"""Throughput benchmark of the online dump writer on real Mooncake and storage.

Producers fill Mooncake once with synthetic samples whose lengths follow a real
dump; writers then copy them out and write native files exactly as an online
run does, in dispatch-sized batches. Producers keep their keys, so several
writer configurations can be measured against the same data.

    plan     --lengths-from <dump>/manifest.json --samples N --producers P --out plan.json
    produce  --plan plan.json --index i --master H:P --metadata URL --ready <file> --stop <file>
    write    --plan plan.json --index k --writers W --master H:P --metadata URL --root <dir> --out result.json

Start a mooncake_master first; each producer runs on its own node until the
stop file appears. Writer k handles every W-th sample, so W writers on W nodes
measure scaling across nodes.
"""

import argparse
import json
import os
import random
import resource
import socket
import time
from pathlib import Path

import torch

LAYOUTS = {
    "hy-v4-flash-dfly2": {"layers": [24, 28, 32, 36, 40], "target_layers": 42, "hidden": 3072},
}
VOCAB = 120832


def _recipe(layout, max_seq_length):
    from angelspec.data.offline import layer_recipe

    spec = LAYOUTS[layout]
    final = spec["target_layers"] - 1
    return {
        **layer_recipe(
            spec["layers"],
            target_num_hidden_layers=spec["target_layers"],
            hidden_size=spec["hidden"],
        ),
        "capture_aux_layer_order": [layer for layer in spec["layers"] if layer != final],
        "schema_version": 1,
        "capture_backend": "angelspec-online-bench",
        "online_capture": True,
        "model_family": "dflash",
        "target_metadata_sha256": {"config.json": "online-dump-bench"},
        "vocab_size": VOCAB,
        "max_seq_length": max_seq_length,
        "storage_dtype": "bfloat16",
        "capture_epochs": 1,
        "multimodal": False,
    }


def plan(args):
    lengths = [
        record["seq_len"] for record in json.loads(Path(args.lengths_from).read_text())["records"]
    ]
    rng = random.Random(args.seed)
    samples = [
        {
            "key": f"bench{index:06d}",
            "length": rng.choice(lengths),
            "producer": index % args.producers,
        }
        for index in range(args.samples)
    ]
    spec = LAYOUTS[args.layout]
    width = spec["hidden"] * len(spec["layers"])
    total = sum(sample["length"] * (9 + 2 * width) for sample in samples)
    Path(args.out).write_text(
        json.dumps(
            {
                "layout": args.layout,
                "max_seq_length": max(lengths),
                "samples": samples,
            }
        )
    )
    print(f"{len(samples)} samples, {total / 2**30:.0f} GiB, longest {max(lengths)} tokens")


def _host():
    return os.environ.get("BENCH_HOST") or socket.gethostbyname(socket.gethostname())


def produce(args):
    from mooncake.store import MooncakeDistributedStore

    layout = json.loads(Path(args.plan).read_text())
    spec = LAYOUTS[layout["layout"]]
    hidden, aux = spec["hidden"], len(spec["layers"]) - 1
    mine = [sample for sample in layout["samples"] if sample["producer"] == args.index]
    longest = max(sample["length"] for sample in mine)
    host = _host()
    os.environ["MC_TCP_BIND_ADDRESS"] = host
    client = MooncakeDistributedStore()
    status = client.setup(
        local_hostname=host,
        metadata_server=args.metadata,
        global_segment_size=args.segment_gb << 30,
        local_buffer_size=1 << 30,
        protocol="tcp",
        rdma_devices="",
        master_server_addr=args.master,
    )
    if status:
        raise RuntimeError(f"Mooncake producer setup failed: {status}")
    generator = torch.Generator().manual_seed(args.index)
    pools = [
        torch.randn(longest, aux * hidden, generator=generator).to(torch.bfloat16),
        torch.randn(longest, hidden, generator=generator).to(torch.bfloat16),
    ]
    for pool in pools:
        if client.register_buffer(pool.data_ptr(), pool.numel() * pool.element_size()):
            raise RuntimeError("Mooncake buffer registration failed")
    started = time.monotonic()
    for sample in mine:
        key, length = sample["key"], sample["length"]
        ids = torch.randint(0, VOCAB, (length,), generator=generator)
        if client.put(f"{key}_ids", ids.numpy().tobytes()):
            raise RuntimeError(f"put failed for {key}_ids")
        codes = client.batch_put_from(
            [f"{key}_hs", f"{key}_lhs"],
            [pool.data_ptr() for pool in pools],
            [length * aux * hidden * 2, length * hidden * 2],
        )
        if any(codes):
            raise RuntimeError(f"put failed for {key}: {codes}")
    Path(args.ready).write_text(
        json.dumps(
            {
                "host": host,
                "samples": len(mine),
                "seconds": round(time.monotonic() - started, 1),
            }
        )
    )
    while not Path(args.stop).exists():
        time.sleep(2)
    client.close()


def _results(layout, samples):
    from angelspec.utils.types import InferenceOutput

    spec = LAYOUTS[layout]
    hidden, aux = spec["hidden"], len(spec["layers"]) - 1
    return [
        InferenceOutput(
            data_id=sample["key"],
            mooncake_key=sample["key"],
            tensor_shapes={
                "input_ids": (sample["length"],),
                "hidden_states": (sample["length"], aux * hidden),
                "last_hidden_states": (sample["length"], hidden),
            },
            tensor_dtypes={
                "input_ids": torch.int64,
                "hidden_states": torch.bfloat16,
                "last_hidden_states": torch.bfloat16,
            },
            packed_loss_mask=f"{sample['length'] // 4},{sample['length'] - sample['length'] // 4}",
        )
        for sample in samples
    ]


def write(args):
    from angelspec.config.mooncake_config import MooncakeConfig
    from angelspec.inference.online_dump import OnlineDumpWriter

    layout = json.loads(Path(args.plan).read_text())
    samples = layout["samples"][args.index :: args.writers]
    results = _results(layout["layout"], samples)
    batches, batch, tokens = [], [], 0
    for result, sample in zip(results, samples):
        batch.append(result)
        tokens += sample["length"]
        if tokens >= args.batch_tokens:
            batches.append(batch)
            batch, tokens = [], 0
    if batch:
        batches.append(batch)
    config = MooncakeConfig(
        local_hostname=_host(),
        metadata_server=args.metadata,
        master_server_address=args.master,
        protocol="tcp",
        async_put_pool_size=0,
        get_retry_max_wait_seconds=60,
    )
    writer = OnlineDumpWriter(
        config,
        args.root,
        _recipe(layout["layout"], layout["max_seq_length"]),
        validation_fraction=0.05,
        seed=0,
        min_loss_tokens=1,
        max_pending_bytes=args.pending_gb << 30,
        writer_threads=args.writer_threads,
        reader_threads=args.reader_threads,
    )
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    started = time.monotonic()
    for batch in batches:
        writer.stage_batch(batch)
    staged = time.monotonic()
    while True:
        status = writer.status()
        if status["written"] >= status["staged"] and not status["pending_bytes"]:
            break
        time.sleep(0.2)
    written = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    final = writer.finalize(verify="header")
    published = time.monotonic()
    gib = final["written_bytes"] / 2**30
    report = {
        "host": _host(),
        "writers": args.writers,
        "reader_threads": args.reader_threads,
        "writer_threads": args.writer_threads,
        "samples": final["written"],
        "batches": len(batches),
        "gib": round(gib, 1),
        "staging_gib_s": round(gib / (staged - started), 2),
        "write_gib_s": round(gib / (written - started), 2),
        "busy_readers": round(final["read_seconds"] / (written - started), 2),
        "busy_writers": round(final["write_seconds"] / (written - started), 2),
        "cores": round(
            (usage.ru_utime + usage.ru_stime - cpu.ru_utime - cpu.ru_stime) / (written - started),
            2,
        ),
        "peak_rss_gib": round(usage.ru_maxrss / 2**20, 1),
        "publish_seconds": round(published - written, 1),
    }
    Path(args.out).write_text(json.dumps(report))
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--lengths-from", required=True)
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--producers", type=int, required=True)
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="hy-v4-flash-dfly2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    for name in ("produce", "write"):
        p = commands.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--index", type=int, required=True)
        p.add_argument("--master", required=True)
        p.add_argument("--metadata", required=True)
    commands.choices["produce"].add_argument("--segment-gb", type=int, default=128)
    commands.choices["produce"].add_argument("--ready", required=True)
    commands.choices["produce"].add_argument("--stop", required=True)
    w = commands.choices["write"]
    w.add_argument("--writers", type=int, default=1)
    w.add_argument("--root", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--reader-threads", type=int, default=4)
    w.add_argument("--writer-threads", type=int, default=16)
    w.add_argument("--pending-gb", type=int, default=32)
    # One GBS-256 dispatch with packing: a 256k-token row for each of 16 DP ranks.
    w.add_argument("--batch-tokens", type=int, default=16 * 262144)
    args = parser.parse_args()
    {"plan": plan, "produce": produce, "write": write}[args.command](args)


if __name__ == "__main__":
    main()
