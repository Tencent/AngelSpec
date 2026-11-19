"""Batch-invariance check for Hy4 flash, the property OPD's teacher engines rely on.

    python scripts/hy_v4_flash_batch_invariance.py --model $TARGET [--max-num-seqs N]

Runs one TP engine in vLLM's batch-invariant mode with prefix caching and the OPD
hidden-state capture, and compares the final hidden rows of the last ``--tail`` tokens of
a prompt (and of branches off it) computed different ways; each pair must be identical:

- the prompt prefilled alone in ``--chunk``-token chunks (an OPD prefill engine's trunk)
  against a short query after its cached prefix (an OPD verify branch);
- branches off the cached prompt verified in one call against each verified alone;
- the prompt prefilled in one call with other prompts against alone.

AngelSpec runs Hy4 flash engines one request per step, the default here. With more
(``--max-num-seqs 8``), public vLLM's results depend on the step's other requests.
"""

import argparse
import os
import sys
from pathlib import Path

CAPTURE_EXT = "angelspec.inference.engine.hidden_capture_worker_ext.HiddenCaptureWorkerExt"


def _rows(llm, outputs, hidden_size):
    import numpy as np
    import torch

    got = llm.collective_rpc("hc_pop", args=([o.request_id for o in outputs],))[0]
    rows = []
    for o in outputs:
        positions, data, n = got[o.request_id]
        rows.append(
            (
                np.frombuffer(positions, dtype=np.int64).tolist(),
                torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view(n, hidden_size),
            )
        )
    return rows


def _same(name, got, want):
    import torch

    (got_positions, got_rows), (want_positions, want_rows) = got, want
    same = got_positions == want_positions and torch.equal(got_rows, want_rows)
    diff = (
        (got_rows.float() - want_rows.float()).abs().max().item()
        if got_positions == want_positions
        else None
    )
    print(f"{name}: positions {got_positions} identical={same} max|diff|={diff}", flush=True)
    return same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=2048, help="max_num_batched_tokens")
    parser.add_argument("--prefix", type=int, default=2560, help="cached prefix, block aligned")
    parser.add_argument("--tail", type=int, default=7, help="tokens a branch scores")
    args = parser.parse_args()
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

    import json

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    text = json.loads((Path(args.model) / "config.json").read_text())["text_config"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    corpus = tokenizer(readme * 4, add_special_tokens=False)["input_ids"]
    prompt = corpus[: args.prefix + args.tail]
    others = [corpus[start : start + length] for start, length in ((97, 1500), (311, 3333))]
    tails = [corpus[-(i + 1) * args.tail :][: args.tail] for i in range(4)]
    branches = [
        prompt[:anchor] + tail
        for anchor, tail in zip((1500, args.prefix - 333, args.prefix - 40, args.prefix), tails)
    ]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="mp",
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.chunk,
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        engram_config={"cpu_offload": False},
        kernel_config={"moe_backend": "triton"},
        worker_extension_cls=CAPTURE_EXT,
    )
    llm.collective_rpc("hc_configure", args=(args.tail,))

    def run(prompts, uncached=True):
        params = SamplingParams(
            max_tokens=1, temperature=0, detokenize=False, skip_reading_prefix_cache=uncached
        )
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        return _rows(llm, outputs, text["hidden_size"])

    (alone,) = run([prompt])
    (branch,) = run([prompt], uncached=False)
    ok = _same("cached-prefix branch vs trunk", branch, alone)
    one_by_one = [run([b], uncached=False)[0] for b in branches]
    together = run(branches, uncached=False)
    for i, (got, want) in enumerate(zip(together, one_by_one)):
        ok &= _same(f"branch {i} verified in one call vs alone", got, want)
    mixed = run([others[0], prompt, others[1]])[1]
    ok &= _same("prompt prefilled with other prompts vs alone", mixed, alone)
    print("BATCH_INVARIANT_OK" if ok else "BATCH_INVARIANT_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
