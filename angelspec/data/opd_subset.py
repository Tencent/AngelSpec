"""Prepare an audited, disjoint Hy4 flash rollout subset without re-tokenization.

The population is the *old* loader's 256k filtered file order. Selection precedes
128k truncation. This command never modifies the source bin or image files.
"""

import argparse
import hashlib
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

OFFSET, SEP, PAD = 300000, 120026, 120002
IMAGE_START, IMAGE_END, VIDEO_START, VIDEO_END = 120118, 120119, 120122, 120123
MEDIA = np.arange(120118, 120125)
VISUAL = np.array([120120, 120121, 120124])
SIDECAR_KEYS = ("img_path", "img_target_size", "is_video", "video_idx")


def read_index(prefix):
    with open(str(prefix) + ".idx", "rb") as f:
        if f.read(9) != b"MMIDIDX\x00\x00":
            raise ValueError("Not a Megatron mmap index")
        version, code, n, _ = struct.unpack("<QBQQ", f.read(25))
        if (version, code) != (1, 4):
            raise ValueError(f"Expected int32 mmap version 1, got {version}/{code}")
        sizes = np.frombuffer(f.read(4 * n), dtype="<i4").astype(np.int64)
    return sizes, np.r_[0, np.cumsum(sizes)]


def decode(raw):
    context = raw >= OFFSET
    ids = np.where(context, raw - OFFSET, raw)
    loss = (~context) & (ids != PAD) & (~np.isin(ids, MEDIA))
    return ids, loss


def prefix_cut(ids, cap):
    cut = min(len(ids), cap)
    # The enclosing video takes precedence over an individual frame.
    for start, end in ((VIDEO_START, VIDEO_END), (IMAGE_START, IMAGE_END)):
        starts, ends = np.flatnonzero(ids[:cut] == start), np.flatnonzero(ids[:cut] == end)
        if len(starts) > len(ends):
            cut = int(starts[-1])
    return cut


def media_slice(sidecar, bi, start, count):
    result = {}
    for key in SIDECAR_KEYS:
        values = sidecar.get(key, {}).get(bi, [])
        if len(values):
            result[key] = list(values[start : start + count])
    return result


def content_hash(raw, media):
    h = hashlib.sha256(memoryview(np.ascontiguousarray(raw)).cast("B"))
    h.update(
        json.dumps(
            media,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda x: x.tolist() if hasattr(x, "tolist") else int(x),
        ).encode()
    )
    return h.hexdigest()


def file_identity(path):
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def scan_population(prefix, sidecar, cap, workers):
    sizes, offsets = read_index(prefix)
    mm = np.memmap(str(prefix) + ".bin", mode="r", dtype="<i4")

    def scan(bounds):
        rows, dropped = [], np.zeros(3, dtype=np.int64)
        for bi in range(*bounds):
            raw = mm[offsets[bi] : offsets[bi + 1]]
            ids, loss = decode(raw)
            start = image_cursor = 0
            for segment, stop in enumerate(np.flatnonzero(ids == SEP)):
                a = start
                start = int(stop) + 1
                x, mask = ids[a:stop], loss[a:stop]
                n_images = int((x == IMAGE_START).sum())
                image_start = image_cursor
                image_cursor += n_images
                if not len(x):
                    continue
                if int(mask.sum()) < 32:
                    dropped[0] += 1
                    continue
                depth = np.cumsum(x == IMAGE_START) - np.cumsum(x == IMAGE_END)
                if np.isin(x[depth <= 0], VISUAL).any():
                    dropped[1] += 1
                    continue
                if len(x) > 262142:  # old max_seq_length=262143, loader reserves 1
                    dropped[2] += 1
                    continue
                cut = prefix_cut(x, cap)
                media = media_slice(sidecar, bi, image_start, n_images)
                rows.append(
                    {
                        "bin": bi,
                        "segment": segment,
                        "offset": int(offsets[bi] + a),
                        "length": len(x),
                        "loss_tokens": int(mask.sum()),
                        "cut": cut,
                        "retained_loss_tokens": int(mask[:cut].sum()),
                        "image_start": image_start,
                        "image_count": n_images,
                        "sha256": content_hash(raw[a:stop], media),
                    }
                )
            if np.any(ids[start:] != PAD):
                raise ValueError(f"bin {bi}: non-padding tail")
            if image_cursor != len(sidecar.get("img_path", {}).get(bi, [])):
                raise ValueError(f"bin {bi}: image count mismatch")
        return rows, dropped

    population, dropped = [], np.zeros(3, dtype=np.int64)
    begun = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = ((i, min(i + 64, len(sizes))) for i in range(0, len(sizes), 64))
        for i, (rows, counts) in enumerate(pool.map(scan, chunks)):
            population.extend(rows)
            dropped += counts
            if (i + 1) % 160 == 0:
                print(
                    f"scanned {min((i + 1) * 64, len(sizes))}/{len(sizes)} bins, "
                    f"valid={len(population)}, seconds={time.monotonic() - begun:.0f}",
                    flush=True,
                )
    return population, dropped.tolist()


def select(population, size, old_seed, seed, min_loss):
    n = len(population)
    old = np.sort(np.random.default_rng(old_seed).choice(n, size=size, replace=False))
    old_set = set(old.tolist())
    old_hashes = {population[i]["sha256"] for i in old}
    remaining = np.array([i for i in range(n) if i not in old_set], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected, rejected, seen_hashes = [], [], set()
    while len(selected) < size:
        need = size - len(selected)
        if len(remaining) < need:
            raise ValueError("Not enough eligible disjoint conversations")
        draw = np.sort(rng.choice(remaining, size=need, replace=False))
        remaining = np.setdiff1d(remaining, draw, assume_unique=True)
        for i in draw.tolist():
            row = population[i]
            reason = (
                "no_supervision_after_cut"
                if row["retained_loss_tokens"] < min_loss
                else (
                    "content_in_old_subset"
                    if row["sha256"] in old_hashes
                    else "duplicate_content" if row["sha256"] in seen_hashes else None
                )
            )
            if reason:
                rejected.append({"population_index": i, "reason": reason})
            else:
                selected.append(i)
                seen_hashes.add(row["sha256"])
    return old, sorted(selected), rejected


def write_subset(prefix, output, sidecar, population, selected, old, rejected, settings):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    mm = np.memmap(str(prefix) + ".bin", mode="r", dtype="<i4")
    new_media = {key: {} for key in SIDECAR_KEYS}
    sizes, total, bin_hash = [], 0, hashlib.sha256()
    with (
        open(staging / "mtp_text_sentence.bin", "wb") as f,
        open(staging / "manifest.jsonl", "w") as manifest,
    ):
        for out_id, source_id in enumerate(selected):
            row = population[source_id]
            raw = mm[row["offset"] : row["offset"] + row["cut"]]
            ids, loss = decode(raw)
            n_images = int((ids == IMAGE_START).sum())
            media = media_slice(sidecar, row["bin"], row["image_start"], n_images)
            for key, values in media.items():
                new_media[key][out_id] = values
            encoded = np.r_[raw, np.int32(SEP)].astype("<i4", copy=False)
            payload = memoryview(encoded).cast("B")
            f.write(payload)
            bin_hash.update(payload)
            sizes.append(len(encoded))
            total += len(raw)
            manifest.write(
                json.dumps(
                    {
                        "output_index": out_id,
                        "population_index": source_id,
                        "source_id": f"single_bin{row['bin']}_conv{source_id}",
                        **row,
                        "retained_sha256": content_hash(raw, media),
                        "retained_images": n_images,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    n = len(sizes)
    with open(staging / "mtp_text_sentence.idx", "wb") as f:
        f.write(b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, n, n + 1))
        np.asarray(sizes, dtype="<i4").tofile(f)
        (np.r_[0, np.cumsum(sizes[:-1])] * 4).astype("<i8").tofile(f)
        np.arange(n + 1, dtype="<i8").tofile(f)
    np.save(staging / "mtp_text_sentence_img.npy", new_media, allow_pickle=True)
    np.save(staging / "excluded_old_indices.npy", old)
    np.save(staging / "selected_indices.npy", np.asarray(selected, dtype=np.int64))
    report = {
        **settings,
        "complete": True,
        "selected": n,
        "tokens": total,
        "overlap_with_old": len(set(selected) & set(old.tolist())),
        "truncated": sum(population[i]["cut"] < population[i]["length"] for i in selected),
        "bin_sha256": bin_hash.hexdigest(),
        "replacements": rejected,
    }
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.rename(output)
    print(
        json.dumps({k: v for k, v in report.items() if k != "replacements"}, indent=2), flush=True
    )
    print(f"replacements={len(rejected)}, output={output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-prefix", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=50000)
    p.add_argument("--expected-population", type=int, default=None)
    p.add_argument("--old-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=131072)
    p.add_argument("--min-loss-tokens", type=int, default=16)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    sidecar = np.load(a.source_prefix + "_img.npy", allow_pickle=True).item()
    source = [file_identity(a.source_prefix + suffix) for suffix in (".bin", ".idx", "_img.npy")]
    population, dropped = scan_population(a.source_prefix, sidecar, a.max_tokens, a.workers)
    if a.expected_population is not None and len(population) != a.expected_population:
        raise ValueError(
            f"Canonical population changed: {len(population)} != {a.expected_population}"
        )
    old, selected, rejected = select(population, a.size, a.old_seed, a.seed, a.min_loss_tokens)
    if source != [
        file_identity(a.source_prefix + suffix) for suffix in (".bin", ".idx", "_img.npy")
    ]:
        raise ValueError("Source changed while scanning")
    write_subset(
        a.source_prefix,
        a.output,
        sidecar,
        population,
        selected,
        old,
        rejected,
        {
            "schema_version": 1,
            "source": source,
            "population": len(population),
            "dropped_no_loss_stray_overlength": dropped,
            "seed": a.seed,
            "old_seed": a.old_seed,
            "max_tokens": a.max_tokens,
            "min_loss_tokens": a.min_loss_tokens,
            "truncation": "prefix_media_boundary",
        },
    )


if __name__ == "__main__":
    main()
