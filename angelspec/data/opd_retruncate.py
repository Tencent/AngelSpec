"""Shorten an existing OPD subset while preserving every selected conversation.

Keep the prefix whenever it has enough usable supervision. Otherwise move a
contiguous window within the same conversation, preserving media boundaries.
No tokens are relabeled and no conversations are replaced or re-sampled.
"""

import argparse
import hashlib
import json
import struct
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np

from angelspec.data.opd_subset import (
    IMAGE_END,
    IMAGE_START,
    MEDIA,
    SEP,
    SIDECAR_KEYS,
    VIDEO_END,
    VIDEO_START,
    content_hash,
    decode,
    file_identity,
    media_slice,
    prefix_cut,
    read_index,
)


def usable_loss(ids, loss):
    """Match the trainer's anchor eligibility, including two context tokens."""
    result = loss.copy()
    result[:2] = False
    result[1:] &= ~np.isin(ids[:-1], MEDIA)
    return result


def media_spans(ids):
    """Return complete outer image/video intervals, with exclusive ends."""
    stack, spans = [], []
    closing = {IMAGE_END: IMAGE_START, VIDEO_END: VIDEO_START}
    markers = np.flatnonzero(np.isin(ids, [IMAGE_START, IMAGE_END, VIDEO_START, VIDEO_END]))
    for pos in markers.tolist():
        token = int(ids[pos])
        if token in (IMAGE_START, VIDEO_START):
            stack.append((token, pos))
        else:
            if not stack or stack[-1][0] != closing[token]:
                raise ValueError(f"Unbalanced media marker at {pos}")
            _, start = stack.pop()
            if not stack:
                spans.append((start, pos + 1))
    if stack:
        raise ValueError("Unclosed image/video in parent conversation")
    return spans


def choose_window(ids, loss, cap, min_loss=16, preferred_loss=1024):
    if cap < min_loss + 2 or min_loss < 1:
        raise ValueError("Window must fit context and the minimum supervision")
    spans = media_spans(ids)
    end = prefix_cut(ids, cap)
    if int(usable_loss(ids[:end], loss[:end]).sum()) >= min_loss:
        return 0, end, "prefix"

    # Prefer the earliest window ending after up to 1024 supervised tokens.
    # This avoids turning a long-prompt example into only 16 supervised tokens.
    supervised = np.flatnonzero(usable_loss(ids, loss))
    if len(supervised) < min_loss:
        raise ValueError("Parent conversation has insufficient usable supervision")
    preferred = int(supervised[min(max(preferred_loss, min_loss), len(supervised)) - 1]) + 1
    starts = [a for a, _ in spans]

    def candidate(end):
        start = max(0, end - cap)
        j = bisect_right(starts, start) - 1
        if j >= 0 and spans[j][0] < start < spans[j][1]:
            start = spans[j][1]
        if int(usable_loss(ids[start:end], loss[start:end]).sum()) >= min_loss:
            return start, end, "shifted_for_supervision"
        return None

    result = candidate(preferred)
    if result is not None:
        return result
    # Handles sparse supervision or large media blocks without re-sampling.
    for pos in supervised[min_loss - 1 :]:
        if int(pos) + 1 != preferred:
            result = candidate(int(pos) + 1)
            if result is not None:
                return result
    raise ValueError("No media-safe window contains enough usable supervision")


def retruncate(parent, output, cap=32768, min_loss=16):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    names = (
        "mtp_text_sentence.bin",
        "mtp_text_sentence.idx",
        "mtp_text_sentence_img.npy",
        "manifest.jsonl",
        "report.json",
        "selected_indices.npy",
        "excluded_old_indices.npy",
    )
    identities = [file_identity(parent / name) for name in names]
    parent_report = json.loads((parent / "report.json").read_text())
    if not parent_report["complete"] or not 0 < cap < parent_report["max_tokens"]:
        raise ValueError("Expected a complete parent dataset with a larger length cap")
    rows = [json.loads(line) for line in (parent / "manifest.jsonl").open()]
    selected = np.load(parent / "selected_indices.npy")
    old = np.load(parent / "excluded_old_indices.npy")
    sizes, offsets = read_index(parent / "mtp_text_sentence")
    if not len(rows) == len(sizes) == len(selected) == parent_report["selected"]:
        raise ValueError("Parent manifest, index and selection counts disagree")
    if len(np.unique(selected)) != len(selected) or np.intersect1d(selected, old).size:
        raise ValueError("Parent selection is not unique and disjoint")
    if [r["population_index"] for r in rows] != selected.tolist():
        raise ValueError("Parent manifest order differs from selected IDs")
    mm = np.memmap(parent / "mtp_text_sentence.bin", mode="r", dtype="<i4")
    if len(mm) != int(offsets[-1]):
        raise ValueError("Parent bin size differs from its index")
    sidecar = np.load(parent / "mtp_text_sentence_img.npy", allow_pickle=True).item()
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    new_media = {key: {} for key in SIDECAR_KEYS}
    bin_hash, parent_hash = hashlib.sha256(), hashlib.sha256()
    new_sizes, lengths, loss_counts, anchor_counts = [], [], [], []
    shifted, shortened, boundary_adjusted, with_media = [], 0, 0, 0
    begun = time.monotonic()
    with (
        (staging / "mtp_text_sentence.bin").open("wb") as data,
        (staging / "manifest.jsonl").open("w") as manifest,
    ):
        for i, row in enumerate(rows):
            encoded = mm[offsets[i] : offsets[i + 1]]
            parent_hash.update(memoryview(encoded).cast("B"))
            raw = encoded[:-1]
            if row["output_index"] != i or encoded[-1] != SEP or len(raw) != row["cut"]:
                raise ValueError(f"Parent row/index mismatch at {i}")
            ids, loss = decode(raw)
            image_count = int((ids == IMAGE_START).sum())
            # Empty per-row keys are significant in the parent's content hash.
            media = {
                key: list(values[i])
                for key, values in sidecar.items()
                if key in SIDECAR_KEYS and i in values
            }
            if image_count != len(sidecar.get("img_path", {}).get(i, [])):
                raise ValueError(f"Parent image count mismatch at {i}")
            if content_hash(raw, media) != row["retained_sha256"]:
                raise ValueError(f"Parent content hash mismatch at {i}")
            start, end, mode = choose_window(ids, loss, cap, min_loss)
            retained = raw[start:end]
            kept_ids, kept_loss = ids[start:end], loss[start:end]
            skipped_images = int((ids[:start] == IMAGE_START).sum())
            kept_images = int((kept_ids == IMAGE_START).sum())
            media = media_slice(sidecar, i, skipped_images, kept_images)
            for key, values in media.items():
                new_media[key][i] = values
            encoded = np.r_[retained, np.int32(SEP)].astype("<i4", copy=False)
            payload = memoryview(encoded).cast("B")
            data.write(payload)
            bin_hash.update(payload)
            new_sizes.append(len(encoded))
            lengths.append(len(retained))
            loss_counts.append(int(kept_loss.sum()))
            anchor_counts.append(int(usable_loss(kept_ids, kept_loss).sum()))
            shortened += int(len(retained) < len(raw))
            boundary_adjusted += int(len(retained) < min(len(raw), cap))
            with_media += int(kept_images > 0)
            if start:
                shifted.append(i)
            manifest.write(
                json.dumps(
                    {
                        **row,
                        "parent_retained_sha256": row["retained_sha256"],
                        "parent_cut": row["cut"],
                        "window_start": start,
                        "window_end": end,
                        "window_policy": mode,
                        "cut": len(retained),
                        "retained_loss_tokens": loss_counts[-1],
                        "usable_anchors": anchor_counts[-1],
                        "retained_images": kept_images,
                        "skipped_parent_images": skipped_images,
                        "retained_sha256": content_hash(retained, media),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if (i + 1) % 10000 == 0:
                print(
                    f"written {i + 1}/{len(rows)}, seconds={time.monotonic() - begun:.1f}",
                    flush=True,
                )
    if parent_hash.hexdigest() != parent_report["bin_sha256"]:
        raise ValueError("Parent bin checksum differs from its report")
    if identities != [file_identity(parent / name) for name in names]:
        raise ValueError("Parent dataset changed during conversion")
    n = len(rows)
    with (staging / "mtp_text_sentence.idx").open("wb") as index:
        index.write(b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, n, n + 1))
        np.asarray(new_sizes, dtype="<i4").tofile(index)
        (np.r_[0, np.cumsum(new_sizes[:-1])] * 4).astype("<i8").tofile(index)
        np.arange(n + 1, dtype="<i8").tofile(index)
    np.save(staging / "mtp_text_sentence_img.npy", new_media, allow_pickle=True)
    for name in ("selected_indices.npy", "excluded_old_indices.npy"):
        (staging / name).write_bytes((parent / name).read_bytes())
    report = {
        "schema_version": 2,
        "complete": True,
        "source": identities,
        "parent_dataset": str(parent),
        "parent_bin_sha256": parent_hash.hexdigest(),
        "population": parent_report["population"],
        "seed": parent_report["seed"],
        "old_seed": parent_report["old_seed"],
        "selected": n,
        "same_selection_as_parent": True,
        "overlap_with_old": 0,
        "replacements": [],
        "max_tokens": cap,
        "min_loss_tokens": min_loss,
        "truncation": "prefix_else_shifted_supervised_media_safe_window",
        "preferred_shifted_loss_tokens": 1024,
        "tokens": sum(lengths),
        "loss_tokens": sum(loss_counts),
        "min_retained_loss_tokens": min(loss_counts),
        "min_usable_anchors": min(anchor_counts),
        "max_retained_tokens": max(lengths),
        "mean_tokens": float(np.mean(lengths)),
        "median_tokens": float(np.median(lengths)),
        "truncated": shortened,
        "media_boundary_adjusted": boundary_adjusted,
        "with_media": with_media,
        "shifted_for_supervision": len(shifted),
        "shifted_output_indices": shifted,
        "bin_sha256": bin_hash.hexdigest(),
    }
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.rename(output)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("source", "shifted_output_indices")},
            indent=2,
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--min-loss-tokens", type=int, default=16)
    args = parser.parse_args()
    retruncate(args.parent, args.output, args.max_tokens, args.min_loss_tokens)


if __name__ == "__main__":
    main()
