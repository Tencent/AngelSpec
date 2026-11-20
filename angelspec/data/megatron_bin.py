# Copyright (c) 2026 Tencent. Licensed under the AngelSpec License.
"""Load a Megatron ``MMapIndexedDataset`` (``.bin``/``.idx``) of pre-packed,
pre-tokenized SFT sequences and unpack it into per-conversation samples that
match the schema produced by :func:`angelspec.data.dataset.load_conversation_dataset`.

Why unpack instead of feeding the packed blocks directly:
    AngelSpec's MTP/DFlash training is disaggregated — the inference workers run
    the target model on each *conversation* to extract ``last_hidden_states``,
    and :class:`MTPPackingCollator` re-packs those hidden states online (with
    per-doc ``ctx_doc_ids`` + doc-local positions for cross-document isolation).
    So the pipeline needs individual conversations, not offline-packed blocks.
    The offline packing arrangement is discarded and rebuilt online anyway.

Data layout:
    * Each ``.idx`` entry is one fixed-length bin (e.g. 262144 tokens), int32.
    * A bin concatenates several whole conversations, then pads the tail.
    * Per-token *loss mask* is encoded by an additive offset: a token stored as
      ``id + LOSS_OFFSET`` (>= LOSS_OFFSET) is CONTEXT (no loss); a token stored
      as plain ``id`` (< LOSS_OFFSET) is a supervised (loss=1) token. Trailing
      ``PAD`` tokens are stored plain but must be excluded from loss.
    * Conversations are separated by a single ``DOC_SEP`` token: every document
      ends ``... <hy_end> <DOC_SEP>`` and the next begins with ``<hy_start>``.
      (Verified: #DOC_SEP == #system messages == #documents per bin; every
      DOC_SEP is preceded by <hy_end> and followed by <hy_start>.)

Decoding rules:
    input_ids  = raw - LOSS_OFFSET  if raw >= LOSS_OFFSET  else raw
    loss_mask  = (raw < LOSS_OFFSET) AND (input_ids != PAD)      # honor offset
    documents  = split at DOC_SEP; drop the DOC_SEP token; drop trailing pad.
"""

import hashlib
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from angelspec.data.utils import serialize_packed_loss_mask
from angelspec.utils.logging import logger

# Hy4 defaults (overridable via args). See module docstring.
DEFAULT_LOSS_OFFSET = 300000
DEFAULT_DOC_SEP_ID = 120026  # <｜hy_place▁holder▁no▁8｜> repurposed as doc separator
DEFAULT_PAD_ID = 120002  # <｜hy_pad:...｜>
DEFAULT_IMAGE_START_TOKEN_ID = 120118  # <img_start>; one per image, used to slice sidecar refs
DEFAULT_IMAGE_END_TOKEN_ID = 120119
DEFAULT_VIDEO_START_TOKEN_ID = 120122
DEFAULT_VIDEO_END_TOKEN_ID = 120123
# Multimodal special tokens that must NOT be supervised. The megatron .bin stores
# image tokens PLAIN (no +LOSS_OFFSET), so the offset decode alone marks them
# loss=1; the real training pipeline masks them by id in process_vlm_special_ids().
# Mirror that here so the draft is only supervised on decode-time text.
# (image_start/end/pad/newline, video_start/end for Hy4 flash.)
DEFAULT_IMAGE_MASK_TOKEN_IDS = (
    120118,
    120119,
    120120,
    120121,
    120122,
    120123,
    120124,
)

# Megatron MMapIndexedDataset dtype codes -> numpy dtype.
_MMAP_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}
_MMAP_MAGIC = b"MMIDIDX\x00\x00"


def _read_mmap_index(idx_path: str):
    """Parse a Megatron MMapIndexedDataset ``.idx`` header.

    Returns (dtype, sizes[int64 array], element_offsets[int64 array]) where
    ``element_offsets[i]`` is the starting element index of doc ``i`` in the
    ``.bin`` file (cumsum of sizes) and ``sizes[i]`` is its token count.
    """
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        if magic != _MMAP_MAGIC:
            raise ValueError(f"{idx_path}: not a Megatron MMapIndexedDataset (magic={magic!r})")
        version = struct.unpack("<Q", f.read(8))[0]
        if version != 1:
            raise ValueError(f"{idx_path}: unsupported .idx version {version}")
        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype = _MMAP_DTYPES.get(dtype_code)
        if dtype is None:
            raise ValueError(f"{idx_path}: unknown dtype code {dtype_code}")
        n = struct.unpack("<Q", f.read(8))[0]  # num sequences (docs)
        struct.unpack("<Q", f.read(8))[0]  # doc_count (unused here)
        sizes = np.frombuffer(f.read(4 * n), dtype=np.int32).astype(np.int64)
    element_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(sizes, out=element_offsets[1:])
    return dtype, sizes, element_offsets


def _resolve_bin_idx(train_data_path: str):
    """Resolve (bin_path, idx_path) from a path that may be the .bin, .idx,
    a directory, or a prefix (``.../train_text_sentence``)."""
    p = train_data_path
    if p.endswith(".idx"):
        return p[:-4] + ".bin", p
    if p.endswith(".bin"):
        return p, p[:-4] + ".idx"
    if os.path.isdir(p):
        idxs = [f for f in os.listdir(p) if f.endswith(".idx")]
        if len(idxs) != 1:
            raise ValueError(
                f"{p}: expected exactly one .idx file, found {idxs}. "
                f"Point train_data_path at the .bin/.idx directly."
            )
        base = os.path.join(p, idxs[0][:-4])
        return base + ".bin", base + ".idx"
    # treat as prefix
    return p + ".bin", p + ".idx"


def is_megatron_bin_path(train_data_path: str) -> bool:
    """True if train_data_path points at a Megatron .bin/.idx dataset."""
    if not train_data_path:
        return False
    if train_data_path.endswith((".bin", ".idx")):
        return True
    try:
        _, idx = _resolve_bin_idx(train_data_path)
        return os.path.exists(idx) and os.path.exists(idx[:-4] + ".bin")
    except Exception:
        return False


def _resolve_img_npy(bin_path: str) -> str | None:
    """Return the `_img.npy` sidecar path for a `.bin` if it exists, else None."""
    cand = bin_path[:-4] + "_img.npy" if bin_path.endswith(".bin") else bin_path + "_img.npy"
    return cand if os.path.exists(cand) else None


def _parse_path_replacements(value) -> list[tuple[str, str]]:
    if not value:
        return []
    if isinstance(value, str):
        pairs = [part.strip() for part in value.split(";") if part.strip()]
    else:
        pairs = list(value)
    replacements = []
    for pair in pairs:
        if isinstance(pair, str):
            if ":" not in pair:
                raise ValueError("Image path replacements must use 'old_prefix:new_prefix'")
            old, new = pair.split(":", 1)
        else:
            old, new = pair
        if not old:
            raise ValueError("Image path replacement source prefix cannot be empty")
        replacements.append((str(old), str(new)))
    return replacements


def _remap_path(path: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if path.startswith(old):
            return new + path[len(old) :]
    return path


def _load_image_sidecar(img_npy_path: str, path_replacements=()):
    """Load the VL media sidecar dict produced by the Megatron VL packer.

    ``img_path`` contains both standalone images and sampled video frames.
    ``is_video`` marks frame entries; the token stream's video delimiters define
    how consecutive frame entries are grouped into videos.
    """
    replacements = _parse_path_replacements(path_replacements)
    meta = np.load(img_npy_path, allow_pickle=True).item()
    img_path = meta.get("img_path", {})
    img_target_size = meta.get("img_target_size", {})
    is_video = meta.get("is_video", {})
    video_idx = meta.get("video_idx", {})
    images_by_bin: dict[int, list] = {}
    sizes_by_bin: dict[int, list] = {}
    for bi, refs in img_path.items():
        bi = int(bi)
        sizes = list(img_target_size.get(bi, []))
        video_flags = list(is_video.get(bi, [0] * len(refs)))
        video_indices = list(video_idx.get(bi, [0] * len(refs)))
        lengths = {
            "img_path": len(refs),
            "is_video": len(video_flags),
            "video_idx": len(video_indices),
        }
        if sizes:
            lengths["img_target_size"] = len(sizes)
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"{img_npy_path}: inconsistent sidecar lengths for bin {bi}: {lengths}"
            )
        parsed = []
        for r, video_flag, video_index in zip(refs, video_flags, video_indices):
            d = json.loads(r) if isinstance(r, (str, bytes)) else r
            missing = {"patch", "start_num", "size"} - d.keys()
            if missing:
                raise ValueError(
                    f"{img_npy_path}: bin {bi} image ref is missing {sorted(missing)}"
                )
            parsed.append(
                {
                    "source_type": "bytearray",
                    "bytearray": {
                        "patch": _remap_path(d["patch"], replacements),
                        "start_num": int(d["start_num"]),
                        "size": int(d["size"]),
                    },
                    "is_video": bool(video_flag),
                    "video_idx": int(video_index),
                }
            )
        images_by_bin[bi] = parsed
        sizes_by_bin[bi] = sizes
    return images_by_bin, sizes_by_bin


def _attach_target_size(item, target_size):
    result = dict(item)
    if target_size:
        result["target_size"] = [int(target_size[0]), int(target_size[1])]
    return result


def _split_segment_media(
    seg_ids,
    refs,
    sizes,
    image_start_token_id,
    image_end_token_id,
    video_start_token_id,
    video_end_token_id,
    *,
    bin_index,
):
    image_starts = np.flatnonzero(seg_ids == image_start_token_id).tolist()
    if len(image_starts) != len(refs):
        raise ValueError(
            f"bin {bin_index}: token stream contains {len(image_starts)} image/frame "
            f"starts but sidecar slice contains {len(refs)} refs"
        )
    image_ends = np.flatnonzero(seg_ids == image_end_token_id).tolist()
    if len(image_ends) != len(image_starts):
        raise ValueError(
            f"bin {bin_index}: image start/end count mismatch "
            f"({len(image_starts)} != {len(image_ends)})"
        )

    videos = []
    video_ranges = []
    open_start = None
    for pos, token in enumerate(seg_ids):
        if token == video_start_token_id:
            if open_start is not None:
                raise ValueError(f"bin {bin_index}: nested video_start at token {pos}")
            open_start = pos
        elif token == video_end_token_id:
            if open_start is None:
                raise ValueError(f"bin {bin_index}: video_end without video_start at token {pos}")
            video_ranges.append((open_start, pos))
            open_start = None
    if open_start is not None:
        raise ValueError(f"bin {bin_index}: unclosed video_start at token {open_start}")

    image_ranges = list(zip(image_starts, image_ends))
    for start, end in image_ranges:
        if end < start:
            raise ValueError(f"bin {bin_index}: image_end precedes image_start")

    standalone_images = []
    consumed = set()
    for video_start, video_end in video_ranges:
        frame_indices = [
            i
            for i, (image_start, image_end) in enumerate(image_ranges)
            if video_start < image_start < image_end < video_end
        ]
        if not frame_indices:
            raise ValueError(
                f"bin {bin_index}: video at tokens {video_start}:{video_end} has no frames"
            )
        frames = []
        frame_prefix_token_ids = []
        cursor = video_start + 1
        target_size = None
        for i in frame_indices:
            image_start, image_end = image_ranges[i]
            item = refs[i]
            if not item.get("is_video", False):
                raise ValueError(
                    f"bin {bin_index}: sidecar ref {i} inside video is not marked as video"
                )
            size = sizes[i] if i < len(sizes) else None
            if size:
                size = (int(size[0]), int(size[1]))
                if target_size is None:
                    target_size = size
                elif size != target_size:
                    raise ValueError(
                        f"bin {bin_index}: video frames have inconsistent target sizes "
                        f"({target_size} != {size})"
                    )
            frames.append(_attach_target_size(item, size))
            frame_prefix_token_ids.append(seg_ids[cursor:image_start].tolist())
            cursor = image_end + 1
            consumed.add(i)
        videos.append(
            {
                "source_type": "frames",
                "frames": frames,
                "frame_prefix_token_ids": frame_prefix_token_ids,
                "suffix_token_ids": seg_ids[cursor:video_end].tolist(),
            }
        )

    for i, item in enumerate(refs):
        if i in consumed:
            continue
        if item.get("is_video", False):
            raise ValueError(f"bin {bin_index}: video sidecar ref {i} is outside video delimiters")
        standalone_images.append(_attach_target_size(item, sizes[i] if i < len(sizes) else None))

    if not standalone_images and not videos:
        return None
    return {
        "images": standalone_images or None,
        "videos": videos or None,
    }


# VL targets whose vision projector appends a learned newline vector to every image
# row (``perceive.image_newline``) when ``vision_config.use_image_newline_token`` is
# not false; HF and vLLM then expect the image placeholder in that slot as well.
IMAGE_NEWLINE_EMBEDDED_MODEL_TYPES = ("hy_v4_flash", "hy_v4_vl")


def image_newline_is_embedded(model_type, use_image_newline_token) -> bool:
    return (
        model_type in IMAGE_NEWLINE_EMBEDDED_MODEL_TYPES and use_image_newline_token is not False
    )


def resolve_image_newline_rewrite(args) -> tuple[int, int, tuple[int, int, int]] | None:
    """``(image_newline_id, image_placeholder_id, (video_start_id, video_end_id,
    video_placeholder_id))`` if newline slots must carry the placeholder, else None.

    The megatron bins mark each image row's last slot with ``image_newline``. When
    the target's vision tower emits that row-newline embedding itself (Hy4 flash),
    serving fills the slot with vision features like any placeholder, so
    vLLM's prompt (and the Engram no-hash set) must see the placeholder id there;
    otherwise every row after the first receives shifted features. Inside a video,
    vLLM lays each frame out with the video placeholder, so the frames' pads and
    newlines carry that one.
    """
    flag = getattr(args, "megatron_image_newline_as_placeholder", None)
    config = {}
    target = getattr(args, "target_model_path", None)
    if target and os.path.isfile(os.path.join(target, "config.json")):
        with open(os.path.join(target, "config.json")) as f:
            config = json.load(f)
    if flag is None:
        vision = config.get("vision_config") or {}
        flag = image_newline_is_embedded(
            config.get("model_type"), vision.get("use_image_newline_token")
        )
    if not flag:
        return None
    video = (
        int(config.get("video_start_token_id", DEFAULT_VIDEO_START_TOKEN_ID)),
        int(config.get("video_end_token_id", DEFAULT_VIDEO_END_TOKEN_ID)),
        int(config.get("video_token_id", 120124)),
    )
    return (
        int(config.get("image_newline_token_id", 120121)),
        int(config.get("image_token_id", 120120)),
        video,
    )


def _rewrite_image_newlines(ids, start_id, end_id, newline_id, placeholder_id, video=None):
    """Returns the rewritten ids and the rewritten newline indices.

    ``video`` is ``(video_start_id, video_end_id, video_placeholder_id)``: inside a
    video, frame pads and newlines take the video placeholder instead.
    """
    depth = np.cumsum(ids == start_id) - np.cumsum(ids == end_id)
    rewrite = (depth > 0) & (ids == newline_id)
    placeholder = np.full_like(ids, placeholder_id)
    pads = np.zeros_like(rewrite)
    if video is not None:
        video_start, video_end, video_placeholder = video
        inside = np.cumsum(ids == video_start) - np.cumsum(ids == video_end) > 0
        placeholder[inside] = video_placeholder
        pads = inside & (depth > 0) & (ids == placeholder_id)
    if not (rewrite.any() or pads.any()):
        return ids, []
    out = ids.copy()
    out[rewrite | pads] = placeholder[rewrite | pads]
    return out, np.nonzero(rewrite)[0].tolist()


def _pack_loss_mask_np(loss_bool: np.ndarray) -> list[int]:
    if loss_bool.size == 0:
        return []
    values = loss_bool.astype(np.int8, copy=False)
    bounds = np.concatenate(([0], np.nonzero(np.diff(values))[0] + 1, [values.size]))
    runs = np.diff(bounds).tolist()
    return [0, *runs] if values[0] else runs


def _unpack_bin_range(
    mm,
    offsets,
    bin_start,
    bin_end,
    loss_offset,
    doc_sep_id,
    pad_id,
    max_length,
    min_loss_tokens,
    image_start_token_id=None,
    images_by_bin=None,
    sizes_by_bin=None,
    image_mask_ids=None,
    image_end_token_id=DEFAULT_IMAGE_END_TOKEN_ID,
    video_start_token_id=DEFAULT_VIDEO_START_TOKEN_ID,
    video_end_token_id=DEFAULT_VIDEO_END_TOKEN_ID,
    image_newline_rewrite=None,
):
    entries = []
    dropped_overlen = 0
    skipped_no_loss = 0
    skipped_stray_visual = 0
    mask_arr = np.asarray(list(image_mask_ids), dtype=np.int64) if image_mask_ids else None
    # Placeholder/newline/video tokens outside an image span get no Hy4 position
    # (data/positions.py raises on them), so such conversations are skipped here
    # instead of failing a trainer mid-run.
    visual_arr = (
        np.setdiff1d(
            mask_arr,
            [image_start_token_id, image_end_token_id, video_start_token_id, video_end_token_id],
        )
        if mask_arr is not None and image_start_token_id is not None
        else None
    )
    for bi in range(bin_start, bin_end):
        a, b = int(offsets[bi]), int(offsets[bi + 1])
        raw = np.asarray(mm[a:b]).astype(np.int64)
        is_ctx = raw >= loss_offset
        ids = np.where(is_ctx, raw - loss_offset, raw)
        sep_pos = np.nonzero(ids == doc_sep_id)[0]
        img_list = images_by_bin.get(bi, []) if images_by_bin is not None else None
        size_list = sizes_by_bin.get(bi, []) if sizes_by_bin is not None else []
        img_cursor = 0
        start = 0
        for sp in sep_pos:
            seg_ids = ids[start:sp]
            seg_ctx = is_ctx[start:sp]
            start = sp + 1
            if seg_ids.size == 0:
                continue
            # Images belong to the bin in stream order (one image_start per image);
            # advance the cursor for every segment — including dropped ones — so a
            # kept segment gets exactly its own images.
            mm_inputs = None
            if img_list is not None and image_start_token_id is not None:
                k = int((seg_ids == image_start_token_id).sum())
                if img_cursor + k > len(img_list):
                    raise ValueError(
                        f"bin {bi}: token stream requires {img_cursor + k} "
                        f"media refs but sidecar contains {len(img_list)}"
                    )
                if k:
                    refs = img_list[img_cursor : img_cursor + k]
                    sizes = size_list[img_cursor : img_cursor + k]
                    mm_inputs = _split_segment_media(
                        seg_ids,
                        refs,
                        sizes,
                        image_start_token_id,
                        image_end_token_id,
                        video_start_token_id,
                        video_end_token_id,
                        bin_index=bi,
                    )
                img_cursor += k
            loss = (~seg_ctx) & (seg_ids != pad_id)
            # Mirror process_vlm_special_ids(): image/video special tokens are
            # stored plain in the .bin but must never be supervised.
            if mask_arr is not None:
                loss &= ~np.isin(seg_ids, mask_arr)
            if not loss.any() or int(loss.sum()) < min_loss_tokens:
                skipped_no_loss += 1
                continue
            if visual_arr is not None:
                depth = np.cumsum(seg_ids == image_start_token_id) - np.cumsum(
                    seg_ids == image_end_token_id
                )
                if np.isin(seg_ids[depth <= 0], visual_arr).any():
                    skipped_stray_visual += 1
                    continue
            if seg_ids.size > max_length:
                dropped_overlen += 1
                continue
            newline_slots = []
            if image_newline_rewrite is not None:
                seg_ids, newline_slots = _rewrite_image_newlines(
                    seg_ids, image_start_token_id, image_end_token_id, *image_newline_rewrite
                )
            entries.append(
                (
                    bi,
                    torch.from_numpy(seg_ids.copy()),
                    serialize_packed_loss_mask(_pack_loss_mask_np(loss)),
                    mm_inputs,
                    newline_slots,
                )
            )
        tail = ids[start:]
        if np.any(tail != pad_id):
            raise ValueError(
                f"bin {bi}: found non-padding tokens after the final document separator"
            )
        if img_list is not None and img_cursor != len(img_list):
            raise ValueError(
                f"bin {bi}: consumed {img_cursor} images from token stream but "
                f"sidecar contains {len(img_list)}"
            )
    return entries, dropped_overlen, skipped_no_loss, skipped_stray_visual


def load_megatron_bin_dataset(args):
    """Unpack a single Megatron .bin/.idx SFT dataset into per-conversation samples.

    Returns a list[dict] identical in schema to
    :func:`load_conversation_dataset` (defer_tokenization=False):
        {data_id, metadata, multimodal_inputs, formatted_prompt,
         input_ids (1D LongTensor), packed_loss_mask (str)}

    Image sidecars (``<prefix>_img.npy``) are auto-detected and, when present,
    attached as ``multimodal_inputs`` per conversation.
    """
    bin_path, _ = _resolve_bin_idx(args.train_data_path)
    load_images = _resolve_img_npy(bin_path) is not None
    prompts = _load_megatron_source(
        args, args.train_data_path, load_images=load_images, source_kind="single"
    )
    return _sample_conversations(args, prompts)


def _sample_conversations(args, prompts):
    """Seeded random subset of ``megatron_sample_size`` conversations, in file order."""
    size = int(getattr(args, "megatron_sample_size", -1))
    if size < 0 or size >= len(prompts):
        return prompts
    seed = int(getattr(args, "megatron_sample_seed", 0))
    keep = np.sort(np.random.default_rng(seed).choice(len(prompts), size=size, replace=False))
    logger.info(f"Megatron-bin: sampled {size} of {len(prompts)} conversations (seed={seed})")
    return [prompts[i] for i in keep]


def _load_megatron_source(args, train_data_path, *, load_images, source_kind):
    bin_path, idx_path = _resolve_bin_idx(train_data_path)
    if not (os.path.exists(bin_path) and os.path.exists(idx_path)):
        raise FileNotFoundError(f"Missing .bin/.idx: {bin_path} / {idx_path}")

    loss_offset = int(getattr(args, "megatron_loss_offset", DEFAULT_LOSS_OFFSET))
    doc_sep_id = int(getattr(args, "megatron_doc_sep_id", DEFAULT_DOC_SEP_ID))
    pad_id = int(getattr(args, "megatron_pad_id", DEFAULT_PAD_ID))
    image_start_token_id = int(
        getattr(args, "megatron_image_start_token_id", DEFAULT_IMAGE_START_TOKEN_ID)
    )
    image_end_token_id = int(
        getattr(args, "megatron_image_end_token_id", DEFAULT_IMAGE_END_TOKEN_ID)
    )
    video_start_token_id = int(
        getattr(args, "megatron_video_start_token_id", DEFAULT_VIDEO_START_TOKEN_ID)
    )
    video_end_token_id = int(
        getattr(args, "megatron_video_end_token_id", DEFAULT_VIDEO_END_TOKEN_ID)
    )
    image_mask_ids = tuple(
        getattr(args, "megatron_image_mask_token_ids", DEFAULT_IMAGE_MASK_TOKEN_IDS) or ()
    )
    # -1 => no cap (load all bins). Set small for smoke tests.
    max_bins = int(getattr(args, "megatron_max_bins", -1))
    bin_start = int(getattr(args, "megatron_bin_start", 0))
    max_length = args.max_seq_length - 1  # engine needs to emit >=1 token
    min_length = int(getattr(args, "min_seq_length", 0) or 0)
    min_loss_tokens = int(getattr(args, "min_loss_tokens", 0))

    img_npy = _resolve_img_npy(bin_path) if load_images else None
    if load_images and img_npy is None:
        raise FileNotFoundError(f"VLM source requires image sidecar: {bin_path[:-4]}_img.npy")
    path_replacements = getattr(args, "megatron_image_path_replacements", "") or os.environ.get(
        "IMAGE_PATH_REPLACEMENTS", ""
    )
    from angelspec.data.positions import IMAGE_NEWLINE_SLOTS_KEY

    image_newline_rewrite = resolve_image_newline_rewrite(args) if img_npy else None
    if image_newline_rewrite is not None:
        logger.info(
            "Megatron-bin: image_newline %d inside image spans -> placeholder %d, inside "
            "video frames -> video placeholder %d with the frame pads (target vision tower "
            "emits the row-newline embedding); rewritten slots kept in metadata[%r] for "
            "position reconstruction",
            *image_newline_rewrite[:2],
            image_newline_rewrite[2][2],
            IMAGE_NEWLINE_SLOTS_KEY,
        )

    # ---- cache (same scheme as load_conversation_dataset) -------------------
    st = os.stat(idx_path)
    bin_st = os.stat(bin_path)
    img_tag = ""
    if img_npy:
        img_st = os.stat(img_npy)
        replacement_tag = hashlib.md5(str(path_replacements).encode()).hexdigest()
        img_tag = (
            f"-img={img_st.st_size}-{img_st.st_mtime}-ist={image_start_token_id}"
            f"-remap={replacement_tag}-newline_slots={image_newline_rewrite}"
        )
    cache_params = (
        f"megatron_bin-{bin_path}-{bin_st.st_size}-{bin_st.st_mtime}"
        f"-idx={st.st_size}-{st.st_mtime}"
        f"-off={loss_offset}-sep={doc_sep_id}-pad={pad_id}"
        f"-maxlen={max_length}-mlt={min_loss_tokens}-maxbins={max_bins}"
        f"{f'-binstart={bin_start}' if bin_start else ''}"
        f"-kind={source_kind}-ims={image_start_token_id}-ime={image_end_token_id}"
        f"-vs={video_start_token_id}-ve={video_end_token_id}"
        f"-immask={'.'.join(map(str, image_mask_ids))}-media-v3{img_tag}"
    )
    if min_length:
        # Only when set, so caches built before min_seq_length existed keep their keys.
        cache_params += f"-minlen={min_length}"
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    cache_dir = os.path.join(getattr(args, "cache_dir", "./cache"), "tokenized_dataset")
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
    if os.path.exists(cache_path):
        logger.info(f"Loading megatron-bin dataset from cache: {cache_path}")
        try:
            prompts = torch.load(cache_path, weights_only=False)
            logger.info(f"Loaded {len(prompts)} cached conversations")
            return prompts
        except Exception as exc:
            corrupt_path = f"{cache_path}.corrupt.{int(time.time())}"
            os.replace(cache_path, corrupt_path)
            logger.warning(
                f"Corrupt megatron-bin cache moved to {corrupt_path}; rebuilding: {exc}"
            )

    images_by_bin = sizes_by_bin = None
    if img_npy:
        images_by_bin, sizes_by_bin = _load_image_sidecar(
            img_npy, path_replacements=path_replacements
        )
        logger.info(f"Loaded image sidecar {img_npy}: {len(images_by_bin)} bins with images")

    dtype, sizes, offsets = _read_mmap_index(idx_path)
    mm = np.memmap(bin_path, dtype=dtype, mode="r")
    if not 0 <= bin_start <= len(sizes):
        raise ValueError(
            f"megatron_bin_start={bin_start} outside [0, {len(sizes)}] for {bin_path}"
        )
    bin_end = len(sizes) if max_bins < 0 else min(len(sizes), bin_start + max_bins)
    n_docs = bin_end - bin_start
    logger.info(
        f"Unpacking Megatron bin ({source_kind}): {bin_path} ({len(sizes)} bins, "
        f"using [{bin_start}, {bin_end})); "
        f"loss_offset={loss_offset} doc_sep={doc_sep_id} pad={pad_id} max_length={max_length} "
        f"min_length={min_length}"
    )

    num_workers = max(1, min(int(getattr(args, "megatron_num_proc", 16)), n_docs or 1))
    bins_per_task = max(1, min(64, (n_docs + num_workers * 8 - 1) // (num_workers * 8)))
    ranges = [
        (start, min(start + bins_per_task, bin_end))
        for start in range(bin_start, bin_end, bins_per_task)
    ]
    logger.info(
        f"Megatron-bin CPU parallelism: workers={num_workers}, "
        f"bins_per_task={bins_per_task}, tasks={len(ranges)}"
    )

    prompts = []
    n_conv = 0
    dropped_overlen = 0
    dropped_short = 0
    skipped_no_loss = 0
    skipped_stray_visual = 0
    work = (
        (
            mm,
            offsets,
            start,
            end,
            loss_offset,
            doc_sep_id,
            pad_id,
            max_length,
            min_loss_tokens,
            image_start_token_id,
            images_by_bin,
            sizes_by_bin,
            image_mask_ids,
            image_end_token_id,
            video_start_token_id,
            video_end_token_id,
            image_newline_rewrite,
        )
        for start, end in ranges
    )
    if num_workers == 1:
        results = map(lambda item: _unpack_bin_range(*item), work)
    else:
        pool = ThreadPoolExecutor(max_workers=num_workers)
        results = pool.map(lambda item: _unpack_bin_range(*item), work)
    try:
        for chunk_entries, chunk_dropped, chunk_skipped, chunk_stray in results:
            dropped_overlen += chunk_dropped
            skipped_no_loss += chunk_skipped
            skipped_stray_visual += chunk_stray
            for bi, input_ids, packed, mm_inputs, newline_slots in chunk_entries:
                if input_ids.numel() < min_length:
                    dropped_short += 1
                    continue
                metadata = {"source": source_kind}
                if newline_slots:
                    metadata[IMAGE_NEWLINE_SLOTS_KEY] = newline_slots
                prompts.append(
                    {
                        "data_id": f"{source_kind}_bin{bi}_conv{n_conv}",
                        "metadata": metadata,
                        "multimodal_inputs": mm_inputs,
                        "formatted_prompt": None,
                        "input_ids": input_ids,
                        "packed_loss_mask": packed,
                    }
                )
                n_conv += 1
    finally:
        if num_workers > 1:
            pool.shutdown(wait=True, cancel_futures=True)

    n_with_img = sum(
        1 for p in prompts if p["multimodal_inputs"] and p["multimodal_inputs"].get("images")
    )
    n_with_video = sum(
        1 for p in prompts if p["multimodal_inputs"] and p["multimodal_inputs"].get("videos")
    )
    logger.info(
        f"Megatron-bin unpack done ({source_kind}): {len(prompts)} conversations "
        f"({n_with_img} with images, {n_with_video} with videos; "
        f"dropped_overlength>{max_length}={dropped_overlen}, "
        f"dropped_short<{min_length}={dropped_short}, "
        f"skipped_no_loss={skipped_no_loss}, "
        f"skipped_visual_outside_image={skipped_stray_visual})"
    )
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            torch.save(prompts, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    logger.info(f"Saved {len(prompts)} conversations to cache: {cache_path}")
    return prompts


def _parse_bin_data_list(list_path: str):
    """Parse a blend spec: each non-empty line is ``<weight> <prefix> <kind>``.

    ``kind`` is a free-form source tag; ``vlm`` enables image-sidecar loading.
    Returns list[(weight: float, prefix: str, kind: str)].
    """
    list_path = os.path.abspath(os.path.expanduser(list_path))
    list_dir = os.path.dirname(list_path)
    sources = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{list_path}: expected '<weight> <prefix> <kind>' per line, got: {line!r}"
                )
            prefix = os.path.expanduser(os.path.expandvars(parts[1]))
            if not os.path.isabs(prefix) and "://" not in prefix:
                prefix = os.path.normpath(os.path.join(list_dir, prefix))
            sources.append((float(parts[0]), prefix, parts[2]))
    if not sources:
        raise ValueError(f"{list_path}: no data sources parsed")
    return sources


def load_megatron_bin_blend(args):
    """Load a weighted blend of Megatron .bin/.idx sources listed in ``args.bin_data_list``.

    Each source is unpacked (and image-loaded for ``vlm`` kinds) via
    :func:`_load_megatron_source`, then subsampled so per-source counts are
    proportional to the given weights (oversampling smaller sources with
    replacement when their weight demands it). Data ids are made unique.
    """
    sources = _parse_bin_data_list(args.bin_data_list)
    seed = int(getattr(args, "seed", 1234) or 1234)
    rng = np.random.default_rng(seed)

    per_source = []
    for weight, prefix, kind in sources:
        prompts = _load_megatron_source(
            args, prefix, load_images=(kind == "vlm"), source_kind=kind
        )
        per_source.append((weight, kind, prompts))

    if any(weight <= 0 for weight, _, _ in per_source):
        raise ValueError("Megatron blend weights must all be positive")
    active_sources = [source for source in per_source if source[2]]
    if not active_sources:
        raise ValueError("All Megatron blend sources unpacked 0 conversations")
    total_weight = sum(w for w, _, _ in active_sources)
    blend_unit = getattr(args, "megatron_blend_unit", "tokens")
    if blend_unit not in ("tokens", "samples"):
        raise ValueError(
            f"Unsupported megatron_blend_unit={blend_unit!r}; " "expected 'tokens' or 'samples'"
        )
    if blend_unit == "tokens":
        total_tokens = sum(
            int(prompt["input_ids"].numel())
            for _, _, prompts in active_sources
            for prompt in prompts
        )
        avg_lengths = [
            (
                sum(int(prompt["input_ids"].numel()) for prompt in prompts) / len(prompts)
                if prompts
                else 0.0
            )
            for _, _, prompts in active_sources
        ]
        count_weights = [
            (weight / total_weight) * total_tokens / avg_length if avg_length > 0 else 0.0
            for (weight, _, _), avg_length in zip(active_sources, avg_lengths)
        ]
    else:
        total_len = sum(len(prompts) for _, _, prompts in active_sources)
        avg_lengths = [0.0] * len(active_sources)
        count_weights = [(weight / total_weight) * total_len for weight, _, _ in active_sources]
    selected = []
    for _weight, kind, prompts in per_source:
        if not prompts:
            logger.warning(f"Blend source {kind!r} unpacked 0 conversations; skipping")
    for source_idx, (weight, kind, prompts) in enumerate(active_sources):
        target = int(round(count_weights[source_idx]))
        n = len(prompts)
        if target <= n:
            idx = rng.choice(n, size=target, replace=False)
        else:
            extra = rng.choice(n, size=target - n, replace=True)
            idx = np.concatenate([np.arange(n), extra])
        logger.info(
            f"Blend source {kind!r}: {n} available, weight={weight}, "
            f"avg_tokens={avg_lengths[source_idx]:.1f}, unit={blend_unit}, "
            f"selected={len(idx)}"
        )
        for j in idx:
            selected.append(prompts[int(j)])

    order = rng.permutation(len(selected))
    blended = []
    for i, pos in enumerate(order):
        e = selected[int(pos)]
        ne = dict(e)
        ne["data_id"] = f"{e['data_id']}#{i}"  # unique key for mooncake hidden-state store
        blended.append(ne)
    logger.info(
        f"Megatron-bin blend done: {len(blended)} conversations from "
        f"{len(active_sources)} active sources"
    )
    return blended
