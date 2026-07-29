# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import IterableDataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from angelspec.models.ops.loss_mask import compute_assistant_loss_mask
from angelspec.utils.logging import logger

_LOCAL_DATA_EXTS = frozenset({".json", ".jsonl", ".parquet", ".arrow", ".csv", ".tsv", ".txt"})


def is_local_data_path(path: str, base_dir: str | None = None) -> bool:
    """True if *path* looks like a local file/directory rather than a HF Hub dataset ID.

    When *base_dir* is given, relative paths are probed against it instead of
    the process CWD.
    """
    if path.startswith((".", "/", "~")):
        return True
    if os.path.splitext(path)[1].lower() in _LOCAL_DATA_EXTS:
        return True
    probe = os.path.join(base_dir, path) if base_dir is not None else path
    return os.path.exists(probe)


class DataCollatorWithPadding:
    def __init__(self, usp_enabled: bool = False):
        self.sp_degree = 1
        self.usp_enabled = usp_enabled

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n, S = intensors.shape
        # Truncate if longer than target (can happen when loss_mask/hidden_states
        # length differs from input_ids after unpacking).
        if n > N:
            return intensors[:, :N, :]
        if n == N:
            return intensors
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        B, n = intensors.shape
        # Truncate if longer than target (prevents negative padding dimension
        # when loss_mask length differs from input_ids after collation).
        if n > N:
            return intensors[:, :N]
        if n == N:
            return intensors
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def _get_loss_mask(self, item: Dict[str, Any]) -> torch.Tensor:
        """Read the materialized loss_mask tensor from the item.

        Callers (e.g. MooncakeDataset) are responsible for computing and
        attaching loss_mask before items reach the collator.
        """
        if "loss_mask" in item and isinstance(item["loss_mask"], torch.Tensor):
            return item["loss_mask"]
        raise KeyError(f"loss_mask not found in item: {item}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        max_length = ((max_length + self.sp_degree - 1) // self.sp_degree) * self.sp_degree

        if self.usp_enabled:
            attention_masks = [item["attention_mask"].long() for item in features]
        else:
            # Round up to nearest bucket to reduce unique shapes for torch.compile.
            # Without this, every batch gets a different padded length, causing
            # FlexAttention recompilation (~1s overhead per new shape).
            _BUCKET = 256
            max_length = ((max_length + _BUCKET - 1) // _BUCKET) * _BUCKET
            attention_masks = [torch.ones_like(item["input_ids"]).long() for item in features]

        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(mask, max_length) for mask in attention_masks]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(self._get_loss_mask(item), max_length) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target": None,
            "last_hidden_states": None,
        }
        if self.usp_enabled:
            max_position_length = max(item["position_ids"].shape[1] for item in features)
            batch["position_ids"] = torch.cat(
                [
                    self.paddingtensor2D(item["position_ids"], max_position_length)
                    for item in features
                ]
            )
        if all("hidden_states" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [self.paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            has_target = all(item.get("target") is not None for item in features)
            has_last_hs = all(item.get("last_hidden_states") is not None for item in features)
            if not has_target and not has_last_hs:
                pass
            if has_target:
                batch["target"] = torch.cat(
                    [self.paddingtensor(item["target"], max_length) for item in features]
                )
            if has_last_hs:
                batch["last_hidden_states"] = torch.cat(
                    [
                        self.paddingtensor(item["last_hidden_states"], max_length)
                        for item in features
                    ]
                )
        return batch


def _packing_loss_mask(item: Dict[str, Any], sample_len: int, owner: str) -> torch.Tensor:
    loss_mask = item.get("loss_mask")
    if not isinstance(loss_mask, torch.Tensor):
        raise KeyError(f"{owner}: loss_mask tensor not found in item")
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)
    if loss_mask.dim() != 2 or loss_mask.shape[0] != 1:
        raise ValueError(f"{owner}: expected loss_mask shape [1, S], got {tuple(loss_mask.shape)}")
    if loss_mask.shape[1] < sample_len:
        raise ValueError(
            f"{owner}: loss_mask length {loss_mask.shape[1]} is shorter than input_ids length {sample_len}"
        )
    # Packed bitmasks may unpack to a byte-aligned length slightly larger than
    # input_ids. Extra bits are padding and are safe to truncate.
    return loss_mask[:, :sample_len]


def _packing_hidden(
    hidden: Any,
    sample_len: int,
    owner: str,
    field_name: str,
) -> torch.Tensor:
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"{owner}: {field_name} must be a tensor")
    if hidden.dim() != 3 or hidden.shape[0] != 1:
        raise ValueError(
            f"{owner}: expected {field_name} shape [1, S, D], got {tuple(hidden.shape)}"
        )
    if hidden.shape[1] < sample_len:
        raise ValueError(
            f"{owner}: {field_name} length {hidden.shape[1]} is shorter than input_ids length {sample_len}"
        )
    # Be defensive against byte-/buffer-aligned over-allocation while preserving
    # exact token alignment for every following packed document.
    return hidden[:, :sample_len, :]


class DFlashPackingCollator:
    """Pack multiple samples into ONE fixed-length sequence for DFlash training.

    Instead of pad-to-longest micro-batches, this collator concatenates the
    given samples end-to-end into a single ``[1, max_seq_length]`` sequence and
    pads the tail to a fixed length, emitting per-token document ids and
    doc-local RoPE positions so the DFlash forward keeps packed documents from
    attending across boundaries.

    Interface contract (deliberately source-agnostic): accepts *any* number of
    input samples and always returns exactly one packed row of length
    ``max_seq_length``. Samples may come from a DataLoader's fixed ``batch_size``
    grouping or from a dynamic bin built in ``MooncakeDataset``.

    Overflow policy: if appending a sample would exceed ``max_seq_length``, the
    WHOLE sample is dropped (never truncated) and a warning is logged. Truncating a
    DFlash sample would silently drop response tokens (input_ids and hidden_states
    are a token-aligned whole), corrupting the data distribution. The streaming
    binner re-bins overflow samples instead, so nothing is dropped there.

    Output batch keys (all with a leading batch dim of 1):
      - input_ids:         [1, max_seq]
      - attention_mask:    [1, max_seq]  (1 for real tokens, 0 for tail padding)
      - loss_mask:         [1, max_seq]
      - hidden_states:     [1, max_seq, D]  (present iff all inputs carry it)
      - ctx_doc_ids:       [1, max_seq] long   (0..K-1 per doc, -1 for padding)
      - base_position_ids: [1, max_seq] long   (doc-local, resets to 0 per doc)
      - target: None (DFlash path does not use it)
      - last_hidden_states: [1, max_seq, D] (present iff all inputs carry it; for DSpark L1)
    """

    def __init__(self, max_seq_length: int):
        if max_seq_length is None or max_seq_length <= 0:
            raise ValueError(
                f"DFlashPackingCollator requires max_seq_length > 0, got {max_seq_length}"
            )
        self.max_seq_length = int(max_seq_length)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_seq = self.max_seq_length
        hidden_presence = [item.get("hidden_states") is not None for item in features]
        if any(hidden_presence) and not all(hidden_presence):
            raise ValueError(
                "DFlashPackingCollator: hidden_states must be present for either all samples or none"
            )
        have_hidden = bool(features) and all(hidden_presence)

        last_hs_presence = [item.get("last_hidden_states") is not None for item in features]
        if any(last_hs_presence) and not all(last_hs_presence):
            raise ValueError(
                "DFlashPackingCollator: last_hidden_states must be present for either all samples or none"
            )
        have_last_hidden = bool(features) and all(last_hs_presence)

        # Greedily append samples until the next one would overflow max_seq.
        # Dropped (never truncated) samples are logged.
        seg_input_ids: List[torch.Tensor] = []
        seg_hidden: List[torch.Tensor] = []
        seg_last_hidden: List[torch.Tensor] = []
        seg_loss_mask: List[torch.Tensor] = []
        seg_doc_ids: List[torch.Tensor] = []
        seg_positions: List[torch.Tensor] = []
        used = 0
        doc_id = 0
        dropped = 0

        for item in features:
            ids = item["input_ids"]  # [1, S]
            if not isinstance(ids, torch.Tensor) or ids.dim() != 2 or ids.shape[0] != 1:
                shape = tuple(ids.shape) if isinstance(ids, torch.Tensor) else type(ids).__name__
                raise ValueError(
                    f"DFlashPackingCollator: expected input_ids shape [1, S], got {shape}"
                )
            s = ids.shape[1]
            if s == 0:
                continue
            if s > max_seq:
                # A single sample longer than the whole packed row can never fit;
                # this should not happen (dataset caps at max_seq_length) but guard.
                logger.warning(
                    f"DFlashPackingCollator: dropping sample of length {s} > max_seq_length {max_seq} (cannot pack)."
                )
                dropped += 1
                continue
            if used + s > max_seq:
                # Overflow: drop the whole sample (the streaming binner re-bins).
                dropped += 1
                continue

            lm = _packing_loss_mask(item, s, "DFlashPackingCollator")
            seg_input_ids.append(ids)
            seg_loss_mask.append(lm)
            seg_doc_ids.append(torch.full((1, s), doc_id, dtype=torch.long))
            seg_positions.append(torch.arange(s, dtype=torch.long).unsqueeze(0))
            if have_hidden:
                seg_hidden.append(
                    _packing_hidden(
                        item["hidden_states"],
                        s,
                        "DFlashPackingCollator",
                        "hidden_states",
                    )
                )
            if have_last_hidden:
                seg_last_hidden.append(
                    _packing_hidden(
                        item["last_hidden_states"],
                        s,
                        "DFlashPackingCollator",
                        "last_hidden_states",
                    )
                )
            used += s
            doc_id += 1

        if dropped:
            logger.warning(
                f"DFlashPackingCollator: dropped {dropped}/{len(features)} sample(s) "
                f"that did not fit into a single max_seq={max_seq} packed row."
            )

        if used == 0:
            # Degenerate batch (all dropped/empty): emit an all-padding row so the
            # training step is a well-formed no-op (loss_mask all zero → zero loss).
            logger.warning("DFlashPackingCollator: empty packed row (all samples dropped).")
            pad_ids = torch.zeros(1, max_seq, dtype=torch.long)
            batch = {
                "input_ids": pad_ids,
                "attention_mask": torch.zeros(1, max_seq, dtype=torch.long),
                "loss_mask": torch.zeros(1, max_seq, dtype=torch.long),
                "ctx_doc_ids": torch.full((1, max_seq), -1, dtype=torch.long),
                "base_position_ids": torch.zeros(1, max_seq, dtype=torch.long),
                "hidden_states": None,
                "target": None,
                "last_hidden_states": None,
            }
            return batch

        pad = max_seq - used

        def _pad2d(parts: List[torch.Tensor], fill: int, dtype) -> torch.Tensor:
            cat = torch.cat(parts, dim=1)
            if pad > 0:
                tail = torch.full((1, pad), fill, dtype=dtype, device=cat.device)
                cat = torch.cat([cat.to(dtype), tail], dim=1)
            return cat.to(dtype)

        input_ids = _pad2d(seg_input_ids, 0, torch.long)
        # loss_mask fill 0 = padding excluded from loss.
        loss_mask = _pad2d(seg_loss_mask, 0, seg_loss_mask[0].dtype)
        # ctx_doc_ids fill -1 = padding (mask_mod requires a_doc>=0 to attend).
        ctx_doc_ids = _pad2d(seg_doc_ids, -1, torch.long)
        base_position_ids = _pad2d(seg_positions, 0, torch.long)
        attention_mask = torch.cat(
            [torch.ones(1, used, dtype=torch.long), torch.zeros(1, pad, dtype=torch.long)], dim=1
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "ctx_doc_ids": ctx_doc_ids,
            "base_position_ids": base_position_ids,
            "hidden_states": None,
            "target": None,
            "last_hidden_states": None,
        }

        if have_hidden:
            d = seg_hidden[0].shape[-1]
            hs_cat = torch.cat(seg_hidden, dim=1)  # [1, used, D]
            if pad > 0:
                tail = torch.zeros(1, pad, d, dtype=hs_cat.dtype, device=hs_cat.device)
                hs_cat = torch.cat([hs_cat, tail], dim=1)
            batch["hidden_states"] = hs_cat

        if have_last_hidden:
            d = seg_last_hidden[0].shape[-1]
            lhs_cat = torch.cat(seg_last_hidden, dim=1)  # [1, used, D]
            if pad > 0:
                tail = torch.zeros(1, pad, d, dtype=lhs_cat.dtype, device=lhs_cat.device)
                lhs_cat = torch.cat([lhs_cat, tail], dim=1)
            batch["last_hidden_states"] = lhs_cat

        return batch


class MTPPackingCollator:
    """Pack samples into one fixed-length sequence for single-head MTP.

    Like DFlashPackingCollator (concatenate end-to-end, emit per-token ctx_doc_ids
    and doc-local base_position_ids) but packs the target's ``last_hidden_states``
    (MTP's draft input + teacher hidden), not the multi-layer ``hidden_states``.
    """

    def __init__(self, max_seq_length: int):
        if max_seq_length is None or max_seq_length <= 0:
            raise ValueError(
                f"MTPPackingCollator requires max_seq_length > 0, got {max_seq_length}"
            )
        self.max_seq_length = int(max_seq_length)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_seq = self.max_seq_length
        hidden_presence = [item.get("last_hidden_states") is not None for item in features]
        if any(hidden_presence) and not all(hidden_presence):
            raise ValueError(
                "MTPPackingCollator: last_hidden_states must be present for either all samples or none"
            )
        have_hidden = bool(features) and all(hidden_presence)

        seg_input_ids: List[torch.Tensor] = []
        seg_hidden: List[torch.Tensor] = []
        seg_loss_mask: List[torch.Tensor] = []
        seg_doc_ids: List[torch.Tensor] = []
        seg_positions: List[torch.Tensor] = []
        used = 0
        doc_id = 0
        dropped = 0

        for item in features:
            ids = item["input_ids"]  # [1, S]
            if not isinstance(ids, torch.Tensor) or ids.dim() != 2 or ids.shape[0] != 1:
                shape = tuple(ids.shape) if isinstance(ids, torch.Tensor) else type(ids).__name__
                raise ValueError(
                    f"MTPPackingCollator: expected input_ids shape [1, S], got {shape}"
                )
            s = ids.shape[1]
            if s == 0:
                continue
            if s > max_seq:
                logger.warning(
                    f"MTPPackingCollator: dropping sample of length {s} > max_seq_length {max_seq} (cannot pack)."
                )
                dropped += 1
                continue
            if used + s > max_seq:
                # Overflow: drop the whole sample (never truncate); the streaming
                # binner re-bins it. Truncating would drop supervised tokens.
                dropped += 1
                continue

            lm = _packing_loss_mask(item, s, "MTPPackingCollator")
            seg_input_ids.append(ids)
            seg_loss_mask.append(lm)
            seg_doc_ids.append(torch.full((1, s), doc_id, dtype=torch.long))
            seg_positions.append(torch.arange(s, dtype=torch.long).unsqueeze(0))
            if have_hidden:
                seg_hidden.append(
                    _packing_hidden(
                        item["last_hidden_states"],
                        s,
                        "MTPPackingCollator",
                        "last_hidden_states",
                    )
                )
            used += s
            doc_id += 1

        if dropped:
            logger.warning(
                f"MTPPackingCollator: dropped {dropped}/{len(features)} sample(s) "
                f"that did not fit into a single max_seq={max_seq} packed row."
            )

        if used == 0:
            logger.warning("MTPPackingCollator: empty packed row (all samples dropped).")
            pad_ids = torch.zeros(1, max_seq, dtype=torch.long)
            return {
                "input_ids": pad_ids,
                "attention_mask": torch.zeros(1, max_seq, dtype=torch.long),
                "loss_mask": torch.zeros(1, max_seq, dtype=torch.long),
                "ctx_doc_ids": torch.full((1, max_seq), -1, dtype=torch.long),
                "base_position_ids": torch.zeros(1, max_seq, dtype=torch.long),
                "hidden_states": None,
                "last_hidden_states": None,
                "target": None,
            }

        pad = max_seq - used

        def _pad2d(parts: List[torch.Tensor], fill: int, dtype) -> torch.Tensor:
            cat = torch.cat(parts, dim=1)
            if pad > 0:
                tail = torch.full((1, pad), fill, dtype=dtype, device=cat.device)
                cat = torch.cat([cat.to(dtype), tail], dim=1)
            return cat.to(dtype)

        input_ids = _pad2d(seg_input_ids, 0, torch.long)
        loss_mask = _pad2d(seg_loss_mask, 0, seg_loss_mask[0].dtype)
        # ctx_doc_ids fill -1 = padding (doc-gate / mask require doc>=0 to attend).
        ctx_doc_ids = _pad2d(seg_doc_ids, -1, torch.long)
        base_position_ids = _pad2d(seg_positions, 0, torch.long)
        attention_mask = torch.cat(
            [torch.ones(1, used, dtype=torch.long), torch.zeros(1, pad, dtype=torch.long)], dim=1
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "ctx_doc_ids": ctx_doc_ids,
            "base_position_ids": base_position_ids,
            "hidden_states": None,
            "last_hidden_states": None,
            "target": None,
        }

        if have_hidden:
            d = seg_hidden[0].shape[-1]
            hs_cat = torch.cat(seg_hidden, dim=1)  # [1, used, D]
            if pad > 0:
                tail = torch.zeros(1, pad, d, dtype=hs_cat.dtype, device=hs_cat.device)
                hs_cat = torch.cat([hs_cat, tail], dim=1)
            batch["last_hidden_states"] = hs_cat

        return batch


def pack_loss_mask(loss_mask: torch.Tensor) -> List[int]:
    """
    Pack a loss_mask tensor into interleaved segment lengths.

    The returned list alternates between prompt and response lengths,
    always starting with prompt (even if length 0).

    Args:
        loss_mask: 1D tensor of 0s and 1s indicating which tokens contribute to loss.

    Returns:
        List of segment lengths: [prompt_len, response_len, prompt_len, response_len, ...]

    Example:
        loss_mask = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        returns: [2, 3, 2, 2, 1]  # 2 prompt, 3 response, 2 prompt, 2 response, 1 prompt
    """
    if loss_mask.dim() > 1:
        loss_mask = loss_mask.squeeze()

    if len(loss_mask) == 0:
        return []

    lengths = []
    mask_list = loss_mask.tolist()
    current_val = 0
    current_len = 0

    for val in mask_list:
        if val == current_val:
            current_len += 1
        else:
            lengths.append(current_len)
            current_val = val
            current_len = 1

    lengths.append(current_len)
    return lengths


def unpack_loss_mask(packed: Union[List[int], str]) -> torch.Tensor:
    """
    Unpack segment lengths back into a loss_mask tensor.

    Args:
        packed: List of segment lengths [prompt_len, response_len, prompt_len, ...],
                or a serialized packed_loss_mask string (e.g. "2,3,2,2,1").

    Returns:
        1D tensor of 0s and 1s.

    Example:
        packed = [2, 3, 2, 2, 1]
        returns: tensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    """
    if isinstance(packed, str):
        packed = deserialize_packed_loss_mask(packed)
    if not packed:
        return torch.tensor([], dtype=torch.long)

    total = sum(packed)
    loss_mask = torch.zeros(total, dtype=torch.long)
    pos = 0

    for i, length in enumerate(packed):
        if i % 2 == 1:
            loss_mask[pos : pos + length] = 1
        pos += length

    return loss_mask


def resolve_loss_mask(
    data: Dict[str, Any],
    *,
    dynamic_loss_mask: bool = False,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
) -> torch.Tensor | None:
    """
    Two strategies, tried in order:
    1. ``packed_loss_mask`` key present → unpack it.
    2. ``dynamic_loss_mask`` enabled with valid header/end ids → compute from
       ``input_ids`` via :func:`compute_assistant_loss_mask`.
    """
    packed = data.get("packed_loss_mask")
    if packed is not None:
        mask = unpack_loss_mask(packed)
        input_ids = data.get("input_ids")
        if input_ids is not None:
            if input_ids.dim() == 2:
                input_ids = input_ids.squeeze(0)
            expected_len = input_ids.shape[-1]
            if mask.shape[0] > expected_len:
                mask = mask[:expected_len]
            elif mask.shape[0] < expected_len:
                mask = torch.nn.functional.pad(mask, (0, expected_len - mask.shape[0]))
        if not mask.any():
            return None
        data["loss_mask"] = mask
        return mask

    if dynamic_loss_mask and assistant_header_ids and end_token_ids:
        input_ids = data.get("input_ids")
        if input_ids is None:
            return None
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        per_sample = data.get("last_turn_loss_only")
        last_turn_only = per_sample if per_sample is not None else last_turn_loss_only
        mask = compute_assistant_loss_mask(
            input_ids,
            assistant_header_ids,
            end_token_ids,
            last_turn_only=last_turn_only,
            skip_after_header=skip_after_header,
        )
        if not mask.any():
            return None
        data["loss_mask"] = mask
        return mask

    return torch.ones(1)


def serialize_packed_loss_mask(packed: List[int]) -> str:
    """
    Serialize packed loss_mask to a comma-separated string.

    Args:
        packed: List of segment lengths from pack_loss_mask().

    Returns:
        Comma-separated string of integers.

    Example:
        packed = [2, 3, 2, 2, 1]
        returns: "2,3,2,2,1"
    """
    return ",".join(str(x) for x in packed)


def deserialize_packed_loss_mask(s: str) -> List[int]:
    """
    Deserialize a comma-separated string back to packed loss_mask.

    Args:
        s: Comma-separated string of integers.

    Returns:
        List of segment lengths.

    Example:
        s = "2,3,2,2,1"
        returns: [2, 3, 2, 2, 1]
    """
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def extract_media_urls(messages: list) -> dict | None:
    """Extract image/video URLs from structured messages without loading them."""
    images = []
    videos = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                if "image" in item:
                    images.append(item["image"])
                else:
                    images.append(None)
            elif item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and "url" in image_url:
                    images.append(image_url["url"])
                else:
                    images.append(None)
            elif item_type == "video":
                if "video" in item:
                    videos.append(item["video"])
                else:
                    videos.append(None)

    if not images and not videos:
        return None
    return {"images": images or None, "videos": videos or None}


def flatten_multimodal_content(messages, image_placeholder="<image>"):
    """Convert list-type content parts to plain text strings.

    Transforms the standard HF multimodal format:
      [{"type":"image"}, {"type":"text","text":"Describe"}]
    into a single string:
      "<image>\\nDescribe"

    Messages with string content are left unchanged.
    Must be called AFTER extract_media_urls so structured info is captured first.
    """
    for msg in messages:
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
            continue
        if not isinstance(content, (str, list)):
            raise ValueError(
                f"Message content must be a str or list, got {type(content).__name__}: {repr(content)[:100]}"
            )
        if not isinstance(content, list):
            continue
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif part_type in ("image", "image_url"):
                text_parts.append(image_placeholder)
            elif part_type == "video":
                text_parts.append("<video>")
        msg["content"] = "\n".join(text_parts)
    return messages


def estimate_row_count(data_path):
    if not os.path.isfile(data_path):
        return None
    if data_path.endswith(".jsonl"):
        with open(data_path, "rb") as f:
            return sum(1 for _ in f)
    if data_path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            return pq.ParquetFile(data_path).metadata.num_rows
        except Exception:
            return None
    if data_path.endswith(".json"):
        return None
    return None


def load_local_json(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            yield from json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_local_json_files(data_files):
    """Yield rows from multiple local JSON/JSONL files via the raw json module.

    Used for local directories with several JSON/JSONL shards. ``load_dataset``
    would force every shard into one unified PyArrow schema and raise CastError
    when shards differ (e.g. some carry a ``tools`` column and some don't); the
    raw json path is schema-agnostic, matching ``_load_hub_json_files``.
    """
    for fp in data_files:
        yield from load_local_json(fp)


def _list_hub_data_files(data_path: str, suffixes: tuple[str, ...]) -> list[str]:
    """List repo files matching ``suffixes``, preferring those under ``data/``."""
    files = list_repo_files(data_path, repo_type="dataset")
    matching = sorted(f for f in files if f.endswith(suffixes))
    preferred = [f for f in matching if f.startswith("data/")]
    return preferred or matching


def _load_hub_json_files(data_path):
    """Download JSON/JSONL files from a HF Hub dataset and yield rows.

    Uses raw json module instead of load_dataset to avoid PyArrow schema
    inference failures on datasets with mixed-type columns.
    """
    data_files = _list_hub_data_files(data_path, (".jsonl", ".json"))
    if not data_files:
        raise ValueError(f"No JSON/JSONL files found in HF Hub dataset '{data_path}'.")

    for filename in data_files:
        local_path = hf_hub_download(repo_id=data_path, filename=filename, repo_type="dataset")
        yield from load_local_json(local_path)


def _load_hub_parquet_dataset(data_path: str):
    """Stream parquet files from a HF Hub dataset.

    Used when ``load_dataset(repo_id, ...)`` fails due to schema/card parsing
    issues but the repo's parquet shards themselves are valid.
    """
    parquet_files = _list_hub_data_files(data_path, (".parquet",))
    if not parquet_files:
        raise ValueError(f"No parquet files found in HF Hub dataset '{data_path}'.")
    urls = [f"https://huggingface.co/datasets/{data_path}/resolve/main/{f}" for f in parquet_files]
    return load_dataset("parquet", data_files=urls, split="train", streaming=True)


def load_hf_dataset(data_path: str):
    """Load dataset as a streaming IterableDataset.

    Local paths are loaded directly; everything else goes to HF Hub.
    """
    data_path = os.path.expanduser(data_path)

    if is_local_data_path(data_path):
        if os.path.isfile(data_path):
            if data_path.endswith((".json", ".jsonl")):
                return IterableDataset.from_generator(
                    load_local_json, gen_kwargs={"data_path": data_path}
                )
            ext = os.path.splitext(data_path)[1].lower()
            fmt = {".parquet": "parquet", ".arrow": "arrow"}.get(ext, "json")
            return load_dataset(fmt, data_files=data_path, split="train", streaming=True)

        if os.path.isdir(data_path):
            # JSON/JSONL: stream via the raw json module so heterogeneous shards
            # (differing optional columns) don't trip load_dataset's single
            # unified PyArrow schema (CastError). Parquet/Arrow keep load_dataset.
            json_files = sorted(
                str(p) for g in ("*.json", "*.jsonl") for p in Path(data_path).rglob(g)
            )
            if json_files:
                return IterableDataset.from_generator(
                    load_local_json_files, gen_kwargs={"data_files": json_files}
                )
            patterns = {
                "parquet": ["*.parquet"],
                "arrow": ["*.arrow"],
            }
            for fmt, globs in patterns.items():
                files = []
                for g in globs:
                    files.extend(str(p) for p in Path(data_path).rglob(g))
                if files:
                    return load_dataset(
                        fmt, data_files=sorted(files), split="train", streaming=True
                    )
            raise ValueError(f"No supported dataset files found in local directory: {data_path}")

        raise FileNotFoundError(f"Local dataset path not found: {data_path}")

    # hub path — try native load_dataset first (handles Arrow, Parquet, etc.),
    # fall back to manual JSON download for repos with mixed-type columns
    _KEEP_COLUMNS = frozenset({"id", "conversations", "text", "messages"})
    try:
        ds = load_dataset(data_path, split="train", streaming=True)
        drop_cols = [c for c in (ds.column_names or []) if c not in _KEEP_COLUMNS]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)
        return ds
    except (ValueError, TypeError, ArithmeticError, KeyError) as e:
        # Schema inference / card-parsing failures (e.g., mixed-type columns in
        # Arrow/Parquet, malformed dataset_info YAML). Fall back to direct file
        # streaming — parquet first, then JSON/JSONL.
        import logging

        log = logging.getLogger(__name__)
        log.info(f"load_dataset failed for '{data_path}' ({e}), trying direct file streaming")
        try:
            return _load_hub_parquet_dataset(data_path)
        except ValueError as parquet_err:
            log.info(f"parquet streaming failed ({parquet_err}), falling back to JSON download")
            return IterableDataset.from_generator(
                _load_hub_json_files, gen_kwargs={"data_path": data_path}
            )
