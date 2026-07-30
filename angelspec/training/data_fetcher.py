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

"""Queue-based data fetching with mooncake store.
Data flow:
  TrainActor -> MooncakeDataFetcher -> MooncakeDataset -> MooncakeStore -> Collator
                     |                      |                  |               |
                iter(fetcher)          queue.get()      store.get(key)     pad & batch
"""

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ray.util.queue import Queue as RayQueue
from torch.utils.data import DataLoader, IterableDataset

from angelspec.data.utils import (
    deserialize_packed_loss_mask,
    resolve_loss_mask,
    unpack_loss_mask,
)
from angelspec.utils.distributed import (
    get_draft_sp_group,
    get_sp_ring_group,
    get_usp_rank_coords,
)
from angelspec.utils.logging import logger
from angelspec.utils.usp import usp_chunk_size


@dataclass
class TrainSample:
    mooncake_key: str
    tensor_shapes: Dict[str, Tuple[int, ...]]
    tensor_dtypes: Optional[Dict[str, torch.dtype]] = None
    packed_loss_mask: Optional[str] = None
    last_turn_loss_only: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    # Fixed-rows packing: the controller stamps a per-queue monotonic row_id and
    # enqueues one row's samples consecutively, marking the last with last_in_row.
    # The trainer flushes on that flag (no cross-step look-ahead → no deadlock).
    # None/False on the non-packing/eval path.
    row_id: Optional[int] = None
    last_in_row: bool = False


class MooncakeDataset(IterableDataset):
    """IterableDataset that loads from mooncake via queue.

    Each DP rank waits on its queue for TrainSample items sent by the
    centralized controller. Data is loaded from mooncake.
    """

    def __init__(
        self,
        ray_queue: RayQueue,
        mooncake_store,
        device: torch.device,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        batch_size: int = 1,
        min_loss_tokens: int = 0,
        usp_enabled: bool = False,
        ttt_length: int = 1,
        max_seq_length: Optional[int] = None,
        usp_local_shard: bool = False,
        packing_enabled: bool = False,
    ):
        self.ray_queue = ray_queue
        self.mooncake_store = mooncake_store
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.assistant_header_ids = assistant_header_ids
        self.end_token_ids = end_token_ids
        self.dynamic_loss_mask = dynamic_loss_mask
        self.last_turn_loss_only = last_turn_loss_only
        self.skip_after_header = skip_after_header
        self._batch_size = batch_size
        self._min_loss_tokens = min_loss_tokens
        self.usp_enabled = usp_enabled
        # USP local-shard: the same Mooncake key is fanned to all SP ranks (each
        # slices its own seq shard). A rank must NOT delete the key — peers still need
        # it; cleanup is suppressed, memory bounded by the pool + segment sizing.
        self.usp_local_shard = usp_local_shard
        self.ttt_length = ttt_length
        self.max_seq_length = max_seq_length
        # DFlash sequence packing: greedily bin queue samples into fixed-length rows
        # in __iter__; each yielded item is a LIST of per-sample dicts (a bin) consumed
        # whole by DFlashPackingCollator. Requires max_seq_length; mutually excl. w/ USP.
        self.packing_enabled = packing_enabled
        if packing_enabled and (max_seq_length is None or max_seq_length <= 0):
            raise ValueError("packing_enabled requires a positive max_seq_length")
        if packing_enabled and usp_enabled:
            raise NotImplementedError("packing_enabled is incompatible with USP")
        self._init_sp_context()

    def _init_sp_context(self) -> None:
        self._sp_group = None
        self._sp_world_size = 1
        self._sp_rank = 0
        self._sp_ring_size = 1
        self._sp_ring_rank = 0
        if not self.usp_enabled:
            return

        sp_group = get_draft_sp_group()
        if sp_group is None:
            return

        self._sp_group = sp_group
        self._sp_world_size = dist.get_world_size(sp_group)
        self._sp_rank = dist.get_rank(sp_group)

        ring_group = get_sp_ring_group()
        if ring_group is not None:
            self._sp_ring_size = dist.get_world_size(ring_group)
            self._sp_ring_rank = dist.get_rank(ring_group)

    def _load_from_mooncake(self, sample: TrainSample) -> Dict[str, Any]:
        """Load tensors from mooncake key into device memory."""
        dtypes_raw = sample.tensor_dtypes or {}

        # Convert string dtypes to torch.dtype objects
        dtypes = {}
        for key, dtype_val in dtypes_raw.items():
            if isinstance(dtype_val, str):
                # Handle "bfloat16" or "torch.bfloat16" format
                dtype_str = dtype_val.replace("torch.", "")
                dtypes[key] = getattr(torch, dtype_str)
            else:
                dtypes[key] = dtype_val

        logger.debug(
            f"_load_from_mooncake: key={sample.mooncake_key}, requesting shapes={sample.tensor_shapes}"
        )

        tensors = self.mooncake_store.get(
            key=sample.mooncake_key,
            shapes=sample.tensor_shapes,
            dtypes=dtypes,
            device=self.device,
        )

        tensor_dict = tensors.to_tensor_dict()
        if self._batch_size > 1 or self.packing_enabled:
            # Clone to prevent use-after-free: the collator/packing bin holds sample N
            # while later samples are fetched, but cleanup frees the Mooncake buffer
            # (Issue 31). Note: clone() unpins, breaking non_blocking H2D — only here.
            result = {k: v.clone() for k, v in tensor_dict.items()}
        else:
            # batch_size=1: safe to use pinned views — consumed immediately.
            # Preserves pinned memory for async H2D via non_blocking=True.
            result = dict(tensor_dict)

        self._cleanup_mooncake_data(sample)
        if sample.packed_loss_mask is not None:
            result["packed_loss_mask"] = sample.packed_loss_mask
        if sample.last_turn_loss_only is not None:
            result["last_turn_loss_only"] = sample.last_turn_loss_only
        return result

    def _cleanup_mooncake_data(self, sample: TrainSample) -> None:
        """Remove data from mooncake store to release buffer space."""
        if self.usp_local_shard:
            # All SP ranks share this key; a per-rank delete would starve peers.
            # The controller pool / Mooncake segment bound memory instead.
            return
        shapes = sample.tensor_shapes or {}
        has_lhs = "last_hidden_states" in shapes
        has_target = "target" in shapes

        self.mooncake_store.remove_eagle3_tensors(
            sample.mooncake_key,
            has_last_hidden_states=has_lhs,
            has_target=has_target,
        )

    def _compute_loss_mask(self, data: Dict[str, Any]) -> torch.Tensor | None:
        return resolve_loss_mask(
            data,
            dynamic_loss_mask=self.dynamic_loss_mask,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
        )

    @staticmethod
    def _fallback_mask_len(data: Dict[str, Any]) -> int:
        """Sequence length for a neutralized sample's zero loss mask."""
        ids = data.get("input_ids")
        if isinstance(ids, torch.Tensor) and ids.dim() >= 1:
            return ids.shape[-1]
        for key in ("hidden_states", "last_hidden_states", "target"):
            t = data.get(key)
            if isinstance(t, torch.Tensor) and t.dim() >= 2:
                return t.shape[-2]
        return 1

    def _resolve_and_neutralize_loss_mask(
        self, data: Dict[str, Any], mooncake_key: str, neutralized_count: int
    ) -> int:
        """Zero an empty / sub-min_loss_tokens loss mask in place instead of
        dropping the sample; a per-rank drop desyncs FSDP collectives."""
        mask = self._compute_loss_mask(data)  # None == all-zero
        neutralize = mask is None or (
            self._min_loss_tokens > 0
            and isinstance(mask, torch.Tensor)
            and mask.sum() < self._min_loss_tokens
        )
        if not neutralize:
            return neutralized_count

        if isinstance(mask, torch.Tensor):
            n_tokens = int(mask.sum())
            data["loss_mask"] = torch.zeros_like(mask)
        else:
            # resolve_loss_mask returns None without setting data["loss_mask"]
            n_tokens = 0
            data["loss_mask"] = torch.zeros(self._fallback_mask_len(data), dtype=torch.long)

        neutralized_count += 1
        logger.warning(
            f"Neutralized loss mask ({n_tokens} < min_loss_tokens="
            f"{self._min_loss_tokens}, mooncake_key={mooncake_key}, "
            f"total_neutralized={neutralized_count})"
        )
        return neutralized_count

    def _prepare_one(self, item, neutralized_count: int):
        """Load + resolve loss mask + add batch dim for a single queue item.

        Returns (data_dict, updated_neutralized_count). Factored out of __iter__
        so both the per-sample and the packed (bin) iteration paths share it.
        """
        data = self._load_from_mooncake(item)

        neutralized_count = self._resolve_and_neutralize_loss_mask(
            data, item.mooncake_key, neutralized_count
        )

        # Note: target is computed in the collator from last_hidden_states for sglang mode

        # Add batch dimension if missing (sglang stores without batch dim)
        for key, tensor in data.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                # 1D tensors (loss_mask, input_ids) should be 2D: (1, seq_len)
                # 2D tensors (hidden_states, last_hidden_states) should be 3D: (1, seq_len, dim)
                if tensor.dim() == 1:
                    data[key] = tensor.unsqueeze(0)
                elif tensor.dim() == 2 and key in [
                    "hidden_states",
                    "last_hidden_states",
                    "target",
                ]:
                    data[key] = tensor.unsqueeze(0)
        return data, neutralized_count

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples synchronously.

        Blocks waiting for each item from the queue and loads from mooncake.
        Empty / sub-min_loss_tokens masks are neutralized, not dropped
        (see _resolve_and_neutralize_loss_mask).

        In packing mode each yielded item is a LIST of sample dicts (one packed
        row, grouped by the controller-stamped row_id); otherwise a single dict.
        """
        if self.packing_enabled:
            yield from self._iter_rows_by_id()
            return

        yield_count = 0
        neutralized_count = 0
        while True:
            if self.usp_enabled:
                data, neutralized = self._usp_get_sharded_item(neutralized_count=neutralized_count)
                neutralized_count += neutralized
                if data is None:
                    break
                yield_count += 1
                yield data
                continue

            logger.debug(f"__iter__: waiting for item from ray_queue (yield_count={yield_count})")
            try:
                item = self.ray_queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.warning(
                    f"__iter__: Exception waiting for data: {e}, timeout={self.timeout}"
                )
                break

            if item is None:
                logger.debug("__iter__: received None sentinel, stopping iteration")
                break

            logger.debug(f"__iter__: got item, mooncake_key={item.mooncake_key}")
            data, neutralized_count = self._prepare_one(item, neutralized_count)

            if data:
                shapes_str = {
                    k: v.shape if hasattr(v, "shape") else type(v) for k, v in data.items()
                }
                logger.debug(f"final shapes (with batch dim): {shapes_str}")
            yield_count += 1
            logger.debug(f"__iter__: yielding batch {yield_count}, keys={list(data.keys())}")
            yield data

    def _iter_rows_by_id(self) -> Iterator[List[Dict[str, torch.Tensor]]]:
        """Assemble packed rows from the queue using controller-stamped row_id.

        The controller (pack_into_rows) is the SINGLE source of truth for how
        samples group into rows: it stamps each TrainSample with a per-queue
        monotonic ``row_id``, enqueues all samples of one row consecutively, and
        marks the LAST sample of each row with ``last_in_row=True``. The trainer
        does NOT re-pack — it accumulates samples into a bin and flushes the bin as
        soon as it sees ``last_in_row`` (no cross-step look-ahead, so training
        never blocks on a not-yet-dispatched next row → no dispatch/consume
        deadlock). One dispatch == one row per rank, so consecutive rows here ==
        consecutive optimizer-step micro-batches.
        """
        neutralized_count = 0
        cur_bin: List[Dict[str, torch.Tensor]] = []

        while True:
            try:
                item = self.ray_queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"_iter_rows_by_id: exception waiting for data: {e}")
                break
            if item is None:
                logger.debug("_iter_rows_by_id: received None sentinel, stopping")
                break

            if getattr(item, "row_id", None) is None:
                raise RuntimeError(
                    "_iter_rows_by_id received a TrainSample with row_id=None; "
                    "packing expects the controller to stamp row_id on every sample."
                )

            data, neutralized_count = self._prepare_one(item, neutralized_count)
            # Keep even zero-length/neutralized samples in the bin: they still
            # occupy their slot and their loss_mask (all-zero) is harmless. Dropping
            # them would desync the doc-aware collator's per-doc bookkeeping.
            cur_bin.append(data)

            if getattr(item, "last_in_row", False):
                yield cur_bin
                cur_bin = []

        if cur_bin:
            # Trailing partial row (sentinel/timeout mid-row) — emit what we have.
            yield cur_bin

    def _usp_global_len(self, sample: TrainSample) -> int:
        global_len = sample.tensor_shapes["input_ids"][-1]
        if self.max_seq_length is not None:
            global_len = min(global_len, self.max_seq_length)
        return global_len

    def _usp_chunk_size(self, global_len: int) -> int:
        return usp_chunk_size(global_len, self._sp_world_size)

    def _usp_loss_mask(self, sample: TrainSample, global_len: int) -> torch.Tensor:
        if sample.packed_loss_mask is None:
            raise RuntimeError("USP sharded Mooncake reads require packed_loss_mask metadata")
        loss_mask = unpack_loss_mask(deserialize_packed_loss_mask(sample.packed_loss_mask))
        loss_mask = loss_mask[:global_len]
        if loss_mask.shape[0] < global_len:
            loss_mask = F.pad(loss_mask, (0, global_len - loss_mask.shape[0]))
        return loss_mask

    def _local_usp_shapes(self, sample: TrainSample) -> dict[str, tuple[int, ...]]:
        local_len = self._usp_chunk_size(self._usp_global_len(sample)) + self.ttt_length
        shapes: dict[str, tuple[int, ...]] = {
            "input_ids": (1, local_len),
            "hidden_states": (1, local_len, sample.tensor_shapes["hidden_states"][-1]),
        }
        if "last_hidden_states" in sample.tensor_shapes:
            shapes["last_hidden_states"] = (
                1,
                local_len,
                sample.tensor_shapes["last_hidden_states"][-1],
            )
        if "target" in sample.tensor_shapes:
            shapes["target"] = (1, local_len, sample.tensor_shapes["target"][-1])
        return shapes

    def _local_usp_loss_and_position(
        self,
        sample: TrainSample,
        local_len: int,
    ) -> dict[str, torch.Tensor]:
        sp_ulysses_size = max(1, self._sp_world_size // self._sp_ring_size)
        global_len = self._usp_global_len(sample)
        chunk_size = self._usp_chunk_size(global_len)
        start = self._sp_rank * chunk_size
        end = min(start + local_len, global_len)
        valid_len = max(0, end - start)

        loss_mask = self._usp_loss_mask(sample, global_len)[start:end].unsqueeze(0)
        if loss_mask.shape[-1] < local_len:
            loss_mask = F.pad(loss_mask, (0, local_len - loss_mask.shape[-1]))

        attention_mask = torch.zeros((1, local_len), dtype=torch.long)
        attention_mask[:, :valid_len] = 1

        usp_chunk_size = max(local_len - self.ttt_length, 0)
        ring_chunk = usp_chunk_size * sp_ulysses_size
        _, ring_rank = get_usp_rank_coords(
            sp_rank=self._sp_rank,
            sp_ulysses_size=sp_ulysses_size,
            sp_ring_size=self._sp_ring_size,
        )
        ring_start = ring_rank * ring_chunk
        position_ids = torch.arange(
            ring_start,
            ring_start + ring_chunk,
            dtype=torch.long,
        ).unsqueeze(0)

        return {
            "loss_mask": loss_mask.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "position_ids": position_ids.to(self.device),
        }

    def _should_skip_usp_sharded_sample(self, sample: TrainSample) -> bool:
        """Return the SP-consistent skip decision for a pre-sharded USP sample."""
        full_loss_mask = self._usp_loss_mask(sample, self._usp_global_len(sample))
        min_tokens = max(1, self._min_loss_tokens)
        return int(full_loss_mask.sum().item()) < min_tokens

    def _usp_get_sharded_item(
        self, neutralized_count: int
    ) -> tuple[Dict[str, torch.Tensor] | None, int]:
        neutralized = 0
        while True:
            try:
                item = self.ray_queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.warning(
                    f"_usp_get_sharded_item: Exception waiting for data: {e}, timeout={self.timeout}"
                )
                return None, neutralized
            if item is None:
                return None, neutralized

            metadata = item.metadata or {}
            if not metadata.get("usp_sharded", False):
                raise RuntimeError(
                    f"USP sharded data fetcher received a non-sharded Mooncake sample. mooncake_key={item.mooncake_key}"
                )

            shapes = self._local_usp_shapes(item)
            dtypes_raw = item.tensor_dtypes or {}
            dtypes = {}
            for key, dtype_val in dtypes_raw.items():
                if isinstance(dtype_val, str):
                    dtypes[key] = getattr(torch, dtype_val.replace("torch.", ""))
                else:
                    dtypes[key] = dtype_val

            should_skip = self._should_skip_usp_sharded_sample(item)
            shard_key = f"{item.mooncake_key}_usp{self._sp_rank}"
            tensors = self.mooncake_store.get(
                key=shard_key,
                shapes=shapes,
                dtypes=dtypes,
                device=self.device,
            ).to_tensor_dict()
            tensors.update(self._local_usp_loss_and_position(item, shapes["input_ids"][-1]))

            self.mooncake_store.remove_eagle3_tensors(
                shard_key,
                has_last_hidden_states="last_hidden_states" in shapes,
                has_target="target" in shapes,
            )

            if should_skip:
                # Neutralize, don't drop: a per-DP-group drop desyncs FSDP.
                neutralized += 1
                tensors["loss_mask"] = torch.zeros_like(tensors["loss_mask"])
                logger.warning(
                    f"Neutralized USP sharded sample (loss mask zeroed): "
                    f"mooncake_key={item.mooncake_key}, sp_rank={self._sp_rank}, "
                    f"total_neutralized={neutralized_count + neutralized}"
                )

            return tensors, neutralized


def create_mooncake_dataloader(
    ray_queue: RayQueue,
    mooncake_store,
    collator: Callable[[List[Dict]], Dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int = 1,
    prefetch_factor: int = 2,
    timeout: Optional[float] = None,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    dynamic_loss_mask: bool = False,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
    min_loss_tokens: int = 0,
    usp_enabled: bool = False,
    ttt_length: int = 1,
    max_seq_length: Optional[int] = None,
    usp_local_shard: bool = False,
    packing_enabled: bool = False,
) -> DataLoader:
    """Create a DataLoader that fetches from mooncake via queue.

    Data flow:
      Controller (dispatches dispatch_batch_size samples) ->
      Ray Queue (per_dp_rank_batch_size samples per rank) ->
      DataLoader (batches per_dp_rank_batch_size samples together with padding) ->
      Training loop (one iteration per step)

    The collator pads sequences within the batch to the same length.

    In packing mode (``packing_enabled=True``), the dataset itself greedily bins
    samples up to ``max_seq_length`` and yields one bin (a ``List[Dict]``) per
    iteration; the DataLoader is set to ``batch_size=None`` so each bin is handed
    to the collator verbatim (the collator packs it into one fixed-length row).

    Args:
        ray_queue: Ray Queue to receive TrainSample from controller.
        mooncake_store: Mooncake store client for loading tensors.
        collator: Collator for padding and batching samples.
        device: Target device for tensors.
        batch_size: Number of samples per batch (= per_dp_rank_batch_size).
        prefetch_factor: Unused, kept for API compatibility.
        timeout: Timeout in seconds for waiting on queue. None means wait forever.
        assistant_header_ids: Token IDs for assistant header (for loss mask skip check).
        end_token_ids: Token IDs for end of turn (for loss mask skip check).
        dynamic_loss_mask: Whether loss mask is computed dynamically from input_ids.
        last_turn_loss_only: Global fallback for last-turn-only loss masking.
        packing_enabled: Enable streaming bin-packing (DFlash A′).

    Returns:
        (DataLoader, MooncakeDataset, collator). The dataset + collator are
        returned so packing callers can drive per-step key-budget bin iteration
        directly (the plain DataLoader __iter__ cannot take a per-step budget).
    """
    dataset = MooncakeDataset(
        ray_queue,
        mooncake_store,
        device,
        prefetch_factor,
        timeout,
        assistant_header_ids=assistant_header_ids,
        end_token_ids=end_token_ids,
        dynamic_loss_mask=dynamic_loss_mask,
        last_turn_loss_only=last_turn_loss_only,
        skip_after_header=skip_after_header,
        batch_size=batch_size,
        min_loss_tokens=min_loss_tokens,
        usp_enabled=usp_enabled,
        ttt_length=ttt_length,
        max_seq_length=max_seq_length,
        usp_local_shard=usp_local_shard,
        packing_enabled=packing_enabled,
    )

    # Packing: the dataset yields whole bins (List[Dict]); batch_size=None makes
    # the DataLoader pass each bin straight to collate_fn without re-batching.
    dl_batch_size = None if packing_enabled else batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=dl_batch_size,
        collate_fn=collator,
        num_workers=0,
    )
    # Return the dataset + collator too so callers can drive per-step packing
    # (key-budget bin iteration) without going through the plain DataLoader
    # __iter__ (which cannot receive a per-step budget).
    return dataloader, dataset, collator


class MooncakeDataFetcher:
    """Queue-based data fetcher for mooncake with DataLoader backend.

    Provides iteration over training samples that are pushed to a Ray queue
    by the AsyncTrainingController and loaded from mooncake.

    Batch size design:
      - micro_batch_size: Samples per GPU per training step (user config)
      - per_dp_rank_batch_size = micro_batch_size * sp_size (derived)
      - dispatch_batch_size = per_dp_rank_batch_size * dp_size (derived)
      - DataLoader batch_size = per_dp_rank_batch_size (all samples batched together)
      - Training loop does ONE iteration per step

    The collator pads sequences within the batch to the max length.
    """

    def __init__(
        self,
        queue: RayQueue,
        mooncake_store,
        collator: Callable[[List[Dict]], Dict[str, torch.Tensor]],
        device: torch.device,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        min_loss_tokens: int = 0,
        usp_enabled: bool = False,
        ttt_length: int = 1,
        max_seq_length: Optional[int] = None,
        usp_local_shard: bool = False,
        packing_enabled: bool = False,
    ):
        self.batch_size = batch_size
        self.packing_enabled = packing_enabled
        self._dataloader, self._dataset, self._collator = create_mooncake_dataloader(
            ray_queue=queue,
            mooncake_store=mooncake_store,
            collator=collator,
            device=device,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            timeout=timeout,
            assistant_header_ids=assistant_header_ids,
            end_token_ids=end_token_ids,
            dynamic_loss_mask=dynamic_loss_mask,
            last_turn_loss_only=last_turn_loss_only,
            skip_after_header=skip_after_header,
            min_loss_tokens=min_loss_tokens,
            usp_enabled=usp_enabled,
            ttt_length=ttt_length,
            max_seq_length=max_seq_length,
            usp_local_shard=usp_local_shard,
            packing_enabled=packing_enabled,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self._dataloader)

    def iter_step_batches(self, num_rows: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Packing only: yield exactly ``num_rows`` collated packed rows for ONE
        optimizer step.

        Rows are grouped by the controller-stamped row_id (single source of truth;
        the trainer never re-packs). The controller emits one row per rank per
        dispatch and the loop dispatches ``num_rows`` (== accumulation_steps) times
        per step, so every rank consumes the SAME ``num_rows`` rows — DP stays in
        lock-step under standard FSDP resharding. Each yielded item is one collated
        fixed-length packed row (a micro-batch).
        """
        if not self.packing_enabled:
            raise RuntimeError("iter_step_batches is only valid in packing mode")
        count = 0
        for packed_bin in self._row_iter():
            yield self._collator(packed_bin)
            count += 1
            if count >= num_rows:
                return

    def _row_iter(self) -> Iterator[List[Dict[str, torch.Tensor]]]:
        """Persistent row iterator (created lazily, reused across steps) so rows
        never straddle a step boundary due to look-ahead flushing."""
        if getattr(self, "_row_iter_cached", None) is None:
            self._row_iter_cached = self._dataset._iter_rows_by_id()
        return self._row_iter_cached


class PrefetchedDataFetcher:
    """Wraps MooncakeDataFetcher with async pre-fetching.

    A background thread continuously fetches batches from the underlying
    MooncakeDataFetcher (which blocks on Mooncake TCP), staging them in a
    thread-safe queue.  The training loop reads from this queue, overlapping
    data transfer with GPU compute.

    Without prefetch: [data] → [compute] → [data] → [compute]  (sequential)
    With prefetch:    [compute] → [compute] → [compute]         (overlapped)
                      [data]      [data]      [data]

    The background thread starts lazily on the first ``__iter__`` call and
    keeps running across multiple ``itertools.islice`` invocations (one per
    training step).  The training loop simply reads from the shared queue.

    Packing mode: ``iter_step_batches(num_rows)`` is the consumption entry point
    (matching MooncakeDataFetcher's). The background thread assembles ONE
    optimizer step's rows per queue item, so ``prefetch_depth`` counts *steps*,
    not batches — depth=D keeps at most ``D * rows_per_step`` rows resident on
    CPU. Rows never straddle a step boundary because inner.iter_step_batches
    consumes exactly ``rows_per_step`` rows from the persistent row iterator.
    """

    _SENTINEL = object()
    # Poll interval for stop-aware blocking puts, so a stalled consumer (e.g.
    # training exits at NUM_STEPS while the queue is full) can't wedge the
    # background thread forever on put().
    _PUT_POLL_SEC = 0.5

    def __init__(
        self,
        inner: MooncakeDataFetcher,
        prefetch_depth: int = 2,
        target_device: Optional[torch.device] = None,
        packing_enabled: bool = False,
        rows_per_step: int = 1,
    ):
        self.inner = inner
        self.prefetch_depth = prefetch_depth
        self.target_device = target_device
        self.packing_enabled = packing_enabled
        self.rows_per_step = rows_per_step
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._error: Optional[BaseException] = None
        self._stop = False

    def _put_with_stop(self, item) -> bool:
        """Blocking put that yields to a stop request. Returns False if the
        fetcher was closed before the item could be enqueued."""
        while not self._stop:
            try:
                self._queue.put(item, timeout=self._PUT_POLL_SEC)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_loop(self) -> None:
        try:
            if self.packing_enabled:
                # Each queue item is ONE optimizer step's worth of packed rows.
                # inner.iter_step_batches consumes exactly rows_per_step rows from
                # the persistent row iterator, so blocks never straddle a step
                # boundary (no cross-step look-ahead → no dispatch/consume deadlock).
                while not self._stop:
                    rows = list(self.inner.iter_step_batches(self.rows_per_step))
                    if not self._put_with_stop(rows):
                        return
                    if not rows:
                        # inner is drained (sentinel/timeout with nothing buffered);
                        # stop so we don't spin emitting empty steps forever.
                        break
            else:
                for batch in self.inner:
                    if not self._put_with_stop(batch):
                        return
        except Exception as e:
            # Preserve the original traceback so re-raise in __next__
            # points to the actual failure site, not to __next__ itself.
            import sys

            self._error = e.with_traceback(sys.exc_info()[2])
        finally:
            self._put_with_stop(self._SENTINEL)

    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Signal the background thread to stop and drain the queue so it can't
        stay blocked on a full-queue put during teardown."""
        self._stop = True
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._ensure_started()
        return self

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch of tensors to the target device (GPU)."""
        if self.target_device is None:
            return batch
        return {
            k: v.to(self.target_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._error is not None:
            raise self._error
        item = self._queue.get()
        if item is self._SENTINEL:
            if self._error is not None:
                raise self._error
            raise StopIteration
        return self._to_device(item)

    def iter_step_batches(self, num_rows: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Packing only: yield one step's prefetched rows, mirroring
        MooncakeDataFetcher.iter_step_batches so the trainer's consumption path is
        identical whether or not prefetch is enabled.

        The background thread has already assembled this step's rows as a single
        queue item; here we just pop it and move each row to the GPU.
        """
        assert self.packing_enabled, "iter_step_batches requires packing_enabled"
        assert num_rows == self.rows_per_step, (
            f"iter_step_batches({num_rows}) != rows_per_step={self.rows_per_step}; "
            "prefetch blocks per-step rows and cannot serve a different budget"
        )
        self._ensure_started()
        if self._error is not None:
            raise self._error
        item = self._queue.get()
        if item is self._SENTINEL:
            if self._error is not None:
                raise self._error
            return
        assert len(item) <= num_rows, (
            f"prefetched step has {len(item)} rows > budget {num_rows}; "
            "inner iter_step_batches must not overshoot rows_per_step"
        )
        for row in item:
            yield self._to_device(row)
