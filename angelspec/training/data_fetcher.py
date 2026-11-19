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

import collections
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ray.util.queue import Queue as RayQueue
from torch.utils.data import DataLoader, IterableDataset

from angelspec.data.positions import prepare_sample_positions
from angelspec.data.utils import (
    MOONCAKE_DELETE_KEY_FIELD,
    deserialize_packed_loss_mask,
    resolve_loss_mask,
    unpack_loss_mask,
)
from angelspec.transfer.mooncake.draft_store import (
    HIDDEN_STATES_STORAGE_DTYPE,
    MooncakeSampleUnavailable,
)
from angelspec.utils.distributed import (
    get_draft_sp_gloo_group,
    get_draft_sp_group,
    get_sp_ring_group,
    get_usp_rank_coords,
)
from angelspec.utils.logging import logger
from angelspec.utils.usp import usp_chunk_size

PLACEHOLDER_SEQ_LEN = 256


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
    # The online dump writer frees this sample's keys after the training step; trainers must not.
    retain_for_dump: bool = False


_STOPPED = object()


class _LeaderReadAhead:
    """Reads the SP leader's next queue items and their Mooncake tensors ahead of use.

    Under ``sp_leader_mooncake_fetch`` only SP rank 0 reads the queue and Mooncake,
    and its broadcasts to the SP peers must stay on the main thread in a fixed
    order, so the generic background prefetch is off. This thread does only the
    leader's reads: it takes the next items off the queue and copies their tensors
    into CPU memory it owns, overlapping the transfers with training, while the
    main thread still broadcasts each item and its tensors in order. It reads ahead
    until the staged items number ``max_items`` or hold ``max_tokens`` tokens
    (0: no limit on that count).
    """

    _POLL_SEC = 1.0

    def __init__(self, dataset: "MooncakeDataset", max_items: int = 0, max_tokens: int = 0):
        self._dataset = dataset
        self._max_items = max_items
        self._max_tokens = max_tokens
        self._entries: collections.deque = collections.deque()
        self._staged_tokens = 0
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="sp-leader-read-ahead", daemon=True)
        self._thread.start()

    def _full(self) -> bool:
        if not self._entries:
            return False
        return bool(
            (self._max_items and len(self._entries) >= self._max_items)
            or (self._max_tokens and self._staged_tokens >= self._max_tokens)
        )

    def _push(self, entry, tokens: int = 0) -> bool:
        with self._cond:
            while self._full() and not self._stop:
                self._cond.wait(self._POLL_SEC)
            if self._stop:
                return False
            self._entries.append((entry, tokens))
            self._staged_tokens += tokens
            self._cond.notify_all()
            return True

    def _next_queue_item(self):
        # Short polls keep close() effective while no item arrives.
        timeout = self._dataset.timeout
        waited = 0.0
        while not self._stop:
            poll = self._POLL_SEC if timeout is None else min(self._POLL_SEC, timeout - waited)
            try:
                return self._dataset.ray_queue.get(block=True, timeout=max(poll, 0.0))
            except queue.Empty:
                waited += poll
                if timeout is not None and waited >= timeout:
                    raise
        return _STOPPED

    def _run(self) -> None:
        try:
            while not self._stop:
                item = self._next_queue_item()
                if item is _STOPPED:
                    return
                if item is None:
                    self._push(("end", None, None))
                    return
                staged = self._dataset._stage_leader_tensors(item)
                tokens = int((item.tensor_shapes or {}).get("input_ids", (0,))[-1])
                if not self._push(("item", item, staged), tokens):
                    return
        except Exception as exc:
            self._push(("error", None, f"{type(exc).__name__}: {exc}"))

    def next(self):
        """Pop the next ``(kind, item, value)`` entry, blocking until there is one."""
        with self._cond:
            while not self._entries:
                if not self._thread.is_alive():
                    raise RuntimeError("SP leader read-ahead ended without another entry")
                self._cond.wait(self._POLL_SEC)
            entry, tokens = self._entries.popleft()
            self._staged_tokens -= tokens
            self._cond.notify_all()
            return entry

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._entries.clear()
            self._staged_tokens = 0
            self._cond.notify_all()


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
        sp_leader_mooncake_fetch: bool = False,
        sequence_shard_after_leader_fetch: bool = False,
        packing_enabled: bool = False,
        copy_cpu_tensors: bool = False,
        position_recipe=None,
        leader_read_ahead_items: int = 0,
        leader_read_ahead_tokens: int = 0,
    ):
        self.ray_queue = ray_queue
        self.mooncake_store = mooncake_store
        self.device = torch.device(device)
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
        self.sp_leader_mooncake_fetch = sp_leader_mooncake_fetch
        self.sequence_shard_after_leader_fetch = sequence_shard_after_leader_fetch
        if self.sp_leader_mooncake_fetch and not self.usp_local_shard:
            raise ValueError("sp_leader_mooncake_fetch requires usp_local_shard=true")
        if self.sequence_shard_after_leader_fetch and not self.sp_leader_mooncake_fetch:
            raise ValueError(
                "sequence_shard_after_leader_fetch requires sp_leader_mooncake_fetch=true"
            )
        if self.sequence_shard_after_leader_fetch and batch_size != 1:
            raise ValueError(
                "DFlash SP leader sequence scatter currently requires micro_batch_size=1"
            )
        self._read_ahead_limits: Optional[Tuple[int, int]] = None
        if leader_read_ahead_items > 0 or leader_read_ahead_tokens > 0:
            if not self.sp_leader_mooncake_fetch:
                raise ValueError("leader read-ahead requires sp_leader_mooncake_fetch=true")
            self._read_ahead_limits = (
                max(leader_read_ahead_items, 0),
                max(leader_read_ahead_tokens, 0),
            )
        self._read_ahead: Optional[_LeaderReadAhead] = None
        self._staged: Optional[Tuple[TrainSample, Tuple[Any, Optional[Exception]]]] = None
        self.copy_cpu_tensors = copy_cpu_tensors
        self.position_recipe = position_recipe
        if position_recipe is not None and usp_enabled:
            raise ValueError(
                "MRoPE preparation requires full sample tokens; use usp_local_shard=true"
            )
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
        self._last_tensor_spec: tuple[Dict[str, Tuple[int, ...]], Dict[str, torch.dtype]] = (
            {},
            {},
        )
        self._loss_mask_dtype = torch.long
        self._placeholder_count = 0
        self._leader_scatter_logged = False
        self._init_sp_context()

    def _init_sp_context(self) -> None:
        self._sp_group = None
        self._sp_gloo_group = None
        self._sp_world_size = 1
        self._sp_rank = 0
        self._sp_ring_size = 1
        self._sp_ring_rank = 0
        if not (self.usp_enabled or self.sp_leader_mooncake_fetch):
            return

        sp_group = get_draft_sp_group()
        if sp_group is None:
            return

        self._sp_group = sp_group
        self._sp_gloo_group = get_draft_sp_gloo_group()
        self._sp_world_size = dist.get_world_size(sp_group)
        self._sp_rank = dist.get_rank(sp_group)

        ring_group = get_sp_ring_group()
        if ring_group is not None:
            self._sp_ring_size = dist.get_world_size(ring_group)
            self._sp_ring_rank = dist.get_rank(ring_group)

    def _sp_root_global_rank(self) -> int:
        if self._sp_group is None:
            raise RuntimeError("SP process group is not initialized")
        return dist.get_global_rank(self._sp_group, 0)

    @staticmethod
    def _resolve_dtypes(dtypes_raw: Optional[Dict[str, Any]]) -> Dict[str, torch.dtype]:
        dtypes: Dict[str, torch.dtype] = {}
        for key, dtype_val in (dtypes_raw or {}).items():
            if isinstance(dtype_val, str):
                dtypes[key] = getattr(torch, dtype_val.replace("torch.", ""))
            else:
                dtypes[key] = dtype_val
        return dtypes

    def _make_placeholder(self, sample: TrainSample) -> Dict[str, torch.Tensor]:
        self._placeholder_count += 1
        shapes = dict(sample.tensor_shapes or {})
        dtypes = self._resolve_dtypes(sample.tensor_dtypes)
        if not any(name in shapes for name in ("hidden_states", "last_hidden_states", "target")):
            shapes, dtypes = self._last_tensor_spec
        if not shapes:
            raise RuntimeError(
                f"Cannot build a placeholder for {sample.mooncake_key}: no tensor metadata"
            )

        seq_len = PLACEHOLDER_SEQ_LEN
        if "input_ids" in shapes:
            seq_len = min(seq_len, int(shapes["input_ids"][-1]))
        data: Dict[str, torch.Tensor] = {
            "_mooncake_unavailable": 1,
            "input_ids": torch.zeros(
                (1, seq_len),
                dtype=dtypes.get("input_ids", torch.long),
                device=self.device,
            ),
            "loss_mask": torch.zeros(
                (1, seq_len),
                dtype=self._loss_mask_dtype,
                device=self.device,
            ),
        }
        for name in ("hidden_states", "last_hidden_states", "target"):
            if name in shapes:
                data[name] = torch.zeros(
                    (1, seq_len, shapes[name][-1]),
                    dtype=dtypes.get(name, HIDDEN_STATES_STORAGE_DTYPE),
                    device=self.device,
                )
        return prepare_sample_positions(data, getattr(self, "position_recipe", None))

    def _sp_broadcast_context(self) -> tuple[dist.ProcessGroup, torch.device, int]:
        device = (
            torch.device("cuda", self.device)
            if isinstance(self.device, int)
            else torch.device(self.device)
        )
        group = self._sp_gloo_group if device.type == "cpu" else self._sp_group
        if group is None:
            raise RuntimeError(f"SP broadcast group is not initialized for device {device}")
        return group, device, dist.get_global_rank(group, 0)

    def _broadcast_object_from_sp_leader(self, value):
        payload = [value if self._sp_rank == 0 else None]
        group, device, src = self._sp_broadcast_context()
        dist.broadcast_object_list(
            payload,
            src=src,
            group=group,
            device=device,
        )
        return payload[0]

    def _get_next_item(self):
        read_ahead = self._read_ahead_limits is not None
        if not read_ahead and (not self.sp_leader_mooncake_fetch or self._sp_world_size == 1):
            return self.ray_queue.get(block=True, timeout=self.timeout)

        payload = None
        if self._sp_rank == 0:
            if read_ahead:
                payload = self._next_read_ahead_item()
            else:
                try:
                    payload = (True, self.ray_queue.get(block=True, timeout=self.timeout), None)
                except Exception as exc:
                    payload = (False, None, f"{type(exc).__name__}: {exc}")
        if self._sp_world_size > 1:
            payload = self._broadcast_object_from_sp_leader(payload)
        ok, item, error = payload
        if not ok:
            raise RuntimeError(f"SP leader failed to read training queue: {error}")
        return item

    def _next_read_ahead_item(self):
        if self._read_ahead is None:
            self._read_ahead = _LeaderReadAhead(self, *self._read_ahead_limits)
        kind, item, value = self._read_ahead.next()
        if kind == "error":
            return (False, None, value)
        self._staged = (item, value) if kind == "item" else None
        return (True, item, None)

    def _stage_leader_tensors(self, sample: TrainSample):
        """Read ``sample`` from Mooncake into CPU tensors this rank owns."""
        try:
            if self.mooncake_store is None:
                raise RuntimeError("SP leader has no Mooncake store")
            tensors = self.mooncake_store.get(
                key=sample.mooncake_key,
                shapes=sample.tensor_shapes,
                dtypes=self._resolve_dtypes(sample.tensor_dtypes),
                device=torch.device("cpu"),
            ).to_tensor_dict()
        except Exception as exc:
            return None, exc
        # Host-buffer reads alias Mooncake's receive buffers, which later reads reuse.
        return {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in tensors.items()
        }, None

    def _leader_tensors(self, sample: TrainSample, dtypes: Dict[str, torch.dtype]):
        """The leader's Mooncake read of ``sample``: staged by the read-ahead, or done now."""
        staged, self._staged = getattr(self, "_staged", None), None
        if staged is not None and staged[0] is sample:
            tensors, error = staged[1]
            if error is not None:
                raise error
            return tensors
        if self.mooncake_store is None:
            raise RuntimeError("SP leader has no Mooncake store")
        return self.mooncake_store.get(
            key=sample.mooncake_key,
            shapes=sample.tensor_shapes,
            dtypes=dtypes,
            device=self.device,
        ).to_tensor_dict()

    def close(self) -> None:
        if self._read_ahead is not None:
            self._read_ahead.close()
            self._read_ahead = None
        self._staged = None

    def _load_from_mooncake_sp_leader(
        self,
        sample: TrainSample,
        dtypes: Dict[str, torch.dtype],
    ) -> Dict[str, torch.Tensor]:
        if self._sp_world_size == 1:
            return {
                name: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for name, value in self._leader_tensors(sample, dtypes).items()
            }

        tensor_names = [
            name
            for name in ("input_ids", "hidden_states", "target", "last_hidden_states")
            if name in sample.tensor_shapes
        ]
        group, device, src = self._sp_broadcast_context()
        tensor_dtypes = {
            name: dtypes.get(name, torch.int64 if name == "input_ids" else torch.bfloat16)
            for name in tensor_names
        }
        result = None
        status = None
        if self._sp_rank == 0:
            try:
                result = self._leader_tensors(sample, dtypes)
                for name in tensor_names:
                    tensor = result[name]
                    if tuple(tensor.shape) != tuple(sample.tensor_shapes[name]):
                        raise ValueError(
                            f"{name}: expected shape {sample.tensor_shapes[name]}, "
                            f"got {tuple(tensor.shape)}"
                        )
                    result[name] = tensor.to(device=device, dtype=tensor_dtypes[name]).contiguous()
                status = (True, False, None)
            except Exception as exc:
                status = (
                    False,
                    isinstance(exc, MooncakeSampleUnavailable),
                    f"{type(exc).__name__}: {exc}",
                )

        ok, unavailable, error = self._broadcast_object_from_sp_leader(status)
        if not ok:
            if unavailable:
                raise MooncakeSampleUnavailable(error)
            raise RuntimeError(
                f"SP leader failed to load Mooncake key {sample.mooncake_key}: {error}"
            )

        tensor_map: Dict[str, torch.Tensor] = {}
        for name in tensor_names:
            shape = tuple(sample.tensor_shapes[name])
            if self._sp_rank == 0:
                tensor = result[name]
            else:
                tensor = torch.empty(shape, dtype=tensor_dtypes[name], device=device)
            dist.broadcast(
                tensor,
                src=src,
                group=group,
            )
            tensor_map[name] = tensor
        return tensor_map

    def _load_from_mooncake(self, sample: TrainSample) -> Dict[str, Any]:
        """Load tensors from mooncake key into device memory."""
        if self.sp_leader_mooncake_fetch:
            if self.sequence_shard_after_leader_fetch and self._sp_world_size > 1:
                return self._load_from_mooncake_leader_scatter(sample)

        dtypes = self._resolve_dtypes(sample.tensor_dtypes)

        logger.debug(
            f"_load_from_mooncake: key={sample.mooncake_key}, requesting shapes={sample.tensor_shapes}"
        )

        if self.sp_leader_mooncake_fetch:
            tensor_dict = self._load_from_mooncake_sp_leader(sample, dtypes)
        else:
            tensors = self.mooncake_store.get(
                key=sample.mooncake_key,
                shapes=sample.tensor_shapes,
                dtypes=dtypes,
                device=self.device,
            )
            tensor_dict = tensors.to_tensor_dict()
        self._last_tensor_spec = (dict(sample.tensor_shapes or {}), dtypes)
        staged_on_cpu = any(
            isinstance(value, torch.Tensor) and value.device.type == "cpu"
            for value in tensor_dict.values()
        )
        if (
            self._batch_size > 1
            or self.packing_enabled
            or (staged_on_cpu and self.copy_cpu_tensors)
        ):
            # Clone to prevent use-after-free: the collator/packing bin holds sample N
            # while later samples are fetched, but cleanup frees the Mooncake buffer
            # (Issue 31). Note: clone() unpins, breaking non_blocking H2D — only here.
            result = {}
            for key, value in tensor_dict.items():
                if not isinstance(value, torch.Tensor):
                    result[key] = value
                elif value.device.type == "cpu" and value.is_pinned():
                    owned = torch.empty_like(value, pin_memory=True)
                    owned.copy_(value)
                    result[key] = owned
                else:
                    result[key] = value.clone()
        else:
            # batch_size=1: safe to use pinned views — consumed immediately.
            # Preserves pinned memory for async H2D via non_blocking=True.
            result = dict(tensor_dict)

        self._cleanup_mooncake_data(sample, result)
        if sample.packed_loss_mask is not None:
            result["packed_loss_mask"] = sample.packed_loss_mask
        if sample.last_turn_loss_only is not None:
            result["last_turn_loss_only"] = sample.last_turn_loss_only
        return result

    @staticmethod
    def _slice_sequence_tensor(
        tensor: torch.Tensor,
        *,
        seq_dim: int,
        start: int,
        shard_len: int,
    ) -> torch.Tensor:
        end = min(start + shard_len, tensor.shape[seq_dim])
        index = [slice(None)] * tensor.dim()
        index[seq_dim] = slice(start, end)
        local = tensor[tuple(index)]
        missing = shard_len - local.shape[seq_dim]
        if missing > 0:
            pad = [0, 0] * tensor.dim()
            pad_index = 2 * (tensor.dim() - seq_dim - 1) + 1
            pad[pad_index] = missing
            local = F.pad(local, pad)
        return local.contiguous()

    def _scatter_cpu_tensor(
        self,
        tensor: Optional[torch.Tensor],
        *,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        global_len: int,
        shard_len: int,
    ) -> torch.Tensor:
        if self._sp_gloo_group is None:
            raise RuntimeError("USP leader scatter requires a draft SP Gloo group")

        src = dist.get_global_rank(self._sp_gloo_group, 0)
        seq_dim = 0 if len(shape) == 1 else len(shape) - 2
        if shape[seq_dim] != global_len:
            raise ValueError(
                f"Cannot find sequence dimension {global_len} in tensor shape {shape}"
            )
        local_shape = list(shape)
        local_shape[seq_dim] = shard_len
        if self._sp_rank == 0:
            if tensor is None:
                raise RuntimeError("SP leader is missing a tensor required for scatter")
            shards = [
                self._slice_sequence_tensor(
                    tensor,
                    seq_dim=seq_dim,
                    start=rank * shard_len,
                    shard_len=shard_len,
                )
                for rank in range(self._sp_world_size)
            ]
            local = torch.empty_like(shards[0])
        else:
            local = torch.empty(local_shape, dtype=dtype, device="cpu")
            shards = None

        dist.scatter(
            local,
            scatter_list=shards,
            src=src,
            group=self._sp_gloo_group,
        )
        return local

    def _load_from_mooncake_leader_scatter(self, sample: TrainSample) -> Dict[str, Any]:
        if self.device.type != "cpu":
            raise ValueError("USP leader scatter requires CPU Mooncake staging")
        if self._sp_gloo_group is None:
            raise RuntimeError("USP leader scatter requires a draft SP Gloo group")

        dtypes = self._resolve_dtypes(sample.tensor_dtypes)
        global_len = sample.tensor_shapes["input_ids"][-1]
        padded_global_len = ((global_len + 255) // 256) * 256
        padded_global_len = (
            (padded_global_len + self._sp_world_size - 1) // self._sp_world_size
        ) * self._sp_world_size
        shard_len = padded_global_len // self._sp_world_size
        tensor_names = [
            name
            for name in ("input_ids", "hidden_states", "target", "last_hidden_states")
            if name in sample.tensor_shapes
        ]
        status = None
        tensor_dict = None
        if self._sp_rank == 0:
            try:
                tensor_dict = self._leader_tensors(sample, dtypes)
                for name in tensor_names:
                    tensor = tensor_dict[name]
                    shape = tuple(sample.tensor_shapes[name])
                    if tuple(tensor.shape) != shape:
                        raise ValueError(
                            f"{name}: expected shape {shape}, got {tuple(tensor.shape)}"
                        )
                    if name != "input_ids" and shape[-2] != global_len:
                        raise ValueError(f"{name}: sequence dimension does not match {global_len}")
                    dtype = dtypes.get(
                        name, torch.int64 if name == "input_ids" else torch.bfloat16
                    )
                    tensor_dict[name] = tensor.to(device=self.device, dtype=dtype).contiguous()
                status = (True, False, None)
            except Exception as exc:
                status = (
                    False,
                    isinstance(exc, MooncakeSampleUnavailable),
                    f"{type(exc).__name__}: {exc}",
                )

        src = dist.get_global_rank(self._sp_gloo_group, 0)
        payload = [status if self._sp_rank == 0 else None]
        dist.broadcast_object_list(payload, src=src, group=self._sp_gloo_group)
        ok, unavailable, error = payload[0]
        if not ok:
            if unavailable:
                raise MooncakeSampleUnavailable(error)
            raise RuntimeError(
                f"SP leader failed to load Mooncake key {sample.mooncake_key}: {error}"
            )
        if self._sp_rank == 0 and not self._leader_scatter_logged:
            self._leader_scatter_logged = True
            total_bytes = sum(
                torch.empty((), dtype=dtypes.get(name, torch.long)).element_size()
                * int(torch.tensor(shape).prod().item())
                for name, shape in sample.tensor_shapes.items()
            )
            logger.info(
                "USP Mooncake leader scatter active: one full get per SP group, "
                "sp_size=%d, full=%.1fMiB, sequence shard≈%.1fMiB/rank",
                self._sp_world_size,
                total_bytes / (1024**2),
                total_bytes / self._sp_world_size / (1024**2),
            )

        result = {}
        for name in tensor_names:
            leader_tensor = tensor_dict[name] if self._sp_rank == 0 else None
            if name == "input_ids":
                if self._sp_rank == 0:
                    result[name] = leader_tensor.clone()
                else:
                    result[name] = torch.empty(
                        sample.tensor_shapes[name],
                        dtype=dtypes.get(name, torch.long),
                    )
                dist.broadcast(result[name], src=src, group=self._sp_gloo_group)
            else:
                result[name] = self._scatter_cpu_tensor(
                    leader_tensor,
                    shape=sample.tensor_shapes[name],
                    dtype=dtypes.get(name, torch.bfloat16),
                    global_len=global_len,
                    shard_len=shard_len,
                )

        self._last_tensor_spec = (dict(sample.tensor_shapes or {}), dtypes)
        self._cleanup_mooncake_data(sample, result)
        if sample.packed_loss_mask is not None:
            result["packed_loss_mask"] = sample.packed_loss_mask
        if sample.last_turn_loss_only is not None:
            result["last_turn_loss_only"] = sample.last_turn_loss_only
        return result

    def _cleanup_mooncake_data(self, sample: TrainSample, result: Dict[str, Any]) -> None:
        """Free the sample's Mooncake entry, or defer it under usp_local_shard.

        Non-USP / data-path-sharded: delete immediately after this rank's read.
        usp_local_shard: the same key is fanned to every SP rank, so a per-rank
        delete-on-fetch would starve peers. Instead stamp the base key + tensor
        flags onto ``result``; the trainer deletes once (sp_rank 0) after the
        optimizer-step barrier, when all SP peers have provably fetched it.
        Samples retained for the online dump are freed by the dump writer instead.
        """
        if sample.retain_for_dump:
            return
        shapes = sample.tensor_shapes or {}
        has_lhs = "last_hidden_states" in shapes
        has_target = "target" in shapes
        has_hidden = "hidden_states" in shapes
        if self.usp_local_shard:
            result[MOONCAKE_DELETE_KEY_FIELD] = (
                sample.mooncake_key,
                has_lhs,
                has_target,
                has_hidden,
            )
            return

        self.mooncake_store.remove_tensors(
            sample.mooncake_key,
            has_last_hidden_states=has_lhs,
            has_target=has_target,
            has_hidden_states=has_hidden,
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
        if isinstance(mask, torch.Tensor):
            self._loss_mask_dtype = mask.dtype
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
        try:
            data = self._load_from_mooncake(item)
        except MooncakeSampleUnavailable as exc:
            data = self._make_placeholder(item)
            if item.metadata and "source" in item.metadata:
                data["data_source"] = str(item.metadata["source"])
            neutralized_count += 1
            logger.warning(
                "Replacing unavailable Mooncake sample with a zero-loss placeholder "
                "(mooncake_key=%s, total_replaced=%d): %s",
                item.mooncake_key,
                self._placeholder_count,
                exc,
            )
            return data, neutralized_count

        if item.metadata and "source" in item.metadata:
            data["data_source"] = str(item.metadata["source"])
        if "opd_media" in (item.metadata or {}):
            data["opd_media"] = item.metadata["opd_media"]
        neutralized_count = self._resolve_and_neutralize_loss_mask(
            data, item.mooncake_key, neutralized_count
        )
        if (item.metadata or {}).get("_eval_padding", False):
            data["loss_mask"] = torch.zeros_like(data["input_ids"])

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
        prepare_sample_positions(
            data, getattr(self, "position_recipe", None), multimodal_metadata=item.metadata
        )
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
            item = self._get_next_item()

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
            item = self._get_next_item()
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
            raise RuntimeError("Packing queue ended before last_in_row; refusing a partial row")

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
        }
        if "hidden_states" in sample.tensor_shapes:
            shapes["hidden_states"] = (
                1,
                local_len,
                sample.tensor_shapes["hidden_states"][-1],
            )
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
            item = self.ray_queue.get(block=True, timeout=self.timeout)
            if item is None:
                return None, neutralized

            metadata = item.metadata or {}
            if not metadata.get("usp_sharded", False):
                raise RuntimeError(
                    f"USP sharded data fetcher received a non-sharded Mooncake sample. mooncake_key={item.mooncake_key}"
                )

            shapes = self._local_usp_shapes(item)
            dtypes = self._resolve_dtypes(item.tensor_dtypes)

            should_skip = self._should_skip_usp_sharded_sample(item)
            should_skip = should_skip or bool(metadata.get("_eval_padding", False))
            shard_key = f"{item.mooncake_key}_usp{self._sp_rank}"
            try:
                tensors = self.mooncake_store.get(
                    key=shard_key,
                    shapes=shapes,
                    dtypes=dtypes,
                    device=self.device,
                ).to_tensor_dict()
            except MooncakeSampleUnavailable as exc:
                self._placeholder_count += 1
                logger.warning(
                    "Zero-filling unavailable USP shard " "(shard_key=%s, total_replaced=%d): %s",
                    shard_key,
                    self._placeholder_count,
                    exc,
                )
                tensors = {
                    name: torch.zeros(
                        shape,
                        dtype=dtypes.get(
                            name,
                            torch.long if name == "input_ids" else HIDDEN_STATES_STORAGE_DTYPE,
                        ),
                        device=self.device,
                    )
                    for name, shape in shapes.items()
                }
                should_skip = True
            tensors.update(self._local_usp_loss_and_position(item, shapes["input_ids"][-1]))
            if item.metadata and "source" in item.metadata:
                tensors["data_source"] = str(item.metadata["source"])

            self.mooncake_store.remove_tensors(
                shard_key,
                has_last_hidden_states="last_hidden_states" in shapes,
                has_target="target" in shapes,
                has_hidden_states="hidden_states" in shapes,
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
    sp_leader_mooncake_fetch: bool = False,
    sequence_shard_after_leader_fetch: bool = False,
    packing_enabled: bool = False,
    copy_cpu_tensors: bool = False,
    position_recipe=None,
    leader_read_ahead_items: int = 0,
    leader_read_ahead_tokens: int = 0,
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
        leader_read_ahead_items: With ``sp_leader_mooncake_fetch``, how many samples the
            SP leader reads ahead from the queue and Mooncake in the background.
        leader_read_ahead_tokens: The same budget in sample tokens (packing).

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
        sp_leader_mooncake_fetch=sp_leader_mooncake_fetch,
        sequence_shard_after_leader_fetch=sequence_shard_after_leader_fetch,
        packing_enabled=packing_enabled,
        copy_cpu_tensors=copy_cpu_tensors,
        position_recipe=position_recipe,
        leader_read_ahead_items=leader_read_ahead_items,
        leader_read_ahead_tokens=leader_read_ahead_tokens,
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
        sp_leader_mooncake_fetch: bool = False,
        sequence_shard_after_leader_fetch: bool = False,
        packing_enabled: bool = False,
        copy_cpu_tensors: bool = False,
        position_recipe=None,
        leader_read_ahead_items: int = 0,
        leader_read_ahead_tokens: int = 0,
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
            sp_leader_mooncake_fetch=sp_leader_mooncake_fetch,
            sequence_shard_after_leader_fetch=sequence_shard_after_leader_fetch,
            packing_enabled=packing_enabled,
            copy_cpu_tensors=copy_cpu_tensors,
            position_recipe=position_recipe,
            leader_read_ahead_items=leader_read_ahead_items,
            leader_read_ahead_tokens=leader_read_ahead_tokens,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self._dataloader)

    def close(self) -> None:
        self._dataset.close()

    def iter_raw_samples(self) -> Iterator[Dict[str, Any]]:
        """Yield uncollated raw samples (single dicts) from the underlying dataset.

        Eval caching uses this so it stores individual, unpadded samples and can
        re-collate them later with an independent micro-batch size.
        """
        return iter(self._dataset)

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
