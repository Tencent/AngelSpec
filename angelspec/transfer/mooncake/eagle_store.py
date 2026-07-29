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

import ctypes
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from angelspec.transfer.mooncake.helpers import _format_bytes
from angelspec.transfer.mooncake.store import MooncakeHiddenStateStore
from angelspec.utils.distributed import get_usp_rank_coords
from angelspec.utils.logging import logger
from angelspec.utils.usp import split_usp_batch

if TYPE_CHECKING:
    from angelspec.models.target.eagle3_target_model import Eagle3TargetOutput

# Static lookup for dtype → element size in bytes (avoids creating a tensor
# on every call to _compute_tensor_size).
_DTYPE_ELEMENT_SIZES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}

# Canonical dtype for hidden-state tensors written to / read from Mooncake.
HIDDEN_STATES_STORAGE_DTYPE = torch.bfloat16


class EagleMooncakeStore(MooncakeHiddenStateStore):
    """
    Mooncake Store wrapper specialized for Eagle3 hidden states.

    Uses RDMA-registered host buffers and put_from for zero-copy transfers.
    Each Eagle3 output is stored as multiple tensors with key suffixes:
    - {key}_hs: hidden_states
    - {key}_tgt: target
    - {key}_ids: input_ids
    - {key}_lhs: last_hidden_states (if present)

    Deletions use ``batch_remove(force=True)`` for immediate cleanup.
    """

    TENSOR_SUFFIXES = ["_hs", "_tgt", "_ids", "_lhs"]

    def _put_raw_tensors(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        if self._gpu_direct_available and self._gpu_send_buffer is not None:
            buf = self._gpu_send_buffer
            buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
            self._do_sync_batch_put(keys, buffer_ptrs, sizes)
        elif self._host_buffer_pool is None or self._async_put_manager is None:
            raise RuntimeError(
                "put() requires either GPU Direct (enable_gpu_direct=True) or "
                "async host-buffer puts (async_put_pool_size > 0). "
                "Current config has async_put_pool_size=0 and GPU Direct is "
                f"{'enabled but gpu_send_buffer failed to initialize' if self._gpu_direct_available else 'disabled'}. "
                "Set async_put_pool_size >= 1 or enable GPU Direct."
            )
        else:
            buf = self._host_buffer_pool.get_buffer()
            self._async_put_manager.check_last_error()
            self._async_put_manager.wait_for_buffer(buf.ptr)

            compute_event = torch.cuda.Event()
            compute_event.record()

            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(compute_event)
                buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
                copy_done = torch.cuda.Event()
                copy_done.record()

            for t in tensors:
                if t.is_cuda:
                    t.record_stream(self._copy_stream)

            self._async_put_manager.submit(
                keys,
                buffer_ptrs,
                sizes,
                buf.ptr,
                wait_event=copy_done,
                device_index=self._copy_stream.device.index,
            )

    def put_usp_shards(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        *,
        sp_size: int,
        sp_ring_size: int,
        ttt_length: int,
        max_seq_length: Optional[int],
    ) -> Dict[str, Any]:
        self._ensure_initialized()
        logger.debug("put_usp_shards: starting for key=%s, sp_size=%s", key, sp_size)

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if (
            last_hidden_states is not None
            and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE
        ):
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        target_hs = last_hidden_states if last_hidden_states is not None else target
        if target_hs is None:
            raise ValueError("USP sharded Mooncake storage requires last_hidden_states or target")

        if input_ids.dim() == 2:
            loss_mask_len = input_ids.shape[-1]
        else:
            loss_mask_len = input_ids.shape[0]
        dummy_loss_mask = torch.ones((1, loss_mask_len), dtype=torch.long, device=input_ids.device)
        sp_ulysses_size = max(1, sp_size // sp_ring_size)

        for sp_rank in range(sp_size):
            _, ring_rank = get_usp_rank_coords(
                sp_rank=sp_rank,
                sp_ulysses_size=sp_ulysses_size,
                sp_ring_size=sp_ring_size,
            )
            (
                shard_input_ids,
                _attention_mask,
                _loss_mask,
                shard_hidden_states,
                shard_target_hs,
                _position_ids,
            ) = split_usp_batch(
                input_ids=input_ids,
                loss_mask=dummy_loss_mask,
                hidden_states=hidden_states,
                target_hidden_states=target_hs,
                ttt_length=ttt_length,
                sp_rank=sp_rank,
                sp_size=sp_size,
                ring_rank=ring_rank,
                sp_ring_size=sp_ring_size,
                max_len=max_seq_length,
            )

            shard_key = f"{key}_usp{sp_rank}"
            keys = [f"{shard_key}_hs", f"{shard_key}_ids"]
            tensors = [shard_hidden_states, shard_input_ids]
            if last_hidden_states is not None:
                keys.append(f"{shard_key}_lhs")
                tensors.append(shard_target_hs)
            else:
                keys.append(f"{shard_key}_tgt")
                tensors.append(shard_target_hs)
            self._put_raw_tensors(keys, tensors)

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        dtypes = {
            "hidden_states": hidden_states.dtype,
            "input_ids": input_ids.dtype,
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype

        logger.debug("put_usp_shards: completed key=%s, shapes=%s", key, shapes)
        return {"shapes": shapes, "dtypes": dtypes}

    def put(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Store Eagle3 output tensors via async batch_put_from.

        DtoH staging runs on ``_copy_stream`` so the caller's compute stream
        is never blocked.  The RDMA transfer runs on a background thread via
        ``AsyncPutManager``.  With *pool_size* host buffers the caller almost
        never waits — ``wait_for_buffer`` only blocks when every buffer is
        still in-flight.

        For GPU Direct send the path is synchronous (no DtoH needed).

        Returns a dict with ``"shapes"`` and ``"dtypes"`` sub-dicts that
        reflect the *actually stored* tensors (post dtype-cast).  Callers
        should forward these to consumers so metadata always matches bytes.
        """
        self._ensure_initialized()
        logger.debug("put: starting for key=%s", key)

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if (
            last_hidden_states is not None
            and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE
        ):
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        keys = [f"{key}_hs", f"{key}_ids"]
        tensors = [hidden_states, input_ids]

        if target is not None:
            keys.append(f"{key}_tgt")
            tensors.append(target)

        if last_hidden_states is not None:
            keys.append(f"{key}_lhs")
            tensors.append(last_hidden_states)

        self._put_raw_tensors(keys, tensors)

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        dtypes = {
            "hidden_states": hidden_states.dtype,
            "input_ids": input_ids.dtype,
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype

        logger.debug("put: completed key=%s, shapes=%s", key, shapes)
        return {"shapes": shapes, "dtypes": dtypes}

    def flush(self) -> None:
        """Block until all in-flight async puts have completed.

        Called before returning mooncake keys to the controller so that
        consumers can GET immediately.  Because ``put()`` stages DtoH on
        ``_copy_stream``, the copies are typically finished by the time
        this is called — the wait is only for the (fast) RDMA transfer.
        """
        self._ensure_initialized()
        if self._async_put_manager is None:
            return
        self._async_put_manager.check_last_error()
        self._async_put_manager.drain()
        self._async_put_manager.check_last_error()

    def _do_sync_batch_put(
        self,
        keys: List[str],
        buffer_ptrs: List[int],
        sizes: List[int],
    ) -> None:
        """Synchronous batch_put_from with error handling."""
        total_bytes = sum(sizes)
        if self._replicate_config is not None:
            results = self._store.batch_put_from(
                keys, buffer_ptrs, sizes, config=self._replicate_config
            )
        else:
            results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
        failures = [(k, r) for k, r in zip(keys, results) if r != 0]
        if failures:
            try:
                self._store.batch_remove(keys, force=True)
            except Exception:
                logger.warning(
                    "Failed to cleanup keys after batch_put_from failure: %s",
                    keys,
                    exc_info=True,
                )
            failure_details = ", ".join(f"{k} (code={r})" for k, r in failures)
            config_details = (
                f"total_bytes={_format_bytes(total_bytes)}, "
                f"global_segment_size={_format_bytes(self.config.global_segment_size)}, "
                f"local_buffer_size={_format_bytes(self.config.local_buffer_size)}, "
                f"host_buffer_size={_format_bytes(self.config.host_buffer_size)}"
            )
            raise RuntimeError(
                f"batch_put_from failed for keys: {failure_details}. "
                f"{config_details}. Consider increasing Mooncake segment/buffer sizes "
                "or reducing batch/sequence length/prefetch depth."
            )

    @staticmethod
    def _stage_tensors_into_buffer(
        buf, tensors: List[torch.Tensor]
    ) -> Tuple[List[int], List[int]]:
        """Copy tensors into a buffer and return (pointers, sizes) for batch_put_from."""
        buffer_ptrs = []
        sizes = []
        offset = 0
        for tensor in tensors:
            nbytes = buf.copy_from_tensor(tensor, offset=offset)
            buffer_ptrs.append(buf.ptr + offset)
            sizes.append(nbytes)
            offset += nbytes
        return buffer_ptrs, sizes

    def get(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ) -> "Eagle3TargetOutput":
        """
        Retrieve Eagle3 tensors into GPU memory.

        For RDMA/InfiniBand: Uses GPUDirect RDMA (batch_get_into directly into GPU).
        For TCP: Uses batch_get_buffer to host buffer, then copies to GPU.

        Automatically falls back to host buffer path if GPUDirect fails.

        Returns:
            Eagle3TargetOutput with the retrieved tensors.
        """
        self._ensure_initialized()

        from angelspec.models.target.eagle3_target_model import Eagle3TargetOutput

        keys = [f"{key}_hs", f"{key}_ids"]
        tensor_specs = [
            (
                "hidden_states",
                shapes["hidden_states"],
                dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE),
            ),
            ("input_ids", shapes["input_ids"], torch.int64),
        ]

        if "target" in shapes:
            keys.append(f"{key}_tgt")
            tensor_specs.append(
                ("target", shapes["target"], dtypes.get("target", HIDDEN_STATES_STORAGE_DTYPE))
            )

        if "last_hidden_states" in shapes:
            keys.append(f"{key}_lhs")
            tensor_specs.append(
                (
                    "last_hidden_states",
                    shapes["last_hidden_states"],
                    dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE),
                )
            )

        tensor_map = None
        if self._gpu_direct_available and self._gpu_receive_buffer is not None:
            tensor_map = self._get_tensors_gpu_direct(keys, tensor_specs, device)
            if tensor_map is None:
                logger.warning(
                    "GPUDirect batch_get_into failed; falling back to host buffer path."
                )

        if tensor_map is None:
            tensor_map = self._get_tensors_via_host_buffer(keys, tensor_specs, device)
            logger.debug("Using host buffer path (TCP)")
        logger.debug("Retrieved Eagle3 tensors with base key: %s", key)

        return Eagle3TargetOutput(
            hidden_states=tensor_map["hidden_states"],
            target=tensor_map.get("target"),
            input_ids=tensor_map["input_ids"],
            last_hidden_states=tensor_map.get("last_hidden_states"),
            input_ids_cpu=tensor_map.get("input_ids_cpu"),
        )

    def _get_tensors_gpu_direct(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Transfer directly into GPU memory using batch_get_into (GPUDirect RDMA).

        Pre-allocates individual destination tensors and transfers directly into
        their memory, avoiding an extra GPU→GPU copy.  Falls back to the
        registered GPU receive buffer when a tensor is too small to be
        page-aligned (RDMA typically requires registered memory).

        Returns None if transfer fails, allowing caller to fall back to host
        buffer path.
        """
        total_size = sum(
            self._compute_tensor_size(shape, dtype) for _, shape, dtype in tensor_specs
        )

        if total_size > self._gpu_receive_buffer.size:
            logger.warning(
                "GPU buffer too small: need %.1fMB, have %.1fMB. Increase gpu_buffer_size in config.",
                total_size / (1024**2),
                self._gpu_receive_buffer.size / (1024**2),
            )
            return None

        # Compute per-tensor sizes and buffer offsets up-front.
        buffer_ptrs: List[int] = []
        sizes: List[int] = []
        offsets: List[int] = []
        offset = 0

        for _, shape, dtype in tensor_specs:
            size = self._compute_tensor_size(shape, dtype)
            buffer_ptrs.append(self._gpu_receive_buffer.ptr + offset)
            sizes.append(size)
            offsets.append(offset)
            offset += size

        try:
            results = self._store.batch_get_into(keys, buffer_ptrs, sizes)
            for i, (k, r) in enumerate(zip(keys, results)):
                if r < 0:
                    logger.warning("batch_get_into failed for %s with error code: %s", k, r)
                    return None
                if r != 0 and r != sizes[i]:
                    logger.warning(
                        "batch_get_into for %s: unexpected return %s (expected 0 or %s)",
                        k,
                        r,
                        sizes[i],
                    )
        except Exception as e:
            logger.warning("batch_get_into exception: %s", e)
            return None

        tensor_map = {}
        for i, (name, shape, dtype) in enumerate(tensor_specs):
            numel = 1
            for dim in shape:
                numel *= dim
            buf_slice = self._gpu_receive_buffer.get_slice(offsets[i], sizes[i])
            # View into the registered buffer; valid until the next call.
            tensor_map[name] = buf_slice.view(dtype)[:numel].reshape(shape)

        logger.debug("GPU Direct RDMA transfer successful for %s tensors", len(keys))
        return tensor_map

    @staticmethod
    def _compute_tensor_size(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Compute the byte size of a tensor with given shape and dtype."""
        numel = 1
        for dim in shape:
            numel *= dim
        return numel * _DTYPE_ELEMENT_SIZES[dtype]

    def _get_tensors_via_host_buffer(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Transfer via Mooncake's registered host buffer, then copy to device."""
        wait_seconds = max(self.config.get_retry_wait_seconds, 0.05)
        log_interval = max(self.config.get_retry_log_interval_seconds, wait_seconds)
        max_wait = max(self.config.get_retry_max_wait_seconds, 0.0)
        start_time = time.time()
        last_log = 0.0

        while True:
            buffers = self._store.batch_get_buffer(keys)
            missing = [i for i, buf in enumerate(buffers) if buf is None]
            if not missing:
                break

            elapsed = time.time() - start_time
            if max_wait > 0 and elapsed >= max_wait:
                missing_keys = ", ".join(keys[i] for i in missing)
                raise RuntimeError(
                    f"batch_get_buffer returned None for keys: {missing_keys}. Waited {elapsed:.1f}s; aborting."
                )

            now = time.time()
            if last_log == 0.0 or (now - last_log) >= log_interval:
                missing_keys = ", ".join(keys[i] for i in missing)
                logger.warning(
                    "batch_get_buffer missing keys (%s); sleeping %.2fs.",
                    missing_keys,
                    wait_seconds,
                )
                last_log = now
            time.sleep(wait_seconds)

        tensor_map = {}
        for i, ((name, shape, dtype), buf) in enumerate(zip(tensor_specs, buffers)):
            if buf is None:
                raise RuntimeError(
                    f"batch_get_buffer returned None for key '{keys[i]}' (tensor: {name}). "
                    "This may indicate the key doesn't exist or RDMA transfer failed."
                )

            numel = 1
            for dim in shape:
                numel *= dim
            element_size = _DTYPE_ELEMENT_SIZES[dtype]
            expected_size = numel * element_size

            buf_size = buf.size()
            if buf_size != expected_size:
                actual_elements = buf_size // element_size if element_size > 0 else 0
                logger.error(
                    f"Size mismatch for {name} (key={keys[i]}): "
                    f"got {buf_size} bytes ({actual_elements} elements), "
                    f"expected {expected_size} bytes ({numel} elements). "
                    f"Expected shape: {shape}, dtype: {dtype}, element_size: {element_size}"
                )
                raise RuntimeError(
                    f"Size mismatch for {name}: got {buf_size} bytes, expected {expected_size} bytes"
                )

            c_array = (ctypes.c_byte * buf_size).from_address(buf.ptr())
            host_tensor = torch.frombuffer(c_array, dtype=dtype, count=numel).reshape(shape)

            tensor_map[name] = host_tensor.to(device)

            if name == "input_ids":
                tensor_map["input_ids_cpu"] = host_tensor.clone()

        return tensor_map

    def remove_eagle3_tensors(
        self,
        key: str,
        has_last_hidden_states: bool = False,
        has_target: bool = False,
    ) -> None:
        """Force-delete all tensors associated with an Eagle3 output.

        Uses ``batch_remove(force=True)`` to bypass lease TTL and delete
        immediately after consumption. Retries up to 3 times on failure.
        Never raises — deletion is best-effort to avoid breaking the
        training fetch path.

        Args:
            key: Base key used when storing
            has_last_hidden_states: Whether last_hidden_states was stored
            has_target: Whether target (logits) was stored
        """
        keys = [f"{key}_hs", f"{key}_ids"]
        if has_target:
            keys.append(f"{key}_tgt")
        if has_last_hidden_states:
            keys.append(f"{key}_lhs")

        for attempt in range(1, 4):
            try:
                results = self._store.batch_remove(keys, force=True)
            except Exception:
                if attempt < 3:
                    logger.warning(
                        "batch_remove raised for %s (attempt %d/3)", key, attempt, exc_info=True
                    )
                    time.sleep(0.5)
                else:
                    logger.error(
                        "Force delete abandoned for %s after 3 exceptions", key, exc_info=True
                    )
                continue
            failed = [(k, r) for k, r in zip(keys, results) if r not in (None, 0, -704)]
            if not failed:
                logger.debug("Force-deleted %s (%d keys)", key, len(results))
                return
            if attempt < 3:
                time.sleep(0.5)
                logger.warning(
                    "Retrying force delete for %s: %d keys failed (attempt %d/3)",
                    key,
                    len(failed),
                    attempt,
                )
            else:
                logger.error("Force delete abandoned for %s: %s", key, failed)
                return
            keys = [k for k, _ in failed]
