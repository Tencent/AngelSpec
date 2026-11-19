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

import os
import resource
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from angelspec.utils.logging import logger

# Upper bound on the temporary allocated when staging a row-block source.
_STAGE_CHUNK_BYTES = 256 * 1024**2
PUT_RETRY_ATTEMPTS = int(os.environ.get("ANGELSPEC_MOONCAKE_PUT_RETRIES", "3"))
PUT_RETRY_BACKOFF_SECONDS = float(os.environ.get("ANGELSPEC_MOONCAKE_PUT_BACKOFF", "0.5"))


class ChunkedRowGather:
    """Lazy ``source[index]``, optionally column-sliced, staged in row blocks.

    Materialising the gather up front costs ~4.4 GiB for a 128k-token
    hidden-state block. Worse, its size tracks the request length, so a few
    hundred requests of varying length fragment the allocator badly — and only
    on the single rank that runs the connector. Producing row blocks on demand
    bounds both.

    The gather reads KV cache slots, which the scheduler recycles once the
    request completes. Staging on ``_copy_stream`` is still safe: the stream
    waits on the compute event before reading, and ``wait_for_save`` drains
    every put before the forward pass returns, so the step cannot end — and
    the blocks cannot be freed — with a gather still in flight.
    """

    def __init__(
        self,
        source: torch.Tensor,
        index: torch.Tensor,
        columns: Optional[slice] = None,
    ):
        if source.dim() != 2:
            raise ValueError(f"source must be 2-D, got {tuple(source.shape)}")
        # Slicing dim 1 is a view, so the gather below lands directly in the
        # final column layout instead of going through the full width.
        self._source = source if columns is None else source[:, columns]
        self._index = index
        self.shape = (index.shape[0], self._source.shape[1])
        self.dtype = self._source.dtype
        self.is_cuda = self._source.is_cuda

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def element_size(self) -> int:
        return self._source.element_size()

    def record_stream(self, stream) -> None:
        self._index.record_stream(stream)

    def rows(self, start: int, stop: int) -> torch.Tensor:
        """Materialise rows ``[start, stop)`` as a contiguous tensor."""
        return self._source[self._index[start:stop]]


def _stage_into_flat_view(dst: torch.Tensor, src) -> None:
    """Copy ``src`` into ``dst``, a flat uint8 view of exactly ``src``'s size.

    ``src`` is either a tensor or a :class:`ChunkedRowGather`. A strided tensor
    cannot be DMA'd as one block, and letting torch handle it materialises a
    contiguous copy of the whole thing on the GPU — ~3 GiB for the column
    slices of a 128k-token hidden-state block. Both cases therefore go through
    row blocks capped at ``_STAGE_CHUNK_BYTES``; row-major order means the
    bytes land exactly as a whole-tensor copy would.
    """
    if isinstance(src, ChunkedRowGather):
        _stage_row_blocks(dst, src.shape[0], src.numel() * src.element_size(), src.rows)
        return

    if src.is_contiguous():
        dst.copy_(src.view(torch.uint8).view(-1), non_blocking=src.is_cuda)
        return

    _stage_row_blocks(
        dst,
        src.shape[0],
        src.numel() * src.element_size(),
        lambda start, stop: src[start:stop].contiguous(),
    )


def _stage_row_blocks(dst: torch.Tensor, rows: int, nbytes: int, make_block) -> None:
    if rows == 0 or nbytes == 0:
        return

    row_bytes = nbytes // rows
    rows_per_chunk = max(1, _STAGE_CHUNK_BYTES // row_bytes)

    written = 0
    for start in range(0, rows, rows_per_chunk):
        block = make_block(start, min(start + rows_per_chunk, rows))
        block_bytes = block.numel() * block.element_size()
        dst[written : written + block_bytes].copy_(
            block.view(torch.uint8).view(-1),
            non_blocking=block.is_cuda,
        )
        written += block_bytes


class HostBuffer:
    """
    RDMA-registered host buffer using pinned CPU memory.

    Provides a reusable buffer for zero-copy transfers to Mooncake Store.
    """

    def __init__(self, size: int):
        self.size = size
        self._tensor = None
        self._ptr = 0
        diagnostics = os.environ.get("ANGELSPEC_PIN_MEMORY_DIAGNOSTICS") == "1"
        phase = "allocation"
        try:
            if diagnostics:
                self._log_allocation_context()
                if torch.cuda.is_initialized():
                    phase = "pre-allocation CUDA synchronization"
                    torch.cuda.synchronize()
            phase = "allocation"
            self._tensor = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
            self._ptr = self._tensor.data_ptr()
        except Exception:
            logger.exception("Pinned host buffer failed during %s (bytes=%s)", phase, size)
            if not diagnostics:
                self._log_allocation_context()
            raise

    def _log_allocation_context(self) -> None:
        try:
            initialized = torch.cuda.is_initialized()
            device = torch.cuda.current_device() if initialized else None
            libraries = sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libcuda" in line
                }
            )
            logger.info(
                "Pinned allocation: bytes=%s pid=%s cuda_initialized=%s device=%s "
                "visible_devices=%s memlock=%s torch=%s cuda=%s allocator=%s libraries=%s",
                self.size,
                os.getpid(),
                initialized,
                device,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
                resource.getrlimit(resource.RLIMIT_MEMLOCK),
                torch.__version__,
                torch.version.cuda,
                os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
                libraries,
            )
        except Exception:
            logger.warning("Could not collect pinned allocation diagnostics", exc_info=True)

    @property
    def ptr(self) -> int:
        return self._ptr

    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        """
        Copy tensor data into this host buffer at the given offset.

        Returns the number of bytes copied.
        """
        nbytes = tensor.numel() * tensor.element_size()

        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        # Get a view into our buffer at the right offset
        host_view = self._tensor[offset : offset + nbytes]

        # PyTorch .copy_() handles CUDA→pinned-CPU directly in one DMA.
        # No need for .cpu() which would create an intermediate unpinned copy.
        _stage_into_flat_view(host_view, tensor)

        return nbytes

    def free(self) -> None:
        self._tensor = None
        self._ptr = 0

    def __del__(self):
        self.free()


class HostBufferPool:
    """
    Pool of RDMA-registered host buffers for tensor storage.
    """

    def __init__(self, buffer_size: int = 4 * 1024**3, pool_size: int = 2):
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._buffers: List[HostBuffer] = []
        self._current_idx = 0

    def initialize(self) -> None:
        """Pre-allocate all buffers."""
        if self._buffers:
            return
        try:
            for _ in range(self.pool_size):
                self._buffers.append(HostBuffer(self.buffer_size))
        except Exception:
            self.shutdown()
            raise
        logger.info(
            "Initialized host buffer pool: %s x %.1fGB",
            self.pool_size,
            self.buffer_size / (1024**3),
        )

    def get_buffer(self) -> HostBuffer:
        """Get the next buffer (round-robin)."""
        if not self._buffers:
            self.initialize()
        buf = self._buffers[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._buffers)
        return buf

    def shutdown(self) -> None:
        for buf in self._buffers:
            buf.free()
        self._buffers.clear()
        self._current_idx = 0


class AsyncPutManager:
    """Runs batch_put_from on a background thread so the caller can return immediately.

    Tracks one in-flight ``Future`` per host-buffer pointer.  Before a buffer is
    reused the caller must call :meth:`wait_for_buffer` to block until the
    previous transfer on that buffer completes.

    Multiple worker threads are used so DtoH-copy and event-synchronization can
    overlap, but ``batch_put_from`` calls are serialized with ``_put_lock``
    because ``MooncakeDistributedStore`` is not thread-safe for concurrent puts.

    A failed transfer never raises into a later, unrelated put: its keys are
    recorded and callers collect them with :meth:`pop_failed_keys` after
    :meth:`drain`.
    """

    def __init__(self, store: Any, max_workers: int = 1, replicate_config: Any = None):
        self._store = store
        self._replicate_config = replicate_config
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="async_put"
        )
        self._in_flight: Dict[int, Future] = {}
        self._put_lock = threading.Lock()
        self._failed_lock = threading.Lock()
        self._failed_keys: Set[str] = set()

    def pop_failed_keys(self) -> Set[str]:
        with self._failed_lock:
            failed = self._failed_keys
            self._failed_keys = set()
        return failed

    def _record_failed_keys(self, keys: List[str]) -> None:
        with self._failed_lock:
            self._failed_keys.update(keys)

    def wait_for_buffer(self, buffer_ptr: int) -> None:
        """Block until the in-flight transfer using *buffer_ptr* finishes.

        A failed transfer only has its keys recorded; the caller is about to
        stage a different sample into the freed buffer.
        """
        future = self._in_flight.pop(buffer_ptr, None)
        if future is None:
            return
        try:
            future.result()
        except Exception as exc:
            logger.warning("async put failed: %s", exc)

    def submit(
        self,
        keys: List[str],
        buffer_ptrs: List[int],
        sizes: List[int],
        owner_buffer_ptr: int,
        wait_event: Optional[Any] = None,
        device_index: Optional[int] = None,
    ) -> None:
        """Submit a ``batch_put_from`` to the background thread.

        *owner_buffer_ptr* is the base pointer of the host buffer that owns the
        staged data — used as the key for :meth:`wait_for_buffer`.

        *wait_event* is an optional CUDA event to synchronize on before the
        RDMA transfer (used when DtoH staging runs on a separate stream).

        *device_index* is the CUDA device ordinal that owns *wait_event*.
        Worker threads do not inherit the caller's device context, so the
        worker must call ``torch.cuda.set_device`` before synchronizing.

        GPU tensor lifetime is managed by the caller via
        ``record_stream`` — the CUDA caching allocator keeps the underlying
        memory alive until the copy stream passes the recorded point.
        """
        future = self._executor.submit(
            self._do_put, keys, buffer_ptrs, sizes, wait_event, device_index
        )
        self._in_flight[owner_buffer_ptr] = future

    def _do_put(
        self,
        keys: List[str],
        buffer_ptrs: List[int],
        sizes: List[int],
        wait_event: Optional[Any] = None,
        device_index: Optional[int] = None,
    ) -> None:
        try:
            self._put_with_retries(keys, buffer_ptrs, sizes, wait_event, device_index)
        except Exception:
            self._record_failed_keys(keys)
            raise

    def _put_with_retries(
        self,
        keys: List[str],
        buffer_ptrs: List[int],
        sizes: List[int],
        wait_event: Optional[Any],
        device_index: Optional[int],
    ) -> None:
        if wait_event is not None:
            if device_index is not None:
                torch.cuda.set_device(device_index)
            wait_event.synchronize()
        detail = ""
        for attempt in range(1, PUT_RETRY_ATTEMPTS + 1):
            with self._put_lock:
                if self._replicate_config is not None:
                    results = self._store.batch_put_from(
                        keys, buffer_ptrs, sizes, config=self._replicate_config
                    )
                else:
                    results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
            failures = [(k, r) for k, r in zip(keys, results) if r != 0]
            if not failures:
                if attempt > 1:
                    logger.info("async batch_put_from succeeded on attempt %d: %s", attempt, keys)
                return

            try:
                self._store.batch_remove(keys, force=True)
            except Exception:
                logger.warning(
                    "Failed to cleanup keys after async batch_put_from failure: %s",
                    keys,
                    exc_info=True,
                )
            detail = ", ".join(f"{k} (code={r})" for k, r in failures)
            if attempt < PUT_RETRY_ATTEMPTS:
                logger.warning(
                    "async batch_put_from failed (attempt %d/%d), retrying: %s",
                    attempt,
                    PUT_RETRY_ATTEMPTS,
                    detail,
                )
                time.sleep(PUT_RETRY_BACKOFF_SECONDS * attempt)

        raise RuntimeError(
            f"async batch_put_from failed after {PUT_RETRY_ATTEMPTS} attempts: {detail}"
        )

    def drain(self) -> None:
        """Wait for every in-flight transfer to finish; failures are only recorded."""
        for _ptr, future in list(self._in_flight.items()):
            try:
                future.result()
            except Exception as exc:
                logger.warning("async put error during drain: %s", exc)
        self._in_flight.clear()

    def shutdown(self) -> None:
        """Drain pending work and shut down the thread pool."""
        self.drain()
        self._executor.shutdown(wait=True)


class _GPUBuffer:
    """Base class for RDMA-registered GPU buffers (send and receive).

    Provides allocation, pointer access, free, and lazy initialization.
    Subclasses add direction-specific methods (copy_from_tensor, get_slice).
    """

    _label: str = "GPU"  # overridden by subclasses for log messages

    def __init__(self, size: int, device: torch.device = None):
        self.size = size
        self.device = torch.device(device) if device is not None else torch.device("cuda")
        self._tensor: Optional[torch.Tensor] = None
        self._ptr: int = 0
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
        self._ptr = self._tensor.data_ptr()
        self._initialized = True
        logger.info(
            "Initialized %s buffer: %.1fMB on %s",
            self._label,
            self.size / (1024**2),
            self.device,
        )

    @property
    def ptr(self) -> int:
        return self._ptr

    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0
            self._initialized = False

    def __del__(self):
        self.free()


class GPUReceiveBuffer(_GPUBuffer):
    """RDMA-registered GPU buffer for receiving tensors via GPU Direct RDMA.

    Allocates contiguous GPU memory that can be registered with Mooncake
    for direct RDMA transfers without CPU involvement.
    """

    _label = "GPU receive"

    def get_slice(self, offset: int, size: int) -> torch.Tensor:
        """Get a slice of the buffer as a tensor view."""
        if not self._initialized:
            raise RuntimeError("GPU buffer not initialized")
        return self._tensor[offset : offset + size]


class GPUSendBuffer(_GPUBuffer):
    """RDMA-registered GPU buffer for sending tensors via GPU Direct RDMA.

    Pre-allocates contiguous GPU memory registered with Mooncake so that
    put operations can stage tensors with a fast GPU-to-GPU copy (~5ms for 1.2GB)
    instead of GPU-to-CPU (~150ms). The NIC reads directly from GPU memory via
    nvidia_peermem (GPU Direct RDMA).
    """

    _label = "GPU send"

    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        """Copy tensor data into this GPU buffer at the given offset.

        The source tensor must be on the same CUDA device. This is a fast
        GPU-to-GPU copy within device memory.

        Returns the number of bytes copied.
        """
        if not self._initialized:
            raise RuntimeError("GPU send buffer not initialized")

        nbytes = tensor.numel() * tensor.element_size()

        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        gpu_view = self._tensor[offset : offset + nbytes]
        _stage_into_flat_view(gpu_view, tensor)

        return nbytes
