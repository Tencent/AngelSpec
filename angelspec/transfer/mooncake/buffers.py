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

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from angelspec.utils.logging import logger


class HostBuffer:
    """
    RDMA-registered host buffer using pinned CPU memory.

    Provides a reusable buffer for zero-copy transfers to Mooncake Store.
    """

    def __init__(self, size: int):
        self.size = size
        # Allocate pinned CPU memory for RDMA compatibility
        self._tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self._ptr = self._tensor.data_ptr()

    @property
    def ptr(self) -> int:
        return self._ptr

    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        """
        Copy tensor data into this host buffer at the given offset.

        Returns the number of bytes copied.
        """
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()

        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        # Get a view into our buffer at the right offset
        host_view = self._tensor[offset : offset + nbytes]

        # PyTorch .copy_() handles CUDA→pinned-CPU directly in one DMA.
        # No need for .cpu() which would create an intermediate unpinned copy.
        host_view.copy_(tensor.view(torch.uint8).view(-1))

        return nbytes

    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
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
        for _ in range(self.pool_size):
            self._buffers.append(HostBuffer(self.buffer_size))
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


class AsyncPutManager:
    """Runs batch_put_from on a background thread so the caller can return immediately.

    Tracks one in-flight ``Future`` per host-buffer pointer.  Before a buffer is
    reused the caller must call :meth:`wait_for_buffer` to block until the
    previous transfer on that buffer completes.

    Multiple worker threads are used so DtoH-copy and event-synchronization can
    overlap, but ``batch_put_from`` calls are serialized with ``_put_lock``
    because ``MooncakeDistributedStore`` is not thread-safe for concurrent puts.
    """

    def __init__(self, store: Any, max_workers: int = 1, replicate_config: Any = None):
        self._store = store
        self._replicate_config = replicate_config
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="async_put"
        )
        self._in_flight: Dict[int, Future] = {}
        self._last_error: Optional[BaseException] = None
        self._put_lock = threading.Lock()

    def check_last_error(self) -> None:
        """Re-raise the first async failure that hasn't been surfaced yet."""
        if self._last_error is not None:
            err = self._last_error
            self._last_error = None
            raise err

    def wait_for_buffer(self, buffer_ptr: int) -> None:
        """Block until the in-flight transfer using *buffer_ptr* finishes.

        Raises the original exception if the transfer failed.
        """
        future = self._in_flight.pop(buffer_ptr, None)
        if future is None:
            return
        try:
            future.result()
        except Exception as exc:
            self._last_error = exc
            raise

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
        if wait_event is not None:
            if device_index is not None:
                torch.cuda.set_device(device_index)
            wait_event.synchronize()
        with self._put_lock:
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
                    "Failed to cleanup keys after async batch_put_from failure: %s",
                    keys,
                    exc_info=True,
                )
            detail = ", ".join(f"{k} (code={r})" for k, r in failures)
            raise RuntimeError(f"async batch_put_from failed: {detail}")

    def drain(self) -> None:
        """Wait for every in-flight transfer to finish."""
        for _ptr, future in list(self._in_flight.items()):
            try:
                future.result()
            except Exception as exc:
                if self._last_error is None:
                    self._last_error = exc
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

        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()

        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        gpu_view = self._tensor[offset : offset + nbytes]
        gpu_view.copy_(tensor.view(torch.uint8).view(-1))

        return nbytes
