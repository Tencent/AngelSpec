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
import threading
from abc import ABC
from typing import Any, Dict, Optional

import torch
from mooncake.store import MooncakeDistributedStore

from angelspec.config.mooncake_config import MooncakeConfig
from angelspec.transfer.mooncake.buffers import (
    AsyncPutManager,
    GPUReceiveBuffer,
    GPUSendBuffer,
    HostBufferPool,
)
from angelspec.utils.logging import logger


class MooncakeHiddenStateStore(ABC):
    """
    Base class for Mooncake Store wrapper to store hidden states from target model.

    Uses RDMA-registered host buffers (MooncakeHostMemAllocator) and put_from
    for zero-copy transfers. Optionally uses GPU Direct RDMA for both sending
    and receiving.
    """

    def __init__(self, config: MooncakeConfig):
        self.config = config
        self._store: Optional[MooncakeDistributedStore] = None
        self._initialized = False
        self._init_event = threading.Event()
        self._registered_buffers: Dict[int, int] = {}
        self._host_buffer_pool: Optional[HostBufferPool] = None
        self._async_put_manager: Optional[AsyncPutManager] = None
        self._gpu_receive_buffer: Optional[GPUReceiveBuffer] = None
        self._gpu_send_buffer: Optional[GPUSendBuffer] = None
        self._gpu_direct_available = False
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._replicate_config: Any = None

    def setup(self, device: torch.device | int | None = None) -> None:
        """Initialize the Mooncake Store client."""
        if self._initialized:
            return

        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)

        if self.config.protocol == "rdma" and not self.config.device_name:
            logger.warning(
                "RDMA protocol with empty device_name will use auto-discovery, "
                "which may fail on hosts with mixed IB subnets. "
                "Set mooncake.device_name to a specific RDMA device (e.g. 'mlx5_0')."
            )

        self.config.apply_env_defaults()
        # local_hostname may have been baked on the DRIVER node (build_mooncake_config
        # runs there) and shipped to every actor, so on a remote training node it
        # wrongly carries the driver's IP. Re-resolve it to THIS process's node IP
        # so each actor registers/binds on its own NIC.
        try:
            from angelspec.ray.ray_actor import RayActor

            local_ip = RayActor.get_node_ip()
            if local_ip and local_ip != self.config.local_hostname:
                logger.info(
                    "Overriding mooncake local_hostname %s -> %s (actual node IP)",
                    self.config.local_hostname,
                    local_ip,
                )
                self.config.local_hostname = local_ip
                os.environ["MOONCAKE_LOCAL_HOSTNAME"] = local_ip
        except Exception:
            logger.warning(
                "Could not re-resolve local node IP for mooncake; using configured value"
            )
        # The transfer engine reaches the HTTP metadata server over intra-cluster
        # IPs; an HTTP proxy (commonly exported by a login shell's .bashrc and
        # inherited by spawned vLLM workers) routes the metadata PUT through it
        # and it's rejected (http=400 -> rc=-200 -> batch_put_from -900). libcurl's
        # no_proxy doesn't honour CIDR, so just hard-unset the proxy vars in THIS
        # process before the transfer engine starts.
        for var in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
        ):
            if os.environ.pop(var, None):
                logger.info("Unset %s for mooncake transfer engine", var)
        # Pin the transfer-engine TCP socket to the interface the cluster
        # addresses this node by (local_hostname == node IP). Without this
        # mooncake auto-picks an arbitrary NIC, so the segment it registers
        # under <node_ip>:<port> is unreachable by peers -> "metadata not
        # found" 404 and
        # batch_put_from fails with code -900. Scoped to the client process
        # only (NOT the master subprocess, whose RPC bind must stay 0.0.0.0).
        if (
            self.config.protocol.lower() == "tcp"
            and "MC_TCP_BIND_ADDRESS" not in os.environ
            and self.config.local_hostname
            and self.config.local_hostname != "localhost"
        ):
            os.environ["MC_TCP_BIND_ADDRESS"] = self.config.local_hostname
            logger.info(
                "Set MC_TCP_BIND_ADDRESS=%s for transfer engine", self.config.local_hostname
            )
        self._store = MooncakeDistributedStore()
        logger.info(
            "Connecting to Mooncake master at %s, metadata server at %s",
            self.config.master_server_address,
            self.config.metadata_server,
        )
        result = self._store.setup(
            local_hostname=self.config.local_hostname,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
        )
        if result is not None and result != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake client (error={result}). "
                f"Check that Mooncake master is running at {self.config.master_server_address} "
                f"and metadata server is available at {self.config.metadata_server}"
            )

        self._verify_force_delete()
        self._build_replicate_config()

        pool_size = self.config.async_put_pool_size
        if pool_size > 0:
            self._host_buffer_pool = HostBufferPool(
                buffer_size=self.config.host_buffer_size,
                pool_size=pool_size,
            )
            self._host_buffer_pool.initialize()

            for buf in self._host_buffer_pool._buffers:
                self._register_buffer(buf.ptr, buf.size)

            self._async_put_manager = AsyncPutManager(
                store=self._store, max_workers=pool_size, replicate_config=self._replicate_config
            )
            logger.info("Async put manager created (pool_size=%d)", pool_size)

        if self.config.enable_gpu_direct and torch.cuda.is_available():
            self._setup_gpu_direct(device)

        if torch.cuda.is_available():
            cuda_device = device if device is not None else torch.device("cuda")
            self._copy_stream = torch.cuda.Stream(device=cuda_device)
            logger.info("DtoH copy stream created on %s", cuda_device)

        self._initialized = True
        self._init_event.set()
        logger.info(
            "Mooncake Store client initialized (protocol=%s, device=%s, gpu_direct=%s, pool_size=%d)",
            self.config.protocol,
            self.config.device_name or "(auto-discovery)",
            self._gpu_direct_available,
            pool_size,
        )

    def _setup_gpu_direct(self, device: torch.device = None) -> None:
        """Initialize GPU send/receive buffers and register for GPU Direct RDMA."""
        try:
            self._gpu_receive_buffer = GPUReceiveBuffer(
                size=self.config.gpu_buffer_size,
                device=device,
            )
            self._gpu_receive_buffer.initialize()

            if not self._register_buffer(
                self._gpu_receive_buffer.ptr, self._gpu_receive_buffer.size
            ):
                logger.warning("Failed to register GPU receive buffer with Mooncake")
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None
                return

            # Allocate GPU send buffer (same size as host buffer — put is 1 sample)
            self._gpu_send_buffer = GPUSendBuffer(
                size=self.config.host_buffer_size,
                device=device,
            )
            self._gpu_send_buffer.initialize()

            if not self._register_buffer(self._gpu_send_buffer.ptr, self._gpu_send_buffer.size):
                logger.warning("Failed to register GPU send buffer with Mooncake")
                self._gpu_send_buffer.free()
                self._gpu_send_buffer = None
                # receive buffer is still usable, continue

            self._gpu_direct_available = True
            send_desc = (
                f"{self._gpu_send_buffer.size / (1024**2):.1f}MB"
                if self._gpu_send_buffer
                else "N/A"
            )
            logger.info(
                "GPU Direct RDMA enabled: receive=%.1fMB, send=%s on %s",
                self._gpu_receive_buffer.size / (1024**2),
                send_desc,
                device or "cuda",
            )

        except Exception as e:
            logger.warning("Failed to setup GPU Direct RDMA: %s", e)
            self._gpu_direct_available = False
            if self._gpu_receive_buffer is not None:
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None
            if self._gpu_send_buffer is not None:
                self._gpu_send_buffer.free()
                self._gpu_send_buffer = None

    def _ensure_initialized(self, timeout: float = 30.0) -> None:
        """Block until setup() has completed (thread-safe via Event).

        If setup() was called from a background thread, this will wait up to
        *timeout* seconds for it to finish.  If nobody has called setup() yet,
        falls back to calling it on the current thread.
        """
        if self._init_event.is_set():
            return
        if not self._init_event.wait(timeout=timeout):
            if not self._initialized:
                self.setup()

    def _register_buffer(self, buffer_ptr: int, size: int) -> bool:
        """Register a buffer for RDMA transfers."""
        if buffer_ptr in self._registered_buffers:
            return True

        try:
            if hasattr(self._store, "register_buffer"):
                result = self._store.register_buffer(buffer_ptr, size)
                if result == 0:
                    self._registered_buffers[buffer_ptr] = size
                    logger.debug("Registered buffer at %#x, size=%s", buffer_ptr, size)
                    return True
                logger.warning("register_buffer returned error code: %s", result)
                return False
        except Exception as e:
            logger.warning("Failed to register buffer: %s", e)

        return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in the store (metadata-only, no data download)."""
        try:
            result = self._store.is_exist(key)
            return result == 1
        except Exception:
            return False

    def _verify_force_delete(self) -> None:
        """Fail-fast if Mooncake doesn't support batch_remove(force=True).

        Requires mooncake-transfer-engine >= 0.3.10.post1.
        Primary check uses package version metadata; falls back to docstring
        heuristic for non-pip installs.
        """
        batch_remove = getattr(self._store, "batch_remove", None)
        if batch_remove is None:
            raise RuntimeError(
                "Mooncake version too old: batch_remove() not found. Requires mooncake-transfer-engine >= 0.3.10.post1."
            )
        try:
            from importlib.metadata import version

            from packaging.version import Version

            installed = Version(version("mooncake-transfer-engine"))
            if installed >= Version("0.3.10.post1"):
                return
        except Exception:
            pass
        doc = getattr(batch_remove, "__doc__", "") or ""
        if "force" not in doc:
            raise RuntimeError(
                "Mooncake version too old: batch_remove(force=True) not supported. "
                "Requires mooncake-transfer-engine >= 0.3.10.post1."
            )

    def _build_replicate_config(self) -> None:
        """Build ReplicateConfig for batch_put_from if hard_pin is enabled and supported."""
        self._replicate_config = None
        if not self.config.enable_hard_pin:
            return
        try:
            from mooncake.store import ReplicateConfig

            cfg = ReplicateConfig()
            if hasattr(cfg, "with_hard_pin"):
                cfg.with_hard_pin = True
                self._replicate_config = cfg
                logger.info("Hard pin enabled for batch_put_from")
            else:
                logger.warning(
                    "enable_hard_pin=True but ReplicateConfig lacks with_hard_pin attr (needs unreleased Mooncake)"
                )
        except ImportError:
            logger.warning("enable_hard_pin=True but ReplicateConfig not importable")

    def close(self) -> None:
        """Close the Mooncake Store client."""
        if self._async_put_manager is not None:
            self._async_put_manager.shutdown()
            self._async_put_manager = None
        if self._gpu_send_buffer is not None:
            self._gpu_send_buffer.free()
            self._gpu_send_buffer = None
        if self._gpu_receive_buffer is not None:
            self._gpu_receive_buffer.free()
            self._gpu_receive_buffer = None
        if self._host_buffer_pool is not None:
            self._host_buffer_pool.shutdown()
            self._host_buffer_pool = None
        self._copy_stream = None
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._initialized = False
        self._init_event.clear()
        self._gpu_direct_available = False
