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
from dataclasses import dataclass
from typing import Tuple

from angelspec.transfer.mooncake.helpers import calculate_eagle3_buffer_size


@dataclass
class MooncakeConfig:
    """
    Configuration for Mooncake Store client.

    Memory parameters (accept ``str`` like ``"4GB"`` or ``int`` in bytes):
        global_segment_size: Memory this client CONTRIBUTES to the distributed pool.
            Other clients can store objects in this space. Inference workers should
            set this large enough to hold multiple outputs (~2-4GB per batch
            for large models).

        local_buffer_size: Memory for RECEIVING data via Get() operations.
            This is the buffer for reading objects from other clients. Trainers
            should set this large enough to receive hidden states (~512MB-2GB).

        gpu_buffer_size: Memory for accepting tensors from store directly into GPU memory.
            This is auto determined based on the training config.

    Note: The metadata_server defaults to the Mooncake Master's built-in HTTP
    metadata server (same host as master, port 8090).
    """

    local_hostname: str = "localhost"
    metadata_server: str = "http://localhost:8090/metadata"
    master_server_address: str = "localhost:50051"
    global_segment_size: str | int = 4 * 1024 * 1024 * 1024
    local_buffer_size: str | int = 512 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    gpu_buffer_size: str | int | None = None
    enable_gpu_direct: bool = False
    replica_num: int = 1
    enable_hard_pin: bool = False
    host_buffer_size: str | int | None = None
    get_batch_size: int = 1
    max_seq_len: int = 8192
    hidden_dim: int = 4096
    # Number of target/aux hidden-state layers concatenated per sample. Drives
    # buffer sizing: a sample carries hidden_dim * num_aux_layers bf16 values.
    # DFlash captures 5, DFlare 9; the default 3 (helpers default) only suited
    # the original Eagle3 and silently under-sized the buffer for deeper drafts.
    num_aux_layers: int = 3
    async_put_pool_size: int | None = None
    store_full_error_codes: Tuple[int, ...] = (-200,)
    store_full_wait_seconds: float = 0.5
    store_full_log_interval_seconds: float = 5.0
    store_full_max_wait_seconds: float = 0.0
    get_retry_wait_seconds: float = 0.5
    get_retry_log_interval_seconds: float = 10.0
    get_retry_max_wait_seconds: float = 60.0
    kv_lease_ttl_s: float = 5.0  # Mooncake master lease TTL only; not used for deletion timing

    def __post_init__(self):
        # Coerce size fields: accept str ("4GB") or int
        for field_name in (
            "global_segment_size",
            "local_buffer_size",
            "host_buffer_size",
            "gpu_buffer_size",
        ):
            val = getattr(self, field_name)
            if isinstance(val, str):
                setattr(self, field_name, self.parse_size(val))

        if self.host_buffer_size is None:
            # put() stages one sample at a time, so batch_size=1
            self.host_buffer_size = calculate_eagle3_buffer_size(
                max_seq_len=self.max_seq_len,
                batch_size=1,
                hidden_dim=self.hidden_dim,
                num_aux_layers=self.num_aux_layers,
                safety_margin=2.0,
            )

        if self.async_put_pool_size is None:
            self.async_put_pool_size = 1

        if self.gpu_buffer_size is None and self.enable_gpu_direct:
            # Size for a single get() call: get_batch_size samples (typically
            # per_dp_rank_batch_size, defaults to 1).
            self.gpu_buffer_size = calculate_eagle3_buffer_size(
                max_seq_len=self.max_seq_len,
                batch_size=self.get_batch_size,
                hidden_dim=self.hidden_dim,
                num_aux_layers=self.num_aux_layers,
            )

    @classmethod
    def from_flat_args(cls, args) -> "MooncakeConfig":
        """Create config from a flat args namespace (mooncake_* prefixed fields).

        Handles:
        - metadata_server URL construction from metadata_port if metadata_server is not set
        - local_hostname auto-resolution via RayActor.get_node_ip()
        - Size string parsing (handled automatically by __post_init__)
        """
        from angelspec.ray.ray_actor import RayActor

        master_server_address = getattr(args, "mooncake_master_server_address", None)
        metadata_port = getattr(args, "mooncake_metadata_port", None)
        metadata_server = getattr(args, "mooncake_metadata_server", None)

        if metadata_server is None and master_server_address is not None:
            master_host = master_server_address.split(":")[0]
            port = metadata_port if metadata_port is not None else 8090
            metadata_server = f"http://{master_host}:{port}/metadata"

        local_hostname = getattr(args, "mooncake_local_hostname", None)
        if local_hostname is None or local_hostname == "localhost":
            local_hostname = RayActor.get_node_ip()

        kwargs = {
            "local_hostname": local_hostname,
            "master_server_address": master_server_address or "localhost:50051",
            "global_segment_size": getattr(args, "mooncake_global_segment_size", "4GB"),
            "local_buffer_size": getattr(args, "mooncake_local_buffer_size", "512MB"),
            "host_buffer_size": getattr(args, "mooncake_host_buffer_size", None),
            "protocol": getattr(args, "mooncake_protocol", "tcp"),
            "device_name": getattr(args, "mooncake_device_name", ""),
            "gpu_buffer_size": getattr(args, "mooncake_gpu_buffer_size", None),
            "enable_gpu_direct": getattr(args, "mooncake_enable_gpu_direct", False),
            "async_put_pool_size": getattr(
                args, "mooncake_async_put_pool_size", getattr(args, "inference_batch_size", 1)
            ),
            "get_batch_size": getattr(
                args, "mooncake_get_batch_size", getattr(args, "per_dp_rank_batch_size", 1)
            ),
            "kv_lease_ttl_s": getattr(args, "mooncake_kv_lease_ttl_s", 5.0),
            "enable_hard_pin": getattr(args, "mooncake_enable_hard_pin", False),
            "max_seq_len": getattr(
                args,
                "mooncake_max_seq_len",
                getattr(args, "max_seq_length", 8192),
            ),
            "hidden_dim": getattr(args, "mooncake_hidden_dim", 4096),
            # Buffer sizing must match how many target layers a sample carries.
            # DFlash/DFlare set dflash_num_target_layers (5 / 9); fall back to it
            # so the host/gpu buffers are sized for the actual draft, not the
            # legacy Eagle3 default of 3.
            "num_aux_layers": getattr(args, "dflash_num_target_layers", 3),
        }

        if metadata_server is not None:
            kwargs["metadata_server"] = metadata_server

        return cls(**kwargs)

    def export_env(self) -> None:
        """Export mooncake configuration as environment variables.

        Sets the environment variables that sglang and other mooncake clients
        read during initialization. The caller is responsible for setting
        ``local_hostname`` to the correct value beforehand.
        """
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = self.local_hostname
        os.environ["MOONCAKE_METADATA_SERVER"] = self.metadata_server
        os.environ["MOONCAKE_MASTER_SERVER"] = self.master_server_address
        os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(self.global_segment_size)
        os.environ["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(self.local_buffer_size)
        os.environ["MOONCAKE_HOST_BUFFER_SIZE"] = str(self.host_buffer_size)
        os.environ["MOONCAKE_PROTOCOL"] = self.protocol
        os.environ["MOONCAKE_DEVICE_NAME"] = self.device_name
        os.environ["MOONCAKE_ENABLE_GPU_DIRECT"] = "1" if self.enable_gpu_direct else "0"
        self.apply_env_defaults()
        if self.async_put_pool_size is not None:
            os.environ["MOONCAKE_ASYNC_PUT_POOL_SIZE"] = str(self.async_put_pool_size)
        os.environ["MOONCAKE_STORE_FULL_WAIT_SECONDS"] = str(self.store_full_wait_seconds)
        os.environ["MOONCAKE_STORE_FULL_LOG_INTERVAL_SECONDS"] = str(
            self.store_full_log_interval_seconds
        )
        os.environ["MOONCAKE_STORE_FULL_MAX_WAIT_SECONDS"] = str(self.store_full_max_wait_seconds)
        os.environ["MOONCAKE_GET_RETRY_WAIT_SECONDS"] = str(self.get_retry_wait_seconds)
        os.environ["MOONCAKE_GET_RETRY_LOG_INTERVAL_SECONDS"] = str(
            self.get_retry_log_interval_seconds
        )
        os.environ["MOONCAKE_GET_RETRY_MAX_WAIT_SECONDS"] = str(self.get_retry_max_wait_seconds)
        os.environ["MOONCAKE_ENABLE_HARD_PIN"] = "1" if self.enable_hard_pin else "0"

    def apply_env_defaults(self) -> None:
        """Apply Mooncake process defaults that are needed before client setup."""
        # Fix: https://github.com/kvcache-ai/Mooncake/issues/1986
        if self.protocol.lower() == "tcp" and "MC_STORE_MEMCPY" not in os.environ:
            # Mooncake's TCP-only memcpy fast path can segfault in same-host
            # multi-process get paths. Preserve an explicit user override.
            os.environ["MC_STORE_MEMCPY"] = "0"

    @classmethod
    def from_env(cls) -> "MooncakeConfig":
        """Create config from environment variables."""
        master_host = os.getenv("MOONCAKE_MASTER_HOST", "localhost")
        master_port = os.getenv("MOONCAKE_MASTER_PORT", "50051")
        metadata_port = os.getenv("MOONCAKE_METADATA_PORT", "8090")
        store_full_wait_seconds = float(os.getenv("MOONCAKE_STORE_FULL_WAIT_SECONDS", "0.5"))
        store_full_log_interval_seconds = float(
            os.getenv("MOONCAKE_STORE_FULL_LOG_INTERVAL_SECONDS", "5.0")
        )
        store_full_max_wait_seconds = float(
            os.getenv("MOONCAKE_STORE_FULL_MAX_WAIT_SECONDS", "0.0")
        )
        get_retry_wait_seconds = float(os.getenv("MOONCAKE_GET_RETRY_WAIT_SECONDS", "0.2"))
        get_retry_log_interval_seconds = float(
            os.getenv("MOONCAKE_GET_RETRY_LOG_INTERVAL_SECONDS", "5.0")
        )
        get_retry_max_wait_seconds = float(os.getenv("MOONCAKE_GET_RETRY_MAX_WAIT_SECONDS", "5.0"))

        host_buffer_env = os.getenv("MOONCAKE_HOST_BUFFER_SIZE")
        host_buffer_size = int(host_buffer_env) if host_buffer_env is not None else None

        pool_size_env = os.getenv("MOONCAKE_ASYNC_PUT_POOL_SIZE")
        async_put_pool_size = int(pool_size_env) if pool_size_env is not None else None

        return cls(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            metadata_server=os.getenv(
                "MOONCAKE_METADATA_SERVER",
                f"http://{master_host}:{metadata_port}/metadata",
            ),
            master_server_address=os.getenv(
                "MOONCAKE_MASTER_SERVER", f"{master_host}:{master_port}"
            ),
            global_segment_size=int(
                os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(4 * 1024 * 1024 * 1024))
            ),
            local_buffer_size=int(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(512 * 1024 * 1024))),
            host_buffer_size=host_buffer_size,
            async_put_pool_size=async_put_pool_size,
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            device_name=os.getenv("MOONCAKE_DEVICE_NAME", ""),
            enable_gpu_direct=os.getenv("MOONCAKE_ENABLE_GPU_DIRECT", "0") == "1",
            store_full_wait_seconds=store_full_wait_seconds,
            store_full_log_interval_seconds=store_full_log_interval_seconds,
            store_full_max_wait_seconds=store_full_max_wait_seconds,
            get_retry_wait_seconds=get_retry_wait_seconds,
            get_retry_log_interval_seconds=get_retry_log_interval_seconds,
            get_retry_max_wait_seconds=get_retry_max_wait_seconds,
            enable_hard_pin=os.getenv("MOONCAKE_ENABLE_HARD_PIN", "0") == "1",
        )

    @staticmethod
    def parse_size(size_str: str) -> int:
        """Parse size string like '4GB', '512MB', or '4G' to bytes."""
        size_str = size_str.upper().strip()
        # Ordered by suffix length (longest first) to avoid "4GB" matching "B".
        multipliers = [
            ("TB", 1024 * 1024 * 1024 * 1024),
            ("GB", 1024 * 1024 * 1024),
            ("MB", 1024 * 1024),
            ("KB", 1024),
            ("T", 1024 * 1024 * 1024 * 1024),
            ("G", 1024 * 1024 * 1024),
            ("M", 1024 * 1024),
            ("K", 1024),
            ("B", 1),
        ]
        for suffix, multiplier in multipliers:
            if size_str.endswith(suffix):
                return int(float(size_str[: -len(suffix)]) * multiplier)
        return int(size_str)
