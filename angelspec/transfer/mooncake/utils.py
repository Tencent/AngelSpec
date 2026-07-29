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

import atexit
import ctypes
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

import ray

from angelspec.ray.ray_actor import RayActor
from angelspec.utils.env import get_angelspec_env_vars
from angelspec.utils.logging import logger


def resolve_mooncake_master_bin() -> str:
    """Resolve the path to the mooncake_master binary."""
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")

    which_result = shutil.which("mooncake_master")
    if which_result:
        return which_result

    home = os.path.expanduser("~")
    return os.path.join(home, "build/mooncake-store/src/mooncake_master")


def _subprocess_preexec():
    """Pre-exec setup for the mooncake master subprocess.

    - os.setpgrp(): Create a new process group so that os.killpg() can kill
      the wrapper script AND the real binary it spawns (grandchild).
    - PR_SET_PDEATHSIG: Kernel sends SIGTERM when the parent (Ray worker) dies,
      preventing orphans on crashes.
    """
    os.setpgrp()
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


class MooncakeMaster(RayActor):
    """Ray actor that wraps the mooncake master subprocess.

    Provides automatic lifecycle management — when the actor is killed or garbage
    collected, the subprocess is terminated. Logs are streamed through Ray's
    logging pipeline instead of written to files.
    """

    def __init__(self):
        self._process = None
        self._info = {}

    def start(
        self,
        port: int,
        http_port: int,
        http_host: str = "0.0.0.0",
        kv_lease_ttl_s: float = 5.0,
    ) -> dict:
        """Launch the mooncake master subprocess.

        Args:
            port: gRPC port for mooncake master.
            http_port: HTTP metadata server port.
            http_host: HTTP metadata server host.
            kv_lease_ttl_s: Default KV object lease TTL in seconds.

        Returns:
            Dict with "master_addr" and "metadata_port".

        Raises:
            FileNotFoundError: If binary is not found.
            RuntimeError: If process fails to start.
        """
        mooncake_bin = resolve_mooncake_master_bin()
        if not os.path.exists(mooncake_bin):
            raise FileNotFoundError(f"mooncake_master binary not found at {mooncake_bin}")

        cmd = [
            mooncake_bin,
            # mooncake >= 0.3.11 deprecates --port in favour of --rpc_port; the
            # old --port path can leave the RPC service stuck in standby
            # (service_ready=false) so clients get connection-refused / err -900.
            f"--rpc_port={port}",
            f"--http_metadata_server_port={http_port}",
            f"--http_metadata_server_host={http_host}",
            "--enable_http_metadata_server=true",
            f"--default_kv_lease_ttl={int(kv_lease_ttl_s * 1000)}",
        ]

        logger.info(f"Starting mooncake master on port {port}")
        logger.info(
            f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}, LD_PRELOAD={os.environ.get('LD_PRELOAD', 'NOT SET')}"
        )

        # The mooncake_master binary links a C++ extension that needs a recent
        # libstdc++ (GLIBCXX). When the active environment's libstdc++ is newer
        # than the system one, prepend the env's lib dir (and preload its
        # libstdc++) so the subprocess resolves the right one. Paths are derived
        # from the active interpreter prefix — not hardcoded — so this is
        # portable across environments. Override via MOONCAKE_SUBPROCESS_LIB_DIR.
        popen_env = os.environ.copy()
        # The master must bind its RPC server on 0.0.0.0; a stray
        # MC_TCP_BIND_ADDRESS (set for store clients to pick the right NIC)
        # would pin the RPC socket to one IP and make gRPC unreachable.
        popen_env.pop("MC_TCP_BIND_ADDRESS", None)
        env_lib_dir = os.environ.get(
            "MOONCAKE_SUBPROCESS_LIB_DIR", os.path.join(sys.prefix, "lib")
        )
        if os.path.isdir(env_lib_dir):
            popen_env["LD_LIBRARY_PATH"] = env_lib_dir + ":" + popen_env.get("LD_LIBRARY_PATH", "")
            libstdcpp = os.path.join(env_lib_dir, "libstdc++.so.6")
            if os.path.exists(libstdcpp):
                popen_env["LD_PRELOAD"] = libstdcpp + (
                    ":" + popen_env["LD_PRELOAD"] if popen_env.get("LD_PRELOAD") else ""
                )
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_subprocess_preexec,
            env=popen_env,
        )

        # Stream stdout/stderr through logger in background daemon threads
        self._start_log_thread(self._process.stdout, "stdout")
        self._start_log_thread(self._process.stderr, "stderr")

        time.sleep(2)

        if self._process.poll() is not None:
            raise RuntimeError(
                f"mooncake master failed to start (exit code: {self._process.returncode})"
            )

        host = self.get_node_ip()
        self._info = {
            "master_addr": f"{host}:{port}",
            "metadata_port": http_port,
        }

        logger.info(f"mooncake master started (PID: {self._process.pid})")
        return self._info

    def health_check(self) -> bool:
        """Check if the subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_info(self) -> dict:
        """Return the master address and metadata port."""
        return self._info

    def _start_log_thread(self, stream, name: str) -> None:
        """Start a daemon thread that reads lines from stream and logs them."""

        def _reader():
            for line in stream:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.rstrip("\n")
                if line:
                    logger.debug(f"[mooncake_master {name}] {line}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def shutdown(self):
        """Gracefully terminate the subprocess and its entire process group."""
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except Exception:
                    pass
            self._process = None

    def __del__(self):
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass


def check_mooncake_master_available(
    master_server_address: str,
    metadata_server: str,
    timeout: float = 5.0,
) -> None:
    """Verify mooncake master services are reachable.

    Probes the gRPC endpoint via TCP connect and the HTTP metadata endpoint
    so actors fail fast with a clear error before expensive model loading.

    Args:
        master_server_address: gRPC address, e.g. "10.1.2.3:50051".
        metadata_server: HTTP metadata URL, e.g. "http://10.1.2.3:8090/metadata".
        timeout: Per-probe connection timeout in seconds.

    Raises:
        RuntimeError: If either service is unreachable.
    """
    # gRPC port check (TCP connect)
    try:
        grpc_host, grpc_port_str = master_server_address.rsplit(":", 1)
        grpc_port = int(grpc_port_str)
    except ValueError as exc:
        raise RuntimeError(f"Invalid master_server_address {master_server_address!r}") from exc

    def _probe(host, port, label, addr):
        # The master subprocess can take a few seconds to bind its RPC port
        # after the wrapper reports "started"; poll instead of one-shot so we
        # don't lose the race.
        deadline = time.monotonic() + 30.0
        while True:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Mooncake {label} unreachable at {addr}: {exc}") from exc
                time.sleep(1.0)

    _probe(grpc_host, grpc_port, "master gRPC", master_server_address)

    # HTTP metadata server check (TCP connect to parsed host:port)
    parsed = urlparse(metadata_server)
    http_host = parsed.hostname
    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if http_host is None:
        raise RuntimeError(f"Cannot parse host from metadata_server URL: {metadata_server!r}")

    _probe(http_host, http_port, "metadata server", metadata_server)

    logger.info(
        "Mooncake services reachable: master=%s, metadata=%s",
        master_server_address,
        metadata_server,
    )


def launch_mooncake_master(args):
    """Launch the mooncake master as a Ray actor.

    Auto-resolves master_server_address and metadata_port if not configured.
    When master_server_address specifies a host IP, pins the actor to that node so the
    mooncake master process starts on the intended machine.
    Writes resolved values back to args for downstream code.

    Args:
        args: Arguments namespace with mooncake_master_server_address, mooncake_metadata_port, etc.

    Returns:
        The MooncakeMasterActor handle, or None if binary not found.
    """
    from angelspec.ray.ray_actor import node_affinity_for_ip

    master_addr = getattr(args, "mooncake_master_server_address", None)
    scheduling_strategy = None

    if master_addr is None:
        host = RayActor.get_node_ip()
        port = RayActor.find_free_port(start_port=random.randint(51000, 52000))
        master_addr = f"{host}:{port}"
        args.mooncake_master_server_address = master_addr
        logger.info(f"Auto-resolved mooncake master_server_address: {master_addr}")
    else:
        if ":" in master_addr:
            host = master_addr.split(":")[0]
            port = int(master_addr.split(":")[1])
        else:
            host = master_addr
            port = getattr(args, "mooncake_master_port", 50051)

    # Pin actor to the user-specified node so the master starts on the right machine.
    scheduling_strategy = node_affinity_for_ip(host, name="mooncake_master")

    http_port = getattr(args, "mooncake_metadata_port", None) or getattr(
        args, "mooncake_http_port", None
    )
    if http_port is None:
        http_port = RayActor.find_free_port(start_port=random.randint(8100, 9100))
        args.mooncake_metadata_port = http_port
        logger.info(f"Auto-resolved mooncake metadata_port: {http_port}")
    http_host = getattr(args, "mooncake_http_host", "0.0.0.0")

    # Check binary existence before creating the actor
    mooncake_bin = resolve_mooncake_master_bin()
    if not os.path.exists(mooncake_bin):
        logger.warning(f"Binary not found at {mooncake_bin}, skipping launch")
        return None

    RemoteActor = ray.remote(num_cpus=0, runtime_env={"env_vars": get_angelspec_env_vars()})(
        MooncakeMaster
    )
    actor_options = {"name": "mooncake_master"}
    if scheduling_strategy is not None:
        actor_options["scheduling_strategy"] = scheduling_strategy
    actor = RemoteActor.options(**actor_options).remote()

    kv_lease_ttl_s = getattr(args, "mooncake_kv_lease_ttl_s", 5.0)

    try:
        info = ray.get(actor.start.remote(port, http_port, http_host, kv_lease_ttl_s))
        # Write back resolved values (actor may have updated host from node IP)
        args.mooncake_master_server_address = info["master_addr"]
        args.mooncake_metadata_port = info["metadata_port"]
        logger.info(f"mooncake master actor started: {info}")
    except Exception as e:
        logger.error(f"Failed to launch mooncake master actor: {e}")
        return None

    args._mooncake_master_actor = actor

    def _cleanup():
        try:
            ray.get(actor.shutdown.remote(), timeout=10)
        except Exception:
            pass

    atexit.register(_cleanup)

    return actor
