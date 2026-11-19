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

import fcntl
import gc
import os
import tempfile
import time
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from angelspec.utils.logging import logger

DTYPE_SIZES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.bool: 1,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "bool": 1,
}


def estimate_tensor_bytes(tensor_shapes: Dict[str, Tuple[int, ...]], tensor_dtypes: Dict) -> int:
    """Calculate bytes from tensor shapes and dtypes."""
    total = 0
    for name, shape in tensor_shapes.items():
        dtype = tensor_dtypes.get(name, torch.bfloat16)
        elem_size = DTYPE_SIZES.get(dtype, 2)
        numel = 1
        for dim in shape:
            numel *= dim
        total += numel * elem_size
    return total


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info


def free_pin_blocks(buddyinfo: str = "/proc/buddyinfo") -> int | None:
    """Free host memory in blocks of 4 MiB or more, counted in 4 MiB blocks."""
    try:
        with open(buddyinfo) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    blocks = 0
    for line in lines:
        counts = [int(count) for count in line.split()[4:]]
        blocks += sum(n << (order - 10) for order, n in enumerate(counts) if order >= 10)
    return blocks


def free_host_bytes(meminfo: str = "/proc/meminfo") -> int | None:
    """MemFree: host memory neither in use nor caching files."""
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemFree:"):
                    return int(line.split()[1]) << 10
    except OSError:
        pass
    return None


def ensure_host_pin_headroom(need_bytes: int, label: str, proc: str = "/proc") -> None:
    """Free and compact host memory until ``need_bytes`` of it lies in blocks of 4 MiB or more.

    The NVIDIA driver holds the page list of every 2 GiB it pins in one physically contiguous
    4 MiB block, and a long-running host whose page cache has fragmented its memory can have
    almost none free; a pin then fails as "CUDA error: invalid argument" (the kernel logs an
    order:10 page allocation failure) however much memory is free. Compaction only gathers free
    memory: when the page cache holds nearly all of it (a staged model is cached twice, the
    network original and the local copy), pins reclaim scattered pages instead, so the clean
    page cache is dropped first. The processes of a host take turns under a lock, so the first
    frees the memory and the others find it free.
    """
    buddyinfo = f"{proc}/buddyinfo"
    need_blocks = need_bytes >> 22
    before = free_pin_blocks(buddyinfo)
    if before is None or before >= need_blocks:
        return
    with open(
        os.path.join(tempfile.gettempdir(), "angelspec_host_pin_headroom.lock"), "w"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = free_pin_blocks(buddyinfo)
        if before is None or before >= need_blocks:
            return
        free = free_host_bytes(f"{proc}/meminfo")
        drop = free is not None and free < need_bytes
        start = time.monotonic()
        try:
            if drop:
                with open(f"{proc}/sys/vm/drop_caches", "w") as f:
                    f.write("1")
            with open(f"{proc}/sys/vm/compact_memory", "w") as f:
                f.write("1")
        except OSError as exc:
            logger.warning(
                "%s: %d free 4 MiB host blocks, too few to pin %d GiB, and freeing them failed "
                "(%s); run `echo 1 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory` "
                "as root",
                label,
                before,
                need_bytes >> 30,
                exc,
            )
            return
        logger.info(
            "%s: %scompacted host memory in %.0fs: free 4 MiB blocks %d -> %s",
            label,
            f"dropped the page cache ({free >> 30} GiB free) and " if drop else "",
            time.monotonic() - start,
            before,
            free_pin_blocks(buddyinfo),
        )
