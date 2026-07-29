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

import torch.distributed as dist

GLOO_GROUP = None

_TP_DEVICE_MESH = None
_TP_GROUP = None
_USP_DEVICE_MESH = None
_USP_GRAD_SYNC_MESH = None
_DRAFT_SP_GROUP = None
_SP_ULYSSES_GROUP = None
_SP_RING_GROUP = None


def init_gloo_group():
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo")
    return GLOO_GROUP


def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    return GLOO_GROUP


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_tp_device_mesh():
    global _TP_DEVICE_MESH
    return _TP_DEVICE_MESH


def get_usp_device_mesh():
    global _USP_DEVICE_MESH
    return _USP_DEVICE_MESH


def get_usp_grad_sync_mesh():
    global _USP_GRAD_SYNC_MESH
    return _USP_GRAD_SYNC_MESH


def _build_usp_group_ranks(
    world_size: int, sp_ulysses_size: int, sp_ring_size: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    sp_size = sp_ulysses_size * sp_ring_size
    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")
    if world_size % sp_size != 0:
        raise ValueError(
            "world_size must be divisible by sp_ulysses_size * sp_ring_size, "
            f"got world_size={world_size}, sp_ulysses_size={sp_ulysses_size}, "
            f"sp_ring_size={sp_ring_size}"
        )

    draft_sp_groups: list[list[int]] = []
    ulysses_groups: list[list[int]] = []
    ring_groups: list[list[int]] = []
    num_ulysses_pgs = sp_ring_size
    num_ring_pgs = sp_ulysses_size
    for base_rank in range(0, world_size, sp_size):
        draft_sp_groups.append(list(range(base_rank, base_rank + sp_size)))
        for idx in range(num_ulysses_pgs):
            ulysses_groups.append(
                list(
                    range(
                        base_rank + idx * sp_ulysses_size,
                        base_rank + (idx + 1) * sp_ulysses_size,
                    )
                )
            )
        for idx in range(num_ring_pgs):
            ring_groups.append(list(range(base_rank + idx, base_rank + sp_size, num_ring_pgs)))
    return draft_sp_groups, ulysses_groups, ring_groups


def init_usp_groups(sp_ulysses_size: int = 1, sp_ring_size: int = 1):
    global _USP_DEVICE_MESH
    global _USP_GRAD_SYNC_MESH
    global _DRAFT_SP_GROUP
    global _SP_ULYSSES_GROUP, _SP_RING_GROUP

    sp_size = sp_ulysses_size * sp_ring_size
    if sp_size == 1:
        _USP_DEVICE_MESH = None
        _USP_GRAD_SYNC_MESH = None
        _DRAFT_SP_GROUP = None
        _SP_ULYSSES_GROUP = None
        _SP_RING_GROUP = None
        return None, None, None

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % sp_size != 0:
        raise ValueError(
            "world_size must be divisible by sp_ulysses_size * sp_ring_size, "
            f"got world_size={world_size}, sp_ulysses_size={sp_ulysses_size}, "
            f"sp_ring_size={sp_ring_size}"
        )

    draft_dp_size = world_size // sp_size

    _DRAFT_SP_GROUP = None
    _SP_ULYSSES_GROUP = None
    _SP_RING_GROUP = None

    _USP_DEVICE_MESH = dist.device_mesh.init_device_mesh(
        "cuda",
        (draft_dp_size, sp_size),
        mesh_dim_names=("draft_dp", "draft_sp"),
    )
    _DRAFT_SP_GROUP = _USP_DEVICE_MESH.get_group("draft_sp")
    _USP_GRAD_SYNC_MESH = dist.device_mesh.init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("draft_dp_with_sp",),
    )

    if sp_ring_size == 1:
        # Pure-Ulysses path (torch-native, no yunchang dependency). The whole
        # draft_sp group IS the Ulysses group; the ring group degenerates to a
        # single rank.
        _SP_ULYSSES_GROUP = _DRAFT_SP_GROUP
        _SP_RING_GROUP = None
    else:
        # Ring>1 relies on yunchang's 2D (ulysses × ring) decomposition.
        import yunchang
        from yunchang.globals import PROCESS_GROUP as YUNCHANG_PROCESS_GROUP

        yunchang.set_seq_parallel_pg(
            sp_ulysses_degree=sp_ulysses_size,
            sp_ring_degree=sp_ring_size,
            rank=rank,
            world_size=world_size,
            use_ulysses_low=True,
        )
        _SP_ULYSSES_GROUP = YUNCHANG_PROCESS_GROUP.ULYSSES_PG
        _SP_RING_GROUP = YUNCHANG_PROCESS_GROUP.RING_PG
        _validate_usp_group_composition()

    return _DRAFT_SP_GROUP, _SP_ULYSSES_GROUP, _SP_RING_GROUP


def get_draft_sp_group():
    global _DRAFT_SP_GROUP
    return _DRAFT_SP_GROUP


def get_sp_ulysses_group():
    global _SP_ULYSSES_GROUP
    return _SP_ULYSSES_GROUP


def get_sp_ring_group():
    global _SP_RING_GROUP
    return _SP_RING_GROUP


def get_sp_rank() -> int:
    sp_group = get_draft_sp_group()
    if sp_group is None:
        return 0
    return dist.get_rank(sp_group)


def get_sp_ulysses_rank() -> int:
    ulysses_group = get_sp_ulysses_group()
    if ulysses_group is None:
        return 0
    return dist.get_rank(ulysses_group)


def get_sp_ring_rank() -> int:
    ring_group = get_sp_ring_group()
    if ring_group is None:
        return 0
    return dist.get_rank(ring_group)


def get_usp_rank_coords(sp_rank: int, sp_ulysses_size: int, sp_ring_size: int) -> tuple[int, int]:
    sp_size = sp_ulysses_size * sp_ring_size
    if sp_rank < 0 or sp_rank >= sp_size:
        raise ValueError(f"sp_rank must be in [0, {sp_size}), got {sp_rank}")
    ulysses_rank = sp_rank % sp_ulysses_size
    ring_rank = sp_rank // sp_ulysses_size
    return ulysses_rank, ring_rank


def _gather_group_members(group) -> tuple[int, ...]:
    group_world_size = dist.get_world_size(group)
    members = [None] * group_world_size
    dist.all_gather_object(members, dist.get_rank(), group=group)
    return tuple(members)


def _validate_usp_group_composition() -> None:
    sp_group = get_draft_sp_group()
    ulysses_group = get_sp_ulysses_group()
    ring_group = get_sp_ring_group()
    if sp_group is None or ulysses_group is None or ring_group is None:
        raise RuntimeError("USP groups must be initialized before validating group composition")

    sp_members = _gather_group_members(sp_group)
    local_record = {
        "world_rank": dist.get_rank(),
        "ring_members": _gather_group_members(ring_group),
        "ulysses_members": _gather_group_members(ulysses_group),
    }
    records = [None] * dist.get_world_size(sp_group)
    dist.all_gather_object(records, local_record, group=sp_group)
    for record in records:
        ring_members = tuple(record["ring_members"])
        ulysses_members = tuple(record["ulysses_members"])
        if any(member not in sp_members for member in ring_members + ulysses_members):
            raise RuntimeError("USP ring/ulysses groups include ranks outside the draft SP group")
