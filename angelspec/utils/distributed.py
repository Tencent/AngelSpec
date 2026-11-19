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

import torch.distributed as dist

from angelspec.utils.logging import logger

GLOO_GROUP = None

_TP_DEVICE_MESH = None
_TP_GROUP = None
_USP_DEVICE_MESH = None
_USP_GRAD_SYNC_MESH = None
_USP_HSDP_DEVICE_MESH = None
_DRAFT_SP_GROUP = None
_DRAFT_SP_GLOO_GROUP = None
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


def get_usp_hsdp_device_mesh():
    global _USP_HSDP_DEVICE_MESH
    return _USP_HSDP_DEVICE_MESH


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


def _build_usp_hsdp_mesh_ranks(
    world_size: int,
    sp_size: int,
    hsdp_shard_size: int,
) -> list[list[int]]:
    if sp_size <= 0 or world_size % sp_size != 0:
        raise ValueError(f"Invalid USP topology: world_size={world_size}, sp_size={sp_size}")
    dp_size = world_size // sp_size
    if hsdp_shard_size <= 0 or world_size % hsdp_shard_size != 0:
        raise ValueError(
            f"hsdp_shard_size ({hsdp_shard_size}) must divide world_size ({world_size})"
        )

    # Global rank layout is dp-major, sp-minor: rank = dp_idx * sp_size + sp_idx.
    if hsdp_shard_size <= dp_size:
        # Shard params over a DP sub-block; SP stays a pure replicate axis. The
        # trainer's mul_(sp_size) turns HSDP's SP-average back into the SP-sum USP
        # needs. (Original behavior — unchanged.)
        if dp_size % hsdp_shard_size != 0:
            raise ValueError(
                f"hsdp_shard_size ({hsdp_shard_size}) must divide dp_size ({dp_size})"
            )
        dp_replicate_size = dp_size // hsdp_shard_size
        return [
            [
                (dp_replica_rank * hsdp_shard_size + dp_shard_rank) * sp_size + sp_rank
                for dp_shard_rank in range(hsdp_shard_size)
            ]
            for dp_replica_rank in range(dp_replicate_size)
            for sp_rank in range(sp_size)
        ]

    # hsdp_shard_size > dp_size: shard spans the whole DP dim plus a contiguous SP
    # block, so params/optimizer are distributed across SP too (long-context memory
    # relief). Correct because HSDP reduces every param shard to the mesh-wide grad
    # MEAN and the trainer's mul_(sp_size) converts the SP portion back to a SUM —
    # invariant to how the ranks split between the replicate and shard axes.
    if hsdp_shard_size % dp_size != 0:
        raise ValueError(
            f"hsdp_shard_size ({hsdp_shard_size}) exceeding dp_size ({dp_size}) "
            "must be a multiple of dp_size"
        )
    sp_shard = hsdp_shard_size // dp_size
    if sp_size % sp_shard != 0:
        raise ValueError(
            f"hsdp_shard_size ({hsdp_shard_size}) implies sp_shard={sp_shard}, "
            f"which must divide sp_size ({sp_size})"
        )
    sp_replicate = sp_size // sp_shard
    return [
        [
            dp_idx * sp_size + (sp_replica_rank * sp_shard + sp_shard_rank)
            for dp_idx in range(dp_size)
            for sp_shard_rank in range(sp_shard)
        ]
        for sp_replica_rank in range(sp_replicate)
    ]


def _resolve_usp_hsdp_shard_size(
    dp_size: int,
    sp_size: int,
    hsdp_shard_size: int | None,
    gpus_per_node: int | None = None,
) -> int:
    if hsdp_shard_size is not None:
        return int(hsdp_shard_size)
    if gpus_per_node is None:
        return dp_size
    gpus_per_node = int(gpus_per_node)
    if gpus_per_node % sp_size != 0:
        return 1
    return min(dp_size, gpus_per_node // sp_size)


def resolve_hsdp_shard_size(
    dp_size: int,
    sp_size: int,
    *,
    hsdp_shard_dp: int | None = None,
    hsdp_shard_sp: int | None = None,
    hsdp_shard_size: int | None = None,
    gpus_per_node: int | None = None,
) -> int:
    """Resolve the FSDP shard-dim size from the explicit DP/SP knobs.

    Preferred inputs are ``hsdp_shard_dp`` (how many DP ranks to shard params
    across, default ``dp_size``) and ``hsdp_shard_sp`` (how many SP ranks to fold
    into the shard dim, default 1 = pure DP with SP replicated); the effective
    shard size is their product. Folding SP (``hsdp_shard_sp`` > 1) requires the
    full DP dim to be sharded first (``hsdp_shard_dp`` == ``dp_size``), matching
    the DP-first layout in :func:`_build_usp_hsdp_mesh_ranks`.

    When neither DP/SP knob is set, falls back to the legacy ``hsdp_shard_size``
    int, then to the conservative per-node default.
    """
    if hsdp_shard_dp is not None or hsdp_shard_sp is not None:
        dp_ways = int(hsdp_shard_dp) if hsdp_shard_dp is not None else dp_size
        sp_ways = int(hsdp_shard_sp) if hsdp_shard_sp is not None else 1
        if dp_ways <= 0 or dp_size % dp_ways != 0:
            raise ValueError(f"hsdp_shard_dp ({dp_ways}) must divide dp_size ({dp_size})")
        if sp_ways <= 0 or sp_size % sp_ways != 0:
            raise ValueError(f"hsdp_shard_sp ({sp_ways}) must divide sp_size ({sp_size})")
        if sp_ways > 1 and dp_ways != dp_size:
            raise ValueError(
                f"hsdp_shard_sp ({sp_ways}) > 1 requires the full DP dim sharded first: "
                f"set hsdp_shard_dp = dp_size ({dp_size}), got {dp_ways}"
            )
        return dp_ways * sp_ways
    return _resolve_usp_hsdp_shard_size(dp_size, sp_size, hsdp_shard_size, gpus_per_node)


def init_usp_hsdp_mesh(hsdp_shard_size: int | None = None):
    global _USP_HSDP_DEVICE_MESH

    usp_mesh = get_usp_device_mesh()
    if usp_mesh is None:
        raise RuntimeError("USP mesh must be initialized before the HSDP mesh")

    world_size = dist.get_world_size()
    sp_size = usp_mesh.shape[1]
    dp_size = usp_mesh.shape[0]
    shard_size = _resolve_usp_hsdp_shard_size(
        dp_size,
        sp_size,
        hsdp_shard_size,
        int(os.environ.get("LOCAL_WORLD_SIZE", dp_size)),
    )
    mesh_ranks = _build_usp_hsdp_mesh_ranks(world_size, sp_size, shard_size)

    if shard_size <= dp_size:
        dp_ways, sp_ways = shard_size, 1
    else:
        dp_ways, sp_ways = dp_size, shard_size // dp_size
    logger.info(
        "HSDP param sharding: shard_size=%d over DP(%d)x SP(%d), replicate=%d, "
        "per-rank param/optimizer fraction=1/%d (dp_size=%d, sp_size=%d)",
        shard_size,
        dp_ways,
        sp_ways,
        world_size // shard_size,
        shard_size,
        dp_size,
        sp_size,
    )

    # Use fresh process groups instead of the USP groups. HSDP's asynchronous
    # gradient collectives may overlap Ulysses all-to-all in backward; sharing a
    # communicator lets their issue order race and can deadlock.
    if world_size == shard_size:
        _USP_HSDP_DEVICE_MESH = dist.device_mesh.DeviceMesh(
            "cuda",
            mesh_ranks[0],
            mesh_dim_names=("hsdp_shard",),
        )
    else:
        _USP_HSDP_DEVICE_MESH = dist.device_mesh.DeviceMesh(
            "cuda",
            mesh_ranks,
            mesh_dim_names=("hsdp_replicate", "hsdp_shard"),
        )
    return _USP_HSDP_DEVICE_MESH


def init_usp_groups(sp_ulysses_size: int = 1, sp_ring_size: int = 1):
    global _USP_DEVICE_MESH
    global _USP_GRAD_SYNC_MESH
    global _USP_HSDP_DEVICE_MESH
    global _DRAFT_SP_GROUP
    global _DRAFT_SP_GLOO_GROUP
    global _SP_ULYSSES_GROUP, _SP_RING_GROUP

    sp_size = sp_ulysses_size * sp_ring_size
    if sp_size == 1:
        _USP_DEVICE_MESH = None
        _USP_GRAD_SYNC_MESH = None
        _USP_HSDP_DEVICE_MESH = None
        _DRAFT_SP_GROUP = None
        _DRAFT_SP_GLOO_GROUP = None
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
    _DRAFT_SP_GLOO_GROUP = None
    _SP_ULYSSES_GROUP = None
    _SP_RING_GROUP = None
    _USP_HSDP_DEVICE_MESH = None

    _USP_DEVICE_MESH = dist.device_mesh.init_device_mesh(
        "cuda",
        (draft_dp_size, sp_size),
        mesh_dim_names=("draft_dp", "draft_sp"),
    )
    _DRAFT_SP_GROUP = _USP_DEVICE_MESH.get_group("draft_sp")
    draft_sp_groups, _, _ = _build_usp_group_ranks(
        world_size,
        sp_ulysses_size,
        sp_ring_size,
    )
    for ranks in draft_sp_groups:
        group = dist.new_group(ranks=ranks, backend="gloo")
        if rank in ranks:
            _DRAFT_SP_GLOO_GROUP = group
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


def warm_up_draft_sp_connections(bytes_per_peer: int = 16 << 20) -> None:
    """Connect the draft SP group's NCCL peers before any compute runs.

    NCCL connects all-to-all peers lazily on first use. If that first use overlaps
    a rank's first load of a cuBLAS kernel (cuLibraryLoadData waits for the
    context's running kernels), the rank's spinning all-to-all kernel and its
    peers' connection handshakes wait on each other forever.
    """
    import torch

    group = _DRAFT_SP_GROUP
    if group is None:
        return
    size = dist.get_world_size(group)
    chunk = bytes_per_peer // 2
    send = torch.zeros(chunk * size, dtype=torch.bfloat16, device="cuda")
    dist.all_to_all_single(torch.empty_like(send), send, group=group)
    dist.all_gather_into_tensor(torch.empty_like(send), send[:chunk], group=group)
    torch.cuda.synchronize()
    del send
    torch.cuda.empty_cache()


def get_draft_sp_gloo_group():
    global _DRAFT_SP_GLOO_GROUP
    return _DRAFT_SP_GLOO_GROUP


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
