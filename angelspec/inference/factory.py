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

"""Inference engine creation and initialization with Ray placement groups."""

import json

import ray
from omegaconf import DictConfig, OmegaConf
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from angelspec.utils.env import get_angelspec_env_vars
from angelspec.utils.logging import logger

# Multi-node TP worker engines must stay alive to participate in NCCL
# operations but are never called directly. Store refs here to prevent GC.
_alive_worker_engines: list = []

# vLLM's get_open_port() binds an ephemeral port (32768+), closes it, then
# rebinds later in create_tcp_store — a TOCTOU window on the OS's busiest port
# range where co-located engines or crash leftovers collide (EADDRINUSE on the
# internal TCPStore). Pinning VLLM_PORT makes vLLM scan upward from a quiet,
# deterministic base instead. Bases are disjoint per engine_group; within a
# group the per-engine offset is taken modulo LOCAL_SLOTS (ports are per-node,
# so only same-node engines must differ) — this stays bounded no matter how
# many GPUs/engines a run spans.
#
# The whole window lives above the training master range (…-21000) and mooncake
# and below the OS ephemeral range (/proc/sys/net/ipv4/ip_local_port_range =
# 32768-60999 here). Budget: (CEIL - BASE) / GROUP_SPAN group slots. With the
# values below that is (32700-21500)/3200 = 3 groups — exactly the ones we run
# (inference=0, eval=1, score=2) — and each engine gets STRIDE headroom for the
# few ports vLLM opens above VLLM_PORT. STRIDE and LOCAL_SLOTS are both 2x the
# original so co-located engines have more room; the assert guarantees we can
# never silently spill into the ephemeral range if either grows.
_VLLM_PORT_GROUP_BASE = 21500
_VLLM_PORT_CEIL = 32700  # exclusive upper bound; just under ephemeral (32768)
_VLLM_PORT_LOCAL_SLOTS = 32  # max engines per node we bother to segregate
_VLLM_PORT_STRIDE = 100  # per-engine headroom for vLLM's upward port scan
_VLLM_PORT_GROUP_SPAN = _VLLM_PORT_LOCAL_SLOTS * _VLLM_PORT_STRIDE  # 3200


def _vllm_config_env_vars(args) -> dict[str, str]:
    """``inference.vllm.env_vars`` as a plain dict, for the engine actor's runtime_env.

    VllmEngine._apply_env_vars() already puts these in the engine process, but that
    is too late for vLLM's own ray workers: under the ray executor backend they are
    separate actors that inherit the raylet's environment, and
    ray_executor_v2.initialize_worker() replays the engine's values with
    os.environ.setdefault() — so a stale value baked into the raylet (e.g. an
    NCCL_SOCKET_IFNAME for a different cluster, which NCCL then reports as
    "Bootstrap : no socket interface found") can never be corrected. Ray's
    runtime_env does override the inherited environment, and vLLM seeds its worker
    runtime_env from the engine actor's own (parallel_config.ray_runtime_env), so
    putting them here is what actually reaches the workers.
    """
    raw = getattr(args, "vllm_env_vars", None)
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    elif isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def _vllm_port(engine_group: int, index: int) -> int:
    port = (
        _VLLM_PORT_GROUP_BASE
        + engine_group * _VLLM_PORT_GROUP_SPAN
        + (index % _VLLM_PORT_LOCAL_SLOTS) * _VLLM_PORT_STRIDE
    )
    assert _VLLM_PORT_GROUP_BASE <= port < _VLLM_PORT_CEIL, (
        f"vLLM port {port} (engine_group={engine_group}, index={index}) escaped the "
        f"safe window [{_VLLM_PORT_GROUP_BASE}, {_VLLM_PORT_CEIL}); shrink STRIDE/"
        f"LOCAL_SLOTS or lower the ephemeral floor before adding more engine groups."
    )
    return port


def prepare_inference_engines(args, inference_pg, mooncake_config, engine_group: int = 0):
    """Create inference engines and fire init calls without waiting.

    Use this to parallelize engine initialization with other setup work
    (e.g., training actor initialization). Call ray.get() on the returned
    init_refs before using the engines.

    Returns:
        Tuple of (head_engines, init_refs) where head_engines are the engines
        for dispatching requests, and init_refs are ObjectRefs to wait on.
    """
    engine_type = getattr(args, "inference_engine_type", "hf")

    if engine_type not in ("hf", "sgl", "vllm"):
        raise ValueError(f"Unknown inference_engine_type: {engine_type}")

    logger.info(f"Preparing {engine_type} inference engines...")

    if engine_type == "hf":
        engines, init_refs = _prepare_hf_engines(args, inference_pg, mooncake_config, engine_group)
    elif engine_type == "sgl":
        engines, init_refs = _prepare_sgl_engines(
            args, inference_pg, mooncake_config, engine_group
        )
    else:
        engines, init_refs = _prepare_vllm_engines(
            args, inference_pg, mooncake_config, engine_group
        )

    return engines, init_refs


def prepare_eval_engine(args, eval_pg, engine_group: int = 1):
    """Create the persistent online-eval engine actor and fire its init.

    Mirrors the vLLM engine's PG-scheduled actor pattern but builds an EvalEngine
    (real spec-decode, offline metrics — see inference/engine/eval_engine.py). One
    actor, ``online_eval_num_gpus_per_engine`` GPUs, pinned to the ``eval`` role's
    bundle. Returns (actor, init_ref) or (None, None) if not configured.
    """
    if eval_pg is None:
        return None, None
    pg_obj, reordered_bundle_indices, reordered_gpu_ids = eval_pg
    if not reordered_gpu_ids:
        return None, None

    from angelspec.inference.engine.eval_engine import EvalEngine

    gpus_per_engine = int(getattr(args, "online_eval_num_gpus_per_engine", 1) or 1)
    base_gpu_id = int(reordered_gpu_ids[0])
    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg_obj,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=reordered_bundle_indices[0],
    )
    EvalRayActor = ray.remote(EvalEngine)
    actor = EvalRayActor.options(
        num_cpus=0.2,
        num_gpus=0.2,
        scheduling_strategy=scheduling_strategy,
        runtime_env={
            "env_vars": {
                **get_angelspec_env_vars(),
                "VLLM_PORT": str(_vllm_port(engine_group, 0)),
            }
        },
    ).remote(
        args=args,
        rank=0,
        base_gpu_id=base_gpu_id,
        num_gpus_per_engine=gpus_per_engine,
        node_rank=0,
        engine_group=engine_group,
    )
    logger.info(
        "Prepared online-eval engine actor (base_gpu_id=%d, gpus=%d)",
        base_gpu_id,
        gpus_per_engine,
    )
    return actor, actor.init.remote()


def prepare_score_engine(args, score_pg, engine_group: int = 2):
    """Create OPD scoring actors and return their initialization references.

    MTP draft OPD and DFlash branch OPD use one TP decoder per training DP rank.
    Packed-tree OPD uses TP1 encoder engines with FlexAttention and no KV cache.
    Both return scores through RPC without a Mooncake connector.
    """
    if score_pg is None or not getattr(args, "enable_opd_score_engine", False):
        return [], []
    if getattr(args, "mtp_draft_opd", False) or (
        getattr(args, "dflash_opd_enabled", False)
        and getattr(args, "dflash_opd_backend", "packed_tree") == "branch_requests"
    ):
        import copy

        verify_args = copy.copy(args)
        verify_args.mtp_opd_verify_engine = bool(getattr(args, "mtp_draft_opd", False))
        verify_args.dfly_opd_verify_engine = not verify_args.mtp_opd_verify_engine
        verify_args.inference_num_gpus = args.opd_score_engine_num_gpus
        return _prepare_vllm_engines(verify_args, score_pg, engine_group=engine_group)
    pg_obj, reordered_bundle_indices, reordered_gpu_ids = score_pg
    if not reordered_gpu_ids:
        return [], []

    import copy as _copy

    from angelspec.inference.engine.vllm_engine import VllmEngine

    ScoreRayActor = ray.remote(VllmEngine)
    engines, init_refs = [], []
    for i, (gpu_id, bundle_idx) in enumerate(zip(reordered_gpu_ids, reordered_bundle_indices)):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_idx,
        )
        score_args = _copy.copy(args)
        score_args.vllm_score_engine = True
        actor = ScoreRayActor.options(
            num_cpus=0.2,
            num_gpus=0.2,
            scheduling_strategy=scheduling_strategy,
            runtime_env={
                "env_vars": {
                    **get_angelspec_env_vars(),
                    "VLLM_PORT": str(_vllm_port(engine_group, i)),
                }
            },
        ).remote(
            args=score_args,
            rank=i,
            base_gpu_id=int(gpu_id),
            num_gpus_per_engine=1,
            node_rank=0,
            engine_group=engine_group,
        )
        engines.append(actor)
        init_refs.append(actor.init.remote())
    logger.info(
        "Prepared %d OPD score engine(s) (gpus=%s, TP=1 each, encoder-only)",
        len(engines),
        list(reordered_gpu_ids),
    )
    return engines, init_refs


def _prepare_hf_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create HF engine actors and fire init calls without waiting.

    Returns:
        Tuple of (engines, init_handles).
    """
    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    num_gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    num_gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)

    num_gpus_per_engine = min(num_gpus_per_engine, num_gpus_per_node)
    num_engines = num_gpus_total // num_gpus_per_engine

    logger.info(f"Initializing {num_engines} HF engines ({num_gpus_per_engine} GPU(s) each)")

    from angelspec.inference.engine.hf_engine import HFEngine

    HFRayActor = ray.remote(HFEngine)
    return _create_and_init_actors(
        args,
        pg,
        num_engines,
        num_gpus_per_engine,
        HFRayActor,
        mooncake_config,
        engine_group=engine_group,
    )


def _init_hf_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize HF engines with Ray placement groups."""
    engines, init_handles = _prepare_hf_engines(args, pg, mooncake_config, engine_group)
    _wait_for_init(init_handles, "HF", timeout=1200)
    return engines


def _prepare_sgl_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create SGL engine actors and fire init calls without waiting.

    Handles three cases:
      - Single-node, multiple engines: one engine per group of GPUs
      - Multi-node, single replica: one engine per node, all forming one TP group
      - Multi-node, multiple replicas: N independent TP groups, each spanning nnodes

    For multi-node, worker engines are stored in a module-level list to prevent GC.

    Returns:
        Tuple of (head_engines, init_handles). head_engines are the engines that
        accept generate() calls. init_handles are ObjectRefs for ALL engines
        (head + worker) that must be waited on before use.
    """

    nnodes = getattr(args, "sglang_nnodes", 1)
    num_gpus_total = getattr(args, "inference_num_gpus", 1)

    if nnodes > 1:
        gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)
        gpus_per_replica = nnodes * gpus_per_node
        num_replicas = num_gpus_total // gpus_per_replica
        num_engines = num_replicas * nnodes
        gpus_per_engine = gpus_per_node
    else:
        gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
        num_replicas = num_gpus_total // gpus_per_engine
        num_engines = num_replicas

    logger.info(
        f"Initializing {num_engines} Sgl engines "
        f"({gpus_per_engine} GPU(s) each, nnodes={nnodes}, replicas={num_replicas})"
    )

    from angelspec.inference.engine.sgl_engine import SglEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    SglRayActor = ray.remote(SglEngine)
    env_vars = get_angelspec_env_vars()

    # Step 1: Create all engine actors (without calling init yet)
    engines = []
    for i in range(num_engines):
        node_rank = i % nnodes if nnodes > 1 else 0

        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        engine = SglRayActor.options(
            num_cpus=0.2,
            num_gpus=0.2,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=node_rank,
            engine_group=engine_group,
        )
        engines.append(engine)

    # Step 2: Pre-allocate ports to avoid TOCTOU races between parallel engines.
    # Each engine needs 2 consecutive ports; allocate sequentially so engines
    # on the same node never collide.
    pre_allocated_ports: dict[int, int] = {}
    next_start = 10000
    for i in range(num_engines):
        port = ray.get(
            engines[i].find_free_port.remote(start_port=next_start, consecutive=2),
            timeout=120,
        )
        pre_allocated_ports[i] = port
        next_start = port + 2
        logger.info(f"Engine {i}: pre-allocated ports {port}, {port + 1}")

    # Step 2.5: Auto-negotiate dist_init_addr for each multi-node replica, drawing
    # from next_start (past every engine's pre-allocated ports). find_free_port only
    # probes — it does not reserve — so negotiating from a fresh 10000 would hand the
    # head node the very port it just got in the pre-allocation pass, colliding the
    # server port with the dist-init TCPStore. Uses RayActor.get_node_ip().
    dist_init_addrs: dict[int, str] = {}
    if nnodes > 1:
        configured_addr = getattr(args, "sglang_dist_init_addr", None)
        for replica_idx in range(num_replicas):
            if configured_addr and num_replicas == 1:
                dist_init_addrs[replica_idx] = configured_addr
                logger.info(
                    f"Replica {replica_idx}: using configured dist_init_addr: {configured_addr}"
                )
            else:
                head_engine = engines[replica_idx * nnodes]
                ip, port = ray.get(
                    [
                        head_engine.get_node_ip.remote(),
                        head_engine.find_free_port.remote(start_port=next_start),
                    ],
                    timeout=120,
                )
                addr = f"{ip}:{port}"
                dist_init_addrs[replica_idx] = addr
                next_start = port + 1
                logger.info(f"Replica {replica_idx}: auto-negotiated dist_init_addr: {addr}")

    # Step 3: Fire init() on all engines (non-blocking)
    init_handles = []
    for i, engine in enumerate(engines):
        replica_idx = i // nnodes if nnodes > 1 else i
        init_handles.append(
            engine.init.remote(
                mooncake_config=mooncake_config,
                dist_init_addr=dist_init_addrs.get(replica_idx),
                pre_allocated_port=pre_allocated_ports.get(i),
            )
        )

    # Separate head/worker engines for multi-node
    if nnodes > 1:
        head_engines = [engines[i] for i in range(num_engines) if i % nnodes == 0]
        worker_engines = [engines[i] for i in range(num_engines) if i % nnodes != 0]
        # Keep worker engines alive (they participate in NCCL TP) via module-level ref
        _alive_worker_engines.extend(worker_engines)
        logger.info(
            f"Prepared multi-node Sgl engines: {len(head_engines)} heads + "
            f"{len(worker_engines)} workers ({num_replicas} replicas)"
        )
        return head_engines, init_handles

    return engines, init_handles


def _init_sgl_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize SGLang engines with Ray placement groups (blocking)."""
    head_engines, init_handles = _prepare_sgl_engines(args, pg, mooncake_config, engine_group)
    nnodes = getattr(args, "sglang_nnodes", 1)
    init_timeout = getattr(args, "sglang_init_timeout", 300 if nnodes == 1 else 600)
    _wait_for_init(init_handles, "Sgl", timeout=init_timeout)
    return head_engines


def _prepare_vllm_dp_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create vLLM data-parallel worlds for extract_hidden_states.

    Each world is dp_size*tp_size actors, ONE GPU each, forming its own
    torch.distributed world via vLLM's external_launcher (SPMD): experts are
    EP-sharded across the world, each DP group has its own KV pool. The inference
    GPUs hold inference_num_gpus // (dp_size*tp_size) such worlds. Ray pins one GPU
    per actor (visible as cuda:0, LOCAL_RANK=0), each world's rank0 hosts its
    rendezvous. Returns ``(head_engines, init_handles)`` where heads are the tp_rank0
    actor of each DP group, world by world (dp_size heads a world); non-head actors
    are kept alive for the SPMD worlds (the controller must still drive generate()
    on every head of a world at once — see AsyncInferenceManager).
    """
    from angelspec.inference.engine.vllm_engine import VllmEngine

    dp_size = int(getattr(args, "vllm_data_parallel_size", 1))
    tp_size = int(
        getattr(args, "vllm_tp_size", None) or getattr(args, "inference_num_gpus_per_engine", 1)
    )
    gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)
    world = dp_size * tp_size
    num_gpus_total = getattr(args, "inference_num_gpus", world)
    if world > num_gpus_total or num_gpus_total % world:
        raise ValueError(
            f"inference_num_gpus ({num_gpus_total}) is not a multiple of dp_size*tp_size ({world})"
        )
    # Independent worlds, each with its own rendezvous and its own experts: a world on one
    # node keeps its MoE all-to-alls on NVLink.
    num_worlds = num_gpus_total // world

    logger.info(
        f"Initializing {num_worlds} vLLM DP world(s): dp={dp_size} tp={tp_size} world={world} "
        f"(1 GPU/actor, external_launcher, EP across each world's ranks)"
    )

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg
    VllmRayActor = ray.remote(VllmEngine)
    env_vars = {**get_angelspec_env_vars(), **_vllm_config_env_vars(args)}

    worlds = []
    for w in range(num_worlds):
        engines = []
        for r in range(world):
            g = w * world + r
            dp_rank = r // tp_size
            tp_rank = r % tp_size
            node_rank = g // gpus_per_node
            # LOCAL_RANK is the node-local physical device index this rank owns, taken
            # from the placement group's inference gpu ids (e.g. [4,5,6,7] under
            # training_first) — NOT g % gpus_per_node, which would land on the training
            # bundles [0..3] and collide with the trainers (OOM).
            local_rank = int(reordered_gpu_ids[g])
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg_obj,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[g],
            )
            engine = VllmRayActor.options(
                num_cpus=0.2,
                num_gpus=1,  # Ray pins one physical GPU per rank -> cuda:0 inside
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": {**env_vars, "VLLM_PORT": str(_vllm_port(engine_group, g))}
                },
            ).remote(
                args=args,
                rank=g,
                base_gpu_id=None,  # Ray already bound the GPU; skip manual setup_gpu
                num_gpus_per_engine=1,
                node_rank=node_rank,
                engine_group=engine_group,
                dp_size=dp_size,
                tp_size=tp_size,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                world_size=world,
                world_rank=r,
                local_rank=local_rank,
            )
            engines.append(engine)
        worlds.append(engines)

    init_handles = []
    head_engines = []
    for w, engines in enumerate(worlds):
        # Each world's rank0 hosts its rendezvous; its ranks get that MASTER via dist_init_addr.
        master_port = ray.get(
            engines[0].find_free_port.remote(start_port=12000, consecutive=1), timeout=300
        )
        master_ip = ray.get(engines[0].get_node_ip.remote(), timeout=30)
        master_addr = f"{master_ip}:{master_port}"
        logger.info(f"vLLM DP world {w} master (rank0) = {master_addr}")
        init_handles += [
            engine.init.remote(mooncake_config=mooncake_config, dist_init_addr=master_addr)
            for engine in engines
        ]
        head_engines += [engines[r] for r in range(world) if r % tp_size == 0]
        worker_engines = [engines[r] for r in range(world) if r % tp_size != 0]
        _alive_worker_engines.extend(worker_engines)
        # Option B: register each DP-group head's tp-workers so the head drives them
        # in SPMD lockstep. Queued after init so it runs once the actors are up; the
        # caller awaits init_handles before any generate.
        for d in range(dp_size):
            group_workers = engines[d * tp_size + 1 : (d + 1) * tp_size]
            if group_workers:
                init_handles.append(engines[d * tp_size].set_tp_group.remote(group_workers))
    logger.info(
        f"Prepared {num_worlds} vLLM DP world(s): {len(head_engines)} DP-head(s) in world "
        f"order, {dp_size} a world (dp={dp_size}, tp={tp_size})"
    )
    return head_engines, init_handles


def _prepare_vllm_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create vLLM engine actors and fire init calls without waiting.

    Handles three cases:
      - Single-node, multiple engines: one engine per group of GPUs
      - Multi-node, single replica: one engine per node, all forming one TP group
      - Multi-node, multiple replicas: N independent TP groups, each spanning nnodes

    For multi-node, worker engines are stored in a module-level list to prevent GC.

    Returns:
        Tuple of (head_engines, init_handles). head_engines are the engines that
        accept generate() calls. init_handles are ObjectRefs for ALL engines
        (head + worker) that must be waited on before use.
    """
    nnodes = getattr(args, "vllm_nnodes", 1)
    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    dp_size = getattr(args, "vllm_data_parallel_size", 1)
    tp_size = int(
        getattr(args, "vllm_tp_size", None) or getattr(args, "inference_num_gpus_per_engine", 1)
    )
    pp_size = int(getattr(args, "vllm_pp_size", 1) or 1)
    if dp_size > 1:
        if pp_size > 1:
            raise NotImplementedError("vLLM data_parallel_size>1 with pp_size>1 is not supported")
        return _prepare_vllm_dp_engines(args, pg, mooncake_config, engine_group)
    backend = getattr(args, "vllm_distributed_executor_backend", None) or (
        "ray" if nnodes > 1 else "mp"
    )
    # Cross-node via ray: ONE EngineCore actor per replica; vLLM's
    # RayDistributedExecutor spawns all workers across the nodes itself (no
    # per-node follower LLM, which is what asserts on collective_rpc).
    xnode_ray = nnodes > 1 and backend == "ray"

    if nnodes > 1:
        gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)
        gpus_per_replica = tp_size * pp_size
        available_per_replica = nnodes * gpus_per_node
        if gpus_per_replica != available_per_replica:
            raise ValueError(
                f"vLLM tp_size*pp_size ({gpus_per_replica}) must equal "
                f"nnodes*inference_num_gpus_per_node ({available_per_replica})"
            )
        if num_gpus_total % gpus_per_replica != 0:
            raise ValueError(
                f"inference_num_gpus ({num_gpus_total}) must be divisible by "
                f"tp_size*pp_size ({gpus_per_replica})"
            )
        num_replicas = num_gpus_total // gpus_per_replica
        if xnode_ray:
            num_engines = num_replicas
            gpus_per_engine = gpus_per_replica  # full cross-node TP width
        else:
            num_engines = num_replicas * nnodes
            gpus_per_engine = gpus_per_node
    else:
        gpus_per_engine = tp_size * pp_size
        if num_gpus_total % gpus_per_engine != 0:
            raise ValueError(
                f"inference_num_gpus ({num_gpus_total}) must be divisible by "
                f"tp_size*pp_size ({gpus_per_engine})"
            )
        num_replicas = num_gpus_total // gpus_per_engine
        num_engines = num_replicas

    logger.info(
        f"Initializing {num_engines} vLLM engines "
        f"({gpus_per_engine} GPU(s) each, tp={tp_size}, pp={pp_size}, "
        f"nnodes={nnodes}, replicas={num_replicas}, "
        f"backend={backend})"
    )

    from angelspec.inference.engine.vllm_engine import VllmEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    engine_cls = VllmEngine
    if getattr(args, "mtp_draft_opd", False):
        from angelspec.inference.engine.mtp_opd_engine import MTPOPDEngine

        engine_cls = MTPOPDEngine
    elif getattr(args, "dfly_opd_verify_engine", False):
        from angelspec.inference.engine.dfly_opd_engine import DFlyOPDEngine

        engine_cls = DFlyOPDEngine
    VllmRayActor = ray.remote(engine_cls)
    env_vars = {**get_angelspec_env_vars(), **_vllm_config_env_vars(args)}

    engines = []
    for i in range(num_engines):
        node_rank = 0 if xnode_ray else (i % nnodes if nnodes > 1 else 0)

        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        actor_env = dict(env_vars)
        pp_layer_partition = getattr(args, "vllm_pp_layer_partition", None)
        if pp_layer_partition:
            actor_env["VLLM_PP_LAYER_PARTITION"] = str(pp_layer_partition)
        # Under the multi-node ray backend the single EngineCore and the shared
        # WORLD rendezvous each call vLLM's get_open_port(); a pinned VLLM_PORT
        # makes both resolve to the SAME port, so the EngineCore binds it and
        # rank-0's TCPStore then hits EADDRINUSE. Let vLLM auto-assign there
        # (it owns worker rendezvous for ray anyway, per the port note below).
        if not xnode_ray:
            actor_env["VLLM_PORT"] = str(_vllm_port(engine_group, i))
        if xnode_ray:
            # Pin vLLM's ray workers to exactly this replica's inference bundles,
            # so they don't grab the training bundles that also live in the shared
            # placement group. These are the raw PG bundle ids (already node/gpu
            # sorted). Actor takes 0 GPUs so each worker can claim a full-GPU bundle
            # (RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES lets the driver still see
            # its local devices for rank-0).
            replica_bundles = reordered_bundle_indices[
                bundle_offset : bundle_offset + gpus_per_engine
            ]
            actor_env["VLLM_RAY_BUNDLE_INDICES"] = ",".join(str(b) for b in replica_bundles)
            actor_num_cpus, actor_num_gpus = 0, 0
        else:
            actor_num_cpus, actor_num_gpus = 0.2, 0.2

        engine = VllmRayActor.options(
            num_cpus=actor_num_cpus,
            num_gpus=actor_num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": actor_env},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=node_rank,
            engine_group=engine_group,
        )
        engines.append(engine)

    # Pre-allocate ports to avoid TOCTOU races between parallel engines.
    # Each engine needs 2 consecutive ports (dist_init + nccl). Not needed for
    # the ray backend — vLLM owns worker rendezvous there.
    pre_allocated_ports: dict[int, int] = {}
    if not xnode_ray:
        next_start = 10000
        for i in range(num_engines):
            port = ray.get(
                engines[i].find_free_port.remote(start_port=next_start, consecutive=2),
                timeout=300,
            )
            pre_allocated_ports[i] = port
            next_start = port + 2
            logger.info(f"vLLM Engine {i}: pre-allocated ports {port}, {port + 1}")

    dist_init_addrs: dict[int, str] = {}
    if nnodes > 1 and not xnode_ray:
        configured_addr = getattr(args, "vllm_dist_init_addr", None)
        for replica_idx in range(num_replicas):
            if configured_addr and num_replicas == 1:
                dist_init_addrs[replica_idx] = configured_addr
                logger.info(
                    f"Replica {replica_idx}: using configured dist_init_addr: {configured_addr}"
                )
            else:
                head_idx = replica_idx * nnodes
                head_engine = engines[head_idx]
                ip = ray.get(head_engine.get_node_ip.remote(), timeout=30)
                addr = f"{ip}:{pre_allocated_ports[head_idx]}"
                dist_init_addrs[replica_idx] = addr
                logger.info(
                    f"Replica {replica_idx}: dist_init_addr from pre-allocated port: {addr}"
                )

    init_handles = []
    for i, engine in enumerate(engines):
        replica_idx = i if xnode_ray else (i // nnodes if nnodes > 1 else i)
        init_handles.append(
            engine.init.remote(
                mooncake_config=mooncake_config,
                dist_init_addr=dist_init_addrs.get(replica_idx),
                pre_allocated_port=pre_allocated_ports.get(i),
            )
        )

    if nnodes > 1 and not xnode_ray:
        head_engines = [engines[i] for i in range(num_engines) if i % nnodes == 0]
        worker_engines = [engines[i] for i in range(num_engines) if i % nnodes != 0]
        _alive_worker_engines.extend(worker_engines)
        logger.info(
            f"Prepared multi-node vLLM engines: {len(head_engines)} heads + "
            f"{len(worker_engines)} workers ({num_replicas} replicas)"
        )
        return head_engines, init_handles

    return engines, init_handles


def _init_vllm_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize vLLM engines with Ray placement groups (blocking)."""
    head_engines, init_handles = _prepare_vllm_engines(args, pg, mooncake_config, engine_group)
    nnodes = getattr(args, "vllm_nnodes", 1)
    init_timeout = getattr(args, "vllm_init_timeout", 300 if nnodes == 1 else 600)
    _wait_for_init(init_handles, "Vllm", timeout=init_timeout)
    return head_engines


def _create_and_init_actors(
    args,
    pg,
    num_engines,
    num_gpus_per_engine,
    ray_actor_cls,
    mooncake_config,
    extra_kwargs=None,
    engine_group: int = 0,
) -> tuple[list, list]:
    """Create Ray actors and start their init calls.

    Args:
        args: Configuration arguments.
        pg: Placement group tuple.
        num_engines: Number of engines to create.
        num_gpus_per_engine: GPUs per engine.
        ray_actor_cls: The ray.remote-wrapped actor class.
        mooncake_config: Mooncake configuration dict.
        extra_kwargs: Extra kwargs to pass to actor constructor (e.g. num_gpus_per_engine for Sgl).
        engine_group: Group index for log file disambiguation.

    Returns:
        Tuple of (engines list, init_handles list).
    """
    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    engines = []
    init_handles = []

    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        base_gpu_id = int(reordered_gpu_ids[i * num_gpus_per_engine])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpus_per_engine],
        )

        env_vars = get_angelspec_env_vars()

        constructor_kwargs = {
            "args": args,
            "rank": i,
            "base_gpu_id": base_gpu_id,
            "engine_group": engine_group,
        }
        if extra_kwargs:
            constructor_kwargs.update(extra_kwargs)

        engine = ray_actor_cls.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(**constructor_kwargs)

        engines.append(engine)
        init_handles.append(engine.init.remote(mooncake_config=mooncake_config))

    return engines, init_handles


def _wait_for_init(init_handles: list, engine_label: str, timeout: int = 300):
    """Wait for engine init calls to complete with error handling.

    Args:
        init_handles: List of init() call ObjectRefs.
        engine_label: Label for log messages (e.g. "HF", "Sgl").
        timeout: Timeout in seconds.
    """
    try:
        logger.info(
            f"Waiting for {len(init_handles)} {engine_label} engines to initialize (timeout={timeout}s)..."
        )
        ray.get(init_handles, timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        logger.error(f"Timeout waiting for {engine_label} engines to initialize after {timeout}s")
        ready, not_ready = ray.wait(init_handles, num_returns=len(init_handles), timeout=0)
        logger.error(f"Initialized: {len(ready)}/{len(init_handles)}, Failed: {len(not_ready)}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize {engine_label} engines: {e}")
        raise

    logger.info(f"Successfully initialized {len(init_handles)} {engine_label} engines")
