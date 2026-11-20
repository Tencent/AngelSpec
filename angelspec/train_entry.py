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

"""Training entry point for AngelSpec draft models."""

import argparse
import os
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON")
# The env var above must be set before importing torch/ray-dependent modules,
# so the following imports intentionally sit below it.
from collections import namedtuple  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from typing import Any, Generator  # noqa: E402

import ray  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # noqa: E402

from angelspec import AutoDraftModelConfig  # noqa: E402
from angelspec.config.train_config import config_to_flat_args, load_config  # noqa: E402
from angelspec.config.utils import generate_draft_model_config  # noqa: E402
from angelspec.controller import (  # noqa: E402
    AsyncTrainingController,
    auto_calculate_training_steps,
    build_mooncake_config,
    run_training_loop,
    setup_async_training_with_engines,
)
from angelspec.controller.loop import run_dump_only_loop  # noqa: E402
from angelspec.controller.setup import (  # noqa: E402
    attach_online_dump_writer,
    create_inference_manager,
)
from angelspec.inference.factory import (  # noqa: E402
    prepare_eval_engine,
    prepare_inference_engines,
    prepare_score_engine,
)
from angelspec.ray.placement_group import (  # noqa: E402
    allocate_train_group,
    create_placement_groups,
    placement_group_node_ips,
)
from angelspec.training.trainer_actor import TrainerActor  # noqa: E402
from angelspec.transfer.mooncake.utils import launch_mooncake_master  # noqa: E402
from angelspec.utils.env import get_angelspec_env_vars  # noqa: E402
from angelspec.utils.logging import init_tracking, logger  # noqa: E402
from angelspec.utils.usp import (  # noqa: E402
    validate_dflash_usp_layout,
    validate_mtp_usp_layout,
)

_Phase = namedtuple("_Phase", ["name", "duration", "is_async", "blocked"])


class _InitTimer:
    """Lightweight segmented timer for initialization phases."""

    def __init__(self) -> None:
        self._t0 = time.time()
        self._phases: list[_Phase] = []
        self._pending: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Time a synchronous phase."""
        start = time.time()
        yield
        self._phases.append(_Phase(name, time.time() - start, is_async=False, blocked=0.0))

    def begin_async(self, name: str) -> None:
        """Mark the start of an async operation (e.g., ray.remote dispatch)."""
        self._pending[name] = time.time()

    def wait(self, name: str, refs, timeout: float | None = None) -> Any:
        """Wrap ray.get for async phases. Returns the result."""
        if name not in self._pending:
            raise ValueError(f"No async phase '{name}' was started via begin_async()")
        t_before = time.time()
        result = ray.get(refs, timeout=timeout)
        t_after = time.time()
        dispatch_time = self._pending.pop(name)
        total = t_after - dispatch_time
        blocked = t_after - t_before
        self._phases.append(_Phase(name, total, is_async=True, blocked=blocked))
        return result

    def log_summary(self) -> None:
        total = time.time() - self._t0
        lines = ["Initialization timing:"]
        for p in self._phases:
            suffix = f"  (blocked {p.blocked:.2f}s)" if p.is_async else ""
            lines.append(f"  {p.name:<48s} {p.duration:>8.2f}s{suffix}")
        lines.append(f"  {'─' * 57}")
        lines.append(f"  {'Total':<48s} {total:>8.2f}s")
        logger.info("\n".join(lines))


def parse_config():
    """Parse YAML config and convert to flat args.

    Supports configs with sections matching the Config dataclass:
    model, dataset, training, debug, inference, logging, mooncake, decode.

    The config is flattened via config_to_flat_args(), with prefixed sections:
    mooncake_*, sglang_*, decode_*.
    """

    parser = argparse.ArgumentParser(description="AngelSpec draft-model training")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--print-config-only", action="store_true", help="Print resolved config and exit"
    )

    args, unknown = parser.parse_known_args()

    config = load_config(
        config_path=args.config, cli_args=unknown if unknown else None, save_snapshot=True
    )

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config))

    if args.print_config_only:
        sys.exit(0)

    flat_args = config_to_flat_args(config)

    flat_args.rank = 0
    flat_args.world_size = flat_args.training_num_nodes * flat_args.training_num_gpus_per_node

    defaults = {
        "colocate": False,
        "debug_train_only": False,
        "debug_inference_only": False,
        "dp_size": None,
        "save_debug_train_data": None,
    }
    for key, value in defaults.items():
        if not hasattr(flat_args, key) or getattr(flat_args, key) is None:
            setattr(flat_args, key, value)

    _resolve_batch_size(flat_args)
    _resolve_default_flags(flat_args)
    _validate_usp_args(flat_args)
    _resolve_dp_hsdp(flat_args)

    return flat_args


def _draft_is_mtp(args) -> bool:
    path = getattr(args, "draft_model_config", None)
    if path is None:
        return False
    from angelspec.models.draft.mtp import MTPConfig

    return isinstance(AutoDraftModelConfig.from_file(path), MTPConfig)


def _resolve_default_flags(args) -> None:
    """Fill training flags left unset (None) from the run's setup.

    Must run before the controller is created: it sizes its queues from
    sp_leader_mooncake_fetch.
    """
    usp = getattr(args, "attention_backend", None) == "usp"
    offline = bool(getattr(args, "offline_manifest_path", None))
    if getattr(args, "usp_local_shard", None) is None:
        args.usp_local_shard = usp and (
            offline or getattr(args, "inference_engine_type", "sgl") == "vllm"
        )
    mtp = _draft_is_mtp(args)
    if getattr(args, "sp_leader_mooncake_fetch", None) is None:
        args.sp_leader_mooncake_fetch = mtp and usp and args.usp_local_shard and not offline
    if getattr(args, "gradient_checkpointing", None) is None:
        args.gradient_checkpointing = mtp


def _maybe_create_scratch_draft(args, train_group):
    """Auto-create scratch draft checkpoint for inference engine if not provided."""
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "decode_speculative_algorithm", None)
        and getattr(args, "decode_speculative_draft_model_path", None) is None
    ):
        scratch_dir = os.path.join(getattr(args, "output_dir", "./outputs"), "scratch_draft_model")
        os.makedirs(scratch_dir, exist_ok=True)
        logger.info(f"Auto-creating scratch draft checkpoint at {scratch_dir}")
        train_group.save_draft_model_for_serving(scratch_dir)
        args.decode_speculative_draft_model_path = scratch_dir
        logger.info(f"Set decode_speculative_draft_model_path = {scratch_dir}")


def _resolve_batch_size(args):
    """Derive dp_size, per_dp_rank_batch_size, dispatch_batch_size, and global_batch_size."""
    packing = getattr(args, "dflash_packing", False) or getattr(args, "mtp_packing", False)
    micro_batch_size = getattr(args, "micro_batch_size", 1)
    if packing and micro_batch_size != 1:
        # Each micro-batch is one packed row; a larger value only shrinks the
        # estimated steps per epoch.
        raise ValueError(
            f"Sequence packing requires training.micro_batch_size=1, got {micro_batch_size}."
        )
    world_size = args.training_num_nodes * args.training_num_gpus_per_node
    if getattr(args, "attention_backend", None) == "usp":
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
        if sp_size <= 0:
            raise ValueError(f"USP requires positive sp_size, got {sp_size}")
        if world_size % sp_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by USP sp_size ({sp_size})"
            )
        dp_size = getattr(args, "dp_size", None) or (world_size // sp_size)
        if dp_size * sp_size != world_size:
            raise ValueError(
                f"dp_size ({dp_size}) * sp_size ({sp_size}) must equal world_size ({world_size})"
            )
        args.dp_size = dp_size
        args.sp_size = sp_size
        args.per_dp_rank_batch_size = 1
    else:
        dp_size = getattr(args, "dp_size", None) or world_size
        args.dp_size = dp_size
        sp_size = getattr(args, "sp_size", None)
        if sp_size is not None and sp_size != 1:
            raise NotImplementedError(
                f"Sequence parallel is not yet supported (got sp_size={sp_size})"
            )
        sp_size = sp_size or 1
        args.per_dp_rank_batch_size = args.micro_batch_size * sp_size

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    args.global_batch_size = args.per_dp_rank_batch_size * dp_size * accumulation_steps


def _resolve_dp_hsdp(args) -> None:
    """Non-USP FULL_SHARD: an explicit shard size turns the 1-D FSDP mesh into HSDP.

    ``hsdp_shard_dp`` (or the legacy ``hsdp_shard_size``) ranks hold one sharded
    copy; the trainer replicates it across ``world_size / shard`` such blocks of
    consecutive ranks. Unset keeps sharding over the whole world.
    """
    if getattr(args, "attention_backend", None) == "usp":
        return
    if getattr(args, "fsdp_strategy", "REPLICATE").upper() != "FULL_SHARD":
        return
    if getattr(args, "hsdp_shard_sp", None) not in (None, 1):
        raise ValueError("training.hsdp_shard_sp needs attention_backend=usp")
    shard = getattr(args, "hsdp_shard_dp", None)
    if shard is None:
        shard = getattr(args, "hsdp_shard_size", None)
    if shard is None:
        return
    shard = int(shard)
    world_size = int(args.dp_size)
    if shard <= 0 or world_size % shard:
        raise ValueError(
            f"invalid HSDP shard config: hsdp_shard_dp ({shard}) must divide the "
            f"training world size ({world_size})"
        )
    args.hsdp_shard_size = shard


def _validate_usp_args(args) -> None:
    if getattr(args, "sp_leader_mooncake_fetch", False) and (
        getattr(args, "attention_backend", None) != "usp"
        or not getattr(args, "usp_local_shard", False)
    ):
        raise ValueError(
            "training.sp_leader_mooncake_fetch requires attention_backend=usp "
            "and usp_local_shard=true"
        )
    if getattr(args, "attention_backend", None) != "usp":
        return

    sp_size = getattr(args, "sp_size", None)
    if sp_size is None:
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
    if sp_size <= 1:
        raise NotImplementedError(f"USP requires sp_size > 1, got {sp_size}")

    inference_engine_type = getattr(args, "inference_engine_type", "sgl")
    if not getattr(args, "offline_manifest_path", None) and inference_engine_type not in (
        "sgl",
        "vllm",
    ):
        raise ValueError(
            f"USP currently supports inference_engine_type in {{sgl, vllm}}, got {inference_engine_type}"
        )

    fsdp_strategy = getattr(args, "fsdp_strategy", "REPLICATE").upper()
    if fsdp_strategy not in ("REPLICATE", "FULL_SHARD"):
        raise ValueError(
            f"USP supports fsdp_strategy in {{REPLICATE, FULL_SHARD}}, got {fsdp_strategy}"
        )
    if fsdp_strategy == "FULL_SHARD":
        dp_size = int(args.dp_size)
        from angelspec.utils.distributed import (
            _build_usp_hsdp_mesh_ranks,
            resolve_hsdp_shard_size,
        )

        try:
            hsdp_shard_size = resolve_hsdp_shard_size(
                dp_size,
                sp_size,
                hsdp_shard_dp=getattr(args, "hsdp_shard_dp", None),
                hsdp_shard_sp=getattr(args, "hsdp_shard_sp", None),
                hsdp_shard_size=getattr(args, "hsdp_shard_size", None),
                gpus_per_node=int(getattr(args, "training_num_gpus_per_node", dp_size * sp_size)),
            )
            # Final safety net: the mesh builder enforces the same DP-first regime.
            _build_usp_hsdp_mesh_ranks(dp_size * sp_size, sp_size, hsdp_shard_size)
        except ValueError as e:
            raise ValueError(f"invalid HSDP shard config: {e}")
        # Downstream (trainer_actor → init_usp_hsdp_mesh) reads hsdp_shard_size.
        args.hsdp_shard_size = hsdp_shard_size

    micro_batch_size = getattr(args, "micro_batch_size", 1)
    if micro_batch_size != 1:
        raise NotImplementedError(
            f"USP currently only supports micro_batch_size=1, got {micro_batch_size}"
        )


def _get_draft_model_config(args):
    """Resolve draft model config from args or auto-generate from target model."""

    draft_config_path = getattr(args, "draft_model_config", None)
    if draft_config_path is not None:
        from angelspec.training.position_config import configure_draft_positions

        return configure_draft_positions(
            AutoDraftModelConfig.from_file(draft_config_path), args.target_model_path
        )

    config_dict = generate_draft_model_config(
        target_model_path=args.target_model_path,
        cache_dir=getattr(args, "model_download_dir", None),
    )
    return AutoDraftModelConfig.from_dict(config_dict)


def _validate_and_configure_dflash(args, draft_model_config) -> None:
    """Validate DFlash-specific config and auto-set aux layer IDs.

    Called before dataset loading to fail fast on misconfigurations.
    """
    from angelspec.models.draft.dflash import DFlashConfig

    if not isinstance(draft_model_config, DFlashConfig):
        return

    from angelspec.data.positions import position_recipe_from_config
    from angelspec.training.dfly_opd import validate_branch_opd

    validate_branch_opd(args, draft_model_config)
    recipe = position_recipe_from_config(draft_model_config)
    if recipe is not None:
        if getattr(args, "dflash_opd_enabled", False) and args.dflash_opd_backend == "packed_tree":
            raise ValueError("DFlash MRoPE packed-tree scoring is not aligned; disable OPD")
        if getattr(args, "online_eval_num_gpus", 0):
            raise ValueError("DFlash MRoPE serving is not supported by the current vLLM adapter")

    if getattr(args, "inference_engine_type", "hf") not in ("sgl", "hf", "vllm"):
        raise NotImplementedError(
            "DFlash currently supports only inference_engine_type in ('sgl', 'hf', 'vllm')."
        )
    if getattr(args, "defer_tokenization", False):
        raise NotImplementedError("DFlash does not support defer_tokenization=True.")
    validate_dflash_usp_layout(
        attention_backend=getattr(args, "attention_backend", None),
        usp_local_shard=bool(getattr(args, "usp_local_shard", False)),
        num_anchors=getattr(args, "dflash_num_anchors", 512),
        sp_ulysses_size=int(getattr(args, "sp_ulysses_size", 1)),
        sp_ring_size=int(getattr(args, "sp_ring_size", 1)),
        opd_enabled=bool(getattr(args, "dflash_opd_enabled", False)),
        num_attention_heads=getattr(draft_model_config, "num_attention_heads", None),
    )
    block_size = getattr(args, "dflash_block_size", 16)
    min_loss = getattr(args, "min_loss_tokens", 0)
    if min_loss < 2 * block_size:
        raise ValueError(
            f"DFlash requires dataset.min_loss_tokens >= 2 * training.dflash_block_size "
            f"({min_loss} < {2 * block_size}). Set dataset.min_loss_tokens={2 * block_size}."
        )

    # Auto-set aux layer IDs from draft config if not explicitly provided
    if not getattr(args, "aux_hidden_states_layers", None):
        from angelspec.models.draft.dflash import build_target_layer_ids

        target_layer_ids = getattr(draft_model_config, "target_layer_ids", None)
        if target_layer_ids is None:
            num_target = getattr(draft_model_config, "num_target_layers", 5)
            target_num_hidden = getattr(draft_model_config, "target_num_hidden_layers", 36)
            target_layer_ids = build_target_layer_ids(num_target, target_num_hidden)
        args.aux_hidden_states_layers = target_layer_ids
        logger.info(f"DFlash: set aux_hidden_states_layers = {target_layer_ids}")

    _validate_dflash_target_layers(args, draft_model_config)

    window = getattr(draft_model_config, "context_local_window", None)
    index_layer = getattr(draft_model_config, "target_index_layer", None)
    kv_layers = getattr(draft_model_config, "target_kv_layers", None)
    if index_layer is not None or kv_layers is not None:
        from angelspec.models.draft.dflash import target_layers_per_draft_layer
        from angelspec.models.target.target_indexer import (
            check_target_index_layer,
            load_target_config,
        )
        from angelspec.models.target.target_kv_latent import (
            check_target_kv_compatible,
            check_target_kv_layer,
        )

        field = "target_index_layer" if index_layer is not None else "target_kv_layers"
        # The layers' attention inputs are read off vLLM's aux capture (see TargetIndexer).
        if getattr(args, "inference_engine_type", "hf") != "vllm":
            raise NotImplementedError(f"{field} needs inference_engine_type=vllm")
        if getattr(args, "attention_backend", None) == "usp" and (
            getattr(draft_model_config, "attentive_correction", False)
            or getattr(args, "dflash_attn_diag_blocks", 0)
            or getattr(args, "dflash_attn_oracle", "")
        ):
            raise NotImplementedError(
                "attentive_correction and dflash_attn_diag/oracle read the target hidden over "
                "the whole context on one rank; they do not support USP"
            )
        target_config = load_target_config(args.target_model_path)
        num_layers = draft_model_config.num_hidden_layers

        def owner(layers):
            if isinstance(layers, int):
                return f"target layer {layers}'s"
            return f"target layers {layers}'"

        if index_layer is not None:
            for layer in dict.fromkeys(target_layers_per_draft_layer(index_layer, num_layers)):
                if layer is not None:
                    check_target_index_layer(target_config, layer)
            union_layers = getattr(draft_model_config, "target_index_union_layers", None)
            if union_layers is not None:
                where = f"union of target layers {list(union_layers)} at anchor-1"
            else:
                where = "anchor-1"
            context = (
                f"{owner(index_layer)} indexer top-{draft_model_config.target_index_topk} at "
                f"{where} + context_local_window={window}"
            )
        else:
            context = f"context_local_window={window}" if window is not None else "all of it"
        if kv_layers is not None:
            for layer in dict.fromkeys(target_layers_per_draft_layer(kv_layers, num_layers)):
                check_target_kv_layer(target_config, layer)
                check_target_kv_compatible(target_config, draft_model_config, layer=layer)
            context += f", read as {owner(kv_layers)} own MLA latents"
        logger.info(f"DFlash: context = {context}")
    elif window is not None:
        logger.info(f"DFlash: context_local_window = {window} (context tokens before each anchor)")
    kv_sources = getattr(draft_model_config, "context_kv_source_layers", None)
    if kv_sources is not None:
        logger.info(f"DFlash: draft layers read the context latents of draft layers {kv_sources}")

    init_layers = getattr(draft_model_config, "target_attn_init_layers", None)
    if init_layers is not None:
        from angelspec.models.draft.dflash import target_layers_per_draft_layer
        from angelspec.models.target.target_attention_init import (
            check_target_attn_init_compatible,
        )
        from angelspec.models.target.target_indexer import load_target_config
        from angelspec.models.target.target_kv_latent import (
            check_target_kv_compatible,
            check_target_kv_layer,
        )

        target_config = load_target_config(args.target_model_path)
        layers = target_layers_per_draft_layer(init_layers, draft_model_config.num_hidden_layers)
        field = "target_attn_init_layers"
        for layer in dict.fromkeys(layers):
            check_target_kv_layer(target_config, layer, field=field)
            # The copied q_b / kv_a rope rows only mean the same under the same RoPE.
            check_target_kv_compatible(target_config, draft_model_config, field=field, layer=layer)
        check_target_attn_init_compatible(target_config, draft_model_config, layers)
        logger.info(f"DFlash: draft attention starts from target layers {layers}")

    mlp_init_layers = getattr(draft_model_config, "target_mlp_init_layers", None)
    if mlp_init_layers is not None:
        from angelspec.models.draft.dflash import target_layers_per_draft_layer
        from angelspec.models.target.target_indexer import load_target_config
        from angelspec.models.target.target_mlp_init import (
            check_target_mlp_init_compatible,
        )

        target_config = load_target_config(args.target_model_path)
        layers = target_layers_per_draft_layer(
            mlp_init_layers, draft_model_config.num_hidden_layers
        )
        check_target_mlp_init_compatible(target_config, draft_model_config, layers)
        if getattr(args, "fsdp_strategy", "REPLICATE").upper() == "FULL_SHARD":
            shard_size = getattr(args, "hsdp_shard_size", None)
            if shard_size is None:
                shard_size = int(getattr(args, "world_size", 0) or 0)
            if not shard_size:
                shard_size = int(getattr(args, "training_num_nodes", 1)) * int(
                    getattr(args, "training_num_gpus_per_node", 1)
                )
            num_experts = int(draft_model_config.num_experts)
            if num_experts % shard_size:
                raise ValueError(
                    f"FSDP shard size ({shard_size}) must divide DFlash num_experts "
                    f"({num_experts}) to avoid padded fused-expert shards"
                )
        subset = getattr(draft_model_config, "target_mlp_init_experts", None)
        experts = (
            "all routed experts"
            if subset is None
            else f"{draft_model_config.num_experts} routed experts "
            f"({subset if isinstance(subset, str) else 'listed'})"
        )
        logger.info(
            f"DFlash: draft post-attention norms and trainable MoEs start from target "
            f"layers {layers} with {experts}"
        )

    if (
        getattr(draft_model_config, "draft_mlp_type", "dense") == "moe"
        and str(getattr(args, "optimizer_type", "adamw")).lower() == "muon"
        and not getattr(args, "muon_per_expert", True)
    ):
        raise ValueError(
            "Muon on a DFlash MoE needs training.muon_per_expert=true: otherwise each fused "
            "expert tensor [E, in, out] is orthogonalized as one [E, in*out] matrix, "
            "gathered in full on every rank"
        )

    dense_init_layers = getattr(draft_model_config, "target_mlp_dense_init_layers", None)
    if dense_init_layers is not None:
        from angelspec.models.draft.dflash import target_layers_per_draft_layer
        from angelspec.models.target.target_indexer import load_target_config
        from angelspec.models.target.target_mlp_init import (
            check_target_mlp_dense_init_compatible,
        )

        layers = target_layers_per_draft_layer(
            dense_init_layers, draft_model_config.num_hidden_layers
        )
        k = check_target_mlp_dense_init_compatible(
            load_target_config(args.target_model_path), draft_model_config, layers
        )
        logger.info(
            f"DFlash: draft dense FFNs start from target layers {layers}: shared expert + "
            f"{k} routed experts ({draft_model_config.target_mlp_dense_init_experts!r})"
        )

    if getattr(draft_model_config, "moe_aux_free_bias_update", False) and (
        getattr(args, "fsdp_strategy", "REPLICATE").upper() == "REPLICATE"
    ):
        raise ValueError(
            "moe_aux_free_bias_update keeps expert_bias trainable without a gradient, "
            "which fsdp_strategy REPLICATE (DDP) rejects; use FULL_SHARD"
        )

    moe_lr_scale = getattr(args, "moe_lr_scale", 1.0)
    moe_router_lr_scale = getattr(args, "moe_router_lr_scale", None)
    if not moe_lr_scale > 0 or (moe_router_lr_scale is not None and not moe_router_lr_scale > 0):
        raise ValueError(
            f"moe_lr_scale ({moe_lr_scale}) and moe_router_lr_scale "
            f"({moe_router_lr_scale}) must be positive"
        )
    if moe_lr_scale != 1.0 or moe_router_lr_scale not in (None, 1.0):
        if getattr(draft_model_config, "draft_mlp_type", "dense") != "moe":
            raise ValueError("moe_lr_scale / moe_router_lr_scale need draft_mlp_type='moe'")
        router_scale = moe_lr_scale if moe_router_lr_scale is None else moe_router_lr_scale
        logger.info(
            f"DFlash: MoE FFN params train at {moe_lr_scale:g}x the LR "
            f"(router {router_scale:g}x)"
        )

    fusion = getattr(draft_model_config, "mtp_fusion", None)
    if fusion is not None:
        # Under vllm capture last_hidden_states is the pre-norm final hidden, which the
        # fusion puts through the target's final norm, as vLLM's MTP layer receives it.
        if getattr(args, "inference_engine_type", "hf") != "vllm":
            raise NotImplementedError("mtp_fusion needs inference_engine_type=vllm")
        if not getattr(args, "store_last_hidden_states", False):
            raise ValueError(
                "mtp_fusion reads the target's final hidden state at each anchor's "
                "predecessor; set inference.store_last_hidden_states=true"
            )
        source = "random init"
        if draft_model_config.mtp_fusion_init == "target":
            from angelspec.models.target.target_indexer import load_target_config
            from angelspec.models.target.target_mtp_fusion import (
                check_target_mtp_fusion_compatible,
            )

            check_target_mtp_fusion_compatible(
                load_target_config(args.target_model_path), draft_model_config
            )
            source = "the target's MTP layer"
        slots = {
            "slot0": "the anchor slot",
            "slot1": "the first predicted slot",
            "broadcast": "every block slot",
        }[fusion]
        logger.info(
            "DFlash: MTP fusion adds the target's final hidden at anchor-1, fused with the "
            f"anchor embedding, to {slots}'s input; the fusion starts from {source}"
        )


def _validate_dflash_target_layers(args, draft_model_config) -> None:
    """Fail fast when the aux-layer request cannot deliver num_target_layers hiddens.

    Three counts must agree, and none of them checks the others at runtime: the
    engine captures one hidden state per aux layer id, the trainer splits the
    concatenated result by ``dflash_num_target_layers``, and the draft's
    ``context_proj`` is sized for ``num_target_layers``. A mismatch surfaces far
    from its cause, as a stack/matmul shape error in the first training step.

    Asking for the target's final layer is legal but travels differently under
    vllm: ids are shifted +1 and ``num_hidden_layers`` is appended for
    ``last_hidden_states``, so the final layer lands on that reserved capture
    slot and arrives as ``last_hidden_states`` instead of inside the concatenated
    aux hidden states. We set ``dflash_final_layer_from_last_hidden`` so the
    trainer takes its last context layer from there; without it the run would
    quietly train on one layer less and die splitting the hiddens. sglang
    captures the requested ids verbatim, so there nothing special is needed.
    """
    from angelspec.models.draft.dspark import DSparkConfig, dspark_training_key

    layer_ids = list(getattr(args, "aux_hidden_states_layers", None) or [])
    num_target = getattr(draft_model_config, "num_target_layers", None)
    # DSpark subclasses DFlashConfig but its trainer splits by its own knob when set.
    count_arg = (
        dspark_training_key(args, "num_target_layers")
        if isinstance(draft_model_config, DSparkConfig)
        else "dflash_num_target_layers"
    )
    if num_target is not None:
        # Unset counts follow the draft config; the dflash one is filled for DSpark too,
        # since the mooncake buffers are sized by it.
        for name in ("dflash_num_target_layers", count_arg):
            if getattr(args, name, None) is None:
                setattr(args, name, num_target)
                logger.info(f"DFlash: set training.{name} = {num_target} from the draft config")
    trainer_num_target = getattr(args, count_arg, None)

    if (
        num_target is not None
        and trainer_num_target is not None
        and trainer_num_target != num_target
    ):
        raise ValueError(
            f"training.{count_arg}={trainer_num_target} but the draft config's "
            f"num_target_layers={num_target}. The trainer splits the concatenated aux hidden "
            "states by the former and the draft projects them with the latter; set them equal."
        )
    if num_target is not None and layer_ids and len(layer_ids) != num_target:
        raise ValueError(
            f"{len(layer_ids)} aux layer ids {layer_ids} for num_target_layers={num_target}"
        )

    target_num_hidden = getattr(draft_model_config, "target_num_hidden_layers", None)
    if not layer_ids or not target_num_hidden:
        return
    if getattr(args, "inference_engine_type", "hf") != "vllm":
        return

    final_layer = target_num_hidden - 1
    beyond = [lid for lid in layer_ids if lid > final_layer]
    if beyond:
        raise ValueError(
            f"target_layer_ids{beyond} are past the target's last layer "
            f"(num_hidden_layers={target_num_hidden}); vllm drops them, leaving fewer "
            f"hidden states than num_target_layers={num_target or len(layer_ids)}."
        )
    if final_layer in layer_ids:
        if layer_ids[-1] != final_layer:
            raise ValueError(
                f"target_layer_ids={layer_ids} must list the target's final layer "
                f"({final_layer}) last: vllm delivers it as last_hidden_states, which the "
                "trainer appends after the other layers, so any other position would "
                "silently reorder the draft's per-layer fusion."
            )
        args.dflash_final_layer_from_last_hidden = True
        logger.info(
            f"DFlash: target layer {final_layer} is the target's final layer; taking it "
            "from last_hidden_states (vllm's reserved capture slot)"
        )


def _validate_and_configure_mtp(args, draft_model_config) -> None:
    """Validate single-head MTP config and configure the single-hidden source.

    MTP consumes only the last-layer hidden state (no multi-layer aux fusion)
    and the full vocab (no pruning). The inference engine always appends the
    final-layer slot for ``last_hidden_states``, so we set a single aux layer to
    keep mooncake buffers small.
    """
    from angelspec.models.draft.mtp import MTPConfig

    if not isinstance(draft_model_config, MTPConfig):
        if getattr(args, "mtp_draft_opd", False):
            raise ValueError("mtp_draft_opd requires an MTP draft model config")
        return

    if getattr(args, "mtp_draft_opd", False):
        _validate_mtp_draft_opd(args, draft_model_config)

    if getattr(args, "inference_engine_type", "hf") not in ("sgl", "hf", "vllm"):
        raise NotImplementedError(
            "MTP currently supports only inference_engine_type in ('sgl', 'hf', 'vllm')."
        )
    if getattr(args, "ttt_length", 0) < 1:
        raise ValueError("MTP requires training.ttt_length >= 1.")
    if getattr(args, "sp_leader_mooncake_fetch", False) and (
        getattr(args, "attention_backend", None) != "usp"
        or not getattr(args, "usp_local_shard", False)
    ):
        raise ValueError(
            "training.sp_leader_mooncake_fetch requires attention_backend=usp "
            "and usp_local_shard=true."
        )

    validate_mtp_usp_layout(
        attention_backend=getattr(args, "attention_backend", None),
        usp_local_shard=bool(getattr(args, "usp_local_shard", False)),
        sp_ring_size=int(getattr(args, "sp_ring_size", 1) or 1),
    )

    # No vocab pruning for MTP.
    draft_vocab = getattr(draft_model_config, "draft_vocab_size", None)
    vocab = getattr(draft_model_config, "vocab_size", None)
    if draft_vocab is not None and vocab is not None and draft_vocab != vocab:
        raise ValueError(
            f"MTP does not support vocab pruning (draft_vocab_size {draft_vocab} != vocab_size {vocab})."
        )

    # Single last-layer hidden source. MTP consumes only last_hidden_states, so
    # the per-layer aux `hidden_states` capture is dead weight (stored/transferred/
    # loaded, never read). Only the vLLM engine appends the final layer on its own
    # (vllm_engine._maybe append), so an empty aux list there captures just the
    # final layer -> connector skips hidden_states entirely, saving the whole chain.
    # HF (eagle3_target_model rejects an empty list) and SGLang (no final-append;
    # num_aux=0 untested) are NOT safe with []; keep the [1] placeholder for them.
    if not getattr(args, "aux_hidden_states_layers", None):
        if getattr(args, "inference_engine_type", "hf") == "vllm":
            args.aux_hidden_states_layers = []
            logger.info(
                "MTP (vllm): set aux_hidden_states_layers = [] "
                "(final layer only; skip unused aux hidden_states)"
            )
        else:
            args.aux_hidden_states_layers = [1]
            logger.info("MTP: set aux_hidden_states_layers = [1] (single last-hidden source)")

    if getattr(args, "mooncake_num_aux_layers", None) is None:
        args.mooncake_num_aux_layers = 0
    if getattr(args, "mooncake_hidden_dim", None) is None:
        args.mooncake_hidden_dim = int(
            getattr(draft_model_config, "target_hidden_size", draft_model_config.hidden_size)
        )
    if getattr(args, "mooncake_async_put_pool_size", None) is None:
        args.mooncake_async_put_pool_size = min(2, int(getattr(args, "inference_batch_size", 1)))

    num_experts = int(getattr(draft_model_config, "num_experts", 0) or 0)
    shard_size = getattr(args, "hsdp_shard_size", None)
    if (
        getattr(args, "fsdp_strategy", "REPLICATE").upper() == "FULL_SHARD"
        and num_experts
        and shard_size is not None
        and num_experts % shard_size != 0
    ):
        raise ValueError(
            f"HSDP shard size ({shard_size}) must divide MTP num_experts ({num_experts}) "
            "to avoid padded fused-expert shards."
        )

    # Half on-policy distillation (packed tree-forward scoring). The block is
    # filled from the TTT per-step argmax, so it can be at most ttt_length long.
    if getattr(args, "mtp_opd_tree_forward", False):
        block_size = getattr(args, "mtp_opd_block_size", 8)
        ttt_length = getattr(args, "ttt_length", 0)
        if block_size < 1:
            raise ValueError("mtp_opd_block_size must be >= 1 when mtp_opd_tree_forward is set.")
        if block_size > ttt_length:
            raise ValueError(
                f"mtp_opd_block_size ({block_size}) must be <= ttt_length ({ttt_length}); "
                "the on-policy block is filled from the TTT per-step argmax."
            )
        logger.info(
            "MTP OPD: packed tree-forward scoring enabled (block_size=%d, max_anchors=%s)",
            block_size,
            getattr(args, "mtp_opd_max_anchors", None),
        )


def _validate_mtp_draft_opd(args, draft_model_config) -> None:
    if args.mtp_opd_loss not in ("lk_alpha", "e2e_tv") or args.mtp_opd_temperature != 1.0:
        raise ValueError("MTP draft-OPD requires full-vocabulary lk_alpha or e2e_tv at T=1")
    if args.inference_engine_type != "vllm" or args.mtp_packing:
        raise ValueError("MTP draft-OPD requires vLLM and one conversation per sample")
    if (
        args.attention_backend != "usp"
        or args.fsdp_strategy.upper() != "FULL_SHARD"
        or not args.usp_local_shard
    ):
        raise ValueError("MTP draft-OPD requires USP with FULL_SHARD and usp_local_shard=true")
    if args.micro_batch_size != 1 or args.sp_ring_size != 1:
        raise ValueError("MTP draft-OPD requires micro_batch_size=1 and pure Ulysses SP")
    if args.continual_training or args.optimizer_type != "adamw":
        raise ValueError("MTP draft-OPD uses AdamW; load_path must resume complete OPD state")
    if args.load_path and args.mtp_opd_init_model:
        raise ValueError("Set either load_path or mtp_opd_init_model, not both")
    if not (args.load_path or args.mtp_opd_init_model):
        raise ValueError("MTP draft-OPD requires mtp_opd_init_model or an OPD load_path")
    if args.mtp_opd_max_anchors is None or args.mtp_opd_max_anchors < 1:
        raise ValueError("MTP draft-OPD requires a positive mtp_opd_max_anchors")
    if (
        min(
            args.mtp_opd_block_size,
            args.mtp_opd_prefix_chunk_size,
            args.mtp_opd_verify_max_num_seqs,
            args.mtp_opd_verify_max_num_batched_tokens,
        )
        < 1
    ):
        raise ValueError("MTP draft-OPD block, chunk and verifier limits must be positive")
    if args.ttt_length != args.mtp_opd_block_size:
        raise ValueError("MTP draft-OPD requires ttt_length=mtp_opd_block_size")
    if args.ploss_weights is None:
        args.ploss_weights = [1.0] * args.mtp_opd_block_size
    if list(args.ploss_weights) != [1.0] * args.mtp_opd_block_size:
        raise ValueError("MTP draft-OPD requires equal per-step ploss_weights=1")
    if (
        not args.enable_opd_score_engine
        or args.opd_score_engine_num_gpus != args.dp_size * args.vllm_tp_size
        or args.vllm_pp_size != 1
        or args.vllm_nnodes != 1
        or args.vllm_data_parallel_size != 1
    ):
        raise ValueError(
            "MTP draft-OPD requires one single-node TP verifier per training DP group"
        )
    if args.last_hidden_states_prenorm or not args.store_last_hidden_states:
        raise ValueError("MTP draft-OPD requires stored postnorm target hidden states")
    if args.mtp_opd_tree_forward or args.mtp_ttt_on_policy:
        raise ValueError("MTP draft-OPD performs its own sampling and branch/sequence sharding")
    if (
        args.compile_model
        or getattr(args, "offline_manifest_path", None)
        or args.online_dump_enabled
        or args.train_with_decode
    ):
        raise ValueError("MTP draft-OPD requires eager online training")
    if args.eval_only or args.eval_data_path or args.eval_datasets or args.online_eval_num_gpus:
        raise ValueError("MTP draft-OPD does not support the teacher-forced evaluation path")
    if (
        draft_model_config.attention_type != "mla"
        or draft_model_config.dsa_backend != "tilelang"
        or not draft_model_config.use_compressed_indexer
        or not draft_model_config.compressed_indexer_skip_fine_select
        or draft_model_config.moe_aux_free_bias_update
    ):
        raise ValueError("MTP draft-OPD requires Hy4 flash coarse tilelang MLA and fixed routing")
    draft_model_config.dsa_index_share = False


def _validate_packing_model_compatibility(args, draft_model_config) -> None:
    """Ensure a packing flag cannot select the wrong model's collator."""
    from angelspec.models.draft.dflash import DFlashConfig
    from angelspec.models.draft.mtp import MTPConfig

    dflash_packing = bool(getattr(args, "dflash_packing", False))
    mtp_packing = bool(getattr(args, "mtp_packing", False))
    if dflash_packing and mtp_packing:
        raise ValueError(
            "training.dflash_packing and training.mtp_packing are mutually exclusive."
        )
    if dflash_packing and not isinstance(draft_model_config, DFlashConfig):
        raise ValueError(
            "training.dflash_packing requires a DFlash/DSpark draft model config; "
            f"got model_type={getattr(draft_model_config, 'model_type', None)!r}."
        )
    if mtp_packing and not isinstance(draft_model_config, MTPConfig):
        raise ValueError(
            "training.mtp_packing requires an MTP draft model config; "
            f"got model_type={getattr(draft_model_config, 'model_type', None)!r}."
        )


def _validate_and_configure_dspark(args, draft_model_config) -> None:
    """Validate DSpark-specific requirements.

    DSpark subclasses DFlashConfig, so ``_validate_and_configure_dflash`` already
    ran (aux layers auto-set, min_loss_tokens checked). This only adds the extra
    DSpark constraint: the L1 distillation / confidence-head objectives read the
    target's final hidden state, so ``store_last_hidden_states`` must be enabled
    (DFlash leaves it off).
    """
    from angelspec.models.draft.dspark import DSparkConfig, dspark_training_key

    if not isinstance(draft_model_config, DSparkConfig):
        return

    l1_alpha = getattr(args, dspark_training_key(args, "l1_loss_alpha"), 0.0)
    conf_alpha = getattr(args, "dspark_confidence_head_alpha", 1.0)
    enable_conf = getattr(draft_model_config, "enable_confidence_head", False)
    needs_target = l1_alpha > 0 or (enable_conf and conf_alpha > 0)
    if needs_target and not getattr(args, "store_last_hidden_states", False):
        raise ValueError(
            "DSpark L1 / confidence objectives require target last_hidden_states; "
            "set inference.store_last_hidden_states=true (or zero out "
            "dspark_l1_loss_alpha and dspark_confidence_head_alpha)."
        )


def _validate_and_configure_online_dump(args, draft_model_config) -> None:
    if not getattr(args, "online_dump_enabled", False):
        return
    from angelspec.inference.online_dump import validate_online_dump_args

    validate_online_dump_args(args, draft_model_config)


def _run_dump_only(args, controller, timer):
    """Capture the dataset through the online inference and dump path, without training."""
    args.debug_inference_only = True
    args.colocate = False
    args.training_num_nodes = 0
    args.online_eval_num_gpus = 0
    args.enable_opd_score_engine = False

    timer.begin_async("Dataset loading")
    dataset_size_ref = controller.load_dataset.remote(args)
    with timer.phase("Driver-side init"):
        pgs = create_placement_groups(args)
        launch_mooncake_master(args)
        mooncake_config = build_mooncake_config(args)
    [dataset_size] = timer.wait("Dataset loading", [dataset_size_ref])
    logger.info(f"Dataset loaded on controller: {dataset_size} samples to dump")

    timer.begin_async("Inference engine initialization")
    with timer.phase("Dispatch inference engine init"):
        inference_engines, engine_init_refs = prepare_inference_engines(
            args, pgs["inference"], mooncake_config
        )
    engine_type = getattr(args, "inference_engine_type", "hf")
    timer.wait(
        "Inference engine initialization",
        engine_init_refs,
        timeout=getattr(args, f"{engine_type}_init_timeout", 300),
    )
    with timer.phase("Setup dump-only capture"):
        attach_online_dump_writer(
            args, controller, mooncake_config, placement_group_node_ips(*pgs["inference"][:2])
        )
        inference_manager = create_inference_manager(args, controller, inference_engines)
    timer.log_summary()
    return run_dump_only_loop(args, controller, inference_manager, inference_engines, dataset_size)


def run_training(args):
    """Entry point for asynchronous draft-model training.

    Supports prefill-only mode (default) and decode mode (train_with_decode=True)
    with speculative decoding. Uses distributed Ray actors with placement groups.
    Engines store tensors in mooncake and return keys to AsyncInferenceManager.
    """
    if getattr(args, "offline_manifest_path", None):
        from angelspec.training.offline import run_offline_training

        return run_offline_training(args)

    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "inference_engine_type", "sgl") != "sgl"
    ):
        raise ValueError("train_with_decode=True requires inference_engine_type=sgl")

    init_tracking(args)
    timer = _InitTimer()

    # [1] Create controller early (lightweight: only needs args + dp_size)
    with timer.phase("Create controller"):
        driver_node_id = ray.get_runtime_context().get_node_id()
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": get_angelspec_env_vars()},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, args.dp_size)

    # [1.5] Parse draft config + DFlash validation (before any async work)
    with timer.phase("Parse draft model config"):
        draft_model_config = _get_draft_model_config(args)
        args.draft_model_config_obj = draft_model_config

        _validate_and_configure_dflash(args, draft_model_config)
        _validate_and_configure_dspark(args, draft_model_config)
        _validate_and_configure_mtp(args, draft_model_config)
        _validate_packing_model_compatibility(args, draft_model_config)
        _validate_and_configure_online_dump(args, draft_model_config)

    if getattr(args, "online_dump_enabled", False):
        with timer.phase("Prepare online dump"):
            from angelspec.inference.online_dump import prepare_online_dump_assets

            args.online_dump_recipe = prepare_online_dump_assets(args, draft_model_config)

    if getattr(args, "online_dump_dump_only", False):
        return _run_dump_only(args, controller, timer)

    # [2] Kick off dataset loading on controller (async — runs on actor while driver continues)
    timer.begin_async("Dataset loading")
    dataset_size_ref = controller.load_dataset.remote(args)
    eval_dataset_size_ref = controller.load_eval_dataset.remote(args)

    # [3] Do initialization that doesn't depend on dataset in parallel
    with timer.phase("Driver-side init"):
        pgs = create_placement_groups(args)
        launch_mooncake_master(args)
        mooncake_config = build_mooncake_config(args)

    # [4] Wait for dataset sizes (small ints)
    dataset_size, eval_dataset_size = timer.wait(
        "Dataset loading", [dataset_size_ref, eval_dataset_size_ref]
    )
    logger.info(f"Dataset loaded on controller: {dataset_size} train, {eval_dataset_size} eval")

    # [5] Auto-calculate training steps (needs dataset_size)
    with timer.phase("Auto-calculate training steps"):
        packing_enabled = bool(
            getattr(args, "dflash_packing", False) or getattr(args, "mtp_packing", False)
        )
        token_lengths = None
        if packing_enabled and not getattr(args, "defer_tokenization", False):
            token_lengths = ray.get(controller.get_token_lengths.remote())
        auto_calculate_training_steps(args, dataset_size, token_lengths)

    # [6] Generate vocab mapping on controller if vocab pruning is enabled
    vocab_mapping = None
    draft_vocab_size = getattr(draft_model_config, "draft_vocab_size", None)
    vocab_size = draft_model_config.vocab_size
    if draft_vocab_size is not None and draft_vocab_size != vocab_size:
        with timer.phase("Vocab mapping"):
            logger.info(
                f"Computing vocab mapping on controller (target={vocab_size}, draft={draft_vocab_size})..."
            )
            vocab_mapping = ray.get(
                controller.compute_vocab_mapping.remote(vocab_size, draft_vocab_size)
            )
            logger.info(
                f"Generated vocab mapping: d2t={vocab_mapping[0].shape}, t2d={vocab_mapping[1].shape}"
            )

    # [7] Create training actors + inference engines (args now has num_train_steps)
    timer.begin_async("Actor initialization")
    with timer.phase("Allocate actors + dispatch init"):
        train_group = allocate_train_group(
            args=args,
            num_nodes=args.training_num_nodes,
            num_gpus_per_node=args.training_num_gpus_per_node,
            pg=pgs["training"],
            training_class=TrainerActor,
        )
        train_init_refs = train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )

        # Decode mode: create scratch draft checkpoint before inference engines
        # are prepared, since they need decode_speculative_draft_model_path on args.
        # This blocks on train actor init (FSDP gather), so inference engines are
        # dispatched after to maximize parallelism with the wait below.
        _maybe_create_scratch_draft(args, train_group)

        inference_engines, engine_init_refs = prepare_inference_engines(
            args, pgs["inference"], mooncake_config
        )

        # Persistent online-eval engine (stage 2): only when the "eval" role PG
        # was carved out (online_eval.num_gpus > 0). Runs real spec-decode +
        # offline acceptance metrics on its own bundle/node.
        eval_engine, eval_init_ref = prepare_eval_engine(args, pgs.get("eval"))

        # The OPD backend chooses TP decoders or packed-tree encoder engines.
        score_engines, score_init_refs = prepare_score_engine(args, pgs.get("score"))

    # [8] Wait for all actor init to complete concurrently
    n_train = len(train_init_refs)
    eval_init_refs = [eval_init_ref] if eval_init_ref is not None else []
    logger.info(
        f"Waiting for {n_train} training actors and {len(engine_init_refs)} "
        f"inference engines to initialize in parallel..."
    )
    all_init_refs = train_init_refs + engine_init_refs + eval_init_refs + score_init_refs
    engine_type = getattr(args, "inference_engine_type", "hf")
    init_timeout = getattr(args, f"{engine_type}_init_timeout", 300)
    try:
        all_results = timer.wait(
            "Actor initialization",
            all_init_refs,
            timeout=init_timeout,
        )
    except ray.exceptions.GetTimeoutError:
        groups = {
            "training": train_init_refs,
            "inference": engine_init_refs,
            "eval": eval_init_refs,
            "score": score_init_refs,
        }
        statuses = []
        for label, refs in groups.items():
            if not refs:
                continue
            ready, pending = ray.wait(refs, num_returns=len(refs), timeout=0)
            statuses.append(f"{label}={len(ready)}/{len(refs)} ready")
            if pending:
                logger.error("%s init pending refs: %s", label, pending)
        logger.error(
            "Actor initialization timed out after %ss: %s",
            init_timeout,
            ", ".join(statuses),
        )
        raise

    train_results = all_results[:n_train]
    assert len(set(train_results)) == 1
    if args.dflash_opd_enabled and args.dflash_opd_backend == "branch_requests":
        contracts = ray.get(
            [
                engine.opd_teacher_contract.remote()
                for engine in [*inference_engines, *score_engines]
            ]
        )
        if any(contract != contracts[0] for contract in contracts):
            raise RuntimeError(f"OPD prefill/verify teacher contracts differ: {contracts}")
        logger.info("OPD all-engine teacher contract verified: %s", contracts[0])
    logger.info(
        f"All {n_train} training actors and {len(engine_init_refs)} inference engines initialized"
    )

    if vocab_mapping is not None:
        train_group.set_vocab_buffers(*vocab_mapping)
        logger.info("Loaded vocab mapping into training actors")

    # [9] Setup async training with pre-created controller
    with timer.phase("Setup async training"):
        controller, inference_manager = setup_async_training_with_engines(
            args,
            train_group,
            mooncake_config,
            inference_engines,
            controller=controller,
            score_engine=score_engines,
            inference_node_ips=placement_group_node_ips(*pgs["inference"][:2]),
        )

    timer.log_summary()

    # [10] Run training loop (no ray.put needed — dataset lives on controller)
    run_training_loop(
        args,
        controller,
        inference_manager,
        train_group,
        inference_engines=inference_engines,
        dataset_size=dataset_size,
        eval_dataset_size=eval_dataset_size,
        eval_engine=eval_engine,
    )


train_async_no_generation = run_training


if __name__ == "__main__":
    args = parse_config()
    run_training(args)
