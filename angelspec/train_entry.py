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

"""Training entry point for Eagle3 speculative decoding."""

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
from angelspec.inference.factory import (  # noqa: E402
    prepare_eval_engine,
    prepare_inference_engines,
    prepare_score_engine,
)
from angelspec.ray.placement_group import (  # noqa: E402
    allocate_train_group,
    create_placement_groups,
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

    def wait(self, name: str, refs) -> Any:
        """Wrap ray.get for async phases. Returns the result."""
        if name not in self._pending:
            raise ValueError(f"No async phase '{name}' was started via begin_async()")
        t_before = time.time()
        result = ray.get(refs)
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

    parser = argparse.ArgumentParser(description="Eagle3 speculative decoding training")
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
    _validate_usp_args(flat_args)

    return flat_args


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


def _validate_usp_args(args) -> None:
    if getattr(args, "attention_backend", None) != "usp":
        return

    sp_size = getattr(args, "sp_size", None)
    if sp_size is None:
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
    if sp_size <= 1:
        raise NotImplementedError(f"USP requires sp_size > 1, got {sp_size}")

    inference_engine_type = getattr(args, "inference_engine_type", "sgl")
    if inference_engine_type not in ("sgl", "vllm"):
        raise ValueError(
            f"USP currently supports inference_engine_type in {{sgl, vllm}}, got {inference_engine_type}"
        )

    fsdp_strategy = getattr(args, "fsdp_strategy", "REPLICATE").upper()
    if fsdp_strategy != "REPLICATE":
        raise NotImplementedError(
            f"USP currently only supports fsdp_strategy=REPLICATE, got {fsdp_strategy}"
        )

    micro_batch_size = getattr(args, "micro_batch_size", 1)
    if micro_batch_size != 1:
        raise NotImplementedError(
            f"USP currently only supports micro_batch_size=1, got {micro_batch_size}"
        )


def _get_draft_model_config(args):
    """Resolve draft model config from args or auto-generate from target model."""

    draft_config_path = getattr(args, "draft_model_config", None)
    if draft_config_path is not None:
        return AutoDraftModelConfig.from_file(draft_config_path)

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

    if getattr(args, "inference_engine_type", "hf") not in ("sgl", "hf", "vllm"):
        raise NotImplementedError(
            "DFlash currently supports only inference_engine_type in ('sgl', 'hf', 'vllm')."
        )
    if getattr(args, "defer_tokenization", False):
        raise NotImplementedError("DFlash does not support defer_tokenization=True.")
    validate_dflash_usp_layout(
        attention_backend=getattr(args, "attention_backend", None),
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


def _validate_and_configure_mtp(args, draft_model_config) -> None:
    """Validate single-head MTP config and configure the single-hidden source.

    MTP consumes only the last-layer hidden state (no multi-layer aux fusion)
    and the full vocab (no pruning). The inference engine always appends the
    final-layer slot for ``last_hidden_states``, so we set a single aux layer to
    keep mooncake buffers small.
    """
    from angelspec.models.draft.mtp import MTPConfig

    if not isinstance(draft_model_config, MTPConfig):
        return

    if getattr(args, "inference_engine_type", "hf") not in ("sgl", "hf", "vllm"):
        raise NotImplementedError(
            "MTP currently supports only inference_engine_type in ('sgl', 'hf', 'vllm')."
        )
    if getattr(args, "ttt_length", 0) < 1:
        raise ValueError("MTP requires training.ttt_length >= 1.")

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

    # Single last-layer hidden source: the engine appends the final layer for
    # last_hidden_states automatically; pick one early aux layer as a small,
    # harmless placeholder so num_aux stays low.
    if not getattr(args, "aux_hidden_states_layers", None):
        args.aux_hidden_states_layers = [1]
        logger.info("MTP: set aux_hidden_states_layers = [1] (single last-hidden source)")

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
    from angelspec.models.draft.dspark import DSparkConfig

    if not isinstance(draft_model_config, DSparkConfig):
        return

    l1_alpha = getattr(args, "dspark_l1_loss_alpha", 0.9)
    conf_alpha = getattr(args, "dspark_confidence_head_alpha", 1.0)
    enable_conf = getattr(draft_model_config, "enable_confidence_head", False)
    needs_target = l1_alpha > 0 or (enable_conf and conf_alpha > 0)
    if needs_target and not getattr(args, "store_last_hidden_states", False):
        raise ValueError(
            "DSpark L1 / confidence objectives require target last_hidden_states; "
            "set inference.store_last_hidden_states=true (or zero out "
            "dspark_l1_loss_alpha and dspark_confidence_head_alpha)."
        )


def train_async_no_generation(args):
    """Entry point for Eagle3 online training.

    Supports prefill-only mode (default) and decode mode (train_with_decode=True)
    with speculative decoding. Uses distributed Ray actors with placement groups.
    Engines store tensors in mooncake and return keys to AsyncInferenceManager.
    """
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
        auto_calculate_training_steps(args, dataset_size)

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

        # Encoder-only score engine(s) (half on-policy OPD): only when the "score"
        # role PG was carved out (enable_opd_score_engine). One TP=1 data-parallel
        # engine per score GPU; the trainer round-robins across them.
        score_engines, score_init_refs = prepare_score_engine(args, pgs.get("score"))

    # [8] Wait for all actor init to complete concurrently
    n_train = len(train_init_refs)
    eval_init_refs = [eval_init_ref] if eval_init_ref is not None else []
    logger.info(
        f"Waiting for {n_train} training actors and {len(engine_init_refs)} "
        f"inference engines to initialize in parallel..."
    )
    all_results = timer.wait(
        "Actor initialization",
        train_init_refs + engine_init_refs + eval_init_refs + score_init_refs,
    )

    train_results = all_results[:n_train]
    assert len(set(train_results)) == 1
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


if __name__ == "__main__":
    args = parse_config()
    train_async_no_generation(args)
