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

"""Eval-specific setup and runtime helpers for the controller loop."""

import hashlib
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from angelspec.config.eval_config import eval_max_seq_length
from angelspec.training.checkpoint import (
    _read_checkpoint_metadata,
    _write_checkpoint_metadata,
)
from angelspec.utils.logging import logger

try:
    import swanlab

    from angelspec.utils import swanlab as swanlab_utils
except ImportError:
    swanlab = None
    swanlab_utils = None

EVAL_CACHE_IDLE_TIMEOUT = 300.0


@dataclass
class EvalSetupState:
    eval_interval: int
    eval_enabled: bool
    eval_cache_loaded: bool
    eval_cache_path: str | None
    best_eval_score: float
    eval_dispatch_bs: int
    eval_dataset_size: int
    dp_size: int
    initial_eval_submit_count: int = 0
    cache_eval_hidden_states: bool = True
    named_datasets: tuple = ()
    dataset_name: str | None = None
    idle_timeout_s: float = EVAL_CACHE_IDLE_TIMEOUT


def _check_idle_timeout(
    dispatched_samples: int,
    last_progress_at: float,
    total_samples: int,
    idle_timeout_s: float = EVAL_CACHE_IDLE_TIMEOUT,
) -> None:
    idle_for = time.monotonic() - last_progress_at
    if idle_for >= idle_timeout_s:
        raise TimeoutError(
            "Timed out while waiting for eval cache generation "
            f"(no progress during eval for {idle_for:.1f}s, "
            f"idle_timeout={idle_timeout_s:.1f}s, "
            f"dispatched={dispatched_samples}/{total_samples} samples)"
        )


def update_checkpoint_eval_meta(
    checkpoint_dir: str | None,
    step: int,
    eval_metrics: dict,
    current_best: float,
) -> float:
    """Append eval metrics to checkpoint meta.json and track best checkpoint."""
    if not checkpoint_dir or not eval_metrics:
        return current_best

    base_dir = Path(checkpoint_dir).expanduser()
    step_id = step + 1
    meta_path = base_dir / f"iter_{step_id:07d}" / "meta.json"

    metadata = _read_checkpoint_metadata(meta_path)
    if not metadata:
        logger.warning(f"Checkpoint meta.json not found at {meta_path}, skipping eval meta update")
        return current_best

    for key in ("eval/avg_loss", "eval/avg_acc", "eval/simulated_acc_len"):
        if key in eval_metrics:
            metadata[key] = eval_metrics[key]
    for key, value in eval_metrics.items():
        if key.startswith("eval/") and key.count("/") > 1:
            metadata[key] = value
    _write_checkpoint_metadata(meta_path, metadata)

    score = eval_metrics.get("eval/simulated_acc_len")
    if score is not None and score > current_best:
        current_best = score
        (base_dir / "best_checkpointed_iteration.txt").write_text(str(step_id))
        _write_checkpoint_metadata(base_dir / "best_meta.json", metadata)
        logger.info(f"New best checkpoint: iter_{step_id:07d} (simulated_acc_len={score:.2f})")

    return current_best


def submit_initial_eval_chunk(controller, eval_state: EvalSetupState, args) -> int:
    """Push the first eval prompt window so inference can start filling the eval pool.

    The window holds an inference batch for every engine, so all of them extract eval
    hidden states at once.
    """
    inference_batch_size = getattr(args, "inference_eval_batch_size", None) or getattr(
        args, "inference_batch_size", eval_state.eval_dispatch_bs
    )
    num_engines = getattr(args, "inference_num_gpus", 0) // max(
        1, getattr(args, "inference_num_gpus_per_engine", 1)
    )
    initial_eval_submit_count = min(
        eval_state.eval_dataset_size,
        max(eval_state.eval_dispatch_bs * 2, inference_batch_size * max(1, num_engines)),
    )
    ray.get(controller.submit_eval_chunk.remote(0, initial_eval_submit_count))
    eval_state.initial_eval_submit_count = initial_eval_submit_count
    logger.info(
        f"Eval: {eval_state.eval_dataset_size} samples, dispatch_bs={eval_state.eval_dispatch_bs}, "
        f"initial_submit={initial_eval_submit_count}"
    )
    return initial_eval_submit_count


def generate_eval_cache(
    controller,
    train_group,
    eval_state: EvalSetupState,
    *,
    persist: bool = True,
) -> None:
    """Drain eval results from inference and cache on each trainer.

    Training samples are not dispatched meanwhile: the trainers only consume them
    after the eval, so queued ones would pile up in Mooncake past the sample pool's
    cap. The inference manager dispatches eval prompts past that cap instead.

    Args:
        persist: If True and ``eval_state.eval_cache_path`` is set, async-save the
            CPU cache to disk for later runs. On-demand eval sets this False.
    """
    eval_dataset_size = eval_state.eval_dataset_size
    dp_size = eval_state.dp_size
    dispatch_bs = eval_state.eval_dispatch_bs
    eval_cache_path = eval_state.eval_cache_path

    last_progress_at = time.monotonic()
    desc = "Eval caching" if persist else "Eval HS generation"
    logger.info(
        f"{desc} from inference engine ({eval_dataset_size} samples, dispatch_bs={dispatch_bs})..."
    )
    dispatched_samples = 0
    next_submit_idx = eval_state.initial_eval_submit_count
    eval_progress = tqdm(total=eval_dataset_size, desc=desc, unit="sample")

    while dispatched_samples < eval_dataset_size:
        ok = ray.get(controller.try_dispatch_eval_batch.remote())
        if ok:
            counts = train_group.cache_eval_samples(dispatch_bs // dp_size)
            dispatched_samples += dispatch_bs
            if eval_state.dataset_name is not None and (
                not counts or any(count != dispatched_samples // dp_size for count in counts)
            ):
                raise RuntimeError(
                    f"Incomplete eval cache for dataset {eval_state.dataset_name!r}: {counts}"
                )
            if next_submit_idx < eval_dataset_size:
                next_end = min(next_submit_idx + dispatch_bs, eval_dataset_size)
                ray.get(controller.submit_eval_chunk.remote(next_submit_idx, next_end))
                next_submit_idx = next_end
            last_progress_at = time.monotonic()
            eval_progress.update(dispatch_bs)
        else:
            _check_idle_timeout(
                dispatched_samples, last_progress_at, eval_dataset_size, eval_state.idle_timeout_s
            )
            time.sleep(0.01)

    eval_progress.close()

    ray.get(controller.finalize_eval_dispatch.remote())
    logger.info(f"{desc} complete ({dispatched_samples} samples)")
    if persist and eval_cache_path:
        train_group.async_save_eval_cache(eval_cache_path)
        logger.info(f"Eval cache save started (async) to {eval_cache_path}")


def _materialize_eval_hidden_states(controller, train_group, eval_state: EvalSetupState, args):
    """Extract eval hidden states via inference and park them on the trainers.

    Used when ``cache_eval_hidden_states=False``: nothing is loaded from disk,
    nothing is kept across eval intervals, and the CPU tensors are dropped by
    the caller after ``run_eval``.
    """
    ray.get(controller.reset_eval_dispatch.remote())
    submit_initial_eval_chunk(controller, eval_state, args)
    generate_eval_cache(controller, train_group, eval_state, persist=False)


def run_eval(
    step: int,
    train_group,
    eval_enabled: bool,
    *,
    controller=None,
    eval_state: EvalSetupState | None = None,
    args=None,
) -> dict:
    """Run forward-only eval. Cached tensors are used when they were preloaded;
    otherwise hidden states are generated via inference just-in-time.
    """
    if not eval_enabled:
        return {}
    if eval_state is not None and eval_state.named_datasets:
        return run_named_eval(step, controller, train_group, eval_state, args)

    on_demand = eval_state is not None and not eval_state.cache_eval_hidden_states
    if on_demand:
        if controller is None or args is None:
            raise ValueError(
                "On-demand eval requires controller and args so hidden states "
                "can be generated via inference."
            )
        logger.info(
            f"Step {step}: generating eval hidden states on demand "
            f"({eval_state.eval_dataset_size} samples)"
        )
        try:
            _materialize_eval_hidden_states(controller, train_group, eval_state, args)
        except Exception:
            train_group.clear_eval_cache()
            raise

    try:
        eval_results = train_group.run_eval()
    finally:
        if on_demand:
            train_group.clear_eval_cache()
            logger.info(f"Step {step}: dropped on-demand eval hidden states from CPU memory")

    eval_metrics = eval_results[0] if eval_results else {}
    if eval_metrics:
        eval_metrics["eval/step"] = step
        if wandb.run is not None:
            wandb.log(eval_metrics)
        if getattr(swanlab, "run", None) is not None:
            swanlab_utils.log(eval_metrics, step=step)
        logger.info(
            f"Step {step} eval: "
            f"loss={eval_metrics.get('eval/avg_loss', 0):.4f}, "
            f"acc={eval_metrics.get('eval/avg_acc', 0):.4f}, "
            f"sim_acc_len={eval_metrics.get('eval/simulated_acc_len', 0):.2f}, "
            f"acc_per_pos=["
            + ", ".join(
                f"{eval_metrics.get(f'eval/acc_{i}', 0):.3f}"
                for i in range(8)
                if f"eval/acc_{i}" in eval_metrics
            )
            + "], "
            f"acc_gt={eval_metrics.get('eval/avg_acc_gt', 0):.4f}, "
            f"sim_acc_len_gt={eval_metrics.get('eval/simulated_acc_len_gt', 0):.2f}, "
            f"acc_gt_per_pos=["
            + ", ".join(
                f"{eval_metrics.get(f'eval/acc_gt_{i}', 0):.3f}"
                for i in range(8)
                if f"eval/acc_gt_{i}" in eval_metrics
            )
            + "]"
        )
    return eval_metrics


def run_named_eval(step, controller, train_group, state, args):
    from angelspec.config.eval_config import named_eval_cache_key
    from angelspec.utils.logging import get_tb_writer

    if controller is None or args is None:
        raise ValueError("Named eval requires controller and args")
    combined, per_dataset = {}, []
    for info in state.named_datasets:
        name = info["name"]
        current = replace(
            state,
            named_datasets=(),
            dataset_name=name,
            eval_dataset_size=info["dispatch_samples"],
            eval_dispatch_bs=state.dp_size,
            initial_eval_submit_count=0,
        )
        current.eval_cache_path = None
        if state.cache_eval_hidden_states:
            key = named_eval_cache_key(args, info)
            current.eval_cache_path = os.path.join(
                os.path.abspath(args.cache_dir),
                "eval_cache",
                "named",
                name,
                key,
            )
        train_group.clear_eval_cache()
        try:
            selected = ray.get(controller.select_eval_dataset.remote(name))
            if selected != info["dispatch_samples"]:
                raise RuntimeError(f"Eval dataset {name!r} changed size during training")
            loaded = (
                train_group.load_eval_cache(current.eval_cache_path)
                if current.eval_cache_path
                else []
            )
            expected = info["dispatch_samples"] // state.dp_size
            complete = bool(loaded) and all(count == expected for count in loaded)
            if not complete:
                train_group.clear_eval_cache()
                submit_initial_eval_chunk(controller, current, args)
                generate_eval_cache(
                    controller,
                    train_group,
                    current,
                    persist=state.cache_eval_hidden_states,
                )
            name_seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            seed = (int(getattr(args, "eval_seed", 42)) + name_seed) % (2**32)
            results = train_group.run_named_eval(seed)
            if not results or any(not result for result in results):
                raise RuntimeError(f"Eval dataset {name!r} produced empty metrics")
            metrics = results[0]
            per_dataset.append(metrics)
            for key, value in metrics.items():
                if key.startswith("eval/") and key != "eval/step":
                    combined[f"eval/{name}/{key[5:]}"] = value
            combined[f"eval/{name}/samples"] = info["samples"]
            logger.info(
                "Step %s eval[%s]: samples=%d loss=%.4f acc=%.4f sim_acc_len=%.2f",
                step,
                name,
                info["samples"],
                metrics.get("eval/avg_loss", 0),
                metrics.get("eval/avg_acc", 0),
                metrics.get("eval/simulated_acc_len", 0),
            )
        finally:
            train_group.clear_eval_cache()
    common = set.intersection(*(set(metrics) for metrics in per_dataset))
    for key in sorted(common):
        if key.startswith("eval/") and key != "eval/step":
            values = [metrics[key] for metrics in per_dataset]
            if all(isinstance(value, (int, float)) for value in values):
                combined[key] = sum(values) / len(values)
    combined["eval/dataset_count"] = len(per_dataset)
    combined["eval/step"] = step
    if wandb.run is not None:
        wandb.log(combined)
    if getattr(swanlab, "run", None) is not None:
        swanlab_utils.log(combined, step=step)
    writer = get_tb_writer()
    if writer is not None:
        for key, value in combined.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, step)
    return combined


def setup_eval(controller, train_group, args, eval_dataset_size: int) -> EvalSetupState:
    """Prepare eval runtime settings and optionally load/submit eval cache input."""
    eval_interval = args.eval_interval
    eval_enabled = eval_dataset_size > 0
    eval_cache_path: str | None = None
    eval_cache_loaded = False
    initial_eval_submit_count = 0
    cache_eval_hidden_states = bool(getattr(args, "cache_eval_hidden_states", True))

    eval_dispatch_bs = min(args.dp_size, eval_dataset_size) if eval_enabled else 0

    best_eval_score = -float("inf")
    if eval_enabled and args.checkpoint_dir:
        best_meta_path = Path(args.checkpoint_dir).expanduser() / "best_meta.json"
        if best_meta_path.exists():
            existing = _read_checkpoint_metadata(best_meta_path)
            if "eval/simulated_acc_len" in existing:
                best_eval_score = existing["eval/simulated_acc_len"]
                logger.info(f"Resumed best eval score: {best_eval_score:.2f}")

    eval_state = EvalSetupState(
        eval_interval=eval_interval,
        eval_enabled=eval_enabled,
        eval_cache_loaded=eval_cache_loaded,
        eval_cache_path=eval_cache_path,
        best_eval_score=best_eval_score,
        eval_dispatch_bs=eval_dispatch_bs,
        eval_dataset_size=eval_dataset_size,
        dp_size=args.dp_size,
        initial_eval_submit_count=initial_eval_submit_count,
        cache_eval_hidden_states=cache_eval_hidden_states,
        idle_timeout_s=float(getattr(args, "eval_idle_timeout_s", EVAL_CACHE_IDLE_TIMEOUT)),
    )

    from angelspec.config.eval_config import named_eval_datasets

    named = named_eval_datasets(args)
    if named:
        infos = ray.get(controller.get_named_eval_datasets.remote())
        if tuple(info["name"] for info in infos) != tuple(name for name, _ in named):
            raise RuntimeError("Controller and driver disagree on named eval datasets")
        if sum(info["dispatch_samples"] for info in infos) != eval_dataset_size:
            raise RuntimeError("Named eval dataset size mismatch")
        eval_state.named_datasets = tuple(infos)
        return eval_state

    if not eval_enabled:
        return eval_state

    if not cache_eval_hidden_states:
        # Do not load disk shards or pin hidden states in trainer CPU memory.
        # Each eval interval will re-extract them from the inference engine.
        logger.info(
            "Eval: cache_eval_hidden_states=False — skipping disk/CPU cache; "
            "hidden states will be generated via inference at each eval interval"
        )
        return eval_state

    cache_dir = os.path.abspath(getattr(args, "cache_dir", "./cache"))
    # dp_size is part of the key: the cache is sharded one file per rank
    # (eval_rank_<r>.pt), so a cache written at a different dp_size has a
    # different set of rank files. Reusing it leaves some ranks with a loaded
    # shard and others empty -> ragged per-rank eval batch counts -> desynced
    # FSDP all-gathers -> NCCL deadlock. Key on dp_size to keep layouts separate.
    from angelspec.data.megatron_bin import resolve_image_newline_rewrite
    from angelspec.data.positions import position_recipe_from_config

    recipe = position_recipe_from_config(getattr(args, "draft_model_config_obj", None))
    position_key = f"|positions={recipe.identity}" if recipe is not None else ""
    eval_path = getattr(args, "eval_data_path", "") or ""
    file_key = ""
    if os.path.isfile(eval_path):
        st = os.stat(eval_path)
        file_key = f"|{st.st_size}-{st.st_mtime}"
    # The cached hidden states also depend on how the samples were rendered and
    # which target layers were captured.
    layout_key = (
        f"|template={getattr(args, 'chat_template', None)}"
        f"|aux={list(getattr(args, 'aux_hidden_states_layers', None) or [])}"
        f"|newline={resolve_image_newline_rewrite(args)}"
    )
    bin_start = int(getattr(args, "megatron_bin_start", 0) or 0)
    window_key = f"|bins={bin_start}+{getattr(args, 'megatron_max_bins', -1)}" if bin_start else ""
    cache_key = hashlib.md5(
        f"{eval_path}{file_key}|"
        f"{getattr(args, 'target_model_path', '')}|"
        f"{eval_max_seq_length(args) or 0}|"
        f"dp{getattr(args, 'dp_size', 1)}{position_key}{layout_key}{window_key}".encode()
    ).hexdigest()[:12]
    eval_state.eval_cache_path = os.path.join(cache_dir, "eval_cache", cache_key)

    loaded = train_group.load_eval_cache(eval_state.eval_cache_path)
    if all(n > 0 for n in loaded):
        eval_state.eval_cache_loaded = True
        logger.info(
            f"Eval: loaded cached tensors from {eval_state.eval_cache_path} "
            f"({loaded[0]} batches per rank)"
        )
        return eval_state

    # Partial load: some ranks loaded a shard (n>0), others got nothing
    # (an interrupted previous save at this same dp_size). Appending fresh
    # batches on top would leave stale+fresh ranks with divergent per-rank
    # counts -> desynced FSDP all-gathers -> NCCL deadlock. Drop every
    # rank's loaded cache so regeneration starts uniformly empty.
    if any(n > 0 for n in loaded):
        logger.warning(
            f"Eval: partial cache load (per-rank counts {loaded}); "
            "clearing all ranks before regenerating"
        )
        train_group.clear_eval_cache()
    submit_initial_eval_chunk(controller, eval_state, args)
    return eval_state
