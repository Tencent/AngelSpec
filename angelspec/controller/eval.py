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
from dataclasses import dataclass
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from angelspec.training.checkpoint import _read_checkpoint_metadata, _write_checkpoint_metadata
from angelspec.utils.logging import logger

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


def _check_idle_timeout(
    dispatched_samples: int, last_progress_at: float, total_samples: int
) -> None:
    idle_for = time.monotonic() - last_progress_at
    if idle_for >= EVAL_CACHE_IDLE_TIMEOUT:
        raise TimeoutError(
            "Timed out while waiting for eval cache generation "
            f"(no progress during eval for {idle_for:.1f}s, "
            f"idle_timeout={EVAL_CACHE_IDLE_TIMEOUT:.1f}s, "
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
    _write_checkpoint_metadata(meta_path, metadata)

    score = eval_metrics.get("eval/simulated_acc_len")
    if score is not None and score > current_best:
        current_best = score
        (base_dir / "best_checkpointed_iteration.txt").write_text(str(step_id))
        _write_checkpoint_metadata(base_dir / "best_meta.json", metadata)
        logger.info(f"New best checkpoint: iter_{step_id:07d} (simulated_acc_len={score:.2f})")

    return current_best


def generate_eval_cache(
    controller,
    train_group,
    eval_state: EvalSetupState,
) -> None:
    """Drain eval results from inference and cache on each trainer."""
    eval_dataset_size = eval_state.eval_dataset_size
    dp_size = eval_state.dp_size
    dispatch_bs = eval_state.eval_dispatch_bs
    eval_cache_path = eval_state.eval_cache_path

    last_progress_at = time.monotonic()
    logger.info(
        f"Caching eval hidden states from inference engine ({eval_dataset_size} samples, dispatch_bs={dispatch_bs})..."
    )
    dispatched_samples = 0
    next_submit_idx = eval_state.initial_eval_submit_count
    eval_progress = tqdm(total=eval_dataset_size, desc="Eval caching", unit="sample")

    while dispatched_samples < eval_dataset_size:
        ok = ray.get(controller.try_dispatch_eval_batch.remote())
        if ok:
            train_group.cache_eval_samples(dispatch_bs // dp_size)
            dispatched_samples += dispatch_bs
            if next_submit_idx < eval_dataset_size:
                next_end = min(next_submit_idx + dispatch_bs, eval_dataset_size)
                ray.get(controller.submit_eval_chunk.remote(next_submit_idx, next_end))
                next_submit_idx = next_end
            last_progress_at = time.monotonic()
            eval_progress.update(dispatch_bs)
        else:
            _check_idle_timeout(dispatched_samples, last_progress_at, eval_dataset_size)
            time.sleep(0.01)

    eval_progress.close()

    ray.get(controller.finalize_eval_dispatch.remote())
    logger.info(f"Eval caching complete ({dispatched_samples} samples)")
    if eval_cache_path:
        train_group.async_save_eval_cache(eval_cache_path)
        logger.info(f"Eval cache save started (async) to {eval_cache_path}")


def run_eval(step: int, train_group, eval_enabled: bool) -> dict:
    """Run forward-only eval from cache. Assumes eval cache is already populated."""
    if not eval_enabled:
        return {}
    eval_results = train_group.run_eval()
    eval_metrics = eval_results[0] if eval_results else {}
    if eval_metrics:
        eval_metrics["eval/step"] = step
        if wandb.run is not None:
            wandb.log(eval_metrics)
        logger.info(
            f"Step {step} eval: "
            f"loss={eval_metrics.get('eval/avg_loss', 0):.4f}, "
            f"acc={eval_metrics.get('eval/avg_acc', 0):.4f}, "
            f"sim_acc_len={eval_metrics.get('eval/simulated_acc_len', 0):.2f}"
        )
    return eval_metrics


def setup_eval(controller, train_group, args, eval_dataset_size: int) -> EvalSetupState:
    """Prepare eval runtime settings and optionally load/submit eval cache input."""
    eval_interval = args.eval_interval
    eval_enabled = eval_dataset_size > 0
    eval_cache_path: str | None = None
    eval_cache_loaded = False
    initial_eval_submit_count = 0

    eval_dispatch_bs = min(args.dp_size, eval_dataset_size)

    best_eval_score = -float("inf")
    if eval_enabled and args.checkpoint_dir:
        best_meta_path = Path(args.checkpoint_dir).expanduser() / "best_meta.json"
        if best_meta_path.exists():
            existing = _read_checkpoint_metadata(best_meta_path)
            if "eval/simulated_acc_len" in existing:
                best_eval_score = existing["eval/simulated_acc_len"]
                logger.info(f"Resumed best eval score: {best_eval_score:.2f}")

    if eval_enabled:
        cache_dir = os.path.abspath(getattr(args, "cache_dir", "./cache"))
        # dp_size is part of the key: the cache is sharded one file per rank
        # (eval_rank_<r>.pt), so a cache written at a different dp_size has a
        # different set of rank files. Reusing it leaves some ranks with a loaded
        # shard and others empty -> ragged per-rank eval batch counts -> desynced
        # FSDP all-gathers -> NCCL deadlock. Key on dp_size to keep layouts separate.
        cache_key = hashlib.md5(
            f"{getattr(args, 'eval_data_path', '')}|"
            f"{getattr(args, 'target_model_path', '')}|"
            f"{getattr(args, 'max_seq_length', 0)}|"
            f"dp{getattr(args, 'dp_size', 1)}".encode()
        ).hexdigest()[:12]
        eval_cache_path = os.path.join(cache_dir, "eval_cache", cache_key)

        loaded = train_group.load_eval_cache(eval_cache_path)
        if all(n > 0 for n in loaded):
            eval_cache_loaded = True
            logger.info(
                f"Eval: loaded cached tensors from {eval_cache_path} ({loaded[0]} batches per rank)"
            )
        else:
            inference_batch_size = args.inference_batch_size
            initial_eval_submit_count = min(
                eval_dataset_size,
                max(eval_dispatch_bs * 2, inference_batch_size),
            )
            ray.get(controller.submit_eval_chunk.remote(0, initial_eval_submit_count))
            logger.info(
                f"Eval: {eval_dataset_size} samples, dispatch_bs={eval_dispatch_bs}, "
                f"initial_submit={initial_eval_submit_count}"
            )

    return EvalSetupState(
        eval_interval=eval_interval,
        eval_enabled=eval_enabled,
        eval_cache_loaded=eval_cache_loaded,
        eval_cache_path=eval_cache_path,
        best_eval_score=best_eval_score,
        eval_dispatch_bs=eval_dispatch_bs,
        eval_dataset_size=eval_dataset_size,
        dp_size=args.dp_size,
        initial_eval_submit_count=initial_eval_submit_count,
    )
