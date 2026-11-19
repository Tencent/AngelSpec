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

"""Pipeline training loop: main training loop with sync training and async inference."""

import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from angelspec.controller.eval import (
    generate_eval_cache,
    run_eval,
    setup_eval,
    update_checkpoint_eval_meta,
)
from angelspec.controller.online_eval import (
    drain_online_eval,
    infer_model_arch,
    maybe_run_online_eval,
    setup_online_eval,
)
from angelspec.training.checkpoint import (
    _read_checkpoint_metadata,
    _write_checkpoint_metadata,
)
from angelspec.utils.logging import get_tb_writer, logger

try:
    import swanlab

    from angelspec.utils import swanlab as swanlab_utils
except ImportError:
    swanlab = None
    swanlab_utils = None


def _validate_train_results(results, expected_workers: int, step: int) -> None:
    if len(results) != expected_workers or not results:
        raise RuntimeError(
            f"step {step}: expected {expected_workers} training results, got {len(results)}"
        )
    failed = [
        rank for rank, result in enumerate(results) if not isinstance(result, dict) or not result
    ]
    if failed:
        raise RuntimeError(
            f"step {step}: workers {failed} returned no training metrics; "
            "refusing to advance completed_steps"
        )


def _maybe_sync_draft_weights(args, completed_steps, train_group, inference_engines):
    """Sync draft model weights to inference engines (decode mode only)."""
    weight_sync_enabled = getattr(args, "decode_weight_sync_enabled", False)
    weight_sync_interval = getattr(args, "decode_weight_sync_interval", 500)
    if not (
        getattr(args, "train_with_decode", False)
        and weight_sync_enabled
        and inference_engines
        and weight_sync_interval > 0
        and completed_steps > 0
        and completed_steps % weight_sync_interval == 0
    ):
        return

    # NOTE: uses local tmp dir; for multi-node, ensure output_dir is on shared filesystem.
    tmp_dir = tempfile.mkdtemp(prefix="draft_weight_sync_")
    try:
        logger.info(f"Step {completed_steps}: saving draft model to {tmp_dir}")
        train_group.save_draft_model_for_serving(tmp_dir)

        logger.info(f"Step {completed_steps}: updating {len(inference_engines)} engine(s)")
        update_results = ray.get(
            [engine.update_weights_from_disk.remote(tmp_dir) for engine in inference_engines]
        )
        for i, res in enumerate(update_results):
            log_fn = logger.info if res.get("success") else logger.warning
            log_fn(f"Engine {i}: success={res.get('success')}, message={res.get('message')}")
    except Exception:
        logger.exception(f"Step {completed_steps}: weight sync failed, skipping")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp dir {tmp_dir}")


def _is_save_interval_step(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def _record_data_position(checkpoint_dir: str | None, step: int, controller) -> None:
    """Store how far the dataset has been trained in the checkpoint's meta.json."""
    if not checkpoint_dir:
        return
    meta_path = Path(checkpoint_dir).expanduser() / f"iter_{step + 1:07d}" / "meta.json"
    metadata = _read_checkpoint_metadata(meta_path)
    if not metadata:
        logger.warning(f"Checkpoint meta.json not found at {meta_path}; data position not saved")
        return
    metadata["data_position"] = ray.get(controller.get_data_position.remote())
    _write_checkpoint_metadata(meta_path, metadata)


def _resume_data_position(args, train_group, start_step, steps_per_epoch, dataset_size):
    """Dataset epoch and number of its samples to skip when resuming at start_step."""
    if start_step <= 0:
        return 0, 0
    if start_step == getattr(args, "data_switch_step", None):
        logger.info(
            f"Training data switched at step {start_step}; starting it from its first sample"
        )
        return 0, 0
    position = ray.get(train_group._actor_handlers[0].get_resume_data_position.remote())
    if position:
        return int(position["dataset_epoch"]), int(position["epoch_consumed_samples"])
    epoch, steps_into_epoch = divmod(start_step, steps_per_epoch)
    if getattr(args, "dflash_packing", False) or getattr(args, "mtp_packing", False):
        # global_batch_size counts packed rows, each holding several samples.
        skip = int(steps_into_epoch * dataset_size / steps_per_epoch)
    else:
        skip = steps_into_epoch * args.global_batch_size
    logger.warning(
        f"Checkpoint has no data position; estimating the resume point as epoch {epoch}, "
        f"{skip} samples in"
    )
    return epoch, skip


#: The iterations of the checkpoints training.keep_epoch_checkpoints keeps as epoch ends,
#: one per line, in the checkpoint directory.
EPOCH_CHECKPOINTS_FILE = "epoch_checkpointed_iterations.txt"


def _epoch_checkpoint_iterations(base_dir: Path) -> set[int] | None:
    """The iterations EPOCH_CHECKPOINTS_FILE lists (none without the file), or None if it
    cannot be read."""
    path = base_dir / EPOCH_CHECKPOINTS_FILE
    if not path.exists():
        return set()
    try:
        return {int(token) for token in path.read_text().split()}
    except (ValueError, OSError) as e:
        logger.warning(f"Could not read the kept epoch checkpoints in {path}: {e}")
        return None


def _keep_epoch_checkpoint(checkpoint_dir: str | None, step: int) -> None:
    """List the checkpoint saved after ``step`` in EPOCH_CHECKPOINTS_FILE, so rotation keeps it."""
    if not checkpoint_dir:
        return
    base_dir = Path(checkpoint_dir).expanduser()
    iteration = step + 1
    if iteration in (_epoch_checkpoint_iterations(base_dir) or set()):
        return
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / EPOCH_CHECKPOINTS_FILE, "a") as f:
        f.write(f"{iteration}\n")
    logger.info(f"Keeping epoch-end checkpoint iter_{iteration:07d} out of rotation")


def _cleanup_old_checkpoints(checkpoint_dir: str | None, max_checkpoints: int) -> None:
    """Delete old checkpoints, keeping only the most recent `max_checkpoints`.

    Checkpoint directories are named ``iter_NNNNNNN`` where N is the step number.
    The ``latest_checkpointed_iteration.txt`` and ``best_*`` files are preserved,
    and the iteration directory that ``best_checkpointed_iteration.txt`` points to
    is exempt from retention so the best pointer never dangles. The epoch ends
    EPOCH_CHECKPOINTS_FILE lists are left out altogether: neither counted nor deleted.
    """
    if not checkpoint_dir or max_checkpoints <= 0:
        return

    base_dir = Path(checkpoint_dir).expanduser()
    if not base_dir.exists():
        return

    kept = _epoch_checkpoint_iterations(base_dir)
    if kept is None:
        # Delete nothing rather than a kept epoch end.
        return

    # Find all iter_* directories, sorted by step number
    iter_dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and re.match(r"iter_\d+", d.name)),
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )
    iter_dirs = [d for d in iter_dirs if int(re.search(r"\d+", d.name).group()) not in kept]

    if len(iter_dirs) <= max_checkpoints:
        return

    # Resolve the best-scoring iteration's directory (written by online eval as the
    # bare step number) so retention keeps it even when it ages out of the window.
    best_dir = None
    best_ptr = base_dir / "best_checkpointed_iteration.txt"
    if best_ptr.exists():
        try:
            best_dir = base_dir / f"iter_{int(best_ptr.read_text().strip()):07d}"
        except (ValueError, OSError) as e:
            logger.warning(f"Could not read best checkpoint pointer {best_ptr}: {e}")

    # Delete oldest checkpoints, keep the newest max_checkpoints (plus the best)
    to_delete = iter_dirs[: len(iter_dirs) - max_checkpoints]
    for old_dir in to_delete:
        if old_dir == best_dir:
            logger.info(f"Keeping best checkpoint (retention exempt): {old_dir}")
            continue
        logger.info(f"Removing old checkpoint: {old_dir}")
        try:
            shutil.rmtree(old_dir)
        except OSError as e:
            logger.warning(f"Failed to remove old checkpoint {old_dir}: {e}")


def _safe_training_cleanup(
    args, inference_manager, inference_future, inference_engines=None
) -> None:
    """Best-effort teardown for inference manager and mooncake master actor."""
    if inference_manager is not None:
        try:
            ray.get(inference_manager.stop.remote())
        except Exception as exc:
            logger.warning(f"Failed to stop inference manager: {exc}")
        if inference_future is not None:
            try:
                ray.get(inference_future)
            except Exception as exc:
                logger.warning(
                    f"Inference manager run loop exited with error during cleanup: {exc}"
                )

    if inference_engines:
        logger.info(f"Shutting down {len(inference_engines)} inference engine(s)...")
        shutdown_refs = []
        for engine in inference_engines:
            try:
                shutdown_refs.append(engine.shutdown.remote())
            except Exception as exc:
                logger.warning(f"Failed to initiate engine shutdown: {exc}")
        for ref in shutdown_refs:
            try:
                ray.get(ref, timeout=30)
            except Exception as exc:
                logger.warning(f"Engine shutdown timed out or failed: {exc}")

    mooncake_master_actor = getattr(args, "_mooncake_master_actor", None)
    if mooncake_master_actor is not None:
        try:
            ray.get(mooncake_master_actor.shutdown.remote(), timeout=10)
        except Exception as exc:
            logger.warning(f"Failed to shutdown mooncake master actor: {exc}")


def training_loop(
    args,
    controller,
    inference_manager,
    train_group,
    inference_engines=None,
    dataset_size=None,
    eval_dataset_size=None,
    eval_engine=None,
):
    """Run the training loop with sync training and async inference.

    Training is synchronous - waits for each step to complete.
    Inference runs in background, continuously producing data.

    Each optimizer step (with draft_accumulation_steps dispatches):
      1. Controller dispatches per_dp_rank_batch_size * dp_size samples, accumulation_steps times
      2. Each DP rank receives per_dp_rank_batch_size * accumulation_steps samples total
      3. train_from_queue(num_batches=accumulation_steps) processes all micro-batches
      4. Optimizer steps after the last micro-batch

    completed_steps counts optimizer steps (consistent with lr_total_steps).

    Args:
        args: Configuration arguments.
        controller: AsyncTrainingController ray actor handle (dataset already loaded).
        inference_manager: AsyncInferenceManager ray actor handle.
        train_group: Training group with set_train_queues method.
        dataset_size: Number of training samples. If None, queried from controller.
        eval_dataset_size: Number of eval samples. If None, queried from controller. 0 means no eval.
    """
    if dataset_size is None:
        dataset_size = ray.get(controller.get_dataset_size.remote())
    if dataset_size <= 0:
        raise ValueError(
            f"Training dataset size is {dataset_size}. "
            f"Ensure controller.load_dataset() was called before run_training_loop()."
        )
    if eval_dataset_size is None:
        eval_dataset_size = ray.get(controller.get_eval_dataset_size.remote())

    # ── Eval setup ──────────────────────────────────────────────
    eval_state = setup_eval(controller, train_group, args, eval_dataset_size)

    # Inference engine is alive. When caching is enabled, extract eval hidden
    # states now (or reuse a disk shard) so later evals are forward-only.
    # cache_eval_hidden_states=False skips this and generates at each eval.
    if (
        eval_state.eval_enabled
        and not eval_state.named_datasets
        and eval_state.cache_eval_hidden_states
        and not eval_state.eval_cache_loaded
    ):
        generate_eval_cache(controller, train_group, eval_state)

    if getattr(args, "eval_only", False):
        if not eval_state.eval_enabled:
            raise ValueError("training.eval_only needs an eval dataset (dataset.eval_data_path)")
        step = ray.get(train_group._actor_handlers[0].get_global_step.remote())
        run_eval(
            step,
            train_group,
            True,
            controller=controller,
            eval_state=eval_state,
            args=args,
        )
        return

    eval_interval = eval_state.eval_interval
    best_eval_score = eval_state.best_eval_score

    # ── Online real spec-decode eval setup (independent of simulated eval) ──
    online_eval_state = setup_online_eval(args, infer_model_arch(args), eval_engine=eval_engine)

    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    num_steps = args.num_train_steps
    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    dumping = getattr(args, "online_dump_enabled", False)
    # steps_per_epoch in optimizer steps, pre-computed in auto_calculate_training_steps
    steps_per_epoch = getattr(args, "steps_per_epoch", dataset_size // args.global_batch_size)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    num_epochs = getattr(args, "num_epochs", 1)

    # Submit training data after optional pre-train eval HS generation so
    # training prompts don't leak into the inference pipeline during that pass.
    # On-demand eval (cache_eval_hidden_states=False) queues its prompts apart;
    # the inference manager dispatches them ahead of the training prompts.
    # Resume skips the samples the checkpoint recorded as trained; prompts that
    # were still in flight when it was saved are inferred again.
    start_step = ray.get(train_group._actor_handlers[0].get_global_step.remote())
    resume_epoch, resume_skip = _resume_data_position(
        args, train_group, start_step, steps_per_epoch, dataset_size
    )
    ray.get(controller.submit_training_dataset.remote(epoch=resume_epoch, skip=resume_skip))

    logger.info(
        f"Starting: num_steps={num_steps}, num_epochs={num_epochs}, "
        f"steps_per_epoch={steps_per_epoch}, global_batch_size={args.global_batch_size}, "
        f"accumulation_steps={accumulation_steps}, "
        f"dp_size={dp_size}, per_dp_rank_batch_size={args.per_dp_rank_batch_size}"
    )

    enable_perf = getattr(args, "enable_perf_metrics", True)

    completed_steps = start_step
    current_epoch = resume_epoch + 1
    steps_in_current_epoch = (
        completed_steps - (getattr(args, "data_switch_step", None) or 0)
    ) % steps_per_epoch
    if start_step > 0:
        logger.info(f"Resuming from step {start_step} (epoch {current_epoch})")
    dispatch_attempts = 0
    consecutive_failures = 0
    last_saved_step: int | None = None
    keep_epoch_checkpoints = bool(getattr(args, "keep_epoch_checkpoints", False))
    # Data-driven epoch boundary: an epoch ends only when the dataset is truly
    # drained (prompt buffer empty + pool can't fill a dispatch + inference idle),
    # not when the advisory steps_per_epoch estimate is hit. The dwell guard
    # (time since last successful dispatch) prevents a transient inference stall
    # from being misread as end-of-epoch; keep it above packing_max_wait_seconds.
    epoch_exhaustion_grace_seconds = float(getattr(args, "epoch_exhaustion_grace_seconds", 30.0))
    t_last_dispatch_success = time.time()
    dump_wait_total = 0.0
    predispatched = 0
    progress = tqdm(total=num_steps, desc="Training", unit="step", initial=start_step)
    while completed_steps < num_steps:
        # Inner loop: dispatch accumulation_steps batches before training
        dispatches_done = predispatched
        predispatched = 0
        if enable_perf:
            t_dispatch = time.time()
        status = None
        while dispatches_done < accumulation_steps:
            dispatch_attempts += 1

            dispatched = ray.get(controller.try_dispatch_batch.remote())
            if dispatched:
                dispatches_done += 1
                consecutive_failures = 0
                t_last_dispatch_success = time.time()
            else:
                consecutive_failures += 1

                # Only fetch status when needed for logging or reload decision
                if dispatch_attempts % 100 == 0 or consecutive_failures >= 500:
                    status = ray.get(controller.get_full_status.remote())

                if dispatch_attempts % 100 == 0 and status is not None:
                    logger.debug(
                        f"Epoch {current_epoch}, Step {completed_steps}: "
                        f"dispatch failed {dispatch_attempts} times "
                        f"(consecutive={consecutive_failures}), "
                        f"pool_size={status['sample_pool_size']}, "
                        f"need={status['dispatch_batch_size']}"
                    )

                # Real end-of-epoch: every prompt has been dispatched to inference
                # (prompt_buffer empty), the pool can't fill another dispatch,
                # inference has gone idle, and we've waited past the grace window
                # (so a transient supply stall isn't misread). NOT the advisory
                # steps_per_epoch estimate. The leftover pool carries over into the
                # next epoch.
                exhausted = (
                    consecutive_failures >= 500
                    and (completed_steps > 0 or dispatches_done > 0)
                    and status is not None
                    and not status["can_dispatch"]
                    and status.get("prompt_buffer_size", 0) == 0
                    and status.get("inference_speed", 0) == 0
                    and (time.time() - t_last_dispatch_success) >= epoch_exhaustion_grace_seconds
                )
                if exhausted:
                    if completed_steps >= num_steps:
                        logger.info("Max steps reached, stopping")
                        break
                    logger.info(
                        f"Epoch {current_epoch} completed "
                        f"({steps_in_current_epoch} steps, "
                        f"{status.get('epoch_consumed_samples', 0)} samples consumed). "
                        f"Total steps: {completed_steps}/{num_steps}"
                    )
                    if (
                        args.save_per_epoch
                        and args.checkpoint_dir
                        and last_saved_step != completed_steps
                    ):
                        logger.info(
                            f"Saving checkpoint at end of epoch {current_epoch} "
                            f"(step {completed_steps})..."
                        )
                        train_group.save_model(completed_steps)
                        _record_data_position(args.checkpoint_dir, completed_steps, controller)
                        last_saved_step = completed_steps
                        if keep_epoch_checkpoints:
                            _keep_epoch_checkpoint(args.checkpoint_dir, completed_steps)
                        eval_metrics = run_eval(
                            completed_steps,
                            train_group,
                            eval_state.eval_enabled,
                            controller=controller,
                            eval_state=eval_state,
                            args=args,
                        )
                        best_eval_score = update_checkpoint_eval_meta(
                            args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                        )
                        max_ckpts = getattr(args, "max_checkpoints", 0)
                        if max_ckpts > 0:
                            _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)
                    elif keep_epoch_checkpoints and last_saved_step == completed_steps:
                        # An interval save already wrote this epoch end.
                        _keep_epoch_checkpoint(args.checkpoint_dir, completed_steps)
                    current_epoch += 1
                    steps_in_current_epoch = 0
                    consecutive_failures = 0
                    t_last_dispatch_success = time.time()
                    logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                    ray.get(controller.reload_dataset.remote())

                time.sleep(0.01)
        else:
            # All accumulation dispatches succeeded — run training
            if enable_perf:
                dispatch_wait = time.time() - t_dispatch
                if dumping:
                    waited = ray.get(controller.get_online_dump_wait_seconds.remote())
                    dump_wait, dump_wait_total = waited - dump_wait_total, waited

            train_futures = [
                actor.train_from_queue.remote(
                    step=completed_steps,
                    num_batches=accumulation_steps,
                )
                for actor in train_group._actor_handlers
            ]

            # Trainers read their queues FIFO, so the next step's micro-batches can be
            # queued while this one trains. Stop at the first miss: waiting and epoch-end
            # handling belong to the dispatch loop above. Not before a save, which records
            # every dispatched sample as trained, nor while dumping, whose release after
            # the step frees the keys of every dispatched sample.
            if (
                completed_steps + 1 < num_steps
                and getattr(args, "predispatch_next_step", True)
                and not dumping
                and not _is_save_interval_step(completed_steps + 1, args.save_interval)
            ):
                while predispatched < accumulation_steps and ray.get(
                    controller.try_dispatch_batch.remote()
                ):
                    predispatched += 1
                if predispatched:
                    t_last_dispatch_success = time.time()

            train_results = ray.get(train_futures)
            if dumping:
                ray.get(controller.release_online_dump_consumed.remote())
            _validate_train_results(train_results, len(train_futures), completed_steps)
            completed_steps += 1

            # Log metrics from training (use rank 0's metrics - they're already all-reduced)
            metrics = train_results[0] if train_results and train_results[0] else {}
            if metrics:
                # Add step counters for wandb x-axis (required in shared mode)
                metrics["train/step"] = completed_steps
                metrics["inference/step"] = completed_steps

                # Add inference metrics (e2e_latency, spec metrics, etc.)
                inference_metrics = ray.get(inference_manager.flush_metrics.remote())
                metrics.update(inference_metrics)

                if enable_perf:
                    metrics["perf/dispatch_wait"] = dispatch_wait
                    if dumping:
                        metrics["perf/online_dump_wait"] = dump_wait
                    step_time = metrics.get("perf/step_time", 0)
                    if step_time > 0:
                        metrics["perf/train_capacity"] = args.global_batch_size / step_time

                    if completed_steps % 5 == 0 or completed_steps <= 5:
                        data_t = metrics.get("perf/data_time", 0)
                        compute_t = metrics.get("perf/compute_time", 0)
                        fwd_t = metrics.get("perf/forward_time", 0)
                        bwd_t = metrics.get("perf/backward_time", 0)
                        opt_t = metrics.get("perf/optimizer_time", 0)
                        fill = metrics.get("data/fill_ratio", 0)
                        eff_tps = metrics.get("perf/effective_tokens_per_sec", 0)
                        if getattr(args, "mtp_draft_opd", False):
                            logger.info(
                                "TIMING step=%d: step=%.3fs dispatch=%.3fs eff_tok/s=%.0f "
                                "verify_sum=%.3fs repair_branches=%.0f/%.0f",
                                completed_steps,
                                step_time,
                                dispatch_wait,
                                eff_tps,
                                metrics.get("opd/verify_seconds_sum", 0),
                                metrics.get("opd/repair_branches", 0),
                                metrics.get("opd/verify_branches", 0),
                            )
                        else:
                            logger.info(
                                f"TIMING step={completed_steps}: "
                                f"step={step_time:.3f}s "
                                f"data={data_t:.3f}s "
                                f"compute={compute_t:.3f}s "
                                f"[fwd={fwd_t:.3f}s bwd={bwd_t:.3f}s opt={opt_t:.3f}s] "
                                f"dispatch={dispatch_wait:.3f}s "
                                + (f"dump_wait={dump_wait:.3f}s " if dumping else "")
                                + f"fill={fill:.1%} eff_tok/s={eff_tps:,.0f}"
                            )

                if getattr(wandb, "run", None) is not None:
                    wandb.log(metrics)
                if getattr(swanlab, "run", None) is not None:
                    swanlab_utils.log(metrics, step=completed_steps)

                tb_writer = get_tb_writer()
                if tb_writer is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(key, value, completed_steps)

            # ── Eval at explicit interval (if configured) ─────────
            # Skip if a checkpoint save is about to run (it will eval anyway)
            save_due = _is_save_interval_step(completed_steps, args.save_interval)
            if eval_interval > 0 and completed_steps % eval_interval == 0 and not save_due:
                run_eval(
                    completed_steps,
                    train_group,
                    eval_state.eval_enabled,
                    controller=controller,
                    eval_state=eval_state,
                    args=args,
                )

            # Online real spec-decode eval (serves current draft, reads engine-native
            # acceptance length). Blocking by design; no-ops unless enabled + due.
            online_metrics = maybe_run_online_eval(
                online_eval_state, completed_steps, train_group, args
            )
            if online_metrics and wandb.run is not None:
                wandb.log({k: v for k, v in online_metrics.items() if isinstance(v, (int, float))})
            if online_metrics and getattr(swanlab, "run", None) is not None:
                swanlab_utils.log(
                    {k: v for k, v in online_metrics.items() if isinstance(v, (int, float))},
                    step=completed_steps,
                )

            steps_in_current_epoch += 1
            dispatch_attempts = 0

            status = ray.get(controller.get_full_status.remote())
            postfix = {
                "loss": f"{metrics.get('train/avg_loss', 0):.3f}",
                "acc": f"{metrics.get('train/avg_acc', 0):.3f}",
                "acc_len": f"{metrics.get('train/simulated_acc_len', 0):.2f}",
                "thru": f"{status['inference_speed']:.1f}",
            }
            if getattr(args, "mtp_draft_opd", False):
                postfix = {
                    "loss": f"{metrics['train/loss']:.4f}",
                    "alpha1": f"{metrics['train/opd_alpha_1']:.3f}",
                    f"alpha{args.mtp_opd_block_size}": f"{metrics[f'train/opd_alpha_{args.mtp_opd_block_size}']:.3f}",
                    "thru": f"{status['inference_speed']:.1f}",
                }
                if "train/opd_e2e_accept_length_proxy" in metrics:
                    postfix["len_proxy"] = f"{metrics['train/opd_e2e_accept_length_proxy']:.3f}"
            for key, value in metrics.items():
                if key.startswith("train_") and key.endswith("/simulated_acc_len"):
                    source = key[len("train_") : -len("/simulated_acc_len")]
                    postfix[f"{source}_len"] = f"{value:.2f}"
            # Only present for DFlash with a distill term on (see
            # DFlashTrainer._reduce_teacher_acc_pp); it tracks the greedy serving
            # accept test, so keep it next to the ground-truth acc_len.
            if "train/simulated_acc_len_tgt" in metrics:
                postfix["acc_len_tgt"] = f"{metrics['train/simulated_acc_len_tgt']:.2f}"
            if enable_perf:
                postfix["I"] = f"{metrics.get('perf/infer_capacity', 0):.1f}"
                postfix["T"] = f"{metrics.get('perf/train_capacity', 0):.1f}"
                postfix["wait"] = f"{dispatch_wait:.1f}s"
                postfix["pool"] = status["sample_pool_size"]
            postfix["epoch"] = f"{current_epoch}/{num_epochs}"
            # Set postfix before update so tqdm emits only one line
            progress.set_postfix(postfix, refresh=False)
            progress.update(1)

            if _is_save_interval_step(completed_steps, args.save_interval):
                # Save before evaluating, so a failing eval cannot lose the checkpoint.
                logger.info(f"Saving checkpoint at step {completed_steps}...")
                train_group.save_model(completed_steps)
                _record_data_position(args.checkpoint_dir, completed_steps, controller)
                last_saved_step = completed_steps
                # Saves between eval steps (save_interval < eval_interval) skip the eval;
                # without an eval interval every save evaluates.
                eval_metrics = {}
                if eval_interval <= 0 or completed_steps % eval_interval == 0:
                    eval_metrics = run_eval(
                        completed_steps,
                        train_group,
                        eval_state.eval_enabled,
                        controller=controller,
                        eval_state=eval_state,
                        args=args,
                    )
                best_eval_score = update_checkpoint_eval_meta(
                    args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                )
                max_ckpts = getattr(args, "max_checkpoints", 0)
                if max_ckpts > 0:
                    _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

            _maybe_sync_draft_weights(args, completed_steps, train_group, inference_engines)
            continue

        # Inner while broke (max steps reached during reload), break outer loop
        break

    progress.close()

    # Drain any in-flight async online eval so the final interval's result isn't lost
    # when the process exits (the eval runs on the background thread + EvalEngine actor).
    final_online_metrics = drain_online_eval(online_eval_state)
    if final_online_metrics and wandb.run is not None:
        wandb.log({k: v for k, v in final_online_metrics.items() if isinstance(v, (int, float))})
    if final_online_metrics and getattr(swanlab, "run", None) is not None:
        swanlab_utils.log(
            {k: v for k, v in final_online_metrics.items() if isinstance(v, (int, float))},
            step=completed_steps,
        )

    # Training that stops on an epoch boundary ends on that epoch's checkpoint.
    final_epoch_end = (
        keep_epoch_checkpoints and completed_steps > 0 and completed_steps % steps_per_epoch == 0
    )
    # Always save a final checkpoint unless saved.
    if args.checkpoint_dir and last_saved_step != completed_steps:
        logger.info(f"Saving final checkpoint at step {completed_steps}...")
        train_group.save_model(completed_steps, force_sync=True)
        _record_data_position(args.checkpoint_dir, completed_steps, controller)
        if final_epoch_end:
            _keep_epoch_checkpoint(args.checkpoint_dir, completed_steps)
        eval_metrics = run_eval(
            completed_steps,
            train_group,
            eval_state.eval_enabled,
            controller=controller,
            eval_state=eval_state,
            args=args,
        )
        best_eval_score = update_checkpoint_eval_meta(
            args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
        )
        max_ckpts = getattr(args, "max_checkpoints", 0)
        if max_ckpts > 0:
            _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)
    elif final_epoch_end:
        # An interval save already wrote this step.
        _keep_epoch_checkpoint(args.checkpoint_dir, completed_steps)

    # Export the final draft in HuggingFace format (pytorch_model.bin + config.json)
    # for serving. Training only writes DCP checkpoints; without this, DFlash/DFlare/
    # Eagle3 runs finish with no directly servable weights. config.save_pretrained
    # carries model_arch (e.g. "dflare"), so the export round-trips correctly.
    if getattr(args, "output_dir", None):
        hf_dir = os.path.join(args.output_dir, "hf_final")
        try:
            logger.info(f"Exporting final draft model (HF format) to {hf_dir}...")
            train_group.save_draft_model_for_serving(hf_dir)
        except Exception:
            logger.exception(f"Final HF export to {hf_dir} failed")

    final_status = ray.get(controller.get_full_status.remote())
    logger.info(
        f"Training completed: {completed_steps} steps in {final_status['elapsed_seconds']:.1f}s | "
        f"avg inference={final_status['avg_inference_speed']:.1f} entries/s | "
        f"avg training={final_status['avg_training_speed']:.1f} entries/s"
    )


def run_training_loop(
    args,
    controller,
    inference_manager,
    train_group,
    inference_engines=None,
    dataset_size=None,
    eval_dataset_size=None,
    eval_engine=None,
):
    inference_future = inference_manager.run.remote()
    succeeded = False
    try:
        result = training_loop(
            args,
            controller,
            inference_manager,
            train_group,
            inference_engines=inference_engines,
            dataset_size=dataset_size,
            eval_dataset_size=eval_dataset_size,
            eval_engine=eval_engine,
        )
        succeeded = True
        return result
    finally:
        _shutdown_and_publish(
            args, controller, inference_manager, inference_future, inference_engines, succeeded
        )


def _shutdown_and_publish(
    args, controller, inference_manager, inference_future, inference_engines, succeeded
):
    """Stop inference and publish the online dump only after a successful run."""
    dumping = getattr(args, "online_dump_enabled", False)
    dump_error = None
    # Staging reads Mooncake, so it must finish before the engines shut down;
    # the disk drain and publication run after the inference GPUs are released.
    if dumping:
        try:
            if succeeded:
                ray.get(controller.flush_online_dump_staging.remote())
            else:
                status = ray.get(controller.abort_online_dump.remote())
                logger.warning("Online dump left incomplete: %s", status)
        except BaseException as exc:  # noqa: B036 - re-raised after cleanup
            if succeeded:
                dump_error = exc
            else:
                logger.warning("Failed to abort online dump cleanly: %s", exc)
    _safe_training_cleanup(
        args=args,
        inference_manager=inference_manager,
        inference_future=inference_future,
        inference_engines=inference_engines,
    )
    if dumping and succeeded:
        try:
            if dump_error is None:
                status = ray.get(controller.finalize_online_dump.remote())
                logger.info("Online dump published: %s", status)
            else:
                status = ray.get(controller.abort_online_dump.remote())
                logger.warning("Online dump left incomplete: %s", status)
        except BaseException as exc:  # noqa: B036 - re-raised below
            dump_error = dump_error or exc
    if dump_error is not None:
        raise dump_error


_DUMP_CHECK_SECONDS = 1.0
_DUMP_WAIT_LOG_SECONDS = 300.0


def _controller_idle(controller):
    status = ray.get(controller.get_full_status.remote())
    if status["prompt_buffer_size"] or status["sample_pool_size"]:
        return None
    return status


def _inference_drained(controller, inference_manager):
    """Controller status once every prompt was inferred and its result taken, else None.

    The manager counts prompts and results moving between it and the controller as
    in transit; a result can still land in the pool after the manager reports idle,
    so the controller is checked again.
    """
    if _controller_idle(controller) is None:
        return None
    manager = ray.get(inference_manager.get_status.remote())
    if manager["prompt_buffer_size"] or manager["pending_tasks"] or manager["in_transit"]:
        return None
    return _controller_idle(controller)


def dump_only_loop(args, controller, inference_manager, inference_future, dataset_size):
    """Pass ``online_dump.capture_epochs`` times over the dataset, handing every
    inferred sample to the dump writer instead of a trainer."""
    epochs = int(getattr(args, "online_dump_capture_epochs", 1))
    # A writer copies one handed batch at a time, so a batch must hold enough
    # samples for every reader thread of every writer, with a second round to
    # cover the slowest.
    max_samples = max(
        int(getattr(args, "inference_batch_size", 1) or 1),
        2
        * int(getattr(args, "online_dump_reader_threads", 8) or 8)
        * int(getattr(args, "online_dump_writers", 1) or 1),
    )
    ray.get(controller.submit_training_dataset.remote(epoch=0, skip=0))
    progress = tqdm(total=epochs * dataset_size, desc="Dumping", unit="sample")
    epoch, last_check, last_log = 0, 0.0, time.time()
    while True:
        drained = ray.get(controller.drain_to_online_dump.remote(max_samples))
        if drained:
            progress.update(drained)
            continue
        now = time.time()
        if now - last_check < _DUMP_CHECK_SECONDS:
            time.sleep(0.05)
            continue
        last_check = now
        if ray.wait([inference_future], timeout=0)[0]:
            ray.get(inference_future)
            raise RuntimeError("Inference manager stopped before the dump-only capture finished")
        status = _inference_drained(controller, inference_manager)
        if status is None:
            if now - last_log >= _DUMP_WAIT_LOG_SECONDS:
                manager = ray.get(inference_manager.get_status.remote())
                logger.info(
                    f"Dump-only epoch {epoch + 1}: waiting for inference "
                    f"({manager['prompt_buffer_size']} prompts queued, "
                    f"{manager['pending_tasks']} batches in flight)"
                )
                last_log = now
            continue
        epoch += 1
        consumed = status["epoch_consumed_samples"]
        if consumed < dataset_size:
            logger.warning(
                f"Dump-only epoch {epoch}: inference returned {consumed} of {dataset_size} samples"
            )
        logger.info(f"Dump-only epoch {epoch}/{epochs} captured {consumed} samples")
        if epoch >= epochs:
            break
        ray.get(controller.reload_dataset.remote())
    progress.close()


def run_dump_only_loop(args, controller, inference_manager, inference_engines, dataset_size):
    inference_future = inference_manager.run.remote()
    succeeded = False
    try:
        dump_only_loop(args, controller, inference_manager, inference_future, dataset_size)
        succeeded = True
    finally:
        _shutdown_and_publish(
            args, controller, inference_manager, inference_future, inference_engines, succeeded
        )
