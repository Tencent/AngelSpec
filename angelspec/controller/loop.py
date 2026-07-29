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
from angelspec.utils.logging import get_tb_writer, logger


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


def _cleanup_old_checkpoints(checkpoint_dir: str | None, max_checkpoints: int) -> None:
    """Delete old checkpoints, keeping only the most recent `max_checkpoints`.

    Checkpoint directories are named ``iter_NNNNNNN`` where N is the step number.
    The ``latest_checkpointed_iteration.txt`` and ``best_*`` files are preserved.
    """
    if not checkpoint_dir or max_checkpoints <= 0:
        return

    base_dir = Path(checkpoint_dir).expanduser()
    if not base_dir.exists():
        return

    # Find all iter_* directories, sorted by step number
    iter_dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and re.match(r"iter_\d+", d.name)),
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if len(iter_dirs) <= max_checkpoints:
        return

    # Delete oldest checkpoints, keep the newest max_checkpoints
    to_delete = iter_dirs[: len(iter_dirs) - max_checkpoints]
    for old_dir in to_delete:
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

    # Inference engine is alive. Run eval hs generation first.
    if eval_state.eval_enabled and not eval_state.eval_cache_loaded:
        generate_eval_cache(controller, train_group, eval_state)

    eval_interval = eval_state.eval_interval
    eval_enabled = eval_state.eval_enabled
    best_eval_score = eval_state.best_eval_score

    # ── Online real spec-decode eval setup (independent of simulated eval) ──
    online_eval_state = setup_online_eval(args, infer_model_arch(args), eval_engine=eval_engine)

    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    num_steps = args.num_train_steps
    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    # steps_per_epoch in optimizer steps, pre-computed in auto_calculate_training_steps
    steps_per_epoch = getattr(args, "steps_per_epoch", dataset_size // args.global_batch_size)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    num_epochs = getattr(args, "num_epochs", 1)

    # Submit training data AFTER eval hs generation so that training prompts don't
    # leak into the inference pipeline during eval.
    # Resume is best-effort: completed optimizer steps determine epoch/skip, but
    # async prompt/result buffers can still lose or replay a small tail.
    start_step = ray.get(train_group._actor_handlers[0].get_global_step.remote())
    resume_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    resume_skip = (start_step % steps_per_epoch) * args.global_batch_size if start_step > 0 else 0
    ray.get(controller.submit_training_dataset.remote(epoch=resume_epoch, skip=resume_skip))

    logger.info(
        f"Starting: num_steps={num_steps}, num_epochs={num_epochs}, "
        f"steps_per_epoch={steps_per_epoch}, global_batch_size={args.global_batch_size}, "
        f"accumulation_steps={accumulation_steps}, "
        f"dp_size={dp_size}, per_dp_rank_batch_size={args.per_dp_rank_batch_size}"
    )

    enable_perf = getattr(args, "enable_perf_metrics", True)

    completed_steps = start_step
    current_epoch = completed_steps // steps_per_epoch + 1
    steps_in_current_epoch = completed_steps % steps_per_epoch
    if start_step > 0:
        logger.info(f"Resuming from step {start_step} (epoch {current_epoch})")
    dispatch_attempts = 0
    consecutive_failures = 0
    last_saved_step: int | None = None
    progress = tqdm(total=num_steps, desc="Training", unit="step", initial=start_step)
    while completed_steps < num_steps:
        # Inner loop: dispatch accumulation_steps batches before training
        dispatches_done = 0
        if enable_perf:
            t_dispatch = time.time()
        status = None
        while dispatches_done < accumulation_steps:
            dispatch_attempts += 1

            dispatched = ray.get(controller.try_dispatch_batch.remote())
            if dispatched:
                dispatches_done += 1
                consecutive_failures = 0
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

                should_reload = False
                if steps_in_current_epoch >= steps_per_epoch:
                    should_reload = True
                elif (
                    consecutive_failures >= 500
                    and (completed_steps > 0 or dispatches_done > 0)
                    and status is not None
                    and status["sample_pool_size"] < status["dispatch_batch_size"]
                    and status.get("prompt_buffer_size", 0) == 0
                ):
                    logger.warning(
                        f"Pool insufficient for dispatch "
                        f"(pool_size={status['sample_pool_size']}, "
                        f"need={status['dispatch_batch_size']}, "
                        f"{dispatches_done}/{accumulation_steps} dispatches done, "
                        f"{steps_in_current_epoch}/{steps_per_epoch} steps in epoch). "
                        f"Reloading dataset."
                    )
                    should_reload = True

                if should_reload:
                    if completed_steps < num_steps:
                        current_epoch += 1
                        steps_in_current_epoch = 0
                        consecutive_failures = 0
                        logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                        ray.get(controller.reload_dataset.remote())
                    else:
                        logger.info("Max steps reached, stopping")
                        break

                time.sleep(0.01)
        else:
            # All accumulation dispatches succeeded — run training
            if enable_perf:
                dispatch_wait = time.time() - t_dispatch

            train_futures = [
                actor.train_from_queue.remote(
                    step=completed_steps,
                    num_batches=accumulation_steps,
                )
                for actor in train_group._actor_handlers
            ]

            train_results = ray.get(train_futures)
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
                        logger.info(
                            f"TIMING step={completed_steps}: "
                            f"step={step_time:.3f}s "
                            f"data={data_t:.3f}s "
                            f"compute={compute_t:.3f}s "
                            f"[fwd={fwd_t:.3f}s bwd={bwd_t:.3f}s opt={opt_t:.3f}s] "
                            f"dispatch={dispatch_wait:.3f}s "
                            f"fill={fill:.1%} eff_tok/s={eff_tps:,.0f}"
                        )

                if getattr(wandb, "run", None) is not None:
                    wandb.log(metrics)

                tb_writer = get_tb_writer()
                if tb_writer is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(key, value, completed_steps)

            # ── Eval at explicit interval (if configured) ─────────
            # Skip if a checkpoint save is about to run (it will eval anyway)
            save_due = _is_save_interval_step(completed_steps, args.save_interval)
            if eval_interval > 0 and completed_steps % eval_interval == 0 and not save_due:
                run_eval(completed_steps, train_group, eval_enabled)

            # Online real spec-decode eval (serves current draft, reads engine-native
            # acceptance length). Blocking by design; no-ops unless enabled + due.
            online_metrics = maybe_run_online_eval(
                online_eval_state, completed_steps, train_group, args
            )
            if online_metrics and wandb.run is not None:
                wandb.log({k: v for k, v in online_metrics.items() if isinstance(v, (int, float))})

            steps_in_current_epoch += 1
            dispatch_attempts = 0

            status = ray.get(controller.get_full_status.remote())
            postfix = {
                "loss": f"{metrics.get('train/avg_loss', 0):.3f}",
                "acc": f"{metrics.get('train/avg_acc', 0):.3f}",
                "acc_len": f"{metrics.get('train/simulated_acc_len', 0):.2f}",
                "thru": f"{status['inference_speed']:.1f}",
            }
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
                eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
                logger.info(f"Saving checkpoint at step {completed_steps}...")
                train_group.save_model(completed_steps)
                last_saved_step = completed_steps
                best_eval_score = update_checkpoint_eval_meta(
                    args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                )
                max_ckpts = getattr(args, "max_checkpoints", 0)
                if max_ckpts > 0:
                    _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

            _maybe_sync_draft_weights(args, completed_steps, train_group, inference_engines)

            # Check if epoch completed
            if steps_in_current_epoch >= steps_per_epoch:
                logger.info(
                    f"Epoch {current_epoch} completed ({steps_in_current_epoch} steps). "
                    f"Total steps: {completed_steps}/{num_steps}"
                )

                if (
                    args.save_per_epoch
                    and args.checkpoint_dir
                    and last_saved_step != completed_steps
                ):
                    eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
                    logger.info(
                        f"Saving checkpoint at end of epoch {current_epoch} (step {completed_steps})..."
                    )
                    train_group.save_model(completed_steps)
                    last_saved_step = completed_steps
                    best_eval_score = update_checkpoint_eval_meta(
                        args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
                    )
                    max_ckpts = getattr(args, "max_checkpoints", 0)
                    if max_ckpts > 0:
                        _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

                if completed_steps < num_steps:
                    current_epoch += 1
                    steps_in_current_epoch = 0
                    logger.info(f"Dataset exhausted, reloading (epoch {current_epoch})...")
                    ray.get(controller.reload_dataset.remote())
                else:
                    logger.info("Max steps reached")
                    break
            continue

        # Inner while broke (max steps reached during reload), break outer loop
        break

    progress.close()

    # Drain any in-flight async online eval so the final interval's result isn't lost
    # when the process exits (the eval runs on the background thread + EvalEngine actor).
    final_online_metrics = drain_online_eval(online_eval_state)
    if final_online_metrics and wandb.run is not None:
        wandb.log({k: v for k, v in final_online_metrics.items() if isinstance(v, (int, float))})

    # Always save a final checkpoint unless saved.
    if args.checkpoint_dir and last_saved_step != completed_steps:
        eval_metrics = run_eval(completed_steps, train_group, eval_enabled)
        logger.info(f"Saving final checkpoint at step {completed_steps}...")
        train_group.save_model(completed_steps, force_sync=True)
        best_eval_score = update_checkpoint_eval_meta(
            args.checkpoint_dir, completed_steps, eval_metrics, best_eval_score
        )
        max_ckpts = getattr(args, "max_checkpoints", 0)
        if max_ckpts > 0:
            _cleanup_old_checkpoints(args.checkpoint_dir, max_ckpts)

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
    try:
        return training_loop(
            args,
            controller,
            inference_manager,
            train_group,
            inference_engines=inference_engines,
            dataset_size=dataset_size,
            eval_dataset_size=eval_dataset_size,
            eval_engine=eval_engine,
        )
    finally:
        _safe_training_cleanup(
            args=args,
            inference_manager=inference_manager,
            inference_future=inference_future,
            inference_engines=inference_engines,
        )
