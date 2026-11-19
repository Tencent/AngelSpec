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

"""Pipeline setup: mooncake config, training steps calculation, async training setup."""

import json
import math

import ray

from angelspec.utils.env import get_angelspec_env_vars
from angelspec.utils.logging import logger


def build_mooncake_config(args):
    """Build MooncakeConfig from flat args namespace."""
    from angelspec.config.mooncake_config import MooncakeConfig

    return MooncakeConfig.from_flat_args(args)


def setup_async_training_with_engines(
    args,
    train_group,
    mooncake_config,
    inference_engines,
    controller=None,
    score_engine=None,
    inference_node_ips=(),
):
    """Setup async training with distributed inference engines (e.g., Eagle3).

    The engines are Ray actors responsible for storing tensors in mooncake and returning keys.
    AsyncInferenceManager forwards the keys to the controller.

    Args:
        args: Configuration arguments.
        train_group: Training group.
        mooncake_config: MooncakeConfig object. Each actor initializes its own store.
        inference_engines: List of Ray actor engine handles for distributed generation.
        controller: Optional pre-created AsyncTrainingController. If None, a new one is created.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from angelspec.controller.training_controller import AsyncTrainingController

    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    driver_node_id = ray.get_runtime_context().get_node_id()

    if controller is None:
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": get_angelspec_env_vars()},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, dp_size)

    if getattr(args, "online_dump_enabled", False):
        attach_online_dump_writer(args, controller, mooncake_config, inference_node_ips)

    inference_manager = create_inference_manager(args, controller, inference_engines)

    train_queues = ray.get(controller.get_train_queues.remote())
    train_group.set_train_queues(
        train_queues, mooncake_config, per_dp_rank_batch_size=args.per_dp_rank_batch_size
    )

    eval_queues = ray.get(controller.get_eval_queues.remote())
    # eval_from_cache re-collates individual samples with eval_micro_batch_size,
    # so the fetcher must yield unbatched (batch_size=1) entries.
    train_group.set_eval_queues(eval_queues, mooncake_config, per_dp_rank_batch_size=1)

    # Half on-policy OPD: hand the encoder-only score engine(s) to every training
    # actor so the DFlash step can score its proposal tree mid-step (round-robin).
    if score_engine:
        train_group.set_score_engine(score_engine)

    return controller, inference_manager


def writer_node_ips(args, inference_node_ips):
    """Ray node of every dump writer; None is the driver node.

    By default each inference node other than the driver's gets one writer. A
    writer copies far faster than a node infers, so capture keeps pace with
    inference at any scale; the driver node only gets one when there is no
    other inference node.
    """
    count = getattr(args, "online_dump_writers", None)
    node_ip = getattr(args, "online_dump_writer_node_ip", None)
    if node_ip not in (None, "auto"):
        return [node_ip] * int(count or 1)
    driver_ip = ray.util.get_node_ip_address()
    nodes = [ip for ip in dict.fromkeys(inference_node_ips) if ip != driver_ip] or [None]
    return [nodes[index % len(nodes)] for index in range(int(count or len(nodes)))]


def attach_online_dump_writer(args, controller, mooncake_config, inference_node_ips=()):
    """Start the CPU dump writer actors and hand them to the controller."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from angelspec.inference.online_dump import OnlineDumpWriter

    writer_class = ray.remote(num_cpus=1)(OnlineDumpWriter)
    writer_env = getattr(args, "online_dump_writer_env_vars", {}) or {}
    if isinstance(writer_env, str):
        writer_env = json.loads(writer_env)
    else:
        writer_env = dict(writer_env)
    runtime_env = {
        "env_vars": {
            **get_angelspec_env_vars(),
            **{str(key): str(value) for key, value in writer_env.items()},
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "0",
        }
    }
    node_ips = writer_node_ips(args, inference_node_ips)
    count = len(node_ips)
    driver_node_id = ray.get_runtime_context().get_node_id()
    writers = []
    for index, writer_ip in enumerate(node_ips):
        options = {"runtime_env": runtime_env}
        if writer_ip:
            options["resources"] = {f"node:{writer_ip}": 0.001}
        else:
            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=driver_node_id, soft=False
            )
        writers.append(
            writer_class.options(**options).remote(
                mooncake_config,
                args.online_dump_dump_path,
                args.online_dump_recipe,
                validation_fraction=args.online_dump_validation_fraction,
                seed=args.seed,
                min_loss_tokens=args.min_loss_tokens,
                max_pending_bytes=args.online_dump_max_pending_bytes,
                writer_threads=getattr(args, "online_dump_writer_threads", 16),
                reader_threads=getattr(args, "online_dump_reader_threads", 8),
                mode="dump_only" if getattr(args, "online_dump_dump_only", False) else "online",
                writer_index=index,
                writer_count=count,
            )
        )
    ray.get([writer.status.remote() for writer in writers])
    ray.get(controller.set_online_dump_writers.remote(writers))
    # The driver-side dump-only loop sizes its handoffs by the writers it got.
    args.online_dump_writers = count
    logger.info(
        "Online dump writers initialized at %s on %s",
        args.online_dump_dump_path,
        ", ".join(ip or "the driver node" for ip in node_ips),
    )
    return writers


def create_inference_manager(args, controller, inference_engines):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from angelspec.controller.inference_manager import AsyncInferenceManager

    return AsyncInferenceManager.options(
        runtime_env={"env_vars": get_angelspec_env_vars()},
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False
        ),
    ).remote(
        args,
        controller,
        inference_engines=inference_engines,
        max_concurrent_batches=getattr(args, "max_concurrent_batches", 1),
    )


def _estimate_packed_steps_per_epoch(
    token_lengths: list[int], max_seq: int, global_batch_size: int
) -> int:
    """steps_per_epoch under packing, from an exact greedy simulation over the
    FULL token-length list.

    This is a lower bound: the runtime packer may flush partially-filled rows
    (min_fill_ratio/max_wait) so real rows >= this. Overlength samples
    (``length > max_seq``) are dropped, matching the runtime collator. Falls
    back to the non-packed count when token lengths are unavailable/degenerate
    (e.g. deferred/streamed tokenization not materialized → all lengths ~1),
    which would otherwise collapse the whole dataset into ~1 row.
    """
    n = len(token_lengths)
    nopack = max(1, n // global_batch_size) if global_batch_size > 0 else 1
    if max_seq <= 0 or global_batch_size <= 0:
        return nopack

    if not token_lengths or max(token_lengths) <= 1:
        logger.warning(
            "Packing step estimate: token lengths look degenerate "
            f"(max={max(token_lengths) if token_lengths else 0}); falling back to "
            f"non-packed steps_per_epoch={nopack}."
        )
        return nopack

    rows = 0
    cur_len = 0
    placed = 0
    dropped = 0
    for length in token_lengths:
        if length > max_seq:
            dropped += 1
            continue
        placed += 1
        if cur_len + length > max_seq:
            rows += 1
            cur_len = length
        else:
            cur_len += length
    if cur_len > 0:
        rows += 1

    if rows == 0 or placed == 0:
        return nopack

    avg_samples_per_row = placed / rows
    steps = rows // global_batch_size

    logger.info(
        f"Packing estimate (full sim): {placed} packed ({dropped} overlength dropped) "
        f"→ {rows} rows, avg {avg_samples_per_row:.1f} samples/row, "
        f"estimated {steps} steps/epoch (vs {nopack} without packing)"
    )
    return max(1, steps)


def auto_calculate_training_steps(args, dataset_size: int, token_lengths: list[int] | None = None):
    """Auto-calculate num_train_steps and lr_total_steps based on dataset size if not explicitly set.

    All step counts are in optimizer steps (not dispatches).
    steps_per_epoch = dataset_size // global_batch_size
    where global_batch_size = per_dp_rank_batch_size * dp_size * draft_accumulation_steps.

    When packing is enabled and token_lengths are provided, steps_per_epoch is
    estimated via subsample packing simulation for accuracy.

    If num_train_steps is set by user, num_epochs is calculated from it.
    Otherwise: lr_total_steps = steps_per_epoch * num_epochs
    """

    global_batch_size = args.global_batch_size
    packing_enabled = bool(
        getattr(args, "dflash_packing", False) or getattr(args, "mtp_packing", False)
    )

    if packing_enabled and token_lengths:
        steps_per_epoch = _estimate_packed_steps_per_epoch(
            token_lengths, int(getattr(args, "max_seq_length", 0)), global_batch_size
        )
    else:
        steps_per_epoch = dataset_size // global_batch_size

    if steps_per_epoch == 0:
        logger.warning(
            f"Dataset size ({dataset_size}) < global_batch_size ({global_batch_size}). Setting steps_per_epoch to 1."
        )
        steps_per_epoch = 1

    args.steps_per_epoch = steps_per_epoch

    current_num_train_steps = getattr(args, "num_train_steps", None)
    current_lr_total_steps = getattr(args, "lr_total_steps", None)

    if current_num_train_steps is not None:
        args.num_epochs = math.ceil(current_num_train_steps / steps_per_epoch)
        logger.info(
            f"Setting num_epochs to {args.num_epochs} based on num_train_steps={current_num_train_steps}!"
        )
        if current_lr_total_steps is None:
            args.lr_total_steps = current_num_train_steps
    else:
        num_epochs = getattr(args, "num_epochs", 1)
        # Epochs over switched-in data are counted from the switch step.
        switch_step = int(getattr(args, "data_switch_step", None) or 0)
        calculated_total_steps = switch_step + num_epochs * steps_per_epoch
        args.num_train_steps = calculated_total_steps
        if current_lr_total_steps is None:
            args.lr_total_steps = calculated_total_steps

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    logger.info(
        f"Training steps (optimizer steps): num_train_steps={args.num_train_steps}, "
        f"lr_total_steps={args.lr_total_steps} "
        f"(dataset_size={dataset_size}, global_batch_size={global_batch_size}, "
        f"per_dp_rank_batch_size={args.per_dp_rank_batch_size}, "
        f"accumulation_steps={accumulation_steps}, "
        f"steps_per_epoch={steps_per_epoch}, num_epochs={args.num_epochs})"
    )
