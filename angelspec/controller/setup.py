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

import math

import ray

from angelspec.utils.env import get_angelspec_env_vars
from angelspec.utils.logging import logger


def build_mooncake_config(args):
    """Build MooncakeConfig from flat args namespace."""
    from angelspec.config.mooncake_config import MooncakeConfig

    return MooncakeConfig.from_flat_args(args)


def setup_async_training_with_engines(
    args, train_group, mooncake_config, inference_engines, controller=None, score_engine=None
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
    from angelspec.controller.inference_manager import AsyncInferenceManager
    from angelspec.controller.training_controller import AsyncTrainingController

    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )

    if controller is None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        driver_node_id = ray.get_runtime_context().get_node_id()
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": get_angelspec_env_vars()},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, dp_size)

    max_concurrent = getattr(args, "max_concurrent_batches", 1)
    inference_manager = AsyncInferenceManager.remote(
        args,
        controller,
        inference_engines=inference_engines,
        max_concurrent_batches=max_concurrent,
    )

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


def auto_calculate_training_steps(args, dataset_size: int):
    """Auto-calculate num_train_steps and lr_total_steps based on dataset size if not explicitly set.

    All step counts are in optimizer steps (not dispatches).
    steps_per_epoch = dataset_size // global_batch_size
    where global_batch_size = per_dp_rank_batch_size * dp_size * draft_accumulation_steps.

    If num_train_steps is set by user, num_epochs is calculated from it.
    Otherwise: lr_total_steps = steps_per_epoch * num_epochs
    """

    global_batch_size = args.global_batch_size
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
        calculated_total_steps = num_epochs * steps_per_epoch
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
