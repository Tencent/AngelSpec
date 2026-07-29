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

import json
import os
from typing import Optional, Type

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from angelspec.utils.distributed import _build_usp_group_ranks
from angelspec.utils.env import get_angelspec_env_vars


class RayTrainGroup:
    """
    A group of ray actors for training
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the training group.
        num_nodes (int): Number of nodes for this training group.
        num_gpus_per_node (int): Number of gpus for this training group.
        pg (PlacementGroup, optional): Placement group to schedule training workers on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each training worker.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each training worker.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        training_class (type, optional): The training class to use. Defaults to None.
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "training",
        training_class: Optional[Type] = None,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self._training_class = training_class

        # Allocate the GPUs for training workers
        self._allocate_gpus_for_training(pg, num_gpus_per_actor)

    def _allocate_gpus_for_training(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        train_env_vars = self.args.train_env_vars
        if isinstance(train_env_vars, str):
            train_env_vars = json.loads(train_env_vars) if train_env_vars else {}

        env_vars = {
            **get_angelspec_env_vars(),
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get(
                "NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"
            ),
            **train_env_vars,
        }
        if "TORCHINDUCTOR_CACHE_DIR" in os.environ:
            env_vars["TORCHINDUCTOR_CACHE_DIR"] = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        env_vars.setdefault(
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
            os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800"),
        )
        env_vars.setdefault(
            "TORCHINDUCTOR_FX_GRAPH_CACHE",
            os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE", "1"),
        )

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(
            self._training_class
        )

        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, mooncake_config=None, with_ref=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [
            actor.init.remote(args, role, mooncake_config=mooncake_config, with_ref=with_ref)
            for actor in self._actor_handlers
        ]

    def train_from_queue(self, step: int, num_batches: int):
        """Do one optimizer step by fetching data from queues"""
        return ray.get(
            [actor.train_from_queue.remote(step, num_batches) for actor in self._actor_handlers]
        )

    def save_model(self, step, force_sync=False):
        """Save training model"""
        return ray.get(
            [
                actor.save_model.remote(step, force_sync=force_sync)
                for actor in self._actor_handlers
            ]
        )

    def save_draft_model_for_serving(self, output_dir: str):
        """Save draft model in HuggingFace format for serving weight update.

        This is a collective FSDP2 operation - all actors must participate.
        """
        return ray.get(
            [
                actor.save_draft_model_for_serving.remote(output_dir)
                for actor in self._actor_handlers
            ]
        )

    def set_vocab_buffers(self, d2t, t2d):
        """Set vocab mapping buffers on draft models on all actors."""
        return ray.get(
            [actor.set_vocab_buffers.remote(d2t, t2d) for actor in self._actor_handlers]
        )

    def set_train_queues(self, queues, mooncake_config, per_dp_rank_batch_size: int = 1):
        """Set training data queues for queue-based training.

        Args:
            queues: List of Ray Queues, one per DP rank.
            mooncake_config: MooncakeConfig object. Each actor initializes its own store.
            per_dp_rank_batch_size: Number of samples per DP rank per training step.
        """
        queues = self._expand_queues_for_usp(queues)
        return ray.get(
            [
                actor.set_train_queue.remote(
                    queue,
                    mooncake_config=mooncake_config,
                    per_dp_rank_batch_size=per_dp_rank_batch_size,
                )
                for actor, queue in zip(self._actor_handlers, queues, strict=True)
            ]
        )

    def set_eval_queues(self, queues, mooncake_config, per_dp_rank_batch_size: int = 1):
        """Set eval data queues — mirrors set_train_queues."""
        queues = self._expand_queues_for_usp(queues)
        return ray.get(
            [
                actor.set_eval_queue.remote(
                    queue,
                    mooncake_config=mooncake_config,
                    per_dp_rank_batch_size=per_dp_rank_batch_size,
                )
                for actor, queue in zip(self._actor_handlers, queues, strict=True)
            ]
        )

    def set_score_engine(self, score_engine):
        """Broadcast the encoder-only score engine handle to all training actors."""
        return ray.get(
            [actor.set_score_engine.remote(score_engine) for actor in self._actor_handlers]
        )

    def _expand_queues_for_usp(self, queues):
        actor_count = len(self._actor_handlers)
        if len(queues) == actor_count:
            return queues

        attention_backend = getattr(self.args, "attention_backend", None)
        if attention_backend != "usp":
            raise ValueError(
                f"Number of queues ({len(queues)}) must match number of actors ({actor_count})"
            )

        sp_size = getattr(self.args, "sp_ulysses_size", 1) * getattr(self.args, "sp_ring_size", 1)
        if sp_size <= 0 or actor_count % sp_size != 0:
            raise ValueError(f"Invalid USP topology: actor_count={actor_count}, sp_size={sp_size}")
        dp_size = actor_count // sp_size
        if len(queues) != dp_size:
            raise ValueError(
                f"USP expects {dp_size} queues for {actor_count} actors with sp_size={sp_size}, got {len(queues)}"
            )
        draft_sp_groups, _, _ = _build_usp_group_ranks(
            world_size=actor_count,
            sp_ulysses_size=getattr(self.args, "sp_ulysses_size", 1),
            sp_ring_size=getattr(self.args, "sp_ring_size", 1),
        )
        rank_to_dp_idx = {
            rank: dp_idx for dp_idx, ranks in enumerate(draft_sp_groups) for rank in ranks
        }
        return [queues[rank_to_dp_idx[rank]] for rank in range(actor_count)]

    def cache_eval_samples(self, count: int):
        """Tell every actor to drain ``count`` individual samples from its eval queue into CPU cache."""
        return ray.get([actor.cache_eval_samples.remote(count) for actor in self._actor_handlers])

    def save_eval_cache(self, cache_dir: str):
        """Persist CPU-cached eval batches to disk on every actor (blocking)."""
        return ray.get([actor.save_eval_cache.remote(cache_dir) for actor in self._actor_handlers])

    def async_save_eval_cache(self, cache_dir: str) -> list[ray.ObjectRef]:
        """Fire-and-forget disk save — returns refs the caller can optionally await."""
        return [actor.save_eval_cache.remote(cache_dir) for actor in self._actor_handlers]

    def load_eval_cache(self, cache_dir: str):
        """Load eval batches from disk on every actor. Returns list of counts (0 = miss)."""
        return ray.get([actor.load_eval_cache.remote(cache_dir) for actor in self._actor_handlers])

    def run_eval(self):
        """Run forward-only eval on every actor using the CPU-cached batches."""
        return ray.get([actor.eval_from_cache.remote() for actor in self._actor_handlers])
