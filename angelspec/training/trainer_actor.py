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

import os
from argparse import Namespace
from datetime import timedelta

import torch.distributed as dist

from angelspec import AutoDraftModelConfig
from angelspec.models.draft.dflash import DFlashConfig
from angelspec.models.draft.dfly import DFlyConfig
from angelspec.models.draft.dspark import DSparkConfig
from angelspec.models.draft.mtp import MTPConfig
from angelspec.ray.ray_actor import RayActor
from angelspec.training.eagle3_trainer import Eagle3Trainer
from angelspec.utils.distributed import init_gloo_group, init_usp_groups
from angelspec.utils.logging import setup_file_logging


class TrainerActor(RayActor):
    def __init__(self, world_size: int, rank: int, master_addr: str, master_port: int):
        self._world_size = world_size
        self._rank = rank

        self.setup_master(master_addr, master_port, port_range=(20000, 21000))

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)

        self.setup_gpu()
        setup_file_logging("training", self._rank)

    def init(
        self, args: Namespace, role: str, mooncake_config=None, with_ref: bool = False
    ) -> int:
        self.args = args

        backend = getattr(args, "distributed_backend", "nccl")
        if getattr(args, "fsdp_cpu_offload", False) and getattr(args, "fsdp_cpu_backend", None):
            cpu_backend = args.fsdp_cpu_backend
            backend = f"cpu:{cpu_backend},cuda:{backend}"

        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=getattr(args, "distributed_timeout_minutes", 30)),
        )

        if getattr(args, "attention_backend", None) == "usp":
            init_usp_groups(
                sp_ulysses_size=getattr(args, "sp_ulysses_size", 1),
                sp_ring_size=getattr(args, "sp_ring_size", 1),
            )

        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        draft_model_config = getattr(args, "draft_model_config_obj", None)
        if draft_model_config is None and getattr(args, "draft_model_config", None):
            draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

        # Config-based trainer dispatch: DSparkConfig → DSparkTrainer,
        # DFlyConfig → DFlyTrainer, DFlashConfig → DFlashTrainer,
        # MTPConfig → MTPTrainer, else Eagle3.
        # DSparkConfig and DFlyConfig both subclass DFlashConfig, so they MUST be
        # checked before the DFlashConfig branch.
        if isinstance(draft_model_config, DSparkConfig):
            from angelspec.training.dspark_trainer import DSparkTrainer

            self._trainer = DSparkTrainer(args)
        elif isinstance(draft_model_config, DFlyConfig):
            from angelspec.training.dfly_trainer import DFlyTrainer

            self._trainer = DFlyTrainer(args)
        elif isinstance(draft_model_config, DFlashConfig):
            from angelspec.training.dflash_trainer import DFlashTrainer

            self._trainer = DFlashTrainer(args)
        elif isinstance(draft_model_config, MTPConfig):
            from angelspec.training.mtp_trainer import MTPTrainer

            self._trainer = MTPTrainer(args)
        else:
            self._trainer = Eagle3Trainer(args)

        target_model_path = getattr(args, "target_model_path", None)

        if draft_model_config is not None:
            self._trainer.init_model(
                draft_model_config=draft_model_config,
                target_model_path=target_model_path,
                mooncake_config=mooncake_config,
            )

        return 0

    def train_from_queue(self, step: int, num_batches: int) -> dict:
        return self._trainer.train_from_queue(step, num_batches)

    def set_train_queue(self, queue, mooncake_config=None, per_dp_rank_batch_size: int = 1):
        return self._trainer.set_train_queue(
            queue, mooncake_config=mooncake_config, per_dp_rank_batch_size=per_dp_rank_batch_size
        )

    def set_score_engine(self, score_engine) -> None:
        return self._trainer.set_score_engine(score_engine)

    def get_global_step(self) -> int:
        return self._trainer.global_step

    def save_model(self, step: int, force_sync: bool = False) -> None:
        self._trainer.save_model(step, force_sync)

    def save_draft_model_for_serving(self, output_dir: str) -> None:
        self._trainer.save_draft_model_for_serving(output_dir)

    def set_vocab_buffers(self, d2t, t2d) -> None:
        if hasattr(self._trainer, "draft_model") and hasattr(
            self._trainer.draft_model, "set_vocab_buffers"
        ):
            self._trainer.draft_model.set_vocab_buffers(d2t, t2d)
        else:
            raise AttributeError(
                "set_vocab_buffers called but draft model does not support vocab pruning. "
                "DFlash training should not use vocab pruning — check train_entry config."
            )

    def set_eval_queue(self, queue, mooncake_config=None, per_dp_rank_batch_size: int = 1):
        return self._trainer.set_eval_queue(
            queue, mooncake_config=mooncake_config, per_dp_rank_batch_size=per_dp_rank_batch_size
        )

    def cache_eval_samples(self, count: int) -> int:
        return self._trainer.cache_eval_samples(count)

    def save_eval_cache(self, cache_dir: str) -> None:
        return self._trainer.save_eval_cache(cache_dir)

    def load_eval_cache(self, cache_dir: str) -> int:
        return self._trainer.load_eval_cache(cache_dir)

    def eval_from_cache(self) -> dict:
        return self._trainer.eval_from_cache()
