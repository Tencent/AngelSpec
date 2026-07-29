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

"""
HF Ray actor engine for distributed deployment across multiple nodes.

This wraps HFRunner as a Ray actor with placement group support.

Accepts pre-tokenized input_ids and loss_mask from batch preprocessing.
"""

from typing import Any

import ray
import torch

from angelspec.inference.engine.base import InferenceEngine
from angelspec.ray.ray_actor import RayActor
from angelspec.utils.logging import logger, setup_file_logging


class HFEngine(InferenceEngine, RayActor):
    """Ray actor wrapper for HFRunner with distributed deployment support.

    This allows HF engines to be deployed across multiple nodes using Ray
    placement groups.

    Accepts pre-tokenized input_ids and loss_mask instead of raw prompts.
    """

    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        engine_group: int = 0,
    ):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self._engine = None
        self._mooncake_config = None
        setup_file_logging("inference", self.rank, group=engine_group)

    def init(self, mooncake_config=None) -> None:
        """Initialize the HFRunner on the allocated GPU.

        This is called after the Ray actor is scheduled on a node.

        Args:
            mooncake_config: MooncakeConfig object for distributed storage.
        """
        from angelspec.inference.engine.hf_runner import HFRunner

        if self.base_gpu_id is not None:
            local_gpu_id = self.setup_gpu(self.base_gpu_id)
            logger.info(
                f"HFEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, using local GPU {local_gpu_id}"
            )

        self._mooncake_config = mooncake_config

        if mooncake_config is not None:
            from angelspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address, mooncake_config.metadata_server
            )

        self._engine = HFRunner.from_pretrained(
            pretrained_model_name_or_path=self.args.target_model_path,
            max_seq_length=getattr(self.args, "max_seq_length", 8192),
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
            aux_hidden_states_layers=getattr(self.args, "aux_hidden_states_layers", None),
            mooncake_config=mooncake_config,
        )

        logger.info(f"HFEngine rank {self.rank}: initialized from {self.args.target_model_path}")

    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate training data from pre-tokenized inputs.

        Args:
            data_id: Data ID(s) for the batch.
            input_ids_ref: Ray ObjectRef or list of input_ids tensors.
            packed_loss_mask_list: List of packed loss_mask strings (e.g. "2,3,2,2,1").
            formatted_prompts: Unused (accepted for interface compatibility).
            return_last_hidden_states: Unused (accepted for interface compatibility).
            return_logits: Unused (accepted for interface compatibility).
            multimodal_inputs: Unused (accepted for interface compatibility).

        Returns:
            List of dicts with mooncake_key or tensors.
        """
        if self._engine is None:
            raise RuntimeError("HFEngine not initialized. Call init() first.")

        assert input_ids_ref is not None, "input_ids_ref must not be None"
        assert packed_loss_mask_list is not None, "packed_loss_mask_list must not be None"

        if isinstance(input_ids_ref, ray.ObjectRef):
            input_ids_list = ray.get(input_ids_ref)
        else:
            input_ids_list = input_ids_ref

        if isinstance(data_id, list):
            batch_data_id = ",".join(data_id)
        else:
            batch_data_id = data_id

        logger.info(
            f"HFEngine rank {self.rank}: processing data_id={batch_data_id}, shapes: {[ids.shape for ids in input_ids_list]}"
        )

        result = self._engine.generate(
            data_id=batch_data_id,
            input_ids_list=input_ids_list,
            packed_loss_mask_list=packed_loss_mask_list,
        )
        return result

    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the engine is healthy."""
        return self._engine is not None

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        logger.info(f"HFEngine rank {self.rank}: shutdown complete")

    def get_status(self) -> dict:
        """Get engine status."""
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
        }
