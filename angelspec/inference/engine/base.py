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

"""Abstract base class for inference engines."""

from abc import ABC, abstractmethod
from typing import Any

import ray
import torch


class InferenceEngine(ABC):
    """Abstract interface for inference engines.

    Concrete implementations (HFEngine, SglEngine) must also inherit from
    RayActor for distributed deployment via Ray placement groups.
    """

    @abstractmethod
    def init(self, mooncake_config=None) -> None:
        """Initialize the engine on the allocated GPU.

        Called after the Ray actor is scheduled on a node.

        Args:
            mooncake_config: MooncakeConfig object for distributed storage.
        """

    @abstractmethod
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
        """Generate training data from inputs.

        Accepts either pre-tokenized input_ids or formatted prompt strings.

        Args:
            data_id: Data ID(s) for the batch.
            input_ids_ref: Ray ObjectRef or list of input_ids tensors.
            packed_loss_mask_list: List of packed loss_mask strings.
            formatted_prompts: List of chat-template-formatted prompt strings.
            return_last_hidden_states: Whether to return last hidden states.
            return_logits: Whether to return target logits.
            multimodal_inputs: List of multimodal input dicts.

        Returns:
            List of dicts with mooncake_key or tensors.
        """

    def get_mooncake_config(self):
        """Get the mooncake configuration.

        Subclasses must set ``self._mooncake_config`` in their ``__init__``.
        """
        return getattr(self, "_mooncake_config", None)

    @abstractmethod
    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the engine is healthy."""

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""

    @abstractmethod
    def get_status(self) -> dict:
        """Get engine status."""
