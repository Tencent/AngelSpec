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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from angelspec.utils.distributed import get_tp_device_mesh, get_tp_group
from angelspec.utils.logging import logger


def _patch_hf_config_compat(config):
    """Reconcile checkpoint config field aliases with transformers expectations.

    Some checkpoints (e.g. hy_v3) ship a config that names the MoE
    expert intermediate dim ``expert_hidden_dim``, while the transformers
    model implementation reads ``moe_intermediate_size``. When the canonical
    field is missing, transformers silently falls back to its default, which
    yields the wrong expert/shared-expert shapes and a load-time size mismatch.
    Copy the alias across so the model is built with the checkpoint's real dim.
    """
    if (
        getattr(config, "moe_intermediate_size", None) is None
        and getattr(config, "expert_hidden_dim", None) is not None
    ):
        config.moe_intermediate_size = config.expert_hidden_dim
        logger.info(
            "HFTargetModel: set moe_intermediate_size=%s from expert_hidden_dim for hy_v3-style config compatibility.",
            config.moe_intermediate_size,
        )
    return config


@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    target: Optional[torch.Tensor] = None
    last_hidden_states: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    # Pre-copied CPU tensor for loss mask computation, avoids GPU→CPU round trip
    # when mooncake uses the TCP/host-buffer path (data already passes through CPU).
    input_ids_cpu: Optional[torch.Tensor] = None

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        tensors = {
            "hidden_states": self.hidden_states,
            "input_ids": self.input_ids,
        }
        if self.loss_mask is not None:
            tensors["loss_mask"] = self.loss_mask
        if self.target is not None:
            tensors["target"] = self.target
        if self.last_hidden_states is not None:
            tensors["last_hidden_states"] = self.last_hidden_states
        if self.input_ids_cpu is not None:
            tensors["input_ids_cpu"] = self.input_ids_cpu
        return tensors


class Eagle3TargetModel(ABC):
    """
    Abstract interface for the target model backend used to generate Eagle3 training data.
    """

    def __init__(self):
        self.aux_hidden_states_layers = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Eagle3TargetModel":
        """
        Initialize the target model backend from a pretrained model path.
        """

    @abstractmethod
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        """
        Generate the eagle3 data from the target model.
        """

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """Set the layers to capture aux hidden states from the target model.

        Generalized to support any number of layers: Eagle3 uses 3, DFlash uses 5.
        When called with None, defaults to 3 Eagle3-style layers for backward compat.
        """
        if aux_hidden_states_layers is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
            else:
                raise ValueError(
                    f"Failed to set aux hidden states layers as model config {self.model.config} does not have num_hidden_layers"
                )
            aux_hidden_states_layers = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        if not self.aux_hidden_states_layers:
            raise ValueError("aux_hidden_states_layers must be a non-empty list")


class HFTargetModel(Eagle3TargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "HFTargetModel":
        """
        Initialize the HuggingFace target model backend from a pretrained model path.
        """
        tp_group = get_tp_group()
        tp_size = tp_group.size() if tp_group is not None else 1

        if tp_size > 1:
            device_kwargs = {
                "tp_plan": "auto",
                "tp_size": tp_size,
                "device_mesh": get_tp_device_mesh(),
            }
        else:
            device_kwargs = {
                "device_map": device or "auto",
            }

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        _patch_hf_config_compat(getattr(config, "text_config", config))

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **device_kwargs,
            **kwargs,
        )
        return cls(target_model)

    def _get_transformer_layers(self):
        """
        Helper to find the module list containing the transformer layers.
        Adapts to common architectures (Llama, Qwen, Mistral, OPT, etc.)
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        else:
            raise ValueError(
                "Could not locate transformer layers in the model architecture to register hooks."
            )

    def _get_final_norm(self) -> nn.Module:
        """
        Helper to find the final norm layer (post-transformer, pre-lm_head).
        Its output is the last hidden states before projection to vocab logits.
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        elif hasattr(self.model, "norm"):
            return self.model.norm
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        else:
            raise ValueError(
                "Could not locate final norm layer in the model architecture to register hook."
            )

    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        """
        Optimized HF backend:
        Instead of returning all hidden states (memory heavy), we use forward hooks
        to capture only the specific layers required by Eagle3.
        """
        captured_states = {}
        last_hidden = {}
        handles = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured_states[layer_idx] = hidden

            return hook

        def norm_hook(module, input, output):
            last_hidden["value"] = output

        layers = self._get_transformer_layers()
        target_indices = self.aux_hidden_states_layers

        for idx in target_indices:
            if 0 <= idx < len(layers):
                handles.append(layers[idx].register_forward_hook(get_hook(idx)))
            else:
                raise ValueError(
                    f"Layer index {idx} out of bounds for model with {len(layers)} layers."
                )

        final_norm = self._get_final_norm()
        handles.append(final_norm.register_forward_hook(norm_hook))

        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                output_router_logits=False,
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        expected = len(target_indices)
        if len(captured_states) != expected:
            raise RuntimeError(
                f"Expected to capture {expected} layers, but captured {len(captured_states)}"
            )
        assert "value" in last_hidden, "Failed to capture last hidden states from final norm layer"

        device = input_ids.device
        hidden_states = torch.cat(
            [captured_states[idx].to(device) for idx in target_indices], dim=-1
        )

        loss_mask = loss_mask[..., None].to(device)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            last_hidden_states=last_hidden["value"].to(device),
            loss_mask=loss_mask,
            input_ids=input_ids,
        )
