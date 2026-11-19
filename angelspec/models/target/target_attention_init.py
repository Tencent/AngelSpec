"""Draft attention initialized from the target's (``DFlashConfig.target_attn_init_layers``).

The drafter's hy4 MLA mirrors the target's name for name, and, with the target's dims,
shape for shape: ``q_a_proj`` / ``q_a_layernorm`` / ``q_b_proj``, ``kv_a_proj_with_mqa`` /
``kv_a_layernorm`` / ``kv_b_proj``, ``o_proj``, ``linear_gate`` and ``learnable_sink_param``
under ``self_attn``, plus the decoder layer's ``input_layernorm``. Copying them makes draft
layer ``i`` start out as target layer ``L``'s attention applied to whatever the drafter
feeds it. Under ``target_kv_layers`` with the same ``L``, that layer then reads ``L``'s
latent with ``L``'s own ``kv_b_proj``, and its block tokens' latents come out of ``L``'s
``kv_a`` path, in the same space as the context's.

The copy runs once, on the rank holding the real weights before FSDP broadcasts them; a
resumed checkpoint or ``draft_init_path`` loaded afterwards overwrites it.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from angelspec.models.draft.llama3_eagle import LlamaRMSNorm
from angelspec.models.target.target_indexer import (
    DSA_LAYER_TYPE,
    _text_config,
    load_layer_tensors,
)

_ANCHOR = "self_attn.kv_b_proj.weight"


def target_attention_keys(layer: nn.Module) -> Dict[str, str]:
    """Draft decoder-layer parameter -> checkpoint key below ``<prefix>layers.<L>.``.

    The layer's ``input_layernorm`` and every attention tensor, leaving out a DSA indexer
    of the drafter's own (``self_attn.indexer``).
    """
    return {
        name: name
        for name, _ in layer.named_parameters()
        if name == "input_layernorm.weight"
        or (name.startswith("self_attn.") and not name.startswith("self_attn.indexer."))
    }


def target_has_post_proj_norm(config: dict, layer: int) -> bool:
    """Whether target ``layer``'s attention ends in Hy4 flash's ``post_proj_norm``: on with
    ``sandwich_norm``, except in DSA layers under ``dsa_skip_attn_post_norm``."""
    text = _text_config(config)
    layer_types = text.get("layer_types")
    skipped = (
        bool(text.get("dsa_skip_attn_post_norm", False))
        and layer_types is not None
        and str(layer_types[layer]) == DSA_LAYER_TYPE
    )
    return bool(text.get("sandwich_norm", False)) and not skipped


def check_target_attn_init_compatible(config: dict, draft_config, layers=()) -> None:
    """Raise unless the drafter's attention has the shapes, norm epsilon and post-projection
    norm of target ``layers``'."""
    text = _text_config(config)
    names = [
        "hidden_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "rms_norm_eps",
        "gated_mla",
        "learnable_sink",
    ]
    if getattr(draft_config, "gated_mla", False):
        names.append("gating_type")
    defaults = {"gated_mla": False, "learnable_sink": False}
    mismatched = [
        f"{name}: target {text.get(name, defaults.get(name))!r} vs drafter "
        f"{getattr(draft_config, name, None)!r}"
        for name in names
        if text.get(name, defaults.get(name)) != getattr(draft_config, name, None)
    ]
    draft_norm = bool(getattr(draft_config, "attn_post_proj_norm", False))
    mismatched += [
        f"attn_post_proj_norm: target layer {layer} {not draft_norm} vs drafter {draft_norm}"
        for layer in dict.fromkeys(layers)
        if target_has_post_proj_norm(config, layer) != draft_norm
    ]
    if mismatched:
        raise ValueError(
            "target_attn_init_layers copies the target's attention into the drafter, so these "
            "must match: " + "; ".join(mismatched)
        )


def init_attention_from_target(draft_model: nn.Module, model_path: str, layers: List[int]) -> None:
    """Copy target layer ``layers[i]``'s ``input_layernorm`` and attention into draft layer ``i``."""
    if len(layers) != len(draft_model.layers):
        raise ValueError(f"{len(layers)} target layers for {len(draft_model.layers)} draft layers")
    for draft_layer, layer in zip(draft_model.layers, layers):
        load_layer_tensors(
            draft_layer, model_path, layer, target_attention_keys(draft_layer), _ANCHOR
        )


class TargetAttentionTeacher(nn.Module):
    """Frozen target layer ``layer``'s ``input_layernorm`` and attention, in the drafter's MLA class.

    The same tensors ``init_attention_from_target`` copies, kept as they are in the
    checkpoint, so the target's attention can be recomputed next to a drafter's
    (``angelspec.models.dflash_attn_diag``). Held outside the FSDP-wrapped drafter.
    """

    def __init__(self, draft_config, layer: int):
        super().__init__()
        from angelspec.models.draft.hy4_mla import Hy4DraftMLAAttention

        self.layer = int(layer)
        self.input_layernorm = LlamaRMSNorm(
            draft_config.hidden_size, eps=draft_config.rms_norm_eps
        )
        self.self_attn = Hy4DraftMLAAttention(draft_config)
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_config(
        cls, draft_config, layer: int, device="cuda", dtype: torch.dtype = torch.bfloat16
    ) -> "TargetAttentionTeacher":
        """Uninitialized weights, e.g. on ranks that receive them by broadcast."""
        module = cls(draft_config, layer)
        module._cast_weights(dtype)
        return module.to(device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        draft_config,
        layer: int,
        device="cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TargetAttentionTeacher":
        module = cls(draft_config, layer)
        load_layer_tensors(module, model_path, layer, target_attention_keys(module), _ANCHOR)
        module._cast_weights(dtype)
        return module.to(device)

    def _cast_weights(self, dtype: torch.dtype) -> None:
        # Projections and norms only: the rotary tables stay fp32, as the target rotates in fp32.
        for module in self.modules():
            if isinstance(module, (nn.Linear, LlamaRMSNorm)):
                module.to(dtype)
