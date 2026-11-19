"""Frozen copy of one target layer's MLA latent projection, for drafters that read the
target's own KV.

A drafter with ``DFlashConfig.target_kv_layers`` gives draft layer ``i`` the compressed
KV that target layer ``target_kv_layers[i]`` holds for the context, in place of a latent
the drafter projects from its own fused context feature. The draft layer still reads it
through its own ``kv_b_proj`` (and absorbed query), and its block tokens' latents still
come from its own ``kv_a_proj_with_mqa``.

hy4's layer-L MLA caches ``kv_a_layernorm(c)`` and the rotated ``k_pe``, both split from
``kv_a_proj_with_mqa(input_layernorm(h))`` with ``h`` the residual stream entering layer
L: the drafter's captured ``target_layer_id`` L-1 (see target_indexer). This module
recomputes the two from that hidden at the target's positions, rotating in fp32 and
rounding to the weights' dtype as vLLM does. ``k_pe`` comes out in the drafter's
split-half RoPE layout (``_interleaved_to_split_rope`` then ``_apply_rope_split``, as in
``Hy4DraftMLAAttention._project_kv_latent``), so it lines up with the drafter's own
block latents. vLLM's default HYV4 rope leaves it interleaved in the KV cache, so a
comparison against that cache needs the same permutation.

Only the three tensors that path reads are loaded. Like the indexer, the trainer holds
the module outside the FSDP-wrapped drafter, so it is neither trained, sharded nor saved
with the drafter.
"""

from typing import Optional

import torch
import torch.nn as nn

from angelspec.models.draft.llama3_eagle import LlamaRMSNorm
from angelspec.models.draft.mtp import _apply_rope_split, _interleaved_to_split_rope
from angelspec.models.target.target_indexer import (
    _text_config,
    load_layer_tensors,
    load_target_config,
    target_cos_sin,
    target_positions,
    target_rope,
    target_rope_parameters,
)

#: Module parameter -> checkpoint key below ``<prefix>layers.<L>.``.
_CHECKPOINT_KEYS = {
    "input_layernorm.weight": "input_layernorm.weight",
    "kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
}

#: Layer types that run MLA, as ``layer_types`` spells them.
_MLA_LAYER_TYPES = ("deepseek_sparse_attention", "full_attention", "attention")


def _rope_parameters(config: dict, layer: Optional[int]) -> dict:
    """The target's ``rope_parameters`` as target ``layer`` applies them: Hy4 flash nests
    them per layer type (see ``target_rope_parameters``)."""
    rope = _text_config(config).get("rope_parameters") or {}
    if any(isinstance(value, dict) for value in rope.values()):
        return target_rope_parameters(config, layer)
    return rope


def check_target_kv_layer(config: dict, layer: int, field: str = "target_kv_layers") -> None:
    """Raise unless target ``layer`` exists and runs MLA with plain (M-)RoPE.

    ``field`` names the config field asking, for the error messages.
    """
    text = _text_config(config)
    if text.get("kv_lora_rank") is None:
        raise ValueError(f"{field} needs an MLA target, but its config has no kv_lora_rank")
    num_layers = int(text["num_hidden_layers"])
    if not 0 <= layer < num_layers:
        raise ValueError(f"{field} entry {layer} is outside the target's {num_layers} layers")
    layer_types = text.get("layer_types")
    if layer_types is not None and str(layer_types[layer]) not in _MLA_LAYER_TYPES:
        raise ValueError(f"target layer {layer} is {layer_types[layer]!r}: it has no MLA latent")
    # target_rope builds plain (M-)RoPE; a scaled variant would rotate k_pe differently.
    rope_type = _rope_parameters(config, layer).get("rope_type", "default")
    scaling = text.get("rope_scaling") or {}
    scaling_type = scaling.get("rope_type", scaling.get("type", "default"))
    if rope_type != "default" or scaling_type not in ("default", "mrope"):
        raise ValueError(
            f"{field} assumes the target's plain RoPE, but the target uses "
            f"rope_type={rope_type!r} / rope_scaling={scaling or None!r}"
        )


def check_target_kv_compatible(
    config: dict, draft_config, field: str = "target_kv_layers", layer: Optional[int] = None
) -> None:
    """Raise unless the drafter can read target ``layer``'s latents as its own.

    The drafter rotates its queries with its own RoPE and expands the latent with its
    own ``kv_b_proj``, so the latent widths and the whole RoPE setup must match.
    ``field`` names the config field asking, for the error message; ``layer`` is needed
    when the target's RoPE differs per layer type (Hy4 flash).
    """
    text = _text_config(config)
    rope = _rope_parameters(config, layer)
    target_mrope = rope.get("mrope_section")
    pairs = {
        "kv_lora_rank": (text.get("kv_lora_rank"), draft_config.kv_lora_rank),
        "qk_rope_head_dim": (text.get("qk_rope_head_dim"), draft_config.qk_rope_head_dim),
        "rope_theta": (
            float(rope.get("rope_theta", text.get("rope_theta", 10000.0))),
            float(draft_config.rope_theta),
        ),
        "mrope_section": (
            [int(s) for s in target_mrope] if target_mrope is not None else None,
            draft_config.mrope_section,
        ),
    }
    if target_mrope is not None:
        pairs["mrope_interleaved"] = (
            bool(rope.get("mrope_interleaved", False)),
            bool(draft_config.mrope_interleaved),
        )
        for key in ("image_start_token_id", "image_end_token_id", "image_newline_token_id"):
            pairs[key] = (config.get(key, text.get(key)), getattr(draft_config, key, None))
    mismatched = [
        f"{name}: target {target!r} vs drafter {draft!r}"
        for name, (target, draft) in pairs.items()
        if target != draft
    ]
    if mismatched:
        raise ValueError(
            f"{field} needs the drafter's latent widths and RoPE to be the target's: "
            + "; ".join(mismatched)
        )


class TargetKVLatent(nn.Module):
    """Target layer ``layer``'s per-token MLA latent ``[kv_a_layernorm(c) | rotated k_pe]``."""

    def __init__(self, config: dict, layer: int):
        super().__init__()
        check_target_kv_layer(config, layer)
        text = _text_config(config)
        hidden_size = int(text["hidden_size"])
        eps = float(text["rms_norm_eps"])
        self.layer = int(layer)
        self.kv_lora_rank = int(text["kv_lora_rank"])
        self.qk_rope_head_dim = int(text["qk_rope_head_dim"])

        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=eps)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = LlamaRMSNorm(self.kv_lora_rank, eps=eps)
        # Kept fp32 whatever dtype the weights take (see _cast_weights), as vLLM
        # rotates k_pe in fp32.
        self.rotary_emb, self._position_recipe = target_rope(
            config, self.qk_rope_head_dim, self.layer
        )

        if set(dict(self.named_parameters())) != set(_CHECKPOINT_KEYS):
            raise RuntimeError("TargetKVLatent parameters drifted from the checkpoint key map")
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_config(
        cls, model_path: str, layer: int, device="cuda", dtype: torch.dtype = torch.bfloat16
    ) -> "TargetKVLatent":
        """Uninitialized weights, e.g. on ranks that receive them by broadcast."""
        module = cls(load_target_config(model_path), layer)
        module._cast_weights(dtype)
        return module.to(device)

    @classmethod
    def from_pretrained(
        cls, model_path: str, layer: int, device="cuda", dtype: torch.dtype = torch.bfloat16
    ) -> "TargetKVLatent":
        module = cls(load_target_config(model_path), layer)
        module.load_weights(model_path)
        module._cast_weights(dtype)
        return module.to(device)

    def _cast_weights(self, dtype: torch.dtype) -> None:
        # Module-wise, so the rotary tables (non-persistent buffers) stay fp32.
        for module in (self.input_layernorm, self.kv_a_proj_with_mqa, self.kv_a_layernorm):
            module.to(dtype)

    def load_weights(self, model_path: str) -> None:
        """Copy the three tensors this module needs from a safetensors checkpoint."""
        load_layer_tensors(
            self, model_path, self.layer, _CHECKPOINT_KEYS, "kv_a_proj_with_mqa.weight"
        )

    @torch.no_grad()
    def latents(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[B, L, kv_lora_rank + qk_rope_head_dim]`` latents, ``k_pe`` in split layout.

        Args:
            hidden: ``[B, L, D]`` captured input of target layer ``layer``'s
                ``input_layernorm`` (the drafter's ``target_layer_id`` ``layer - 1``).
            input_ids: ``[B, L]`` token ids, for the target's (M-RoPE) positions.
            ctx_doc_ids: ``[B, L]`` packed-document ids, or ``None``.
            position_ids: ``[B, 3, L]`` prepared M-RoPE positions, or ``None`` to
                build them from ``input_ids``. Ignored by a 1D-RoPE target.
        """
        positions = target_positions(input_ids, self._position_recipe, ctx_doc_ids, position_ids)
        cos, sin = target_cos_sin(
            self.rotary_emb, self._position_recipe, positions, hidden.shape[1]
        )
        normed = self.input_layernorm(hidden.to(self.kv_a_proj_with_mqa.weight.dtype))
        latent, k_pe = torch.split(
            self.kv_a_proj_with_mqa(normed), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = _apply_rope_split(_interleaved_to_split_rope(k_pe.unsqueeze(1)), cos, sin)
        return torch.cat([self.kv_a_layernorm(latent), k_pe.squeeze(1)], dim=-1)
