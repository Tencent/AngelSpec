"""DFly draft model: DFlash shared-FC context + DFlare per-layer fusion residual.

    layer_context_i = RMSNorm(FC(concat(target_hidden)) + fusion_i(target_hidden))

Selected via ``DFlyConfig`` (architecture ``"Qwen3DFlyModel"``). The optional
hidden-states correction (TreeFlash formula (1)) is applied at train time by the
``DFlyModel`` wrapper's ``_compute_draft_logits`` hook.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.draft.dflare import DFlareDraftModel
from angelspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDecoderLayer,
    DFlashRMSNorm,
)


class DFlyConfig(DFlashConfig):
    """DFly config: :class:`DFlashConfig` plus TreeFlash hidden-correction knobs."""

    model_type = "qwen3"

    def __init__(
        self,
        enable_hidden_correction: bool = True,
        hidden_correction_intermediate_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_hidden_correction = enable_hidden_correction
        self.hidden_correction_intermediate_size = hidden_correction_intermediate_size


class HiddenStatesCorrection(nn.Module):
    """TreeFlash hidden-states correction (formula (1)), residual + zero-init::

        h'_{t+i} = h_{t+i} + SwiGLU( norm(h_{t+i}) :: norm(e_{t+i-1}) )

    ``e_{t+i-1}`` is the previous (teacher-forced) token embedding, ``::`` is
    feature-dim concat. The output projection is zero-initialized, so the
    correction starts at 0 and the model degenerates back to DFlash.
    """

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embed_size = int(embed_size)
        self.intermediate_size = int(intermediate_size)

        # Normalize each stream independently before concat so the hidden state
        # and the differently-scaled token embedding contribute comparably.
        self.hidden_norm = DFlashRMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.embed_norm = DFlashRMSNorm(self.embed_size, eps=rms_norm_eps)

        in_features = self.hidden_size + self.embed_size
        self.gate_proj = nn.Linear(in_features, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Residual zero-init: correction starts at 0 -> exactly recovers DFlash.
        nn.init.zeros_(self.down_proj.weight)

    def forward(
        self, hidden_states: torch.Tensor, prev_token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Apply the residual correction.

        Args:
            hidden_states: ``[..., hidden_size]`` drafter output hidden states.
            prev_token_embeds: ``[..., embed_size]`` previous-token embeddings
                (same leading shape as ``hidden_states``).

        Returns:
            Corrected hidden states, same shape/dtype as ``hidden_states``.
        """
        h_norm = self.hidden_norm(hidden_states)
        e_norm = self.embed_norm(prev_token_embeds.to(hidden_states.dtype))
        gate_in = torch.cat([h_norm, e_norm], dim=-1)
        delta = self.down_proj(F.silu(self.gate_proj(gate_in)) * self.up_proj(gate_in))
        delta = delta.to(hidden_states.dtype)

        return hidden_states + delta


def build_hidden_correction(config) -> Optional[nn.Module]:
    """Build the hidden-states correction module, or ``None`` if disabled.

    The previous-token embedding dim equals the draft ``hidden_size`` (the
    drafter reuses the target token embedding). The SwiGLU intermediate width
    defaults to ``hidden_size`` unless ``hidden_correction_intermediate_size``
    is set.
    """
    if not bool(getattr(config, "enable_hidden_correction", False)):
        return None

    hidden_size = int(config.hidden_size)
    intermediate = getattr(config, "hidden_correction_intermediate_size", None)
    intermediate = int(intermediate) if intermediate else hidden_size
    return HiddenStatesCorrection(
        hidden_size=hidden_size,
        embed_size=hidden_size,
        intermediate_size=intermediate,
        rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
    )


class DFlyDraftModel(DFlareDraftModel):
    """DFlash decoder layers + DFlash FC context + DFlare fusion residual.

    Carries the optional ``hidden_correction`` module, applied by the
    ``DFlyModel`` wrapper via the shared ``_compute_draft_logits`` hook.
    """

    config_class = DFlyConfig

    def __init__(self, config):
        super().__init__(config)

        # Restore DFlash layers so context and draft share k_proj/v_proj, while
        # keeping DFlare's per-layer fusion (context_norm + layer_fusion_weights).
        self.layers = nn.ModuleList([DFlashDecoderLayer(config) for _ in range(self.num_layers)])

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        if target_hidden_size != config.hidden_size:
            raise ValueError(
                "DFly residual fusion requires target_hidden_size == hidden_size, "
                f"got target_hidden_size={target_hidden_size} and hidden_size={config.hidden_size}"
            )

        self.context_proj = nn.Linear(
            self.num_target_layers * target_hidden_size,
            self.hidden_size,
            bias=False,
        )

        # TreeFlash correction (formula (1)); ``None`` unless enabled. Applied by DFlyModel.
        self.hidden_correction = build_hidden_correction(config)

    def _project_base_context(self, context_feature: torch.Tensor) -> torch.Tensor:
        """Apply DFlash's FC to stacked ``[B, S, T, D]`` target features."""
        flat_context = context_feature.flatten(start_dim=2)
        return self.context_proj(flat_context.to(self.context_proj.weight.dtype))

    def _build_layer_context(
        self,
        context_feature: torch.Tensor,
        base_context: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Add layer-wise DFlare fusion as a residual, then RMS-normalize."""
        fusion_probs = F.softmax(self.layer_fusion_weights[layer_idx], dim=-1)
        fusion_probs = fusion_probs.to(base_context.dtype)
        residual_context = torch.einsum(
            "t,bstd->bsd",
            fusion_probs,
            context_feature.to(base_context.dtype),
        )
        return self.context_norm(base_context + residual_context)

    def forward(
        self,
        draft_input_ids: Optional[torch.Tensor],
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the draft model with FC + fusion-residual context per layer."""
        if context_feature.ndim != 4:
            raise ValueError(
                f"DFly context_feature must have shape [B, S, T, D], got {tuple(context_feature.shape)}"
            )
        if context_feature.shape[2] != self.num_target_layers:
            raise ValueError(
                f"Expected {self.num_target_layers} target layers, got {context_feature.shape[2]}"
            )

        base_context = self._project_base_context(context_feature)
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(base_context.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(base_context.dtype)

        for i, layer in enumerate(self.layers):
            layer_context = self._build_layer_context(context_feature, base_context, i)
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=layer_context,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
            )

        return self.final_norm(draft_hidden)


__all__ = ["DFlyConfig", "DFlyDraftModel", "HiddenStatesCorrection", "build_hidden_correction"]
