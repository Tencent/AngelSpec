"""DFly draft model: DFlash shared-FC context + DFlare per-layer fusion residual.

    layer_context_i = RMSNorm(FC(concat(target_hidden)) + fusion_i(target_hidden))

Selected via ``DFlyConfig`` (architecture ``"Qwen3DFlyModel"``). The hidden-states
correction (TreeFlash formula (1)) is a DFlash-family module that DFly turns on by
default; it is applied at train time by the ``DFlashModel`` wrapper's
``_compute_draft_logits`` hook.

``layer_arch`` (a DFlash-family field, see :func:`~.dflash.build_draft_layers`)
picks the decoder-layer body: ``"dflash"`` (default) is the dense SwiGLU layer
with a plain residual. Within it, ``attention_type`` picks DFlash's GQA block
(default) or a target family's MLA from
:mod:`angelspec.models.draft.attention_registry`.
"""

# The hidden correction and the decoder-layer bodies (``layer_arch``) live with
# the DFlash family — every drafter in it can carry them. They are re-exported
# here because DFly is the path that defaults the correction on and ships the hy4
# layer variant as its own config class.
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.draft.dflare import DFlareDraftModel
from angelspec.models.draft.dflash import (
    DFlashConfig,
    HiddenStatesCorrection,
    build_draft_layers,
    build_hidden_correction,
)


class DFlyConfig(DFlashConfig):
    """DFly config: :class:`DFlashConfig` with the hidden correction on by default.

    Every field is inherited, ``layer_arch`` and the hy4 iHC / MoE / attention
    fields it selects included. The correction is what DFly adds over DFlare, so
    the only difference is that ``enable_hidden_correction`` defaults to ``True``.
    """

    model_type = "qwen3"

    def __init__(self, **kwargs):
        kwargs.setdefault("enable_hidden_correction", True)
        super().__init__(**kwargs)


class DFlyDraftModel(DFlareDraftModel):
    """DFlash decoder layers + DFlash FC context + DFlare fusion residual.

    Inherits the ``hidden_correction`` module from ``DFlashDraftModel`` (on by
    default here), applied by the training wrapper's ``_compute_draft_logits`` hook.

    When hy4's ``indexer_types`` schedule shares DSA indexers between layers,
    they are held in ``indexers`` here (see
    :func:`~angelspec.models.draft.hy4_mla.build_hy4_self_attns`).
    """

    config_class = DFlyConfig

    def __init__(self, config):
        super().__init__(config)
        self.context_chunk_size = 0
        self.context_checkpoint = False

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

    def _build_layers(self, config):
        """Back to the DFlash-family layer stack, per ``layer_arch``.

        DFly keeps DFlash's shared k_proj/v_proj for context and draft — DFlare's
        separate context projections are the one thing it does not inherit — while
        the fusion residual (``context_norm`` + ``layer_fusion_weights``) stays.
        """
        return build_draft_layers(config, self.num_layers)

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
        full_ctx_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the draft model with FC + fusion-residual context per layer.

        ``full_ctx_len`` is the global context length under USP (local shard in
        ``context_feature``); forwarded into each decoder layer's attention.
        """
        if context_feature.ndim != 4:
            raise ValueError(
                f"DFly context_feature must have shape [B, S, T, D], got {tuple(context_feature.shape)}"
            )
        if context_feature.shape[2] != self.num_target_layers:
            raise ValueError(
                f"Expected {self.num_target_layers} target layers, got {context_feature.shape[2]}"
            )

        from angelspec.models.ops.memory_chunks import chunked_context_call

        base_context = chunked_context_call(
            self._project_base_context,
            (context_feature,),
            self.context_chunk_size,
            self.context_checkpoint and self.training,
        )
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(base_context.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(base_context.dtype)

        for i, layer in enumerate(self.layers):
            layer_context = chunked_context_call(
                partial(self._build_layer_context, layer_idx=i),
                (context_feature, base_context),
                self.context_chunk_size,
                self.context_checkpoint and self.training,
            )
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=layer_context,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
                full_ctx_len=full_ctx_len,
            )

        return self.final_norm(draft_hidden)


__all__ = [
    "DFlyConfig",
    "DFlyDraftModel",
    "HiddenStatesCorrection",
    "build_hidden_correction",
]
