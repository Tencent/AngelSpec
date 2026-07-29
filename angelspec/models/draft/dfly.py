"""DFlareV2 (``dfly``) draft model.

DFlareV2 combines DFlash's shared FC context with DFlare's per-draft-layer
target fusion:

    layer_context_i = RMSNorm(FC(concat(target_hidden)) + fusion_i(target_hidden))

The resulting context has shape ``[B, S, hidden_size]`` and is consumed by
DFlash decoder layers, where target context and draft hidden states share the
same K/V projections.

Selected via ``DSparkConfig`` + ``model_arch == "dfly"`` (dispatched in
``AutoEagle3DraftModel.from_config`` / ``DSparkTrainer.init_model``). The
optional hidden-states correction is the shared TreeFlash module from
``dspark.py`` and is applied at train time by the ``DSparkModel`` wrapper's
``_compute_draft_logits`` hook (which reads ``draft_model.hidden_correction``
generically) — this drafter only carries the module.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.draft.dflare import DFlareDraftModel
from angelspec.models.draft.dflash import DFlashDecoderLayer
from angelspec.models.draft.dspark import DSparkConfig, build_hidden_correction


class DFlyDraftModel(DFlareDraftModel):
    """DFlash layers with DFlash FC context plus DFlare fusion residual.

    Rides the DSpark path (``DSparkConfig`` / ``DSparkTrainer`` / ``DSparkModel``)
    so the shared ``hidden_correction`` runs through the wrapper hook; it builds
    no Markov or confidence head (the hooks read those via ``getattr(..., None)``
    and no-op when absent).
    """

    config_class = DSparkConfig

    def __init__(self, config):
        super().__init__(config)

        # DFlare builds layers with separate context/draft K/V projections and
        # deletes DFlash's ``context_proj`` (keeping the reinitialized
        # ``context_norm`` + ``layer_fusion_weights``). DFlareV2 intentionally
        # restores DFlash layers so both sources share the same k_proj/v_proj,
        # and re-adds the FC ``context_proj`` while retaining the DFlare fusion.
        self.layers = nn.ModuleList([DFlashDecoderLayer(config) for _ in range(self.num_layers)])

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        if target_hidden_size != config.hidden_size:
            raise ValueError(
                "DFlareV2 residual fusion requires target_hidden_size == hidden_size, "
                f"got target_hidden_size={target_hidden_size} and hidden_size={config.hidden_size}"
            )

        self.context_proj = nn.Linear(
            self.num_target_layers * target_hidden_size,
            self.hidden_size,
            bias=False,
        )

        # Shared TreeFlash hidden-states correction (dspark.build_hidden_correction);
        # ``None`` when ``enable_hidden_correction`` is unset. Applied by DSparkModel.
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
                f"DFlareV2 context_feature must have shape [B, S, T, D], got {tuple(context_feature.shape)}"
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


__all__ = ["DFlyDraftModel"]
