"""DFlare draft model: DFlash with layer-wise target fusion.

Two structural changes over DFlash; the training stack (DFlashModel wrapper,
DFlashTrainer, anchor sampling, block-causal mask, CE loss) is reused unchanged.

1. Separate context/noise K,V projections (DFlareAttention): target hidden uses
   dedicated k_proj_target / v_proj_target; draft tokens use k_proj / v_proj.
   DFlash shares one k_proj / v_proj for both.

2. Learnable per-draft-layer fusion (DFlareDraftModel): instead of collapsing the
   T captured target layers into one shared context (DFlash's context_proj =
   Linear(T*D, D)), DFlare keeps them un-collapsed and learns a
   layer_fusion_weights matrix [num_draft_layers, T], so each draft layer attends
   to its own softmax-weighted mix of the target layers.

DFlare reuses DFlashConfig and the "DFlashDraftModel" architecture name; the
dflash-vs-dflare choice is DFlashConfig.model_arch, read by
DFlashTrainer.init_model.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse DFlash's shared building blocks verbatim — only attention and the
# top-level model change for DFlare.
from angelspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDraftModel,
    DFlashMLP,
    DFlashRMSNorm,
    DFlashRotaryEmbedding,
    _repeat_kv,
    _rotate_half,
    build_target_layer_ids,
)


class DFlareAttention(nn.Module):
    """Dual-source KV attention with SEPARATE context/draft projections.

    Identical to ``DFlashAttention`` except context K/V come from dedicated
    ``k_proj_target`` / ``v_proj_target`` rather than the shared ``k_proj`` /
    ``v_proj`` used for draft K/V. The concat -> k_norm -> RoPE flow is unchanged.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # Noise (draft) K/V projections.
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        # Context (target hidden) K/V projections — DFlare-specific.
        self.k_proj_target = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj_target = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Q-norm and K-norm (Qwen3 architecture requirement)
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = DFlashRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
        )

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        bsz, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]

        # Q only from draft
        q = self.q_proj(draft_hidden)
        q = q.view(bsz, draft_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, num_heads, draft_len, head_dim]

        # K/V from both sources, SEPARATE projections (DFlare change).
        k_ctx = self.k_proj_target(context_hidden)
        v_ctx = self.v_proj_target(context_hidden)
        k_draft = self.k_proj(draft_hidden)
        v_draft = self.v_proj(draft_hidden)

        # Concatenate K and V along sequence dimension BEFORE normalization
        # (matches DFlash: concat -> K-norm -> RoPE).
        k = torch.cat([k_ctx, k_draft], dim=1)  # [B, ctx+draft, kv_dim]
        v = torch.cat([v_ctx, v_draft], dim=1)

        total_len = ctx_len + draft_len
        k = k.view(bsz, total_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # [B, num_kv_heads, ctx+draft, head_dim]
        v = v.view(bsz, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE with concatenated position IDs (context + draft)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        cos, sin = self.rotary_emb(q, seq_len=total_len)
        cos = cos.to(q.device)
        sin = sin.to(q.device)

        # Apply RoPE to Q (draft positions only)
        cos_q = cos.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
        sin_q = sin.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)

        # Apply RoPE to K (full concatenated positions)
        cos_k = cos.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(1)
        sin_k = sin.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(1)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

        if block_mask is not None:
            from angelspec.models.ops.flex_attention import compile_friendly_flex_attention

            attn_output = compile_friendly_flex_attention(
                query=q,
                key=k,
                value=v,
                block_mask=block_mask,
                enable_gqa=True,
            )
        else:
            k = _repeat_kv(k, self.num_kv_groups)
            v = _repeat_kv(v, self.num_kv_groups)
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, draft_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class DFlareDecoderLayer(nn.Module):
    """Single decoder layer for DFlare. Identical structure to DFlash's, but the
    ``context_hidden`` passed in differs per layer (see DFlareDraftModel.forward).
    """

    def __init__(self, config):
        super().__init__()
        self.self_attn = DFlareAttention(config)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        residual = draft_hidden
        draft_hidden = self.input_layernorm(draft_hidden)
        draft_hidden = self.self_attn(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
        )
        draft_hidden = residual + draft_hidden

        residual = draft_hidden
        draft_hidden = self.post_attention_layernorm(draft_hidden)
        draft_hidden = self.mlp(draft_hidden)
        draft_hidden = residual + draft_hidden
        return draft_hidden


class DFlareDraftModel(DFlashDraftModel):
    """DFlare draft model with learnable per-layer target fusion.

    Reuses ``DFlashConfig`` (config_class) and DFlash's embedding/lm_head plumbing
    (load_embedding/freeze_embedding inherited). Two overrides vs DFlash:
      * ``layer_fusion_weights[num_draft_layers, T]`` replaces ``context_proj``.
      * ``extract_context_feature`` stacks (no projection); ``forward`` fuses
        per-layer.
    """

    config_class = DFlashConfig

    def __init__(self, config):
        # DFlashDraftModel.__init__ builds context_proj / context_norm / layers
        # (as DFlashDecoderLayer). We rebuild layers as DFlareDecoderLayer and
        # swap the fusion mechanism below.
        super().__init__(config)

        # Replace DFlash decoder layers with DFlare layers (separate ctx/draft KV).
        self.layers = nn.ModuleList([DFlareDecoderLayer(config) for _ in range(self.num_layers)])

        # Drop the DFlash single-context projection; DFlare fuses per draft layer.
        # (Removing the attributes keeps the state_dict clean and avoids dead
        # params being sharded/optimized.)
        if hasattr(self, "context_proj"):
            del self.context_proj
        if hasattr(self, "context_norm"):
            del self.context_norm

        # Learnable fusion logits: one softmax distribution over the T target
        # layers per draft layer. [num_draft_layers, num_target_layers]
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(self.num_layers, self.num_target_layers)
        )
        # Normalize fused context per draft layer before it enters attention
        # (replaces DFlash's context_norm, but applied to the per-layer fusion).
        self.context_norm = DFlashRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self._init_fusion_weights()

    def _init_fusion_weights(self) -> None:
        """Distance-based prior: each draft layer initially prefers the target
        layer at the matching relative depth, with a soft (not one-hot) falloff.

        For draft layer d (of D) the preferred target index is
        ``t* = round(d/(D-1) * (T-1))``; logits decay linearly with |t - t*| so
        softmax peaks at t* but still spreads mass to neighbours. This breaks the
        symmetry that a zeros-init (uniform 1/T == average pooling) would impose,
        giving training a structured shallow->shallow / deep->deep starting point.
        """
        D, T = self.num_layers, self.num_target_layers
        alpha = 2.0  # peak sharpness: larger -> closer to one-hot
        with torch.no_grad():
            for d in range(D):
                t_pref = round(d / max(D - 1, 1) * (T - 1))
                dist = torch.arange(T, dtype=torch.float32)
                self.layer_fusion_weights[d].copy_(-alpha * (dist - t_pref).abs())

    def extract_context_feature(self, all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Stack the T target-layer hidden states WITHOUT collapsing them.

        Returns ``[B, seq_len, T, hidden_size]``; per-layer fusion happens in
        ``forward``. (DFlash instead projects to a single ``[B, seq_len, D]``.)
        The training wrapper (DFlashModel) treats this return value as an opaque
        ``context_feature`` and passes it straight through, so the 4-D shape is
        invisible to it.

        Hidden states arrive from the trainer already in bf16; we keep them as-is
        rather than casting to the fusion-weight dtype, so a non-FSDP fp32-param
        debug run does not silently upcast (and bloat) the context tensor.
        """
        return torch.stack(all_hidden_states, dim=2)  # [B, seq, T, D]

    def forward(
        self,
        draft_input_ids: Optional[torch.Tensor],
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with per-layer target fusion.

        Args:
            context_feature: [B, ctx_len, T, D] — stacked target hidden states
                (output of ``extract_context_feature``).
        """
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_feature.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(context_feature.dtype)

        # Softmax over target layers, per draft layer: [num_layers, T].
        fusion_probs = F.softmax(self.layer_fusion_weights, dim=-1)

        for i, layer in enumerate(self.layers):
            # Weighted combination of the T target layers for draft layer i:
            # [T] x [B, ctx_len, T, D] -> [B, ctx_len, D]
            fused_context = torch.einsum("t,bstd->bsd", fusion_probs[i], context_feature)
            fused_context = self.context_norm(fused_context)
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=fused_context,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
            )

        return self.final_norm(draft_hidden)


__all__ = [
    "DFlareAttention",
    "DFlareDecoderLayer",
    "DFlareDraftModel",
    "build_target_layer_ids",
]
