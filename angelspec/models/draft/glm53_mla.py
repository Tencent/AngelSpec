"""GLM-5.3 attention for drafters (``attention_type="glm53_mla"``): NoPE MLA in absorbed form,
multi-query attention over the 512-wide latent. With no positions in the scores, a block's slots
differ only through the mask, so train it with ``dflash_causal_attn``."""

from typing import Optional

import torch
import torch.nn as nn

from angelspec.models.draft.dflash import DFlashRMSNorm
from angelspec.models.ops.dsa_kernels import bind_dsa_kernel
from angelspec.models.ops.latent_attention import (
    flex_sparse_latent_attention,
    gather_context_latent,
    grouped_sparse_latent_attention,
    latent_attention,
)
from angelspec.models.ops.sparse_mla import torch_sparse_mla
from angelspec.models.ops.tilelang_dsa import DSA_LATENT_DIM, tilelang_dsa_available


def resolve_dsa_backend(config) -> str:
    """Like ``hy4_mla.resolve_dsa_backend``, but "auto" takes the TileLang kernel only for
    target_index_layer's candidates over a latent as wide as the kernel's."""
    requested = str(getattr(config, "dsa_backend", "torch"))
    if requested != "auto":
        return requested
    usable = (
        getattr(config, "target_index_layer", None) is not None
        and int(getattr(config, "kv_lora_rank", 0) or 0) == DSA_LATENT_DIM
    )
    return "tilelang" if usable and tilelang_dsa_available() else "torch"


class Glm53DraftAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, "qk_rope_head_dim", 0):
            raise NotImplementedError("glm53_mla is NoPE, as GLM-5.3 is: set qk_rope_head_dim=0")
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = 0
        self.v_head_dim = config.v_head_dim
        self.q_a_proj = nn.Linear(hidden, config.q_lora_rank, bias=False)
        self.q_a_layernorm = DFlashRMSNorm(config.q_lora_rank, eps=eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(hidden, self.kv_lora_rank, bias=False)
        self.kv_a_layernorm = DFlashRMSNorm(self.kv_lora_rank, eps=eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)
        self.scaling = self.qk_nope_head_dim**-0.5
        self.dsa_backend = resolve_dsa_backend(config)
        self.dsa_kernel = bind_dsa_kernel(self.dsa_backend, config)

    def muon_head_rows(self):
        return {"q_b_proj.weight": self.qk_nope_head_dim}

    def _latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))

    def _attend_sparse(
        self, q_latent: torch.Tensor, latent: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Attention over each row's listed keys of ``latent``; a row with no key gives zero."""
        if self.dsa_backend == "grouped":
            return grouped_sparse_latent_attention(q_latent, latent, indices, scaling=self.scaling)
        if self.dsa_backend == "flex_attention":
            return flex_sparse_latent_attention(q_latent, latent, indices, scaling=self.scaling)
        if self.dsa_kernel is None:
            return torch_sparse_mla(
                q_latent, latent, indices, scaling=self.scaling, d_v=self.kv_lora_rank
            )[0]
        from angelspec.models.draft.hy4_mla import _finalize_sparse_indices

        keys, alive = _finalize_sparse_indices(indices, self.dsa_kernel.index_alignment)
        out = self.dsa_kernel.sparse_mla(
            q_latent.to(self.dsa_kernel.dtype),
            latent.unsqueeze(2).to(self.dsa_kernel.dtype),
            keys,
            scaling=self.scaling,
        )[0]
        return torch.where(alive[..., None, None], out, torch.zeros_like(out)).to(q_latent.dtype)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: Optional[torch.Tensor],
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        full_ctx_len: Optional[int] = None,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        context_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``context_latent`` (target_kv_layers) replaces the latents of ``context_hidden``."""
        if self.dsa_kernel is not None and sparse_kv_indices is None:
            raise ValueError(
                f"on {self.dsa_backend} this layer reads the keys target_index_layer lists, but "
                "no sparse_kv_indices were passed"
            )
        bsz, q_len, _ = draft_hidden.shape
        if context_latent is None:
            ctx_latent = gather_context_latent(self._latent(context_hidden), full_ctx_len)
        elif full_ctx_len is not None and int(full_ctx_len) != context_latent.shape[1]:
            raise NotImplementedError("context_latent (target_kv_layers) does not support USP")
        else:
            ctx_latent = context_latent
        draft_latent = self._latent(draft_hidden)
        latent = torch.cat([ctx_latent.to(draft_latent.dtype), draft_latent], dim=1)

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(draft_hidden)))
        q = q.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
        weight = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        w_uk, w_uv = weight[:, : self.qk_nope_head_dim], weight[:, self.qk_nope_head_dim :]
        q_latent = torch.einsum("bqhd,hdc->bhqc", q, w_uk)
        if sparse_kv_indices is None:
            out = latent_attention(q_latent, latent, block_mask, self.scaling)
        else:
            out = self._attend_sparse(q_latent.transpose(1, 2), latent, sparse_kv_indices)
            out = out.transpose(1, 2)
        out = torch.einsum("bhqc,hvc->bqhv", out, w_uv)
        return self.o_proj(out.reshape(bsz, q_len, self.num_heads * self.v_head_dim))


def build_glm53_draft_attention(config, layer_idx: int) -> Glm53DraftAttention:
    return Glm53DraftAttention(config)
