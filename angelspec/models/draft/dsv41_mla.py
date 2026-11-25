"""DeepSeek-V4.1 attention for drafters (``attention_type="dsv41_mla"``): latent MQA with one
512-wide latent per token as every head's key and value, per-head sinks and a grouped output."""

import math
from typing import Optional

import torch
import torch.nn as nn

from angelspec.models.ops.dsa_kernels import BoundDSAKernel, bind_dsa_kernel
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
        and int(getattr(config, "head_dim", 0) or 0) == DSA_LATENT_DIM
    )
    return "tilelang" if usable and tilelang_dsa_available() else "torch"


class DSV41RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * h).to(x.dtype)


def rope_inv_freq(
    rope_dim: int,
    theta: float,
    device=None,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32,
    beta_slow: float = 1,
) -> torch.Tensor:
    """DeepSeek-V4's YaRN frequencies (plain RoPE if ``original_seq_len`` <= 0), op for op as
    vLLM's ``_compute_inv_freq`` so the target latents a drafter reads match bit for bit."""
    pos_freqs = theta ** (
        torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim
    )
    extrapolation = 1.0 / pos_freqs
    if original_seq_len <= 0:
        return extrapolation
    interpolation = 1.0 / (factor * pos_freqs)

    def pair_turning(rotations: float) -> float:
        return (
            rope_dim
            * math.log(original_seq_len / (rotations * 2 * math.pi))
            / (2 * math.log(theta))
        )

    low = max(math.floor(pair_turning(beta_fast)), 0)
    high = min(math.ceil(pair_turning(beta_slow)), rope_dim - 1)
    if low == high:
        high += 0.001
    ramp = (
        (torch.arange(rope_dim // 2, device=device, dtype=torch.float32) - low) / (high - low)
    ).clamp(0, 1)
    keep = 1 - ramp
    return interpolation * (1 - keep) + extrapolation * keep


def rope_cos_sin(positions: torch.Tensor, rope_dim: int, theta: float, **yarn):
    """fp32 ``(cos, sin)``, one rounded product per angle as in vLLM's table."""
    angles = positions.float().unsqueeze(-1) * rope_inv_freq(
        rope_dim, theta, positions.device, **yarn
    )
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False):
    """Rotate the last ``2 * cos.size(-1)`` channels, adjacent pairs as (real, imag)."""
    rope_dim = cos.size(-1) * 2
    tail = x[..., -rope_dim:].float().unflatten(-1, (-1, 2))
    real, imag = tail[..., 0], tail[..., 1]
    if inverse:
        sin = -sin
    rotated = torch.stack([real * cos - imag * sin, real * sin + imag * cos], dim=-1).flatten(-2)
    return torch.cat([x[..., :-rope_dim], rotated.to(x.dtype)], dim=-1)


def sparse_latent_attention(
    q: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float,
    sink: torch.Tensor,
    kernel: Optional[BoundDSAKernel] = None,
) -> torch.Tensor:
    """Per-head-sink attention over each row's listed keys (``-1`` unused); no key gives zero."""
    if kernel is None:
        return torch_sparse_mla(q, latent, indices, scaling=scaling, d_v=q.shape[-1], sink=sink)[0]
    from angelspec.models.draft.hy4_mla import _finalize_sparse_indices

    keys, alive = _finalize_sparse_indices(indices, kernel.index_alignment)
    out = kernel.sparse_mla(
        q.to(kernel.dtype),
        latent.unsqueeze(2).to(kernel.dtype),
        keys,
        scaling=scaling,
        sink=sink,
    )[0]
    return torch.where(alive[..., None, None], out, torch.zeros_like(out)).to(q.dtype)


def listed_latent_attention(
    q: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    backend: str,
    scaling: float,
    sink: torch.Tensor,
    kernel: Optional[BoundDSAKernel] = None,
) -> torch.Tensor:
    if backend == "grouped":
        return grouped_sparse_latent_attention(q, latent, indices, scaling=scaling, sink=sink)
    if backend == "flex_attention":
        return flex_sparse_latent_attention(q, latent, indices, scaling=scaling, sink=sink)
    return sparse_latent_attention(q, latent, indices, scaling=scaling, sink=sink, kernel=kernel)


def grouped_output(
    out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: nn.Linear,
    wo_b: nn.Linear,
    n_groups: int,
) -> torch.Tensor:
    bsz, length = out.shape[:2]
    out = apply_rope(out, cos.unsqueeze(2), sin.unsqueeze(2), inverse=True)
    out = out.view(bsz, length, n_groups, -1)
    weight = wo_a.weight.view(n_groups, wo_a.out_features // n_groups, -1)
    return wo_b(torch.einsum("blgd,grd->blgr", out, weight).flatten(2))


class DSV41Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.rope_theta = float(config.rope_theta or 10000.0)
        self.rope_yarn = dict(
            original_seq_len=int(getattr(config, "original_seq_len", 0) or 0),
            factor=float(getattr(config, "rope_factor", 1.0) or 1.0),
            beta_fast=float(getattr(config, "beta_fast", 32)),
            beta_slow=float(getattr(config, "beta_slow", 1)),
        )
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))
        self.wq_a = nn.Linear(hidden, config.q_lora_rank, bias=False)
        self.q_norm = DSV41RMSNorm(config.q_lora_rank, eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(hidden, self.head_dim, bias=False)
        self.kv_norm = DSV41RMSNorm(self.head_dim, eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, hidden, bias=False)
        self.scaling = self.head_dim**-0.5
        self.dsa_backend = resolve_dsa_backend(config)
        self.dsa_kernel = bind_dsa_kernel(self.dsa_backend, config)

    def muon_head_rows(self):
        """The KV projection is a single head already; only the query's up-projection splits."""
        return {"wq_b.weight": self.head_dim}

    def _rope(self, positions: torch.Tensor):
        return rope_cos_sin(positions, self.rope_dim, self.rope_theta, **self.rope_yarn)

    def _query(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(-1, (self.n_heads, self.head_dim))
        return apply_rope(q, cos.unsqueeze(2), sin.unsqueeze(2))

    def _latent(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return apply_rope(self.kv_norm(self.wkv(x)), cos, sin)

    def _output(self, out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return grouped_output(out, cos, sin, self.wo_a, self.wo_b, self.n_groups)

    def _attend_sparse(
        self, q: torch.Tensor, latent: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        return listed_latent_attention(
            q,
            latent,
            indices,
            backend=self.dsa_backend,
            scaling=self.scaling,
            sink=self.attn_sink,
            kernel=self.dsa_kernel,
        )


class DSV41DraftAttention(DSV41Attention):
    """Under USP each rank rotates its context shard at its own positions before the all-gather."""

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
        draft_cos, draft_sin = self._rope(draft_position_ids)
        if context_latent is None:
            ctx_cos, ctx_sin = self._rope(context_position_ids)
            ctx_latent = gather_context_latent(
                self._latent(context_hidden, ctx_cos, ctx_sin), full_ctx_len
            )
        elif full_ctx_len is not None:
            raise NotImplementedError("context_latent (target_kv_layers) does not support USP")
        else:
            ctx_latent = context_latent
        draft_latent = self._latent(draft_hidden, draft_cos, draft_sin)
        latent = torch.cat([ctx_latent.to(draft_latent.dtype), draft_latent], dim=1)
        q = self._query(draft_hidden, draft_cos, draft_sin)
        if sparse_kv_indices is None:
            out = latent_attention(
                q.transpose(1, 2), latent, block_mask, self.scaling, sink=self.attn_sink
            ).transpose(1, 2)
        else:
            out = self._attend_sparse(q, latent, sparse_kv_indices)
        return self._output(out, draft_cos, draft_sin)


def build_dsv41_draft_attention(config, layer_idx: int) -> DSV41DraftAttention:
    return DSV41DraftAttention(config)
