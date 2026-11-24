"""Attentive hidden-states correction for dsv41_mla drafters, reading the context in
DeepSeek-V4.1's latent MQA. ``wo_b`` starts at zero, so a fresh drafter's output is unchanged."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from angelspec.models.draft.attentive_correction import (
    MAX_BLOCK_SLOTS,
    AttentiveCorrection,
)
from angelspec.models.draft.dsv41_mla import (
    DSV41RMSNorm,
    apply_rope,
    grouped_output,
    listed_latent_attention,
    resolve_dsa_backend,
    rope_cos_sin,
)


class DSV41AttentiveCorrection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.rope_dim = int(config.qk_rope_head_dim)
        self.q_lora_rank = int(config.q_lora_rank)
        self.n_groups = int(config.o_groups)
        self.o_lora_rank = int(config.o_lora_rank)
        self.scaling = self.head_dim**-0.5
        eps = config.rms_norm_eps
        self.rope_theta = float(config.rope_theta or 10000.0)
        self.rope_yarn = dict(
            original_seq_len=int(getattr(config, "original_seq_len", 0) or 0),
            factor=float(getattr(config, "rope_factor", 1.0) or 1.0),
            beta_fast=float(getattr(config, "beta_fast", 32)),
            beta_slow=float(getattr(config, "beta_slow", 1)),
        )

        self.q_in = nn.Linear(2 * self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DSV41RMSNorm(self.q_lora_rank, eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads))
        self.wo_a = nn.Linear(
            self.num_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.hidden_size, bias=False)
        nn.init.zeros_(self.wo_b.weight)
        self.block_keys = bool(getattr(config, "attentive_correction_block_keys", False))
        self.block_gated = (
            self.block_keys
            and getattr(config, "attentive_correction_block_mode", "shared") == "gated"
        )
        self.block_kv_in_embed = (
            getattr(config, "attentive_correction_block_kv_in", "hidden_embed") == "hidden_embed"
        )
        self.block_kv_b_separate = False
        if self.block_keys:
            self.kv_in = nn.Linear(
                (2 if self.block_kv_in_embed else 1) * self.hidden_size, self.head_dim, bias=False
            )
            self.kv_norm = DSV41RMSNorm(self.head_dim, eps)
        if self.block_gated:
            self.block_sink = nn.Parameter(torch.zeros(self.num_heads))
            # Logit bias by how far back the key is (0 = the slot itself).
            self.block_rel_bias = nn.Parameter(torch.zeros(self.num_heads, MAX_BLOCK_SLOTS))
            self.block_gate = nn.Linear(2 * self.hidden_size, self.num_heads, bias=False)
            nn.init.zeros_(self.block_gate.weight)
            self.block_gate_bias = nn.Parameter(torch.zeros(self.num_heads, MAX_BLOCK_SLOTS))

        self.backend = getattr(
            config, "attentive_correction_backend", None
        ) or resolve_dsa_backend(config)

    def _attend_block_gated(
        self, query: torch.Tensor, block: torch.Tensor, inputs: torch.Tensor, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, slots, heads, _ = query.shape
        groups = slots // block_size
        q = query.float().view(bsz, groups, block_size, heads, -1)
        keys = block.float().view(bsz, groups, block_size, -1)
        logits = torch.einsum("bgqhm,bgkm->bghqk", q, keys) * self.scaling
        steps = torch.arange(block_size, device=query.device)
        back = steps.view(-1, 1) - steps.view(1, -1)
        logits = logits + self.block_rel_bias.float()[:, back.clamp(min=0)]
        logits = logits.masked_fill(back < 0, torch.finfo(logits.dtype).min)
        total = torch.logaddexp(
            torch.logsumexp(logits, dim=-1, keepdim=True),
            self.block_sink.float().view(1, 1, -1, 1, 1),
        )
        read = torch.einsum("bghqk,bgkm->bgqhm", torch.exp(logits - total), keys).reshape(
            bsz, slots, heads, -1
        )
        slot = torch.arange(slots, device=query.device) % block_size
        gate = torch.tanh(
            self.block_gate(inputs).float() + self.block_gate_bias.float()[:, slot].transpose(0, 1)
        )
        return read * gate.unsqueeze(-1), gate

    def forward(
        self,
        hidden_norm: torch.Tensor,
        embed_norm: torch.Tensor,
        position_ids: torch.Tensor,
        latent: torch.Tensor,
        indices: torch.Tensor,
        block_size: Optional[int] = None,
        stats: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The correction and the mask of slots that had a key to read (the others get 0)."""
        lead = hidden_norm.shape[:-1]
        bsz = lead[0]
        hidden_norm = hidden_norm.reshape(bsz, -1, self.hidden_size)
        slots = hidden_norm.shape[1]
        embed_norm = embed_norm.reshape(bsz, slots, self.hidden_size).to(hidden_norm.dtype)
        position_ids = position_ids.reshape(bsz, slots)
        indices = indices.reshape(bsz, slots, indices.shape[-1])

        inputs = torch.cat([hidden_norm, embed_norm], -1)
        cos, sin = rope_cos_sin(position_ids, self.rope_dim, self.rope_theta, **self.rope_yarn)
        q = self.wq_b(self.q_norm(self.q_in(inputs))).unflatten(
            -1, (self.num_heads, self.head_dim)
        )
        q = apply_rope(q, cos.unsqueeze(2), sin.unsqueeze(2))
        block = None
        if self.block_keys:
            if block_size is None or block_size < 1 or slots % block_size:
                raise ValueError(
                    f"block keys need the {slots} slots as whole blocks; got block_size {block_size}"
                )
            if self.block_gated and block_size > MAX_BLOCK_SLOTS:
                raise ValueError(
                    f"the gated block read covers blocks of up to {MAX_BLOCK_SLOTS} slots, "
                    f"got {block_size}"
                )
            block_in = inputs if self.block_kv_in_embed else hidden_norm
            block = apply_rope(self.kv_norm(self.kv_in(block_in)), cos, sin)
        if self.block_keys and not self.block_gated:
            ctx_len = latent.shape[1]
            latent = torch.cat([latent.to(block.dtype), block], dim=1)
            indices = torch.cat(
                [
                    indices,
                    AttentiveCorrection._block_indices(
                        bsz, slots, block_size, ctx_len, indices.device
                    ).to(indices.dtype),
                ],
                dim=-1,
            )
        read = listed_latent_attention(
            q,
            latent.to(q.dtype),
            indices,
            backend="torch" if self.backend == "tilelang" and not q.is_cuda else self.backend,
            scaling=self.scaling,
            sink=self.attn_sink,
        )
        alive = (indices >= 0).any(-1)
        if self.block_gated:
            block_read, gate = self._attend_block_gated(q, block, inputs, block_size)
            read = torch.where(alive[..., None, None], read.float(), 0.0) + block_read
            alive = torch.ones_like(alive)
            if stats is not None:
                with torch.no_grad():
                    per_slot = gate.abs().mean(dim=-1).view(bsz, -1, block_size)
                    stats["block_gate"] = per_slot.sum(dim=(0, 1))
                    stats["block_gate_count"] = torch.full_like(
                        stats["block_gate"], float(bsz * per_slot.shape[1])
                    )
        read = torch.where(alive[..., None, None], read, torch.zeros_like(read)).to(q.dtype)
        out = grouped_output(read, cos, sin, self.wo_a, self.wo_b, self.n_groups)
        return out.reshape(*lead, self.hidden_size), alive.reshape(lead)


def _check_source(module: DSV41AttentiveCorrection, attention: nn.Module) -> None:
    for name, mine in (("wq_b", module.wq_b), ("wo_a", module.wo_a)):
        theirs = getattr(attention, name, None)
        if theirs is None or mine.weight.shape != theirs.weight.shape:
            raise ValueError(
                f"{name}: correction {tuple(mine.weight.shape)} vs source "
                f"{None if theirs is None else tuple(theirs.weight.shape)}"
            )
    wq_a = getattr(attention, "wq_a", None)
    if wq_a is None or tuple(wq_a.weight.shape) != (module.q_lora_rank, module.hidden_size):
        raise ValueError(
            "the source attention needs a wq_a of shape "
            f"{(module.q_lora_rank, module.hidden_size)} for the correction to copy"
        )


@torch.no_grad()
def init_from_attention(module: DSV41AttentiveCorrection, attention: nn.Module) -> None:
    """Copy a dsv41_mla draft layer's attention into ``module``, ``wo_b`` zeroed."""
    _check_source(module, attention)
    hidden = module.hidden_size
    module.q_in.weight.zero_()
    module.q_in.weight[:, :hidden].copy_(attention.wq_a.weight)
    module.q_norm.weight.copy_(attention.q_norm.weight)
    module.wq_b.weight.copy_(attention.wq_b.weight)
    module.attn_sink.copy_(attention.attn_sink)
    module.wo_a.weight.copy_(attention.wo_a.weight)
    module.wo_b.weight.zero_()
    if module.block_keys:
        init_block_keys_from_attention(module, attention)


@torch.no_grad()
def init_block_keys_from_attention(module: DSV41AttentiveCorrection, attention: nn.Module) -> None:
    hidden = module.hidden_size
    wkv = getattr(attention, "wkv", None)
    if wkv is None or tuple(wkv.weight.shape) != (module.head_dim, hidden):
        raise ValueError(
            f"the source attention needs a wkv of shape {(module.head_dim, hidden)} for the "
            "block keys to copy"
        )
    module.kv_in.weight.zero_()
    module.kv_in.weight[:, :hidden].copy_(wkv.weight)
    module.kv_norm.weight.copy_(attention.kv_norm.weight)
    if module.block_gated:
        init_block_gate_from_attention(module, attention)


@torch.no_grad()
def init_block_gate_from_attention(module: DSV41AttentiveCorrection, attention: nn.Module) -> None:
    """Start ``module``'s gated block read closed: ``attention``'s sink and a zero gate."""
    module.block_sink.copy_(attention.attn_sink)
    module.block_rel_bias.zero_()
    module.block_gate.weight.zero_()
    module.block_gate_bias.zero_()
