"""Attentive hidden-states correction (``DFlashConfig.attentive_correction``).

The TreeFlash correction (:class:`~angelspec.models.draft.dflash.HiddenStatesCorrection`)
adjusts slot k's hidden ``h_k`` from it and the previous token's embedding with a SwiGLU,
but cannot go back to the context with that token in hand: after "the value in row 3 is",
the number still has to be read from somewhere. This module adds that read, an MLA attention
over the context candidates of one draft layer (the target's own latent there, under
``target_kv_layers``), whose query is projected from the same two normalized inputs the
SwiGLU sees::

    delta_k += W_o Attn(q = W_qb norm(W_qin [norm(h_k); norm(e_{y_{k-1}})]), latent)

It starts out reading the context like that draft layer's attention (:func:`init_from_attention`
copies the layer's ``q_a_proj`` into the hidden half of ``q_in``, its ``q_a_layernorm``,
``q_b_proj``, ``kv_b_proj`` and sink) with ``o_proj`` at zero, so a fresh drafter's output is
unchanged. No output gate. Teacher-forced in training like the SwiGLU part; serving runs it
slot by slot, the previous token being the one just drafted.

With ``attentive_correction_block_keys`` slot k also reads slots 0..k of its own block, each
projected by ``kv_in`` from the same two inputs (its hidden and the token before it), so the
drafted prefix is there to read; ``kv_in`` starts as the source layer's ``kv_a_proj_with_mqa``
on the hidden half. ``attentive_correction_block_mode`` places them:

- ``shared``: after the context latents in the kernel's key list, like the draft attention's
  block keys, one softmax over both (the prefix draws attention away from the context);
- ``gated``: in a softmax of their own (sink and a learned bias by how far back the key is),
  added to the context read through a per-head gate ``tanh(a[slot] + w . [h; e])`` that starts
  at zero, so the context read stays as it is, a fresh run starts as without the block, and
  each slot learns how much of its prefix to add, per head and from its inputs.

With ``attentive_correction_block_kv_in`` "hidden", ``kv_in`` projects each slot's hidden alone
(starting as the source layer's whole ``kv_a_proj_with_mqa``): the block's keys and values no
longer depend on the drafted tokens, only the query and the gate follow the previous token, as
they do for the context, so a block's keys are fixed once the backbone has run.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from angelspec.models.draft.attention_registry import resolve_draft_dsa_backend
from angelspec.models.draft.hy4_mla import _finalize_sparse_indices
from angelspec.models.draft.llama3_eagle import LlamaRMSNorm, LlamaRotaryEmbedding
from angelspec.models.draft.mtp import _apply_rope_split, _interleaved_to_split_rope
from angelspec.models.ops.dsa_kernels import bind_dsa_kernel
from angelspec.models.ops.latent_attention import (
    flex_sparse_latent_attention,
    grouped_sparse_latent_attention,
)
from angelspec.models.ops.mrope import MRotaryEmbedding

# Slots per block the gated block read's per-slot tables cover.
MAX_BLOCK_SLOTS = 16


class AttentiveCorrection(nn.Module):
    """MLA cross-attention over a draft layer's context latents, queried per slot from the
    slot's normalized hidden and the previous token's normalized embedding."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim ** (-0.5)

        # q_in replaces the source layer's q_a_proj: it reads [norm(h); norm(e)].
        self.q_in = nn.Linear(2 * self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = LlamaRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        nn.init.zeros_(self.o_proj.weight)
        self.block_keys = bool(getattr(config, "attentive_correction_block_keys", False))
        self.block_gated = (
            self.block_keys
            and getattr(config, "attentive_correction_block_mode", "shared") == "gated"
        )
        self.block_kv_in_embed = (
            getattr(config, "attentive_correction_block_kv_in", "hidden_embed") == "hidden_embed"
        )
        if self.block_keys:
            # kv_in replaces the source layer's kv_a_proj_with_mqa for the block's own slots.
            self.kv_in = nn.Linear(
                (2 if self.block_kv_in_embed else 1) * self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
            )
            self.kv_a_layernorm = LlamaRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        if self.block_gated:
            self.block_sink = nn.Parameter(torch.zeros(self.num_heads))
            # Logit bias by how far back the key is (0 = the slot itself).
            self.block_rel_bias = nn.Parameter(torch.zeros(self.num_heads, MAX_BLOCK_SLOTS))
            self.block_gate = nn.Linear(2 * self.hidden_size, self.num_heads, bias=False)
            nn.init.zeros_(self.block_gate.weight)
            self.block_gate_bias = nn.Parameter(torch.zeros(self.num_heads, MAX_BLOCK_SLOTS))
        self.block_kv_b_separate = (
            self.block_gated
            and getattr(config, "attentive_correction_block_kv_b", "shared") == "separate"
        )
        if self.block_kv_b_separate:
            # The block read's own key / value up-projections from the latent, started as
            # kv_b_proj (see init_block_kv_b).
            self.block_kv_b_proj = nn.Linear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
            )
        self.learnable_sink = bool(getattr(config, "learnable_sink", False))
        if self.learnable_sink:
            self.learnable_sink_param = nn.Parameter(
                torch.full(
                    (self.num_heads,),
                    float(getattr(config, "learnable_sink_init", 0.0)),
                    dtype=torch.float32,
                )
            )

        # A NoPE latent (glm53_mla's) enters the TileLang kernel zero-padded, see
        # tilelang_sparse_mla.
        self.backend = getattr(config, "attentive_correction_backend", None) or (
            resolve_draft_dsa_backend(config)
        )
        self.dsa_kernel = bind_dsa_kernel(self.backend, config)
        # The same RoPE as the draft attention's queries (Hy4DraftMLAAttention).
        rope = getattr(config, "rope_parameters", None) or {}
        self.uses_mrope = bool(rope.get("mrope_section"))
        self.rotary_emb = (
            MRotaryEmbedding(self.qk_rope_head_dim, rope)
            if self.uses_mrope
            else LlamaRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=int(getattr(config, "max_position_embeddings", 32768)),
                base=getattr(config, "rope_theta", 10000),
            )
        )

    def _cos_sin(
        self, ref: torch.Tensor, position_ids: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.uses_mrope:
            return self.rotary_emb(ref, position_ids)
        if position_ids.ndim != 2:
            raise ValueError("3D positions require rope_parameters.mrope_section")
        cache_len = max(int(seq_len), int(position_ids.max().item()) + 1)
        cos, sin = self.rotary_emb(ref, seq_len=cache_len)
        return cos.squeeze(1).squeeze(0)[position_ids], sin.squeeze(1).squeeze(0)[position_ids]

    def _kv_b_weights(self, kv_b: Optional[nn.Linear] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """``kv_b``'s (default ``kv_b_proj``) per-head key and value up-projections."""
        weight = (self.kv_b_proj if kv_b is None else kv_b).weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        return torch.split(weight, [self.qk_nope_head_dim, self.v_head_dim], dim=1)

    def _block_latent(
        self, inputs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """``[B, S, 2D]`` slot inputs (``[B, S, D]`` hiddens under block_kv_in "hidden") ->
        ``[B, S, kv_lora_rank + qk_rope_head_dim]`` latents in the context latents' form
        (normalized, RoPE at each slot's position, split layout)."""
        bsz, slots, _ = inputs.shape
        latent, k_pe = torch.split(
            self.kv_in(inputs), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, 1, slots, self.qk_rope_head_dim)
        k_pe = _apply_rope_split(_interleaved_to_split_rope(k_pe), cos, sin)
        return torch.cat([self.kv_a_layernorm(latent), k_pe.squeeze(1)], dim=-1)

    @staticmethod
    def _block_indices(
        bsz: int, slots: int, block_size: int, offset: int, device: torch.device
    ) -> torch.Tensor:
        """``[B, S, block_size]``: row r reads rows ``block_start(r) .. r`` of its block,
        shifted by ``offset``; ``-1`` past r."""
        rows = torch.arange(slots, device=device)
        keys = (rows - rows % block_size).unsqueeze(-1) + torch.arange(block_size, device=device)
        keys = torch.where(keys <= rows.unsqueeze(-1), keys + offset, torch.full_like(keys, -1))
        return keys.unsqueeze(0).expand(bsz, -1, -1)

    def _attend_block_gated(
        self, query: torch.Tensor, block: torch.Tensor, inputs: torch.Tensor, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The gated block read: ``[B, S, H, kv_lora_rank]``, each slot's read of its block up
        to itself already scaled by its gate, and the ``[B, S, H]`` gates."""
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
        read = torch.einsum(
            "bghqk,bgkm->bgqhm", torch.exp(logits - total), keys[..., : self.kv_lora_rank]
        ).reshape(bsz, slots, heads, -1)
        slot = torch.arange(slots, device=query.device) % block_size
        gate = torch.tanh(
            self.block_gate(inputs).float() + self.block_gate_bias.float()[:, slot].transpose(0, 1)
        )
        return read * gate.unsqueeze(-1), gate

    def _attend_torch(
        self, query: torch.Tensor, latent: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense reference over the listed keys: ``[B, S, H, kv_lora_rank]`` and the live rows."""
        bsz, slots = query.shape[:2]
        ctx_len = latent.shape[1]
        keep = torch.zeros(bsz, slots, ctx_len + 1, dtype=torch.bool, device=latent.device)
        keep.scatter_(-1, torch.where(indices >= 0, indices.long(), ctx_len), True)
        keep = keep[..., :ctx_len]
        logits = torch.einsum("bshm,bkm->bhsk", query.float(), latent.float()) * self.scaling
        logits = logits.masked_fill(~keep.unsqueeze(1), torch.finfo(logits.dtype).min)
        total = torch.logsumexp(logits, dim=-1, keepdim=True)
        if self.learnable_sink:
            sink = self.learnable_sink_param.float().view(1, -1, 1, 1)
            total = torch.logaddexp(total, sink)
        probs = torch.exp(logits - total)
        read = torch.einsum("bhsk,bkm->bshm", probs, latent[..., : self.kv_lora_rank].float())
        return read, keep.any(-1)

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
        """The attention's share of the correction.

        Args:
            hidden_norm / embed_norm: ``[B, *L, D]`` normalized slot hiddens and
                previous-token embeddings, the SwiGLU's inputs.
            position_ids: the slots' RoPE positions, ``[B, *L]`` or ``[B, 3, *L]``.
            latent: ``[B, ctx_len, kv_lora_rank + qk_rope_head_dim]`` context latents,
                RoPE applied in the split layout.
            indices: ``[B, *L, K]`` keys into ``latent`` per slot, ``-1`` unused.
            block_size: with block keys, the slots (flattened in order) as consecutive
                blocks of this many from slot 0, a slot reading its own block up to itself.
                A block cut short (slots 0..k only) gives slot k what the full block does.
            stats: filled, when given and the block read is gated, with each block slot's
                summed absolute gate (mean over heads) and its slot count.

        Returns ``[B, *L, D]`` and the ``[B, *L]`` mask of slots that had a key to read
        (the others get 0).
        """
        lead = hidden_norm.shape[:-1]
        bsz = lead[0]
        hidden_norm = hidden_norm.reshape(bsz, -1, self.hidden_size)
        slots = hidden_norm.shape[1]
        embed_norm = embed_norm.reshape(bsz, slots, self.hidden_size).to(hidden_norm.dtype)
        mrope = position_ids.dim() == len(lead) + 1
        position_ids = (
            position_ids.reshape(bsz, 3, slots) if mrope else position_ids.reshape(bsz, slots)
        )
        indices = indices.reshape(bsz, slots, indices.shape[-1])

        inputs = torch.cat([hidden_norm, embed_norm], -1)
        q = self.q_b_proj(self.q_a_layernorm(self.q_in(inputs)))
        q = q.view(bsz, slots, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        cos, sin = self._cos_sin(q_nope, position_ids, latent.shape[1])
        q_pe = _apply_rope_split(_interleaved_to_split_rope(q_pe), cos, sin)
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
            block = self._block_latent(inputs if self.block_kv_in_embed else hidden_norm, cos, sin)
        if self.block_keys and not self.block_gated:
            ctx_len = latent.shape[1]
            latent = torch.cat([latent.to(block.dtype), block], dim=1)
            indices = torch.cat(
                [
                    indices,
                    self._block_indices(bsz, slots, block_size, ctx_len, indices.device).to(
                        indices.dtype
                    ),
                ],
                dim=-1,
            )
        w_kc, w_vc = self._kv_b_weights()
        query = torch.cat(
            [torch.einsum("bhsd,hdm->bshm", q_nope, w_kc), q_pe.transpose(1, 2)], dim=-1
        )
        if self.backend in ("grouped", "flex_attention"):
            listed = (
                grouped_sparse_latent_attention
                if self.backend == "grouped"
                else flex_sparse_latent_attention
            )
            read = listed(
                query,
                latent.to(query.dtype),
                indices,
                scaling=self.scaling,
                sink=self.learnable_sink_param if self.learnable_sink else None,
            )[..., : self.kv_lora_rank]
            alive = (indices >= 0).any(-1)
        elif self.dsa_kernel is not None and query.is_cuda:
            keys, alive = _finalize_sparse_indices(indices, self.dsa_kernel.index_alignment)
            read = self.dsa_kernel.sparse_mla(
                query.to(self.dsa_kernel.dtype),
                latent.unsqueeze(2).to(self.dsa_kernel.dtype),
                keys,
                scaling=self.scaling,
                sink=self.learnable_sink_param if self.learnable_sink else None,
            )[0]
        else:
            read, alive = self._attend_torch(query, latent, indices)
        block_out = None
        if self.block_gated:
            block_query = query
            if self.block_kv_b_separate:
                w_kc_b, w_vc_b = self._kv_b_weights(self.block_kv_b_proj)
                block_query = torch.cat(
                    [torch.einsum("bhsd,hdm->bshm", q_nope, w_kc_b), q_pe.transpose(1, 2)],
                    dim=-1,
                )
            block_read, gate = self._attend_block_gated(block_query, block, inputs, block_size)
            read = torch.where(alive[..., None, None], read.float(), 0.0)
            if self.block_kv_b_separate:
                block_out = torch.einsum("bshm,hdm->bshd", block_read.to(w_vc_b.dtype), w_vc_b)
            else:
                read = read + block_read
            alive = torch.ones_like(alive)
            if stats is not None:
                with torch.no_grad():
                    per_slot = gate.abs().mean(dim=-1).view(bsz, -1, block_size)
                    stats["block_gate"] = per_slot.sum(dim=(0, 1))
                    stats["block_gate_count"] = torch.full_like(
                        stats["block_gate"], float(bsz * per_slot.shape[1])
                    )
                    if self.block_kv_b_separate:
                        own = self.kv_b_proj.weight.float()
                        drift = self.block_kv_b_proj.weight.float() - own
                        stats["block_kv_b_drift"] = drift.norm() / own.norm().clamp(min=1e-12)
        out = torch.einsum("bshm,hdm->bshd", read.to(w_vc.dtype), w_vc)
        if block_out is not None:
            out = out + block_out
        out = torch.where(alive[..., None, None], out, torch.zeros_like(out))
        out = self.o_proj(out.reshape(bsz, slots, self.num_heads * self.v_head_dim))
        return out.reshape(*lead, self.hidden_size), alive.reshape(lead)


@torch.no_grad()
def init_from_attention(module: AttentiveCorrection, attention: nn.Module) -> None:
    """Start ``module`` reading the context like ``attention`` (hy4's or glm53's MLA).

    Copies ``q_a_proj`` into the hidden half of ``q_in`` (the previous-token half is zeroed),
    ``q_a_layernorm``, ``q_b_proj``, ``kv_b_proj`` and the sink, and zeroes ``o_proj``; with
    block keys also :func:`init_block_keys_from_attention`.
    """
    hidden = module.hidden_size
    q_a = getattr(attention, "q_a_proj", None)
    if q_a is None or tuple(q_a.weight.shape) != (module.q_lora_rank, hidden):
        raise ValueError(
            "the source attention needs a q_a_proj of shape "
            f"{(module.q_lora_rank, hidden)} for the correction to copy"
        )
    for name in ("q_b_proj", "kv_b_proj"):
        mine, theirs = getattr(module, name).weight, getattr(attention, name).weight
        if mine.shape != theirs.shape:
            raise ValueError(
                f"{name}: correction {tuple(mine.shape)} vs source {tuple(theirs.shape)}"
            )
    if module.learnable_sink and not getattr(attention, "learnable_sink", False):
        raise ValueError("the correction has a learnable sink but its source attention has none")

    module.q_in.weight.zero_()
    module.q_in.weight[:, :hidden].copy_(q_a.weight)
    module.q_a_layernorm.weight.copy_(attention.q_a_layernorm.weight)
    module.q_b_proj.weight.copy_(attention.q_b_proj.weight)
    module.kv_b_proj.weight.copy_(attention.kv_b_proj.weight)
    if module.learnable_sink:
        module.learnable_sink_param.copy_(attention.learnable_sink_param)
    module.o_proj.weight.zero_()
    if module.block_keys:
        init_block_keys_from_attention(module, attention)


@torch.no_grad()
def init_block_keys_from_attention(module: AttentiveCorrection, attention: nn.Module) -> None:
    """Start ``module``'s block keys as ``attention`` projects its block: its
    ``kv_a_proj_with_mqa`` into the hidden half of ``kv_in`` (the previous-token half is
    zeroed; under block_kv_in "hidden" that half is all of kv_in) and its
    ``kv_a_layernorm``; a gated block read also gets :func:`init_block_gate_from_attention`,
    and its own kv_b :func:`init_block_kv_b`."""
    hidden = module.hidden_size
    kv_a = getattr(attention, "kv_a_proj_with_mqa", None)
    width = module.kv_lora_rank + module.qk_rope_head_dim
    if kv_a is None or tuple(kv_a.weight.shape) != (width, hidden):
        raise ValueError(
            "the source attention needs a kv_a_proj_with_mqa of shape "
            f"{(width, hidden)} for the block keys to copy"
        )
    module.kv_in.weight.zero_()
    module.kv_in.weight[:, :hidden].copy_(kv_a.weight)
    module.kv_a_layernorm.weight.copy_(attention.kv_a_layernorm.weight)
    if module.block_gated:
        init_block_gate_from_attention(module, attention)
    if module.block_kv_b_separate:
        init_block_kv_b(module)


@torch.no_grad()
def init_block_kv_b(module: AttentiveCorrection) -> None:
    """Start the block read's own kv_b (``attentive_correction_block_kv_b='separate'``) as
    ``module``'s ``kv_b_proj``, so it reads its block as the shared one would."""
    module.block_kv_b_proj.weight.copy_(module.kv_b_proj.weight)


@torch.no_grad()
def init_block_gate_from_attention(module: AttentiveCorrection, attention: nn.Module) -> None:
    """Start ``module``'s gated block read closed: ``attention``'s sink (zero without one), no
    distance bias and a zero gate. The block keys are left as they are."""
    if getattr(attention, "learnable_sink", False):
        module.block_sink.copy_(attention.learnable_sink_param)
    else:
        module.block_sink.zero_()
    module.block_rel_bias.zero_()
    module.block_gate.weight.zero_()
    module.block_gate_bias.zero_()
