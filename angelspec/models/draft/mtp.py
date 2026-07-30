"""Single-head MTP draft model (DeepSeek-V3 / Hy3 style).

The MTP head is the Hy3 MTP layer stored as
model.layers.<num_hidden_layers>.* inside the target checkpoint:

    enorm(embed(shifted_tokens))  +  hnorm(last_hidden)
        -> cat -> eh_proj(2h -> h)
        -> one HYV3 MoE decoder layer (qk_norm GQA attn + 192-expert MoE)
        -> final_layernorm
        -> shared (tied, frozen) lm_head over the full vocab

It consumes a single last-layer hidden state (no aux fusion, no vocab pruning)
and shares the target lm_head, plugging into the TTT loop (MTPModel) via the
Eagle3DraftModel interface.

Weight names follow the checkpoint layout (self_attn.{q,k,v,o}_proj,
self_attn.{q,k}_norm, mlp.experts.<e>.*, mlp.shared_mlp.*, mlp.router.gate,
mlp.expert_bias, input_layernorm, post_attention_layernorm) so the CPT loader
can map model.layers.<N>.* directly.
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from angelspec.models.draft.base import Eagle3DraftModel
from angelspec.models.draft.llama3_eagle import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from angelspec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
)
from angelspec.utils.distributed import get_sp_ulysses_group


def _mtp_dense_block0(q, k0, v0, scale, valid_mask, doc_ids=None):
    """Dense causal block-0 attention (CPU fallback / numerical reference).

    q/k0/v0: [b, H, s, d] (heads already repeated). valid_mask: [b, s] bool.
    Returns (out0, lse0), lse0 = logsumexp of the block-0 logits.
    """
    bsz, H, s, d = q.shape
    logits = torch.matmul(q, k0.transpose(-1, -2)) * scale  # [b,H,s,s]
    causal = torch.triu(torch.ones(s, s, dtype=torch.bool, device=q.device), diagonal=1)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(causal[None, None], neg_inf)
    if valid_mask is not None:
        logits = logits.masked_fill(~valid_mask[:, None, None, :], neg_inf)
    if doc_ids is not None:
        cross = doc_ids[:, :, None] != doc_ids[:, None, :]  # [b,s,s]
        logits = logits.masked_fill(cross[:, None], neg_inf)
    lse0 = torch.logsumexp(logits, dim=-1)  # [b,H,s]
    probs = torch.softmax(logits, dim=-1)
    out0 = torch.matmul(probs, v0)  # [b,H,s,d]
    return out0, lse0


def _mtp_flex_block0(q, k0, v0, scale, valid_mask, doc_ids=None):
    """block-0 causal attention via compiled flex_attention (GPU, O(seq) memory).

    Returns (out0, lse0) matching _mtp_dense_block0's semantics. `valid_mask` [b,s]
    gates keys via the block mask.
    """
    bsz, H, s, d = q.shape
    vm = valid_mask
    di = doc_ids

    def mask_mod(b, h, q_idx, kv_idx):
        m = q_idx >= kv_idx
        if vm is not None:
            m = m & vm[b, kv_idx]
        if di is not None:
            m = m & (di[b, q_idx] == di[b, kv_idx])
        return m

    block_mask = compile_friendly_create_block_mask(
        mask_mod,
        bsz,
        None,
        s,
        s,
        device=q.device,
        # KV_LEN == full seq, so the un-compiled builder materializes the full
        # [s, s] grid (128 GiB at 128k). Compiled builder is O(num_blocks).
        _compile=True,
    )
    out0, lse0 = compile_friendly_flex_attention(
        q, k0, v0, block_mask=block_mask, scale=scale, return_lse=True
    )
    return out0, lse0


def _mtp_cached_merge(q, cache_k, cache_v, valid_mask, scale, use_flex=None, doc_ids=None):
    """Memory-lean cached TTT attention.

      - block-0: causal attention of q over the full-seq cached step-0 K/V,
      - step i>=1: query j additionally attends the single diagonal key ki[j]
        (the EAGLE per-step chain) with value vi[j],
      - all merged in one softmax via log-sum-exp,
    without materializing the [b,H,q,seq] score matrix: block-0 goes through
    flex_attention (compiled, O(seq)); the i>=1 terms are per-query scalars.
    ``use_flex=None`` auto-selects flex on CUDA, dense block-0 on CPU (flex has no
    CPU backward).

    Layout (KV heads already repeated so num_groups==1):
      q          : [b, H, s, d]
      cache_k/v  : [b, H, nc, s, d]   (nc = number of cached steps)
      valid_mask : [b, s] bool, or None
    """
    bsz, H, s, d = q.shape
    nc = cache_k.shape[2]
    # fp32 accum for low-precision inputs; keep native dtype for fp32/fp64.
    cdt = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    if use_flex is None:
        use_flex = q.is_cuda
    k0 = cache_k[:, :, 0]
    v0 = cache_v[:, :, 0]

    if use_flex:
        out0, lse0 = _mtp_flex_block0(q, k0, v0, scale, valid_mask, doc_ids)
        out0 = out0.to(cdt)
        lse0 = lse0.to(cdt)
    else:
        out0, lse0 = _mtp_dense_block0(
            q.to(cdt), k0.to(cdt), v0.to(cdt), scale, valid_mask, doc_ids
        )

    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=cdt)
    vmask = valid_mask[:, None, :] if valid_mask is not None else None  # [b,1,s]
    lse0 = torch.where(vmask, lse0, neg_inf) if vmask is not None else lse0
    lse_terms = [lse0]
    for i in range(1, nc):
        ki = cache_k[:, :, i].to(cdt)
        lse_i = (q.to(cdt) * ki).sum(-1) * scale  # [b,H,s]
        lse_terms.append(torch.where(vmask, lse_i, neg_inf) if vmask is not None else lse_i)

    merged_lse = torch.logsumexp(torch.stack(lse_terms, dim=-1), dim=-1)  # [b,H,s]
    # All-invalid-keys query (e.g. a USP shard with zero valid tokens) → merged_lse
    # = -inf → exp(-inf - -inf) = nan poisons grads (and nan all-reduce desyncs USP
    # ranks). Those positions are masked to 0 below anyway, so make -inf finite first.
    merged_lse = torch.where(torch.isfinite(merged_lse), merged_lse, torch.zeros_like(merged_lse))
    out = out0 * torch.exp(lse0 - merged_lse).unsqueeze(-1)
    for i in range(1, nc):
        vi = cache_v[:, :, i].to(cdt)
        out = out + vi * torch.exp(lse_terms[i] - merged_lse).unsqueeze(-1)

    if vmask is not None:
        out = torch.where(vmask.unsqueeze(-1), out, 0.0)

    return out.to(q.dtype)


class _MTPSeqAllToAll(torch.autograd.Function):
    """Ulysses all-to-all over a [b, H, S, d] tensor: swap the head and seq dims
    across the sequence-parallel group.

    scatter_dim/gather_dim are dims of the 4D tensor (1=head, 2=seq). Backward is
    the symmetric all-to-all (scatter/gather swapped).
    """

    @staticmethod
    def forward(ctx, x, group, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _mtp_all_to_all_4d(x, group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad):
        g = _mtp_all_to_all_4d(grad, ctx.group, ctx.gather_dim, ctx.scatter_dim)
        return g, None, None, None


def _mtp_all_to_all_4d(x, group, scatter_dim, gather_dim):
    """Ulysses all-to-all for a 4D [b, H, S, d] tensor (eager, differentiable-safe).

    Splits ``scatter_dim`` into ``world`` equal parts (one sent to each rank) and
    gathers ``gather_dim`` from all ranks. Used to swap between seq-sharded
    ([b, H, S_local, d]) and head-sharded ([b, H_local, S, d]) layouts.
    """
    if os.environ.get("ANGELSPEC_USP_COLLECTIVE_DEBUG") == "1":
        print(
            f"[USP-A2A rank{dist.get_rank()}] all_to_all scatter={scatter_dim} gather={gather_dim} shape={tuple(x.shape)}",
            flush=True,
        )
    world = dist.get_world_size(group)
    if world == 1:
        return x
    b, H, S, d = x.shape
    if scatter_dim == 1 and gather_dim == 2:
        # [b, H, S_local, d] -> [b, H_local, S, d]  (scatter heads, gather seq)
        assert H % world == 0, f"num_heads {H} not divisible by sp world {world}"
        Hl = H // world
        # -> [world, b, Hl, S, d]
        x = x.reshape(b, world, Hl, S, d).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        # out[r]: this rank's Hl heads for rank r's seq shard -> gather seq
        out = out.permute(1, 2, 0, 3, 4).contiguous().reshape(b, Hl, world * S, d)
        return out
    elif scatter_dim == 2 and gather_dim == 1:
        # [b, H_local, S, d] -> [b, H, S_local, d]  (scatter seq, gather heads)
        assert S % world == 0, f"seq {S} not divisible by sp world {world}"
        Sl = S // world
        Hl = H
        # -> [world, b, Hl, Sl, d]
        x = x.reshape(b, Hl, world, Sl, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        # out[r]: rank r's heads for this rank's seq shard -> gather heads
        out = out.permute(1, 0, 2, 3, 4).contiguous().reshape(b, world * Hl, Sl, d)
        return out
    else:
        raise ValueError(
            f"unsupported (scatter_dim={scatter_dim}, gather_dim={gather_dim}); expected (1,2) or (2,1)"
        )


def _mtp_ulysses_all_to_all(x, group, scatter_dim, gather_dim):
    """Differentiable Ulysses all-to-all wrapper (see _MTPSeqAllToAll)."""
    return _MTPSeqAllToAll.apply(x, group, scatter_dim, gather_dim)


class MTPConfig(PretrainedConfig):
    """Configuration for the single-head MTP draft model.

    Defaults match the Hy3 (hy_v3) target so a config JSON only needs to
    override what differs. ``vocab_size`` is the FULL target vocab (no pruning),
    and the lm_head is tied to the frozen target head, so it is not a trainable
    parameter of this model.
    """

    model_type = "mtp"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        vocab_size: int = 120832,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 262144,
        rope_theta: float = 11158840.0,
        hidden_act: str = "silu",
        qk_norm: bool = True,
        # MoE
        num_experts: int = 192,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 1536,
        moe_router_use_sigmoid: bool = True,
        moe_router_enable_expert_bias: bool = True,
        route_norm: bool = True,
        router_scaling_factor: float = 2.826,
        first_k_dense_replace: int = 1,
        # FFN variant: "moe" (Hy3 MoE, default) or "dense" (plain SwiGLU,
        # for dense targets like Qwen3/Llama).
        mlp_type: str = "moe",
        # target / draft wiring
        target_hidden_size: int = 4096,
        tie_lm_head: bool = True,
        mtp_init_from_target_layer: Optional[int] = None,
        # loss
        mtp_ce_loss_scaling_factor: float = 0.1,
        mtp_kl_loss_scaling_factor: float = 0.0,
        mtp_kl_topk: Optional[int] = None,
        mtp_kl_variant: str = "a",
        mtp_ce_use_ground_truth: bool = False,
        mtp_single_step_loss: Optional[str] = None,
        mtp_kl_tv_eta: float = 3.0,
        mtp_e2e_weighting: bool = False,
        mtp_e2e_direct: bool = False,
        mtp_e2e_tv_loss_scaling_factor: float = 1.0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.qk_norm = qk_norm
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_router_use_sigmoid = moe_router_use_sigmoid
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.route_norm = route_norm
        self.router_scaling_factor = router_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        if mlp_type not in ("moe", "dense"):
            raise ValueError(f"mlp_type must be 'moe' or 'dense', got {mlp_type!r}")
        self.mlp_type = mlp_type
        self.target_hidden_size = target_hidden_size
        self.tie_lm_head = tie_lm_head
        self.mtp_init_from_target_layer = mtp_init_from_target_layer
        self.mtp_ce_loss_scaling_factor = mtp_ce_loss_scaling_factor
        self.mtp_kl_loss_scaling_factor = mtp_kl_loss_scaling_factor
        self.mtp_kl_topk = mtp_kl_topk
        self.mtp_kl_variant = mtp_kl_variant
        self.mtp_ce_use_ground_truth = mtp_ce_use_ground_truth
        if mtp_single_step_loss not in (None, "tv", "kl", "lk"):
            raise ValueError(
                f"mtp_single_step_loss must be None/'tv'/'kl'/'lk', got {mtp_single_step_loss!r}"
            )
        if (mtp_e2e_weighting or mtp_e2e_direct) and mtp_single_step_loss is None:
            raise ValueError(
                "mtp_e2e_weighting/mtp_e2e_direct require mtp_single_step_loss to be set "
                "(use 'tv' for the e2e acceptance-length objective)"
            )
        self.mtp_single_step_loss = mtp_single_step_loss
        self.mtp_kl_tv_eta = mtp_kl_tv_eta
        self.mtp_e2e_weighting = mtp_e2e_weighting
        self.mtp_e2e_direct = mtp_e2e_direct
        self.mtp_e2e_tv_loss_scaling_factor = mtp_e2e_tv_loss_scaling_factor
        self.rope_scaling = kwargs.get("rope_scaling", None)


class MTPAttention(nn.Module):
    """qk_norm GQA attention with EAGLE-style per-step KV cache.

    qk_norm is applied per-head before RoPE. Each TTT step appends one
    full-sequence K/V block along a "cached steps" dim (cache shape
    ``[bsz, num_heads, num_cached, seq_len, head_dim]``).
    """

    def __init__(self, config: MTPConfig, attention_backend: str = "sdpa"):
        super().__init__()
        self.config = config
        self.attention_backend = attention_backend
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.use_qk_norm = getattr(config, "qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
        )

        # Pure-Ulysses sequence parallelism: each rank holds a seq shard; before
        # attention we all-to-all to a head-shard so local flex attention is exact.
        # Group resolved lazily on first cached-forward (after init_usp_groups).
        self._usp_group = None
        self._usp_size = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        ctx_doc_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        import math

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # qk_norm is applied per-head on the head_dim dimension, before transpose.
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if not use_cache:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=0.0,
            )
        else:
            lck = 0 if cache_keys is None else cache_keys.shape[2]

            if self.attention_backend == "usp" and self._usp_group is None:
                self._usp_group = get_sp_ulysses_group()
                self._usp_size = (
                    dist.get_world_size(self._usp_group) if self._usp_group is not None else 1
                )
            usp = self.attention_backend == "usp" and self._usp_size > 1

            # RoPE positions are GLOBAL (MTPModel offsets position_ids per rank), so
            # the cos/sin cache must span the global sequence, not the local shard.
            global_q_len = q_len * self._usp_size if usp else q_len
            cos, sin = self.rotary_emb(query_states, seq_len=global_q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck
            )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if usp:
                # Ulysses: swap seq-shard -> head-shard so local attention sees the
                # full sequence. K/V were repeat_kv'd to num_heads, so their head
                # count is divisible by the SP world.
                query_states = _mtp_ulysses_all_to_all(query_states, self._usp_group, 1, 2)
                key_states = _mtp_ulysses_all_to_all(key_states, self._usp_group, 1, 2)
                value_states = _mtp_ulysses_all_to_all(value_states, self._usp_group, 1, 2)

            if cache_keys is None:
                cache_keys = key_states.unsqueeze(2)
                cache_values = value_states.unsqueeze(2)
            else:
                cache_keys = torch.cat([cache_keys, key_states.unsqueeze(2)], dim=2)
                cache_values = torch.cat([cache_values, value_states.unsqueeze(2)], dim=2)

            lck = cache_keys.shape[2]
            k0 = cache_keys[:, :, 0]
            v0 = cache_values[:, :, 0]

            if self.attention_backend in ("fa", "usp"):
                # Memory-lean path: block-0 via flex_attention (O(seq)), chain terms
                # merged by log-sum-exp. attention_mask here is the [b, seq] padding
                # mask (1=valid), NOT the dense additive mask.
                if attention_mask is None:
                    valid_mask = torch.ones(
                        bsz, q_len, dtype=torch.bool, device=query_states.device
                    )
                else:
                    valid_mask = attention_mask.to(torch.bool)
                    if valid_mask.dim() > 2:
                        valid_mask = valid_mask.view(bsz, -1)[:, :q_len]
                if usp:
                    # After all-to-all each rank holds the FULL sequence; gather the
                    # per-rank [b, S_local] key-validity mask into [b, S_full] in rank
                    # order. all_gather_into_tensor concatenates along dim 0.
                    vm = valid_mask.contiguous()
                    gathered = torch.empty(
                        self._usp_size * bsz, q_len, dtype=vm.dtype, device=vm.device
                    )
                    if os.environ.get("ANGELSPEC_USP_COLLECTIVE_DEBUG") == "1":
                        print(
                            f"[USP-AG rank{dist.get_rank()}] all_gather valid_mask shape={tuple(vm.shape)}",
                            flush=True,
                        )
                    dist.all_gather_into_tensor(gathered, vm, group=self._usp_group)
                    valid_mask = (
                        gathered.view(self._usp_size, bsz, q_len)
                        .permute(1, 0, 2)
                        .reshape(bsz, self._usp_size * q_len)
                    )
                # Packing doc-gate for block-0; gather to full seq under USP like valid_mask.
                doc_ids = None
                if ctx_doc_ids is not None:
                    doc_ids = ctx_doc_ids
                    if doc_ids.dim() > 2:
                        doc_ids = doc_ids.view(bsz, -1)[:, :q_len]
                    if usp:
                        di = doc_ids.contiguous().to(torch.long)
                        gathered_d = torch.empty(
                            self._usp_size * bsz, q_len, dtype=di.dtype, device=di.device
                        )
                        dist.all_gather_into_tensor(gathered_d, di, group=self._usp_group)
                        doc_ids = (
                            gathered_d.view(self._usp_size, bsz, q_len)
                            .permute(1, 0, 2)
                            .reshape(bsz, self._usp_size * q_len)
                        )
                attn_output = _mtp_cached_merge(
                    query_states,
                    cache_keys,
                    cache_values,
                    valid_mask,
                    1.0 / math.sqrt(self.head_dim),
                    # Under USP, flex's lazy compile can desync ranks (one enters the
                    # grad all-reduce while peers still compile → NCCL hang), so it must
                    # be warmed up identically on every rank (MTPModel.warmup_usp_flex).
                    use_flex=None,
                    doc_ids=doc_ids,
                ).to(query_states.dtype)
                if usp:
                    # Swap head-shard -> seq-shard so o_proj / downstream see the
                    # local seq slice with all heads again.
                    attn_output = _mtp_ulysses_all_to_all(attn_output, self._usp_group, 2, 1)
            else:
                attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(
                    self.head_dim
                )
                attn_weights = attn_weights + attention_mask

                for i in range(1, lck):
                    ki = cache_keys[:, :, i]
                    attn_weightsi = (query_states * ki).sum(-1) / math.sqrt(self.head_dim)
                    attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_weights0 = attn_weights[..., :q_len]
                attn_output = torch.matmul(attn_weights0, v0)

                for i in range(1, lck):
                    vi = cache_values[:, :, i]
                    attn_weightsi = attn_weights[..., q_len + i - 1]
                    attn_output = attn_output + attn_weightsi[..., None] * vi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)

        return attn_output, cache_keys, cache_values


class SwiGLUMLP(nn.Module):
    """Dense SwiGLU MLP (shared expert / dense layer)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Hy3MoE(nn.Module):
    """Training-friendly MoE matching hy_v3 routing semantics.

    Sigmoid router scores + ``expert_bias`` correction for top-k selection,
    optional ``route_norm`` renormalisation of the selected (pre-bias) weights,
    ``router_scaling_factor`` applied to the routed output, plus an always-on
    shared expert.

    Expert weights are stored FUSED as three ``[E, in, out]`` parameters
    (``experts_{gate,up,down}_proj``) so the routed computation is three
    ``torch._grouped_mm`` calls instead of a per-expert Python loop. This layout
    is the transpose of the per-expert ``nn.Linear.weight`` (``[out, in]``)
    checkpoint layout; the CPT loader transposes+stacks accordingly. Set
    ``ANGELSPEC_MTP_MOE_REFERENCE=1`` to fall back to the per-expert reference loop.
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.route_norm = config.route_norm
        self.router_scaling_factor = config.router_scaling_factor
        self.use_sigmoid = config.moe_router_use_sigmoid

        H = config.hidden_size
        inter = config.moe_intermediate_size
        E = self.num_experts
        # Fused expert weights [E, in, out] (transpose of nn.Linear.weight [out, in])
        # so they feed straight into torch._grouped_mm(mat_a=[T, in], mat_b) → [T, out].
        self.experts_gate_proj = nn.Parameter(torch.empty(E, H, inter))  # [E, H, I]
        self.experts_up_proj = nn.Parameter(torch.empty(E, H, inter))  # [E, H, I]
        self.experts_down_proj = nn.Parameter(torch.empty(E, inter, H))  # [E, I, H]
        self.act_fn = nn.SiLU()
        # Match nn.Linear's default init (kaiming_uniform_, a=sqrt(5)) with the
        # correct fan_in: init in the [out, in] layout then transpose into place.
        with torch.no_grad():
            for w, out_dim, in_dim in (
                (self.experts_gate_proj, inter, H),
                (self.experts_up_proj, inter, H),
                (self.experts_down_proj, H, inter),
            ):
                for e in range(E):
                    tmp = torch.empty(out_dim, in_dim)
                    nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
                    w[e].copy_(tmp.t())

        self.router = nn.Module()
        self.router.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        # Router gate stays fp32: a bf16 gate with fp32 input would crash F.linear.
        self.router.gate.weight.data = self.router.gate.weight.data.float()
        if config.moe_router_enable_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))
        else:
            self.register_parameter("expert_bias", None)

        if config.num_shared_experts > 0:
            self.shared_mlp = SwiGLUMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
            )
        else:
            self.shared_mlp = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)
        n_tokens = x.shape[0]

        # Routing math runs in fp32 for numerical stability.
        logits = self.router.gate(x.to(self.router.gate.weight.dtype)).float()
        scores = logits.sigmoid() if self.use_sigmoid else logits.softmax(dim=-1)

        scores_for_choice = scores
        if self.expert_bias is not None:
            scores_for_choice = scores + self.expert_bias.float().unsqueeze(0)
        topk_idx = torch.topk(scores_for_choice, self.top_k, dim=-1).indices  # [N, top_k]
        # weights come from the un-biased scores at the chosen experts
        topk_weights = torch.gather(scores, 1, topk_idx)
        if self.route_norm:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = (topk_weights * self.router_scaling_factor).to(x.dtype)

        use_reference = os.environ.get("ANGELSPEC_MTP_MOE_REFERENCE", "0") not in {
            "0",
            "false",
            "False",
        }
        # ``torch._grouped_mm`` is a private API and is absent from some supported
        # PyTorch builds (for example 2.7.x).  Falling back here keeps MTP, including
        # the USP + packing path, functional instead of failing before attention.
        use_reference = use_reference or not hasattr(torch, "_grouped_mm")
        if use_reference:
            out = self._routed_reference(x, topk_idx, topk_weights)
        else:
            out = self._routed_grouped(x, topk_idx, topk_weights)

        if self.shared_mlp is not None:
            out = out + self.shared_mlp(x)

        if os.environ.get("ANGELSPEC_MTP_ROUTER_DEBUG", "0") not in {"0", "false", "False"}:
            self._log_router_stats(topk_idx, n_tokens)

        return out.view(bsz, seq_len, hidden_dim)

    def _routed_grouped(
        self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """Fused expert compute via three torch._grouped_mm calls.

        Sort the (token, slot) pairs by expert so each expert's tokens are
        contiguous, run one grouped GEMM per proj over the fused ``[E, in, out]``
        weights, then scatter the weighted results back.
        """
        N, k = topk_idx.shape
        flat_expert = topk_idx.reshape(-1)  # [N*k]
        flat_token = (
            torch.arange(N, device=x.device).unsqueeze(1).expand(N, k).reshape(-1)
        )  # [N*k]
        flat_weight = topk_weights.reshape(-1)  # [N*k]

        # Sort by expert id so tokens routed to the same expert are contiguous.
        order = torch.argsort(flat_expert)
        sorted_expert = flat_expert[order]
        sorted_token = flat_token[order]
        sorted_weight = flat_weight[order]

        # group_sizes per expert → cumulative end offsets (int32) for _grouped_mm.
        group_sizes = torch.bincount(sorted_expert, minlength=self.num_experts)
        offs = torch.cumsum(group_sizes, 0).to(torch.int32)

        xs = x[sorted_token]  # [N*k, H] — routed token inputs (k copies per token)
        g = torch._grouped_mm(xs, self.experts_gate_proj, offs=offs)  # [N*k, I]
        u = torch._grouped_mm(xs, self.experts_up_proj, offs=offs)  # [N*k, I]
        h = self.act_fn(g) * u
        o = torch._grouped_mm(h, self.experts_down_proj, offs=offs)  # [N*k, H]
        o = o * sorted_weight.unsqueeze(-1).to(o.dtype)

        out = torch.zeros_like(x)
        out.index_add_(0, sorted_token, o)
        return out

    def _routed_reference(
        self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """Slow per-expert reference loop, kept for numerical cross-checks.

        Enabled via ``ANGELSPEC_MTP_MOE_REFERENCE=1``.
        """
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            token_idx, slot_idx = torch.where(topk_idx == e)
            if token_idx.numel() == 0:
                continue
            w = topk_weights[token_idx, slot_idx].unsqueeze(-1)
            xe = x[token_idx]
            g = torch.nn.functional.linear(xe, self.experts_gate_proj[e].t())
            u = torch.nn.functional.linear(xe, self.experts_up_proj[e].t())
            ye = torch.nn.functional.linear(self.act_fn(g) * u, self.experts_down_proj[e].t())
            out.index_add_(0, token_idx, ye * w)
        return out

    @torch.no_grad()
    def _log_router_stats(self, topk_idx: torch.Tensor, n_tokens: int) -> None:
        """Log expert-selection distribution (gated by ``ANGELSPEC_MTP_ROUTER_DEBUG``).

        A near-degenerate distribution (a few experts taking almost all tokens)
        after CPT usually means a routing mismatch (scaling/bias/sigmoid).
        """
        from angelspec.utils.logging import logger

        counts = torch.bincount(topk_idx.flatten(), minlength=self.num_experts).float()
        total = counts.sum().clamp_min(1.0)
        frac = counts / total
        used = int((counts > 0).sum().item())
        ideal = self.top_k / self.num_experts
        cv = (frac.std() / frac.mean().clamp_min(1e-9)).item()  # 0 == perfectly balanced
        logger.info(
            f"[MTP router] tokens={n_tokens} experts_used={used}/{self.num_experts} "
            f"max_share={frac.max().item():.4f} min_share={frac.min().item():.4f} "
            f"ideal_share={ideal:.4f} load_cv={cv:.3f}"
        )


class MTPDecoderLayer(nn.Module):
    """One Hy3 decoder layer (qk_norm attn + MoE), KV-cache aware.

    Pre-norm residual structure identical to the HF/vLLM reference:
        h = h + attn(input_layernorm(h))
        h = h + mlp(post_attention_layernorm(h))
    """

    def __init__(self, config: MTPConfig, attention_backend: str = "sdpa"):
        super().__init__()
        self.self_attn = MTPAttention(config, attention_backend=attention_backend)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # FFN follows the target architecture: dense SwiGLU for dense targets
        # (Qwen3/Llama, mlp_type="dense"), Hy3 MoE otherwise. MTPAttention
        # (qk-norm GQA) is shared by both and weight-compatible with a Qwen3 layer.
        if getattr(config, "mlp_type", "moe") == "dense":
            self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        else:
            self.mlp = Hy3MoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        ctx_doc_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, cache_keys, cache_values = self.self_attn(
            hidden_states=hidden_states,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            ctx_doc_ids=ctx_doc_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache_keys, cache_values


class MTPDraftModel(Eagle3DraftModel):
    """Single-head MTP draft model conforming to the Eagle3DraftModel interface.

    The lm_head is *tied* to the frozen target head: its weight is injected by
    the trainer via :meth:`set_lm_head_weight` and is not a trainable parameter
    here (no optimizer state, no FSDP shard).
    """

    config_class = MTPConfig

    def __init__(self, config: MTPConfig, attention_backend: str = "sdpa") -> None:
        super().__init__(config)
        self.config = config
        self.attention_backend = attention_backend

        self.target_vocab_size = config.vocab_size
        self.vocab_size = config.vocab_size  # no pruning for MTP

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None)
        )

        # MTP-specific fusion of (shifted-token embedding, last hidden state).
        self.enorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.midlayer = MTPDecoderLayer(config, attention_backend=attention_backend)
        self.final_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Optional projection if the target hidden size differs from the draft
        # hidden size (identity by default for hy_v3 where both are 4096).
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        if target_hidden_size != config.hidden_size:
            self.hidden_proj: Optional[nn.Linear] = nn.Linear(
                target_hidden_size, config.hidden_size, bias=False
            )
        else:
            self.hidden_proj = None

        # Tied frozen lm_head (full vocab), injected by the trainer. Held as a PLAIN
        # ATTRIBUTE (not Parameter/buffer) to exclude it from FSDP2 sharding: a
        # Parameter would be collected as a DTensor and break the later rebind's
        # lazy_init; a buffer would force a vocab×hidden broadcast on every rank.
        self.lm_head_weight: Optional[torch.Tensor] = None

    @property
    def norm(self) -> nn.Module:
        """Alias for final_layernorm — a property, NOT a submodule, so it does not
        duplicate ``final_layernorm`` in the state dict."""
        return self.final_layernorm

    # -- tied lm_head injection ------------------------------------------------

    @torch.no_grad()
    def set_lm_head_weight(self, weight: torch.Tensor) -> None:
        """Tie the lm_head to the (frozen) target head weight (vocab, hidden).

        Stored as a plain attribute (NOT a Parameter or buffer) so it shares memory
        with the frozen target head and is never picked up by FSDP2.
        """
        self.lm_head_weight = weight

    # -- Eagle3DraftModel interface -------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Single last-layer hidden state. Project only if dims differ.
        if self.hidden_proj is not None:
            return self.hidden_proj(hidden_states)
        return hidden_states

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        ctx_doc_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # hy_v3 masks the position-0 token embedding (per-doc under packing).
        if position_ids is not None:
            mask0 = (position_ids == 0).unsqueeze(-1).to(input_embeds.dtype)
            input_embeds = input_embeds * (1.0 - mask0)

        de = self.enorm(input_embeds)
        h = self.hnorm(hidden_states)
        x = self.eh_proj(torch.cat([de, h], dim=-1))

        x, cache_keys, cache_values = self.midlayer(
            hidden_states=x,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            ctx_doc_ids=ctx_doc_ids,
        )
        x = self.final_layernorm(x)
        return x, cache_keys, cache_values

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states already final_layernorm'd by backbone.
        return F.linear(hidden_states, self.lm_head_weight)

    def get_lm_head_params(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # backbone already applies final_layernorm, so the fused loss path must NOT
        # re-normalise; it consumes post-norm hidden directly.
        return (
            self.final_layernorm.weight,
            self.lm_head_weight,
            self.final_layernorm.variance_epsilon,
        )
