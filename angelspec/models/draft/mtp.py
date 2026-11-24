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

import functools
import math
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import PretrainedConfig

from angelspec.models.draft.base import Eagle3DraftModel
from angelspec.models.draft.llama3_eagle import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from angelspec.models.ops.attention_mask import (
    mtp_causal_mask,
    mtp_validity_mask,
    validate_mtp_backend,
)
from angelspec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
)
from angelspec.models.ops.mrope import MRotaryEmbedding, advance_mtp_positions
from angelspec.models.ops.sparse_indices import (
    append_diagonal_kv_indices,
    build_contiguous_causal_ranges,
    build_forced_tail_candidates,
    compressed_group_token_offsets,
    compressed_middle_group_ranges,
    compute_compressed_cu_seqlens,
    expand_compressed_groups_to_tokens,
)
from angelspec.models.ops.sparse_mla import torch_lightning_indexer, torch_sparse_mla
from angelspec.models.ops.tilelang_dsa import (
    tilelang_lightning_indexer,
    tilelang_sparse_mla,
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


@torch.compiler.disable
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


class _MTPSeqAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        world = dist.get_world_size(group)
        ctx.world = world
        if world == 1:
            return x
        b, s = x.shape[:2]
        tail = x.shape[2:]
        gathered = torch.empty(
            (world * b, s, *tail),
            dtype=x.dtype,
            device=x.device,
        )
        dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
        return gathered.view(world, b, s, *tail).movedim(0, 1).reshape(b, world * s, *tail)

    @staticmethod
    def backward(ctx, grad):
        if ctx.world == 1:
            return grad, None
        b, full_s = grad.shape[:2]
        local_s = full_s // ctx.world
        tail = grad.shape[2:]
        stacked = (
            grad.reshape(b, ctx.world, local_s, *tail)
            .movedim(1, 0)
            .reshape(ctx.world * b, local_s, *tail)
            .contiguous()
        )
        local = torch.empty(
            (b, local_s, *tail),
            dtype=grad.dtype,
            device=grad.device,
        )
        dist.reduce_scatter_tensor(local, stacked, group=ctx.group)
        return local, None


def _mtp_seq_all_gather(x, group):
    return _MTPSeqAllGather.apply(x, group)


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
        rms_norm_type: str = "llama",
        attn_output_gate: bool = False,
        partial_rotary_factor: float = 1.0,
        rope_parameters: Optional[dict] = None,
        # MoE
        num_experts: int = 192,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 1536,
        moe_router_use_sigmoid: bool = True,
        moe_router_enable_expert_bias: bool = True,
        route_norm: bool = True,
        router_scaling_factor: float = 2.826,
        swiglu_limit: float = 0.0,
        # Qwen3.5-MoE scales the shared expert by sigmoid(shared_expert_gate(x)).
        shared_expert_gate: bool = False,
        # Skip the expensive per-expert random fill when a trainer will
        # immediately overwrite every expert from a target checkpoint.
        moe_skip_expert_init: bool = False,
        # DeepSeek-style aux-loss-free load balancing. The forward accumulates
        # hard top-k assignment counts; the trainer all-reduces them and updates
        # expert_bias manually after each optimizer step.
        moe_aux_free_bias_update: bool = False,
        moe_aux_free_bias_beta: float = 0.9,
        moe_aux_free_bias_gamma: float = 1e-4,
        first_k_dense_replace: int = 1,
        # FFN variant: "moe" (Hy3 MoE, default) or "dense" (plain SwiGLU,
        # for dense targets like Qwen3/Llama).
        mlp_type: str = "moe",
        # Attention variant: "gqa" (qk-norm GQA, default) or "mla" (gated MLA +
        # optional DSA indexer, to reuse a DeepSeek-V3-style pretrained MTP head).
        attention_type: str = "gqa",
        # MLA config (only read when attention_type == "mla")
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        gated_mla: bool = False,
        gating_type: str = "elementwise",
        learnable_sink: bool = False,
        learnable_sink_init: float = 0.0,
        # DSA indexer (only active when attention_type == "mla" and seq >= index_topk)
        index_topk: Optional[int] = None,
        index_head_dim: int = 128,
        index_n_heads: int = 16,
        dsa_backend: str = "torch",
        dsa_fp8_indexer: bool = False,
        # True: TTT steps >= 1 reuse step 0's top-k. False: every step selects its
        # own top-k, scoring its queries against the step-0 indexer keys.
        dsa_index_share: bool = True,
        # Hy4 flash: two-level compressed DSA indexer (coarse group recall over
        # ratio-pooled keys; with skip_fine_select the recalled groups plus the
        # current group are the final sparse-attention candidates).
        use_compressed_indexer: bool = False,
        compressed_indexer_compress_ratio: int = 4,
        compressed_indexer_num_attention_heads: int = 16,
        compressed_indexer_head_dim: int = 128,
        compressed_indexer_group_topk: int = 512,
        compressed_indexer_candidate_topk: int = 2048,
        compressed_indexer_skip_fine_select: bool = True,
        compressed_indexer_key_pooling: str = "gated",
        # Hy4 flash: post-sublayer RMSNorms (attention post_proj_norm, MoE
        # combine_output_norm) and latent routed experts.
        sandwich_norm: bool = False,
        routed_expert_hidden_size: Optional[int] = None,
        # target / draft wiring
        target_hidden_size: int = 4096,
        tie_lm_head: bool = True,
        mtp_init_from_target_layer: Optional[int] = None,
        mtp_init_from_mtp_layer: Optional[int] = None,
        # Fail (instead of warn) when warm-starting from the target MTP layer
        # leaves draft weights uninitialized or drops checkpoint tensors.
        mtp_init_strict: bool = False,
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
        if rope_parameters is None:
            rope_parameters = kwargs.get("rope_scaling")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        if rope_parameters is not None:
            nested_theta = rope_parameters.get("rope_theta")
            if nested_theta is not None:
                rope_theta = float(nested_theta)
            nested_partial = rope_parameters.get("partial_rotary_factor")
            if nested_partial is not None:
                partial_rotary_factor = float(nested_partial)
        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError(
                f"partial_rotary_factor must be in (0, 1], got {partial_rotary_factor!r}"
            )
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_parameters = rope_parameters
        self.attn_output_gate = attn_output_gate
        self.hidden_act = hidden_act
        if rms_norm_type not in ("llama", "qwen3_5"):
            raise ValueError(f"rms_norm_type must be 'llama' or 'qwen3_5', got {rms_norm_type!r}")
        self.rms_norm_type = rms_norm_type
        self.qk_norm = qk_norm
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_router_use_sigmoid = moe_router_use_sigmoid
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.route_norm = route_norm
        self.router_scaling_factor = router_scaling_factor
        self.swiglu_limit = float(swiglu_limit or 0.0)
        self.shared_expert_gate = bool(shared_expert_gate)
        self.moe_skip_expert_init = bool(moe_skip_expert_init)
        self.moe_aux_free_bias_update = bool(moe_aux_free_bias_update)
        self.moe_aux_free_bias_beta = float(moe_aux_free_bias_beta)
        self.moe_aux_free_bias_gamma = float(moe_aux_free_bias_gamma)
        self.first_k_dense_replace = first_k_dense_replace
        if mlp_type not in ("moe", "dense"):
            raise ValueError(f"mlp_type must be 'moe' or 'dense', got {mlp_type!r}")
        self.mlp_type = mlp_type
        if attention_type not in ("gqa", "mla"):
            raise ValueError(f"attention_type must be 'gqa' or 'mla', got {attention_type!r}")
        self.attention_type = attention_type
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.gated_mla = gated_mla
        self.gating_type = gating_type
        self.learnable_sink = learnable_sink
        self.learnable_sink_init = learnable_sink_init
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        if dsa_backend not in ("torch", "tilelang", "torch_absorbed"):
            raise ValueError(
                f"dsa_backend must be 'torch', 'tilelang' or 'torch_absorbed', got {dsa_backend!r}"
            )
        self.dsa_backend = dsa_backend
        self.dsa_fp8_indexer = dsa_fp8_indexer
        self.dsa_index_share = dsa_index_share
        if use_compressed_indexer:
            if not compressed_indexer_skip_fine_select:
                raise ValueError(
                    "compressed indexer supports only compressed_indexer_skip_fine_select=True"
                )
            if compressed_indexer_key_pooling != "gated":
                raise ValueError("compressed indexer supports only gated key pooling")
            if dsa_backend not in ("tilelang", "torch_absorbed"):
                raise ValueError("compressed indexer needs dsa_backend tilelang or torch_absorbed")
            if index_topk != compressed_indexer_candidate_topk:
                raise ValueError("index_topk must equal compressed_indexer_candidate_topk")
            # group_topk - 1 recalled groups plus the forced current group.
            if (
                compressed_indexer_group_topk * compressed_indexer_compress_ratio
                != compressed_indexer_candidate_topk
            ):
                raise ValueError("candidate_topk must equal group_topk * compress_ratio")
        self.use_compressed_indexer = use_compressed_indexer
        self.compressed_indexer_compress_ratio = compressed_indexer_compress_ratio
        self.compressed_indexer_num_attention_heads = compressed_indexer_num_attention_heads
        self.compressed_indexer_head_dim = compressed_indexer_head_dim
        self.compressed_indexer_group_topk = compressed_indexer_group_topk
        self.compressed_indexer_candidate_topk = compressed_indexer_candidate_topk
        self.compressed_indexer_skip_fine_select = compressed_indexer_skip_fine_select
        self.compressed_indexer_key_pooling = compressed_indexer_key_pooling
        self.sandwich_norm = sandwich_norm
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.target_hidden_size = target_hidden_size
        self.tie_lm_head = tie_lm_head
        self.mtp_init_from_target_layer = mtp_init_from_target_layer
        self.mtp_init_from_mtp_layer = mtp_init_from_mtp_layer
        self.mtp_init_strict = mtp_init_strict
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
        rope_scaling = kwargs.get("rope_scaling", None)
        if rope_parameters is None and rope_scaling is not None:
            self.rope_parameters = rope_scaling


class Qwen35RMSNorm(nn.Module):
    """Qwen3.5's ``(1 + w) * RMS(x)`` in fp32; a Llama-style weight (near 1) loads as ``w - 1``.
    ``variance_epsilon`` aliases ``eps`` for get_lm_head_params."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * (1.0 + self.weight.float())).to(input_dtype)


def build_mtp_rms_norm(config: "MTPConfig", hidden_size: int) -> nn.Module:
    eps = config.rms_norm_eps
    if getattr(config, "rms_norm_type", "llama") == "qwen3_5":
        return Qwen35RMSNorm(hidden_size, eps=eps)
    return LlamaRMSNorm(hidden_size, eps=eps)


class MTPAttention(nn.Module):
    """qk_norm GQA attention with EAGLE-style per-step KV cache.

    qk_norm is applied per-head before RoPE. Each TTT step appends one
    full-sequence K/V block along a "cached steps" dim (cache shape
    ``[bsz, num_heads, num_cached, seq_len, head_dim]``).
    """

    def __init__(self, config: MTPConfig, attention_backend: str = "sdpa"):
        super().__init__()
        self.config = config
        self.attention_backend = validate_mtp_backend(attention_backend)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.attn_output_gate = getattr(config, "attn_output_gate", False)
        q_out = self.num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        self.q_proj = nn.Linear(self.hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.use_qk_norm = getattr(config, "qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = build_mtp_rms_norm(config, self.head_dim)
            self.k_norm = build_mtp_rms_norm(config, self.head_dim)

        partial = getattr(config, "partial_rotary_factor", 1.0)
        rotary_dim = int(self.head_dim * partial)
        if rotary_dim % 2 != 0 or rotary_dim <= 0:
            raise ValueError(
                f"partial RoPE dim must be a positive even integer, got {rotary_dim} "
                f"(head_dim={self.head_dim}, partial_rotary_factor={partial})"
            )
        self.rotary_dim = rotary_dim
        self.rotary_emb = LlamaRotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
        )

        # Pure-Ulysses sequence parallelism: each rank holds a seq shard; before
        # attention we all-to-all to a head-shard so local flex attention is exact.
        # Group resolved lazily on first cached-forward (after init_usp_groups).
        self._usp_group = None
        self._usp_size = 1

    def _project_q(
        self, hidden_states: torch.Tensor, bsz: int, q_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """With the output gate each head is ``[q | gate]``; the gate skips qk-norm and RoPE."""
        if not self.attn_output_gate:
            query_states = self.q_proj(hidden_states).view(
                bsz, q_len, self.num_heads, self.head_dim
            )
            return query_states, None
        mixed = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim * 2)
        query_states, gate = mixed.split(self.head_dim, dim=-1)
        return query_states, gate.reshape(bsz, q_len, self.num_heads * self.head_dim)

    def _apply_rope(self, query_states, key_states, cos, sin, position_ids):
        """Partial RoPE; equals Qwen3.5's MRoPE only for text, whose T/H/W positions are equal."""
        if self.rotary_dim == self.head_dim:
            return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        q_rot, q_pass = query_states[..., : self.rotary_dim], query_states[..., self.rotary_dim :]
        k_rot, k_pass = key_states[..., : self.rotary_dim], key_states[..., self.rotary_dim :]
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)
        return (
            torch.cat([q_rot, q_pass], dim=-1),
            torch.cat([k_rot, k_pass], dim=-1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        precomputed_index=None,
        return_index: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        import math

        bsz, q_len, _ = hidden_states.size()
        if not use_cache or self.attention_backend == "sdpa":
            attention_mask = mtp_causal_mask(
                attention_mask, ctx_doc_ids, bsz, q_len, hidden_states.device, hidden_states.dtype
            )
        else:
            attention_mask = mtp_validity_mask(attention_mask, bsz, q_len, hidden_states.device)

        query_states, gate = self._project_q(hidden_states, bsz, q_len)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

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
            query_states, key_states = self._apply_rope(
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
            query_states, key_states = self._apply_rope(
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

            if self.attention_backend in ("fa", "flex_attention", "usp"):
                # Memory-lean path: block-0 via flex_attention (O(seq)), chain terms
                # merged by log-sum-exp. attention_mask here is the [b, seq] padding
                # mask (1=valid), NOT the dense additive mask.
                if attention_mask is None:
                    valid_mask = torch.ones(
                        bsz, q_len, dtype=torch.bool, device=query_states.device
                    )
                else:
                    valid_mask = attention_mask
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
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        if return_index:
            return attn_output, cache_keys, cache_values, None
        return attn_output, cache_keys, cache_values


class SwiGLUMLP(nn.Module):
    """Dense SwiGLU MLP (shared expert / dense layer)."""

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.swiglu_limit = float(swiglu_limit or 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.swiglu_limit > 0:
            gate, up = _clamp_swiglu_inputs(gate, up, self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class FusedExpertProj(nn.Module):
    def __init__(self, num_experts: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, in_dim, out_dim))

    def forward(self, xs: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        return torch._grouped_mm(xs, self.weight, offs=offs)


@functools.lru_cache(maxsize=None)
def _exact_in_dtype(value: float, dtype: torch.dtype) -> bool:
    return torch.tensor(value, dtype=dtype).item() == value


def _clamp_swiglu_inputs(gate: torch.Tensor, up: torch.Tensor, limit: float):
    """Hy4 routed-expert clamp of the gate (above) and up (both sides) projections."""
    if _exact_in_dtype(limit, gate.dtype):
        return gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    # A limit that rounds in the activation dtype would move the gradient cutoff.
    return (
        gate.float().clamp(max=limit).to(gate.dtype),
        up.float().clamp(min=-limit, max=limit).to(up.dtype),
    )


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

    Latent MoE (``routed_expert_hidden_size`` set, Hy4 flash): the router and shared
    expert see the full hidden, routed experts run between ``routed_expert_down_proj``
    and ``routed_expert_up_proj``. ``sandwich_norm`` adds ``combine_output_norm`` over
    the routed + shared sum.
    """

    # Grad-free parameters kept in fp32 end to end (see MTPDraftModel.frozen_fp32_parameters).
    _frozen_fp32_params = ("expert_bias",)

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.route_norm = config.route_norm
        self.router_scaling_factor = config.router_scaling_factor
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)
        self.use_sigmoid = config.moe_router_use_sigmoid
        self.score_func = getattr(config, "moe_router_score_func", None) or (
            "sigmoid" if self.use_sigmoid else "softmax"
        )
        if self.score_func not in ("sigmoid", "softmax", "sqrtsoftplus"):
            raise ValueError(f"unknown moe_router_score_func {self.score_func!r}")

        H = getattr(config, "routed_expert_hidden_size", None) or config.hidden_size
        inter = config.moe_intermediate_size
        E = self.num_experts
        # Fused expert weights [E, in, out] (transpose of nn.Linear.weight [out, in])
        # so they feed straight into torch._grouped_mm(mat_a=[T, in], mat_b) → [T, out].
        self.experts_gate_proj = FusedExpertProj(E, H, inter)
        self.experts_up_proj = FusedExpertProj(E, H, inter)
        self.experts_down_proj = FusedExpertProj(E, inter, H)
        self._register_load_state_dict_pre_hook(self._load_legacy_fused_experts)
        self.act_fn = nn.SiLU()
        # Match nn.Linear's default init (kaiming_uniform_, a=sqrt(5)) with the
        # correct fan_in: init in the [out, in] layout then transpose into place.
        if not getattr(config, "moe_skip_expert_init", False):
            with torch.no_grad():
                for w, out_dim, in_dim in (
                    (self.experts_gate_proj.weight, inter, H),
                    (self.experts_up_proj.weight, inter, H),
                    (self.experts_down_proj.weight, H, inter),
                ):
                    for e in range(E):
                        tmp = torch.empty(out_dim, in_dim)
                        nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
                        w[e].copy_(tmp.t())

        self.router = nn.Module()
        self.router.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        # MTP construction keeps the router parameter fp32. DFlash later casts
        # its whole FSDP layer to bf16 (FSDP requires a uniform original dtype);
        # forward still evaluates the small router matmul in fp32 below.
        self.router.gate.weight.data = self.router.gate.weight.data.float()
        if config.moe_router_enable_expert_bias:
            # Only steers top-k selection, so it never receives a gradient; DDP
            # (fsdp_strategy REPLICATE) rejects trainable parameters without one.
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
        else:
            self.register_parameter("expert_bias", None)
        self.aux_free_bias_update = bool(getattr(config, "moe_aux_free_bias_update", False))
        self.aux_free_bias_beta = float(getattr(config, "moe_aux_free_bias_beta", 0.9))
        self.aux_free_bias_gamma = float(getattr(config, "moe_aux_free_bias_gamma", 1e-4))
        if self.aux_free_bias_update:
            if self.expert_bias is None:
                raise ValueError("moe_aux_free_bias_update needs moe_router_enable_expert_bias")
            self.register_buffer(
                "_expert_load_counts",
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "expert_load_ema",
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "expert_load_ema_steps",
                torch.zeros((), dtype=torch.int64),
                persistent=False,
            )

        if config.num_shared_experts > 0:
            clamp_shared = getattr(config, "shared_expert_swiglu_clamp", False)
            self.shared_mlp = SwiGLUMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                swiglu_limit=self.swiglu_limit if clamp_shared else 0.0,
            )
        else:
            self.shared_mlp = None
        self.shared_expert_gate = (
            nn.Linear(config.hidden_size, 1, bias=False)
            if getattr(config, "shared_expert_gate", False)
            else None
        )

        if H != config.hidden_size:
            self.routed_expert_down_proj = nn.Linear(config.hidden_size, H, bias=False)
            self.routed_expert_up_proj = nn.Linear(H, config.hidden_size, bias=False)
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_up_proj = None
        self.combine_output_norm = (
            build_mtp_rms_norm(config, config.hidden_size)
            if getattr(config, "sandwich_norm", False)
            else None
        )

    _LEGACY_FUSED_NAMES = ("experts_gate_proj", "experts_up_proj", "experts_down_proj")

    def _load_legacy_fused_experts(self, state_dict, prefix, *_args) -> None:
        for name in self._LEGACY_FUSED_NAMES:
            old_key = prefix + name
            if old_key in state_dict:
                state_dict[old_key + ".weight"] = state_dict.pop(old_key)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)
        n_tokens = x.shape[0]

        # Routing math runs in fp32 for numerical stability.
        logits = F.linear(x.float(), self.router.gate.weight.float())
        if self.score_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.score_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            scores = F.softplus(logits).sqrt()

        scores_for_choice = scores
        if self.expert_bias is not None:
            scores_for_choice = scores + self.expert_bias.float().unsqueeze(0)
        topk_idx = torch.topk(scores_for_choice, self.top_k, dim=-1).indices  # [N, top_k]
        if self.training and self.aux_free_bias_update:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).to(
                    self._expert_load_counts.dtype
                )
                self._expert_load_counts.add_(counts)
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
        routed_x = x if self.routed_expert_down_proj is None else self.routed_expert_down_proj(x)
        if use_reference:
            out = self._routed_reference(routed_x, topk_idx, topk_weights)
        else:
            out = self._routed_grouped(routed_x, topk_idx, topk_weights)
        if self.routed_expert_up_proj is not None:
            out = self.routed_expert_up_proj(out)

        if self.shared_mlp is not None:
            shared = self.shared_mlp(x)
            if self.shared_expert_gate is not None:
                shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
            out = out + shared
        if self.combine_output_norm is not None:
            out = self.combine_output_norm(out)

        if os.environ.get("ANGELSPEC_MTP_ROUTER_DEBUG", "0") not in {"0", "false", "False"}:
            self._log_router_stats(topk_idx, n_tokens)

        return out.view(bsz, seq_len, hidden_dim)

    @torch.no_grad()
    def take_expert_load_counts(self) -> Optional[torch.Tensor]:
        """Return and clear this optimizer step's local hard-routing counts."""
        if not self.aux_free_bias_update:
            return None
        counts = self._expert_load_counts.clone()
        self._expert_load_counts.zero_()
        return counts

    @torch.no_grad()
    def update_expert_load_ema(self, fractions: torch.Tensor) -> torch.Tensor:
        """Update and return the fp32 EMA of globally reduced expert load fractions."""
        fractions = fractions.float()
        if int(self.expert_load_ema_steps.item()) == 0:
            self.expert_load_ema.copy_(fractions)
        else:
            self.expert_load_ema.mul_(self.aux_free_bias_beta).add_(
                fractions, alpha=1.0 - self.aux_free_bias_beta
            )
        self.expert_load_ema_steps.add_(1)
        return self.expert_load_ema

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
        g = self.experts_gate_proj(xs, offs)  # [N*k, I]
        u = self.experts_up_proj(xs, offs)  # [N*k, I]
        if self.swiglu_limit > 0:
            g, u = _clamp_swiglu_inputs(g, u, self.swiglu_limit)
        h = self.act_fn(g) * u
        o = self.experts_down_proj(h, offs)  # [N*k, H]
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
            g = torch.nn.functional.linear(xe, self.experts_gate_proj.weight[e].t())
            u = torch.nn.functional.linear(xe, self.experts_up_proj.weight[e].t())
            if self.swiglu_limit > 0:
                g, u = _clamp_swiglu_inputs(g, u, self.swiglu_limit)
            ye = torch.nn.functional.linear(
                self.act_fn(g) * u, self.experts_down_proj.weight[e].t()
            )
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


def _interleaved_to_split_rope(x: torch.Tensor) -> torch.Tensor:
    """Permute interleaved RoPE layout [re0,im0,re1,im1,...] to split layout
    [re0,re1,...,im0,im1,...] (shape-preserving), matching the HYV4 checkpoint.

    HYV4 stores q_pe/k_pe RoPE rows interleaved (Megatron ordering); the split
    layout is what ``rotate_half``-based ``apply_rotary_pos_emb`` expects. Doing
    the permute at runtime lets us load ``q_b_proj``/``kv_a_proj_with_mqa``
    weights bit-for-bit as stored.
    """
    return x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).transpose(-2, -1).reshape(x.shape)


def _apply_rope_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE (fp32) to a [b, h, s, rope_dim] tensor. cos/sin are [b, s, rope_dim]."""
    dtype = x.dtype
    cos = cos.to(torch.float32).unsqueeze(1)
    sin = sin.to(torch.float32).unsqueeze(1)
    xf = x.to(torch.float32)
    return ((xf * cos) + (_rotate_half_split(xf) * sin)).to(dtype)


def _rotate_half_split(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """fp32 ``x`` as vLLM's DSA indexer scores it: e4m3 values with one power-of-two
    (ue8m0) scale per last-dim vector, as its q and k quantization produce."""
    x = x.float()
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-10)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale


def _hadamard_fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """fp32 ``x`` as Hy4 flash's compressed indexer scores it in fp8: rotated by the orthonormal
    Hadamard matrix over the last dim (which leaves q.k unchanged), then e4m3 with one
    power-of-two scale per vector. The values are exact in bf16."""
    dim = x.shape[-1]
    hadamard = torch.ones(1, 1, device=x.device)
    while hadamard.shape[0] < dim:
        hadamard = torch.cat(
            [torch.cat([hadamard, hadamard], 1), torch.cat([hadamard, -hadamard], 1)], 0
        )
    rotated = x.float() @ (hadamard * dim**-0.5)
    amax = rotated.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (rotated / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale


class MTPIndexer(nn.Module):
    """DSA (DeepSeek Sparse Attention) indexer: selects top-k key positions.

    Weight names / math mirror the HYV4 ``self_attn.indexer`` so a pretrained
    MTP head loads directly. Returns an additive block-0 mask [b, s, s] (0 for
    kept keys, -inf for dropped). When ``seq_len < index_topk`` every causal key
    survives top-k, so this returns all-zeros (dense) — the pretrained indexer
    weights are still loaded but inactive at short/medium context.
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def project_keys(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        k = self.wk(hidden_states)
        k = F.layer_norm(
            k.to(torch.float32),
            self.k_norm.normalized_shape,
            self.k_norm.weight.to(torch.float32),
            self.k_norm.bias.to(torch.float32),
            self.k_norm.eps,
        ).to(k.dtype)
        k_nope, k_pe = torch.split(
            k, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        k_pe = _interleaved_to_split_rope(k_pe.unsqueeze(2)).transpose(1, 2)
        k_pe = _apply_rope_split(k_pe, cos, sin).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1)

    @torch.no_grad()
    def sparse_indices(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        doc_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        precomputed_keys: Optional[torch.Tensor] = None,
        query_offset: int = 0,
        use_fp8: bool = False,
        kernel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        q = self.wq_b(q_resid).view(bsz, seq_len, self.n_heads, self.head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = _interleaved_to_split_rope(q_pe).transpose(1, 2)
        q_pe = _apply_rope_split(q_pe, cos, sin).transpose(1, 2)
        q = torch.cat([q_nope, q_pe], dim=-1)

        k = (
            precomputed_keys
            if precomputed_keys is not None
            else self.project_keys(hidden_states, cos, sin)
        )
        weights = (
            self.weights_proj(hidden_states).float() * (self.n_heads**-0.5) * self.softmax_scale
        )
        if use_fp8:
            q, k = _fp8_round_trip(q), _fp8_round_trip(k)
        lightning_indexer = tilelang_lightning_indexer if kernel else torch_lightning_indexer

        total_len = k.shape[1]
        all_indices = []
        for batch_idx in range(bsz):
            docs = doc_ids[batch_idx] if doc_ids is not None else None
            valid = valid_mask[batch_idx] if valid_mask is not None else None
            starts, ends = build_contiguous_causal_ranges(
                total_len,
                query_offset=query_offset,
                query_len=seq_len,
                doc_ids=docs,
                valid_mask=valid,
                device=hidden_states.device,
            )
            _, indices = lightning_indexer(
                q[batch_idx].to(torch.bfloat16),
                k[batch_idx].to(torch.bfloat16),
                weights[batch_idx],
                starts.squeeze(0),
                ends.squeeze(0),
                self.index_topk,
            )
            all_indices.append(indices)

        indices = torch.stack(all_indices, dim=0)
        return indices, (indices < 0).any(-1, keepdim=True)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        return_indices: bool = False,
        doc_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        precomputed_keys: Optional[torch.Tensor] = None,
        query_offset: int = 0,
        use_fp8: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.wq_b(q_resid).view(bsz, seq_len, self.n_heads, self.head_dim)
        q_nope, q_pe = torch.split(
            q, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        # RoPE on q_pe: [b, s, h, d] -> [b, h, s, d] -> rotate -> back
        q_pe = _interleaved_to_split_rope(q_pe).transpose(1, 2)
        q_pe = _apply_rope_split(q_pe, cos, sin).transpose(1, 2)
        q = torch.cat([q_nope, q_pe], dim=-1)

        k = (
            precomputed_keys
            if precomputed_keys is not None
            else self.project_keys(hidden_states, cos, sin)
        )
        q_score, k_score = (_fp8_round_trip(q), _fp8_round_trip(k)) if use_fp8 else (q, k)

        weights = (
            self.weights_proj(hidden_states).float() * (self.n_heads**-0.5) * self.softmax_scale
        )  # [b, s, n_heads]
        # scores [b,s,h,t] and index_scores [b,s,t] are O(seq^2), and Ulysses shards
        # heads (not the sequence) so each rank scores the full 128k grid + a dense
        # [s,t] causal mask -> tens of GiB, OOM. Each query row's top-k is independent,
        # so tile over the query dim (identical math, bounded peak) and build the
        # causal add-mask per chunk instead of a full [s,s] tensor.
        total_len = k.shape[1]
        topk = min(self.index_topk, total_len)
        # scores_c is [b, chunk, n_heads, total_len] fp32; bound it (~2 GiB) so the
        # per-chunk transient stays small at long context.
        chunk = max(1, min(2048, (512 << 20) // max(1, self.n_heads * total_len * 4)))
        is_4d = attention_mask is not None and attention_mask.dim() == 4
        neg = torch.finfo(q.dtype).min
        # A masked key gets ~finfo.min added (dwarfing any finite score, whose sign is
        # not fixed since the head weights are unconstrained), so a row whose top-k
        # still contains such an entry has fewer than top-k valid keys -> "all open".
        # Compare against finfo.min/2, NOT float("-inf"): the additive masks use
        # finfo.min (see base._make_causal_mask), never -inf.
        open_thresh = neg / 2

        def _causal_add(start, end):
            if is_4d:
                return attention_mask[:, 0, start:end, :]
            qi = torch.arange(
                query_offset + start,
                query_offset + end,
                device=hidden_states.device,
            )[:, None]
            kj = torch.arange(total_len, device=hidden_states.device)[None, :]
            add = torch.where(
                kj > qi,
                torch.full((), neg, dtype=q.dtype, device=hidden_states.device),
                torch.zeros((), dtype=q.dtype, device=hidden_states.device),
            )[
                None
            ]  # [1, chunk, total_len]
            if valid_mask is not None:
                add = add + torch.where(
                    valid_mask[:, None, :].to(torch.bool),
                    torch.zeros((), dtype=q.dtype, device=hidden_states.device),
                    torch.full((), neg, dtype=q.dtype, device=hidden_states.device),
                )
            if doc_ids is not None:
                same = (
                    doc_ids[:, query_offset + start : query_offset + end, None]
                    == doc_ids[:, None, :]
                )
                add = add + torch.where(
                    same,
                    torch.zeros((), dtype=q.dtype, device=hidden_states.device),
                    torch.full((), neg, dtype=q.dtype, device=hidden_states.device),
                )
            return add  # [b, chunk, total_len]

        def _chunk_topk(start, end):
            sc = F.relu(
                torch.einsum("bshd,btd->bsht", q_score[:, start:end].float(), k_score.float())
            )
            isc = torch.einsum("bsht,bsh->bst", sc, weights[:, start:end])
            isc = isc + _causal_add(start, end)
            return isc.topk(topk, dim=-1)

        if return_indices:
            idx_parts, open_parts = [], []
            for start in range(0, seq_len, chunk):
                end = min(start + chunk, seq_len)
                ts, ti = _chunk_topk(start, end)
                # int32 halves the persistent [b, S, topk] index (positions < 2^31);
                # gather callers cast the small per-chunk slice back to int64.
                allowed = ts > open_thresh
                idx_parts.append(torch.where(allowed, ti, -1).to(torch.int32))
                open_parts.append((~allowed).any(-1, keepdim=True))
            return torch.cat(idx_parts, dim=1), torch.cat(open_parts, dim=1)

        neg_inf = torch.finfo(hidden_states.dtype).min
        out_mask = torch.full(
            (bsz, seq_len, total_len),
            neg_inf,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            ts, ti = _chunk_topk(start, end)
            om = out_mask[:, start:end]
            om.scatter_(-1, ti, 0.0)
            # Positions whose whole causal window is < topk keep every key (top-k of
            # an all-masked row would scatter garbage); force those rows fully open.
            out_mask[:, start:end] = torch.where(
                (ts <= open_thresh).any(-1, keepdim=True), torch.zeros_like(om), om
            )
        return out_mask


class MTPCompressedIndexer(nn.Module):
    """Hy4 flash two-level compressed DSA indexer (``self_attn.compressed_indexer``).

    Weight names and math mirror vLLM's ``FlashCompressedIndexer`` (``select_groups`` +
    ``expand_groups``) with ``compressed_indexer_skip_fine_select``: token keys are pooled
    per document in groups of ``compress_ratio`` with a softmax gate, each query scores
    the complete groups before its own with the lightning indexer, and the
    ``group_topk - 1`` recalled groups expanded to tokens, plus the query's current
    (possibly partial) group, are its sparse-attention keys. The coarse query reads the
    same post-``q_a_layernorm`` LoRA residual as the attention. Grad-free: everything is
    read from the target MTP layer.
    """

    _frozen_fp32_params = (
        "key_value_projection.weight",
        "key_gate_projection.weight",
        "key_norm.weight",
        "intra_group_ape",
    )

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.ratio = config.compressed_indexer_compress_ratio
        self.n_heads = config.compressed_indexer_num_attention_heads
        self.head_dim = config.compressed_indexer_head_dim
        self.key_input_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.group_topk = config.compressed_indexer_group_topk
        self.index_topk = config.compressed_indexer_candidate_topk
        eps = config.rms_norm_eps
        self.query_projection = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.head_weight_projection = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.token_key_projection = nn.Linear(config.hidden_size, self.key_input_dim, bias=False)
        self.token_key_norm = nn.RMSNorm(self.key_input_dim, eps=eps)
        self.key_value_projection = nn.Linear(self.key_input_dim, self.head_dim, bias=False)
        self.key_gate_projection = nn.Linear(self.key_input_dim, self.head_dim, bias=False)
        self.key_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.intra_group_ape = nn.Parameter(torch.zeros(self.ratio, self.head_dim))
        self.requires_grad_(False)

    @torch.no_grad()
    def project_keys(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Per-token key inputs ``[b, s, key_input_dim + 2 * rope_dim]`` (fp32).

        Groups are pooled over the full sequence, so under USP the token keys are
        gathered together with their cos/sin, which ``sparse_indices`` unpacks.
        """
        keys = self.token_key_norm(self.token_key_projection(hidden_states)).to(torch.bfloat16)
        return torch.cat([keys.float(), cos.float(), sin.float()], dim=-1)

    def _pool(self, token_keys: torch.Tensor, cu_seqlens: torch.Tensor):
        compressed_cu = compute_compressed_cu_seqlens(cu_seqlens, self.ratio)
        group_starts = compressed_group_token_offsets(cu_seqlens, self.ratio)
        offsets = torch.arange(self.ratio, device=token_keys.device, dtype=torch.long)
        retained = token_keys[(group_starts.unsqueeze(-1) + offsets).flatten()].float()
        values = F.linear(retained, self.key_value_projection.weight.float())
        gates = F.linear(retained, self.key_gate_projection.weight.float())
        values = values.view(-1, self.ratio, self.head_dim)
        gates = gates.view(-1, self.ratio, self.head_dim) + self.intra_group_ape.float()
        pooled = (values * gates.softmax(dim=1)).sum(dim=1)
        pooled = F.rms_norm(
            pooled, (self.head_dim,), self.key_norm.weight.float(), self.key_norm.eps
        )
        return pooled.to(torch.bfloat16), compressed_cu, group_starts

    def _rope_last(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Interleaved->split RoPE on the last ``rope_dim`` channels of ``[b, s, h, d]``."""
        nope, pe = torch.split(x, [x.shape[-1] - self.rope_dim, self.rope_dim], dim=-1)
        pe = _interleaved_to_split_rope(pe).transpose(1, 2)
        pe = _apply_rope_split(pe, cos, sin).transpose(1, 2)
        return torch.cat([nope, pe], dim=-1)

    @torch.no_grad()
    def sparse_indices(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        doc_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        precomputed_keys: Optional[torch.Tensor] = None,
        query_offset: int = 0,
        query_positions: Optional[torch.Tensor] = None,
        use_fp8: bool = False,
        kernel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Candidate token indices per query row, ``[b, s, candidate_topk]`` int32 (``-1`` unused).

        Query row ``i`` sits at key position ``query_offset + i`` unless ``query_positions``
        (``[b, s]`` long) places each row at its own position, so a caller can score a few
        rows of a long sequence without scoring all of them.

        ``use_fp8`` scores q and the pooled keys as the fp8 inference indexer quantizes them
        (:func:`_hadamard_fp8_round_trip`), through the bf16 kernel.
        """
        bsz, seq_len, _ = hidden_states.shape
        key_inputs = (
            precomputed_keys
            if precomputed_keys is not None
            else self.project_keys(hidden_states, cos, sin)
        )
        total_len = key_inputs.shape[1]
        token_keys, key_cos, key_sin = key_inputs.split(
            [self.key_input_dim, self.rope_dim, self.rope_dim], dim=-1
        )
        q = self.query_projection(q_resid).view(bsz, seq_len, self.n_heads, self.head_dim)
        q = self._rope_last(q, cos, sin).to(torch.bfloat16)
        if use_fp8:
            q = _hadamard_fp8_round_trip(q).to(torch.bfloat16)
        weights = self.head_weight_projection(hidden_states).float() * (
            self.n_heads**-0.5 * self.head_dim**-0.5
        )
        device = hidden_states.device
        positions = torch.arange(total_len, device=device, dtype=torch.long)
        if query_positions is None:
            query_positions = positions[query_offset : query_offset + seq_len].expand(bsz, -1)
        else:
            query_positions = query_positions.to(device=device, dtype=torch.long)

        all_indices = []
        for b in range(bsz):
            starts, ends = build_contiguous_causal_ranges(
                total_len,
                doc_ids=doc_ids[b] if doc_ids is not None else None,
                valid_mask=valid_mask[b] if valid_mask is not None else None,
                device=device,
            )
            starts, ends = starts.squeeze(0).long(), ends.squeeze(0).long()
            # Documents of valid tokens; every invalid token is its own segment so
            # no pooled group mixes padding into a document.
            segment = torch.where(ends > starts, starts, positions)
            boundary = torch.ones_like(segment, dtype=torch.bool)
            boundary[1:] = segment[1:] != segment[:-1]
            cu_seqlens = torch.cat([positions[boundary], positions.new_tensor([total_len])]).to(
                torch.int32
            )

            pooled, compressed_cu, group_starts = self._pool(token_keys[b], cu_seqlens)
            pooled = self._rope_last(
                pooled.view(1, -1, 1, self.head_dim),
                key_cos[b, group_starts].unsqueeze(0),
                key_sin[b, group_starts].unsqueeze(0),
            ).view(-1, self.head_dim)
            if use_fp8:
                pooled = _hadamard_fp8_round_trip(pooled)

            q_starts, q_ends = starts[query_positions[b]], ends[query_positions[b]]
            sample = torch.searchsorted(
                cu_seqlens[1:].long().contiguous(), query_positions[b], right=True
            ).clamp(max=cu_seqlens.numel() - 2)
            middle_starts, middle_ends = compressed_middle_group_ranges(
                q_starts, q_ends, compressed_cu.long()[:-1][sample], self.ratio
            )
            if pooled.shape[0] == 0:
                groups = torch.full(
                    (seq_len, self.group_topk - 1), -1, dtype=torch.long, device=device
                )
            else:
                lightning_indexer = (
                    tilelang_lightning_indexer if kernel else torch_lightning_indexer
                )
                _, groups = lightning_indexer(
                    q[b],
                    pooled.to(torch.bfloat16),
                    weights[b],
                    middle_starts,
                    middle_ends,
                    self.group_topk - 1,
                )
            tail = build_forced_tail_candidates(q_starts, q_ends, self.ratio)
            candidates = torch.cat(
                [expand_compressed_groups_to_tokens(groups.long(), cu_seqlens, self.ratio), tail],
                dim=-1,
            )
            all_indices.append(candidates.sort(dim=-1).values.to(torch.int32))

        indices = torch.stack(all_indices, dim=0)
        return indices, (indices < 0).any(-1, keepdim=True)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("the compressed indexer runs only on the absorbed cached path")


class MTPMLAAttention(nn.Module):
    """Gated-MLA (+ optional DSA indexer) attention with the MTP per-step KV cache.

    Faithful to the HYV4 ``mtp_layers.*.self_attn``: LoRA q/kv projections,
    interleaved->split RoPE on the rope slice, elementwise sigmoid gate, and a
    per-head learnable attention sink. The cached (TTT) path reuses the same
    block-0-causal + per-step-diagonal-suffix merge as :class:`MTPAttention`,
    but stores decompressed per-head K/V so no MLA-specific cache math is needed.
    The DSA indexer only masks the block-0 (full-sequence) term and only when
    ``seq_len >= index_topk``; the diagonal suffix terms are unaffected.

    ``return_index`` hands the TTT loop the DSA state that later steps take back as
    ``precomputed_index``. With ``dsa_index_share`` (default) that is the top-k
    selection, which later steps reuse; without it, it is the step-0 indexer keys,
    since block 0 holds the step-0 K/V, and every step selects its own top-k
    against them.
    """

    def __init__(self, config: MTPConfig, attention_backend: str = "sdpa"):
        super().__init__()
        self.config = config
        self.attention_backend = validate_mtp_backend(attention_backend)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = build_mtp_rms_norm(config, self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = build_mtp_rms_norm(config, self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        self.scaling = self.qk_head_dim ** (-0.5)

        self.gated_mla = config.gated_mla
        if self.gated_mla:
            self.gate_projection_size = 1 if config.gating_type == "headwise" else self.v_head_dim
            self.linear_gate = nn.Linear(
                self.hidden_size, self.num_heads * self.gate_projection_size, bias=False
            )

        self.learnable_sink = config.learnable_sink
        if self.learnable_sink:
            self.learnable_sink_param = nn.Parameter(
                torch.full(
                    (self.num_heads,), float(config.learnable_sink_init), dtype=torch.float32
                )
            )

        self.use_indexer = config.index_topk is not None
        self.use_compressed_indexer = bool(getattr(config, "use_compressed_indexer", False))
        if self.use_compressed_indexer:
            self.compressed_indexer = MTPCompressedIndexer(config)
        elif self.use_indexer:
            self.indexer = MTPIndexer(config)
        self.post_proj_norm = (
            build_mtp_rms_norm(config, self.hidden_size)
            if getattr(config, "sandwich_norm", False)
            else None
        )
        self.dsa_backend = getattr(config, "dsa_backend", "torch")
        self.fp8_indexer = bool(getattr(config, "dsa_fp8_indexer", False))
        self.dsa_index_share = bool(getattr(config, "dsa_index_share", True))
        if self.dsa_backend == "tilelang":
            if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
                raise ValueError(
                    "tilelang sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64"
                )
        if self.dsa_backend in ("tilelang", "torch_absorbed") and not self.use_indexer:
            raise ValueError(f"{self.dsa_backend} sparse MLA requires a DSA indexer")

        # Pure-Ulysses SP: seq-shard -> head-shard around attention so the MLA
        # backbone attends across the full sequence (resolved lazily, like GQA).
        self._usp_group = None
        self._usp_size = 1
        # Under USP the softmax sees only this rank's head shard, so the per-head
        # learnable sink must be sliced to those heads (set transiently in forward).
        self._head_slice = slice(None)

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.uses_mrope = bool(rope_parameters.get("mrope_section"))
        if self.qk_rope_head_dim == 0:
            if self.dsa_backend not in ("tilelang", "torch_absorbed"):
                raise ValueError("MLA without RoPE needs the absorbed cached path")
            self.rotary_emb = None
        elif self.uses_mrope:
            self.rotary_emb = MRotaryEmbedding(self.qk_rope_head_dim, rope_parameters)
        else:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000),
            )

    @property
    def indexer(self) -> nn.Module:
        """The DSA indexer, registered as ``indexer`` (Hy4) or ``compressed_indexer``
        (Hy4 flash) so its weights keep the checkpoint names."""
        name = "compressed_indexer" if self.use_compressed_indexer else "indexer"
        if name not in self._modules:
            raise AttributeError(f"{type(self).__name__} has no DSA indexer")
        return self._modules[name]

    def _output_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        output = self.o_proj(attn_output)
        if self.post_proj_norm is not None:
            output = self.post_proj_norm(output)
        return output

    def _cos_sin(self, ref: torch.Tensor, position_ids: torch.Tensor, seq_len: int):
        """Return per-position (cos, sin) of shape [b, s, rope_dim] for given position_ids."""
        if self.rotary_emb is None:
            empty = ref.new_zeros(position_ids.shape[0], position_ids.shape[-1], 0)
            return empty, empty
        if self.uses_mrope:
            return self.rotary_emb(ref, position_ids)
        if position_ids.ndim != 2:
            raise ValueError("3D positions require MRoPE configuration")
        cache_len = max(int(seq_len), int(position_ids.max().item()) + 1)
        cos, sin = self.rotary_emb(ref, seq_len=cache_len)
        cos = cos.squeeze(1).squeeze(0)  # [max_seq, rope_dim]
        sin = sin.squeeze(1).squeeze(0)
        return cos[position_ids], sin[position_ids]  # [b, s, rope_dim]

    def _project(self, hidden_states: torch.Tensor):
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is not None:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_resid)
        else:
            q_resid = None
            q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)  # [b, h, s, nope]
        value = value.transpose(1, 2)  # [b, h, s, v]
        k_pe = k_pe.view(bsz, 1, q_len, self.qk_rope_head_dim)  # [b, 1, s, rope]
        return q_nope, q_pe, k_nope, k_pe, value, q_resid

    def _assemble(self, q_nope, q_pe, k_nope, k_pe, cos, sin):
        q_pe = _apply_rope_split(_interleaved_to_split_rope(q_pe), cos, sin)
        k_pe = _apply_rope_split(_interleaved_to_split_rope(k_pe), cos, sin)
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)
        return query_states, key_states

    def _kv_b_weights(self):
        from torch.distributed.tensor import DTensor

        weight = self.kv_b_proj.weight
        if isinstance(weight, DTensor):
            # full_tensor() treats the gradient as replicated, so a sharded weight
            # here would keep only this rank's own contribution to its shard.
            if torch.is_grad_enabled() and weight.requires_grad:
                raise RuntimeError(
                    "kv_b_proj.weight is sharded in a gradient-enabled forward: the "
                    "absorbed MLA reads it without calling kv_b_proj.forward, so it must "
                    "belong to an FSDP unit whose forward runs (see mtp_linear_shard_units)"
                )
            weight = weight.full_tensor()
        weight = weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        return torch.split(
            weight,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=1,
        )

    def _project_absorbed(self, hidden_states: torch.Tensor):
        bsz, q_len, _ = hidden_states.shape
        if self.q_lora_rank is not None:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_resid)
        else:
            q_resid = None
            q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_compressed = self.kv_a_layernorm(kv_compressed)
        return q_nope, q_pe, kv_compressed, k_pe, q_resid

    def _assemble_absorbed(self, q_nope, q_pe, kv_compressed, k_pe, cos, sin):
        w_kc, w_vc = self._kv_b_weights()
        q_nope = torch.einsum("bshd,hdm->bshm", q_nope, w_kc)
        if self.rotary_emb is None:
            return q_nope, kv_compressed.unsqueeze(2), w_vc
        q_pe = q_pe.transpose(1, 2)
        q_pe = _apply_rope_split(_interleaved_to_split_rope(q_pe), cos, sin)
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.unsqueeze(1)
        k_pe = _apply_rope_split(_interleaved_to_split_rope(k_pe), cos, sin)
        k_pe = k_pe.transpose(1, 2)

        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = torch.cat([kv_compressed.unsqueeze(2), k_pe], dim=-1)
        return q, kv, w_vc

    def _absorbed_cached_forward(
        self,
        hidden_states,
        cache_keys,
        cache_values,
        attention_mask,
        position_ids,
        ctx_doc_ids,
        precomputed_index,
    ):
        bsz, q_len, _ = hidden_states.shape
        if self.attention_backend == "usp" and self._usp_group is None:
            self._usp_group = get_sp_ulysses_group()
            self._usp_size = (
                dist.get_world_size(self._usp_group) if self._usp_group is not None else 1
            )
        usp = self.attention_backend == "usp" and self._usp_size > 1
        full_seq_len = q_len * self._usp_size if usp else q_len
        cached_steps = 0 if cache_keys is None else cache_keys.shape[1] // full_seq_len

        q_nope, q_pe, kv_compressed, k_pe, q_resid = self._project_absorbed(hidden_states)
        cos, sin = self._cos_sin(
            q_nope,
            advance_mtp_positions(position_ids, cached_steps),
            full_seq_len + cached_steps,
        )
        q, kv, w_vc = self._assemble_absorbed(
            q_nope,
            q_pe,
            kv_compressed,
            k_pe,
            cos,
            sin,
        )

        if attention_mask is None:
            valid_local = torch.ones(
                bsz,
                q_len,
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            valid_local = mtp_validity_mask(attention_mask, bsz, q_len, hidden_states.device)

        if usp:
            q = _mtp_ulysses_all_to_all(q.transpose(1, 2), self._usp_group, 1, 2)
            q = q.transpose(1, 2).contiguous()
            kv = _mtp_seq_all_gather(kv, self._usp_group)
            valid_full = self._usp_gather_seq(valid_local)
            doc_full = (
                self._usp_gather_seq(ctx_doc_ids.contiguous().long())
                if ctx_doc_ids is not None
                else None
            )
            rank = dist.get_rank(self._usp_group)
            h_local = q.shape[2]
            head_slice = slice(rank * h_local, (rank + 1) * h_local)
        else:
            valid_full = valid_local
            doc_full = ctx_doc_ids
            rank = 0
            h_local = self.num_heads
            head_slice = slice(None)

        index_state = precomputed_index
        computed_index = precomputed_index if self.dsa_index_share else None
        if computed_index is None:
            index_keys = self._dsa_index_keys(hidden_states, cos, sin, precomputed_index, usp)
            local_index = self.indexer.sparse_indices(
                hidden_states,
                q_resid,
                cos,
                sin,
                doc_ids=doc_full,
                valid_mask=valid_full,
                precomputed_keys=index_keys,
                query_offset=rank * q_len,
                use_fp8=self.fp8_indexer,
                kernel=self.dsa_backend == "tilelang",
            )
            if usp:
                computed_index = (
                    self._usp_gather_seq(local_index[0]),
                    self._usp_gather_seq(local_index[1]),
                )
            else:
                computed_index = local_index
            index_state = computed_index if self.dsa_index_share else index_keys

        cache_keys = kv if cache_keys is None else torch.cat([cache_keys, kv], dim=1)
        cache_values = cache_keys.new_empty(0)
        num_cached_steps = cache_keys.shape[1] // full_seq_len
        indices = append_diagonal_kv_indices(
            computed_index[0],
            seq_len=full_seq_len,
            num_cached_steps=num_cached_steps,
        )

        sink = self.learnable_sink_param[head_slice] if self.learnable_sink else None
        if self.dsa_backend == "torch_absorbed":
            result = torch_sparse_mla(
                q.to(torch.bfloat16),
                cache_keys.to(torch.bfloat16),
                indices,
                scaling=self.scaling,
                d_v=self.kv_lora_rank,
                sink=sink,
            )
        else:
            result = tilelang_sparse_mla(
                q.to(torch.bfloat16),
                cache_keys.to(torch.bfloat16),
                indices,
                scaling=self.scaling,
                sink=sink,
            )
        latent_output = result[0]
        local_w_vc = w_vc[head_slice]
        attn_output = torch.einsum(
            "bshm,hdm->bshd",
            latent_output.to(local_w_vc.dtype),
            local_w_vc,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        if usp:
            attn_output = _mtp_ulysses_all_to_all(
                attn_output,
                self._usp_group,
                2,
                1,
            )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(
                bsz,
                q_len,
                self.num_heads * self.v_head_dim,
            )
        )
        if self.gated_mla:
            gate = self.linear_gate(hidden_states).view(
                bsz,
                q_len,
                self.num_heads,
                self.gate_projection_size,
            )
            attn_output = (
                attn_output.view(bsz, q_len, self.num_heads, self.v_head_dim) * torch.sigmoid(gate)
            ).reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self._output_proj(attn_output), cache_keys, cache_values, index_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        precomputed_index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_index: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.dsa_backend in ("tilelang", "torch_absorbed") and use_cache:
            output, cache_keys, cache_values, index_out = self._absorbed_cached_forward(
                hidden_states,
                cache_keys,
                cache_values,
                attention_mask,
                position_ids,
                ctx_doc_ids,
                precomputed_index,
            )
            if return_index:
                return output, cache_keys, cache_values, index_out
            return output, cache_keys, cache_values

        bsz, q_len, _ = hidden_states.size()
        q_nope, q_pe, k_nope, k_pe, value_states, q_resid = self._project(hidden_states)

        lck = 0 if cache_keys is None else cache_keys.shape[2]
        cos, sin = self._cos_sin(q_nope, advance_mtp_positions(position_ids, lck), q_len + lck)
        query_states, key_states = self._assemble(q_nope, q_pe, k_nope, k_pe, cos, sin)

        if self.attention_backend == "usp" and self._usp_group is None:
            self._usp_group = get_sp_ulysses_group()
            self._usp_size = (
                dist.get_world_size(self._usp_group) if self._usp_group is not None else 1
            )
        usp = self.attention_backend == "usp" and self._usp_size > 1

        if not usp:
            attention_mask = mtp_causal_mask(
                attention_mask, ctx_doc_ids, bsz, q_len, hidden_states.device, hidden_states.dtype
            )
        elif not use_cache:
            raise ValueError("MTP MLA USP requires use_cache=True")

        # Non-USP: the indexer runs on the (full) local sequence. Under USP the local
        # tensor is only a seq shard, so the index is (re)computed on the gathered
        # full sequence inside _usp_cached_forward instead.
        reuse_index = self.dsa_index_share and precomputed_index is not None
        computed_index = None
        index_state = precomputed_index
        if not usp and self.use_indexer and q_len >= self.indexer.index_topk and not reuse_index:
            index_keys = self._dsa_index_keys(hidden_states, cos, sin, precomputed_index, usp)
            computed_index = self.indexer(
                hidden_states,
                q_resid,
                cos,
                sin,
                attention_mask,
                return_indices=True,
                precomputed_keys=index_keys,
            )
            index_state = computed_index if self.dsa_index_share else index_keys

        active_index = precomputed_index if reuse_index else computed_index

        if not use_cache:
            attn_output = self._sdpa_block0(
                query_states,
                key_states,
                value_states,
                attention_mask,
                hidden_states,
                q_resid,
                cos,
                sin,
                active_index,
            )
        elif usp:
            attn_output, cache_keys, cache_values, index_state = self._usp_cached_forward(
                query_states,
                key_states,
                value_states,
                cache_keys,
                cache_values,
                hidden_states,
                q_resid,
                cos,
                sin,
                ctx_doc_ids,
                attention_mask,
                precomputed_index,
                q_len,
            )
        else:
            if cache_keys is None:
                cache_keys = key_states.unsqueeze(2)
                cache_values = value_states.unsqueeze(2)
            else:
                cache_keys = torch.cat([cache_keys, key_states.unsqueeze(2)], dim=2)
                cache_values = torch.cat([cache_values, value_states.unsqueeze(2)], dim=2)
            nc = cache_keys.shape[2]
            k0, v0 = cache_keys[:, :, 0], cache_values[:, :, 0]

            use_sparse = (
                self.use_indexer and q_len >= self.indexer.index_topk and active_index is not None
            )
            if use_sparse:
                topk_idx, all_open = active_index
                chain_keys = [cache_keys[:, :, i] for i in range(1, nc)]
                chain_values = [cache_values[:, :, i] for i in range(1, nc)]
                attn_output = self._sparse_cached_attn(
                    query_states,
                    k0,
                    v0,
                    topk_idx,
                    all_open,
                    chain_keys,
                    chain_values,
                    attention_mask,
                )
            else:
                chain_keys = [cache_keys[:, :, i] for i in range(1, nc)]
                chain_values = [cache_values[:, :, i] for i in range(1, nc)]
                attn_output = self._dense_block0(
                    query_states, k0, v0, chain_keys, chain_values, attention_mask
                )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        )
        if self.gated_mla:
            gate = self.linear_gate(hidden_states).view(
                bsz, q_len, self.num_heads, self.gate_projection_size
            )
            attn_output = (
                attn_output.view(bsz, q_len, self.num_heads, self.v_head_dim) * torch.sigmoid(gate)
            ).reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self._output_proj(attn_output)
        if return_index:
            return attn_output, cache_keys, cache_values, index_state
        return attn_output, cache_keys, cache_values

    def _softmax_with_sink(self, logits: torch.Tensor, head_slice=None):
        """Softmax over the last dim; if learnable_sink, fold a per-head sink logit
        into the denominator (equivalent to appending one extra key with score
        ``sink_param`` whose value is 0). Returns (probs, sink_scale) where
        sink_scale downweights the output by the sink's probability mass.

        ``head_slice`` pins this rank's head range explicitly: a sparse-chunk
        activation-checkpoint recompute runs in backward, AFTER _usp_cached_forward
        has reset ``self._head_slice``, so the recompute must be told which heads to
        slice the sink to rather than reading the (stale) instance attribute.
        """
        if not self.learnable_sink:
            alive = torch.isfinite(logits).any(dim=-1, keepdim=True)
            safe_logits = torch.where(alive, logits, torch.zeros_like(logits))
            probs = F.softmax(safe_logits, dim=-1, dtype=torch.float32).to(logits.dtype)
            return torch.where(alive, probs, torch.zeros_like(probs)), None
        hs = self._head_slice if head_slice is None else head_slice
        lf = logits.to(torch.float32)
        sink = self.learnable_sink_param[hs].to(torch.float32).view(1, -1, 1, 1)
        total = torch.logsumexp(
            torch.cat([lf, sink.expand(*lf.shape[:-1], 1)], dim=-1), dim=-1, keepdim=True
        )
        probs = torch.exp(lf - total).to(logits.dtype)
        return probs, None

    def _dense_block0(self, query_states, k0, v0, chain_keys, chain_values, attention_mask):
        """Dense block-0 (+ per-step diagonal chain) attention with joint softmax.

        Shared by the non-sparse path and the DSA ``all_open`` fallback so both
        apply the same causal(+doc) additive mask; chain terms are the
        position-diagonal dot ``q_i · k_i``, never a full cross-product.
        """
        b0 = k0.shape[2]
        logits = torch.matmul(query_states, k0.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            logits = logits + attention_mask
        for ck in chain_keys:
            wi = (query_states * ck).sum(-1) * self.scaling
            logits = torch.cat((logits, wi[..., None]), dim=-1)
        probs, sink_scale = self._softmax_with_sink(logits)
        out = torch.matmul(probs[..., :b0], v0)
        for j, cv in enumerate(chain_values):
            out = out + probs[..., b0 + j][..., None] * cv
        if sink_scale is not None:
            out = out * sink_scale
        return out

    def _build_causal_doc_mask(self, valid, doc_ids, n, device, dtype):
        """Additive [b, 1, n, n] block-0 mask: causal + padding(valid) + doc-gate.

        Uses ``finfo.min`` fill (matching ``base._make_causal_mask``) so the indexer's
        all-open threshold and the dense mask agree on which keys are masked.
        """
        neg = torch.finfo(dtype).min
        idx = torch.arange(n, device=device)
        causal = idx[None, :] > idx[:, None]  # [n, n]: True == future key
        mask = torch.zeros(1, 1, n, n, dtype=dtype, device=device).masked_fill(
            causal[None, None], neg
        )
        if valid is not None:
            v = valid[:, :n].to(torch.bool)
            mask = mask.masked_fill(~v[:, None, None, :], neg)
        if doc_ids is not None:
            d = doc_ids[:, :n]
            same = d[:, :, None] == d[:, None, :]
            mask = mask.masked_fill(~same[:, None], neg)
        return mask

    def _usp_gather_seq(self, x):
        """All-gather a seq-sharded [b, S_local, *] tensor into [b, S_full, *] in
        rank order. Non-differentiable: only feeds the no-grad indexer and masks."""
        xc = x.contiguous()
        b = xc.shape[0]
        tail = tuple(xc.shape[2:])
        out = torch.empty((self._usp_size * b,) + xc.shape[1:], dtype=xc.dtype, device=xc.device)
        dist.all_gather_into_tensor(out, xc, group=self._usp_group)
        out = out.view((self._usp_size, b) + xc.shape[1:]).movedim(0, 1)
        return out.reshape((b, self._usp_size * xc.shape[1]) + tail)

    def _dsa_index_keys(self, hidden_states, cos, sin, precomputed_index, usp):
        """Full-sequence indexer keys this step scores its queries against: the
        step-0 keys a later TTT step was handed when ``dsa_index_share`` is off,
        else this step's own (gathered across the SP group under USP)."""
        if not self.dsa_index_share and precomputed_index is not None:
            return precomputed_index
        keys = self.indexer.project_keys(hidden_states, cos, sin)
        return self._usp_gather_seq(keys) if usp else keys

    def _usp_cached_forward(
        self,
        query_states,
        key_states,
        value_states,
        cache_keys,
        cache_values,
        hidden_states,
        q_resid,
        cos,
        sin,
        ctx_doc_ids,
        attention_mask,
        precomputed_index,
        q_len,
    ):
        """Ulysses seq->head all-to-all around the cached MLA attention so each rank
        attends over the FULL sequence (not just its seq shard). The indexer, causal
        and doc masks are all rebuilt on the gathered full sequence; the per-head
        learnable sink is sliced to this rank's heads. Returns (attn_output on the
        local seq shard, cache_keys, cache_values, DSA state for later TTT steps)."""
        group = self._usp_group
        bsz = query_states.shape[0]

        # seq-shard -> head-shard: [b, H, S_loc, d] -> [b, H/sp, S_full, d].
        query_states = _mtp_ulysses_all_to_all(query_states, group, 1, 2)
        key_states = _mtp_ulysses_all_to_all(key_states, group, 1, 2)
        value_states = _mtp_ulysses_all_to_all(value_states, group, 1, 2)
        h_local, s_full = query_states.shape[1], query_states.shape[2]
        rank = dist.get_rank(group)
        previous_head_slice = self._head_slice
        self._head_slice = slice(rank * h_local, (rank + 1) * h_local)

        # Full-sequence key-validity and doc ids for the block-0 masks.
        if attention_mask is None:
            valid_loc = torch.ones(bsz, q_len, dtype=torch.bool, device=query_states.device)
        else:
            valid_loc = mtp_validity_mask(attention_mask, bsz, q_len, query_states.device)
        valid_full = self._usp_gather_seq(valid_loc)
        doc_full = None
        if ctx_doc_ids is not None:
            d = ctx_doc_ids
            if d.dim() > 2:
                d = d.view(bsz, -1)[:, :q_len]
            doc_full = self._usp_gather_seq(d.contiguous().long())

        index_state = precomputed_index
        active_index = precomputed_index if self.dsa_index_share else None
        if self.use_indexer and s_full >= self.indexer.index_topk and active_index is None:
            index_keys = self._dsa_index_keys(hidden_states, cos, sin, precomputed_index, usp=True)
            local_index = self.indexer(
                hidden_states,
                q_resid,
                cos,
                sin,
                None,
                return_indices=True,
                doc_ids=doc_full,
                valid_mask=valid_full,
                precomputed_keys=index_keys,
                query_offset=rank * q_len,
            )
            active_index = (
                self._usp_gather_seq(local_index[0]),
                self._usp_gather_seq(local_index[1]),
            )
            index_state = active_index if self.dsa_index_share else index_keys
            del index_keys, local_index

        if cache_keys is None:
            cache_keys = key_states.unsqueeze(2)
            cache_values = value_states.unsqueeze(2)
        else:
            cache_keys = torch.cat([cache_keys, key_states.unsqueeze(2)], dim=2)
            cache_values = torch.cat([cache_values, value_states.unsqueeze(2)], dim=2)
        nc = cache_keys.shape[2]
        k0, v0 = cache_keys[:, :, 0], cache_values[:, :, 0]
        chain_keys = [cache_keys[:, :, i] for i in range(1, nc)]
        chain_values = [cache_values[:, :, i] for i in range(1, nc)]

        use_sparse = (
            self.use_indexer and s_full >= self.indexer.index_topk and active_index is not None
        )
        try:
            if use_sparse:
                topk_idx, all_open = active_index
                attn_output = self._sparse_cached_attn(
                    query_states, k0, v0, topk_idx, all_open, chain_keys, chain_values
                )
            else:
                if s_full > 8192:
                    raise RuntimeError(
                        "MTP MLA USP dense attention is disabled above 8192 tokens; "
                        "enable the DSA indexer or use a sparse/linear-memory backend."
                    )
                m = self._build_causal_doc_mask(
                    valid_full, doc_full, s_full, query_states.device, query_states.dtype
                )
                attn_output = self._dense_block0(query_states, k0, v0, chain_keys, chain_values, m)

            attn_output = _mtp_ulysses_all_to_all(attn_output, group, 2, 1)
        finally:
            self._head_slice = previous_head_slice
        return attn_output, cache_keys, cache_values, index_state

    @staticmethod
    def _sparse_query_chunk(chunk_size, num_heads, topk, feat_dim, elem_size, budget_bytes=None):
        """Cap the query chunk so the [b, heads, chunk*topk, dim] K/V gather stays
        within budget_bytes. topk is fixed by the sparsity budget, so a large chunk
        at long context would otherwise materialize tens of GiB and OOM."""
        if budget_bytes is None:
            budget_mb = int(os.environ.get("ANGELSPEC_MTP_SPARSE_CHUNK_MB", "128"))
            budget_bytes = max(1, budget_mb) << 20
        per_q = max(1, num_heads * topk * feat_dim * elem_size)
        return max(1, min(chunk_size, budget_bytes // per_q))

    def _sparse_attn_chunk(self, query_chunk, key_states, value_states, topk_idx_chunk):
        """Sparse attention on a query chunk using gathered K/V at topk indices."""
        bsz, num_heads, cs, qk_dim = query_chunk.shape
        v_dim = value_states.shape[-1]
        k = topk_idx_chunk.shape[-1]

        idx_flat = topk_idx_chunk.clamp_min(0).reshape(bsz, 1, cs * k, 1).long()
        k_gathered = key_states.gather(2, idx_flat.expand(bsz, num_heads, cs * k, qk_dim))
        k_sparse = k_gathered.view(bsz, num_heads, cs, k, qk_dim)

        v_gathered = value_states.gather(2, idx_flat.expand(bsz, num_heads, cs * k, v_dim))
        v_sparse = v_gathered.view(bsz, num_heads, cs, k, v_dim)

        logits = torch.einsum("bhcd,bhckd->bhck", query_chunk, k_sparse) * self.scaling
        logits = logits.masked_fill(topk_idx_chunk[:, None] < 0, float("-inf"))
        probs, _ = self._softmax_with_sink(logits)
        return torch.einsum("bhck,bhckd->bhcd", probs, v_sparse)

    def _sparse_cached_attn(
        self,
        query_states,
        key_states,
        value_states,
        topk_idx,
        all_open,
        chain_keys,
        chain_values,
        attention_mask=None,
        chunk_size=2048,
    ):
        """Sparse block-0 + chain terms with joint softmax (cached TTT path)."""
        bsz, num_heads, q_len, qk_dim = query_states.shape
        v_dim = value_states.shape[-1]
        k = topk_idx.shape[-1]
        chunk_size = self._sparse_query_chunk(
            chunk_size, num_heads, k, qk_dim + v_dim, query_states.element_size()
        )

        num_chain = len(chain_keys)
        # Captured by value so a checkpoint recompute (backward, after
        # _usp_cached_forward reset self._head_slice) slices the sink to the same
        # heads the forward used.
        sink_slice = self._head_slice

        def _chunk_fn(q_chunk, idx_chunk, key_states, value_states, *chain_tensors):
            cs = q_chunk.shape[2]
            idx_flat = idx_chunk.clamp_min(0).reshape(bsz, 1, cs * k, 1).long()
            k_sparse = key_states.gather(2, idx_flat.expand(bsz, num_heads, cs * k, qk_dim))
            k_sparse = k_sparse.view(bsz, num_heads, cs, k, qk_dim)
            v_sparse = value_states.gather(2, idx_flat.expand(bsz, num_heads, cs * k, v_dim))
            v_sparse = v_sparse.view(bsz, num_heads, cs, k, v_dim)

            block0_logits = torch.einsum("bhcd,bhckd->bhck", q_chunk, k_sparse) * self.scaling
            block0_logits = block0_logits.masked_fill(idx_chunk[:, None] < 0, float("-inf"))

            chunk_chain_keys = chain_tensors[:num_chain]
            chunk_chain_values = chain_tensors[num_chain:]
            chain_logit_list = []
            for ck in chunk_chain_keys:
                wi = (q_chunk * ck).sum(-1, keepdim=True) * self.scaling
                chain_logit_list.append(wi)

            all_logits = torch.cat([block0_logits] + chain_logit_list, dim=-1)
            probs, _ = self._softmax_with_sink(all_logits, head_slice=sink_slice)

            block0_probs = probs[..., :k]
            out = torch.einsum("bhck,bhckd->bhcd", block0_probs, v_sparse)
            for i, cv in enumerate(chunk_chain_values):
                chain_prob = probs[..., k + i : k + i + 1]
                out = out + chain_prob * cv
            return out

        checkpoint_chunks = torch.is_grad_enabled() and os.environ.get(
            "ANGELSPEC_MTP_SPARSE_CHUNK_CHECKPOINT", "1"
        ) not in {"0", "false", "False"}
        output_chunks = []
        for start in range(0, q_len, chunk_size):
            end = min(start + chunk_size, q_len)
            chunk_args = (
                query_states[:, :, start:end],
                topk_idx[:, start:end].detach(),
                key_states,
                value_states,
                *[ck[:, :, start:end] for ck in chain_keys],
                *[cv[:, :, start:end] for cv in chain_values],
            )
            if checkpoint_chunks:
                chunk_out = torch_checkpoint(
                    _chunk_fn,
                    *chunk_args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                chunk_out = _chunk_fn(*chunk_args)
            output_chunks.append(chunk_out)
        output = torch.cat(output_chunks, dim=2)

        return output

    def _sparse_sdpa_block0(
        self,
        query_states,
        key_states,
        value_states,
        topk_idx,
        all_open,
        attention_mask=None,
        chunk_size=2048,
    ):
        """Sparse block-0 attention: only attend to DSA-selected topk keys per query."""
        bsz, num_heads, q_len, _ = query_states.shape
        v_dim = value_states.shape[-1]
        k = topk_idx.shape[-1]
        chunk_size = self._sparse_query_chunk(
            chunk_size, num_heads, k, query_states.shape[-1] + v_dim, query_states.element_size()
        )

        if q_len <= chunk_size:
            output = self._sparse_attn_chunk(query_states, key_states, value_states, topk_idx)
        else:
            output = torch.zeros(
                bsz, num_heads, q_len, v_dim, dtype=query_states.dtype, device=query_states.device
            )
            for start in range(0, q_len, chunk_size):
                end = min(start + chunk_size, q_len)
                output[:, :, start:end] = self._sparse_attn_chunk(
                    query_states[:, :, start:end],
                    key_states,
                    value_states,
                    topk_idx[:, start:end],
                )

        return output

    def _sdpa_block0(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        hidden_states,
        q_resid,
        cos,
        sin,
        precomputed_index=None,
    ):
        bsz, _, q_len, _ = query_states.shape
        if self.use_indexer and q_len >= self.indexer.index_topk:
            if precomputed_index is not None:
                topk_idx, all_open = precomputed_index
            else:
                topk_idx, all_open = self.indexer(
                    hidden_states, q_resid, cos, sin, attention_mask, return_indices=True
                )
            return self._sparse_sdpa_block0(
                query_states, key_states, value_states, topk_idx, all_open, attention_mask
            )
        logits = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            logits = logits + attention_mask
        probs, _ = self._softmax_with_sink(logits)
        return torch.matmul(probs, value_states)

    @staticmethod
    def _causal_add_mask(attention_mask, q_len, dtype, device):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask[:, 0]  # [b, s, s]
        neg_inf = torch.finfo(dtype).min
        m = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=device), diagonal=1)
        return torch.where(
            m,
            torch.tensor(neg_inf, dtype=dtype, device=device),
            torch.zeros(1, dtype=dtype, device=device),
        )[None]


class MTPDecoderLayer(nn.Module):
    """One decoder layer (GQA or MLA attn + MoE), KV-cache aware.

    Pre-norm residual structure identical to the HF/vLLM reference:
        h = h + attn(input_layernorm(h))
        h = h + mlp(post_attention_layernorm(h))
    """

    def __init__(self, config: MTPConfig, attention_backend: str = "sdpa"):
        super().__init__()
        if getattr(config, "attention_type", "gqa") == "mla":
            self.self_attn = MTPMLAAttention(config, attention_backend=attention_backend)
        else:
            self.self_attn = MTPAttention(config, attention_backend=attention_backend)
        self.input_layernorm = build_mtp_rms_norm(config, config.hidden_size)
        self.post_attention_layernorm = build_mtp_rms_norm(config, config.hidden_size)
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
        precomputed_index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_index: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_result = self.self_attn(
            hidden_states=hidden_states,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            ctx_doc_ids=ctx_doc_ids,
            precomputed_index=precomputed_index,
            return_index=return_index,
        )
        if return_index:
            hidden_states, cache_keys, cache_values, index_out = attn_result
        else:
            hidden_states, cache_keys, cache_values = attn_result
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if return_index:
            return hidden_states, cache_keys, cache_values, index_out
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
        self.attention_backend = validate_mtp_backend(attention_backend)

        self.target_vocab_size = config.vocab_size
        self.vocab_size = config.vocab_size  # no pruning for MTP

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None)
        )

        # MTP-specific fusion of (shifted-token embedding, last hidden state).
        self.enorm = build_mtp_rms_norm(config, config.hidden_size)
        self.hnorm = build_mtp_rms_norm(config, config.hidden_size)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.midlayer = MTPDecoderLayer(config, attention_backend=attention_backend)
        # The DSA indexer only selects keys (and is idle below index_topk), so the MTP
        # loss never reaches it; keep the target's weights instead of leaving
        # gradient-free trainable parameters, which DDP rejects.
        if getattr(config, "mtp_freeze_indexer", False):
            indexer = getattr(self.midlayer.self_attn, "indexer", None)
            if indexer is not None:
                indexer.requires_grad_(False)
        self.final_layernorm = build_mtp_rms_norm(config, config.hidden_size)

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
        if getattr(config, "rms_norm_type", "llama") == "qwen3_5":
            if config.attention_type != "gqa" or not config.tie_lm_head:
                raise ValueError("Qwen3.5 MTP requires GQA and a frozen target LM head")
            with torch.no_grad():
                for module in self.modules():
                    if isinstance(module, (nn.Linear, FusedExpertProj)):
                        nn.init.normal_(
                            module.weight, std=getattr(config, "initializer_range", 0.02)
                        )
                    elif isinstance(module, Qwen35RMSNorm):
                        nn.init.zeros_(module.weight)

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
        # Assigning an nn.Parameter through Module.__setattr__ registers it even
        # though this attribute was initialized to None. Keep the documented
        # plain-tensor contract, including when rebinding a previously tied head.
        self._parameters.pop("lm_head_weight", None)
        self.lm_head_weight = weight.detach()

    def frozen_fp32_parameters(self) -> dict[str, nn.Parameter]:
        """Grad-free parameters that stay fp32 in storage and compute.

        Modules declare them in ``_frozen_fp32_params`` (the router ``expert_bias``,
        the compressed indexer's fp32 pooling tensors); the target checkpoint stores
        them in F32. Trainers keep them out of FSDP's bf16 mixed precision.
        """
        params = {}
        for prefix, module in self.named_modules():
            for name in getattr(type(module), "_frozen_fp32_params", ()):
                owner_name, _, leaf = name.rpartition(".")
                owner = module.get_submodule(owner_name) if owner_name else module
                param = getattr(owner, leaf, None)
                if isinstance(param, nn.Parameter):
                    params[f"{prefix}.{name}" if prefix else name] = param
        return params

    def restore_frozen_fp32(self) -> None:
        """Undo a model-wide ``.to(bf16)`` for :meth:`frozen_fp32_parameters`."""
        for param in self.frozen_fp32_parameters().values():
            param.data = param.data.float()

    # -- Eagle3DraftModel interface -------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.placement_types import Replicate

        if isinstance(self.embed_tokens.weight, DTensor) and not isinstance(input_ids, DTensor):
            input_ids = DTensor.from_local(
                input_ids,
                device_mesh=self.embed_tokens.weight.device_mesh,
                placements=[Replicate()] * self.embed_tokens.weight.device_mesh.ndim,
            )
            return self.embed_tokens(input_ids).to_local()
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
        precomputed_index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_index: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # hy_v3 masks the position-0 token embedding (per-doc under packing).
        if (
            position_ids is not None
            and getattr(self.config, "rms_norm_type", "llama") != "qwen3_5"
        ):
            mask0 = position_ids == 0
            if position_ids.ndim == 3:
                mask0 = mask0.all(dim=1)
                if cache_keys is not None:
                    mask0 = torch.zeros_like(mask0)
            mask0 = mask0.unsqueeze(-1).to(input_embeds.dtype)
            input_embeds = input_embeds * (1.0 - mask0)

        de = self.enorm(input_embeds)
        h = self.hnorm(hidden_states)
        x = self.eh_proj(torch.cat([de, h], dim=-1))

        result = self.midlayer(
            hidden_states=x,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            ctx_doc_ids=ctx_doc_ids,
            precomputed_index=precomputed_index,
            return_index=return_index,
        )
        if return_index:
            x, cache_keys, cache_values, index_out = result
        else:
            x, cache_keys, cache_values = result
        x = self.final_layernorm(x)
        if return_index:
            return x, cache_keys, cache_values, index_out
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
