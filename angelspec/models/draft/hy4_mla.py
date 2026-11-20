"""hy4 gated-MLA attention (+ optional DSA indexer) for the DFly drafter.

Ports the attention half of the hy4 decoder layer — what
:class:`angelspec.models.draft.mtp.MTPMLAAttention` implements for the MTP head —
onto DFly's dual-source KV layout:

  * Q comes from the draft block only, through the LoRA q path.
  * K/V come from ``context ++ draft`` through ONE shared MLA down-projection,
    the same way :class:`~angelspec.models.draft.dflash.DFlashAttention` shares
    ``k_proj``/``v_proj`` between the two sources. DFly already requires
    ``target_hidden_size == hidden_size``, so both sources live in one space.

Reused rather than re-derived (``mtp.py`` is not modified):
  * ``mtp.MTPIndexer`` — the DSA top-k indexer, weight names and math included, so
    a pretrained hy4 ``self_attn.indexer`` still loads. Its ``precomputed_keys``
    and 4-D ``attention_mask`` arguments are exactly the hooks needed to score
    draft queries against context keys under an arbitrary mask.
  * ``mtp._interleaved_to_split_rope`` / ``mtp._apply_rope_split`` — hy4 stores
    the rope slice interleaved.
  * ``llama3_eagle.LlamaRMSNorm`` / ``LlamaRotaryEmbedding`` — same modules the
    MTP MLA path uses, so weight shapes and RoPE match.

Masking: DFly trains under a FlexAttention ``BlockMask`` (block-causal, and with
``causal_attn`` the intra-block direction is causal too). MLA cannot just hand
that object to ``flex_attention``, because both the learnable sink and the DSA
top-k gate have to take part in the softmax, which needs an *additive* mask. So
the mask is materialised from the BlockMask's own ``mask_mod``
(:func:`materialize_keep_mask`) — the training semantics are reused verbatim
instead of being re-implemented here, and they cannot drift.

Two DSA backends, same weights and same masking semantics:

``dsa_backend="torch"``
    Dense path: decompresses the latent to per-head K/V and attends over every key
    the mask allows, with the top-k gate applied as part of that mask. On CUDA it
    goes through ``flex_attention``, which tiles the scores and hands back the
    log-sum-exp the sink needs (:meth:`Hy4DraftMLAAttention._attend_flex`), so
    nothing quadratic in the context length is materialised. Off CUDA (flex has no
    CPU backward) it falls back to :meth:`Hy4DraftMLAAttention._attend_torch`,
    which builds the ``[B, H, draft_len, kv_len]`` score tensor outright — exact
    and simple, but short-context only.

``dsa_backend="tilelang"``
    Same math through the TileLang kernels the hy4 MTP head uses
    (:mod:`angelspec.models.ops.tilelang_dsa`), in the *absorbed* MLA form: keys
    stay compressed as the ``kv_lora_rank + qk_rope_head_dim`` latent and the
    kernel only ever reads the ``index_topk`` keys the indexer selected, so
    nothing quadratic is ever materialised. This is the path that makes a
    long-context drafter trainable. It needs the kernel's own constraints
    (``kv_lora_rank=512``, ``qk_rope_head_dim=64``, an indexer) — see
    ``DFlyConfig``.

    The one thing DFly must supply itself is the key list, because the kernels take
    masking *only* as indices: one contiguous ``[key_starts, key_ends)`` range
    per query for the indexer, and an explicit index list for the attention.
    DFlash's mask is exactly expressible that way — a block's context window is
    the interval ``[doc_start(anchor), anchor)`` and its draft window is its own
    block — and both are read back out of ``mask_mod`` (:func:`keep_mask_to_ranges`,
    :func:`keep_mask_to_indices`) rather than re-derived from anchor positions,
    so the two backends cannot disagree about what a token may see.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from angelspec.models.draft.llama3_eagle import LlamaRMSNorm, LlamaRotaryEmbedding
from angelspec.models.draft.mtp import (
    MTPIndexer,
    _apply_rope_split,
    _fp8_round_trip,
    _interleaved_to_split_rope,
)
from angelspec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
)
from angelspec.models.ops.mrope import MRotaryEmbedding
from angelspec.models.ops.tilelang_dsa import (
    DSA_INDEX_ALIGNMENT,
    DSA_KERNEL_DTYPE,
    tilelang_dsa_available,
    tilelang_lightning_indexer,
    tilelang_sparse_mla,
)
from angelspec.models.ops.ulysses import all_gather_seq
from angelspec.utils.distributed import get_sp_ulysses_group

logger = logging.getLogger(__name__)


def resolve_dsa_backend(config: PretrainedConfig) -> str:
    """Which DSA implementation this process will actually run.

    ``"auto"`` asks for the TileLang kernels where they run and the dense torch
    path elsewhere, so one config travels between a training image that ships
    TileLang and a box that does not (e.g. checkpoint conversion). It is resolved
    per module, at construction, because that happens in the process that will
    run the kernels — the driver's environment may differ from the trainer's, and
    a resolved value baked into the config would follow the checkpoint around.
    """
    requested = str(getattr(config, "dsa_backend", "torch"))
    if requested != "auto":
        return requested
    # The kernel only reads a key list, which comes from the drafter's own DSA
    # indexer or, for DFlash, from the target indexer's candidates.
    keyed = (
        getattr(config, "index_topk", None) is not None
        or getattr(config, "target_index_layer", None) is not None
    )
    usable = (
        str(getattr(config, "attention_type", "gqa")) == "mla"
        and keyed
        and int(getattr(config, "kv_lora_rank", 0) or 0) == 512
        and int(getattr(config, "qk_rope_head_dim", 0) or 0) == 64
    )
    return "tilelang" if usable and tilelang_dsa_available() else "torch"


def materialize_keep_mask(
    block_mask,
    bsz: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Materialise a FlexAttention ``BlockMask`` as ``[B, 1, q_len, kv_len]`` bool.

    ``True`` means "may attend". Returns ``None`` for ``block_mask=None``, which
    the callers treat as fully visible — matching DFlashAttention's no-mask
    fallback.
    """
    if block_mask is None:
        return None

    mask_mod = getattr(block_mask, "mask_mod", None)
    if mask_mod is None:
        raise ValueError(
            "hy4 MLA needs the BlockMask's mask_mod to build an additive mask "
            f"(the sink and the DSA gate live inside the softmax); got {type(block_mask)!r}"
        )

    from torch.nn.attention.flex_attention import create_mask

    return create_mask(mask_mod, bsz, 1, q_len, kv_len, device=device)


def keep_mask_to_ranges(keep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[B, q_len, L]`` bool keep-mask -> per-query half-open ``(starts, ends)``.

    The lightning indexer expresses "which keys may this query score" as one
    contiguous range per query row, which is all DFlash's mask needs over the
    context axis: a draft block sees ``kv_idx < anchor``, intersected with the
    anchor's document when sequences are packed, and that is an interval.

    A row with nothing visible gets ``starts == ends`` (empty). Rows whose mask is
    *not* an interval are rejected rather than silently widened, since the kernel
    has no way to express them.
    """
    counts = keep.sum(-1)
    as_int = keep.int()
    first = torch.argmax(as_int, dim=-1)
    last = keep.shape[-1] - 1 - torch.argmax(as_int.flip(-1), dim=-1)
    empty = counts == 0
    starts = torch.where(empty, torch.zeros_like(first), first)
    ends = torch.where(empty, torch.zeros_like(last), last + 1)
    if not torch.equal(ends - starts, counts):
        raise ValueError(
            "the DSA indexer needs each query's visible key range to be contiguous, but "
            "the block mask leaves gaps; only the DFlash-style block-causal mask is "
            "supported by this backend."
        )
    return starts.to(torch.int32), ends.to(torch.int32)


def _finalize_sparse_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[B, S, K]`` key indices -> the kernel's ``[B, S, 1, K']`` list, plus live rows.

    Pads ``K`` up to :data:`DSA_INDEX_ALIGNMENT` and returns the ``[B, S]`` mask
    of rows with at least one key.
    """
    width = indices.shape[-1]
    blocks = max(1, (width + DSA_INDEX_ALIGNMENT - 1) // DSA_INDEX_ALIGNMENT)
    target = blocks * DSA_INDEX_ALIGNMENT
    if target != width:
        indices = F.pad(indices, (0, target - width), value=-1)
    return indices.unsqueeze(2).to(torch.int32).contiguous(), (indices >= 0).any(-1)


def keep_mask_to_indices(keep: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """``[B, q_len, L]`` bool keep-mask -> ``[B, q_len, K]`` int32 key indices.

    Indices are ascending and shifted by ``offset`` (DFly's draft keys sit after
    the context in the KV layout); ``K`` is the widest row and short rows are
    ``-1``-padded, which is how the sparse kernels mark an unused slot.
    """
    width = int(keep.sum(-1).max().item()) if keep.numel() else 0
    if width == 0:
        return keep.new_zeros((*keep.shape[:2], 0), dtype=torch.int32)
    # Sort on the negated mask so kept columns come first; `stable` is what keeps
    # them in ascending column order (plain argsort may permute equal keys).
    order = torch.argsort((~keep).int(), dim=-1, stable=True)[..., :width]
    kept = torch.gather(keep, -1, order)
    indices = order.to(torch.int32) + int(offset)
    return torch.where(kept, indices, torch.full_like(indices, -1))


class Hy4DraftMLAAttention(nn.Module):
    """hy4 gated MLA over DFly's ``context ++ draft`` keys, with optional DSA.

    Submodule names and shapes match ``MTPMLAAttention`` (hence the hy4 target's
    ``self_attn.*``): LoRA q, ``kv_a_proj_with_mqa`` + ``kv_b_proj``, elementwise
    sigmoid output gate, per-head learnable sink.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        indexer: Optional[nn.Module] = None,
        own_indexer: bool = True,
    ):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = LlamaRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
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
        self.kv_a_layernorm = LlamaRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        self.scaling = self.qk_head_dim ** (-0.5)

        self.gated_mla = bool(getattr(config, "gated_mla", False))
        if self.gated_mla:
            self.gate_projection_size = (
                1
                if getattr(config, "gating_type", "elementwise") == "headwise"
                else self.v_head_dim
            )
            self.linear_gate = nn.Linear(
                self.hidden_size, self.num_heads * self.gate_projection_size, bias=False
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

        # Hy4 flash's RMSNorm after o_proj (its sandwich_norm), so an attention copied from
        # such a target (target_attn_init_layers) computes what the target's does.
        self.post_proj_norm = (
            LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, "attn_post_proj_norm", False)
            else None
        )

        # By default the layer owns its indexer, so the weights sit at
        # ``self_attn.indexer.*`` exactly as hy4 names them. Under a sharing
        # ``indexer_types`` schedule the drafter owns them instead and passes the
        # module in; it is then held in a plain list rather than assigned to an
        # attribute, so it is NOT registered as a submodule here. A parameter
        # reachable by two FQNs is what breaks the FSDP loader — see
        # :func:`build_hy4_self_attns`. Read it through :attr:`dsa_indexer`.
        if own_indexer:
            indexer = MTPIndexer(config) if getattr(config, "index_topk", None) else None
            self.indexer = indexer
        elif indexer is None:
            raise ValueError("own_indexer=False needs an indexer to borrow")
        self._indexer_ref = [indexer]
        self.use_indexer = indexer is not None
        if self.use_indexer and self.q_lora_rank is None:
            raise ValueError("the DSA indexer scores from the LoRA q residual; set q_lora_rank")

        self.dsa_backend = resolve_dsa_backend(config)
        self.fp8_indexer = bool(getattr(config, "dsa_fp8_indexer", False))
        # Without its own indexer the layer is handed its keys per call
        # (``sparse_kv_indices``, from the target indexer's candidates).
        self.external_keys = getattr(config, "target_index_layer", None) is not None
        if self.dsa_backend == "tilelang":
            if not self.use_indexer and not self.external_keys:
                raise ValueError(
                    "tilelang sparse MLA is a sparse kernel; it needs the DSA indexer or "
                    "target_index_layer to supply its keys"
                )
            if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
                raise ValueError(
                    "tilelang sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64"
                )

        rope = getattr(config, "rope_parameters", None) or {}
        self.uses_mrope = bool(rope.get("mrope_section"))
        self.rotary_emb = (
            MRotaryEmbedding(self.qk_rope_head_dim, rope)
            if self.uses_mrope
            else LlamaRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000),
            )
        )

        self._usp_group = None
        self._usp_size = 1
        self._usp_rank = 0

    @property
    def dsa_indexer(self) -> Optional[nn.Module]:
        """The DSA indexer this layer scores with — its own, or a borrowed one."""
        return self._indexer_ref[0]

    def _ensure_usp(self) -> None:
        if self._usp_group is None:
            self._usp_group = get_sp_ulysses_group()
            if self._usp_group is not None:
                self._usp_size = dist.get_world_size(self._usp_group)
                self._usp_rank = dist.get_rank(self._usp_group)
            else:
                self._usp_size = 1
                self._usp_rank = 0

    def _cos_sin(
        self, ref: torch.Tensor, position_ids: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-position ``(cos, sin)`` of shape ``[B, S, rope_dim]``.

        ``position_ids`` is ``[B, S]`` (1D RoPE) or ``[B, 3, S]`` (M-RoPE rows t/h/w,
        each rotary column reading the row its frequency is assigned to).
        """
        if self.uses_mrope:
            return self.rotary_emb(ref, position_ids)
        if position_ids.ndim != 2:
            raise ValueError("3D positions require rope_parameters.mrope_section")
        cache_len = max(int(seq_len), int(position_ids.max().item()) + 1)
        cos, sin = self.rotary_emb(ref, seq_len=cache_len)
        return cos.squeeze(1).squeeze(0)[position_ids], sin.squeeze(1).squeeze(0)[position_ids]

    def _project_q(self, draft_hidden: torch.Tensor):
        bsz, q_len, _ = draft_hidden.shape
        if self.q_lora_rank is not None:
            q_resid = self.q_a_layernorm(self.q_a_proj(draft_hidden))
            q = self.q_b_proj(q_resid)
        else:
            q_resid = None
            q = self.q_proj(draft_hidden)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        return q_nope, q_pe, q_resid

    def _project_kv_latent(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Down-project one KV source, normalize it, and RoPE its shared rope slice.

        Returns the concatenated ``[B, S, kv_lora_rank + qk_rope_head_dim]`` latent
        — the same normalized-then-roped form ``MTPMLAAttention._project_absorbed``
        caches, so it can go to the sparse kernel as-is or be decompressed by
        :meth:`_decompress_kv`. RoPE is applied here, before any USP gather,
        because only the owning rank knows its shard's positions;
        ``kv_a_layernorm`` is position-wise, so its placement is unaffected.
        """
        bsz, seq_len, _ = hidden.shape
        kv = self.kv_a_proj_with_mqa(hidden)
        latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, 1, seq_len, self.qk_rope_head_dim)
        k_pe = _apply_rope_split(_interleaved_to_split_rope(k_pe), cos, sin)
        return torch.cat([self.kv_a_layernorm(latent), k_pe.squeeze(1)], dim=-1)

    def _decompress_kv(self, latent_with_pe: torch.Tensor):
        """``[B, S, kv_lora+rope]`` (already normalized) -> per-head ``key``/``value``."""
        bsz, seq_len, _ = latent_with_pe.shape
        latent, k_pe = torch.split(
            latent_with_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(latent)
        kv = kv.view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)  # [B, H, S, nope]
        # Copied out rather than left as a view: a view keeps the whole
        # (nope+v)-wide ``kv`` resident for the backward pass, which at 128k is
        # ~2 GiB per layer more than ``value`` itself needs.
        value = value.transpose(1, 2).contiguous()  # [B, H, S, v]
        k_pe = k_pe.view(bsz, 1, seq_len, self.qk_rope_head_dim).expand(-1, self.num_heads, -1, -1)
        return torch.cat([k_nope, k_pe], dim=-1), value

    def _kv_b_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split ``kv_b_proj`` into its per-head key (``w_kc``) and value (``w_vc``) halves."""
        weight = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        return torch.split(weight, [self.qk_nope_head_dim, self.v_head_dim], dim=1)

    def _absorb_q(self, q_nope: torch.Tensor, q_pe: torch.Tensor) -> torch.Tensor:
        """Fold ``w_kc`` into the query so it can score the *compressed* latent.

        ``q_nope @ w_kc`` lives in the ``kv_lora_rank`` space, so
        ``q_absorbed · latent`` reproduces the expanded ``q · k`` logits exactly
        (that identity is what ``tests/test_sparse_dsa_adapter.py`` pins down for
        the MTP head) without ever decompressing K. Returns ``[B, S, H, kv_lora +
        rope]``, the layout the sparse MLA kernel expects.
        """
        w_kc, _ = self._kv_b_weights()
        q_nope = torch.einsum("bhsd,hdm->bshm", q_nope, w_kc)
        return torch.cat([q_nope, q_pe.transpose(1, 2)], dim=-1)

    def _softmax_with_sink(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax over the key axis, folding the per-head sink into the denominator.

        Same construction as ``MTPMLAAttention._softmax_with_sink``: equivalent to
        one extra key whose score is ``learnable_sink_param`` and whose value is 0,
        so the rows sum to less than 1 and the sink drains probability mass. No
        head slicing here (the MTP path needs it): USP shards the context sequence
        and gathers it, so this softmax always sees every head.
        """
        if not self.learnable_sink:
            return F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        lf = logits.to(torch.float32)
        sink = self.learnable_sink_param.to(torch.float32).view(1, -1, 1, 1)
        # logaddexp of the key-axis logsumexp, not a logsumexp over the concatenated
        # tensor: same value, without a second fp32 copy of the scores.
        total = torch.logaddexp(torch.logsumexp(lf, dim=-1, keepdim=True), sink)
        return torch.exp(lf - total).to(logits.dtype)

    def _dsa_keep(
        self,
        draft_hidden: torch.Tensor,
        q_resid: torch.Tensor,
        index_keys_ctx: torch.Tensor,
        cos_q: torch.Tensor,
        sin_q: torch.Tensor,
        keep: Optional[torch.Tensor],
        ctx_len: int,
    ) -> torch.Tensor:
        """DSA top-k gate as a bool ``[B, 1, draft_len, kv_len]`` keep-mask.

        The indexer scores draft queries against ``index_keys_ctx`` (one row per
        context position) and keeps the top ``index_topk``. The base mask is handed
        to it as its additive mask, so a key the training mask already forbids can
        never win a slot.

        Only the context is scored, and the draft block is kept whole: it is a
        handful of tokens and the one region the drafter must always see. That also
        keeps this identical to the ``tilelang`` backend, whose kernel takes the
        selected keys and the block's own keys as two separate index lists (as
        MTP's ``append_diagonal_kv_indices`` does for its current step).
        """
        bsz, draft_len, _ = draft_hidden.shape
        neg = torch.finfo(draft_hidden.dtype).min

        if keep is None:
            add_mask = torch.zeros(
                bsz, 1, draft_len, ctx_len, dtype=draft_hidden.dtype, device=draft_hidden.device
            )
        else:
            add_mask = torch.where(
                keep[..., :ctx_len],
                torch.zeros((), dtype=draft_hidden.dtype, device=draft_hidden.device),
                torch.full((), neg, dtype=draft_hidden.dtype, device=draft_hidden.device),
            )

        dsa_add = self.dsa_indexer(
            hidden_states=draft_hidden,
            q_resid=q_resid,
            cos=cos_q,
            sin=sin_q,
            attention_mask=add_mask,
            precomputed_keys=index_keys_ctx,
        )  # [B, draft_len, ctx_len]: 0 kept, finfo.min dropped

        ctx_keep = (dsa_add > neg / 2).unsqueeze(1)
        draft_keep = ctx_keep.new_ones((bsz, 1, draft_len, draft_len))
        return torch.cat([ctx_keep, draft_keep], dim=-1)

    @torch.no_grad()
    def _context_topk(
        self,
        draft_hidden: torch.Tensor,
        q_resid: torch.Tensor,
        index_keys: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        keep_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Top-``index_topk`` context keys per draft query, via the lightning indexer kernel.

        Mirrors ``MTPIndexer.sparse_indices`` — same weights, same projection,
        same kernel — apart from the one thing that call cannot express: it derives
        its scoring ranges from a self-attention causal layout
        (``build_contiguous_causal_ranges``, i.e. ``[doc_start, q_pos]``), whereas a
        DFly draft block scores a *separate* context axis over
        ``[doc_start(anchor), anchor)``. Since ``mtp.py`` stays untouched, the query
        projection is repeated here and the ranges come from the block mask.
        """
        idx = self.dsa_indexer
        bsz, draft_len, _ = draft_hidden.shape
        q = idx.wq_b(q_resid).view(bsz, draft_len, idx.n_heads, idx.head_dim)
        q_nope, q_pe = torch.split(
            q, [idx.head_dim - idx.qk_rope_head_dim, idx.qk_rope_head_dim], dim=-1
        )
        q_pe = _interleaved_to_split_rope(q_pe).transpose(1, 2)
        q_pe = _apply_rope_split(q_pe, cos, sin).transpose(1, 2)
        q = torch.cat([q_nope, q_pe], dim=-1)
        weights = idx.weights_proj(draft_hidden).float() * (idx.n_heads**-0.5) * idx.softmax_scale
        if self.fp8_indexer:
            q, index_keys = _fp8_round_trip(q), _fp8_round_trip(index_keys)

        starts, ends = keep_mask_to_ranges(keep_ctx)
        per_sequence = []
        for b in range(bsz):
            # The kernel takes one sequence at a time, as in MTP's adapter.
            _, indices = tilelang_lightning_indexer(
                q[b].to(DSA_KERNEL_DTYPE),
                index_keys[b].to(DSA_KERNEL_DTYPE),
                weights[b],
                starts[b],
                ends[b],
                int(idx.index_topk),
            )
            per_sequence.append(indices)
        return torch.stack(per_sequence, dim=0).to(torch.int32)

    @torch.no_grad()
    def _sparse_kernel_indices(
        self,
        draft_hidden: torch.Tensor,
        q_resid: torch.Tensor,
        index_keys_ctx: Optional[torch.Tensor],
        cos_q: torch.Tensor,
        sin_q: torch.Tensor,
        keep: Optional[torch.Tensor],
        ctx_len: int,
        draft_len: int,
        run_dsa: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Key list for the sparse kernel, ``[B, draft_len, 1, K]`` int32.

        Also returns the ``[B, draft_len]`` mask of rows that ended up with at
        least one key. The two halves of DFly's KV are built separately: the
        context half is the indexer's top-k within the block's allowed range, and
        the draft half is the block's own tokens, which are never sparsified — a
        handful of positions, and the one region the drafter must always see.
        """
        bsz = draft_hidden.shape[0]
        if keep is None:
            keep_ctx = draft_hidden.new_ones((bsz, draft_len, ctx_len), dtype=torch.bool)
            keep_draft = draft_hidden.new_ones((bsz, draft_len, draft_len), dtype=torch.bool)
        else:
            flat = keep[:, 0]
            keep_ctx, keep_draft = flat[..., :ctx_len], flat[..., ctx_len:]

        if run_dsa:
            ctx_indices = self._context_topk(
                draft_hidden, q_resid, index_keys_ctx, cos_q, sin_q, keep_ctx
            )
        else:
            # Top-k over fewer keys than it would keep is a no-op, so hand the
            # kernel every key the training mask allows.
            ctx_indices = keep_mask_to_indices(keep_ctx)
        indices = torch.cat(
            [ctx_indices, keep_mask_to_indices(keep_draft, offset=ctx_len)], dim=-1
        )
        return _finalize_sparse_indices(indices)

    def _attend_sparse(
        self,
        q_absorbed: torch.Tensor,
        latent: torch.Tensor,
        indices: torch.Tensor,
        alive: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse absorbed MLA through the TileLang kernel. Returns ``[B, S, H, v_head_dim]``."""
        latent_output = tilelang_sparse_mla(
            q_absorbed.to(DSA_KERNEL_DTYPE),
            latent.unsqueeze(2).to(DSA_KERNEL_DTYPE),
            indices,
            scaling=self.scaling,
            sink=self.learnable_sink_param if self.learnable_sink else None,
        )[0]
        _, w_vc = self._kv_b_weights()
        attn_output = torch.einsum("bshm,hdm->bshd", latent_output.to(w_vc.dtype), w_vc)
        return torch.where(alive[..., None, None], attn_output, torch.zeros_like(attn_output))

    def _keep_to_block_mask(self, keep: torch.Tensor):
        """``[B, 1, q_len, kv_len]`` bool keep-mask -> flex ``BlockMask``.

        The DSA gate is data-dependent, so unlike the block-causal training mask it
        has no closed-form ``mask_mod``; hand flex the tensor itself as one. Tiles
        the gate empties out are then skipped instead of scored, which is the whole
        reason for routing this path through flex.
        """
        rows = keep[:, 0]
        bsz, q_len, kv_len = rows.shape

        def mask_mod(b, h, q_idx, kv_idx):
            return rows[b, q_idx, kv_idx]

        return compile_friendly_create_block_mask(
            mask_mod, bsz, None, q_len, kv_len, rows.device, _compile=True
        )

    def _attend_flex(
        self,
        query_states: torch.Tensor,
        latent: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        """Dense attention via flex_attention. Returns ``[B, S, H, v_head_dim]``.

        Same result as :meth:`_attend_torch`, but the ``[B, H, draft_len, kv_len]``
        score tensor never exists — flex tiles it, and the sink is folded in
        afterwards from the log-sum-exp flex returns. With ``lse = logsumexp_j l_j``
        and ``total = logsumexp([l, sink])``, the sink softmax is the plain one
        rescaled per row::

            exp(l - total) @ V = exp(lse - total) * (exp(l - lse) @ V)
                               = sigmoid(lse - sink) * flex_out

        A dropped block (``block_keep_mask=False``) has its whole row masked, so
        flex returns ``lse = -inf`` and ``out = 0`` for it; ``sigmoid(-inf - sink)``
        is 0, which keeps that row zero without the explicit zeroing
        :meth:`_attend_torch` needs.
        """
        key_states, value_states = self._decompress_kv(latent)
        attn_output, lse = compile_friendly_flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            scale=self.scaling,
            return_lse=True,
        )
        if self.learnable_sink:
            sink = self.learnable_sink_param.to(torch.float32).view(1, -1, 1)
            rescale = torch.sigmoid(lse - sink).unsqueeze(-1)
            attn_output = attn_output * rescale.to(attn_output.dtype)
        return attn_output.transpose(1, 2)

    def _attend_torch(
        self,
        query_states: torch.Tensor,
        latent: torch.Tensor,
        keep: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Dense reference attention. Returns ``[B, S, H, v_head_dim]``."""
        key_states, value_states = self._decompress_kv(latent)
        logits = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if keep is not None:
            # finfo.min, not -inf: a draft token in a dropped block sees nothing at
            # all, and -inf rows would make the softmax NaN before it is zeroed.
            logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)

        attn_output = torch.matmul(self._softmax_with_sink(logits), value_states)
        if keep is not None:
            # A dropped block (block_keep_mask=False) masks a whole row; softmax of
            # an all-min row is uniform, so zero those rows to match what
            # FlexAttention returns for them.
            attn_output = attn_output * keep.any(-1, keepdim=True).to(attn_output.dtype)
        return attn_output.transpose(1, 2)

    def _output_tail(self, attn_output: torch.Tensor, draft_hidden: torch.Tensor) -> torch.Tensor:
        """hy4's sigmoid output gate, then ``o_proj`` (and ``post_proj_norm``), on
        ``[B, S, H, v_head_dim]``."""
        bsz, draft_len = attn_output.shape[:2]
        if self.gated_mla:
            gate = self.linear_gate(draft_hidden).view(
                bsz, draft_len, self.num_heads, self.gate_projection_size
            )
            attn_output = attn_output * torch.sigmoid(gate)
        output = self.o_proj(
            attn_output.contiguous().reshape(bsz, draft_len, self.num_heads * self.v_head_dim)
        )
        return output if self.post_proj_norm is None else self.post_proj_norm(output)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        full_ctx_len: Optional[int] = None,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        context_latent: Optional[torch.Tensor] = None,
        return_context_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Dual-source MLA.

        Args:
            draft_hidden: ``[B, draft_len, D]`` draft-block hidden states (Q, and
                one of the two KV sources).
            context_hidden: ``[B, ctx_len, D]`` per-layer target context (local
                shard under USP; ``full_ctx_len`` is then the global length).
                Unread, and may be ``None``, when ``context_latent`` is given.
            draft_position_ids / context_position_ids: RoPE positions per source,
                ``[B, S]``, or ``[B, 3, S]`` with M-RoPE.
            block_mask: FlexAttention BlockMask; its ``mask_mod`` defines which
                keys each draft token may read.
            full_ctx_len: global context length under USP, else ``None``.
            sparse_kv_indices: ``[B, draft_len, K]`` int32 keys per draft token, as
                indices into ``context ++ draft`` (``-1`` unused), in place of
                ``block_mask``. Only the tilelang backend reads them; they come from
                ``target_index_layer``'s candidates, which this layer has no
                indexer of its own to pick.
            context_latent: ``[B, ctx_len, kv_lora_rank + qk_rope_head_dim]``
                context latents, RoPE applied in the split layout, in place of the
                ones this layer would project from ``context_hidden``: the target's
                own under ``target_kv_layers``, another draft layer's under
                ``context_kv_source_layers``. This layer's ``kv_b_proj`` reads them
                like its own.
            return_context_latent: also return the context latent this layer
                projected (local shard), for the layers that read it under
                ``context_kv_source_layers``.
        """
        if sparse_kv_indices is not None and self.dsa_backend != "tilelang":
            raise ValueError(
                f"sparse_kv_indices need dsa_backend='tilelang', not {self.dsa_backend!r}: "
                "without them block_mask is None, which would let every key through"
            )
        if self.external_keys and self.dsa_backend == "tilelang" and sparse_kv_indices is None:
            raise ValueError(
                "this layer takes its keys from target_index_layer's candidates, but no "
                "sparse_kv_indices were passed"
            )
        bsz, draft_len, _ = draft_hidden.shape
        if context_latent is not None and self.use_indexer:
            raise ValueError(
                "context_latent replaces context_hidden, which this layer's DSA indexer "
                "scores; target_kv_layers needs index_topk unset"
            )
        ctx_len = (context_hidden if context_latent is None else context_latent).shape[1]

        usp = False
        if full_ctx_len is not None and int(full_ctx_len) != ctx_len:
            if context_latent is not None:
                raise NotImplementedError(
                    "context_latent must span the whole context under USP; DFlashModel gathers "
                    "the target's latents (target_kv_layers) before handing them over"
                )
            self._ensure_usp()
            # The caller's block_mask is sized for the global layout, so a shard
            # that cannot be gathered would silently attend over the wrong keys.
            if self._usp_size <= 1:
                raise ValueError(
                    f"full_ctx_len={full_ctx_len} != local ctx_len={ctx_len} but no "
                    "Ulysses group is initialised"
                )
            if full_ctx_len != ctx_len * self._usp_size:
                raise ValueError(
                    f"USP context shard mismatch: local={ctx_len}, "
                    f"full_ctx_len={full_ctx_len}, sp={self._usp_size}"
                )
            usp = True

        q_nope, q_pe, q_resid = self._project_q(draft_hidden)

        ctx_len_total = int(full_ctx_len) if usp else ctx_len
        kv_len = ctx_len_total + draft_len
        rope_len = max(
            kv_len,
            int(draft_position_ids.max().item()) + 1 if draft_position_ids.numel() else 1,
            int(context_position_ids.max().item()) + 1 if context_position_ids.numel() else 1,
        )
        cos_q, sin_q = self._cos_sin(q_nope, draft_position_ids, rope_len)
        cos_c, sin_c = (
            self._cos_sin(q_nope, context_position_ids, rope_len)
            if context_latent is None
            else (None, None)
        )

        # Top-k over fewer context keys than it would keep is a no-op, so skip it.
        run_dsa = self.use_indexer and ctx_len_total > int(self.dsa_indexer.index_topk)

        latent_ctx = (
            self._project_kv_latent(context_hidden, cos_c, sin_c)
            if context_latent is None
            else context_latent
        )
        local_latent_ctx = latent_ctx
        latent_draft = self._project_kv_latent(draft_hidden, cos_q, sin_q)
        index_keys_ctx = (
            self.dsa_indexer.project_keys(context_hidden, cos_c, sin_c) if run_dsa else None
        )
        if usp:
            # Gather the compressed latent (and the indexer's key rows), not
            # per-head K/V: kv_lora+rope and index_head_dim are both far narrower
            # than num_heads * qk_head_dim, so this moves much less than
            # DFlashAttention's context gather. Both projections are position-wise
            # and RoPE is already applied, so gathering after them is exact.
            latent_ctx = all_gather_seq(latent_ctx.unsqueeze(1), self._usp_group).squeeze(1)
            if index_keys_ctx is not None:
                index_keys_ctx = all_gather_seq(
                    index_keys_ctx.unsqueeze(1), self._usp_group
                ).squeeze(1)

        latent = torch.cat([latent_ctx.to(latent_draft.dtype), latent_draft], dim=1)
        q_pe = _apply_rope_split(_interleaved_to_split_rope(q_pe), cos_q, sin_q)

        if sparse_kv_indices is not None:
            # The keys are already listed; no keep-mask is materialised, so nothing
            # here grows with draft_len * kv_len.
            indices, alive = _finalize_sparse_indices(sparse_kv_indices)
            attn_output = self._attend_sparse(self._absorb_q(q_nope, q_pe), latent, indices, alive)
            out = self._output_tail(attn_output, draft_hidden)
            return (out, local_latent_ctx) if return_context_latent else out

        # flex has no CPU backward, so off-GPU falls back to the dense reference.
        flex = draft_hidden.is_cuda and self.dsa_backend != "tilelang"
        # The dense keep-mask is only materialised where something needs it as a
        # tensor: tilelang turns it into index lists, the DSA gate ANDs into it, and
        # _attend_torch masks with it. The flex path reads the BlockMask directly.
        keep = (
            None
            if flex and not run_dsa
            else materialize_keep_mask(block_mask, bsz, draft_len, kv_len, draft_hidden.device)
        )

        if self.dsa_backend == "tilelang":
            indices, alive = self._sparse_kernel_indices(
                draft_hidden,
                q_resid,
                index_keys_ctx,
                cos_q,
                sin_q,
                keep,
                ctx_len_total,
                draft_len,
                run_dsa,
            )
            attn_output = self._attend_sparse(self._absorb_q(q_nope, q_pe), latent, indices, alive)
        else:
            if run_dsa:
                dsa_keep = self._dsa_keep(
                    draft_hidden, q_resid, index_keys_ctx, cos_q, sin_q, keep, ctx_len_total
                )
                keep = dsa_keep if keep is None else (keep & dsa_keep)
            query_states = torch.cat([q_nope, q_pe], dim=-1)
            if flex:
                attn_output = self._attend_flex(
                    query_states,
                    latent,
                    self._keep_to_block_mask(keep) if run_dsa else block_mask,
                )
            else:
                attn_output = self._attend_torch(query_states, latent, keep)

        out = self._output_tail(attn_output, draft_hidden)
        return (out, local_latent_ctx) if return_context_latent else out


def build_hy4_self_attn(
    config: PretrainedConfig,
    layer_idx: int,
    indexer: Optional[nn.Module] = None,
    own_indexer: bool = True,
) -> nn.Module:
    """One decoder layer's attention: DFlash GQA (default) or hy4 gated MLA."""
    if str(getattr(config, "attention_type", "gqa")) != "mla":
        from angelspec.models.draft.dflash import DFlashAttention

        return DFlashAttention(config)
    return Hy4DraftMLAAttention(config, indexer=indexer, own_indexer=own_indexer)


def resolve_indexer_owners(config: PretrainedConfig, num_layers: int) -> List[int]:
    """Owning layer index per layer, from hy4's ``indexer_types`` schedule.

    ``"full"`` opens a new indexer; ``"shared"`` reads the most recent ``"full"``
    layer's. Without a schedule every layer owns its own, so this is the identity.
    ``DFlyConfig`` has already rejected a schedule that opens with ``"shared"``,
    so every layer has an owner.
    """
    indexer_types = getattr(config, "indexer_types", None)
    if not indexer_types:
        return list(range(num_layers))

    owners: List[int] = []
    owner = 0
    for layer_idx in range(num_layers):
        if str(indexer_types[layer_idx]) == "full":
            owner = layer_idx
        owners.append(owner)
    return owners


def build_hy4_self_attns(
    config: PretrainedConfig, num_layers: int
) -> Tuple[List[nn.Module], Optional[nn.ModuleList]]:
    """Per-layer attention modules for the hy4 layer path.

    Returns the attentions plus, when hy4's ``indexer_types`` schedule asks layers
    to share a DSA indexer, the module list of unique indexers for the *drafter* to
    own (``None`` otherwise). Two layouts, because where an indexer lives decides
    whether it survives training:

    * No sharing: each MLA layer owns one at ``layers.N.self_attn.indexer.*``,
      hy4's own naming.
    * Sharing: every MLA layer borrows, and the indexers move to the drafter root
      (``indexers.K.*``). They cannot stay in the owning layer, because
      ``DFlashTrainer`` shards each decoder layer as its own FSDP unit — layer 0's
      indexer is resharded back to a ``DTensor`` the moment its forward returns,
      i.e. before a borrowing layer reads it. At the root they belong to the root
      FSDP unit, which is never resharded mid-step. Borrowers hold the module
      outside ``nn.Module``'s registry so it keeps a single FQN: DCP's
      ``set_model_state_dict`` deduplicates shared parameters and then loads with
      ``strict=True``, which would report every borrowed path as missing.
    """
    owners = resolve_indexer_owners(config, num_layers)
    shares = (
        str(getattr(config, "attention_type", "gqa")) == "mla"
        and getattr(config, "index_topk", None) is not None
        and owners != list(range(num_layers))
    )

    if not shares:
        attns = [build_hy4_self_attn(config, layer_idx) for layer_idx in range(num_layers)]
        indexers = None
    else:
        indexers = nn.ModuleList()
        slot_of_owner = {}
        for layer_idx, owner in enumerate(owners):
            if owner == layer_idx:
                slot_of_owner[owner] = len(indexers)
                indexers.append(MTPIndexer(config))
        attns = [
            build_hy4_self_attn(
                config,
                layer_idx,
                indexer=indexers[slot_of_owner[owner]],
                own_indexer=False,
            )
            for layer_idx, owner in enumerate(owners)
        ]
        logger.info(
            f"DSA indexers shared per indexer_types: {len(indexers)} indexer(s) for "
            f"{num_layers} layers, owners={owners}"
        )

    if str(getattr(config, "dsa_backend", "torch")) == "auto" and attns:
        resolved = getattr(attns[0], "dsa_backend", "torch")
        if resolved == "tilelang":
            logger.info("dsa_backend='auto' resolved to the TileLang kernels")
        else:
            logger.warning(
                "dsa_backend='auto' resolved to the dense torch path (TileLang kernels "
                "unavailable or the config does not meet their constraints). Its score "
                "tensor is [batch, heads, draft_len, context+draft], i.e. quadratic in "
                "context length, so long-context training will run out of memory."
            )
    return attns, indexers


__all__ = [
    "Hy4DraftMLAAttention",
    "build_hy4_self_attn",
    "build_hy4_self_attns",
    "keep_mask_to_indices",
    "keep_mask_to_ranges",
    "materialize_keep_mask",
    "resolve_indexer_owners",
]
