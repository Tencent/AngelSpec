# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DFlash training model: wraps the DFlash draft model with training-specific logic.

Handles anchor sampling, block-causal mask generation, noise input construction,
and cross-entropy loss with exponential decay weighting.
"""

import os
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.draft.dflash import (
    KERNEL_FREE_DSA_BACKENDS,
    DFlashDraftModel,
    target_layers_per_draft_layer,
    windows_per_draft_layer,
)
from angelspec.models.draft.path_selector import selector_training_candidates
from angelspec.models.ops.dsa_kernels import is_dsa_kernel
from angelspec.models.ops.flex_attention import compile_friendly_create_block_mask
from angelspec.models.ops.loss import _kl_variant_b, lk_tv_kl_per_pos
from angelspec.models.ops.mrope import (
    draft_block_positions,
    mrope_text_block_mask,
    validate_mrope_positions,
)
from angelspec.models.ops.tilelang_dsa import DSA_INDEX_ALIGNMENT
from angelspec.models.ops.ulysses import all_gather_seq
from angelspec.utils.logging import logger

_VALID_DFLASH_LOSS_OBJECTIVES = {"decay", "dpace"}


def _dflash_loss_chunk() -> int:
    """Row budget for chunked draft-logit projection (0/unset => no chunking).

    Mirrors ANGELSPEC_MTP_LOSS_CHUNK. Only the decay + no-distill forward path
    honors it (the production path); dpace / distillation / subclass heads keep
    the single full-vocab projection.
    """
    return int(os.environ.get("ANGELSPEC_DFLASH_LOSS_CHUNK", "0") or 0)


def _dflash_e2e_tv_vocab_chunk() -> int:
    """Vocab-dim chunk size for the e2e-TV loss streaming softmax+min.

    Controls the peak-memory / kernel-launch trade-off inside
    ``_compute_e2e_tv_loss``. ``0`` / unset => fall back to the module default
    (8192). Smaller values reduce peak GPU memory (useful for large vocab or
    long sequences that trigger OOM) at the cost of more kernel launches.
    """
    return int(os.environ.get("ANGELSPEC_DFLASH_E2E_TV_VOCAB_CHUNK", "0") or 0)


# Default vocab-dim chunk size when neither the env var nor the call-site
# override is set. Chosen to keep per-chunk fp32 tensors small while
# amortizing kernel-launch overhead on typical (B*nb*γ, V~150k) shapes.
_DFLASH_E2E_TV_VOCAB_CHUNK_DEFAULT = 8192


def _dpace_position_weights(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute detached D-PACE weights from per-position draft confidences."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights.to(dtype=confidences.dtype)


def _hc_topk_hits(
    candidates: torch.Tensor, label_ids: torch.Tensor, binary_mask: torch.Tensor
) -> torch.Tensor:
    """Per-slot masked count of positions whose label is among their top-k candidate ids.

    The numerator of ``hc_topk_recall``: ``binary_mask`` is ``[..., block_size]``,
    ``candidates`` ``[..., k]`` in the same flattened position order as ``label_ids``.
    Returns ``[block_size]``.
    """
    hits = (candidates.reshape(label_ids.numel(), -1) == label_ids.reshape(-1, 1)).any(dim=-1)
    masked = hits.view(binary_mask.shape).float() * binary_mask.float()
    return masked.reshape(-1, binary_mask.shape[-1]).sum(dim=0)


@torch.no_grad()
def _accept_rates(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    draft_ids: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row speculative-sampling acceptance at ``temperature``, ``q`` and ``p`` the softmax
    of ``student_logits / temperature`` and of ``teacher_logits / temperature`` (``[R, V]``):

    - the drafter samples its token from ``q``: ``sum_v min(p_v, q_v)``;
    - the drafter proposes ``draft_ids`` (``[R]``, its argmax), its distribution then being
      one-hot in the ratio test: ``p`` of that token.

    Returns both, ``[R]`` each, in float32.
    """
    sampled, proposed = [], []
    for start in range(0, student_logits.size(0), 512):
        rows = slice(start, start + 512)
        q = torch.softmax(student_logits[rows].float() / temperature, dim=-1)
        p = torch.softmax(teacher_logits[rows].float() / temperature, dim=-1)
        proposed.append(p.gather(-1, draft_ids[rows].unsqueeze(-1)).squeeze(-1))
        sampled.append(torch.minimum(p, q, out=p).sum(dim=-1))
    return torch.cat(sampled), torch.cat(proposed)


def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    ctx_doc_ids: Optional[torch.Tensor] = None,
    causal: bool = False,
    context_local_window: Optional[int] = None,
    ctx_candidates: Optional[torch.Tensor] = None,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention is bidirectional by default; when ``causal`` is
         True it is causal instead (a draft token only sees draft tokens at or
         before its own position within the same block).
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing
      5. Sequence packing (ctx_doc_ids given): a block additionally only sees
         context tokens in the SAME document as its anchor (and non-padding),
         so packed docs never leak across boundaries.
      6. Context selection, on top of rules 1 and 5: with context_local_window=W
         a block keeps the W context tokens right before its anchor
         (kv_idx >= anchor_pos - W); with ctx_candidates ([B, n_blocks, ctx_len]
         bool) it keeps the positions marked for it; with both, their union.
         One candidate row per block, so all its queries and all layers share
         it. A block's own draft tokens are unaffected.
    """
    num_anchors = anchor_positions.shape[1]
    packed = ctx_doc_ids is not None
    windowed = context_local_window is not None
    selected = ctx_candidates is not None

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < ctx_len
        mask_context = is_context & (kv_idx < anchor_pos)
        if packed or selected:
            # Clamp context index so gathers stay in-bounds for block-region kv
            # positions; only used when is_context is True.
            kv_ctx = torch.where(is_context, kv_idx, torch.zeros_like(kv_idx))
        if windowed and selected:
            in_window = kv_idx >= anchor_pos - context_local_window
            mask_context = mask_context & (in_window | ctx_candidates[b, q_block_id, kv_ctx])
        elif windowed:
            mask_context = mask_context & (kv_idx >= anchor_pos - context_local_window)
        elif selected:
            mask_context = mask_context & ctx_candidates[b, q_block_id, kv_ctx]

        if packed:
            a_doc = ctx_doc_ids[b, anchor_pos]
            kv_doc = ctx_doc_ids[b, kv_ctx]
            mask_context = mask_context & (a_doc >= 0) & (kv_doc == a_doc)

        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        if causal:
            # Intra-block causal: draft token at q_idx only attends to draft
            # tokens at kv positions <= its own position within the block.
            q_pos_in_block = q_idx - q_block_id * block_size
            kv_pos_in_block = (kv_idx - ctx_len) - kv_block_id * block_size
            mask_draft = mask_draft & (kv_pos_in_block <= q_pos_in_block)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    suffix = "_packed" if packed else ""
    if causal:
        suffix += "_causal"
    if windowed:
        suffix += f"_W{context_local_window}"
    if selected:
        suffix += "_cand"
    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}{suffix}"
    return dflash_mask_mod


def _create_dflash_sparse_kv_indices(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    ctx_candidates: Optional[torch.Tensor],
    width: int,
    ctx_doc_ids: Optional[torch.Tensor] = None,
    causal: bool = False,
    context_local_window: Optional[int] = None,
    two_segment: bool = False,
) -> torch.Tensor:
    """The keys :func:`_create_dflash_mask_mod` allows, as one index list per query.

    Same KV layout as the mask (``[context | block_0 | block_1 | ...]``) and the same
    rules, with ``ctx_candidates`` (``[B, n_blocks, K]`` indices, ``-1`` unused, or
    ``None`` for a layer that reads its window alone) in place of the bool table: each
    row lists the context keys the mask keeps for its
    block, then the block's own draft keys. Returns ``[B, n_blocks * block_size,
    width]`` int32, ``-1``-padded, which is what a sparse-attention kernel reads
    instead of evaluating the mask per element — the mask's per-element candidate
    lookup is what keeps flex from skipping any tile.

    A row must not list a key twice (the kernel would weigh it twice), so candidates
    inside the local window are dropped in favour of the window's own entries.
    ``two_segment`` lays the context out twice (window latents, then global latents): candidates
    index the second copy and the draft keys start at ``2 * ctx_len``.
    ``width`` is fixed by the caller so the kernel's tile choice, keyed on it, does
    not change from step to step.
    """
    n_blocks = anchor_positions.shape[1]
    device = anchor_positions.device
    anchors = anchor_positions.to(device=device, dtype=torch.long)
    keep = block_keep_mask.to(device=device, dtype=torch.bool)
    draft_len = n_blocks * block_size

    # Context rules, as in the mask: strictly before the anchor, inside the context, and
    # with packing, in the anchor's (non-padding) document.
    ctx_keep = keep
    docs = anchor_doc = None
    if ctx_doc_ids is not None:
        docs = ctx_doc_ids.to(device=device, dtype=torch.long)
        anchor_doc = docs.gather(1, anchors.clamp(0, ctx_len - 1))
        ctx_keep = ctx_keep & (anchor_doc >= 0)

    def visible(pos: torch.Tensor) -> torch.Tensor:
        ok = (pos >= 0) & (pos < ctx_len) & (pos < anchors.unsqueeze(-1))
        if docs is not None:
            pos_doc = docs.gather(1, pos.clamp(0, ctx_len - 1).flatten(1)).view_as(pos)
            ok = ok & (pos_doc == anchor_doc.unsqueeze(-1))
        return ok & ctx_keep.unsqueeze(-1)

    if ctx_candidates is None and context_local_window is None:
        raise ValueError("sparse KV rows need ctx_candidates, a context_local_window, or both")
    parts = []
    if ctx_candidates is not None:
        candidates = ctx_candidates.to(device=device, dtype=torch.long)
        cand_ok = visible(candidates)
    if context_local_window is not None:
        window = int(context_local_window)
        start = anchors - window
        positions = start.unsqueeze(-1) + torch.arange(window, device=device)
        parts.append(torch.where(visible(positions), positions, -1))
        if ctx_candidates is not None and not two_segment:
            cand_ok = cand_ok & (candidates < start.unsqueeze(-1))
    if ctx_candidates is not None:
        offset = ctx_len if two_segment else 0
        parts.insert(0, torch.where(cand_ok, candidates + offset, -1))
    ctx_rows = torch.cat(parts, dim=-1).to(torch.int32).repeat_interleave(block_size, dim=1)

    # Draft rules: a query sees its own block only, causally within it if asked.
    q_idx = torch.arange(draft_len, device=device)
    in_block = torch.arange(block_size, device=device)
    draft_start = 2 * ctx_len if two_segment else ctx_len
    draft_keys = draft_start + (q_idx // block_size).unsqueeze(-1) * block_size + in_block
    draft_ok = keep.repeat_interleave(block_size, dim=1).unsqueeze(-1)
    if causal:
        draft_ok = draft_ok & (in_block <= (q_idx % block_size).unsqueeze(-1))
    draft_rows = torch.where(draft_ok, draft_keys.to(torch.int32), -1)

    rows = torch.cat([ctx_rows, draft_rows], dim=-1)
    if rows.shape[-1] > width:
        raise ValueError(
            f"sparse KV rows need {rows.shape[-1]} slots but width={width}; size it for "
            "target_index_topk + context_local_window + block_size"
        )
    return F.pad(rows, (0, width - rows.shape[-1]), value=-1).contiguous()


def _build_per_layer(
    candidates: Optional[List[Optional[torch.Tensor]]], windows: List[Optional[int]], build
):
    """``build(c, w)`` once per distinct per-layer pair of candidate tensor ``c`` (``None``
    for a layer without candidates) and context window ``w``.

    ``candidates`` is ``None`` when no layer has candidates. One result when every layer
    shares the same pair, else a list with one result per layer.
    """
    if candidates is None:
        candidates = [None] * len(windows)
    built, keys = {}, []
    for c, w in zip(candidates, windows):
        key = (None if c is None else id(c), w)
        if key not in built:
            built[key] = build(c, w)
        keys.append(key)
    if len(built) == 1:
        return next(iter(built.values()))
    return [built[key] for key in keys]


def _candidate_overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[B, n_blocks]``: the share of each block's ``a`` candidates that ``b`` also picks.

    Both are ``[B, n_blocks, K]`` candidate indices, ``-1`` in unused slots.
    """
    b_sorted = b.sort(-1).values
    pos = torch.searchsorted(b_sorted, a.contiguous()).clamp(max=b_sorted.shape[-1] - 1)
    hit = (b_sorted.gather(-1, pos) == a) & (a >= 0)
    return hit.sum(-1) / (a >= 0).sum(-1).clamp(min=1)


#: Candidate-set metrics, in the order their sums and counts are all_reduced. Every key
#: is emitted whenever any of them is on (a config fact, so the collective stays
#: symmetric); the trainer drops the ones no rank counted.
INDEX_METRIC_KEYS = ("index_union_size",)


def _sum_and_count(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(sum, count)`` of ``values``, which the trainer divides after its all_reduce.

    Reducing a per-rank mean instead would weigh a rank holding one block the same as one
    holding a thousand, and a rank holding none would fold in a 0: on an eval set with
    fewer samples than ranks (codespan_64k has 32 for 64) that halved the reported value.
    Carrying both parts also keeps the collective symmetric, which is what a metric a rank
    could otherwise drop needs.
    """
    flat = values.detach().float()
    return flat.sum(), torch.full((), float(flat.numel()), device=flat.device)


def _union_candidate_rows(indices: torch.Tensor) -> torch.Tensor:
    """Set-union along the last dim of ``[B, N, *]`` candidate indices (``-1`` unused).

    Returns a sorted, ``-1``-padded tensor of the same rank; used by
    ``target_index_union_layers`` (several target indexers at the same anchors).
    """
    if indices.ndim != 3:
        raise ValueError(f"expected [B, N, K], got {tuple(indices.shape)}")
    pad = torch.iinfo(indices.dtype).max
    ids = indices.masked_fill(indices < 0, pad).sort(dim=-1).values
    repeat = F.pad(ids[..., 1:] == ids[..., :-1], (1, 0), value=False)
    ids = ids.masked_fill(repeat, pad).sort(dim=-1).values
    return ids.masked_fill(ids == pad, -1)


def _union_layer_candidates(indices_by_layer: List[torch.Tensor]) -> torch.Tensor:
    """``[B, N, n_layers * K]``: set-union of several ``[B, N, K]`` indexer picks."""
    if not indices_by_layer:
        raise ValueError("need at least one layer's candidates to union")
    shape0 = indices_by_layer[0].shape
    for i, indices in enumerate(indices_by_layer):
        if indices.ndim != 3 or indices.shape[:2] != shape0[:2]:
            raise ValueError(
                f"layer candidates[{i}] shape {tuple(indices.shape)} != "
                f"{tuple(shape0[:2])} + K"
            )
    return _union_candidate_rows(torch.cat(indices_by_layer, dim=-1))


def _target_latent_mismatch(module: nn.Module, attn: nn.Module, draft_config) -> Optional[str]:
    """Why ``module``'s latents cannot replace ``attn``'s own context latents, else None.

    The drafter reads them with its own RoPE'd queries and ``kv_b_proj``, so the latent
    widths, the RoPE frequencies and the M-RoPE layout must all agree. A module with a
    ``mismatch`` method (DeepSeek-V4.1's) answers for itself.
    """
    if hasattr(module, "mismatch"):
        return module.mismatch(attn)
    widths = (module.kv_lora_rank, module.qk_rope_head_dim)
    expected = (attn.kv_lora_rank, attn.qk_rope_head_dim)
    if widths != expected:
        return f"(kv_lora_rank, qk_rope_head_dim) {widths} vs the drafter's {expected}"
    if not attn.qk_rope_head_dim:
        # NoPE (glm53_mla): no positions in the latent to agree on.
        return None
    # rtol: the drafter's rotary buffers may have been cast to bf16 with its weights.
    if not torch.allclose(
        module.rotary_emb.inv_freq.float().cpu(),
        attn.rotary_emb.inv_freq.float().cpu(),
        rtol=1e-2,
        atol=0.0,
    ):
        return "the RoPE frequencies (rope_theta) differ"
    recipe = module._position_recipe
    if (recipe is not None) != bool(attn.uses_mrope):
        return "the M-RoPE layout (mrope_section / mrope_interleaved) differs"
    if recipe is None:
        return None
    layout = (tuple(module.rotary_emb.section), bool(module.rotary_emb.interleaved))
    draft_layout = (tuple(attn.rotary_emb.section), bool(attn.rotary_emb.interleaved))
    if layout != draft_layout:
        return (
            f"the M-RoPE layout (mrope_section, mrope_interleaved) {layout} vs the "
            f"drafter's {draft_layout}"
        )
    names = ("image_start_token_id", "image_end_token_id", "image_newline_token_id")
    token_ids = tuple(getattr(recipe, name) for name in names)
    draft_token_ids = tuple(getattr(draft_config, name, None) for name in names)
    if getattr(draft_config, "position_provider", None) != "provided" and (
        token_ids != draft_token_ids
    ):
        return f"image token ids {token_ids} vs the drafter's {draft_token_ids}"
    return None


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with exponential decay weighting
      - Per-position loss_mask application
      - The drafter's optional hidden-states correction, before the LM head
      - The drafter's optional path selector, reranking the unary top-k
      - The drafter's optional MTP fusion, added to the block input
    """

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        fp32_lm_head: bool = True,
        gate_entropy_weight: float = 0.0,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        ce_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 0.0,
        kl_loss_weight: float = 0.0,
        kl_topk: int = 10,
        lk_loss_weight: float = 0.0,
        lk_loss_type: str = "hybrid",
        lk_eta: float = 3.0,
        e2e_tv_loss_weight: float = 0.0,
        e2e_tv_vocab_chunk: Optional[int] = None,
        path_selector_loss_alpha: float = 1.0,
        path_selector_teacher_labels: bool = True,
        path_selector_objective_weighting: bool = True,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group=None,
        causal_attn: bool = False,
        loss_checkpoint_chunks: bool = False,
        attn_diag_blocks: int = 0,
        attn_oracle: str = "",
        attn_oracle_layers: Optional[List[int]] = None,
        hc_topk_diag: Optional[List[int]] = None,
        eval_accept_temperature: Optional[float] = None,
        attentive_correction_eval_off: bool = False,
    ):
        super().__init__()
        loss_objective = loss_objective.lower()
        if loss_objective not in _VALID_DFLASH_LOSS_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_DFLASH_LOSS_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash loss objective {loss_objective!r}; expected one of {valid}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.loss_checkpoint_chunks = loss_checkpoint_chunks
        self.draft_model = draft_model
        cfg = getattr(draft_model, "config", None)
        self.uses_mrope = bool(
            (getattr(cfg, "rope_parameters", None) or {}).get("mrope_section")
            and getattr(cfg, "attention_type", None) == "mla"
        )
        # The context window rides on the drafter's config too, so the exported
        # config.json records the window the drafter was trained under.
        self.context_local_window = getattr(cfg, "context_local_window", None)
        # vLLM's sample_from_anchor layout (DeepSeek's DSpark): slot 0 holds the anchor
        # token and, like every slot, predicts the next one, so the labels sit one
        # position right of DFlash's and slot 0 is supervised.
        self.sample_from_anchor = bool(getattr(cfg, "sample_from_anchor", False))
        self._label_shift = int(self.sample_from_anchor)
        self.first_pred_slot = 1 - self._label_shift
        # Number of highest uncorrected logits the hidden correction reaches
        # (DFlashConfig.enable_hidden_correction_topk); 0 corrects every logit.
        self.hidden_correction_topk = (
            int(cfg.hidden_correction_topk)
            if getattr(cfg, "enable_hidden_correction_topk", False)
            else 0
        )
        if (
            self.hidden_correction_topk
            and type(self)._compute_draft_logits is not DFlashModel._compute_draft_logits
        ):
            raise ValueError(
                f"{type(self).__name__} computes its own draft logits and would ignore "
                "enable_hidden_correction_topk"
            )
        # Top-k candidate ids of the latest _compute_draft_logits call, read right after
        # it for the hc_topk_recall metric; stays None without hidden_correction_topk.
        self._hc_topk_candidates: Optional[torch.Tensor] = None
        # Whether each slot's previous token tops the previous slot's uncorrected logits
        # (``[B, N]``, slots 0 and 1 False) in the latest _compute_draft_logits call, read
        # right after it for the hc_prev_top1_rate / hc_acc_prev_* metrics; stays None
        # without hidden_correction_topk.
        self._hc_prev_top1: Optional[torch.Tensor] = None
        # Target-indexer context candidates (DFlashConfig.target_index_layer) and target
        # KV reuse (DFlashConfig.target_kv_layers), each resolved to one target layer per
        # draft layer. The trainer attaches one frozen TargetIndexer / TargetKVLatent per
        # distinct target layer, held in plain lists rather than registered, so they are
        # neither FSDP-wrapped nor saved with the drafter.
        num_layers = len(getattr(draft_model, "layers", []))
        # A drafter without .layers still needs its one mask under the shared window.
        self._context_windows = windows_per_draft_layer(self.context_local_window, num_layers) or [
            self.context_local_window
        ]
        self.target_index_layer = getattr(cfg, "target_index_layer", None)
        self.target_index_topk = getattr(cfg, "target_index_topk", None)
        self._target_index_layers = target_layers_per_draft_layer(
            self.target_index_layer, num_layers
        )
        self.target_kv_layers = target_layers_per_draft_layer(
            getattr(cfg, "target_kv_layers", None), num_layers
        )
        # Shared union of several target indexers' top-k. Mean cardinality over kept blocks
        # is logged as index_union_size.
        union_layers = getattr(cfg, "target_index_union_layers", None)
        self._index_union_layers = (
            [int(layer) for layer in union_layers] if union_layers is not None else None
        )
        self._index_union_size: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # Target layer L reads its input from captured target_layer_id L - 1. Modules that
        # read further layers (DeepSeek-V4.1's KV source) add theirs when attached.
        self._target_hidden_index = {
            layer: list(draft_model.target_layer_ids).index(layer - 1)
            for layer in (
                (self._target_index_layers or [])
                + (self.target_kv_layers or [])
                + (self._index_union_layers or [])
            )
            if layer is not None
        }
        # DeepSeek-V4.1's context is window + global latents, which only key lists can address.
        self._two_segment_kv = (
            self.target_kv_layers is not None
            and getattr(cfg, "attention_type", None) == "dsv41_mla"
        )
        if (
            self.target_kv_layers is not None
            or len(set(self._target_index_layers or [])) > 1
            or len(set(self._context_windows)) > 1
            or getattr(cfg, "context_kv_source_layers", None) is not None
        ) and type(draft_model).forward is not DFlashDraftModel.forward:
            raise ValueError(
                "per-layer target layers (target_kv_layers, or a target_index_layer list naming "
                "several layers), per-layer context_local_window lists and "
                "context_kv_source_layers are routed by DFlashDraftModel.forward, which "
                f"{type(draft_model).__name__} overrides"
            )
        self._target_indexer_ref: list = []
        self._target_kv_latent_ref: list = []
        self._target_index_logged = False
        self._target_kv_logged = False
        # Key-list backends (DSA kernels or KERNEL_FREE_DSA_BACKENDS) take the candidates as index
        # lists with no block mask; flex would look each up per element and score every tile.
        attns = [getattr(layer, "self_attn", None) for layer in getattr(draft_model, "layers", [])]
        backends = [getattr(attn, "dsa_backend", None) for attn in attns]
        listed = [
            is_dsa_kernel(backend) or backend in KERNEL_FREE_DSA_BACKENDS for backend in backends
        ]
        self._sparse_kv_alignment = max(
            [
                attn.dsa_kernel.index_alignment
                for attn in attns
                if getattr(attn, "dsa_kernel", None)
            ]
            or [DSA_INDEX_ALIGNMENT]
        )
        if self.target_index_layer is not None and any(listed) and not all(listed):
            raise ValueError(
                f"target_index_layer needs every attention layer on one backend, got {backends}"
            )
        self._sparse_kv_attention = (
            (self.target_index_layer is not None and bool(listed) and all(listed))
            or bool(getattr(draft_model, "sparse_kv_attention", False))
            or self._two_segment_kv
        )
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        # fp32 draft logits before CE: FSDP2's bf16 compute otherwise rounds ~2% of
        # small-margin argmax decisions that CE grad and the acc metric depend on.
        self.fp32_lm_head = fp32_lm_head
        # Optional sparsity penalty on the gated_sum layer-selection gate (plan B).
        # Default 0 => no-op; only takes effect on a DFlashGatedDraftModel.
        self.gate_entropy_weight = gate_entropy_weight
        # Loss objective + optional distillation terms (unified loss architecture).
        self.loss_objective = loss_objective
        self.dpace_alpha = dpace_alpha
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        # KL / LK distillation against the target's true last-layer logits. Both
        # are convex-mix coefficients in [0, 1] against CE; LK and KL are mutually
        # exclusive (LK takes precedence when both are set).
        self.kl_loss_weight = float(kl_loss_weight)
        self.kl_topk = int(kl_topk)
        self.lk_loss_weight = float(lk_loss_weight)
        self.lk_loss_type = str(lk_loss_type)
        self.lk_eta = float(lk_eta)
        # End-to-end multi-step TV loss (independent term, added to the total;
        # not mutually exclusive with KL/LK). 0 => off. Fixed T=1.
        self.e2e_tv_loss_weight = float(e2e_tv_loss_weight)
        # Vocab-dim chunk size for the streaming softmax+min inside
        # ``_compute_e2e_tv_loss``. ``None`` defers to env var
        # ``ANGELSPEC_DFLASH_E2E_TV_VOCAB_CHUNK`` and then the module default
        # (see ``_compute_e2e_tv_loss``). Expose as a knob so users can trade
        # peak GPU memory vs. kernel launches without touching env vars.
        self.e2e_tv_vocab_chunk = (
            int(e2e_tv_vocab_chunk) if e2e_tv_vocab_chunk is not None else None
        )
        # Weight of the path selector's K-way CE. Independent term added on top of
        # the total (like e2e-TV), and a no-op unless the drafter built a selector
        # (``enable_path_selector`` in the draft config).
        self.path_selector_loss_alpha = float(path_selector_loss_alpha)
        # Label the selector with the target's argmax once the corpus CE no longer
        # drives the unary head (KL/LK displace it, ce_loss_alpha=0 drops it), so both
        # heads optimize the same token distribution (see the selector block in
        # ``forward``). False pins it to the corpus token.
        self.path_selector_teacher_labels = bool(path_selector_teacher_labels)
        # Whether the selector's K-way CE rides ``objective_weights`` (decay /
        # D-PACE) like the unary terms, or weighs every supervised position
        # equally. The continuation-value argument behind D-PACE is about which
        # block position is worth fixing, but the selector's parameters are two
        # token-indexed codebooks plus one projection — shared across positions
        # and teacher-forced on ground-truth predecessors at every slot — so the
        # position weighting mostly reshapes the bigram statistics it sees rather
        # than allocating capacity. False trains it on the unweighted mask.
        self.path_selector_objective_weighting = bool(path_selector_objective_weighting)
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        self.causal_attn = bool(causal_attn)
        # One-shot runtime proof that context/anchors were actually sharded.
        self._usp_runtime_logged = False
        if getattr(cfg, "context_kv_source_layers", None) is not None and self.sp_size > 1:
            raise NotImplementedError(
                "context_kv_source_layers hands a layer's local context latent to later layers, "
                "which would then read one shard of it; USP (sp_size > 1) is not supported"
            )
        # loss_components entries from the last backbone pass (see _add_mtp_fusion).
        self._mtp_fusion_stats: dict = {}
        self._mtp_fusion_logged = False
        # Attentive correction (DFlashConfig.attentive_correction): the context of its
        # source draft layer, set by each backbone pass, and its output's size per block
        # range for logging. attentive_correction_eval_off leaves it out at eval.
        correction = getattr(draft_model, "hidden_correction", None)
        self._acorr = getattr(correction, "attention", None) is not None
        self._acorr_source: Optional[int] = None
        if self._acorr:
            if self.target_kv_layers is None:
                raise ValueError(
                    "attentive_correction reads a draft layer's target latents; set "
                    "target_kv_layers"
                )
            if not self._sparse_kv_attention:
                raise ValueError(
                    "attentive_correction reads a draft layer's context candidates and target "
                    "latents; it needs target_index_layer on key lists "
                    "(a DSA kernel backend such as tilelang, or grouped / flex_attention)"
                )
            self._acorr_source = int(draft_model.config.attentive_correction_source_layer) % len(
                draft_model.layers
            )
        self._acorr_block_keys = self._acorr and bool(correction.attention.block_keys)
        self.acorr_eval_off = bool(attentive_correction_eval_off)
        self._acorr_context: Optional[dict] = None
        self._acorr_sums: dict = {}
        # Eval-only draft-vs-target attention comparison (dflash_attn_diag) or intervention
        # (dflash_attn_oracle, which takes precedence); both need the frozen target attention
        # copies the trainer attaches.
        self.attn_diag_blocks = int(attn_diag_blocks)
        self.attn_oracle = str(attn_oracle or "")
        self.attn_oracle_layers = list(attn_oracle_layers) if attn_oracle_layers else None
        if self.sp_size > 1 and (self._acorr or self.attn_diag_blocks > 0 or self.attn_oracle):
            raise NotImplementedError(
                "the attentive correction and the attention diagnostic/oracle read the target "
                "hidden over the whole context on one rank; USP (sp_size > 1) is not supported"
            )
        self._attn_diag_teachers: list = []
        self._attn_diag_stats: dict = {}
        # Eval-only fidelity check of the top-K hidden correction (dflash_hc_topk_diag).
        self.hc_topk_diag = sorted({int(k) for k in hc_topk_diag}) if hc_topk_diag else []
        if self.hc_topk_diag and self._acorr_block_keys:
            raise ValueError(
                "dflash_hc_topk_diag walks one slot at a time and does not feed the attentive "
                "correction's block keys (attentive_correction_block_keys)"
            )
        # Eval-only sampling acceptance at this temperature (dflash_eval_accept_temperature).
        if eval_accept_temperature is not None and not eval_accept_temperature > 0:
            raise ValueError(
                f"dflash_eval_accept_temperature must be > 0, got {eval_accept_temperature}"
            )
        self.eval_accept_temperature = eval_accept_temperature

    def attach_target_indexer(self, indexer: nn.Module) -> None:
        """Hand over the frozen ``TargetIndexer`` that picks each block's context candidates."""
        self.attach_target_indexers([indexer])

    def attach_target_indexers(self, indexers: List[nn.Module]) -> None:
        """Hand over one frozen ``TargetIndexer`` per scored target index layer."""
        if self.target_index_layer is None:
            raise ValueError("the drafter config sets no target_index_layer")
        want = sorted(
            set(self._index_union_layers)
            if self._index_union_layers is not None
            else {layer for layer in self._target_index_layers if layer is not None}
        )
        if sorted(indexer.layer for indexer in indexers) != want or any(
            indexer.topk != self.target_index_topk for indexer in indexers
        ):
            got = ", ".join(
                f"TargetIndexer(layer={indexer.layer}, topk={indexer.topk})"
                for indexer in indexers
            )
            raise ValueError(
                f"{got} does not match the drafter config (target_index_layer="
                f"{self.target_index_layer}, target_index_union_layers="
                f"{self._index_union_layers}, target_index_topk={self.target_index_topk})"
            )
        self._index_target_inputs(indexers)
        self._target_indexer_ref = list(indexers)

    def _index_target_inputs(self, modules: List) -> None:
        captured = list(self.draft_model.target_layer_ids)
        for module in modules:
            for layer in getattr(module, "input_layers", ()):
                if layer - 1 not in captured:
                    raise ValueError(
                        f"{type(module).__name__}(layer={module.layer}) reads target layer "
                        f"{layer}'s input, but target_layer_id {layer - 1} is not captured "
                        f"(target_layer_ids={captured})"
                    )
                self._target_hidden_index[layer] = captured.index(layer - 1)

    def _target_inputs(self, module, hidden_states_list: List[torch.Tensor]):
        layers = getattr(module, "input_layers", None)
        if layers is None:
            return hidden_states_list[self._target_hidden_index[module.layer]]
        return [hidden_states_list[self._target_hidden_index[layer]] for layer in layers]

    def attach_target_kv_latents(self, modules: List[nn.Module]) -> None:
        """Hand over one frozen ``TargetKVLatent`` per target layer in ``target_kv_layers``."""
        if self.target_kv_layers is None:
            raise ValueError("the drafter config sets no target_kv_layers")
        if sorted(module.layer for module in modules) != sorted(set(self.target_kv_layers)):
            raise ValueError(
                f"TargetKVLatent layers {[module.layer for module in modules]} do not match "
                f"the drafter config (target_kv_layers={self.target_kv_layers})"
            )
        layouts = {getattr(module, "context_layout", "shared") for module in modules}
        expected = "window_global" if self._two_segment_kv else "shared"
        if layouts != {expected}:
            raise ValueError(
                f"the target KV modules lay the context out as {sorted(layouts)}, but a "
                f"{getattr(self.draft_model.config, 'attention_type', None)} drafter reads it "
                f"as {expected!r}"
            )
        self._index_target_inputs(modules)
        attn = self.draft_model.layers[0].self_attn
        for module in modules:
            mismatch = _target_latent_mismatch(module, attn, self.draft_model.config)
            if mismatch is not None:
                raise ValueError(
                    f"TargetKVLatent(layer={module.layer}) cannot stand in for the drafter's "
                    f"own context latents: {mismatch}"
                )
        self._target_kv_latent_ref = list(modules)

    def attach_attn_diag_teachers(self, modules: List[nn.Module]) -> None:
        """Hand over one frozen ``TargetAttentionTeacher`` per layer in ``target_kv_layers``."""
        if self.target_kv_layers is None:
            raise ValueError("the attention diagnostic compares against target_kv_layers")
        if getattr(self.draft_model.config, "attention_type", None) != "mla":
            raise ValueError("the attention diagnostic reads hy4's MLA (attention_type='mla')")
        if sorted(module.layer for module in modules) != sorted(set(self.target_kv_layers)):
            raise ValueError(
                f"TargetAttentionTeacher layers {[module.layer for module in modules]} do not "
                f"match target_kv_layers={self.target_kv_layers}"
            )
        self._attn_diag_teachers = list(modules)

    def _attn_probe(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
    ):
        """This forward's eval-time attention intervention or diagnostic, or ``None`` when off."""
        if (
            self.training
            or (self.attn_diag_blocks <= 0 and not self.attn_oracle)
            or not self._attn_diag_teachers
            or not self._sparse_kv_attention
        ):
            return None
        from angelspec.data.positions import position_recipe_from_config
        from angelspec.models.dflash_attn_diag import AttnDiag, AttnOracle

        # Image spans are only known from the Hy4 flash token layout; provided positions carry none.
        recipe = position_recipe_from_config(self.draft_model.config)
        image_token_ids = (
            (recipe.image_start_token_id, recipe.image_end_token_id, recipe.image_newline_token_id)
            if recipe is not None and recipe.provider == "hy_v4_flash"
            else None
        )
        teachers = {module.layer: module for module in self._attn_diag_teachers}
        teachers = [teachers[layer] for layer in self.target_kv_layers]
        target_hidden = [
            hidden_states_list[self._target_hidden_index[layer]] for layer in self.target_kv_layers
        ]
        if self.attn_oracle:
            return AttnOracle(
                mode=self.attn_oracle,
                layers=self.draft_model.layers,
                teachers=teachers,
                target_hidden=target_hidden,
                input_ids=input_ids,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                block_size=self.block_size,
                image_token_ids=image_token_ids,
                layer_indices=self.attn_oracle_layers,
            )
        return AttnDiag(
            layers=self.draft_model.layers,
            teachers=teachers,
            names=[f"d{i}-L{layer}" for i, layer in enumerate(self.target_kv_layers)],
            target_hidden=target_hidden,
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            ctx_doc_ids=ctx_doc_ids,
            block_size=self.block_size,
            num_blocks=self.attn_diag_blocks,
            index_topk=self.target_index_topk,
            image_token_ids=image_token_ids,
        )

    def _target_index_candidates(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Context candidates from the attached ``TargetIndexer``s, per draft layer.

        Returns per-layer lists of ``[B, n_blocks, L]`` bool tables and ``[B, n_blocks,
        K]`` indices; layers reading the same target layer share one tensor, and a layer
        without a target layer gets ``None`` for both. The table is what ``mask_mod``
        reads, so it is ``None`` when the attention takes the indices directly
        (``_sparse_kv_attention``) and no mask is built. Under USP
        ``hidden_states_list`` is this rank's context shard and ``anchor_positions`` its
        own anchors; candidates are still positions in the whole sequence.
        """
        if not self._target_indexer_ref:
            raise RuntimeError(
                "target_index_layer is set but no TargetIndexer is attached "
                "(DFlashTrainer attaches it in init_model; call attach_target_indexer)"
            )
        from angelspec.models.target.target_indexer import candidates_to_table

        by_layer = {}
        for indexer in self._target_indexer_ref:
            indices = indexer.candidate_indices(
                self._target_inputs(indexer, hidden_states_list),
                input_ids,
                anchor_positions,
                ctx_doc_ids,
                position_ids=position_ids,
                **({"sp_group": self.sp_group} if self.sp_size > 1 else {}),
            )
            table = (
                None
                if self._sparse_kv_attention
                else candidates_to_table(indices, input_ids.shape[1])
            )
            by_layer[indexer.layer] = (table, indices)
        if self._index_union_layers is not None:
            missing = [layer for layer in self._index_union_layers if layer not in by_layer]
            if missing:
                raise RuntimeError(
                    f"target_index_union_layers={self._index_union_layers} missing indexers "
                    f"for {missing}"
                )
            unioned = _union_layer_candidates(
                [by_layer[layer][1] for layer in self._index_union_layers]
            )
            self._index_union_size = _sum_and_count((unioned >= 0).sum(dim=-1)[block_keep_mask])
            table = (
                None
                if self._sparse_kv_attention
                else candidates_to_table(unioned, input_ids.shape[1])
            )
            shared = (table, unioned)
        else:
            self._index_union_size = None
        if not self._target_index_logged:
            self._target_index_logged = True
            if not dist.is_initialized() or dist.get_rank() == 0:
                log_indices = (
                    {"∪": shared[1]}
                    if self._index_union_layers is not None
                    else {layer: indices for layer, (_, indices) in by_layer.items()}
                )
                self._log_target_index(
                    log_indices,
                    input_ids,
                    anchor_positions,
                    block_keep_mask,
                    ctx_doc_ids,
                )
        if self._index_union_layers is not None:
            per_layer = [
                (None, None) if layer is None else shared for layer in self._target_index_layers
            ]
        else:
            per_layer = [
                (None, None) if layer is None else by_layer[layer]
                for layer in self._target_index_layers
            ]
        return [table for table, _ in per_layer], [indices for _, indices in per_layer]

    def _log_target_index(
        self,
        indices_by_layer: dict,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
    ) -> None:
        """One-shot candidate stats per target layer, and how much the layers agree."""
        from angelspec.models.target.target_indexer import document_starts

        bsz, seq_len = input_ids.shape
        doc_start = document_starts(ctx_doc_ids, bsz, seq_len, anchor_positions.device)
        visible = anchor_positions - doc_start.gather(1, anchor_positions)
        if not bool(block_keep_mask.any()):
            return

        if self._index_union_layers is not None:
            where = f"union of target layers {self._index_union_layers} at anchor-1"
        else:
            where = "anchor-1"
        for layer, indices in indices_by_layer.items():
            # The indexer's top-k slots are distinct positions or -1, so this counts
            # the same set the table holds. For a layer union the last dim is up to
            # n_layers * topk and the mean is the logged union size.
            picked = (indices >= 0).sum(-1)[block_keep_mask]
            size_name = (
                "union size"
                if self._index_union_layers is not None
                else f"top-{self.target_index_topk}"
            )
            logger.info(
                f"[target index] layer {layer} {size_name} at {where} "
                f"(+ context_local_window={self.context_local_window}): "
                f"{picked.float().mean():.0f} candidates per block over "
                f"{visible[block_keep_mask].float().mean():.0f} visible context tokens; "
                f"{(visible[block_keep_mask] > self.target_index_topk).float().mean():.1%} "
                "of blocks with more visible tokens than a single top-k"
            )
        cut = block_keep_mask & (visible > self.target_index_topk)
        layers = list(indices_by_layer)
        if len(layers) > 1 and bool(cut.any()):
            pairs = [
                f"{a}/{b}="
                f"{_candidate_overlap(indices_by_layer[a], indices_by_layer[b])[cut].mean():.2f}"
                for i, a in enumerate(layers)
                for b in layers[i + 1 :]
            ]
            logger.info(
                "[target index] share of candidates two target layers both pick, over cut "
                f"blocks: {' '.join(pairs)}"
            )

    def _target_kv_context(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        ctx_doc_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Each draft layer's context latent, its target layer's own (``target_kv_layers``).

        ``[B, L, kv_lora_rank + qk_rope_head_dim]`` per layer over the whole sequence;
        layers reading the same target layer share one tensor. Under USP
        ``hidden_states_list`` is this rank's context shard: each rank projects its own
        positions and the latents are gathered, so every rank's anchors can read any key.
        """
        if not self._target_kv_latent_ref:
            raise RuntimeError(
                "target_kv_layers is set but no TargetKVLatent is attached "
                "(DFlashTrainer attaches them in init_model; call attach_target_kv_latents)"
            )
        sp_group = self.sp_group if self.sp_size > 1 else None
        by_layer = {}
        for module in self._target_kv_latent_ref:
            latent = module.latents(
                self._target_inputs(module, hidden_states_list),
                input_ids,
                ctx_doc_ids,
                position_ids=position_ids,
                sp_group=sp_group,
            )
            if sp_group is not None:
                with torch.no_grad():
                    latent = all_gather_seq(latent.unsqueeze(1), sp_group).squeeze(1)
            by_layer[module.layer] = latent
        if not self._target_kv_logged:
            self._target_kv_logged = True
            if not dist.is_initialized() or dist.get_rank() == 0:
                rank = self._target_kv_latent_ref[0].kv_lora_rank
                rms = " ".join(
                    f"{layer}={latent[..., :rank].float().pow(2).mean().sqrt():.3f}"
                    + (
                        f"/{latent[..., rank:].float().pow(2).mean().sqrt():.3f}"
                        if latent.shape[-1] > rank
                        else ""
                    )
                    for layer, latent in by_layer.items()
                )
                logger.info(
                    f"[target kv] draft layers read target layers {self.target_kv_layers}' MLA "
                    f"latents as their context KV; latent / k_pe RMS per target layer: {rms}"
                )
        return [by_layer[layer] for layer in self.target_kv_layers]

    def _usp_pad(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        last_hidden_states: Optional[torch.Tensor] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        context_pre_sharded: bool = False,
    ):
        """Right-pad the full sequence to a multiple of ``sp_size`` (pad tokens
        get loss/attention mask 0) so every SP rank gets an equal context shard.
        """
        if self.sp_size <= 1:
            return (
                input_ids,
                loss_mask,
                hidden_states_list,
                last_hidden_states,
                ctx_doc_ids,
                base_position_ids,
            )
        seq_len = input_ids.shape[1]
        rem = seq_len % self.sp_size
        if rem == 0:
            return (
                input_ids,
                loss_mask,
                hidden_states_list,
                last_hidden_states,
                ctx_doc_ids,
                base_position_ids,
            )
        pad = self.sp_size - rem
        input_ids = F.pad(input_ids, (0, pad), value=0)
        loss_mask = F.pad(loss_mask, (0, pad), value=0)
        if not context_pre_sharded:
            hidden_states_list = [F.pad(h, (0, 0, 0, pad), value=0.0) for h in hidden_states_list]
        if last_hidden_states is not None:
            last_hidden_states = F.pad(last_hidden_states, (0, 0, 0, pad), value=0.0)
        if ctx_doc_ids is not None:
            ctx_doc_ids = F.pad(ctx_doc_ids, (0, pad), value=-1)
        if base_position_ids is not None:
            base_position_ids = F.pad(base_position_ids, (0, pad), value=0)
        return (
            input_ids,
            loss_mask,
            hidden_states_list,
            last_hidden_states,
            ctx_doc_ids,
            base_position_ids,
        )

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        anchor_eligibility: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchor positions per sample; returns (anchors, keep_mask).

        Always returns exactly ``self.num_anchors`` anchor slots so that
        ``Q_LEN = num_anchors * block_size`` is constant across steps,
        preventing FlexAttention recompilation from shape changes.  Samples
        with fewer valid positions use ``block_keep_mask=False`` for the
        excess slots (those blocks are skipped by the block-sparse kernel).

        Args:
            seq_len: sequence length
            loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
            device: torch device
            ctx_doc_ids: [B, seq_len] long — per-token document id (padding=-1)
                for sequence packing. When provided, an anchor ``a`` is eligible
                only if its whole block ``[a, a+block_size)`` stays inside a single
                document (``ctx_doc_ids[a] == ctx_doc_ids[a+block_size-1] >= 0``),
                so blocks never straddle a packed doc boundary. When None (legacy
                pad-to-longest path), only ``loss_mask`` gates eligibility.
            injected_anchors: [B, num_anchors] — if provided, bypass random
                sampling and use these anchors verbatim (for deterministic tests).
            injected_keep_mask: [B, num_anchors] bool — validity mask paired with
                ``injected_anchors``. If None while ``injected_anchors`` is given,
                all slots are treated as valid.

        Returns:
            anchors: [B, num_anchors] — sampled anchor positions (sorted)
            keep_mask: [B, num_anchors] — True for valid sampled anchors
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        # The last slot's label, at a + bs - 1 + label shift, must exist.
        span = bs + self._label_shift
        max_anchor = max(seq_len - span, 0)
        max_n = self.num_anchors
        if anchor_eligibility is not None and anchor_eligibility.shape != loss_mask.shape:
            raise ValueError("Anchor eligibility must match loss_mask [B, L]")

        # Deterministic injection path: use given anchors as-is.
        if injected_anchors is not None:
            anchors = injected_anchors.to(device=device, dtype=torch.long)
            if injected_keep_mask is not None:
                keep_mask = injected_keep_mask.to(device=device, dtype=torch.bool)
            else:
                keep_mask = torch.ones(bsz, max_n, dtype=torch.bool, device=device)
            anchors = torch.where(keep_mask, anchors, 0)
            if anchor_eligibility is not None:
                in_bounds = (anchors >= 0) & (anchors < seq_len)
                eligible = torch.gather(anchor_eligibility, 1, anchors.clamp(0, seq_len - 1))
                if bool((keep_mask & ~(in_bounds & eligible)).any()):
                    raise ValueError(
                        "DFlash MRoPE blocks require consecutive text positions without visual tokens"
                    )
            return anchors, keep_mask

        if max_anchor == 0:
            logger.warning(
                f"Sequence too short for anchor sampling (seq_len={seq_len}, "
                f"block_size={bs}). Returning dummy anchors so loss is zero."
            )
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        if anchor_eligibility is not None:
            valid = valid & anchor_eligibility[:, : max_anchor + 1]

        # Sequence packing: an anchor's block and labels [a, a+span) must not straddle
        # a doc boundary. Documents are contiguous, so requiring the first and last
        # token to share a non-padding doc id is sufficient.
        if ctx_doc_ids is not None:
            doc = ctx_doc_ids.to(device=device)
            head_doc = doc[:, : max_anchor + 1]
            tail_doc = doc[:, span - 1 : span - 1 + max_anchor + 1]
            same_doc = (head_doc == tail_doc) & (head_doc >= 0)
            valid = valid & same_doc

        valid_counts = valid.sum(dim=1)

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, seq_len + 1)

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        # Take up to num_anchors slots; pad with zeros if fewer valid positions
        take_n = min(max_n, gathered.shape[1])
        selected = gathered[:, :take_n].sort(dim=1).values
        if take_n < max_n:
            pad = torch.zeros(bsz, max_n - take_n, dtype=torch.long, device=device)
            selected = torch.cat([selected, pad], dim=1)
        anchors = selected

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, 0)

        return anchors, keep_mask

    def _mrope_anchor_mask(self, input_ids, position_ids):
        if not self.uses_mrope:
            return None
        position_ids = validate_mrope_positions(
            position_ids, input_ids.shape[0], input_ids.shape[-1]
        )
        from angelspec.data.positions import position_recipe_from_config

        recipe = position_recipe_from_config(self.draft_model.config)
        visual_ids = ()
        if recipe.provider == "hy_v4_flash":
            visual_ids = tuple(
                value for name, value in vars(recipe).items() if name.endswith("_token_id")
            )
        return mrope_text_block_mask(
            position_ids,
            self.block_size,
            input_ids=input_ids,
            visual_token_ids=visual_ids,
        )

    def _create_position_ids(
        self,
        anchor_positions: torch.Tensor,
        seq_len: int,
        base_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position IDs for context and draft tokens.

        Args:
            anchor_positions: [B, n_blocks] anchor start indices.
            seq_len: context sequence length.
            base_position_ids: [B, seq_len] long — doc-local context positions
                (reset to 0 at each doc boundary) for sequence packing. When
                None (legacy path), context positions are the global
                ``arange(seq_len)`` and draft positions are ``anchor + offset``.
                When provided, context positions are taken verbatim and draft
                positions are ``base_position_ids[anchor] + offset`` so RoPE is
                doc-aware (each packed document starts at position 0).
                ``[B, 3, seq_len]`` (M-RoPE t/h/w rows) yields ``[B, 3, *]`` outputs.
        """
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)

        if base_position_ids is not None:
            base = base_position_ids.to(device=device, dtype=torch.long)
            context_position_ids = base
            if base.dim() == 3:
                return context_position_ids, draft_block_positions(
                    anchor_positions, base, self.block_size
                )
            # Doc-local base position at each anchor, then add within-block offset.
            anchor_base = torch.gather(base, 1, anchor_positions)  # [B, n_blocks]
            draft_position_ids = anchor_base.unsqueeze(-1) + offsets
            draft_position_ids = draft_position_ids.view(bsz, -1)
            return context_position_ids, draft_position_ids

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        draft_position_ids = draft_position_ids.view(bsz, -1)

        return context_position_ids, draft_position_ids

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise embeddings: anchor token at block starts, MASK elsewhere."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )

        return self.draft_model.embed_tokens(noise_ids)

    def _add_mtp_fusion(
        self,
        noise_embedding: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
        last_hidden_states: Optional[torch.Tensor],
        target_norm: Optional[nn.Module],
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The block input with the drafter's MTP fusion added (``DFlashConfig.mtp_fusion``).

        Fuses each anchor's embedding (its block's slot 0) with the target's final hidden
        at the anchor's predecessor, the last accepted token, put through the target's
        final norm as vLLM hands it to the MTP layer. A block whose anchor has no
        predecessor in its own document gets nothing. ``prev_hidden`` (``[B, n_blocks,
        D]``) hands over those rows directly, in place of reading them out of the
        whole-sequence ``last_hidden_states``. Leaves the gates' RMS and the injection's
        size, relative to the input it lands on in the slot ``gate`` acts on, in
        ``_mtp_fusion_stats``.
        """
        self._mtp_fusion_stats = {}
        fusion = getattr(self.draft_model, "mtp_fusion", None)
        if fusion is None:
            return noise_embedding
        if (last_hidden_states is None and prev_hidden is None) or target_norm is None:
            missing = "the target norm" if target_norm is None else "last_hidden_states"
            raise ValueError(
                "mtp_fusion reads the target's final hidden at each anchor's predecessor "
                f"through the target's final norm, but got no {missing}; set "
                "inference.store_last_hidden_states=true"
            )
        bsz, n_blocks = anchor_positions.shape
        if prev_hidden is None:
            seq_len, dim = last_hidden_states.shape[1], last_hidden_states.shape[2]
            anchors = anchor_positions.clamp(0, seq_len - 1)
        else:
            anchors = anchor_positions
        prev = (anchors - 1).clamp(min=0)
        valid = block_keep_mask & (anchor_positions > 0)
        if ctx_doc_ids is not None:
            valid = valid & (ctx_doc_ids.gather(1, prev) == ctx_doc_ids.gather(1, anchors))
        with torch.no_grad():
            if prev_hidden is None:
                prev_hidden = torch.gather(
                    last_hidden_states, 1, prev.unsqueeze(-1).expand(-1, -1, dim)
                )
            prev_hidden = target_norm(prev_hidden)
        noise = noise_embedding.view(bsz, n_blocks, self.block_size, -1)
        delta = fusion(noise[:, :, 0], prev_hidden, self.block_size)
        delta = delta * valid[..., None, None].to(delta.dtype)

        with torch.no_grad():
            keep = valid.unsqueeze(-1).float()
            slot = fusion.gate_slot
            injected = (delta[:, :, slot].float().pow(2) * keep).sum()
            base = (noise[:, :, slot].float().pow(2) * keep).sum()
            gates = [g for g in (fusion.gate, fusion.mask_gate) if g is not None]
            gate_rms = torch.cat(gates).detach().float().pow(2).mean().sqrt()
            self._mtp_fusion_stats = {
                "mtp_fusion_gate": gate_rms,
                "mtp_fusion_ratio": (injected / base.clamp(min=1e-12)).sqrt(),
            }
            if not self._mtp_fusion_logged and bool(valid.any()):
                self._mtp_fusion_logged = True
                if not dist.is_initialized() or dist.get_rank() == 0:
                    fused = fusion.fuse(noise[:, :, 0], prev_hidden)[valid]
                    anchor_rms = noise[:, :, 0][valid].float().pow(2).mean().sqrt()
                    logger.info(
                        f"[mtp fusion] {self.draft_model.config.mtp_fusion}: "
                        f"{int(valid.sum())}/{valid.numel()} blocks fused; RMS of the fusion "
                        f"{fused.float().pow(2).mean().sqrt():.3f}, of the anchor embeddings "
                        f"{anchor_rms:.3f}, of the gate {gate_rms:.3g}"
                    )
        return (noise + delta).view_as(noise_embedding)

    def _with_components(self, out: tuple, components: dict) -> tuple:
        """``out`` (a forward result) with ``components`` merged into its loss_components,
        and the attentive correction's, which only exist once the logits are computed."""
        out[5].update(components)
        out[5].update(self._acorr_components())
        out[5].update(self._index_components())
        return out

    def _index_components(self) -> dict:
        """Candidate-set sizes from the last pass (set in ``_target_index_candidates``):
        ``index_union_size`` for ``target_index_union_layers``."""
        totals = {"index_union_size": self._index_union_size}
        if all(total is None for total in totals.values()):
            return {}
        # Which of them a run has is a config fact, identical on every rank, but the key
        # set handed over is the full one either way: the trainer all_reduces it as a
        # block and drops what no rank counted.
        present = next(total for total in totals.values() if total is not None)
        zero = torch.zeros((), device=present[0].device)
        return {
            "_index": {
                key: totals[key] if totals[key] is not None else (zero, zero)
                for key in INDEX_METRIC_KEYS
            }
        }

    def _acorr_components(self) -> dict:
        """``acorr_out_ratio``: the attentive correction's output size relative to the
        hidden it lands on, over the slots it read context for in this pass; with a gated
        block read also ``acorr_gate_s<slot>``, each predicted slot's mean absolute gate;
        with its own kv_b (``attentive_correction_block_kv_b``) also
        ``acorr_block_kv_b_drift``, how far that kv_b is from kv_b_proj, relative to it."""
        if not self._acorr_sums:
            return {}
        attention = sum(stats["attention"] for stats in self._acorr_sums.values())
        hidden = sum(stats["hidden"] for stats in self._acorr_sums.values())
        out = {"acorr_out_ratio": (attention / hidden.clamp(min=1e-12)).sqrt()}
        gated = [stats for stats in self._acorr_sums.values() if "block_gate" in stats]
        if gated:
            gate = sum(stats["block_gate"] for stats in gated)
            count = sum(stats["block_gate_count"] for stats in gated).clamp(min=1.0)
            for slot in range(1, gate.numel()):
                out[f"acorr_gate_s{slot}"] = gate[slot] / count[slot]
        drift = [stats["block_kv_b_drift"] for stats in gated if "block_kv_b_drift" in stats]
        if drift:
            out["acorr_block_kv_b_drift"] = drift[0]
        return out

    def _acorr_context_for(self, rows: torch.Tensor, block_size: int) -> Optional[dict]:
        """The attentive correction's context for draft rows ``rows`` (any shape, the same
        for every batch row; flattened in order they are whole blocks of ``block_size``
        slots from slot 0, which the block keys read); ``None`` without the correction or
        with it off at eval."""
        if not self._acorr or (self.acorr_eval_off and not self.training):
            return None
        context = self._acorr_context
        if context is None:
            raise RuntimeError("the attentive correction's context is set by _draft_backbone")
        flat = rows.reshape(-1)
        positions = context["position_ids"]
        out = {
            "latent": context["latent"],
            "position_ids": positions[..., flat].reshape(*positions.shape[:-1], *rows.shape),
        }
        indices = context["indices"]
        out["indices"] = indices[:, flat].reshape(indices.shape[0], *rows.shape, indices.shape[-1])
        if self._acorr_block_keys:
            out["block_size"] = int(block_size)
        return out

    def _draft_backbone(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        context_pre_sharded: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.Tensor] = None,
        target_norm: Optional[nn.Module] = None,
        fusion_prev_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Shared DFlash backbone (steps 1-6): context features → anchor
        sampling → noise embedding → position ids → block-causal mask → draft
        forward. ``DFlashModel.forward`` and DSpark/DFly subclasses build the
        draft hidden states this way; only the label/loss tail differs.

        Doc-aware (``ctx_doc_ids`` / ``base_position_ids``) and anchor-injection
        args are threaded through for packing and parity tests. The target's final
        hidden states and norm feed the drafter's MTP fusion, if it has one; under USP,
        where a rank holds no whole sequence, ``fusion_prev_hidden`` carries the final
        hidden at each of this rank's anchors' predecessors instead.

        Returns:
            draft_hidden: [B, n_blocks*block_size, D] pre-loss draft hidden states
            anchor_positions: [B, n_blocks] sampled anchor positions
            block_keep_mask: [B, n_blocks] bool validity of each anchor slot
            n_blocks: number of anchor slots (== num_anchors // sp_size under USP)
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        if self.uses_mrope:
            base_position_ids = validate_mrope_positions(position_ids, bsz, seq_len)
        # Under USP the sequence must already be padded to a multiple of sp_size
        # (see ``_usp_pad``). We keep full input_ids / loss_mask / doc ids for
        # anchor sampling, labels, and the block mask; only the heavy context
        # hidden states are sliced to this rank's shard.
        full_ctx_len = None
        context_hidden_list = hidden_states_list
        local_base_position_ids = base_position_ids
        usp_global_offset = 0

        if self.sp_size > 1:
            if seq_len % self.sp_size != 0:
                raise ValueError(
                    f"USP requires seq_len divisible by sp_size "
                    f"(got seq_len={seq_len}, sp_size={self.sp_size}); call _usp_pad first"
                )
            full_ctx_len = seq_len
            shard = seq_len // self.sp_size
            usp_global_offset = self.sp_rank * shard
            sl = slice(usp_global_offset, usp_global_offset + shard)
            if context_pre_sharded:
                context_hidden_list = hidden_states_list
                expected = shard
                actual = context_hidden_list[0].shape[1] if context_hidden_list else 0
                if actual != expected:
                    raise ValueError(
                        f"Pre-sharded USP context length mismatch: got {actual}, expected {expected}"
                    )
            else:
                context_hidden_list = [h[:, sl] for h in hidden_states_list]
            if base_position_ids is not None:
                local_base_position_ids = base_position_ids[..., sl]

        # A drafter reusing the target's latents has no context projection to run.
        context_feature = (
            None
            if self.target_kv_layers is not None
            else self.draft_model.extract_context_feature(context_hidden_list)
        )

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            ctx_doc_ids=ctx_doc_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
            anchor_eligibility=self._mrope_anchor_mask(input_ids, position_ids),
        )

        if self.sp_size > 1:
            # All ranks sample (keeping RNG consumption uniform); rank 0's
            # result wins via broadcast. Skip broadcast when test-injected.
            # Then stride anchors across SP (compute shard) while context is
            # seq-sharded (memory shard) and gathered inside attention.
            if injected_anchors is None:
                sp_src = dist.get_global_rank(self.sp_group, 0)
                dist.broadcast(anchor_positions, src=sp_src, group=self.sp_group)
                bkm_u8 = block_keep_mask.to(torch.uint8)
                dist.broadcast(bkm_u8, src=sp_src, group=self.sp_group)
                block_keep_mask = bkm_u8.bool()
            anchor_positions = anchor_positions[:, self.sp_rank :: self.sp_size].contiguous()
            block_keep_mask = block_keep_mask[:, self.sp_rank :: self.sp_size].contiguous()

        n_blocks = anchor_positions.shape[1]

        if self.sp_size > 1 and not self._usp_runtime_logged:
            self._usp_runtime_logged = True
            # Rank 0 only: one clear proof line; other ranks would only repeat
            # the same ratios with a different sp_rank offset.
            if dist.get_rank() == 0:
                local_ctx = int(full_ctx_len) // self.sp_size
                n_ctx_layers = len(context_hidden_list)
                hidden_dim = int(context_hidden_list[0].shape[-1]) if context_hidden_list else 0
                elem_bytes = context_hidden_list[0].element_size() if context_hidden_list else 2
                # Approximate resident target-hidden footprint before vs after shard.
                full_bytes = n_ctx_layers * bsz * int(full_ctx_len) * hidden_dim * elem_bytes
                local_bytes = n_ctx_layers * bsz * local_ctx * hidden_dim * elem_bytes
                mem_msg = ""
                if torch.cuda.is_available() and input_ids.is_cuda:
                    alloc_gb = torch.cuda.memory_allocated(input_ids.device) / (1024**3)
                    peak_gb = torch.cuda.max_memory_allocated(input_ids.device) / (1024**3)
                    mem_msg = f" cuda_alloc={alloc_gb:.2f}GiB cuda_peak={peak_gb:.2f}GiB"
                logger.info(
                    f"[USP] runtime shard active: sp_rank={self.sp_rank}/{self.sp_size} "
                    f"full_ctx={full_ctx_len} local_ctx={local_ctx} "
                    f"n_blocks={n_blocks} (global_anchors~{n_blocks * self.sp_size}) "
                    f"context_feature="
                    f"{None if context_feature is None else tuple(context_feature.shape)} "
                    f"target_hidden~{local_bytes / (1024**2):.1f}MiB/rank "
                    f"(vs ~{full_bytes / (1024**2):.1f}MiB if unsharded, "
                    f"~{self.sp_size}x resident save){mem_msg}"
                )

        # Noise embed gathers from FULL input_ids at global anchor positions.
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)
        if (
            self.sp_size > 1
            and fusion_prev_hidden is None
            and getattr(self.draft_model, "mtp_fusion", None) is not None
            and (last_hidden_states is None or last_hidden_states.shape[1] != seq_len)
        ):
            raise ValueError(
                "under USP the MTP fusion reads the target's final hidden at this rank's "
                "anchors' predecessors: pass teacher_hidden_at_labels, or the whole-sequence "
                "last_hidden_states"
            )
        noise_embedding = self._add_mtp_fusion(
            noise_embedding,
            anchor_positions,
            block_keep_mask,
            ctx_doc_ids,
            last_hidden_states,
            target_norm,
            prev_hidden=fusion_prev_hidden,
        )

        _, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len, base_position_ids=base_position_ids
        )
        if self.sp_size > 1:
            shard = seq_len // self.sp_size
            if local_base_position_ids is not None:
                context_position_ids = local_base_position_ids
            else:
                context_position_ids = (
                    torch.arange(
                        usp_global_offset,
                        usp_global_offset + shard,
                        device=device,
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
        else:
            context_position_ids, _ = self._create_position_ids(
                anchor_positions, seq_len, base_position_ids=base_position_ids
            )

        draft_len = n_blocks * self.block_size
        # Block mask / attention see the FULL context length under USP (Ulysses
        # all-to-all reconstructs it); local shard length only feeds projections.
        mask_ctx_len = full_ctx_len if full_ctx_len is not None else seq_len
        kv_len = mask_ctx_len + draft_len

        # Per draft layer; layers reading the same target layer share one tensor, so
        # each distinct candidate set gets one block mask / index list below.
        ctx_candidates, ctx_candidate_indices = (
            self._target_index_candidates(
                input_ids,
                context_hidden_list,
                anchor_positions,
                block_keep_mask,
                ctx_doc_ids,
                position_ids=base_position_ids if self.uses_mrope else None,
            )
            if self.target_index_layer is not None
            else (None, None)
        )
        block_mask = sparse_kv_indices = None
        if self._sparse_kv_attention:
            alignment = self._sparse_kv_alignment

            def build_sparse_kv_indices(candidates, window):
                # An indexer union, or GLM-5.3's top-k plus its unpooled tail, can exceed topk.
                slots = (
                    (
                        max(self.target_index_topk or 0, candidates.shape[-1])
                        if candidates is not None
                        else 0
                    )
                    + (window or 0)
                    + self.block_size
                )
                return _create_dflash_sparse_kv_indices(
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    ctx_len=mask_ctx_len,
                    block_size=self.block_size,
                    ctx_candidates=candidates,
                    width=-(-slots // alignment) * alignment,
                    ctx_doc_ids=ctx_doc_ids,
                    causal=self.causal_attn,
                    context_local_window=window,
                    two_segment=self._two_segment_kv,
                )

            sparse_kv_indices = _build_per_layer(
                ctx_candidate_indices, self._context_windows, build_sparse_kv_indices
            )
        else:

            def build_block_mask(candidates, window):
                mask_mod = _create_dflash_mask_mod(
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    ctx_len=mask_ctx_len,
                    block_size=self.block_size,
                    ctx_doc_ids=ctx_doc_ids,
                    causal=self.causal_attn,
                    context_local_window=window,
                    ctx_candidates=candidates,
                )
                return compile_friendly_create_block_mask(
                    mask_mod=mask_mod,
                    B=bsz,
                    H=None,
                    Q_LEN=draft_len,
                    KV_LEN=kv_len,
                    device=device,
                    _compile=device.type == "cuda",
                )

            block_mask = _build_per_layer(ctx_candidates, self._context_windows, build_block_mask)

        # Index lists go to DFlashDraftModel (DSpark's too) and to drafters that declare
        # sparse_kv_attention; only DFlashDraftModel takes target latents.
        draft_kwargs = {}
        if sparse_kv_indices is not None:
            draft_kwargs["sparse_kv_indices"] = sparse_kv_indices
        probe = None
        if self.target_kv_layers is not None:
            draft_kwargs["context_latents"] = self._target_kv_context(
                input_ids,
                context_hidden_list,
                ctx_doc_ids,
                position_ids=base_position_ids if self.uses_mrope else None,
            )
            probe = self._attn_probe(
                input_ids,
                hidden_states_list,
                loss_mask,
                anchor_positions,
                block_keep_mask,
                ctx_doc_ids,
            )
        with probe.hooks() if probe is not None else nullcontext():
            draft_hidden = self.draft_model(
                draft_input_ids=None,
                context_feature=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
                noise_embedding=noise_embedding,
                full_ctx_len=full_ctx_len,
                **draft_kwargs,
            )
        self._attn_diag_stats = probe.stats() if probe is not None else {}

        self._acorr_sums = {}
        self._acorr_context = None
        if self._acorr_source is not None:
            latent = draft_kwargs["context_latents"][self._acorr_source]
            keys = (
                sparse_kv_indices[self._acorr_source]
                if isinstance(sparse_kv_indices, list)
                else sparse_kv_indices
            )
            # The source layer's keys minus its block's own (those latents are the
            # drafter's, not the target's).
            in_context = (keys >= 0) & (keys < latent.shape[1])
            self._acorr_context = {
                "latent": latent,
                "indices": torch.where(in_context, keys, torch.full_like(keys, -1)),
                "position_ids": draft_position_ids,
            }

        return draft_hidden, anchor_positions, block_keep_mask, n_blocks

    @staticmethod
    def _compute_l1_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """L1 distribution-distillation loss (DSpark ``l1_per_token``).

        Per-position L1 distance ``Σ_i |softmax(student)_i - softmax(teacher)_i|``
        between the full-vocab next-token distributions, which equals ``2·TV``.
        Returns [N].
        """
        tv, _ = lk_tv_kl_per_pos(student_logits, teacher_logits, form="tv")
        return 2.0 * tv

    def _compute_topk_kl_loss_variant_b(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        topk: int = 10,
    ) -> torch.Tensor:
        """Top-K KL divergence (Variant B) for DFlash distillation. Returns [N]."""
        return _kl_variant_b(student_logits, teacher_logits, topk)

    def _compute_lk_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        loss_type: str = "hybrid",
        eta: float = 3.0,
    ) -> torch.Tensor:
        """LK (acceptance-rate) distillation loss for DFlash.

          * ``loss_type="alpha"``  →  -log( sum_i min(p_i, q_i) )
          * ``loss_type="hybrid"`` →  lambda*KL(p||q) + (1-lambda)*TV(p, q),
                lambda = exp(-eta * sg[alpha]), alpha = sum_i min(p_i, q_i).

        ``p`` = teacher (detached), ``q`` = student, both full-vocab. Returns [N]
        per-position LK loss.
        """
        if loss_type == "alpha":
            # alpha = Σ_i min(p_i, q_i) == 1 − TV(p, q); reuse the shared TV term.
            tv, _ = lk_tv_kl_per_pos(student_logits, teacher_logits, form="tv")
            alpha = (1.0 - tv).clamp_min(1e-10)  # [N]
            return -torch.log(alpha)

        if loss_type == "hybrid":
            ell, _tv = lk_tv_kl_per_pos(student_logits, teacher_logits, form="lk", eta=eta)
            return ell

        raise ValueError(f"Unknown lk_loss_type={loss_type!r}; expected 'alpha' or 'hybrid'.")

    def _compute_e2e_tv_loss(
        self,
        student_logits_pb: torch.Tensor,
        teacher_logits_pb: torch.Tensor,
        valid_mask_pb: torch.Tensor,
        vocab_chunk: Optional[int] = None,
    ):
        """End-to-end multi-step TV loss (γ-step accepted-length objective)::

            α_i     = 1 - TV(p_i, q_i) = Σ_v min(p_i,v, q_i,v)  ∈ (0, 1]
            L_e2e   = 1 - (1/γ) * Σ_{j=1..γ}  Π_{i=1..j} α_i

        γ = block_size. The prefix product couples steps inside a block, giving
        intrinsic per-step weighting, so this term ignores decay / flat_weights.
        Inputs are ``[B, n_blocks, block_size, V]`` logits and a
        ``[B, n_blocks, block_size]`` validity mask (T=1). Returns
        unnormalized ``(loss_sum, accept_length_sum, block_count)``. The caller
        applies the local or global denominator.

        Under USP the loss divides by the SP-global valid-block count (the same
        contract the CE term follows), so the later gradient SUM reconstructs the
        full-sequence mean. ``accept_length`` stays on the local count to match how
        the trainer reduces logged components.

        ``vocab_chunk`` controls the streaming-softmax chunk size along the
        vocab dimension (see the memory-optimized implementation below).
        ``None`` (default) resolves in order:
          1. ``ANGELSPEC_DFLASH_E2E_TV_VOCAB_CHUNK`` env var, if set to > 0.
          2. Module default ``_DFLASH_E2E_TV_VOCAB_CHUNK_DEFAULT`` (8192).
        Smaller values reduce peak GPU memory at the cost of more kernel
        launches; useful when large-V + long-seq training OOMs here.
        """
        # fp32 for the O(V) min-sum and the chain of up-to-γ products.
        # Memory-optimized: compute α = Σ_v min(softmax(t), softmax(s)) in a
        # vocab-chunked streaming manner so at most one chunk's fp32 tensor is
        # materialized at a time (peak memory ≈ chunk_size fp32 slices instead
        # of a full fp32 softmax over the vocab dimension). ``t_lse`` / ``s_lse``
        # are the log-sum-exp normalizers, so each softmax slice is computed as
        # ``exp(logits_slice - lse)`` without ever forming the full-vocab
        # softmax. Gradient flows through the student branch of min.

        # Teacher lse: cast to fp32 for the reduction, drop the fp32 copy.
        _t_logits_f32_tmp = teacher_logits_pb.float()
        t_lse = torch.logsumexp(_t_logits_f32_tmp, dim=-1, keepdim=True)  # [B, nb, γ, 1]
        del _t_logits_f32_tmp

        # Resolve chunk size: explicit arg > env var > module default.
        if vocab_chunk is None or vocab_chunk <= 0:
            env_chunk = _dflash_e2e_tv_vocab_chunk()
            vocab_chunk = env_chunk if env_chunk > 0 else _DFLASH_E2E_TV_VOCAB_CHUNK_DEFAULT

        # Compute α = Σ_v min(softmax(t), softmax(s)) via vocab-chunked
        # streaming. Materializes at most one chunk's fp32 tensor at a time.
        s_logits_f32 = student_logits_pb.float()
        s_lse = torch.logsumexp(s_logits_f32, dim=-1, keepdim=True)  # [B, nb, γ, 1]
        vocab_size_local = teacher_logits_pb.size(-1)
        alpha = torch.zeros(
            s_logits_f32.shape[:-1], dtype=torch.float32, device=s_logits_f32.device
        )  # [B, nb, γ]
        for start_ in range(0, vocab_size_local, vocab_chunk):
            end_ = min(start_ + vocab_chunk, vocab_size_local)
            # softmax slices via exp(logits - lse); grad flows through s.
            t_slice = torch.exp(teacher_logits_pb[..., start_:end_].float() - t_lse)
            s_slice = torch.exp(s_logits_f32[..., start_:end_] - s_lse)
            alpha = alpha + torch.minimum(t_slice, s_slice).sum(dim=-1)
            del t_slice, s_slice
        # α ∈ (0, 1]; gradient flows through the student branch of min.

        # Set α:=1 on invalid slots so cumprod treats them as identity.
        m = valid_mask_pb.float()
        alpha_effective = alpha * m + (1.0 - m)
        prefix_prod = torch.cumprod(alpha_effective, dim=-1)  # [B, nb, γ]

        gamma_valid = m.sum(dim=-1).clamp(min=1.0)  # [B, nb]
        accept_length_pb = (prefix_prod * m).sum(dim=-1)  # [B, nb]
        e2e_per_block = 1.0 - accept_length_pb / gamma_valid

        block_has_valid = (m.sum(dim=-1) > 0).float()
        local_denom = block_has_valid.sum().clamp(min=1.0)

        # Same global-count contract as the CE term: divide the LOCAL numerator by
        # the SP-GLOBAL valid-block count so the gradient SUM over SP reconstructs
        # the full-sequence mean. Dividing by the local count inflates the term by
        # ~sp_size, and because the per-rank valid-block counts differ (anchors are
        # strided, then masked by bounds / loss_mask), it also reweights each block
        # by N_global/N_local on top of that — a per-step distortion, not a constant.
        # Unconditional within this branch: every rank enters it together (the
        # weight and last_hidden_states are batch-uniform), so the scalar
        # all-reduce stays symmetric and cannot deadlock.
        if self.sp_size > 1:
            global_blocks = block_has_valid.sum().detach().clone()
            dist.all_reduce(global_blocks, op=dist.ReduceOp.SUM, group=self.sp_group)
            loss_denom = global_blocks.clamp(min=1.0)
        else:
            loss_denom = local_denom
        e2e_tv_loss = (e2e_per_block * block_has_valid).sum() / loss_denom

        with torch.no_grad():
            # The metric keeps the LOCAL denominator: the trainer reduces logged
            # components with a plain mean over ranks, which reads a local mean
            # correctly but would show a global-count fraction at 1/sp_size.
            accept_length = (accept_length_pb * block_has_valid).sum() / local_denom

        return e2e_tv_loss, accept_length.detach()

    # ------------------------------------------------------------------
    # Path selector (drafter head; see models/draft/path_selector.py)
    # ------------------------------------------------------------------

    def _path_selector(self):
        """The drafter's path selector, or ``None`` when the config disables it."""
        return getattr(self.draft_model, "path_selector", None)

    def _path_selector_sums(
        self,
        *,
        unary_logits: torch.Tensor,  # [B, N, V]
        hidden: torch.Tensor,  # [B, N, D]
        prev_token_ids: torch.Tensor,  # [B, N]
        label_ids: torch.Tensor,  # [B, N]
        objective_weights: torch.Tensor,  # [B, N]
        binary_mask: torch.Tensor,  # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Un-normalized selector CE and diagnostic sums over one set of positions.

        ``label_ids`` is the token the selector must rank first — the corpus token,
        or the target model's argmax when that is what the unary head is trained
        against (the caller decides; see the selector block in ``forward``). The
        loss and both metrics read the same labels, so ``acc_sum`` always measures
        agreement with whatever acceptance is tested against.

        Everything here is position-local (each position reranks its own unary
        top-k under its own teacher-forced predecessor), so the caller is free to
        pass the whole batch or one chunk of blocks and normalize afterwards —
        both give identical results up to summation order.

        Returns ``(ce_weighted_sum, recall_sum, acc_sum, serving_count)``:

          * ``ce_weighted_sum``: objective-weighted K-way CE, the trained term.
          * ``recall_sum``: valid positions whose label was in the unary top-k;
            the selector's ceiling, since it can only rerank that set.
          * ``acc_sum`` / ``serving_count``: teacher-forced selector hits over
            exactly those recoverable positions (injected labels excluded, so
            the ratio is not inflated by labels serving could never pick).
        """
        selector = self._path_selector()
        candidate_ids = unary_logits.topk(selector.top_k, dim=-1).indices
        training_candidate_ids, target_positions, contains_target = selector_training_candidates(
            candidate_ids, label_ids
        )
        candidate_logits = selector.score_candidates(
            unary_logits, hidden, prev_token_ids, training_candidate_ids
        )

        # fp32 for the same reason as the unary CE: bf16 rounding flips
        # small-margin rankings, and here the whole term IS a ranking decision.
        ce = F.cross_entropy(
            candidate_logits.flatten(0, -2).float(),
            target_positions.flatten(),
            reduction="none",
        ).view_as(target_positions)
        ce_weighted_sum = (ce * objective_weights.float()).sum()

        with torch.no_grad():
            valid = binary_mask.float()
            recall_sum = (contains_target.float() * valid).sum()
            picked_ids = training_candidate_ids.gather(
                -1, candidate_logits.detach().argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
            serving_valid = valid * contains_target.float()
            acc_sum = (picked_ids.eq(label_ids).float() * serving_valid).sum()
            serving_count = serving_valid.sum()

        return ce_weighted_sum, recall_sum, acc_sum, serving_count

    def _greedy_draft_ids(
        self,
        unary_logits: torch.Tensor,
        hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Token the serving path would emit at each position, teacher-forced.

        Without a path selector this is the unary argmax. With one it is the
        selector's rerank of the unary top-k, conditioned on ``prev_token_ids``
        (the same teacher-forced predecessor ``forward`` already builds). Logged
        ``acc_per_position`` / ``teacher_acc_per_position`` — and therefore the
        progress-bar ``acc_len`` / ``acc_len_tgt`` — must use this, otherwise
        they score a head the serving path does not emit.
        """
        selector = self._path_selector()
        if selector is None:
            return unary_logits.argmax(dim=-1)
        candidate_ids, scores = selector.select(unary_logits, hidden, prev_token_ids)
        return candidate_ids.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)

    def _apply_hidden_correction(
        self, draft_hidden: torch.Tensor, prev_token_ids: torch.Tensor, block_start: int = 0
    ) -> torch.Tensor:
        """TreeFlash hidden-states correction (formula (1)), or a pass-through.

        Applied before the LM head, so the token distribution is conditioned on
        the teacher-forced previous token. Returns ``draft_hidden`` unchanged
        unless the drafter carries a ``hidden_correction`` module
        (``enable_hidden_correction`` in the draft config). ``draft_hidden`` holds
        the blocks from ``block_start`` on, which the attentive correction needs to
        pick its rows of context.
        """
        correction = getattr(self.draft_model, "hidden_correction", None)
        if correction is None:
            return draft_hidden

        bsz = draft_hidden.size(0)
        prev_embeds = self.draft_model.embed_tokens(prev_token_ids)
        prev_embeds = prev_embeds.view(bsz, -1, prev_embeds.size(-1))
        if not self._acorr:
            return correction(draft_hidden, prev_embeds)
        start = block_start * self.block_size
        rows = torch.arange(start, start + draft_hidden.size(1), device=draft_hidden.device)
        stats: dict = {}
        corrected = correction(
            draft_hidden,
            prev_embeds,
            context=self._acorr_context_for(rows, self.block_size),
            attention_stats=stats,
        )
        if stats:
            self._acorr_sums[block_start] = stats
        return corrected

    @torch.no_grad()
    def _prev_top1(self, logits: torch.Tensor, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """``[B, N]``: whether slot k's previous token is the argmax of slot k-1's
        uncorrected ``logits``; False at slots 0 and 1, which have no supervised previous slot."""
        bsz, n_blocks, bs = prev_token_ids.shape
        top1 = torch.zeros_like(prev_token_ids, dtype=torch.bool)
        if bs > 2:
            prev_argmax = logits.view(bsz, n_blocks, bs, -1)[:, :, 1:-1].argmax(dim=-1)
            top1[:, :, 2:] = prev_argmax == prev_token_ids[:, :, 2:]
        return top1.view(bsz, -1)

    # ------------------------------------------------------------------
    # Subclass extension hooks (no-ops for base DFlash). DSpark overrides these
    # to inject the Markov logit bias and the confidence-head loss. Signatures
    # are frozen here so subclasses attach without reworking forward. See
    # models/dspark.py.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _served_draft_chain(
        self,
        draft_hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
        lm_head_weight: torch.Tensor,
        n_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the previous-token inputs and proposals used by serving."""
        unsupported = [
            name
            for name, present in (
                ("attentive_correction", self._acorr),
                ("hidden_correction_topk", bool(self.hidden_correction_topk)),
                ("a path selector", self._path_selector() is not None),
                ("a Markov head", getattr(self.draft_model, "markov_head", None) is not None),
                ("a drafter lm_head", hasattr(self.draft_model, "lm_head")),
            )
            if present
        ]
        if unsupported:
            raise NotImplementedError(
                "dflash_opd_on_policy_prev unrolls a plain hidden correction over the "
                f"target lm_head; this drafter has {', '.join(unsupported)}."
            )
        correction = self.draft_model.hidden_correction
        embed = self.draft_model.embed_tokens
        bsz = draft_hidden.shape[0]
        hidden = draft_hidden.view(bsz, n_blocks, self.block_size, -1)
        prev = prev_token_ids.clone()
        proposals = torch.empty_like(prev)
        for j in range(self.block_size):
            corrected = correction(hidden[:, :, j], embed(prev[:, :, j]))
            proposals[:, :, j] = F.linear(corrected, lm_head_weight).argmax(dim=-1)
            if 1 <= j < self.block_size - 1:
                prev[:, :, j + 1] = proposals[:, :, j]
        return prev, proposals

    def _compute_draft_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
        block_start: int = 0,
    ) -> torch.Tensor:
        """Correct the draft hidden states (if enabled), then project to vocab
        logits through the frozen LM head. DSpark overrides to add the Markov bias.

        ``prev_token_ids`` is ``[B, n_blocks, block_size]`` — the ground-truth
        token preceding each draft slot's target (aligned with ``draft_hidden``),
        which holds ``n_blocks`` blocks from ``block_start`` on.

        With ``hidden_correction_topk`` set, only that many highest uncorrected
        logits per position take their corrected value and the rest keep the
        uncorrected one, which is the distribution serving draws from. The
        selection itself carries no gradient.
        """

        def head(hidden: torch.Tensor) -> torch.Tensor:
            return (
                self.draft_model.lm_head(hidden)
                if hasattr(self.draft_model, "lm_head")
                else F.linear(hidden, lm_head_weight)
            )

        corrected = head(self._apply_hidden_correction(draft_hidden, prev_token_ids, block_start))
        if not self.hidden_correction_topk:
            return corrected
        base = head(draft_hidden)
        self._hc_prev_top1 = self._prev_top1(base, prev_token_ids)
        top = base.detach().topk(self.hidden_correction_topk, dim=-1).indices
        self._hc_topk_candidates = top
        return base.scatter(-1, top, corrected.gather(-1, top))

    @torch.no_grad()
    def _hc_topk_diag(
        self,
        *,
        draft_hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
        lm_head_weight: torch.Tensor,
        teacher_ids_flat: Optional[torch.Tensor],
        binary_eval_mask: torch.Tensor,
        n_blocks: int,
    ) -> dict:
        """Eval-only fidelity of the top-K hidden correction (``dflash_hc_topk_diag``).

        Serving can skip the per-slot full-vocabulary LM head by correcting only the K
        highest uncorrected logits and keeping the rest uncorrected (the distribution
        ``hidden_correction_topk`` trains); on those K the scores equal the full
        correction. The top-K pick is that distribution's argmax, so the K+1-th
        uncorrected logit can still win. Per slot, over valid label positions:

        - ``changes_argmax``: the full correction moves the argmax at all;
        - ``rank_lt<t>``: the full-correction argmax is among the t highest
          uncorrected logits (the top-K pick is exact iff that holds for t = K);
        - ``tf_agree_k<K>``, ``acc_tgt_k<K>`` vs ``acc_tgt_full``: teacher-forced
          (corpus predecessor, as in training) agreement of the top-K pick with the
          full argmax, and each pick's agreement with the target argmax;
        - ``tv09_k<K>``: TV at T=0.9 between the top-K distribution (uncorrected
          tail) and the full one;
        - ``chain_agree_k<K>``, ``chain_tgt_k<K>`` vs ``chain_tgt_full``: walking the
          drafter's own picks from the anchor, as serving does, whether the whole
          prefix up to the slot equals the full-correction walk, and whether it
          equals the target argmax (read along the corpus prefix).

        Logits are recomputed in fp32 so bf16 ties do not count as disagreements.
        Returns ``hc_topk/<metric>/s<slot>`` sums plus ``hc_topk/count/s<slot>``;
        the trainer divides them after summing over batches and ranks.
        """
        correction = self.draft_model.hidden_correction
        embed = self.draft_model.embed_tokens
        weight = lm_head_weight.float()
        topks = self.hc_topk_diag

        bsz, bs = draft_hidden.size(0), self.block_size
        shape = (bsz, n_blocks, bs)
        valid = binary_eval_mask.view(shape) > 0.5
        stats: dict = {}

        def add(name: str, values: torch.Tensor) -> None:
            sums = (values.view(shape).float() * valid).sum(dim=(0, 1))
            for slot in range(1, bs):
                stats[f"hc_topk/{name}/s{slot}"] = sums[slot]

        def head(hidden: torch.Tensor) -> torch.Tensor:
            return F.linear(hidden.float(), weight)

        hidden = draft_hidden.reshape(*shape, -1)
        teacher = teacher_ids_flat.view(shape) if teacher_ids_flat is not None else None
        add("count", torch.ones(shape, device=draft_hidden.device))

        # The attentive correction's context, by draft row (None for other drafters).
        slots = torch.arange(n_blocks * bs, device=draft_hidden.device).view(n_blocks, bs)
        base = head(hidden)
        corrected = head(
            correction(hidden, embed(prev_token_ids), context=self._acorr_context_for(slots, bs))
        )
        full_pick = corrected.argmax(dim=-1)
        add("changes_argmax", full_pick != base.argmax(dim=-1))
        rank = (base > base.gather(-1, full_pick.unsqueeze(-1))).sum(dim=-1)
        for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024):
            add(f"rank_lt{t}", rank < t)
        if teacher is not None:
            add("acc_tgt_full", full_pick == teacher)

        tops = {k: base.topk(k, dim=-1).indices for k in topks}
        base_rows = base.view(-1, base.size(-1))
        corrected_rows = corrected.view(-1, corrected.size(-1))
        for k, top in tops.items():
            top_rows = top.view(-1, k)
            pick = torch.empty(base_rows.size(0), dtype=torch.long, device=base.device)
            tv = torch.empty(base_rows.size(0), device=base.device)
            for start in range(0, base_rows.size(0), 1024):
                sl = slice(start, start + 1024)
                rows_top = top_rows[sl]
                hybrid = base_rows[sl].scatter(
                    -1, rows_top, corrected_rows[sl].gather(-1, rows_top)
                )
                pick[sl] = hybrid.argmax(dim=-1)
                p = torch.softmax(hybrid / 0.9, dim=-1)
                q = torch.softmax(corrected_rows[sl] / 0.9, dim=-1)
                tv[sl] = 0.5 * (p - q).abs().sum(dim=-1)
            pick = pick.view(shape)
            add(f"tf_agree_k{k}", pick == full_pick)
            if teacher is not None:
                add(f"acc_tgt_k{k}", pick == teacher)
            add(f"tv09_k{k}", tv)
        del corrected, corrected_rows

        anchor = prev_token_ids[..., 0]
        full_prev, full_ok = anchor, torch.ones_like(anchor, dtype=torch.bool)
        chain_tgt_full = torch.zeros(shape, dtype=torch.bool, device=anchor.device)
        prev = {k: anchor for k in topks}
        same = {k: torch.ones_like(full_ok) for k in topks}
        ok = {k: torch.ones_like(full_ok) for k in topks}
        chain_agree = {k: torch.zeros_like(chain_tgt_full) for k in topks}
        chain_tgt = {k: torch.zeros_like(chain_tgt_full) for k in topks}
        for slot in range(1, bs):
            h = hidden[:, :, slot]
            context = self._acorr_context_for(slots[:, slot], 1)
            full_prev = head(correction(h, embed(full_prev), context=context)).argmax(dim=-1)
            if teacher is not None:
                full_ok = full_ok & (full_prev == teacher[:, :, slot])
                chain_tgt_full[:, :, slot] = full_ok
            for k in topks:
                scores = head(correction(h, embed(prev[k]), context=context))
                top = tops[k][:, :, slot]
                pick = base[:, :, slot].scatter(-1, top, scores.gather(-1, top)).argmax(dim=-1)
                same[k] = same[k] & (pick == full_prev)
                chain_agree[k][:, :, slot] = same[k]
                if teacher is not None:
                    ok[k] = ok[k] & (pick == teacher[:, :, slot])
                    chain_tgt[k][:, :, slot] = ok[k]
                prev[k] = pick
        for k in topks:
            add(f"chain_agree_k{k}", chain_agree[k])
            if teacher is not None:
                add(f"chain_tgt_k{k}", chain_tgt[k])
        if teacher is not None:
            add("chain_tgt_full", chain_tgt_full)
        return stats

    def _extra_distill_needed(self) -> bool:
        """Whether a subclass head needs the teacher logits even when KL/LK are
        off. DFlash: no. DSpark confidence head overrides to True."""
        return False

    def _compute_extra_loss(
        self,
        loss: torch.Tensor,
        flat_logits: torch.Tensor,
        teacher_logits_flat: Optional[torch.Tensor],
        flat_weights: torch.Tensor,
        loss_denom: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Add subclass-specific loss terms on top of the DFlash objective.

        ``loss_denom`` is what the other ``flat_weights``-weighted terms divide by: the
        weighted count of this rank's tokens, of the whole SP group's under USP (so the
        gradient SUM over SP rebuilds the full-sequence mean). Returns ``(loss,
        extra_components)``, where ``extra_components`` holds detached scalars merged
        into ``loss_components`` for logging. DFlash adds nothing; DSpark adds e.g.
        ``{"confidence_loss": ...}``."""
        return loss, {}

    @torch.no_grad()
    def propose_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        context_pre_sharded: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.Tensor] = None,
        target_norm: Optional[nn.Module] = None,
    ):
        """On-policy DFlash proposal for OPD packed tree-forward scoring (no grad).

        Runs the block-parallel draft and returns the argmax'd proposals. The
        within-block layout is ``[anchor_slot(0), pred@1, ..., pred@{B-1}]``: slot
        0 is the anchor input, slots 1..B-1 are the draft's proposed tokens for
        absolute positions ``anchor+1 .. anchor+B-1``. The caller drops slot 0 and
        feeds the B-1 proposals as branch tokens to ``build_tree_layout`` (block
        length B-1), then dispatches ``score_packed`` for the teacher distribution.
        ``last_hidden_states`` / ``target_norm`` are only read by an MTP fusion.

        Returns:
            proposals: [B, n_blocks, block_size] long — argmax per within-block slot.
            anchor_positions: [B, n_blocks] long.
            block_keep_mask: [B, n_blocks] bool — valid anchors.
        """
        if self.sample_from_anchor:
            raise NotImplementedError(
                "OPD proposals assume slot 0 is the anchor input, not a prediction"
            )
        if self.uses_mrope:
            position_ids = validate_mrope_positions(
                position_ids, input_ids.shape[0], input_ids.shape[-1]
            )
            position_pad = (-input_ids.shape[-1]) % self.sp_size if self.sp_size > 1 else 0
            position_ids = F.pad(position_ids, (0, position_pad))
        (
            input_ids,
            loss_mask,
            hidden_states_list,
            last_hidden_states,
            ctx_doc_ids,
            base_position_ids,
        ) = self._usp_pad(
            input_ids,
            loss_mask,
            hidden_states_list,
            last_hidden_states=last_hidden_states,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            context_pre_sharded=context_pre_sharded,
        )
        bsz = input_ids.shape[0]
        draft_hidden, anchor_positions, block_keep_mask, n_blocks = self._draft_backbone(
            input_ids,
            hidden_states_list,
            loss_mask,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
            position_ids=position_ids,
            context_pre_sharded=context_pre_sharded,
            last_hidden_states=last_hidden_states,
            target_norm=target_norm,
        )
        logits = (
            self.draft_model.lm_head(draft_hidden)
            if hasattr(self.draft_model, "lm_head")
            else F.linear(draft_hidden, lm_head_weight)
        )
        proposals = logits.view(bsz, n_blocks, self.block_size, -1).argmax(-1)
        return proposals, anchor_positions, block_keep_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
        teacher_hidden_at_labels: Optional[torch.Tensor] = None,
        target_norm: Optional[nn.Module] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        context_pre_sharded: bool = False,
        return_draft: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        opd_on_policy_prev: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full DFlash training forward pass.

        Args:
            input_ids: [B, seq_len] token IDs.
            hidden_states_list: per-target-layer [B, seq_len, D] hidden states.
            loss_mask: [B, seq_len] 1 for supervised positions.
            lm_head_weight: frozen target LM head weight.
            last_hidden_states: [B, seq_len, D] target pre-norm final hidden states.
            teacher_hidden_at_labels: [B, n_blocks, block_size, D] pre-gathered
                teacher rows for this rank's anchors. USP uses this to avoid moving
                the full final-layer sequence to GPU.
            target_norm: target model's final RMSNorm module, applied to
                ``last_hidden_states`` before the LM head so teacher logits match
                inference-time output. None => teacher logits from raw hidden.
            ctx_doc_ids: [B, seq_len] long — per-token document id (padding=-1)
                for sequence packing. When None (default), the whole sequence is
                treated as one document (legacy pad-to-longest path). When given,
                anchor sampling, the block-causal mask, and RoPE positions all
                become doc-aware so packed segments never attend across doc
                boundaries.
            base_position_ids: [B, seq_len] long — doc-local context positions
                (reset to 0 at each doc boundary) used for RoPE when packing.
                Required to be paired with ``ctx_doc_ids``; ignored when
                ``ctx_doc_ids`` is None.
            injected_anchors: [B, num_anchors] — bypass random anchor sampling
                (parity tests only). See ``_sample_anchor_positions``.
            injected_keep_mask: [B, num_anchors] bool — validity for injected
                anchors.
            return_draft: half on-policy OPD — additionally return the OPD dict
                (grad-carrying draft hidden + detached proposals/anchors) as a
                trailing 7th element so the trainer can build the tree, score it,
                and add the OPD-KL term.

        Returns:
            loss: scalar training loss (objective-weighted, + optional distill)
            accuracy: scalar accuracy (binary mask, no decay)
            loss_per_position: [block_size] mean loss at each within-block position
                (index 0 is the anchor slot and always 0; indices 1..B-1 are the
                predicted tokens at 1..B-1 steps past the anchor)
            acc_per_position: [block_size] mean accuracy at each within-block position
            count_per_position: [block_size] valid label count at each within-block
                position before loss decay is applied
            loss_components: dict of per-component loss scalars for logging
                (``ce_loss``/``kl_loss``/``lk_loss``; subclasses add more).
            opd (only when return_draft=True): dict for OPD tree scoring.
        """
        if self.uses_mrope:
            position_ids = validate_mrope_positions(
                position_ids, input_ids.shape[0], input_ids.shape[-1]
            )
            position_pad = (-input_ids.shape[-1]) % self.sp_size if self.sp_size > 1 else 0
            position_ids = F.pad(position_ids, (0, position_pad))
        (
            input_ids,
            loss_mask,
            hidden_states_list,
            last_hidden_states,
            ctx_doc_ids,
            base_position_ids,
        ) = self._usp_pad(
            input_ids,
            loss_mask,
            hidden_states_list,
            last_hidden_states=last_hidden_states,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            context_pre_sharded=context_pre_sharded,
        )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1-6. Shared backbone → draft hidden states + anchor bookkeeping
        #      (doc-aware + injection args threaded through for packing/parity).
        #      Under USP, context hidden states are seq-sharded inside the backbone;
        #      input_ids / labels stay full-sequence.
        # Teacher row 0 of each anchor is the final hidden at its predecessor, the row the
        # MTP fusion reads; under USP it is the only copy of it this rank holds.
        draft_hidden, anchor_positions, block_keep_mask, n_blocks = self._draft_backbone(
            input_ids,
            hidden_states_list,
            loss_mask,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
            position_ids=position_ids,
            context_pre_sharded=context_pre_sharded,
            last_hidden_states=last_hidden_states,
            target_norm=target_norm,
            fusion_prev_hidden=(
                teacher_hidden_at_labels[:, :, 0]
                if self.sp_size > 1 and teacher_hidden_at_labels is not None
                else None
            ),
        )
        fusion_components = {**self._mtp_fusion_stats, **self._attn_diag_stats}

        # 7. Labels first (same-position prediction: slot k predicts token at
        #    anchor+k, or anchor+k+1 under sample_from_anchor), so we can build
        #    ``prev_token_ids`` for the logits hook.
        label_offsets = torch.arange(
            self._label_shift, self.block_size + self._label_shift, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, n_blocks, block_size]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )  # [B, n_blocks, block_size]

        # Teacher-forced previous token at each draft slot: slot 0 is the anchor
        # token, slot j (>0) is target_ids[j-1]. Read by the hidden correction and
        # by the DSpark ``_compute_draft_logits`` / ``_compute_extra_loss`` hooks.
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions.clamp(0, seq_len - 1))
        prev_token_ids = torch.cat([anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)

        def finish(out):
            if not return_draft:
                return out
            prev = prev_token_ids
            proposals = out[6]["proposals"]
            if opd_on_policy_prev and self.draft_model.hidden_correction is not None:
                prev, proposals = self._served_draft_chain(
                    draft_hidden, prev, lm_head_weight, n_blocks
                )
            student = self._apply_hidden_correction(draft_hidden, prev)
            return (*out[:6], {**out[6], "draft_hidden": student, "proposals": proposals})

        # Chunked draft-logit projection (full-vocab logits + their CE gradient
        # are ~half the training-step peak). Only the production path — decay
        # objective, no distillation, no subclass teacher head — is chunked.
        chunk = _dflash_loss_chunk()
        distill_active = (
            self.l1_loss_alpha > 0
            or (
                self.lk_loss_weight > 0.0
                and (last_hidden_states is not None or teacher_hidden_at_labels is not None)
            )
            or (
                self.kl_loss_weight > 0.0
                and (last_hidden_states is not None or teacher_hidden_at_labels is not None)
            )
            or (
                self.e2e_tv_loss_weight > 0.0
                and (last_hidden_states is not None or teacher_hidden_at_labels is not None)
            )
            or self._extra_distill_needed()
        )
        if (
            chunk > 0
            and self.loss_checkpoint_chunks
            and type(self)._compute_extra_loss is DFlashModel._compute_extra_loss
            and not self._extra_distill_needed()
        ):
            from angelspec.models.ops.dflash_chunked_loss import chunked_dflash_loss

            return finish(
                self._with_components(
                    chunked_dflash_loss(
                        self,
                        draft_hidden,
                        target_ids,
                        prev_token_ids,
                        block_keep_mask,
                        valid_label_mask,
                        safe_label_indices,
                        loss_mask,
                        lm_head_weight,
                        n_blocks,
                        anchor_positions,
                        last_hidden_states,
                        teacher_hidden_at_labels,
                        target_norm,
                        chunk,
                        return_draft,
                    ),
                    fusion_components,
                )
            )
        if chunk > 0 and self.loss_objective == "decay" and not distill_active:
            return finish(
                self._with_components(
                    self._forward_chunked_decay(
                        draft_hidden=draft_hidden,
                        target_ids=target_ids,
                        prev_token_ids=prev_token_ids,
                        block_keep_mask=block_keep_mask,
                        valid_label_mask=valid_label_mask,
                        safe_label_indices=safe_label_indices,
                        loss_mask=loss_mask,
                        lm_head_weight=lm_head_weight,
                        n_blocks=n_blocks,
                        anchor_positions=anchor_positions,
                        chunk=chunk,
                        return_draft=return_draft,
                    ),
                    fusion_components,
                )
            )

        # 8. Draft logits (hook): optional hidden-states correction, then the
        #    frozen LM head. DSpark adds a Markov bias on top.
        logits = self._compute_draft_logits(draft_hidden, lm_head_weight, prev_token_ids, n_blocks)
        hc_candidates = self._hc_topk_candidates
        hc_prev_top1 = self._hc_prev_top1

        # Cast to fp32 before CE / argmax so bf16 rounding does not perturb
        # small-margin token rankings (local override of FSDP's bf16 compute).
        if self.fp32_lm_head:
            logits = logits.float()

        # 9. Weight mask: block validity × bounds × exclude
        #    anchor (pos 0, unless it predicts) × loss_mask.
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block >= self.first_pred_slot).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # Binary mask BEFORE objective weighting — accuracy measures "did we
        # predict correctly?" uniformly; weighting only shapes gradient.
        binary_eval_mask = weight_mask.view(-1)

        # 9a. Per-token loss: ce_loss_alpha*CE + l1_loss_alpha*L1.
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = target_ids.view(-1)
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        loss_per_token = self.ce_loss_alpha * ce_per_token
        l1_per_token = None
        if self.l1_loss_alpha > 0:
            if last_hidden_states is None and teacher_hidden_at_labels is None:
                raise ValueError(
                    "DFlash L1 distillation (l1_loss_alpha > 0) requires target "
                    "last_hidden_states; set inference.store_last_hidden_states=true in the "
                    "run config."
                )
            with torch.no_grad():
                if teacher_hidden_at_labels is not None:
                    aligned_hidden = teacher_hidden_at_labels.reshape(
                        bsz, n_blocks * self.block_size, -1
                    )
                else:
                    tgt_idx = (safe_label_indices - 1).clamp(min=0)
                    hdim = last_hidden_states.size(-1)
                    gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
                    aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
                # last_hidden_states is the pre-`norm` slot under vllm capture;
                # apply the target's final RMSNorm before lm_head so the L1
                # teacher matches the KL/LK teacher and the target's real output.
                if target_norm is not None:
                    aligned_hidden = target_norm(aligned_hidden)
                aligned_hidden = aligned_hidden.to(lm_head_weight.dtype)
                target_logits = F.linear(aligned_hidden, lm_head_weight).view(-1, vocab_size)
            l1_per_token = self._compute_l1_loss(flat_logits, target_logits)
            loss_per_token = loss_per_token + self.l1_loss_alpha * l1_per_token

        # Per-position metric uses pure CE (objective-independent).
        loss_per_token_by_position = ce_per_token.view(bsz, n_blocks, self.block_size)

        # 9b. Objective weighting: exp-decay or D-PACE continuation value.
        objective_weights = weight_mask
        if (
            self.loss_objective == "decay"
            and self.loss_decay_gamma is not None
            and self.loss_decay_gamma > 0
        ):
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - self.first_pred_slot).clamp(min=0).float() / self.loss_decay_gamma
            )
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            first = self.first_pred_slot
            dpace_weights = torch.ones_like(weight_mask)
            if self.block_size > first:
                with torch.no_grad():
                    target_confidences = torch.exp(
                        -loss_per_token_by_position[..., first:].float()
                    )
                    dpace_pred_weights = _dpace_position_weights(
                        target_confidences, self.dpace_alpha
                    ).to(dtype=weight_mask.dtype)
                dpace_weights[..., first:] = dpace_pred_weights
            objective_weights = weight_mask * dpace_weights

        flat_weights = objective_weights.view(-1)
        valid_token_count = flat_weights.sum()
        binary_token_count = binary_eval_mask.sum()
        loss_num = (loss_per_token * flat_weights).sum()

        if self.sp_size > 1:
            # Only all-reduce the counts (detached). Each rank's loss = local_sum /
            # global_count; SP grad SUM reconstructs the full-sequence mean grad.
            # The unweighted count rides the same collective so the selector can
            # normalize by it without a second barrier.
            counts = torch.stack([valid_token_count.detach(), binary_token_count.detach()])
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.sp_group)
            global_binary_count = counts[1]
            loss_denom = counts[0].clamp(min=1e-6)
            # Keep the context gather on the backward graph even when this shard
            # has zero supervised tokens (else peers enter reduce-scatter and we don't).
            connect = draft_hidden.sum() * 0.0
            loss = loss_num / loss_denom + connect
        else:
            global_binary_count = binary_token_count
            loss_denom = valid_token_count.clamp(min=1e-6)
            loss = loss_num / loss_denom

        # 9c. Optional KL / LK distillation vs the target's true last-layer logits.
        #     Convex-mix in [0,1] against the base loss; LK precedes KL. Teacher
        #     logits are also computed when a subclass head needs them
        #     (``_extra_distill_needed``) even if KL/LK are off.
        base_loss = loss
        kl_loss = torch.zeros((), device=device, dtype=base_loss.dtype)
        lk_loss = torch.zeros((), device=device, dtype=base_loss.dtype)
        e2e_tv_loss = torch.zeros((), device=device, dtype=base_loss.dtype)
        e2e_accept_len = None

        teacher_available = last_hidden_states is not None or teacher_hidden_at_labels is not None
        lk_active = self.lk_loss_weight > 0.0 and teacher_available
        kl_active = (not lk_active) and self.kl_loss_weight > 0.0 and teacher_available
        e2e_tv_active = self.e2e_tv_loss_weight > 0.0 and teacher_available
        want_teacher = lk_active or kl_active or e2e_tv_active or self._extra_distill_needed()

        teacher_logits_flat = None
        teacher_ids_flat = None
        if want_teacher and teacher_available:
            with torch.no_grad():
                if teacher_hidden_at_labels is not None:
                    teacher_hidden = teacher_hidden_at_labels
                else:
                    teacher_label_indices = (safe_label_indices - 1).clamp(min=0)
                    gather_idx = teacher_label_indices.unsqueeze(-1).expand(
                        -1, -1, -1, last_hidden_states.size(-1)
                    )
                    teacher_hidden = torch.gather(
                        last_hidden_states.unsqueeze(1).expand(-1, n_blocks, -1, -1),
                        2,
                        gather_idx,
                    )
                if target_norm is not None:
                    teacher_hidden = target_norm(teacher_hidden)
                teacher_hidden = teacher_hidden.to(lm_head_weight.dtype)
                teacher_logits = F.linear(teacher_hidden, lm_head_weight)
                teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1)).detach()
                # The token a greedy target would emit: the accept test at serving
                # time, so both the selector label and the teacher_acc metric read
                # it. Full-vocab argmax, hence computed once.
                teacher_ids_flat = teacher_logits_flat.argmax(dim=-1)

        # Distill terms use the same global count under USP as the CE term.
        distill_denom = (
            loss_denom.float() if self.sp_size > 1 else valid_token_count.float().clamp(min=1e-6)
        )
        # Same local_sum / global_count contract, but over the unweighted mask —
        # the denominator the selector needs when it opts out of objective
        # weighting (``path_selector_objective_weighting=False``).
        binary_denom = global_binary_count.float()
        if lk_active:
            lk_per_position = self._compute_lk_loss(
                student_logits=flat_logits,
                teacher_logits=teacher_logits_flat,
                loss_type=self.lk_loss_type,
                eta=self.lk_eta,
            )
            lk_loss = ((lk_per_position * flat_weights.float()).sum() / distill_denom).to(
                base_loss.dtype
            )
            distill_w = max(0.0, min(1.0, self.lk_loss_weight))
            loss = (
                lk_loss
                if distill_w >= 1.0
                else distill_w * lk_loss + (1.0 - distill_w) * base_loss
            )
        elif kl_active:
            kl_per_position = self._compute_topk_kl_loss_variant_b(
                student_logits=flat_logits,
                teacher_logits=teacher_logits_flat,
                topk=self.kl_topk,
            )
            kl_loss = ((kl_per_position * flat_weights.float()).sum() / distill_denom).to(
                base_loss.dtype
            )
            distill_w = max(0.0, min(1.0, self.kl_loss_weight))
            loss = (
                kl_loss
                if distill_w >= 1.0
                else distill_w * kl_loss + (1.0 - distill_w) * base_loss
            )

        # 9c'. Independent e2e multi-step TV term, added on top of the total
        #      (not mutually exclusive with KL/LK). Bypasses flat_weights/decay.
        if e2e_tv_active and teacher_logits_flat is not None:
            vocab_size_e2e = flat_logits.size(-1)
            e2e_tv_loss, e2e_accept_len = self._compute_e2e_tv_loss(
                student_logits_pb=flat_logits.view(bsz, n_blocks, self.block_size, vocab_size_e2e),
                teacher_logits_pb=teacher_logits_flat.view(
                    bsz, n_blocks, self.block_size, vocab_size_e2e
                ),
                valid_mask_pb=weight_mask,
                vocab_chunk=self.e2e_tv_vocab_chunk,
            )
            loss = loss + self.e2e_tv_loss_weight * e2e_tv_loss

        # 9c''. Path selector: independent K-way CE over the unary top-k, added on
        #       top of the total. Teacher-forces the predecessor token, exactly as
        #       ``prev_token_ids`` already does for the hidden correction, and
        #       reranks from the same logits the drafter serves. The hidden input is
        #       the backbone output (pre-correction), which is what the selector
        #       sees at inference.
        selector_components: dict = {}
        if self._path_selector() is not None and self.path_selector_loss_alpha != 0.0:
            # The K-way CE is a hard-label ranking term, so its label has to name the
            # token the unary head is pushed toward. The corpus token qualifies only
            # while the corpus CE actually carries weight: KL/LK displace that term
            # (fully at weight 1.0) and ce_loss_alpha=0 drops it outright, either way
            # leaving the drafter aligned to the target — whose argmax is also what
            # acceptance is tested against at serving time. Labelling the selector
            # with the corpus token there would train it to rank a token the verifier
            # rejects wherever the two disagree. Note that e2e-TV alone does not flip
            # the label: it is added on top of the CE rather than replacing it, so a
            # nonzero ce_loss_alpha still trains the tokens against the corpus.
            corpus_ce_active = self.ce_loss_alpha > 0.0 and not (lk_active or kl_active)
            selector_label_ids = target_ids.view(bsz, -1)
            if (
                self.path_selector_teacher_labels
                and not corpus_ce_active
                and teacher_ids_flat is not None
            ):
                selector_label_ids = teacher_ids_flat.view(bsz, -1)
            # Numerator and denominator must use the same weights, or the term
            # stops being a weighted mean of the K-way CE.
            sel_weights, sel_denom = (
                (objective_weights, distill_denom)
                if self.path_selector_objective_weighting
                else (weight_mask, binary_denom)
            )
            sel_ce_sum, sel_recall_sum, sel_acc_sum, sel_serving = self._path_selector_sums(
                unary_logits=logits,
                hidden=draft_hidden,
                prev_token_ids=prev_token_ids.view(bsz, -1),
                label_ids=selector_label_ids,
                objective_weights=sel_weights.view(bsz, -1),
                binary_mask=weight_mask.view(bsz, -1),
            )
            # Clamped: a fully masked micro-batch leaves the denominator at 0, and
            # the numerator is 0 there too, so the term must read 0 and not NaN.
            selector_loss = (sel_ce_sum / sel_denom.clamp(min=1e-6)).to(base_loss.dtype)
            loss = loss + self.path_selector_loss_alpha * selector_loss
            selector_components = {
                "selector_loss": selector_loss.detach(),
                "selector_recall": (
                    sel_recall_sum / binary_eval_mask.sum().clamp(min=1e-6)
                ).detach(),
                "selector_acc": (sel_acc_sum / sel_serving.clamp(min=1e-6)).detach(),
            }

        # 9d. Subclass extra-loss hook (DSpark confidence head; no-op for DFlash).
        loss, extra_components = self._compute_extra_loss(
            loss,
            flat_logits,
            teacher_logits_flat,
            flat_weights,
            distill_denom,
            prev_token_ids,
            n_blocks,
        )

        # 9e. Optional gate-sparsity penalty (gated_sum layer-selection; default off).
        #     Added to the final total so it applies regardless of distillation.
        if self.gate_entropy_weight > 0 and hasattr(self.draft_model, "gate_entropy"):
            loss = loss + self.gate_entropy_weight * self.draft_model.gate_entropy()

        # 10. Accuracy (using binary mask without decay). Score the token the
        #     serving path actually emits: selector rerank when the head exists,
        #     otherwise the unary argmax.
        with torch.no_grad():
            pred_ids = self._greedy_draft_ids(
                logits,
                draft_hidden,
                prev_token_ids.view(bsz, -1),
            ).reshape(-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count

            # Per-position-within-block metrics (index 0 = anchor, masked out;
            # indices 1..block_size-1 correspond to 1..B-1 tokens past the anchor).
            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size)
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)

            loss_per_position = (loss_per_token_by_position * binary_weights).sum(
                dim=(0, 1)
            ) / count_per_pos
            acc_per_position = (correct.view(bsz, n_blocks, self.block_size).float()).sum(
                dim=(0, 1)
            ) / count_per_pos

            # ``acc_per_position`` above scores the serving draft token (selector
            # pick, else unary argmax) against the corpus token. The serving
            # accept test is ``draft token == target argmax``, so score that too
            # (MTP splits the same pair as acces / acces_gt). Teacher logits are
            # already materialised for the distill terms; absent only when every
            # distill term is off.
            teacher_acc_per_position = None
            if teacher_ids_flat is not None:
                correct_vs_teacher = (pred_ids == teacher_ids_flat) & (binary_eval_mask > 0.5)
                teacher_acc_per_position = (
                    correct_vs_teacher.view(bsz, n_blocks, self.block_size).float()
                ).sum(dim=(0, 1)) / count_per_pos

            # Under sampling the target accepts a draft token with probability
            # min(1, p/q), which the greedy agreement above does not see: sum_v
            # min(p_v, q_v) when the drafter samples, p of its token when it proposes
            # its argmax (vLLM's default, the draft then counting as one-hot).
            sample_accept_per_position = argmax_accept_per_position = None
            if (
                self.eval_accept_temperature
                and not self.training
                and teacher_logits_flat is not None
            ):
                sampled, proposed = _accept_rates(
                    flat_logits, teacher_logits_flat, pred_ids, self.eval_accept_temperature
                )
                sample_accept_per_position, argmax_accept_per_position = (
                    (rate.view(bsz, n_blocks, self.block_size) * binary_weights).sum(dim=(0, 1))
                    / count_per_pos
                    for rate in (sampled, proposed)
                )

        # Single registration point for all loss terms (for logging). Components
        # are the PURE objective-weighted means of each term (``ce`` and ``l1``
        # separately, not the ``ce_α·ce + l1_α·l1`` combination), so they read as
        # an interpretable decomposition; ``kl``/``lk`` are the distill terms; the
        # extra-loss hook merges its own named components (e.g. confidence_loss).
        # Add a term here + list its key in the trainer's
        # ``_extra_loss_component_keys`` — never change the tuple arity.
        ce_component = (ce_per_token.detach() * flat_weights).sum() / valid_token_count.clamp(
            min=1e-6
        )
        loss_components = {
            "ce_loss": ce_component.detach(),
            "kl_loss": kl_loss.detach(),
            "lk_loss": lk_loss.detach(),
            "e2e_tv_loss": e2e_tv_loss.detach(),
        }
        if l1_per_token is not None:
            loss_components["l1_loss"] = (
                (l1_per_token.detach() * flat_weights.float()).sum()
                / valid_token_count.clamp(min=1e-6).float()
            ).detach()
        if e2e_accept_len is not None:
            # Expected accepted length under the sampling-mode coupling
            # (Σ_v min(p,q) is exactly the speculative-sampling accept rate), so
            # it reads as an upper companion to the greedy ``simulated_acc_len``.
            loss_components["e2e_accept_len"] = e2e_accept_len.detach()
        if teacher_acc_per_position is not None:
            # Per-position tensor, not a scalar: reduced by the trainer's
            # count-weighted position reducer, not ``_reduce_loss_components``.
            loss_components["teacher_acc_per_position"] = teacher_acc_per_position
        if sample_accept_per_position is not None:
            loss_components["sample_accept_per_position"] = sample_accept_per_position
            loss_components["argmax_accept_per_position"] = argmax_accept_per_position
        loss_components.update(selector_components)
        loss_components.update(fusion_components)
        loss_components.update(self._acorr_components())
        loss_components.update(self._index_components())
        loss_components.update(extra_components)
        if hc_candidates is not None:
            # Acceptance is tested against the target's argmax; the corpus token
            # stands in when no teacher logits were computed. The per-position tensor
            # goes through the trainer's count-weighted position reducer.
            hc_labels = teacher_ids_flat if teacher_ids_flat is not None else flat_targets
            hc_hits = _hc_topk_hits(hc_candidates, hc_labels, binary_weights)
            loss_components["hc_topk_recall"] = (
                hc_hits.sum() / binary_eval_mask.sum().clamp(min=1e-6)
            ).detach()
            loss_components["hc_topk_recall_per_position"] = (hc_hits / count_per_pos).detach()
        if hc_prev_top1 is not None:
            # Slots >= 2 split by whether the previous token tops the previous slot's
            # uncorrected logits; labelled like hc_topk_recall.
            with torch.no_grad():
                labels = teacher_ids_flat if teacher_ids_flat is not None else flat_targets
                later = (binary_weights[:, :, 2:] > 0.5).reshape(-1)
                top1 = hc_prev_top1.view(bsz, n_blocks, -1)[:, :, 2:].reshape(-1)
                hit = (pred_ids == labels).view(bsz, n_blocks, -1)[:, :, 2:].reshape(-1)
                expected, other = later & top1, later & ~top1
                loss_components["hc_prev_top1_rate"] = (
                    expected.sum() / later.sum().clamp(min=1)
                ).float()
                loss_components["hc_acc_prev_top1"] = (
                    (hit & expected).sum() / expected.sum().clamp(min=1)
                ).float()
                loss_components["hc_acc_prev_other"] = (
                    (hit & other).sum() / other.sum().clamp(min=1)
                ).float()
        if (
            self.hc_topk_diag
            and not self.training
            and self.sp_size == 1
            and getattr(self.draft_model, "hidden_correction", None) is not None
        ):
            loss_components.update(
                self._hc_topk_diag(
                    draft_hidden=draft_hidden,
                    prev_token_ids=prev_token_ids,
                    lm_head_weight=lm_head_weight,
                    teacher_ids_flat=teacher_ids_flat,
                    binary_eval_mask=binary_eval_mask,
                    n_blocks=n_blocks,
                )
            )

        if return_draft:
            # Half on-policy OPD: hand back the grad-carrying draft hidden + the
            # (detached) proposal ids / anchors so the trainer can build the tree,
            # score it on the target, and add the OPD-KL term. proposals[..,0] is
            # the anchor slot.
            opd = {
                "draft_hidden": draft_hidden,
                "proposals": logits.detach().view(bsz, n_blocks, self.block_size, -1).argmax(-1),
                "anchor_positions": anchor_positions,
                "block_keep_mask": block_keep_mask,
            }
            return finish(
                (
                    loss,
                    accuracy,
                    loss_per_position,
                    acc_per_position,
                    count_per_position,
                    loss_components,
                    opd,
                )
            )

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )

    def _forward_chunked_decay(
        self,
        *,
        draft_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        prev_token_ids: torch.Tensor,
        block_keep_mask: torch.Tensor,
        valid_label_mask: torch.Tensor,
        safe_label_indices: torch.Tensor,
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        n_blocks: int,
        anchor_positions: torch.Tensor,
        chunk: int,
        return_draft: bool,
    ):
        """Memory-lean equivalent of ``forward``'s decay + no-distill tail.

        Projects draft logits one block-group at a time instead of materializing
        the full ``[B, n_blocks*block_size, V]`` tensor (and its CE gradient) at
        once. Numerically equals the full path up to summation order; gated on in
        ``forward`` only when the objective is decay and no distillation / subclass
        teacher head is active, so every value here mirrors the corresponding full
        path line exactly. Chunk unit is blocks, so per-block-position weighting
        (decay) and metrics stay intact.
        """
        bsz = draft_hidden.shape[0]
        device = draft_hidden.device
        bs = self.block_size
        D = draft_hidden.shape[-1]

        if not getattr(self, "_dflash_chunk_logged", False):
            logger.info(
                "DFlash chunked-projection loss active "
                f"(ANGELSPEC_DFLASH_LOSS_CHUNK={chunk} rows, {max(1, chunk // bs)} blocks/chunk)"
            )
            self._dflash_chunk_logged = True

        # weight_mask — identical to the full path (block validity × bounds ×
        # exclude anchor pos 0 × loss_mask). Logit-free, so computed up front.
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, bs).float()
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(bs, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block >= self.first_pred_slot).float()
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        weight_mask = weight_mask * original_loss_mask_gathered  # [B, n_blocks, bs]

        # Binary (accuracy) mask is weight_mask before objective weighting.
        binary_eval_mask = weight_mask

        # Decay objective weights (logit-free); matches full path lines 799-801.
        objective_weights = weight_mask
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(bs, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - self.first_pred_slot).clamp(min=0).float() / self.loss_decay_gamma
            )
            objective_weights = weight_mask * decay_weights

        valid_token_count = objective_weights.view(-1).sum().clamp(min=1e-6)
        actual_token_count = binary_eval_mask.view(-1).sum().clamp(min=1e-6)
        count_per_position = binary_eval_mask.sum(dim=(0, 1))
        count_per_pos = count_per_position.clamp(min=1.0)

        blocks_per_chunk = max(1, chunk // bs)
        dh = draft_hidden.view(bsz, n_blocks, bs, D)

        loss_num = draft_hidden.new_zeros(())
        ce_comp_num = draft_hidden.new_zeros(())
        correct_num = torch.zeros((), device=device)
        loss_pos_num = torch.zeros(bs, device=device)
        acc_pos_num = torch.zeros(bs, device=device)
        proposals = (
            torch.empty(bsz, n_blocks, bs, dtype=torch.long, device=device)
            if return_draft
            else None
        )
        # Path selector is position-local, so it accumulates per chunk and
        # normalizes once below — same value as the full path. This path is only
        # taken when no distillation is active, so the corpus token is still what
        # the unary CE trains and therefore the selector's label (see ``forward``).
        selector_active = (
            self._path_selector() is not None and self.path_selector_loss_alpha != 0.0
        )
        sel_ce_num = draft_hidden.new_zeros(())
        sel_recall_num = torch.zeros((), device=device)
        sel_acc_num = torch.zeros((), device=device)
        sel_serving_num = torch.zeros((), device=device)
        sel_weights = (
            objective_weights if self.path_selector_objective_weighting else binary_eval_mask
        )
        hc_hits_num = torch.zeros(bs, device=device)

        for start in range(0, n_blocks, blocks_per_chunk):
            end = min(start + blocks_per_chunk, n_blocks)
            nb = end - start
            dh_chunk = dh[:, start:end].reshape(bsz, nb * bs, D)
            # Keep prev_token_ids [B, nb, bs] (not flattened): the logits hook
            # indexes it as [B, n_blocks, block_size] for the hidden correction
            # and the DSpark Markov head, matching the non-chunked path.
            prev_chunk = prev_token_ids[:, start:end]
            logits_chunk = self._compute_draft_logits(
                dh_chunk, lm_head_weight, prev_chunk, nb, block_start=start
            )
            if self.fp32_lm_head:
                logits_chunk = logits_chunk.float()
            flat = logits_chunk.view(-1, logits_chunk.size(-1))
            tgt = target_ids[:, start:end].reshape(-1)
            if self._hc_topk_candidates is not None:
                hc_hits_num = hc_hits_num + _hc_topk_hits(
                    self._hc_topk_candidates, tgt, binary_eval_mask[:, start:end]
                )
            ce = F.cross_entropy(flat, tgt, reduction="none")

            w = objective_weights[:, start:end].reshape(-1)
            loss_num = loss_num + (self.ce_loss_alpha * ce * w).sum()
            ce_comp_num = ce_comp_num + (ce.detach() * w).sum()

            if selector_active:
                chunk_ce, chunk_recall, chunk_acc, chunk_serving = self._path_selector_sums(
                    unary_logits=logits_chunk,
                    hidden=dh_chunk,
                    prev_token_ids=prev_chunk.reshape(bsz, -1),
                    label_ids=target_ids[:, start:end].reshape(bsz, -1),
                    objective_weights=sel_weights[:, start:end].reshape(bsz, -1),
                    binary_mask=binary_eval_mask[:, start:end].reshape(bsz, -1),
                )
                sel_ce_num = sel_ce_num + chunk_ce
                sel_recall_num = sel_recall_num + chunk_recall
                sel_acc_num = sel_acc_num + chunk_acc
                sel_serving_num = sel_serving_num + chunk_serving

            with torch.no_grad():
                b = binary_eval_mask[:, start:end].reshape(-1)
                pred = self._greedy_draft_ids(
                    logits_chunk,
                    dh_chunk,
                    prev_chunk.reshape(bsz, -1),
                ).reshape(-1)
                correct = (pred == tgt) & (b > 0.5)
                correct_num = correct_num + correct.sum().float()
                loss_pos_num = loss_pos_num + (ce.view(bsz, nb, bs) * b.view(bsz, nb, bs)).sum(
                    dim=(0, 1)
                )
                acc_pos_num = acc_pos_num + correct.view(bsz, nb, bs).float().sum(dim=(0, 1))
                if proposals is not None:
                    proposals[:, start:end] = pred.view(bsz, nb, bs)

        if self.sp_size > 1:
            # Both counts ride one collective; the second is the denominator the
            # selector needs when it opts out of objective weighting.
            counts = torch.stack([valid_token_count.detach(), actual_token_count.detach()])
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.sp_group)
            connect = draft_hidden.sum() * 0.0
            loss_denom = counts[0].clamp(min=1e-6)
            binary_denom = counts[1].clamp(min=1e-6)
            loss = loss_num / loss_denom + connect
        else:
            loss_denom = valid_token_count
            binary_denom = actual_token_count
            loss = loss_num / loss_denom

        selector_components: dict = {}
        if selector_active:
            sel_denom = loss_denom if self.path_selector_objective_weighting else binary_denom
            selector_loss = (sel_ce_num / sel_denom.float()).to(loss.dtype)
            loss = loss + self.path_selector_loss_alpha * selector_loss
            selector_components = {
                "selector_loss": selector_loss.detach(),
                "selector_recall": (sel_recall_num / actual_token_count).detach(),
                "selector_acc": (sel_acc_num / sel_serving_num.clamp(min=1e-6)).detach(),
            }

        if self.gate_entropy_weight > 0 and hasattr(self.draft_model, "gate_entropy"):
            loss = loss + self.gate_entropy_weight * self.draft_model.gate_entropy()
        accuracy = correct_num / actual_token_count
        loss_per_position = loss_pos_num / count_per_pos
        acc_per_position = acc_pos_num / count_per_pos

        zero = torch.zeros((), device=device, dtype=loss.dtype)
        loss_components = {
            "ce_loss": (ce_comp_num / valid_token_count.clamp(min=1e-6)).detach(),
            "kl_loss": zero,
            "lk_loss": zero,
            **selector_components,
        }
        if self.hidden_correction_topk:
            # No distillation on this path, so the corpus token is the label.
            loss_components["hc_topk_recall"] = (hc_hits_num.sum() / actual_token_count).detach()
            loss_components["hc_topk_recall_per_position"] = (hc_hits_num / count_per_pos).detach()

        if return_draft:
            opd = {
                "draft_hidden": draft_hidden,
                "proposals": proposals,
                "anchor_positions": anchor_positions,
                "block_keep_mask": block_keep_mask,
            }
            return (
                loss,
                accuracy,
                loss_per_position,
                acc_per_position,
                count_per_position,
                loss_components,
                opd,
            )
        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )
