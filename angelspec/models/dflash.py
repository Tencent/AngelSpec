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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.ops.flex_attention import compile_friendly_create_block_mask
from angelspec.models.ops.loss import _kl_variant_b, lk_tv_kl_per_pos
from angelspec.utils.logging import logger

_VALID_DFLASH_LOSS_OBJECTIVES = {"decay", "dpace"}


def _dflash_loss_chunk() -> int:
    """Row budget for chunked draft-logit projection (0/unset => no chunking).

    Mirrors ANGELSPEC_MTP_LOSS_CHUNK. Only the decay + no-distill forward path
    honors it (the production path); dpace / distillation / subclass heads keep
    the single full-vocab projection.
    """
    return int(os.environ.get("ANGELSPEC_DFLASH_LOSS_CHUNK", "0") or 0)


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


def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    ctx_doc_ids: Optional[torch.Tensor] = None,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention is bidirectional
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing
      5. Sequence packing (ctx_doc_ids given): a block additionally only sees
         context tokens in the SAME document as its anchor (and non-padding),
         so packed docs never leak across boundaries.
    """
    num_anchors = anchor_positions.shape[1]
    packed = ctx_doc_ids is not None

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < ctx_len
        mask_context = is_context & (kv_idx < anchor_pos)

        if packed:
            a_doc = ctx_doc_ids[b, anchor_pos]
            # Clamp context index so the gather stays in-bounds for block-region
            # kv positions; only used when is_context is True.
            kv_ctx = torch.where(is_context, kv_idx, torch.zeros_like(kv_idx))
            kv_doc = ctx_doc_ids[b, kv_ctx]
            mask_context = mask_context & (a_doc >= 0) & (kv_doc == a_doc)

        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    suffix = "_packed" if packed else ""
    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}{suffix}"
    return dflash_mask_mod


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with exponential decay weighting
      - Per-position loss_mask application
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

        self.draft_model = draft_model
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

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
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
        max_anchor = max(seq_len - bs, 0)
        max_n = self.num_anchors

        # Deterministic injection path: use given anchors as-is.
        if injected_anchors is not None:
            anchors = injected_anchors.to(device=device, dtype=torch.long)
            if injected_keep_mask is not None:
                keep_mask = injected_keep_mask.to(device=device, dtype=torch.bool)
            else:
                keep_mask = torch.ones(bsz, max_n, dtype=torch.bool, device=device)
            anchors = torch.where(keep_mask, anchors, 0)
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

        # Sequence packing: an anchor's block [a, a+bs) must not straddle a doc
        # boundary. Documents are contiguous, so requiring the block's first and
        # last token to share a non-padding doc id is sufficient.
        if ctx_doc_ids is not None:
            doc = ctx_doc_ids.to(device=device)
            head_doc = doc[:, : max_anchor + 1]
            tail_doc = doc[:, bs - 1 : bs - 1 + max_anchor + 1]
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
        """
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)

        if base_position_ids is not None:
            base = base_position_ids.to(device=device, dtype=torch.long)
            context_position_ids = base
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

    def _draft_backbone(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Shared DFlash backbone (steps 1-6): context features → anchor
        sampling → noise embedding → position ids → block-causal mask → draft
        forward. ``DFlashModel.forward`` and DSpark/DFly subclasses build the
        draft hidden states this way; only the label/loss tail differs.

        Doc-aware (``ctx_doc_ids`` / ``base_position_ids``) and anchor-injection
        args are threaded through for packing and parity tests.

        Returns:
            draft_hidden: [B, n_blocks*block_size, D] pre-loss draft hidden states
            anchor_positions: [B, n_blocks] sampled anchor positions
            block_keep_mask: [B, n_blocks] bool validity of each anchor slot
            n_blocks: number of anchor slots (== num_anchors)
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        context_feature = self.draft_model.extract_context_feature(hidden_states_list)

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            ctx_doc_ids=ctx_doc_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
        )
        n_blocks = anchor_positions.shape[1]

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len, base_position_ids=base_position_ids
        )

        draft_len = n_blocks * self.block_size
        kv_len = seq_len + draft_len

        block_mask = None
        if device.type == "cuda":
            mask_mod = _create_dflash_mask_mod(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.block_size,
                ctx_doc_ids=ctx_doc_ids,
            )
            block_mask = compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=bsz,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=kv_len,
                device=device,
            )

        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )

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

    @staticmethod
    def _compute_e2e_tv_loss(
        student_logits_pb: torch.Tensor,
        teacher_logits_pb: torch.Tensor,
        valid_mask_pb: torch.Tensor,
    ):
        """End-to-end multi-step TV loss (γ-step accepted-length objective)::

            α_i     = 1 - TV(p_i, q_i) = Σ_v min(p_i,v, q_i,v)  ∈ (0, 1]
            L_e2e   = 1 - (1/γ) * Σ_{j=1..γ}  Π_{i=1..j} α_i

        γ = block_size. The prefix product couples steps inside a block, giving
        intrinsic per-step weighting, so this term ignores decay / flat_weights.
        Inputs are ``[B, n_blocks, block_size, V]`` logits and a
        ``[B, n_blocks, block_size]`` validity mask (T=1). Returns
        ``(e2e_tv_loss, accept_length)`` (the latter detached, for logging).
        """
        # fp32 for the O(V) min-sum and the chain of up-to-γ products.
        t = torch.softmax(teacher_logits_pb.float(), dim=-1)
        s = torch.softmax(student_logits_pb.float(), dim=-1)

        # α = Σ_v min(p, q); gradient flows through the student branch of min.
        alpha = torch.minimum(t, s).sum(dim=-1)  # [B, nb, γ]

        # Set α:=1 on invalid slots so cumprod treats them as identity.
        m = valid_mask_pb.float()
        alpha_effective = alpha * m + (1.0 - m)
        prefix_prod = torch.cumprod(alpha_effective, dim=-1)  # [B, nb, γ]

        gamma_valid = m.sum(dim=-1).clamp(min=1.0)  # [B, nb]
        accept_length_pb = (prefix_prod * m).sum(dim=-1)  # [B, nb]
        e2e_per_block = 1.0 - accept_length_pb / gamma_valid

        block_has_valid = (m.sum(dim=-1) > 0).float()
        denom = block_has_valid.sum().clamp(min=1.0)
        e2e_tv_loss = (e2e_per_block * block_has_valid).sum() / denom

        with torch.no_grad():
            accept_length = (accept_length_pb * block_has_valid).sum() / denom

        return e2e_tv_loss, accept_length.detach()

    # ------------------------------------------------------------------
    # Subclass extension hooks (no-ops for base DFlash). DSpark / TreeFlash
    # override these to inject hidden-state correction + Markov logit bias and
    # the confidence-head loss. Signatures are frozen here so subclasses attach
    # without reworking forward. See models/dspark.py.
    # ------------------------------------------------------------------
    def _compute_draft_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ) -> torch.Tensor:
        """Project draft hidden states to vocab logits. DFlash uses the frozen
        LM head directly; DSpark overrides to add hidden-correction + Markov bias."""
        return (
            self.draft_model.lm_head(draft_hidden)
            if hasattr(self.draft_model, "lm_head")
            else F.linear(draft_hidden, lm_head_weight)
        )

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
        valid_token_count: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Add subclass-specific loss terms on top of the DFlash objective.

        Returns ``(loss, extra_components)``, where ``extra_components`` holds
        detached scalars merged into ``loss_components`` for logging. DFlash adds
        nothing; DSpark adds e.g. ``{"confidence_loss": ...}``."""
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
    ):
        """On-policy DFlash proposal for OPD packed tree-forward scoring (no grad).

        Runs the block-parallel draft and returns the argmax'd proposals. The
        within-block layout is ``[anchor_slot(0), pred@1, ..., pred@{B-1}]``: slot
        0 is the anchor input, slots 1..B-1 are the draft's proposed tokens for
        absolute positions ``anchor+1 .. anchor+B-1``. The caller drops slot 0 and
        feeds the B-1 proposals as branch tokens to ``build_tree_layout`` (block
        length B-1), then dispatches ``score_packed`` for the teacher distribution.

        Returns:
            proposals: [B, n_blocks, block_size] long — argmax per within-block slot.
            anchor_positions: [B, n_blocks] long.
            block_keep_mask: [B, n_blocks] bool — valid anchors.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            ctx_doc_ids=ctx_doc_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
        )
        n_blocks = anchor_positions.shape[1]
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len, base_position_ids=base_position_ids
        )
        draft_len = n_blocks * self.block_size
        block_mask = None
        if device.type == "cuda":
            mask_mod = _create_dflash_mask_mod(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.block_size,
                ctx_doc_ids=ctx_doc_ids,
            )
            block_mask = compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=bsz,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=seq_len + draft_len,
                device=device,
            )
        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
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
        target_norm: Optional[nn.Module] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        return_draft: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full DFlash training forward pass.

        Args:
            input_ids: [B, seq_len] token IDs.
            hidden_states_list: per-target-layer [B, seq_len, D] hidden states.
            loss_mask: [B, seq_len] 1 for supervised positions.
            lm_head_weight: frozen target LM head weight.
            last_hidden_states: [B, seq_len, D] target pre-norm final hidden states
                (from mooncake); required only for KL/LK/L1 distillation. None
                disables the distillation terms (base CE path).
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
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1-6. Shared backbone → draft hidden states + anchor bookkeeping
        #      (doc-aware + injection args threaded through for packing/parity).
        draft_hidden, anchor_positions, block_keep_mask, n_blocks = self._draft_backbone(
            input_ids,
            hidden_states_list,
            loss_mask,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
        )

        # 7. Labels first (same-position prediction: slot k predicts token at
        #    anchor+k), so we can build ``prev_token_ids`` for the logits hook.
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, n_blocks, block_size]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )  # [B, n_blocks, block_size]

        # Teacher-forced previous token at each draft slot: slot 0 is the anchor
        # token, slot j (>0) is target_ids[j-1]. DFlash ignores it; the DSpark
        # ``_compute_draft_logits`` / ``_compute_extra_loss`` hooks condition on it.
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions.clamp(0, seq_len - 1))
        prev_token_ids = torch.cat([anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)

        # Chunked draft-logit projection (full-vocab logits + their CE gradient
        # are ~half the training-step peak). Only the production path — decay
        # objective, no distillation, no subclass teacher head — is chunked.
        chunk = _dflash_loss_chunk()
        distill_active = (
            self.l1_loss_alpha > 0
            or (self.lk_loss_weight > 0.0 and last_hidden_states is not None)
            or (self.kl_loss_weight > 0.0 and last_hidden_states is not None)
            or (self.e2e_tv_loss_weight > 0.0 and last_hidden_states is not None)
            or self._extra_distill_needed()
        )
        if chunk > 0 and self.loss_objective == "decay" and not distill_active:
            return self._forward_chunked_decay(
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
            )

        # 8. Draft logits (hook). DFlash: frozen LM head. DSpark: hidden-states
        #    correction + Markov bias conditioned on ``prev_token_ids``.
        logits = self._compute_draft_logits(draft_hidden, lm_head_weight, prev_token_ids, n_blocks)

        # Cast to fp32 before CE / argmax so bf16 rounding does not perturb
        # small-margin token rankings (local override of FSDP's bf16 compute).
        if self.fp32_lm_head:
            logits = logits.float()

        # 9. Weight mask: block validity × bounds × exclude
        #    anchor (pos 0) × loss_mask.
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

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
            if last_hidden_states is None:
                raise ValueError(
                    "DFlash L1 distillation (l1_loss_alpha > 0) requires target "
                    "last_hidden_states; set inference.store_last_hidden_states=true in the "
                    "run config."
                )
            tgt_idx = (safe_label_indices - 1).clamp(min=0)  # [B, n_blocks, block_size]
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
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
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            dpace_weights = torch.ones_like(weight_mask)
            if self.block_size > 1:
                with torch.no_grad():
                    target_confidences = torch.exp(-loss_per_token_by_position[..., 1:].float())
                    dpace_pred_weights = _dpace_position_weights(
                        target_confidences, self.dpace_alpha
                    ).to(dtype=weight_mask.dtype)
                dpace_weights[..., 1:] = dpace_pred_weights
            objective_weights = weight_mask * dpace_weights

        flat_weights = objective_weights.view(-1)
        valid_token_count = flat_weights.sum().clamp(min=1e-6)
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        # 9c. Optional KL / LK distillation vs the target's true last-layer logits.
        #     Convex-mix in [0,1] against the base loss; LK precedes KL. Teacher
        #     logits are also computed when a subclass head needs them
        #     (``_extra_distill_needed``) even if KL/LK are off.
        base_loss = loss
        kl_loss = torch.zeros((), device=device, dtype=base_loss.dtype)
        lk_loss = torch.zeros((), device=device, dtype=base_loss.dtype)
        e2e_tv_loss = torch.zeros((), device=device, dtype=base_loss.dtype)

        lk_active = self.lk_loss_weight > 0.0 and last_hidden_states is not None
        kl_active = (
            (not lk_active) and self.kl_loss_weight > 0.0 and last_hidden_states is not None
        )
        e2e_tv_active = self.e2e_tv_loss_weight > 0.0 and last_hidden_states is not None
        want_teacher = lk_active or kl_active or e2e_tv_active or self._extra_distill_needed()

        teacher_logits_flat = None
        if want_teacher and last_hidden_states is not None:
            with torch.no_grad():
                # last_hidden_states is the pre-`norm` slot under vllm capture;
                # apply the target's final RMSNorm before lm_head so teacher
                # logits match what the target emits at inference time.
                lhs = last_hidden_states
                if target_norm is not None:
                    lhs = target_norm(lhs)
                lhs = lhs.to(lm_head_weight.dtype)

                # Teacher LM at position p emits next-token logits, so gather
                # teacher hidden at (anchor+k - 1) to match input_ids[anchor+k].
                teacher_label_indices = (safe_label_indices - 1).clamp(min=0)
                gather_idx = teacher_label_indices.unsqueeze(-1).expand(-1, -1, -1, lhs.size(-1))
                teacher_hidden_at_labels = torch.gather(
                    lhs.unsqueeze(1).expand(-1, n_blocks, -1, -1), 2, gather_idx
                )
                teacher_logits = F.linear(teacher_hidden_at_labels, lm_head_weight)
                teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1)).detach()

        if lk_active:
            lk_per_position = self._compute_lk_loss(
                student_logits=flat_logits,
                teacher_logits=teacher_logits_flat,
                loss_type=self.lk_loss_type,
                eta=self.lk_eta,
            )
            lk_loss = (
                (lk_per_position * flat_weights.float()).sum() / valid_token_count.float()
            ).to(base_loss.dtype)
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
            kl_loss = (
                (kl_per_position * flat_weights.float()).sum() / valid_token_count.float()
            ).to(base_loss.dtype)
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
            e2e_tv_loss, _accept_len = self._compute_e2e_tv_loss(
                student_logits_pb=flat_logits.view(bsz, n_blocks, self.block_size, vocab_size_e2e),
                teacher_logits_pb=teacher_logits_flat.view(
                    bsz, n_blocks, self.block_size, vocab_size_e2e
                ),
                valid_mask_pb=weight_mask,
            )
            e2e_tv_loss = e2e_tv_loss.to(base_loss.dtype)
            loss = loss + self.e2e_tv_loss_weight * e2e_tv_loss

        # 9d. Subclass extra-loss hook (DSpark confidence head; no-op for DFlash).
        loss, extra_components = self._compute_extra_loss(
            loss,
            flat_logits,
            teacher_logits_flat,
            flat_weights,
            valid_token_count,
            prev_token_ids,
            n_blocks,
        )

        # 9e. Optional gate-sparsity penalty (gated_sum layer-selection; default off).
        #     Added to the final total so it applies regardless of distillation.
        if self.gate_entropy_weight > 0 and hasattr(self.draft_model, "gate_entropy"):
            loss = loss + self.gate_entropy_weight * self.draft_model.gate_entropy()

        # 10. Accuracy (using binary mask without decay)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
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

        # Single registration point for all loss terms (for logging). Components
        # are the PURE objective-weighted means of each term (``ce`` and ``l1``
        # separately, not the ``ce_α·ce + l1_α·l1`` combination), so they read as
        # an interpretable decomposition; ``kl``/``lk`` are the distill terms; the
        # extra-loss hook merges its own named components (e.g. confidence_loss).
        # Add a term here + list its key in the trainer's
        # ``_extra_loss_component_keys`` — never change the tuple arity.
        ce_component = (ce_per_token * flat_weights).sum() / valid_token_count
        loss_components = {
            "ce_loss": ce_component.detach(),
            "kl_loss": kl_loss.detach(),
            "lk_loss": lk_loss.detach(),
            "e2e_tv_loss": e2e_tv_loss.detach(),
        }
        if l1_per_token is not None:
            loss_components["l1_loss"] = (
                (l1_per_token * flat_weights.float()).sum() / valid_token_count.float()
            ).detach()
        loss_components.update(extra_components)

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
        weight_mask = weight_mask * (pos_in_block > 0).float()
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
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
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

        for start in range(0, n_blocks, blocks_per_chunk):
            end = min(start + blocks_per_chunk, n_blocks)
            nb = end - start
            dh_chunk = dh[:, start:end].reshape(bsz, nb * bs, D)
            prev_chunk = prev_token_ids[:, start:end].reshape(bsz, nb * bs)
            logits_chunk = self._compute_draft_logits(dh_chunk, lm_head_weight, prev_chunk, nb)
            if self.fp32_lm_head:
                logits_chunk = logits_chunk.float()
            flat = logits_chunk.view(-1, logits_chunk.size(-1))
            tgt = target_ids[:, start:end].reshape(-1)
            ce = F.cross_entropy(flat, tgt, reduction="none")

            w = objective_weights[:, start:end].reshape(-1)
            loss_num = loss_num + (self.ce_loss_alpha * ce * w).sum()
            ce_comp_num = ce_comp_num + (ce * w).sum()

            with torch.no_grad():
                b = binary_eval_mask[:, start:end].reshape(-1)
                pred = flat.argmax(dim=-1)
                correct = (pred == tgt) & (b > 0.5)
                correct_num = correct_num + correct.sum().float()
                loss_pos_num = loss_pos_num + (ce.view(bsz, nb, bs) * b.view(bsz, nb, bs)).sum(
                    dim=(0, 1)
                )
                acc_pos_num = acc_pos_num + correct.view(bsz, nb, bs).float().sum(dim=(0, 1))
                if proposals is not None:
                    proposals[:, start:end] = pred.view(bsz, nb, bs)

        loss = loss_num / valid_token_count
        if self.gate_entropy_weight > 0 and hasattr(self.draft_model, "gate_entropy"):
            loss = loss + self.gate_entropy_weight * self.draft_model.gate_entropy()
        accuracy = correct_num / actual_token_count
        loss_per_position = loss_pos_num / count_per_pos
        acc_per_position = acc_pos_num / count_per_pos

        zero = torch.zeros((), device=device, dtype=loss.dtype)
        loss_components = {
            "ce_loss": (ce_comp_num / valid_token_count).detach(),
            "kl_loss": zero,
            "lk_loss": zero,
        }

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
