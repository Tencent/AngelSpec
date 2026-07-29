"""DSpark training model: DFlash training wrapper + Markov / confidence heads.

:class:`DSparkModel` reuses the **entire** :class:`DFlashModel` training forward
— anchor sampling, block-causal FlexAttention mask, MASK-token noise, and the
full CE / KL / LK / L1 / D-PACE loss pipeline (with the same label alignment and
loss knobs). The only additions ride on the frozen DFlash subclass hooks so none
of the DFlash loss is re-implemented:

  - ``_compute_draft_logits``: applies the TreeFlash hidden-states correction
    (``h' = h + SwiGLU(norm(h) :: norm(e_prev))``) before the LM head and the
    Markov logit bias after it, both conditioned on the teacher-forced previous
    token. Zero-init keeps these no-ops at start (degenerates to DFlash).
  - ``_compute_extra_loss``: adds a confidence-head BCE against the empirical
    per-token accept rate ``1 - 0.5 * L1(draft, teacher)`` (+ any
    position-adaptive alpha smoothness penalty), surfaced as ``confidence_loss``.

Total loss = ``<DFlash loss> + confidence_head_alpha * confidence`` where
``<DFlash loss>`` is the objective selected by the shared knobs
(``loss_objective`` / ``ce_loss_alpha`` / ``l1_loss_alpha`` / ``kl_*`` / ``lk_*``).
"""

import torch
import torch.nn.functional as F

from angelspec.models.dflash import DFlashModel


class DSparkModel(DFlashModel):
    """DSpark training wrapper (DFlash backbone + Markov / confidence heads)."""

    def __init__(
        self,
        draft_model,
        block_size: int = 7,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.0,
        kl_loss_weight: float = 0.0,
        kl_temperature: float = 1.0,
        kl_topk: int = 10,
        kl_topk_renormalize: bool = True,
        lk_loss_weight: float = 0.0,
        lk_loss_type: str = "hybrid",
        lk_eta: float = 3.0,
        lk_temperature: float = 1.0,
        fp32_lm_head: bool = True,
        gate_entropy_weight: float = 0.0,
        confidence_head_alpha: float = 1.0,
    ):
        # Forward the full DFlash loss configuration to the parent so DSpark's
        # base loss is identical to DFlash's; ``confidence`` is the only
        # DSpark-specific term (added in ``_compute_extra_loss``).
        super().__init__(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            fp32_lm_head=fp32_lm_head,
            gate_entropy_weight=gate_entropy_weight,
            loss_objective=loss_objective,
            dpace_alpha=dpace_alpha,
            ce_loss_alpha=ce_loss_alpha,
            l1_loss_alpha=l1_loss_alpha,
            kl_loss_weight=kl_loss_weight,
            kl_temperature=kl_temperature,
            kl_topk=kl_topk,
            kl_topk_renormalize=kl_topk_renormalize,
            lk_loss_weight=lk_loss_weight,
            lk_loss_type=lk_loss_type,
            lk_eta=lk_eta,
            lk_temperature=lk_temperature,
        )
        self.confidence_head_alpha = float(confidence_head_alpha)
        # Handoff buffer for the corrected draft hidden states between the
        # ``_compute_draft_logits`` and ``_compute_extra_loss`` hooks within a
        # single forward pass.
        self._dspark_hidden_4d = None

    # ------------------------------------------------------------------
    # DFlash subclass hooks
    # ------------------------------------------------------------------

    def _compute_draft_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ) -> torch.Tensor:
        """Inject hidden-states correction + Markov bias, then project to logits.

        ``prev_token_ids`` is ``[B, n_blocks, block_size]`` — the ground-truth
        token immediately preceding each draft slot's target (aligned
        slot-for-slot with the flattened ``draft_hidden`` layout).
        """
        bsz = draft_hidden.size(0)

        # Hidden-states correction (TreeFlash formula (1)), BEFORE the LM head so
        # the token distribution is conditioned on the previous token.
        if getattr(self.draft_model, "hidden_correction", None) is not None:
            prev_embeds = self.draft_model.embed_tokens(prev_token_ids)
            prev_embeds = prev_embeds.view(bsz, -1, prev_embeds.size(-1))
            draft_hidden = self.draft_model.hidden_correction(draft_hidden, prev_embeds)

        # Cache the corrected hidden states for the confidence head.
        self._dspark_hidden_4d = draft_hidden.view(bsz, n_blocks, self.block_size, -1)

        logits = F.linear(draft_hidden, lm_head_weight)

        # Markov-biased draft logits (teacher-forced previous token).
        if getattr(self.draft_model, "markov_head", None) is not None:
            logits_4d = self.draft_model.markov_head.apply_block_logits(
                logits.view(bsz, n_blocks, self.block_size, -1),
                token_ids=prev_token_ids,
            )
            logits = logits_4d.reshape(bsz, n_blocks * self.block_size, -1)

        return logits

    def _extra_distill_needed(self) -> bool:
        """Confidence head needs the teacher logits (for the accept-rate target)."""
        return (
            getattr(self.draft_model, "confidence_head", None) is not None
            and self.confidence_head_alpha > 0
        )

    def _compute_extra_loss(
        self,
        loss: torch.Tensor,
        flat_logits: torch.Tensor,
        teacher_logits_flat,
        flat_weights: torch.Tensor,
        valid_token_count: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ):
        """Add the confidence-head BCE against the empirical accept rate.

        Uses the SAME objective-weighted validity mask (``flat_weights``) and
        weighted-mean reduction as the DFlash CE / distillation terms. Returns
        ``(loss, {"confidence_loss": ...})`` so the component is logged via the
        shared ``loss_components`` mechanism.
        """
        confidence_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)

        # Position-adaptive alpha smoothness regularizer (independent of the
        # confidence head / teacher logits): ``lambda * sum_i (alpha_i - alpha_{i-1})^2``
        # over any head (Markov / hidden-correction) that enables it.
        for _head in (
            getattr(self.draft_model, "markov_head", None),
            getattr(self.draft_model, "hidden_correction", None),
        ):
            _pa = getattr(_head, "pos_alpha", None) if _head is not None else None
            if _pa is not None:
                _reg = _pa.smooth_loss()
                if _reg is not None:
                    loss = loss + _reg.to(loss.dtype)

        # Confidence BCE needs the teacher accept-rate target; skip when the head
        # is off or target last_hidden_states weren't delivered this step.
        if not self._extra_distill_needed() or teacher_logits_flat is None:
            return loss, {"confidence_loss": confidence_loss.detach()}

        # accept_rate = 1 - 0.5 * L1(draft, teacher)  in [0, 1], per position.
        l1_per_position = self._compute_l1_loss(flat_logits, teacher_logits_flat)  # [N]
        accept_rate = (1.0 - 0.5 * l1_per_position).clamp(0.0, 1.0)

        hidden_4d = self._dspark_hidden_4d
        if getattr(self.draft_model, "confidence_head_with_markov", False):
            prev_emb = self.draft_model.markov_head.get_prev_embeddings(prev_token_ids).to(
                hidden_4d.dtype
            )
            conf_features = torch.cat([hidden_4d, prev_emb], dim=-1)
        else:
            conf_features = hidden_4d

        confidence_pred = self.draft_model.confidence_head(conf_features).float().reshape(-1)
        conf_bce = F.binary_cross_entropy_with_logits(
            confidence_pred, accept_rate.detach(), reduction="none"
        )
        confidence_loss = ((conf_bce * flat_weights.float()).sum() / valid_token_count.float()).to(
            loss.dtype
        )

        loss = loss + self.confidence_head_alpha * confidence_loss
        return loss, {"confidence_loss": confidence_loss.detach()}
