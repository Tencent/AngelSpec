from typing import Optional

import torch
import torch.nn.functional as F

from angelspec.models.dflash import (
    _DFLASH_E2E_TV_VOCAB_CHUNK_DEFAULT,
    DFlashModel,
    _dflash_e2e_tv_vocab_chunk,
)


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
        kl_topk: int = 10,
        lk_loss_weight: float = 0.0,
        lk_loss_type: str = "hybrid",
        lk_eta: float = 3.0,
        e2e_tv_loss_weight: float = 0.0,
        e2e_tv_vocab_chunk: Optional[int] = None,
        path_selector_loss_alpha: float = 1.0,
        path_selector_teacher_labels: bool = True,
        path_selector_objective_weighting: bool = True,
        fp32_lm_head: bool = True,
        gate_entropy_weight: float = 0.0,
        confidence_head_alpha: float = 1.0,
        confidence_detach: bool = False,
        causal_attn: bool = False,
        loss_checkpoint_chunks: bool = False,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group=None,
    ):
        # Forward the full DFlash loss config to the parent; ``confidence`` is
        # DSpark's only extra term (added in ``_compute_extra_loss``).
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
            kl_topk=kl_topk,
            lk_loss_weight=lk_loss_weight,
            lk_loss_type=lk_loss_type,
            lk_eta=lk_eta,
            e2e_tv_loss_weight=e2e_tv_loss_weight,
            e2e_tv_vocab_chunk=e2e_tv_vocab_chunk,
            path_selector_loss_alpha=path_selector_loss_alpha,
            path_selector_teacher_labels=path_selector_teacher_labels,
            path_selector_objective_weighting=path_selector_objective_weighting,
            causal_attn=causal_attn,
            loss_checkpoint_chunks=loss_checkpoint_chunks,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
        )
        self.confidence_head_alpha = float(confidence_head_alpha)
        # Stops the confidence BCE at the head, so the drafter trains on the DFlash
        # loss alone.
        self.confidence_detach = bool(confidence_detach)
        # Handoff buffer for corrected draft hidden states, between the
        # ``_compute_draft_logits`` and ``_compute_extra_loss`` hooks.
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
        block_start: int = 0,
    ) -> torch.Tensor:
        """Inject hidden-states correction + Markov bias, then project to logits.

        ``prev_token_ids`` is ``[B, n_blocks, block_size]`` — the ground-truth
        token preceding each draft slot's target (aligned with ``draft_hidden``).
        """
        bsz = draft_hidden.size(0)
        draft_hidden = self._apply_hidden_correction(draft_hidden, prev_token_ids, block_start)

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

    @torch.no_grad()
    def _accept_rate(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Per-position ``Σ_v min(p_v, q_v) = 1 - 0.5·L1(p, q)`` in [0, 1]. Returns [N].

        The target is detached, so it is streamed over the vocab like the e2e-TV
        term: only one chunk of full-vocab fp32 probabilities is alive at a time.
        """
        vocab_chunk = self.e2e_tv_vocab_chunk
        if vocab_chunk is None or vocab_chunk <= 0:
            vocab_chunk = _dflash_e2e_tv_vocab_chunk() or _DFLASH_E2E_TV_VOCAB_CHUNK_DEFAULT
        t_lse = torch.logsumexp(teacher_logits.float(), dim=-1, keepdim=True)
        s_lse = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
        rate = torch.zeros(
            student_logits.shape[:-1], dtype=torch.float32, device=student_logits.device
        )
        for start in range(0, student_logits.size(-1), vocab_chunk):
            end = start + vocab_chunk
            p = torch.exp(teacher_logits[..., start:end].float() - t_lse)
            q = torch.exp(student_logits[..., start:end].float() - s_lse)
            rate += torch.minimum(p, q).sum(dim=-1)
        return rate.clamp_(0.0, 1.0)

    def _compute_extra_loss(
        self,
        loss: torch.Tensor,
        flat_logits: torch.Tensor,
        teacher_logits_flat,
        flat_weights: torch.Tensor,
        loss_denom: torch.Tensor,
        prev_token_ids: torch.Tensor,
        n_blocks: int,
    ):
        """Add the confidence-head BCE against the empirical accept rate.

        Uses the same objective-weighted mask (``flat_weights``) and denominator
        (``loss_denom``, SP-global under USP) as the DFlash CE / distillation terms.
        The logged ``confidence_loss`` / ``confidence_mae`` are means over this rank's
        tokens, like ``ce_loss``, since the trainer averages logged components over
        ranks. Returns ``(loss, {"confidence_loss": ...})`` for shared logging.
        """
        confidence_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)

        # Confidence BCE needs the teacher accept-rate target; skip when the head
        # is off or target last_hidden_states weren't delivered this step.
        if not self._extra_distill_needed() or teacher_logits_flat is None:
            return loss, {"confidence_loss": confidence_loss.detach()}

        accept_rate = self._accept_rate(flat_logits, teacher_logits_flat)  # [N]

        hidden_4d = self._dspark_hidden_4d
        if getattr(self.draft_model.config, "confidence_on_prenorm_hidden", False):
            hidden_4d = self.draft_model.confidence_hidden.reshape(hidden_4d.shape)
        if getattr(self.draft_model, "confidence_head_with_markov", False):
            prev_emb = self.draft_model.markov_head.get_prev_embeddings(prev_token_ids).to(
                hidden_4d.dtype
            )
            conf_features = torch.cat([hidden_4d, prev_emb], dim=-1)
        else:
            conf_features = hidden_4d
        if self.confidence_detach:
            conf_features = conf_features.detach()

        confidence_pred = self.draft_model.confidence_head(conf_features).float().reshape(-1)
        conf_bce = F.binary_cross_entropy_with_logits(
            confidence_pred, accept_rate.detach(), reduction="none"
        )
        weights = flat_weights.float()
        conf_sum = (conf_bce * weights).sum()
        local_denom = weights.sum().clamp(min=1e-6)
        with torch.no_grad():
            # BCE on a soft target has an entropy floor; the absolute error reads
            # calibration directly.
            confidence_mae = (
                (confidence_pred.sigmoid() - accept_rate).abs() * weights
            ).sum() / local_denom

        loss = loss + self.confidence_head_alpha * (conf_sum / loss_denom.float()).to(loss.dtype)
        return loss, {
            "confidence_loss": (conf_sum.detach() / local_denom).to(loss.dtype),
            "confidence_mae": confidence_mae,
        }
