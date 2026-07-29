"""DFlash gated-sum drafter: learnable target-LAYER SELECTION over dense candidates.

Purpose: an experiment harness to discover WHICH target layers matter for the
DFlash context fusion. It extracts a dense candidate set (e.g. 16 layers) and
learns a single global softmax gate over them; reading the gate after training
ranks the layers so a standard 5-layer DFlash run can be trained on the top-k.

Why a weighted SUM, not DFlash's concat -> Linear (identifiability):
  A per-layer scalar gate placed BEFORE ``context_proj`` (Linear(T*D, D)) is
  losslessly absorbable — ``context_proj(concat(w_j * h_j)) = sum_j (w_j W_j) h_j``,
  so the Linear can rescale ``W_j`` to undo any ``w_j``. Gradient descent then
  leaves the gate near its init and does the selection inside the Linear, making
  the read-out uniform and meaningless. Collapsing the T layers into ONE vector by
  a weighted sum removes that freedom: the downstream projection can only absorb a
  global scale, not the inter-layer RELATIVE weights. This is exactly DFlare's
  ``layer_fusion_weights`` construction (dflare.py) — validated to read non-uniform
  with no sparsity pressure. Here the gate is a single global ``[T]`` vector (one
  ranking of the candidate layers) rather than DFlare's per-draft-layer ``[D, T]``.

Per-candidate RMSNorm BEFORE the sum is essential: it strips the raw-magnitude
differences between shallow/deep target hidden states so the gate compares them on
equal footing (otherwise a layer's norm scale, not its usefulness, drives its weight).

Selected via ``DFlashConfig.fusion_type == "gated_sum"`` (dispatched in
``DFlashTrainer._build_draft_model``); the default ``"concat_fc"`` is unchanged
DFlash. Reuses DFlashConfig, the DFlash decoder layers and ``forward`` unchanged —
only ``__init__`` and ``extract_context_feature`` are overridden, and the fused
output is a single ``[B, seq, D]`` tensor just like DFlash's context feature.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel, DFlashRMSNorm


class DFlashGatedDraftModel(DFlashDraftModel):
    """DFlash drafter with a learnable global gate over candidate target layers."""

    config_class = DFlashConfig

    def __init__(self, config):
        # Builds context_proj / context_norm / layers / embed_tokens as DFlash.
        super().__init__(config)

        T = self.num_target_layers
        target_hidden = getattr(config, "target_hidden_size", config.hidden_size)
        # The weighted sum stays in target_hidden dim and feeds the decoder layers'
        # attention (k_proj expects hidden_size), so the two must match. True for
        # Qwen3-4B (both 2560). A projection could relax this, but is unneeded here.
        if target_hidden != config.hidden_size:
            raise ValueError(
                "DFlashGatedDraftModel (fusion_type=gated_sum) requires "
                f"target_hidden_size == hidden_size, got {target_hidden} != {config.hidden_size}."
            )

        # Single global gate over the T candidate layers; zero init => uniform 1/T.
        self.layer_gate = nn.Parameter(torch.zeros(T))
        # Per-candidate RMSNorm to equalise per-layer magnitude before the sum.
        self.candidate_norms = nn.ModuleList(
            [DFlashRMSNorm(target_hidden, eps=config.rms_norm_eps) for _ in range(T)]
        )
        self.gate_temperature = float(getattr(config, "gate_temperature", 1.0))

        # Weighted sum does not need the concat projection; drop it to keep the
        # state_dict clean and avoid a dead param being sharded/optimized (as DFlare).
        if hasattr(self, "context_proj"):
            del self.context_proj

    def extract_context_feature(self, all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Per-candidate RMSNorm -> softmax-gated weighted sum -> single [B, seq, D].

        Called inside the FSDP2-wrapped forward, so ``layer_gate`` / candidate norm
        weights are already all-gathered to full plain tensors by FSDP's pre-forward
        hook (same as DFlash's context_norm / DFlare's layer_fusion_weights).
        """
        normed = [self.candidate_norms[i](h) for i, h in enumerate(all_hidden_states)]
        stacked = torch.stack(normed, dim=2)  # [B, seq, T, D]
        w = F.softmax(self.layer_gate / self.gate_temperature, dim=0)  # [T]
        fused = torch.einsum("t,bstd->bsd", w.to(stacked.dtype), stacked)  # [B, seq, D]
        return self.context_norm(fused)

    # ------------------------------------------------------------------
    # Read-out helpers (called from the trainer OUTSIDE the wrapped forward,
    # where root params are re-sharded DTensors). full_tensor() is a collective:
    # callers must invoke these on ALL ranks, then use the result on rank 0 only.
    # ------------------------------------------------------------------

    @staticmethod
    def _full(t: torch.Tensor) -> torch.Tensor:
        return t.full_tensor() if hasattr(t, "full_tensor") else t

    def gate_logits(self) -> torch.Tensor:
        """Raw gate logits [T] (pre-softmax) on CPU — for post-hoc diagnosis."""
        return self._full(self.layer_gate).detach().float().cpu()

    def gate_weights(self) -> torch.Tensor:
        """Softmax gate weights [T] on CPU — used to rank / select layers."""
        g = self._full(self.layer_gate)
        return F.softmax(g / self.gate_temperature, dim=0).detach().float().cpu()

    def candidate_rmsnorm_scales(self) -> torch.Tensor:
        """Per-candidate effective scale [T] = RMS(norm.weight) on CPU.

        The RMSNorm gain co-determines a layer's effective contribution together
        with the gate weight; useful to tie-break layers with near-equal gate mass.
        """
        scales = [self._full(n.weight).pow(2).mean().sqrt() for n in self.candidate_norms]
        return torch.stack(scales).detach().float().cpu()

    def gate_entropy(self) -> torch.Tensor:
        """Differentiable entropy H(softmax(gate)) — for the optional sparsity penalty."""
        w = F.softmax(self.layer_gate / self.gate_temperature, dim=0)
        return -(w * (w + 1e-12).log()).sum()


__all__ = ["DFlashGatedDraftModel"]
