"""DSpark draft model: DFlash backbone + EAGLE-style Markov and confidence heads.

Shares DFlash's block-diffusion drafter and adds two heads:

  - Markov head: low-rank learned bigram logit bias on the (teacher-forced)
    previous token; refines the per-token distribution without touching the
    backbone.
  - Confidence head (AcceptRatePredictor): predicts per-position acceptance
    probability, trained against the empirical accept rate (used at inference
    for adaptive block length).
"""

from typing import Optional

import torch
import torch.nn as nn

from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel


class DSparkConfig(DFlashConfig):
    """Config for the DSpark draft model. Extends :class:`DFlashConfig`."""

    model_type = "dspark"

    def __init__(
        self,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov


class VanillaMarkov(nn.Module):
    """Low-rank learned bigram (Markov) logit bias on the previous token."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        assert (
            self.markov_rank > 0
        ), f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        # TODO: markov_w2 out_features should match "draft_vocab_size" if pruning is used.
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        bias = self.compute_step_bias(token_ids)
        return base_logits + bias


class AcceptRatePredictor(nn.Module):
    """Per-position acceptance-probability head (linear projection to a logit)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None

    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkov(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
        )
    raise NotImplementedError(
        f"markov_head_type={markov_head_type!r} is not supported yet; only 'vanilla' "
        "is implemented in AngelSpec as it is recommended by the authors."
    )


class DSparkDraftModel(DFlashDraftModel):
    config_class = DSparkConfig

    def __init__(self, config: DSparkConfig):
        super().__init__(config)

        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )

        self.markov_head = build_markov_head(config)

        self.confidence_head: Optional[nn.Module] = None
        if getattr(config, "enable_confidence_head", False):
            conf_input_dim = self.hidden_size
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "confidence_head_with_markov=True requires a Markov head (markov_rank > 0)."
                    )
                conf_input_dim += self.markov_rank
            self.confidence_head = AcceptRatePredictor(conf_input_dim)
