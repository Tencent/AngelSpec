"""TreeFlash draft model: DFlare backbone + DSpark Markov / confidence heads +
TreeFlash hidden-states correction.

``TreeflashDSparkDFlareDraftModel`` is a :class:`DFlareDraftModel` (the DFlash
variant with learnable per-layer target fusion) that additionally carries the
DSpark heads — a low-rank Markov logit bias, an optional accept-rate confidence
head, and the TreeFlash hidden-states correction (formula (1), zero-init
residual). It holds only the modules; the DFlare backbone forward is reused
unchanged, and the heads are applied by the :class:`DSparkModel` training wrapper
through its ``_compute_draft_logits`` / ``_compute_extra_loss`` hooks (which read
``draft_model.{markov_head, hidden_correction, confidence_head}`` generically).

Selected via ``DSparkConfig`` + ``model_arch == "dflare"`` (dispatched in
``AutoEagle3DraftModel.from_config`` / ``dspark_trainer._build_draft_model``).
"""

from typing import Optional

import torch.nn as nn

from angelspec.models.draft.dflare import DFlareDraftModel
from angelspec.models.draft.dspark import (
    AcceptRatePredictor,
    DSparkConfig,
    build_hidden_correction,
    build_markov_head,
)


class TreeflashDSparkDFlareDraftModel(DFlareDraftModel):
    config_class = DSparkConfig

    def __init__(self, config: DSparkConfig):
        super().__init__(config)

        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )

        self.markov_head = build_markov_head(config)

        # TreeFlash hidden-states correction (formula (1)); ``None`` when disabled.
        self.hidden_correction = build_hidden_correction(config)

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
