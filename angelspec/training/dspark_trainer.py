"""DSpark trainer — DFlash trainer + Markov / confidence heads.

Subclasses DFlashTrainer, reusing its whole pipeline. Swaps in the DSpark draft
model and the ``DSparkModel`` wrapper through the ``_build_draft_model`` /
``_build_training_wrapper`` seams, and extends ``_extra_loss_component_keys``
with ``confidence_loss``. Heads inject through the shared DFlash hooks, so
loss / metric plumbing is inherited unchanged.
"""

from argparse import Namespace

import torch.distributed as dist

from angelspec.models.draft.dfly2 import DFly2Config, DFly2DraftModel
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.dspark import DSparkModel
from angelspec.training.dflash_trainer import DFlashTrainer


class DSparkTrainer(DFlashTrainer):
    """DSpark-specific trainer (DFlash backbone + EAGLE-style heads)."""

    # DSpark reuses the unified DFlash loss (ce/kl/lk/l1 via loss_components) and
    # adds the confidence-head term.
    _extra_loss_component_keys = [
        "ce_loss",
        "kl_loss",
        "lk_loss",
        "l1_loss",
        "e2e_tv_loss",
        "confidence_loss",
        "confidence_mae",
        # Path selector; only emitted when the draft config enables it.
        "selector_loss",
        "selector_recall",
        "selector_acc",
    ]

    def __init__(self, args: Namespace):
        super().__init__(args)
        # A dspark_* knob that is set overrides the dflash_* value the parent read;
        # unset (None), the run keeps the DFlash one.
        for attr in (
            "num_anchors",
            "num_target_layers",
            "loss_decay_gamma",
            "ce_loss_alpha",
            "l1_loss_alpha",
        ):
            value = getattr(args, f"dspark_{attr}", None)
            if value is not None:
                setattr(self, attr, value)
        self.confidence_head_alpha = getattr(args, "dspark_confidence_head_alpha", 1.0)
        self.confidence_detach = bool(getattr(args, "dspark_confidence_detach", False))

    def init_model(
        self,
        draft_model_config,
        target_model_path: str,
        mooncake_config=None,
    ) -> int:
        # The parent parses str/dict as a plain DFlashConfig; resolve DSparkConfig
        # here so the head flags are typed config fields.
        if isinstance(draft_model_config, str):
            draft_model_config = DSparkConfig.from_pretrained(draft_model_config)
        elif isinstance(draft_model_config, dict):
            draft_model_config = DSparkConfig(**draft_model_config)
        elif not isinstance(draft_model_config, DSparkConfig):
            raise TypeError(
                f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                f"Expected str, dict, or DSparkConfig."
            )
        return super().init_model(draft_model_config, target_model_path, mooncake_config)

    def _build_draft_model(self, config):
        if isinstance(config, DFly2Config):
            return DFly2DraftModel(config)
        return DSparkDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        usp = getattr(self.args, "attention_backend", None) == "usp"
        sp_size = int(getattr(self.args, "sp_ulysses_size", 1)) if usp else 1
        sp_rank = dist.get_rank(self.sp_group) if self.sp_group is not None else 0
        self._log_usp_model_config(sp_size=sp_size, sp_rank=sp_rank, causal_attn=self.causal_attn)
        return DSparkModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_decay_gamma=self.loss_decay_gamma,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            kl_loss_weight=self.kl_loss_weight,
            kl_topk=self.kl_topk,
            lk_loss_weight=self.lk_loss_weight,
            lk_loss_type=self.lk_loss_type,
            lk_eta=self.lk_eta,
            e2e_tv_loss_weight=self.e2e_tv_loss_weight,
            e2e_tv_vocab_chunk=self.e2e_tv_vocab_chunk,
            path_selector_loss_alpha=self.path_selector_loss_alpha,
            path_selector_teacher_labels=self.path_selector_teacher_labels,
            path_selector_objective_weighting=self.path_selector_objective_weighting,
            fp32_lm_head=self.fp32_lm_head,
            gate_entropy_weight=getattr(self.args, "dflash_gate_entropy_weight", 0.0),
            confidence_head_alpha=self.confidence_head_alpha,
            confidence_detach=self.confidence_detach,
            causal_attn=self.causal_attn,
            loss_checkpoint_chunks=getattr(self.args, "dflash_loss_checkpoint_chunks", False),
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=self.sp_group,
        )
