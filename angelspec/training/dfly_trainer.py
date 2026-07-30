"""DFly trainer — DFlashTrainer with DFly's draft model and training wrapper.

Reuses the whole DFlash pipeline (FSDP init, optimizer, checkpoint, schedule,
forward/metrics/eval) and overrides only the two model-build seams: builds the
DFly draft model (DFlash FC context + DFlare fusion residual) and wraps it in
``DFlyModel`` for the optional hidden-states correction. Reads the ``dflash_*``
hyperparameter namespace.
"""

from angelspec.training.dflash_trainer import DFlashTrainer


class DFlyTrainer(DFlashTrainer):
    """DFly-specific trainer (DFlash backbone + hidden-states correction)."""

    def _build_draft_model(self, config):
        from angelspec.models.draft.dfly import DFlyDraftModel

        return DFlyDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        from angelspec.models.dfly import DFlyModel

        return DFlyModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_decay_gamma=self.loss_decay_gamma,
            fp32_lm_head=self.fp32_lm_head,
            gate_entropy_weight=getattr(self.args, "dflash_gate_entropy_weight", 0.0),
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
        )
