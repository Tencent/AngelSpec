"""DFly trainer — DFlashTrainer with DFly's draft model and training wrapper.

Reuses the whole DFlash pipeline (FSDP init, optimizer, checkpoint, schedule,
forward/metrics/eval) and overrides only the two model-build seams: builds the
DFly draft model (DFlash FC context + DFlare fusion residual) and wraps it in
``DFlyModel``. The hidden-states correction rides the shared DFlash wrapper hook.
Reads the ``dflash_*`` hyperparameter namespace.
"""

from angelspec.training.dflash_trainer import DFlashTrainer


class DFlyTrainer(DFlashTrainer):
    """DFly-specific trainer (DFlash backbone + DFlare fusion residual)."""

    def _build_draft_model(self, config):
        from angelspec.models.draft.dfly import DFlyDraftModel

        model = DFlyDraftModel(config)
        model.context_chunk_size = int(getattr(self.args, "dfly_context_chunk_size", 0))
        model.context_checkpoint = bool(getattr(self.args, "dfly_context_checkpoint", False))
        if model.context_chunk_size < 0:
            raise ValueError("dfly_context_chunk_size must be nonnegative")
        return model

    def _build_training_wrapper(self, draft_model):
        import torch.distributed as dist

        from angelspec.models.dfly import DFlyModel

        usp = getattr(self.args, "attention_backend", None) == "usp"
        sp_size = int(getattr(self.args, "sp_ulysses_size", 1)) if usp else 1
        sp_rank = dist.get_rank(self.sp_group) if self.sp_group is not None else 0
        causal_attn = self.causal_attn
        self._log_usp_model_config(sp_size=sp_size, sp_rank=sp_rank, causal_attn=causal_attn)
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
            e2e_tv_vocab_chunk=self.e2e_tv_vocab_chunk,
            loss_checkpoint_chunks=getattr(self.args, "dflash_loss_checkpoint_chunks", False),
            path_selector_loss_alpha=self.path_selector_loss_alpha,
            path_selector_teacher_labels=self.path_selector_teacher_labels,
            path_selector_objective_weighting=self.path_selector_objective_weighting,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=self.sp_group,
            causal_attn=causal_attn,
        )
