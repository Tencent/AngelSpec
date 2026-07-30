"""DSpark trainer — DFlash trainer + Markov / confidence heads.

Subclasses DFlashTrainer, reusing its whole pipeline. Overrides ``init_model``
to swap in the DSpark draft model and the ``DSparkModel`` wrapper, and extends
``_extra_loss_component_keys`` with ``confidence_loss``. Heads inject through the
shared DFlash hooks, so loss / metric plumbing is inherited unchanged.
"""

from argparse import Namespace

import torch
import torch.distributed as dist

from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.dspark import DSparkModel
from angelspec.training import checkpoint
from angelspec.training.dflash_trainer import DFlashTrainer
from angelspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from angelspec.training.optimizer import BF16Optimizer
from angelspec.utils.distributed import get_gloo_group
from angelspec.utils.logging import logger


class DSparkTrainer(DFlashTrainer):
    """DSpark-specific trainer (DFlash backbone + EAGLE-style heads)."""

    # DSpark reuses the unified DFlash loss (ce/kl/lk/l1 via loss_components) and
    # adds the confidence-head term.
    _extra_loss_component_keys = ["ce_loss", "kl_loss", "lk_loss", "l1_loss", "confidence_loss"]

    def __init__(self, args: Namespace):
        super().__init__(args)
        # DSpark uses its own knobs; override the dflash_* defaults read by the
        # parent so the shared machinery picks them up.
        self.block_size = getattr(args, "dflash_block_size", 7)
        self.num_anchors = getattr(args, "dspark_num_anchors", 512)
        self.num_target_layers = getattr(args, "dspark_num_target_layers", 5)
        self.loss_decay_gamma = getattr(args, "dspark_loss_decay_gamma", 4.0)
        self.ce_loss_alpha = getattr(args, "dspark_ce_loss_alpha", 0.1)
        self.l1_loss_alpha = getattr(args, "dspark_l1_loss_alpha", 0.9)
        self.confidence_head_alpha = getattr(args, "dspark_confidence_head_alpha", 1.0)

    # ------------------------------------------------------------------
    # Model init (copied from DFlashTrainer.init_model; the two model-build
    # sites are swapped for DSpark. This is the B-migration seam — see header.)
    # ------------------------------------------------------------------

    def init_model(
        self,
        draft_model_config,
        target_model_path: str,
        mooncake_config=None,
    ) -> int:
        if mooncake_config is not None:
            from angelspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address, mooncake_config.metadata_server
            )

        init_context = self._get_init_weight_context_manager()

        with init_context():
            if isinstance(draft_model_config, DSparkConfig):
                config = draft_model_config
            elif isinstance(draft_model_config, str):
                config = DSparkConfig.from_pretrained(draft_model_config)
            elif isinstance(draft_model_config, dict):
                config = DSparkConfig(**draft_model_config)
            else:
                raise TypeError(
                    f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                    f"Expected str, dict, or DSparkConfig."
                )

            if not hasattr(config, "num_target_layers") or config.num_target_layers is None:
                config.num_target_layers = self.num_target_layers
            if not hasattr(config, "target_hidden_size") or config.target_hidden_size is None:
                config.target_hidden_size = config.hidden_size
            if (
                not hasattr(config, "target_num_hidden_layers")
                or config.target_num_hidden_layers is None
            ):
                from transformers import AutoConfig

                target_config = AutoConfig.from_pretrained(
                    target_model_path,
                    trust_remote_code=getattr(self.args, "trust_remote_code", True),
                )
                config.target_num_hidden_layers = target_config.num_hidden_layers

            # --- B-seam: swapped model construction ---
            draft_model = DSparkDraftModel(config)

        if dist.get_rank() == 0:
            draft_model.load_embedding(
                target_model_path,
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens.weight"),
            )

        draft_model.freeze_embedding()
        draft_model = draft_model.to(torch.bfloat16)

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] DSpark draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen (embedding) parameters"
        )

        # --- B-seam: swapped training wrapper construction ---
        dspark_model = DSparkModel(
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
            fp32_lm_head=self.fp32_lm_head,
            gate_entropy_weight=getattr(self.args, "dflash_gate_entropy_weight", 0.0),
            confidence_head_alpha=self.confidence_head_alpha,
        )

        full_state = dspark_model.state_dict() if dist.get_rank() == 0 else {}

        dspark_model = apply_fsdp2(
            dspark_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=list(draft_model.layers),
        )

        dspark_model = fsdp2_load_full_state_dict(
            dspark_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        if getattr(self.args, "compile_model", False):
            logger.info("Compiling DSpark model with torch.compile (inductor backend)")
            dspark_model = torch.compile(dspark_model)

        self.model = dspark_model
        _unwrapped = getattr(self.model, "_orig_mod", self.model)  # torch.compile
        self.dflash = getattr(_unwrapped, "module", _unwrapped)  # DDP/replicate
        self.draft_model = self.dflash.draft_model

        total_steps = self.args.lr_total_steps
        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.1)

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=warmup_ratio,
            total_steps=total_steps,
            decay_style=decay_style if decay_style != "WSD" else "cosine",
            min_lr=getattr(self.args, "min_lr", 0.0),
        )

        if decay_style == "WSD" and total_steps:
            from angelspec.training.lr_scheduler import LRSchedulerWithWarmup

            wsd_ratio = getattr(self.args, "wsd_decay_ratio", 0.2)
            self.optimizer.scheduler = LRSchedulerWithWarmup(
                self.optimizer.optimizer,
                max_lr=self.args.learning_rate,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
                decay_style="WSD",
                min_lr=getattr(self.args, "min_lr", 0.0),
                wsd_decay_steps=int(wsd_ratio * total_steps),
                wsd_decay_style=getattr(self.args, "wsd_decay_style", "cosine"),
            )

        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        self._init_target_lm_head(target_model_path)

        if self.target_lm_head is None:
            raise ValueError(
                "target_lm_head is required but was None. Ensure _init_target_lm_head succeeded."
            )
        self.target_lm_head_weight = self.target_lm_head.lm_head.weight

        self.prof.on_init_end()

        logger.info(f"[Rank {self.dp_rank}] DSpark model initialized with FSDP2")

        return 0
