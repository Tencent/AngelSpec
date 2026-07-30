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

"""DFlash trainer — extends Trainer with DFlash-specific model init, forward, and metrics."""

import time
from argparse import Namespace
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import DFlashDraftModel
from angelspec.training import checkpoint
from angelspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from angelspec.training.optimizer import BF16Optimizer
from angelspec.training.trainer import Trainer
from angelspec.utils.distributed import get_gloo_group
from angelspec.utils.logging import logger


class DFlashTrainer(Trainer):
    """DFlash-specific trainer.

    Extends ``Trainer`` with DFlash model initialisation (dual-source KV draft model),
    forward/backward with anchor sampling + block-causal mask, and metric aggregation.
    """

    # Per-component loss scalars surfaced from DFlashModel.forward's
    # ``loss_components`` dict; reduced to global means for logging (each key is
    # reduced only when present, so subclasses just extend this list). Subclasses
    # (DSpark) add e.g. "confidence_loss".
    _extra_loss_component_keys: list[str] = [
        "ce_loss",
        "kl_loss",
        "lk_loss",
        "l1_loss",
        "e2e_tv_loss",
    ]

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head: Optional[torch.nn.Module] = None
        self.num_target_layers = getattr(args, "dflash_num_target_layers", 5)
        self.block_size = getattr(args, "dflash_block_size", 16)
        self.num_anchors = getattr(args, "dflash_num_anchors", 512)
        self.loss_decay_gamma = getattr(args, "dflash_loss_decay_gamma", 7.0)
        self.fp32_lm_head = getattr(args, "dflash_fp32_lm_head", True)
        # Unified loss architecture: objective (decay / D-PACE) + optional
        # CE-side distillation (L1 / top-K KL / LK). Defaults reproduce the
        # legacy decay CE baseline (ce_loss_alpha=1.0, all distill weights 0).
        self.loss_objective = getattr(args, "dflash_loss_objective", "decay")
        self.dpace_alpha = getattr(args, "dflash_dpace_alpha", 0.5)
        self.ce_loss_alpha = getattr(args, "dflash_ce_loss_alpha", 1.0)
        self.l1_loss_alpha = getattr(args, "dflash_l1_loss_alpha", 0.0)
        self.kl_loss_weight = float(getattr(args, "dflash_kl_loss_weight", 0.0))
        self.kl_topk = int(getattr(args, "dflash_kl_topk", 10))
        self.lk_loss_weight = float(getattr(args, "dflash_lk_loss_weight", 0.0))
        self.lk_loss_type = str(getattr(args, "dflash_lk_loss_type", "hybrid"))
        self.lk_eta = float(getattr(args, "dflash_lk_eta", 3.0))
        # Independent e2e multi-step TV loss (added on top; not KL/LK-exclusive).
        self.e2e_tv_loss_weight = float(getattr(args, "dflash_e2e_tv_loss_weight", 0.0))
        self._lk_enabled = self.lk_loss_weight > 0.0
        self._kl_enabled = (not self._lk_enabled) and self.kl_loss_weight > 0.0
        # last_hidden_states (target final norm) is required for KL/LK/e2e_tv
        # teacher logits; L1 uses raw last_hidden_states directly (no norm).
        self._distill_enabled = (
            self._lk_enabled or self._kl_enabled or self.e2e_tv_loss_weight > 0.0
        )
        # Rolling window of the top-5 candidate-layer set for the gated_sum layer
        # selection run; drives the topk_jaccard / backbone_size early-stop signals.
        self._gate_topk_window: deque = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Model-build seams (overridable by DFlash-family subclasses)
    # ------------------------------------------------------------------

    def _build_draft_model(self, config):
        """Construct the draft model from ``config`` (dispatch by config flags).

        Subclasses (e.g. DFly) override this to build their own draft model.
        """
        if getattr(config, "fusion_type", "concat_fc") == "gated_sum":
            from angelspec.models.draft.dflash_gated import DFlashGatedDraftModel

            return DFlashGatedDraftModel(config)
        elif getattr(config, "model_arch", "dflash") == "dflare":
            from angelspec.models.draft.dflare import DFlareDraftModel

            return DFlareDraftModel(config)
        return DFlashDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        """Wrap ``draft_model`` in the training module (loss / forward plumbing).

        Subclasses (e.g. DFly) override this to swap in a wrapper that injects
        their extra behavior through the shared DFlash hooks.
        """
        return DFlashModel(
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
            from angelspec.models.draft.dflash import DFlashConfig

            if isinstance(draft_model_config, str):
                config = DFlashConfig.from_pretrained(draft_model_config)
            elif isinstance(draft_model_config, dict):
                config = DFlashConfig(**draft_model_config)
            elif isinstance(draft_model_config, DFlashConfig):
                config = draft_model_config
            else:
                raise TypeError(
                    f"Unsupported draft_model_config type: {type(draft_model_config).__name__}. "
                    f"Expected str, dict, or DFlashConfig."
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

            draft_model = self._build_draft_model(config)

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
            f"[Rank {self.dp_rank}] DFlash draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen (embedding) parameters"
        )

        dflash_model = self._build_training_wrapper(draft_model)

        full_state = dflash_model.state_dict() if dist.get_rank() == 0 else {}

        dflash_model = apply_fsdp2(
            dflash_model,
            mesh=self.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=list(draft_model.layers),
        )

        dflash_model = fsdp2_load_full_state_dict(
            dflash_model,
            full_state,
            self.dp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        if getattr(self.args, "compile_model", False):
            logger.info("Compiling DFlash model with torch.compile (inductor backend)")
            dflash_model = torch.compile(dflash_model)

        self.model = dflash_model
        # Unwrap torch.compile and/or DDP module wrappers to access underlying DFlashModel
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
            betas=(
                getattr(self.args, "adam_beta1", 0.9),
                getattr(self.args, "adam_beta2", 0.999),
            ),
            optimizer_type=getattr(self.args, "optimizer_type", "adamw"),
            muon_momentum=getattr(self.args, "muon_momentum", 0.95),
            muon_nesterov=getattr(self.args, "muon_nesterov", True),
            muon_ns_steps=getattr(self.args, "muon_ns_steps", 5),
            muon_matched_adamw_rms=getattr(self.args, "muon_matched_adamw_rms", 0.2),
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

        logger.info(f"[Rank {self.dp_rank}] DFlash model initialized with FSDP2")

        return 0

    # ------------------------------------------------------------------
    # Target LM head (same as Eagle3Trainer)
    # ------------------------------------------------------------------

    def _init_target_lm_head(self, target_model_path: str) -> None:
        from angelspec.models.target.target_utils import TargetLMHead

        # KL/LK distillation needs the target's final RMSNorm to normalise the
        # (pre-norm) last_hidden_states from mooncake before the lm_head. Skipped
        # for the base CE path (load_norm=False → norm=None → target_norm=None).
        load_norm = self._distill_enabled

        if dist.get_rank() == 0:
            self.target_lm_head = TargetLMHead.from_pretrained(
                model_path=target_model_path,
                lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
                norm_key=getattr(self.args, "norm_key", "model.norm.weight"),
                load_norm=load_norm,
                device="cuda",
                dtype=torch.bfloat16,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            logger.info(
                f"[Rank 0] TargetLMHead loaded from {target_model_path} "
                f"(load_norm={load_norm}, norm={'yes' if self.target_lm_head.norm is not None else 'no'})"
            )
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                target_model_path,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            self.target_lm_head = TargetLMHead(config)
            if load_norm:
                # Build the empty norm structure on non-rank-0 so parameter counts
                # match before the broadcast sync below.
                self.target_lm_head._init_norm_structure()
            self.target_lm_head.to(device="cuda", dtype=torch.bfloat16)
            self.target_lm_head.eval()
            self.target_lm_head.requires_grad_(False)

        dist.barrier()

        for param in self.target_lm_head.parameters():
            dist.broadcast(param.data, src=0)

        logger.info(f"[Rank {self.dp_rank}] TargetLMHead initialized and synced")

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _split_hidden_states(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated hidden states [B, seq_len, num_layers*D] into per-layer list.

        The target model concatenates hidden states from `num_target_layers` layers
        along the last dimension. We split them back into a list of [B, seq_len, D] tensors.
        """
        total_dim = hidden_states.shape[-1]
        per_layer_dim = total_dim // self.num_target_layers
        return list(hidden_states.split(per_layer_dim, dim=-1))

    def _forward(
        self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.to(device, non_blocking=True)

        # Sequence packing (DFlashPackingCollator): doc ids + doc-local positions.
        # Absent under the legacy pad-to-longest collator, in which case forward
        # falls back to the single-document path.
        ctx_doc_ids = batch.get("ctx_doc_ids")
        if ctx_doc_ids is not None:
            if ctx_doc_ids.dim() == 3:
                ctx_doc_ids = ctx_doc_ids.squeeze(-1)
            ctx_doc_ids = ctx_doc_ids.to(device, non_blocking=True)
        base_position_ids = batch.get("base_position_ids")
        if base_position_ids is not None:
            if base_position_ids.dim() == 3:
                base_position_ids = base_position_ids.squeeze(-1)
            base_position_ids = base_position_ids.to(device, non_blocking=True)

        # Target final hidden states + norm — only needed for L1/KL/LK distillation
        # (None on the base CE path, in which case those loss terms are skipped).
        last_hidden_states = batch.get("last_hidden_states")
        if last_hidden_states is not None:
            last_hidden_states = last_hidden_states.to(device, non_blocking=True)
        target_norm = getattr(self.target_lm_head, "norm", None) if self._distill_enabled else None

        hidden_states_list = self._split_hidden_states(hidden_states)
        del hidden_states

        opd_on = bool(self.score_engine) and getattr(self.args, "dflash_opd_enabled", False)
        out = self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=self.target_lm_head_weight,
            last_hidden_states=last_hidden_states,
            target_norm=target_norm,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            return_draft=opd_on,
        )
        if opd_on:
            (
                loss,
                accuracy,
                loss_per_position,
                acc_per_position,
                count_per_position,
                loss_components,
                opd,
            ) = out
            opd_loss, opd_metrics = self._compute_opd_loss(input_ids, opd)
            loss = loss + float(getattr(self.args, "dflash_opd_weight", 1.0)) * opd_loss
            self._last_opd_metrics = opd_metrics
        else:
            (
                loss,
                accuracy,
                loss_per_position,
                acc_per_position,
                count_per_position,
                loss_components,
            ) = out
            self._last_opd_metrics = {}

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )

    def _compute_opd_loss(self, input_ids: torch.Tensor, opd: dict):
        """Half on-policy OPD two-stream reverse-KL over the draft's proposal tree.

        Packs ALL sequences in the micro-batch into ONE multi-sequence tree and
        scores them in a SINGLE score_packed forward (cross-sequence isolation via
        the generalized tree mask). Student = the grad-carrying draft hidden at
        each proposed token's block slot; teacher = target hidden from the score
        engine (detached); both project through the shared frozen lm_head.

        The scored slots are then split per branch by greedy verify (draft-proposed
        token vs teacher argmax) into an accepted prefix (response stream) and a
        rejected suffix (rejected-draft stream, position-decayed), each scored with
        the k3 reverse-KL estimator (two KL terms: response + rejected-draft).

        Returns:
            (opd_loss, opd_metrics) — opd_metrics holds un-averaged sum/count
            scalars keyed by ``_OPD_METRIC_KEYS`` for DP-correct aggregation.
        """
        import ray

        from angelspec.models.ops.loss import (
            _OPD_METRIC_KEYS,
            opd_two_stream_kl_from_hs,
        )
        from angelspec.models.ops.tree_layout import build_dflash_opd_batch_layout

        draft_hidden = opd["draft_hidden"]  # [B, n_blocks*bs, D] (grad)
        proposals = opd["proposals"].cpu()  # [B, n_blocks, bs] (detached ids)
        anchors = opd["anchor_positions"].cpu()
        keep = opd["block_keep_mask"].cpu()
        ids_cpu = input_ids.detach().cpu()
        bs = int(getattr(self.args, "dflash_block_size", 16))
        device = draft_hidden.device
        lm_head = self.target_lm_head_weight
        b = ids_cpu.shape[0]

        bl = build_dflash_opd_batch_layout(
            [ids_cpu[i] for i in range(b)],
            [proposals[i] for i in range(b)],
            [anchors[i] for i in range(b)],
            [keep[i] for i in range(b)],
            bs,
            max_anchors=getattr(self.args, "dflash_opd_max_anchors", None),
        )
        if bl.score_index.numel() == 0:
            z = draft_hidden.new_zeros(())
            return z, {k: z.detach().clone() for k in _OPD_METRIC_KEYS}

        # Round-robin across the data-parallel score engines; offset by dp_rank so
        # the training ranks don't all pile onto engine 0 first.
        rr = getattr(self, "_score_rr", self.dp_rank)
        self._score_rr = rr + 1
        engine = self.score_engine[rr % len(self.score_engine)]

        _score_t0 = time.perf_counter()
        teacher_bytes = ray.get(
            engine.score_packed.remote(
                bl.packed_ids.tolist(),
                bl.positions.tolist(),
                bl.doc_ids.tolist(),
                bl.anchor_of.tolist(),
                bl.score_index.tolist(),
                bl.trunk_doc_of.tolist(),
            )
        )
        self._log_opd_score_wait(time.perf_counter() - _score_t0)

        # score_packed returns a raw bf16 (M, H) blob (see score_worker_ext); rebuild
        # in-place instead of materializing a ~M*H nested Python list. bf16 already
        # matches draft_hidden.dtype, so this is numerically identical to the old path.
        H = int(draft_hidden.shape[-1])
        M = int(bl.score_index.numel())
        teacher = (
            torch.frombuffer(bytearray(teacher_bytes), dtype=torch.bfloat16)
            .view(M, H)
            .to(device=device, dtype=draft_hidden.dtype)
        )
        student = draft_hidden[bl.student_seq.to(device), bl.student_slot.to(device)]
        opd_loss, opd_metrics = opd_two_stream_kl_from_hs(
            student,
            teacher,
            bl.score_target_ids.to(device),
            lm_head,
            bs,
            response_stream_weight=float(
                getattr(self.args, "dflash_opd_response_stream_weight", 1.0)
            ),
            rejected_stream_weight=float(
                getattr(self.args, "dflash_opd_rejected_stream_weight", 1.0)
            ),
            position_decay=float(getattr(self.args, "dflash_opd_position_decay", 0.8)),
            position_decay_enabled=bool(
                getattr(self.args, "dflash_opd_position_decay_enabled", True)
            ),
            loss_max_clamp=getattr(self.args, "dflash_opd_loss_max_clamp", 10.0),
            logprob_min_clamp=getattr(self.args, "dflash_opd_logprob_min_clamp", -10.0),
        )
        return opd_loss, opd_metrics

    def _log_opd_score_wait(self, dt: float) -> None:
        """Accumulate score_packed RPC wall-time and log the running mean periodically
        (every 50 calls) so the OPD scoring overhead is visible without per-call spam."""
        self._opd_score_wait_s = getattr(self, "_opd_score_wait_s", 0.0) + dt
        self._opd_score_calls = getattr(self, "_opd_score_calls", 0) + 1
        if self._opd_score_calls % 50 == 0:
            logger.info(
                "OPD score_packed: %d calls, %.1f ms/call (%.1fs total)",
                self._opd_score_calls,
                1000.0 * self._opd_score_wait_s / self._opd_score_calls,
                self._opd_score_wait_s,
            )

    def _backward(
        self,
        loss: torch.Tensor,
        accumulation_steps: int = 1,
        loss_scale: float | None = None,
    ) -> torch.Tensor:
        # Packing (A′): loss_scale is the row's token-weighted accumulation factor
        # (row_tokens / step_total_tokens), summing to 1 across the step's rows.
        # Non-packing: divide by the fixed accumulation_steps as before.
        if loss_scale is not None:
            scaled_loss = loss * loss_scale
        else:
            scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        return loss

    # ------------------------------------------------------------------
    # Eval (no-grad forward on CPU-cached data) — mirrors Eagle3Trainer
    # ------------------------------------------------------------------

    def eval_forward(self, batch: dict) -> dict:
        """Single forward pass without backward — returns per-position tensors."""
        with torch.no_grad():
            _, _, loss_per_position, acc_per_position, count_per_position, loss_components = (
                self._forward(batch)
            )
        return {
            "loss_pp": loss_per_position.detach(),
            "acc_pp": acc_per_position.detach(),
            "count_pp": count_per_position.detach(),
            **loss_components,
        }

    def _reduce_position_metrics(
        self,
        all_step_metrics: list[dict],
        *,
        loss_key: str,
        acc_key: str,
        count_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_sum_pp = torch.stack([m[loss_key] * m[count_key] for m in all_step_metrics]).sum(
            dim=0
        )
        correct_sum_pp = torch.stack([m[acc_key] * m[count_key] for m in all_step_metrics]).sum(
            dim=0
        )
        count_pp = torch.stack([m[count_key] for m in all_step_metrics]).sum(dim=0)

        dist.all_reduce(loss_sum_pp, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_sum_pp, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_pp, op=dist.ReduceOp.SUM)

        safe_count_pp = count_pp.clamp(min=1.0)
        avg_loss_pp = loss_sum_pp / safe_count_pp
        avg_acc_pp = correct_sum_pp / safe_count_pp
        return avg_loss_pp, avg_acc_pp, count_pp

    def _reduce_loss_components(self, all_step_metrics: list[dict], prefix: str) -> dict:
        """Reduce extra scalar loss components into ``{prefix}{key}`` global means."""
        out: dict = {}
        for key in self._extra_loss_component_keys:
            vals = [m[key] for m in all_step_metrics if key in m]
            if not vals:
                continue
            value = torch.stack([v.float() for v in vals]).mean()
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value = value / dist.get_world_size()
            out[f"{prefix}{key}"] = value.item()
        return out

    def _reduce_opd_metrics(self, all_step_metrics: list[dict]) -> dict:
        """DP-reduce the two-stream OPD sum/count scalars into logged means.

        Sums the un-averaged per-micro-batch sum/count scalars, all_reduces (SUM)
        over the DP group, then forms the global ratios. Returns {} when OPD is off
        (no ``_opd`` payload) — every opd-on rank emits the identical key set (via
        ``_OPD_METRIC_KEYS`` zero-fill) so this collective stays symmetric.
        """
        opd_dicts = [m["_opd"] for m in all_step_metrics if m.get("_opd")]
        if not opd_dicts:
            return {}
        from angelspec.models.ops.loss import _OPD_METRIC_KEYS

        totals = {k: torch.stack([d[k] for d in opd_dicts]).sum() for k in _OPD_METRIC_KEYS}
        flat = torch.stack([totals[k] for k in _OPD_METRIC_KEYS]).float()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        t = {k: flat[i] for i, k in enumerate(_OPD_METRIC_KEYS)}

        accepted = t["opd/accepted_cnt"]
        rejected = t["opd/rejected_cnt"]
        scored = t["opd/scored_cnt"].clamp(min=1.0)
        return {
            "opd/resp_kl": (t["opd/resp_kl_sum"] / accepted.clamp(min=1.0)).item(),
            "opd/rej_kl": (t["opd/rej_kl_wsum"] / t["opd/rej_eff_w_sum"].clamp(min=1.0)).item(),
            "opd/accept_rate": (accepted / (accepted + rejected).clamp(min=1.0)).item(),
            "opd/accepted_cnt": accepted.item(),
            "opd/rejected_cnt": rejected.item(),
            "opd/clamp_fraction": (t["opd/clamp_frac_sum"] / scored).item(),
        }

    def _compute_scalar_metrics(
        self,
        pred_loss_pp: torch.Tensor,
        pred_acc_pp: torch.Tensor,
        pred_count_pp: torch.Tensor,
    ) -> Tuple[float, float]:
        safe_total_count = pred_count_pp.sum().clamp(min=1.0)
        avg_acc = ((pred_acc_pp * pred_count_pp).sum() / safe_total_count).item()

        gamma = self.loss_decay_gamma
        if gamma is not None and gamma > 0:
            k = torch.arange(pred_loss_pp.shape[0], device=pred_loss_pp.device)
            weights = torch.exp(-k.float() / gamma)
        else:
            weights = torch.ones_like(pred_loss_pp)

        weighted_counts = pred_count_pp * weights
        safe_weighted_count = weighted_counts.sum().clamp(min=1.0)
        avg_loss = ((pred_loss_pp * weighted_counts).sum() / safe_weighted_count).item()
        return avg_loss, avg_acc

    def eval_from_cache(self) -> dict:
        """Run forward-only eval over all CPU-cached eval samples.

        Samples are stored individually (no padding). Re-collate into batches
        of ``eval_micro_batch_size`` (or ``micro_batch_size``) so the eval
        forward batch size is independent of cache generation throughput.
        """
        if not getattr(self, "_eval_cache", None):
            return {}

        eval_mbs = getattr(self.args, "eval_micro_batch_size", None) or self.args.micro_batch_size

        self.model.eval()
        all_metrics: list[dict] = []
        for i in range(0, len(self._eval_cache), eval_mbs):
            chunk = self._eval_cache[i : i + eval_mbs]
            batch = self._eval_collator(chunk)
            gpu_batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            all_metrics.append(self.eval_forward(gpu_batch))

        self.model.train()

        return self._aggregate_eval_metrics(all_metrics)

    def _aggregate_eval_metrics(self, all_step_metrics: list[dict]) -> dict:
        if not all_step_metrics:
            return {}

        avg_loss_pp, avg_acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_pp",
            acc_key="acc_pp",
            count_key="count_pp",
        )

        # Drop anchor slot (index 0) — see _aggregate_metrics for rationale.
        pred_loss_pp = avg_loss_pp[1:]
        pred_acc_pp = avg_acc_pp[1:]
        pred_count_pp = count_pp[1:]

        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            simulated_acc_len += cumulative

        weighted_avg_loss, avg_acc = self._compute_scalar_metrics(
            pred_loss_pp, pred_acc_pp, pred_count_pp
        )

        metrics: dict = {
            "eval/avg_loss": weighted_avg_loss,
            "eval/avg_acc": avg_acc,
            "eval/simulated_acc_len": simulated_acc_len,
        }
        for i in range(pred_loss_pp.shape[0]):
            metrics[f"eval/ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"eval/acc_{i}"] = pred_acc_pp[i].item()

        metrics.update(self._reduce_loss_components(all_step_metrics, "eval/"))

        if dist.get_rank() == 0:
            logger.info(
                f"eval: loss={weighted_avg_loss:.4f}, acc={avg_acc:.4f}, sim_acc_len={simulated_acc_len:.2f}"
            )

        return metrics

    # ------------------------------------------------------------------
    # Subclass contract implementations
    # ------------------------------------------------------------------

    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        evt_fwd_s = torch.cuda.Event(enable_timing=True)
        evt_fwd_e = torch.cuda.Event(enable_timing=True)
        evt_bwd_s = torch.cuda.Event(enable_timing=True)
        evt_bwd_e = torch.cuda.Event(enable_timing=True)

        evt_fwd_s.record()
        (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        ) = self._forward(batch)
        evt_fwd_e.record()

        evt_bwd_s.record()
        loss_scale = batch.get("_loss_scale") if isinstance(batch, dict) else None
        total_loss = self._backward(
            loss, accumulation_steps=accumulation_steps, loss_scale=loss_scale
        )
        evt_bwd_e.record()

        return {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_per_position.detach(),
            "acc_per_position": acc_per_position.detach(),
            "count_per_position": count_per_position.detach(),
            "total_loss": total_loss.detach(),
            "_opd": getattr(self, "_last_opd_metrics", {}),
            "_fwd_events": (evt_fwd_s, evt_fwd_e),
            "_bwd_events": (evt_bwd_s, evt_bwd_e),
            **loss_components,
        }

    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        if not all_step_metrics:
            return {}

        avg_loss_pp, avg_acc_pp, count_pp = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="loss_per_position",
            acc_key="acc_per_position",
            count_key="count_per_position",
        )

        # Skip index 0 (anchor slot, always zero); indices 1..B-1 are the
        # predicted tokens at 1..B-1 steps past the anchor. Re-index to 0..B-2
        # so the naming matches Eagle3 (acc_0 = first predicted token).
        pred_loss_pp = avg_loss_pp[1:]
        pred_acc_pp = avg_acc_pp[1:]
        pred_count_pp = count_pp[1:]

        # Simulated accepted length: acc_0 + acc_0*acc_1 + ... + prod(acc_0..acc_{B-2})
        # Models the expected number of consecutively accepted draft tokens.
        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            simulated_acc_len += cumulative

        avg_loss, avg_acc = self._compute_scalar_metrics(pred_loss_pp, pred_acc_pp, pred_count_pp)

        metrics = {
            "train/avg_loss": avg_loss,
            "train/avg_acc": avg_acc,
            "train/simulated_acc_len": simulated_acc_len,
            "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "train/global_step": self.global_step,
            "train/lr": self.optimizer.get_learning_rate(),
            "train/step": step,
        }

        for i in range(pred_loss_pp.shape[0]):
            metrics[f"train/ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"train/acc_{i}"] = pred_acc_pp[i].item()

        metrics.update(self._reduce_loss_components(all_step_metrics, "train/"))

        # Sub-timing breakdown (forward vs backward)
        fwd_ms = sum(
            m["_fwd_events"][0].elapsed_time(m["_fwd_events"][1])
            for m in all_step_metrics
            if "_fwd_events" in m
        )
        bwd_ms = sum(
            m["_bwd_events"][0].elapsed_time(m["_bwd_events"][1])
            for m in all_step_metrics
            if "_bwd_events" in m
        )
        metrics["perf/forward_time"] = fwd_ms / 1000.0
        metrics["perf/backward_time"] = bwd_ms / 1000.0

        if dist.get_rank() == 0 and (step % 5 == 0 or step <= 5):
            logger.info(
                f"COMPUTE_BREAKDOWN step={step}: forward={fwd_ms:.1f}ms backward={bwd_ms:.1f}ms"
            )

        # Layer-gate read-out (gated_sum selection run only; all-rank collective).
        metrics.update(self._gate_metrics())

        # Two-stream OPD read-out (all-rank collective; {} when OPD is off).
        opd_metrics = self._reduce_opd_metrics(all_step_metrics)
        metrics.update(opd_metrics)
        if opd_metrics and dist.get_rank() == 0 and (step % 5 == 0 or step <= 5):
            logger.info(
                "OPD step=%d: resp_kl=%.4f rej_kl=%.4f accept_rate=%.3f accepted=%d rejected=%d clamp_frac=%.3f",
                step,
                opd_metrics["opd/resp_kl"],
                opd_metrics["opd/rej_kl"],
                opd_metrics["opd/accept_rate"],
                int(opd_metrics["opd/accepted_cnt"]),
                int(opd_metrics["opd/rejected_cnt"]),
                opd_metrics["opd/clamp_fraction"],
            )

        if dist.get_rank() == 0:
            logger.debug(f"step {step}: {metrics}")

        return metrics

    def _gate_metrics(self) -> dict:
        """Layer-gate read-out + early-stop signals for the gated_sum selection run.

        No-op unless the draft model is a ``DFlashGatedDraftModel``. ``gate_weights``
        gathers a sharded DTensor via a collective, so this MUST run on all ranks;
        it returns metrics only on rank 0 (empty elsewhere). Called once per
        optimizer step from ``_aggregate_metrics`` (already an all-rank collective).
        """
        dm = getattr(self, "draft_model", None)
        if dm is None or not hasattr(dm, "gate_weights"):
            return {}

        w = dm.gate_weights()  # [T] CPU — collective (full_tensor) on ALL ranks
        if dist.get_rank() != 0:
            return {}

        layer_ids = list(getattr(dm, "target_layer_ids", None) or range(w.numel()))
        # 16 gate curves keyed by REAL target-layer id (not candidate index).
        metrics = {f"layer_gate/w_{layer_ids[i]}": w[i].item() for i in range(w.numel())}

        # Rolling-window top-5 stability: pairwise Jaccard mean + backbone size.
        k = min(5, w.numel())
        topk_ids = frozenset(layer_ids[j] for j in torch.topk(w, k).indices.tolist())
        self._gate_topk_window.append(topk_ids)
        sets = list(self._gate_topk_window)
        if len(sets) >= 2:
            pair_j = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
            metrics["layer_gate/topk_jaccard"] = sum(pair_j) / len(pair_j)
        else:
            metrics["layer_gate/topk_jaccard"] = 1.0
        metrics["layer_gate/backbone_size"] = float(len(set.intersection(*map(set, sets))))
        return metrics

    def save_model(self, step: int, force_sync: bool = False) -> None:
        super().save_model(step, force_sync=force_sync)
        self._write_gate_sidecar(step)

    def _write_gate_sidecar(self, step: int) -> None:
        """Write raw logits + softmax weights + RMSNorm scales next to the checkpoint.

        No-op unless gated_sum. The read-out helpers gather sharded DTensors via
        collectives, so they run on all ranks; only rank 0 writes the JSON. Mirrors
        ``checkpoint.save``'s ``iter_{step+1:07d}`` directory.
        """
        if not self.args.checkpoint_dir:
            return
        dm = getattr(self, "draft_model", None)
        if dm is None or not hasattr(dm, "gate_weights"):
            return

        logits = dm.gate_logits()  # collectives on ALL ranks
        weights = dm.gate_weights()
        scales = dm.candidate_rmsnorm_scales()
        if dist.get_rank() != 0:
            return

        import json
        from pathlib import Path

        layer_ids = list(getattr(dm, "target_layer_ids", None) or range(weights.numel()))
        payload = {
            "step": step,
            "target_layer_ids": layer_ids,
            "raw_logits": logits.tolist(),
            "softmax_weights": weights.tolist(),
            "rmsnorm_scale": scales.tolist(),
            "effective_contribution": (weights * scales).tolist(),
        }
        ckpt_dir = Path(self.args.checkpoint_dir).expanduser() / f"iter_{step + 1:07d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out = ckpt_dir / "layer_gate_readout.json"
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote layer_gate read-out to {out}")
