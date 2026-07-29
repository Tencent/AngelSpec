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

from argparse import Namespace
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from angelspec import AutoDraftModelConfig, AutoEagle3DraftModel, Eagle3Model
from angelspec.models.eagle3 import compute_lazy_target_padded, compute_target_p_padded
from angelspec.training import checkpoint
from angelspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from angelspec.training.optimizer import BF16Optimizer
from angelspec.training.trainer import Trainer
from angelspec.utils.distributed import get_gloo_group
from angelspec.utils.logging import logger
from angelspec.utils.tensor import padding
from angelspec.utils.train_dump import dump_eagle3_batch


def _position_decay_weights(n: int, weights: Optional[list[float]] = None) -> list[float]:
    """Per-position TTT loss weights.

    If ``weights`` is supplied, it is used as-is (after coercion to ``list[float]``).
    Otherwise defaults to ``[0.8**i for i in range(n)]``.
    """
    if weights is not None:
        if len(weights) != n:
            raise ValueError(f"ploss_weights length {len(weights)} does not match ttt_length {n}")
        return [float(w) for w in weights]
    return [0.8**i for i in range(n)]


class Eagle3Trainer(Trainer):
    """Eagle3-specific trainer.

    Extends ``Trainer`` with Eagle3 model initialisation, forward/backward,
    and metric aggregation.
    """

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head: Optional[torch.nn.Module] = None

    def init_model(
        self,
        draft_model_config: AutoDraftModelConfig,
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
            draft_model = AutoEagle3DraftModel.from_config(
                draft_model_config,
                attention_backend=self.args.attention_backend,
                torch_dtype=torch.bfloat16,
            )

        if dist.get_rank() == 0:
            draft_model.load_embedding(
                target_model_path,
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens"),
            )

        draft_model.freeze_embedding()

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] Draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen (embedding) parameters"
        )

        eagle3_model = Eagle3Model(
            draft_model=draft_model,
            length=self.args.ttt_length,
            attention_backend=self.args.attention_backend,
            gradient_checkpointing=getattr(self.args, "gradient_checkpointing", True),
        )

        full_state = eagle3_model.state_dict() if dist.get_rank() == 0 else {}

        midlayer_modules = [
            m
            for name, m in eagle3_model.named_modules()
            if isinstance(m, torch.nn.Linear) and "midlayer" in name
        ]
        eagle3_model = apply_fsdp2(
            eagle3_model,
            mesh=self.grad_sync_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=midlayer_modules,
        )

        eagle3_model = fsdp2_load_full_state_dict(
            eagle3_model,
            full_state,
            self.grad_sync_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        self.model = eagle3_model
        self.eagle3 = self.model.module if hasattr(self.model, "module") else self.model
        self.draft_model = self.eagle3.draft_model
        self.t2d = self.draft_model.t2d if self.eagle3.vocab_pruning else None

        decay_style = getattr(self.args, "lr_decay_style", "cosine")
        wsd_decay_steps = None
        wsd_decay_style = None
        if decay_style == "WSD":
            wsd_ratio = getattr(self.args, "lr_wsd_decay_ratio", 0.2)
            wsd_decay_steps = int(wsd_ratio * self.args.lr_total_steps)
            wsd_decay_style = getattr(self.args, "lr_wsd_decay_style", "cosine")
        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=getattr(self.args, "warmup_ratio", 0.1),
            total_steps=self.args.lr_total_steps,
            decay_style=decay_style,
            min_lr=getattr(self.args, "min_lr", 0.0),
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style,
        )
        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        self._last_hs_prenorm = getattr(self.args, "last_hidden_states_prenorm", False)

        if getattr(self.args, "compute_logits_in_trainer", True):
            self._init_target_lm_head(target_model_path)

        if self.target_lm_head is None:
            raise ValueError(
                "target_lm_head is required but was None. Set compute_logits_in_trainer=True or provide a TargetLMHead."
            )
        self.target_lm_head_weight = self.target_lm_head.lm_head.weight
        self.verifier_norm = self.target_lm_head.norm

        if getattr(self.args, "attention_backend", None) == "fa4":
            from angelspec.models.draft.llama3_eagle import (
                _has_cute_dsl,
                warmup_flash_attention_masked,
            )

            if _has_cute_dsl:
                cfg = self.draft_model.config
                head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
                warmup_flash_attention_masked(
                    q_len=self.args.max_seq_length,
                    num_heads=cfg.num_attention_heads,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=head_dim,
                    dtype=torch.bfloat16,
                    device=torch.cuda.current_device(),
                )

        self.prof.on_init_end()

        logger.info(f"[Rank {self.dp_rank}] Eagle3 model initialized with FSDP2")

        return 0

    # ------------------------------------------------------------------
    # Eagle3-specific helpers
    # ------------------------------------------------------------------

    def _init_target_lm_head(self, target_model_path: str) -> None:
        """Initialize TargetLMHead for computing logits from last_hidden_states.

        Only rank 0 loads the weights, then broadcasts to other ranks.
        The lm_head is kept frozen and not wrapped with FSDP.
        """
        from angelspec.models.target.target_utils import TargetLMHead

        if dist.get_rank() == 0:
            self.target_lm_head = TargetLMHead.from_pretrained(
                model_path=target_model_path,
                lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
                norm_key=getattr(self.args, "norm_key", "model.norm.weight"),
                load_norm=self._last_hs_prenorm,
                device="cuda",
                dtype=torch.bfloat16,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            logger.info(
                f"[Rank 0] TargetLMHead loaded from {target_model_path}"
                f"{' (with verifier norm)' if self._last_hs_prenorm else ''}"
            )
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                target_model_path,
                trust_remote_code=getattr(self.args, "trust_remote_code", True),
            )
            self.target_lm_head = TargetLMHead(config)
            if self._last_hs_prenorm:
                self.target_lm_head._init_norm_structure()
            self.target_lm_head.to(device="cuda", dtype=torch.bfloat16)
            self.target_lm_head.eval()
            self.target_lm_head.requires_grad_(False)

        # Sync norm status from rank 0 so all ranks have the same parameter count
        # before the broadcast loop (prevents NCCL deadlock if norm loading
        # silently failed on rank 0 but structure creation succeeded elsewhere).
        has_norm = torch.tensor(
            [self.target_lm_head.norm is not None], dtype=torch.int32, device="cuda"
        )
        dist.broadcast(has_norm, src=0)
        if has_norm.item():
            if self.target_lm_head.norm is None:
                logger.warning(
                    f"[Rank {self.dp_rank}] Rank 0 has norm but this rank does not — "
                    "this indicates _init_norm_structure failed; attempting recovery"
                )
                self.target_lm_head._init_norm_structure()
                self.target_lm_head.norm = self.target_lm_head.norm.to(
                    device="cuda", dtype=torch.bfloat16
                )
        else:
            if self.target_lm_head.norm is not None:
                logger.warning(
                    f"[Rank {self.dp_rank}] Rank 0 does not have norm — removing norm on this rank to match"
                )
                self.target_lm_head.norm = None

        dist.barrier()

        for param in self.target_lm_head.parameters():
            dist.broadcast(param.data, src=0)

        logger.info(f"[Rank {self.dp_rank}] TargetLMHead initialized and synced")

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(self, batch: dict) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        input_ids = padding(batch["input_ids"], left=False).cuda()
        target_hidden_states = padding(batch["last_hidden_states"], left=False).cuda()

        if self.verifier_norm is not None:
            with torch.no_grad():
                target_hidden_states = self.verifier_norm(target_hidden_states)

        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.cuda()

        if self.t2d is not None:
            target = compute_target_p_padded(
                target_hidden_states=target_hidden_states,
                target_lm_head_weight=self.target_lm_head_weight,
                t2d=self.t2d,
                loss_mask=loss_mask,
                length=self.eagle3.length,
            )
        else:
            target = compute_lazy_target_padded(
                target_hidden_states,
                self.target_lm_head_weight,
                self.eagle3.length,
            )
        del target_hidden_states

        plosses, vlosses, acces, acc_counts = self.model(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].cuda(),
            target=target,
            loss_mask=loss_mask,
            hidden_states=batch["hidden_states"].cuda(),
            position_ids=batch.get("position_ids").cuda()
            if batch.get("position_ids") is not None
            else None,
        )
        return plosses, vlosses, acces, acc_counts

    def _backward(self, plosses: List[torch.Tensor], accumulation_steps: int = 1) -> torch.Tensor:
        ploss_weight = _position_decay_weights(
            len(plosses), getattr(self.args, "ploss_weights", None)
        )
        ploss = sum(ploss_weight[i] * plosses[i] for i in range(len(plosses))) / accumulation_steps
        ploss.backward()
        return ploss

    # ------------------------------------------------------------------
    # Eval (no-grad forward on CPU-cached data)
    # ------------------------------------------------------------------

    def eval_forward(self, batch: dict) -> dict:
        """Single forward pass without backward — returns per-position metrics."""
        with torch.no_grad():
            _plosses, vlosses, acces, acc_counts = self._forward(batch)
        return {
            "vlosses": torch.stack(vlosses).detach(),
            "acces": torch.stack(acces).detach(),
            "acc_counts": torch.stack(acc_counts).detach(),
        }

    def eval_from_cache(self) -> dict:
        """Run forward-only eval over all CPU-cached eval samples.

        Samples are stored individually (no padding). We re-collate them into
        batches of ``eval_micro_batch_size`` (or ``micro_batch_size``) so the
        eval forward batch size is independent of cache generation throughput.
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

        # Switch back to training mode
        self.model.train()

        return self._aggregate_eval_metrics(all_metrics)

    def _aggregate_eval_metrics(self, all_step_metrics: list[dict]) -> dict:
        if not all_step_metrics:
            return {}

        avg_vlosses = torch.stack([m["vlosses"] for m in all_step_metrics]).mean(dim=0)
        avg_acces = torch.stack([m["acces"] for m in all_step_metrics]).mean(dim=0)

        dist.all_reduce(avg_vlosses, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_acces, op=dist.ReduceOp.AVG)

        avg_acc_scalar = avg_acces.mean().item()

        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(avg_acces.shape[0]):
            cumulative *= avg_acces[i].item()
            simulated_acc_len += cumulative

        ploss_weights = torch.tensor(
            _position_decay_weights(
                avg_vlosses.shape[0], getattr(self.args, "ploss_weights", None)
            ),
            device=avg_vlosses.device,
        )
        weighted_avg_loss = (avg_vlosses * ploss_weights).sum().item() / ploss_weights.sum().item()

        metrics: dict = {
            "eval/avg_loss": weighted_avg_loss,
            "eval/avg_acc": avg_acc_scalar,
            "eval/simulated_acc_len": simulated_acc_len,
        }
        for i in range(avg_vlosses.shape[0]):
            metrics[f"eval/ploss_{i}"] = avg_vlosses[i].item()
            metrics[f"eval/acc_{i}"] = avg_acces[i].item()

        if dist.get_rank() == 0:
            logger.info(
                f"eval: loss={weighted_avg_loss:.4f}, acc={avg_acc_scalar:.4f}, sim_acc_len={simulated_acc_len:.2f}"
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
        plosses, vlosses, acces, acc_counts = self._forward(batch)
        total_loss = self._backward(plosses, accumulation_steps=accumulation_steps)

        return {
            "plosses": torch.stack(plosses).detach(),
            "vlosses": torch.stack(vlosses).detach(),
            "acces": torch.stack(acces).detach(),
            "acc_counts": torch.stack(acc_counts).detach(),
            "plosses_raw": [p.detach() for p in plosses],
            "acces_raw": [a.detach() for a in acces],
            "total_loss": total_loss.detach(),
        }

    def _save_dump_data(
        self,
        *,
        batch: dict,
        step_metrics: dict,
        gradients: dict,
        model_weights: dict,
        step: int,
        batch_idx: int,
    ) -> None:
        dump_eagle3_batch(
            self.args,
            step=step,
            batch_idx=batch_idx,
            batch=batch,
            plosses=step_metrics.get("plosses_raw"),
            acces=step_metrics.get("acces_raw"),
            gradients=gradients,
            total_loss=step_metrics.get("total_loss"),
            model_weights=model_weights,
        )

    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        if not all_step_metrics:
            return {}

        avg_vlosses = torch.stack([m["vlosses"] for m in all_step_metrics]).mean(dim=0)
        avg_acces = torch.stack([m["acces"] for m in all_step_metrics]).mean(dim=0)

        dist.all_reduce(avg_vlosses, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_acces, op=dist.ReduceOp.AVG)

        avg_acc_scalar = avg_acces.mean().item()

        # Simulated acceptance length: acc_0 + acc_0*acc_1 + acc_0*acc_1*acc_2 + ...
        # Models the expected number of consecutively accepted draft tokens,
        # which better reflects actual speculative decoding performance.
        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(avg_acces.shape[0]):
            cumulative *= avg_acces[i].item()
            simulated_acc_len += cumulative

        ploss_weights = torch.tensor(
            _position_decay_weights(
                avg_vlosses.shape[0], getattr(self.args, "ploss_weights", None)
            ),
            device=avg_vlosses.device,
        )
        weighted_avg_loss = (avg_vlosses * ploss_weights).sum().item() / ploss_weights.sum().item()

        metrics = {
            "train/avg_loss": weighted_avg_loss,
            "train/avg_acc": avg_acc_scalar,
            "train/simulated_acc_len": simulated_acc_len,
            "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "train/global_step": self.global_step,
            "train/lr": self.optimizer.get_learning_rate(),
            "train/step": step,
        }

        for i in range(avg_vlosses.shape[0]):
            metrics[f"train/ploss_{i}"] = avg_vlosses[i].item()
            metrics[f"train/acc_{i}"] = avg_acces[i].item()

        if dist.get_rank() == 0:
            logger.debug(f"step {step}: {metrics}")

        return metrics
