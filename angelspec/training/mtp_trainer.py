"""Trainer for the single-head MTP draft (path B).

The draft is a single MoE decoder layer with a tied, frozen lm_head, trained
with CE + optional KL. Can warm-start from the target's trained MTP layer.
"""

import glob
import json
import os
from argparse import Namespace
from typing import Optional

import torch
import torch.distributed as dist
from safetensors import safe_open

from angelspec import AutoDraftModelConfig, AutoEagle3DraftModel
from angelspec.models.mtp import MTPModel
from angelspec.training import checkpoint
from angelspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from angelspec.training.optimizer import BF16Optimizer
from angelspec.training.trainer import Trainer
from angelspec.utils.distributed import get_gloo_group
from angelspec.utils.logging import logger
from angelspec.utils.metrics import means_from_token_totals, token_metric_totals
from angelspec.utils.tensor import padding, same_doc_next_mask
from angelspec.utils.usp import usp_chunk_size, usp_dp_average_factor


def _position_decay_weights(n: int, weights: Optional[list] = None, theta_decay=0.8) -> list:
    """Per-position TTT loss weights (defaults to 0.8**i)."""
    if weights is not None:
        if len(weights) != n:
            raise ValueError(f"ploss_weights length {len(weights)} does not match ttt_length {n}")
        return [float(w) for w in weights]
    return [theta_decay**i for i in range(n)]


def align_mtp_inputs(
    raw_input_ids: torch.Tensor,
    raw_last_hidden: torch.Tensor,
    raw_loss_mask: torch.Tensor,
    *,
    draft_input_postnorm: bool = False,
    verifier_norm=None,
    ctx_doc_ids: torch.Tensor = None,
):
    """Shift the raw batch tensors to match how vLLM serve feeds the MTP block.

    The inference engine emits un-shifted, per-token-aligned tensors: input_ids[j]
    == token[j], and last_hidden[j] is the target's last-layer output after
    token[0..j] (the state used to predict token[j+1]).

    At position j, serve feeds  embed(token[j+1]) + raw_h[j] -> predict token[j+2]
    (input_ids left-shifted once, hidden left UNSHIFTED). Training must match that:
    the draft-input hidden stays UN-shifted (raw_h[j]) while the teacher hidden is
    left-shifted so its argmax agrees with the left-shifted label (both token[j+2]).
    The two therefore use different shifts despite both being "the target's last
    hidden" — shifting the draft input too lets it read the answer out of raw_h[j+1]
    (train acc looks fine but serve acceptance collapses).

    loss_mask is left-shifted TWICE: a position's loss is gated by whether the
    PREDICTED token is supervised, not the input token. step 0 predicts token[j+2],
    so +2 gates on raw_mask[j+2]; mtp.py shifts once more per subsequent step.

    Returns (input_ids, draft_input, target_hidden_states, loss_mask). draft_input
    and target_hidden_states pass through verifier_norm when a pre-norm capture
    must be restored to post-norm.
    """
    # embed source: shifted so position j embeds token[j+1].
    input_ids = padding(raw_input_ids, left=False)

    # teacher hidden: shifted so argmax(token[j+2]) matches the shifted label.
    last_hidden_shifted = padding(raw_last_hidden, left=False)
    if verifier_norm is not None:
        with torch.no_grad():
            target_hidden_states = verifier_norm(last_hidden_shifted)
    else:
        target_hidden_states = last_hidden_shifted

    # draft input hidden: un-shifted (raw_h[j]) to match serve.
    if draft_input_postnorm and verifier_norm is not None:
        with torch.no_grad():
            draft_input = verifier_norm(raw_last_hidden)
    else:
        draft_input = raw_last_hidden

    # loss mask: shifted twice so mask[j] gates on the predicted token (token[j+2]).
    # Packing: gate doc seams after each shift so a prediction crossing into the
    # next doc is not supervised.
    if ctx_doc_ids is not None:
        sd = same_doc_next_mask(ctx_doc_ids).to(
            device=raw_loss_mask.device, dtype=raw_loss_mask.dtype
        )
        loss_mask = padding(raw_loss_mask, left=False) * sd
        loss_mask = padding(loss_mask, left=False) * sd
    else:
        loss_mask = padding(padding(raw_loss_mask, left=False), left=False)

    return input_ids, draft_input, target_hidden_states, loss_mask


class MTPTrainer(Trainer):
    """MTP-specific trainer."""

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head = None
        self.target_lm_head_weight = None
        self.verifier_norm = None

    # Model init

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
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens.weight"),
            )

        draft_model.freeze_embedding()

        # Optional CPT: load the trained MTP layer from the target checkpoint.
        init_layer = getattr(draft_model_config, "mtp_init_from_target_layer", None)
        if init_layer is None:
            init_layer = getattr(self.args, "mtp_init_from_target_layer", None)
        if init_layer is not None:
            self._load_mtp_from_target(draft_model, target_model_path, int(init_layer))

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] MTP draft model: {trainable_count:,} trainable, {frozen_count:,} frozen parameters"
        )

        mtp_model = MTPModel(
            draft_model=draft_model,
            length=self.args.ttt_length,
            attention_backend=self.args.attention_backend,
            gradient_checkpointing=getattr(self.args, "gradient_checkpointing", True),
            use_kl=getattr(draft_model_config, "mtp_kl_loss_scaling_factor", 0.0) > 0.0,
            kl_topk=getattr(draft_model_config, "mtp_kl_topk", None),
            kl_variant=getattr(draft_model_config, "mtp_kl_variant", "a"),
            ce_use_ground_truth=getattr(draft_model_config, "mtp_ce_use_ground_truth", False),
            on_policy=getattr(self.args, "mtp_ttt_on_policy", False),
            single_step_form=getattr(draft_model_config, "mtp_single_step_loss", None),
            single_step_eta=getattr(draft_model_config, "mtp_kl_tv_eta", 3.0),
            e2e_weighting=getattr(draft_model_config, "mtp_e2e_weighting", False),
            e2e_direct=getattr(draft_model_config, "mtp_e2e_direct", False),
        )

        # Loss scaling factors (read once).
        self.ce_scale = float(getattr(draft_model_config, "mtp_ce_loss_scaling_factor", 1.0))
        self.kl_scale = float(getattr(draft_model_config, "mtp_kl_loss_scaling_factor", 0.0))
        self.single_step_form = getattr(draft_model_config, "mtp_single_step_loss", None)
        self.e2e_weighting = getattr(draft_model_config, "mtp_e2e_weighting", False) and (
            self.single_step_form is not None
        )
        self.e2e_scale = float(getattr(draft_model_config, "mtp_e2e_tv_loss_scaling_factor", 1.0))

        full_state = mtp_model.state_dict() if dist.get_rank() == 0 else {}

        # Shard the MoE layer's Linears (attn + shared_mlp + eh_proj) as their own
        # FSDP units. Fused [E,in,out] experts + fp32 router.gate fall into the ROOT
        # flat param (still sharded); excluding the gate keeps it out of a bf16 shard.
        midlayer_modules = [
            m
            for name, m in mtp_model.named_modules()
            if isinstance(m, torch.nn.Linear)
            and ("midlayer" in name or "eh_proj" in name)
            and "lm_head" not in name
            and "router.gate" not in name
        ]
        mtp_model = apply_fsdp2(
            mtp_model,
            mesh=self.grad_sync_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=midlayer_modules,
        )

        mtp_model = fsdp2_load_full_state_dict(
            mtp_model,
            full_state,
            self.grad_sync_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        self.model = mtp_model
        self.mtp = self.model.module if hasattr(self.model, "module") else self.model
        self.draft_model = self.mtp.draft_model

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
            optimizer_type=getattr(self.args, "optimizer_type", "adamw"),
            muon_momentum=getattr(self.args, "muon_momentum", 0.95),
            muon_nesterov=getattr(self.args, "muon_nesterov", True),
            muon_ns_steps=getattr(self.args, "muon_ns_steps", 5),
            muon_matched_adamw_rms=getattr(self.args, "muon_matched_adamw_rms", 0.2),
        )
        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        self._last_hs_prenorm = getattr(self.args, "last_hidden_states_prenorm", False)

        # Load the (frozen) target lm_head and tie it into the draft.
        self._init_target_lm_head(target_model_path)
        if self.target_lm_head is None:
            raise ValueError("target_lm_head is required for MTP but was None.")
        self.target_lm_head_weight = self.target_lm_head.lm_head.weight
        self.verifier_norm = self.target_lm_head.norm
        self.draft_model.set_lm_head_weight(self.target_lm_head_weight)

        # Post-norm alignment: the draft consumes the post-final-norm hidden, so
        # pre-norm captures (vLLM) need verifier_norm to restore it.
        if getattr(self.args, "mtp_draft_input_postnorm", False):
            if self._last_hs_prenorm and self.verifier_norm is None:
                raise ValueError(
                    "mtp_draft_input_postnorm=True with pre-norm captures "
                    "(last_hidden_states_prenorm=True) requires the target final "
                    "norm, but it was not loaded. Check norm_key / load_norm."
                )
            logger.info(f"[Rank {self.dp_rank}] MTP draft input = POST-norm (norm-on-norm)")

        self.prof.on_init_end()
        logger.info(
            f"[Rank {self.dp_rank}] MTP TTT = "
            f"{'ON-policy (argmax feedback)' if getattr(self.args, 'mtp_ttt_on_policy', False) else 'off-policy (teacher-forced)'}"
        )

        # USP flex warmup: compile the flex block-0 kernel identically on every SP rank
        # before training, so no rank triggers a mid-step compile that desyncs the
        # Ulysses all-to-all. Use the same ceil-div shard shape as the real
        # forward, which pads non-divisible full sequences to a multiple of SP.
        if getattr(self.args, "attention_backend", None) == "usp":
            sp_size = getattr(self.args, "sp_ulysses_size", 1) * getattr(
                self.args, "sp_ring_size", 1
            )
            max_seq = getattr(self.args, "max_seq_length", None)
            if max_seq and sp_size > 1:
                shard_len = usp_chunk_size(max_seq, sp_size)
                hidden = (
                    getattr(draft_model_config, "target_hidden_size", None)
                    or draft_model_config.hidden_size
                )
                logger.info(
                    f"[Rank {self.dp_rank}] USP flex warmup: shard_len={shard_len} "
                    f"(full_seq={max_seq}, sp_size={sp_size})"
                )
                self.mtp.warmup_usp_flex(
                    shard_len,
                    int(hidden),
                    self.target_lm_head_weight,
                    packing=getattr(self.args, "mtp_packing", False),
                )
                logger.info(f"[Rank {self.dp_rank}] USP flex warmup done")

        logger.info(f"[Rank {self.dp_rank}] MTP model initialized with FSDP2")
        return 0

    # CPT: load trained MTP layer from target checkpoint

    def _load_mtp_from_target(self, draft_model, target_model_path: str, layer_idx: int) -> None:
        """Load model.layers.<layer_idx>.* from the target ckpt into the MTP draft.

        Hy3 stores the trained MTP block as a decoder layer right after the
        backbone (index == num_hidden_layers). Key mapping into MTPDraftModel:
            model.layers.<N>.enorm/hnorm/eh_proj/final_layernorm      -> <same>
            model.layers.<N>.input_layernorm/post_attention_layernorm -> midlayer.<same>
            model.layers.<N>.self_attn.*                              -> midlayer.self_attn.*
            model.layers.<N>.mlp.*                                    -> midlayer.mlp.*
        embed_tokens / lm_head are not in this layer (loaded/tied separately).
        """
        if dist.get_rank() != 0:
            return

        prefix = f"model.layers.{layer_idx}."
        state = self._read_target_layer_weights(target_model_path, prefix)
        if not state:
            raise FileNotFoundError(
                f"MTP CPT init: no weights with prefix '{prefix}' found in {target_model_path}"
            )

        top_level = {"enorm", "hnorm", "eh_proj", "final_layernorm"}
        # The checkpoint stores experts per-expert as
        #   mlp.experts.<e>.{gate,up,down}_proj.weight  (each [out, in], Linear layout)
        # but Hy3MoE now holds FUSED [E, in, out] params experts_{gate,up,down}_proj.
        # Collect per-expert slices, transpose to [in, out], and stack on dim 0.
        import re

        expert_pat = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
        expert_slices = {"gate_proj": {}, "up_proj": {}, "down_proj": {}}

        remapped = {}
        for k, v in state.items():
            sub = k[len(prefix) :]  # e.g. "self_attn.q_proj.weight" or "enorm.weight"
            m = expert_pat.match(sub)
            if m is not None:
                e_idx, proj = int(m.group(1)), m.group(2)
                expert_slices[proj][e_idx] = v
                continue
            top = sub.split(".")[0]
            if top in top_level:
                remapped[sub] = v
            else:
                # input_layernorm / post_attention_layernorm / self_attn.* /
                # mlp.{router.gate,expert_bias,shared_mlp.*}
                remapped[f"midlayer.{sub}"] = v

        # Stack per-expert weights into the fused [E, in, out] layout.
        for proj, fused_name in (
            ("gate_proj", "experts_gate_proj"),
            ("up_proj", "experts_up_proj"),
            ("down_proj", "experts_down_proj"),
        ):
            slices = expert_slices[proj]
            if not slices:
                continue
            n_e = max(slices) + 1
            # checkpoint weight is [out, in]; fused layout is [in, out] → transpose
            stacked = torch.stack([slices[e].t().contiguous() for e in range(n_e)], dim=0)
            remapped[f"midlayer.mlp.{fused_name}"] = stacked

        missing, unexpected = draft_model.load_state_dict(remapped, strict=False)
        # embed_tokens.weight + lm_head.weight are expected to be missing here.
        real_missing = [
            m for m in missing if not (m.startswith("embed_tokens") or m.startswith("lm_head"))
        ]
        if real_missing:
            logger.warning(f"MTP CPT init: missing keys (will keep init values): {real_missing}")
        if unexpected:
            logger.warning(f"MTP CPT init: unexpected keys (ignored): {unexpected}")
        logger.info(
            f"[Rank 0] MTP CPT init from target layer {layer_idx}: loaded {len(remapped)} tensors"
        )

    @staticmethod
    def _read_target_layer_weights(model_path: str, prefix: str) -> dict:
        """Read all tensors whose key starts with ``prefix`` from a HF checkpoint."""
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(repo_id=model_path)

        index_paths = glob.glob(os.path.join(model_path, "*.index.json"))
        out: dict = {}
        if index_paths:
            with open(index_paths[0], "r") as f:
                weight_map = json.load(f)["weight_map"]
            shards = {}
            for k, shard in weight_map.items():
                if k.startswith(prefix):
                    shards.setdefault(shard, []).append(k)
            for shard, keys in shards.items():
                shard_path = os.path.join(model_path, shard)
                if shard_path.endswith(".safetensors"):
                    with safe_open(shard_path, framework="pt") as f:
                        for k in keys:
                            out[k] = f.get_tensor(k)
                else:
                    sd = torch.load(shard_path, map_location="cpu")
                    for k in keys:
                        out[k] = sd[k]
        else:
            single = os.path.join(model_path, "model.safetensors")
            if os.path.exists(single):
                with safe_open(single, framework="pt") as f:
                    for k in f.keys():
                        if k.startswith(prefix):
                            out[k] = f.get_tensor(k)
        return out

    # Target lm_head (frozen teacher, tied to draft)

    def _init_target_lm_head(self, target_model_path: str) -> None:
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

        has_norm = torch.tensor(
            [self.target_lm_head.norm is not None], dtype=torch.int32, device="cuda"
        )
        dist.broadcast(has_norm, src=0)
        if has_norm.item() and self.target_lm_head.norm is None:
            self.target_lm_head._init_norm_structure()
            self.target_lm_head.norm = self.target_lm_head.norm.to(
                device="cuda", dtype=torch.bfloat16
            )
        elif not has_norm.item() and self.target_lm_head.norm is not None:
            self.target_lm_head.norm = None

        dist.barrier()
        for param in self.target_lm_head.parameters():
            dist.broadcast(param.data, src=0)
        logger.info(f"[Rank {self.dp_rank}] target lm_head loaded and tied into MTP draft")

    # Forward / backward

    def _forward(self, batch: dict):
        # Align raw (un-shifted) batch tensors to the vLLM-serve MTP convention.
        # CRITICAL alignments (see align_mtp_inputs for the full derivation):
        #  - draft-input hidden stays UN-shifted (raw_h[j], paired with
        #    embed(token[j+1])) to match serve;
        #  - teacher hidden stays left-shifted once (raw_h[j+1]) so its argmax
        #    agrees with the ground-truth label;
        #  - loss_mask is left-shifted TWICE so it gates on the PREDICTED token
        #    (token[j+2]).
        raw_loss_mask = batch["loss_mask"]
        if raw_loss_mask.dim() == 3:
            raw_loss_mask = raw_loss_mask.squeeze(-1)

        # Packing doc ids (None on the non-packing path → all doc-aware logic inert).
        ctx_doc_ids = batch.get("ctx_doc_ids")
        if ctx_doc_ids is not None:
            if ctx_doc_ids.dim() == 3:
                ctx_doc_ids = ctx_doc_ids.squeeze(-1)
            ctx_doc_ids = ctx_doc_ids.cuda()
        base_position_ids = batch.get("base_position_ids")
        if base_position_ids is not None:
            if base_position_ids.dim() == 3:
                base_position_ids = base_position_ids.squeeze(-1)
            base_position_ids = base_position_ids.cuda()

        input_ids, draft_input, target_hidden_states, loss_mask = align_mtp_inputs(
            batch["input_ids"],
            batch["last_hidden_states"],
            raw_loss_mask,
            draft_input_postnorm=getattr(self.args, "mtp_draft_input_postnorm", False),
            verifier_norm=self.verifier_norm,
            ctx_doc_ids=ctx_doc_ids,
        )
        input_ids = input_ids.cuda()
        draft_input = draft_input.cuda()
        target_hidden_states = target_hidden_states.cuda()
        loss_mask = loss_mask.cuda()

        ce_losses, kl_losses, vlosses, acces, acc_counts, acces_gt, e2e_loss, vklosses = (
            self.model(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].cuda(),
                target_hidden_states=target_hidden_states,
                target_lm_head_weight=self.target_lm_head_weight,
                loss_mask=loss_mask,
                hidden_states=draft_input,
                position_ids=(
                    batch.get("position_ids").cuda()
                    if batch.get("position_ids") is not None
                    else None
                ),
                ctx_doc_ids=ctx_doc_ids,
                base_position_ids=base_position_ids,
            )
        )
        return ce_losses, kl_losses, vlosses, acces, acc_counts, acces_gt, e2e_loss, vklosses

    def _backward(
        self, ce_losses, kl_losses, e2e_loss=None, accumulation_steps: int = 1
    ) -> torch.Tensor:
        w = _position_decay_weights(len(ce_losses), getattr(self.args, "ploss_weights", None))
        ce_term = sum(w[i] * self.ce_scale * ce_losses[i] for i in range(len(ce_losses)))
        if self.e2e_weighting and e2e_loss is not None:
            total = (ce_term + self.e2e_scale * e2e_loss) / accumulation_steps
        else:
            distill = sum(w[i] * self.kl_scale * kl_losses[i] for i in range(len(ce_losses)))
            total = (ce_term + distill) / accumulation_steps
        total.backward()
        return total

    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        ce_losses, kl_losses, vlosses, acces, acc_counts, acces_gt, e2e_loss, vklosses = (
            self._forward(batch)
        )
        total_loss = self._backward(
            ce_losses, kl_losses, e2e_loss=e2e_loss, accumulation_steps=accumulation_steps
        )
        metrics = {
            # These are local means accompanied by acc_counts. Aggregation converts
            # them back to sums before the cross-rank reduction; ce_losses/kl_losses
            # remain the separate GLOBAL-count-normalized backward values.
            "ce": torch.stack([c.detach() for c in vlosses]),
            "kl": torch.stack([k.detach() for k in vklosses]),
            "vlosses": torch.stack(vlosses).detach(),
            "acces": torch.stack(acces).detach(),
            "acces_gt": torch.stack(acces_gt).detach(),
            "acc_counts": torch.stack(acc_counts).detach(),
            "total_loss": total_loss.detach(),
        }
        if e2e_loss is not None:
            metrics["e2e"] = e2e_loss.detach()
        return metrics

    def _save_dump_data(self, **kwargs) -> None:
        # MTP dump not yet wired; no-op to satisfy the base contract.
        pass

    # Eval

    def eval_forward(self, batch: dict) -> dict:
        with torch.no_grad():
            _, _, vlosses, acces, acc_counts, acces_gt, _, _ = self._forward(batch)
        return {
            "vlosses": torch.stack(vlosses).detach(),
            "acces": torch.stack(acces).detach(),
            "acces_gt": torch.stack(acces_gt).detach(),
            "acc_counts": torch.stack(acc_counts).detach(),
        }

    def _eval_metric_group(self):
        # USP metric collectives MUST use the (NCCL-warm) grad-sync group, not the
        # default WORLD PG (a bare all_reduce lazily makes a fresh WORLD communicator
        # that deadlocks against the sub-group comms in flight). Non-USP -> WORLD.
        if getattr(self.args, "attention_backend", None) == "usp":
            mesh = getattr(self, "grad_sync_mesh", None)
            return mesh.get_group() if mesh is not None else None
        return None

    def eval_from_cache(self) -> dict:
        eval_mbs = getattr(self.args, "eval_micro_batch_size", None) or self.args.micro_batch_size
        cache = getattr(self, "_eval_cache", None) or []
        metric_group = self._eval_metric_group()
        # Every training rank runs FSDP, which all-gathers params on EACH forward, so
        # ranks must run the SAME number of eval batches or the all-gathers (and the
        # final metric all_reduces) desync -> NCCL deadlock. A ragged per-rank eval
        # cache (e.g. reused across a different dp_size) would otherwise hang here.
        # Truncate every rank to the global-min batch count to keep collectives in
        # lockstep; if the min is 0, all ranks return {} symmetrically (no hang).
        local_batches = (len(cache) + eval_mbs - 1) // eval_mbs
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        n_t = torch.tensor([local_batches], device=dev)
        if dist.is_initialized():
            dist.all_reduce(n_t, op=dist.ReduceOp.MIN, group=metric_group)
        n_batches = int(n_t.item())
        if n_batches == 0:
            return {}
        self.model.eval()
        all_metrics: list = []
        for b in range(n_batches):
            chunk = cache[b * eval_mbs : (b + 1) * eval_mbs]
            batch = self._eval_collator(chunk)
            gpu_batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            all_metrics.append(self.eval_forward(gpu_batch))
        self.model.train()
        return self._aggregate_eval_metrics(all_metrics)

    def _aggregate_eval_metrics(self, all_step_metrics: list) -> dict:
        if not all_step_metrics:
            return {}
        metric_group = self._eval_metric_group()
        totals = token_metric_totals(all_step_metrics, ce_key="vlosses")
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=metric_group)
        avg_vlosses, _, avg_acces, avg_acces_gt, _ = means_from_token_totals(totals)

        def _sim_acc_len(acc_vec) -> float:
            cum, total = 1.0, 0.0
            for i in range(acc_vec.shape[0]):
                cum *= acc_vec[i].item()
                total += cum
            return total

        simulated_acc_len = _sim_acc_len(avg_acces)
        simulated_acc_len_gt = _sim_acc_len(avg_acces_gt)

        metrics = {
            "eval/avg_loss": avg_vlosses.mean().item(),
            "eval/avg_acc": avg_acces.mean().item(),
            "eval/avg_acc_gt": avg_acces_gt.mean().item(),
            "eval/simulated_acc_len": simulated_acc_len,
            "eval/simulated_acc_len_gt": simulated_acc_len_gt,
        }
        for i in range(avg_vlosses.shape[0]):
            metrics[f"eval/ce_{i}"] = avg_vlosses[i].item()
            metrics[f"eval/acc_{i}"] = avg_acces[i].item()
            metrics[f"eval/acc_gt_{i}"] = avg_acces_gt[i].item()
        return metrics

    # Metrics

    def _aggregate_metrics(
        self, all_step_metrics: list, step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        if not all_step_metrics:
            return {}
        # Under USP, metric collectives MUST use the (NCCL-warm) grad-sync group, not
        # the default WORLD PG — a bare dist.all_reduce lazily creates a fresh WORLD
        # communicator that deadlocks against the sub-group communicators in flight.
        metric_group = None
        if getattr(self.args, "attention_backend", None) == "usp":
            mesh = getattr(self, "grad_sync_mesh", None)
            metric_group = mesh.get_group() if mesh is not None else None
        totals = token_metric_totals(all_step_metrics, ce_key="ce")
        _dbg = os.environ.get("ANGELSPEC_USP_COLLECTIVE_DEBUG") == "1"
        _r = dist.get_rank()
        if _dbg:
            logger.info(
                f"[USP-COLL rank{_r}] agg: token totals shape={tuple(totals.shape)} "
                f"finite={bool(torch.isfinite(totals).all())} grp={metric_group is not None}"
            )
        # One small collective aggregates every TTT step's sums and counts across
        # both sequence-parallel shards and data-parallel replicas.
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=metric_group)
        if _dbg:
            logger.info(f"[USP-COLL rank{_r}] agg: token totals reduce END")
        avg_ce, avg_kl, avg_acces, avg_acces_gt, _ = means_from_token_totals(totals)

        def _sim_acc_len(acc_vec) -> float:
            cum, total = 1.0, 0.0
            for i in range(acc_vec.shape[0]):
                cum *= acc_vec[i].item()
                total += cum
            return total

        simulated_acc_len = _sim_acc_len(avg_acces)
        simulated_acc_len_gt = _sim_acc_len(avg_acces_gt)

        w = _position_decay_weights(avg_ce.shape[0], getattr(self.args, "ploss_weights", None))
        w_t = torch.tensor(w, device=avg_ce.device)
        weighted_ce = (avg_ce * w_t).sum().item() / w_t.sum().item()

        e2e_metric = None
        if any("e2e" in m for m in all_step_metrics):
            avg_e2e = torch.stack([m["e2e"] for m in all_step_metrics if "e2e" in m]).mean()
            if getattr(self.args, "attention_backend", None) == "usp":
                # Each SP rank reports local_sum / SP-global_count. SUM reconstructs
                # each DP replica's full-sequence mean; divide by DP replicas to AVG.
                dist.all_reduce(avg_e2e, op=dist.ReduceOp.SUM, group=metric_group)
                group_world_size = dist.get_world_size(metric_group)
                sp_size = int(getattr(self.args, "sp_ulysses_size", 1)) * int(
                    getattr(self.args, "sp_ring_size", 1)
                )
                avg_e2e.div_(usp_dp_average_factor(group_world_size, sp_size))
            else:
                dist.all_reduce(avg_e2e, op=dist.ReduceOp.AVG, group=metric_group)
            e2e_metric = avg_e2e.item()

        metrics = {
            "train/avg_loss": weighted_ce,
            "train/avg_ce": avg_ce.mean().item(),
            "train/avg_kl": avg_kl.mean().item(),
            "train/avg_acc": avg_acces.mean().item(),
            "train/avg_acc_gt": avg_acces_gt.mean().item(),
            "train/simulated_acc_len": simulated_acc_len,
            "train/simulated_acc_len_gt": simulated_acc_len_gt,
            "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
            "train/global_step": self.global_step,
            "train/lr": self.optimizer.get_learning_rate(),
            "train/step": step,
        }
        if e2e_metric is not None:
            metrics["train/e2e_loss"] = e2e_metric
        # When single_step_form is set the kl slot holds the tv/kl/lk distill term;
        # surface it under its real name too.
        if self.single_step_form is not None:
            metrics[f"train/{self.single_step_form}_loss"] = avg_kl.mean().item()
        for i in range(avg_ce.shape[0]):
            metrics[f"train/ce_{i}"] = avg_ce[i].item()
            metrics[f"train/kl_{i}"] = avg_kl[i].item()
            metrics[f"train/acc_{i}"] = avg_acces[i].item()
            metrics[f"train/acc_gt_{i}"] = avg_acces_gt[i].item()

        if dist.get_rank() == 0:
            logger.debug(f"step {step}: {metrics}")
        return metrics
