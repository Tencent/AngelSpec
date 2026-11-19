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

import json
import math
import time
from argparse import Namespace
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import (
    DFlashDraftModel,
    target_layers_per_draft_layer,
)
from angelspec.training import checkpoint
from angelspec.training.activation_checkpointing import (
    enable_dflash_activation_checkpointing,
)
from angelspec.training.dfly_opd import DFlashOPDMixin
from angelspec.training.fsdp import (
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fused_expert_shard_units,
)
from angelspec.training.optimizer import BF16Optimizer, moe_lr_scales, muon_head_rows
from angelspec.training.trainer import Trainer
from angelspec.utils.distributed import get_gloo_group
from angelspec.utils.logging import logger
from angelspec.utils.usp import slice_usp_sequence_shard


class DFlashTrainer(DFlashOPDMixin, Trainer):
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
        "e2e_accept_len",
        # Path selector; only emitted when the draft config enables it.
        "selector_loss",
        "selector_recall",
        "selector_acc",
        # MTP fusion (gates' RMS, injection vs the input it lands on); only with mtp_fusion.
        "mtp_fusion_gate",
        "mtp_fusion_ratio",
        # Attentive correction's output vs the hidden it lands on; only with
        # attentive_correction (and not at eval under dflash_acorr_eval_off).
        "acorr_out_ratio",
        # Each predicted slot's mean absolute block gate; only with
        # attentive_correction_block_mode "gated".
        *[f"acorr_gate_s{slot}" for slot in range(1, 16)],
        # With attentive_correction_block_kv_b "separate": how far the block read's own
        # kv_b has moved from the context read's, relative to the latter.
        "acorr_block_kv_b_drift",
        # The candidate-set size (index_union_size) is not here: it arrives as a sum/count
        # pair under "_index" and goes through _reduce_index_metrics, which weighs it by
        # block instead of averaging each rank's own mean.
        # Share of supervised positions whose label is among the k highest uncorrected
        # logits, the candidates the correction can reach; only with
        # enable_hidden_correction_topk.
        "hc_topk_recall",
        # Slots >= 2 split by whether the previous token tops the previous slot's
        # uncorrected logits: that share, and the accuracy on each side; only with
        # enable_hidden_correction_topk.
        "hc_prev_top1_rate",
        "hc_acc_prev_top1",
        "hc_acc_prev_other",
    ]
    # The components above that are eval metrics rather than loss terms, so named eval
    # sets report them as well.
    _named_eval_component_keys: tuple[str, ...] = (
        "hc_topk_recall",
        "hc_prev_top1_rate",
        "hc_acc_prev_top1",
        "hc_acc_prev_other",
        "acorr_out_ratio",
        *[f"acorr_gate_s{slot}" for slot in range(1, 16)],
    )

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.target_lm_head: Optional[torch.nn.Module] = None
        self.num_target_layers = getattr(args, "dflash_num_target_layers", None) or 5
        # vllm hands the target's final layer over as last_hidden_states rather
        # than inside the concatenated aux hiddens (set by train_entry).
        self.final_layer_from_last_hidden = bool(
            getattr(args, "dflash_final_layer_from_last_hidden", False)
        )
        self.block_size = getattr(args, "dflash_block_size", 16)
        self.num_anchors = getattr(args, "dflash_num_anchors", 512)
        self.loss_decay_gamma = getattr(args, "dflash_loss_decay_gamma", 7.0)
        self.fp32_lm_head = getattr(args, "dflash_fp32_lm_head", True)
        # Intra-block attention direction (dflash / dfly / dspark / dflare share
        # this). False => block-bidirectional (legacy); True => intra-block
        # causal (required for USP long-context training).
        self.causal_attn = bool(getattr(args, "dflash_causal_attn", False))
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
        # Vocab-dim chunk size for the e2e-TV streaming softmax+min. 0 => defer
        # to env var / model default (see DFlashModel._compute_e2e_tv_loss).
        _e2e_tv_chunk = int(getattr(args, "dflash_e2e_tv_vocab_chunk", 0) or 0)
        self.e2e_tv_vocab_chunk = _e2e_tv_chunk if _e2e_tv_chunk > 0 else None
        # Path selector K-way CE weight. Only takes effect when the draft config
        # sets enable_path_selector (the head has to exist to be trained).
        self.path_selector_loss_alpha = float(
            getattr(args, "dflash_path_selector_loss_alpha", 1.0)
        )
        # Whether the selector's label follows the unary head onto the target's
        # argmax once KL/LK replaces the corpus CE (see DFlashModel.forward).
        self.path_selector_teacher_labels = bool(
            getattr(args, "dflash_path_selector_teacher_labels", True)
        )
        # Whether the selector's K-way CE rides decay / D-PACE position weights.
        self.path_selector_objective_weighting = bool(
            getattr(args, "dflash_path_selector_objective_weighting", True)
        )
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

    def _defer_batch_h2d(self) -> bool:
        return getattr(self.args, "attention_backend", None) == "usp" and bool(
            getattr(self.args, "usp_local_shard", False)
        )

    def _sequence_shard_after_leader_fetch(self) -> bool:
        return (
            self._defer_batch_h2d()
            and bool(getattr(self.args, "sp_leader_mooncake_fetch", False))
            and not bool(getattr(self.args, "dflash_packing", False))
        )

    def _sample_usp_anchors(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
        padded_seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pad = padded_seq_len - input_ids.shape[1]
            sample_loss_mask = (
                torch.nn.functional.pad(loss_mask, (0, pad), value=0) if pad > 0 else loss_mask
            )
            sample_doc_ids = (
                torch.nn.functional.pad(ctx_doc_ids, (0, pad), value=-1)
                if pad > 0 and ctx_doc_ids is not None
                else ctx_doc_ids
            )
            eligibility_kwargs = {}
            if self.dflash.uses_mrope:
                eligibility = self.dflash._mrope_anchor_mask(input_ids, position_ids)
                if pad > 0:
                    eligibility = torch.nn.functional.pad(eligibility, (0, pad), value=False)
                eligibility_kwargs["anchor_eligibility"] = eligibility
            anchors, keep_mask = self.dflash._sample_anchor_positions(
                padded_seq_len,
                sample_loss_mask,
                input_ids.device,
                ctx_doc_ids=sample_doc_ids,
                **eligibility_kwargs,
            )
            sp_src = dist.get_global_rank(self.sp_group, 0)
            packed_anchors = torch.where(keep_mask, anchors, -1)
            dist.broadcast(packed_anchors, src=sp_src, group=self.sp_group)
            keep_mask = packed_anchors >= 0
            anchors = packed_anchors.clamp_min(0)
            local_anchors = anchors[:, self._sp_rank() :: self._sp_size()]
        return anchors, keep_mask, local_anchors

    def _sp_size(self) -> int:
        return int(getattr(self.args, "sp_ulysses_size", 1))

    def _sp_rank(self) -> int:
        return dist.get_rank(self.sp_group) if self.sp_group is not None else 0

    def _gather_teacher_rows(
        self,
        last_hidden_states: torch.Tensor,
        local_anchors: torch.Tensor,
    ) -> torch.Tensor:
        if last_hidden_states.dim() != 3:
            raise ValueError(
                "Expected last_hidden_states shape [B, S, D], "
                f"got {tuple(last_hidden_states.shape)}"
            )
        offsets = torch.arange(
            self.block_size,
            device=last_hidden_states.device,
            dtype=torch.long,
        ).view(1, 1, -1)
        indices = (local_anchors.to(last_hidden_states.device).unsqueeze(-1) + offsets - 1).clamp(
            min=0,
            max=last_hidden_states.shape[1] - 1,
        )
        gather_idx = indices.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            last_hidden_states.shape[-1],
        )
        gathered = torch.gather(
            last_hidden_states.unsqueeze(1).expand(-1, local_anchors.shape[1], -1, -1),
            2,
            gather_idx,
        )
        return gathered.contiguous()

    def _gather_sharded_teacher_rows(
        self,
        local_last_hidden_states: torch.Tensor,
        global_anchors: torch.Tensor,
        padded_seq_len: int,
    ) -> torch.Tensor:
        if local_last_hidden_states.dim() != 3:
            raise ValueError(
                "Expected sharded last_hidden_states shape [B, S/SP, D], "
                f"got {tuple(local_last_hidden_states.shape)}"
            )
        sp_size = self._sp_size()
        sp_rank = self._sp_rank()
        batch_size, local_seq_len, hidden_size = local_last_hidden_states.shape
        if padded_seq_len != local_seq_len * sp_size:
            raise ValueError(
                f"Sharded teacher length mismatch: local={local_seq_len}, "
                f"sp_size={sp_size}, padded_global={padded_seq_len}"
            )
        if global_anchors.shape[1] % sp_size != 0:
            raise ValueError(
                f"Anchor count {global_anchors.shape[1]} must divide sp_size {sp_size}"
            )
        local_anchor_count = global_anchors.shape[1] // sp_size
        anchors_cpu = global_anchors.detach().cpu()
        offsets = torch.arange(self.block_size, dtype=torch.long).view(1, 1, 1, -1)
        requests = torch.stack(
            [anchors_cpu[:, rank::sp_size] for rank in range(sp_size)],
            dim=0,
        )
        global_indices = (requests.unsqueeze(-1) + offsets - 1).clamp(
            min=0,
            max=padded_seq_len - 1,
        )
        owners = torch.div(global_indices, local_seq_len, rounding_mode="floor")
        local_indices = global_indices.remainder(local_seq_len)
        destination_ids = (
            torch.arange(sp_size, dtype=torch.long).view(sp_size, 1, 1, 1).expand_as(owners)
        )
        batch_ids = (
            torch.arange(batch_size, dtype=torch.long).view(1, batch_size, 1, 1).expand_as(owners)
        )

        send_mask = owners == sp_rank
        send_splits = (
            torch.bincount(destination_ids[send_mask], minlength=sp_size).to(torch.int64).tolist()
        )
        send_batch = batch_ids[send_mask].to(local_last_hidden_states.device)
        send_index = local_indices[send_mask].to(local_last_hidden_states.device)
        send_rows = local_last_hidden_states[send_batch, send_index].contiguous()

        local_owners = owners[sp_rank].reshape(-1)
        recv_splits = torch.bincount(local_owners, minlength=sp_size).to(torch.int64).tolist()
        recv_rows = torch.empty(
            sum(recv_splits),
            hidden_size,
            dtype=local_last_hidden_states.dtype,
            device=local_last_hidden_states.device,
        )
        dist.all_to_all_single(
            recv_rows,
            send_rows,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.sp_group,
        )

        recv_positions = torch.cat(
            [
                torch.nonzero(local_owners == owner, as_tuple=False).flatten()
                for owner in range(sp_size)
            ]
        ).to(local_last_hidden_states.device)
        gathered = torch.empty(
            batch_size * local_anchor_count * self.block_size,
            hidden_size,
            dtype=local_last_hidden_states.dtype,
            device=local_last_hidden_states.device,
        )
        gathered[recv_positions] = recv_rows
        return gathered.view(
            batch_size,
            local_anchor_count,
            self.block_size,
            hidden_size,
        )

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

    def _init_draft_attention_from_target(self, draft_model, target_model_path: str) -> None:
        """Copy target layers' ``input_layernorm`` and attention into the drafter
        (``target_attn_init_layers``).

        Rank 0, pre-FSDP, like ``_load_draft_init_weights``, which runs after it and so
        takes precedence; a checkpoint resume overwrites both.
        """
        init_layers = getattr(draft_model.config, "target_attn_init_layers", None)
        if init_layers is None:
            return
        from angelspec.models.target.target_attention_init import (
            init_attention_from_target,
        )

        layers = target_layers_per_draft_layer(init_layers, len(draft_model.layers))
        init_attention_from_target(draft_model, target_model_path, layers)
        logger.info(f"[Rank 0] draft attention initialized from target layers {layers}")

    def _init_draft_mlp_from_target(self, draft_model, target_model_path: str) -> None:
        """Copy target layers' post-attention norm and FFN weights into the drafter.

        ``target_mlp_init_layers`` copies the complete MoE (router, all routed
        experts, shared expert); ``target_mlp_dense_init_layers`` builds each dense
        FFN from the shared expert plus k routed experts. Runs rank-0/pre-FSDP
        before ``draft_init_path`` for the same precedence as target attention
        init, and every copied parameter remains trainable.
        """
        config = draft_model.config
        init_layers = getattr(config, "target_mlp_init_layers", None)
        if init_layers is not None:
            from angelspec.models.target.target_mlp_init import init_mlp_from_target

            layers = target_layers_per_draft_layer(init_layers, len(draft_model.layers))
            experts = init_mlp_from_target(
                draft_model,
                target_model_path,
                layers,
                config.target_mlp_init_experts,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            copied = "all routed experts" if experts is None else f"routed experts {experts}"
            logger.info(
                f"[Rank 0] draft post-attention norms and MoEs initialized from target layers "
                f"{layers}: shared expert, router and {copied}"
            )
        dense_layers = getattr(config, "target_mlp_dense_init_layers", None)
        if dense_layers is not None:
            from angelspec.models.target.target_mlp_init import (
                init_dense_mlp_from_target,
            )

            layers = target_layers_per_draft_layer(dense_layers, len(draft_model.layers))
            experts = init_dense_mlp_from_target(
                draft_model,
                target_model_path,
                layers,
                config.target_mlp_dense_init_experts,
                config.target_mlp_dense_init_routed_scale,
            )
            logger.info(
                f"[Rank 0] draft post-attention norms and dense FFNs initialized from target "
                f"layers {layers}: shared expert + routed experts {experts}"
            )

    def _init_mtp_fusion_from_target(self, draft_model, target_model_path: str) -> None:
        """Copy the target MTP layer's input fusion into the drafter's (``mtp_fusion_init``).

        Rank 0, pre-FSDP, before ``_load_draft_init_weights``: a ``draft_init_path``
        that carries fusion weights takes precedence, and one from a drafter trained
        without the fusion leaves this copy (and the zero gate) in place.
        """
        fusion = getattr(draft_model, "mtp_fusion", None)
        if fusion is None or draft_model.config.mtp_fusion_init != "target":
            return
        from angelspec.models.target.target_mtp_fusion import (
            init_mtp_fusion_from_target,
        )

        init_mtp_fusion_from_target(fusion, target_model_path)
        logger.info("[Rank 0] draft MTP fusion initialized from the target's MTP layer")

    def _load_draft_init_weights(self, draft_model) -> Optional[list]:
        """Seed the drafter from ``args.draft_init_path`` (rank 0, pre-FSDP); returns the
        drafter's keys the load left at init, or ``None`` without a ``draft_init_path``.

        An HF-style directory of the drafter's OWN weights — a previous run
        exported by ``tools/convert_to_hf.py`` (safetensors), or an ``hf_final``
        dir written by ``save_draft_model_for_serving`` (.bin).
        Distinct from ``load_path``, which is a DCP training checkpoint
        (step + optimizer + LR).

        Runs before ``apply_fsdp2``, so rank 0's ``state_dict()`` is what
        ``fsdp2_load_full_state_dict`` broadcasts to every rank. ``strict=False``
        because a warm start is legitimately partial in both directions: the MTP
        head carries no ``context_proj`` / ``hidden_correction`` (those keep
        their init), and a drafter trained with ``enable_hidden_correction`` has
        correction weights this config may not declare (those are dropped).
        """
        init_path = getattr(self.args, "draft_init_path", None)
        if not init_path:
            return None

        path = Path(init_path).expanduser()
        shards = None
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_file = path / index_name
            if index_file.exists():
                with open(index_file) as f:
                    shards = sorted(set(json.load(f)["weight_map"].values()))
                break
        if shards is None:
            for single_name in ("model.safetensors", "pytorch_model.bin"):
                if (path / single_name).exists():
                    shards = [single_name]
                    break
        if shards is None:
            raise FileNotFoundError(
                f"draft_init_path={init_path} has no "
                "model.safetensors(.index.json) or pytorch_model.bin(.index.json)"
            )

        state = {}
        for shard in shards:
            if shard.endswith(".safetensors"):
                from safetensors.torch import load_file

                state.update(load_file(str(path / shard)))
            else:
                state.update(torch.load(str(path / shard), map_location="cpu", weights_only=True))

        offline = bool(getattr(self.args, "offline_manifest_path", None))
        file_embedding = state.pop("embed_tokens.weight", None)
        target_embedding = draft_model.embed_tokens.weight
        if file_embedding is not None and not torch.equal(
            file_embedding.to(target_embedding.dtype), target_embedding
        ):
            if offline:
                raise ValueError("Offline warm-start embedding differs from frozen target")
            logger.warning(
                f"[Rank 0] draft init: {path} carries an embed_tokens that differs from the "
                "current target's; keeping the target's"
            )
        missing, unexpected = draft_model.load_state_dict(state, strict=False)
        missing = [k for k in missing if k != "embed_tokens.weight"]
        if offline and (missing or unexpected):
            raise RuntimeError(
                f"Offline warm start must match the draft exactly: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        logger.info(
            f"[Rank 0] draft init from {path}: loaded {len(state)} tensors "
            f"({len(missing)} left at init, {len(unexpected)} ignored)"
        )
        if missing:
            logger.info(f"[Rank 0] draft init: kept init values for {sorted(missing)}")
        if unexpected:
            logger.warning(f"[Rank 0] draft init: ignored unexpected keys {sorted(unexpected)}")
        return list(missing)

    def _init_attentive_correction(self, draft_model, init_missing: Optional[list]) -> None:
        """Start the attentive correction reading the context like its source draft layer's
        attention (``attentive_correction``), unless ``draft_init_path`` carried it.

        Rank 0, pre-FSDP, after ``_load_draft_init_weights``, so it copies the layer as it
        stands by then: the target's attention on a fresh run under
        ``target_attn_init_layers``, the warm start's otherwise. A warm start that lacks
        only block-read parameters gets just those: without the block keys
        (``attentive_correction_block_keys``) the whole block read; with them, the closed
        gate if the gated read's gate is missing, and the block read's own kv_b
        (``attentive_correction_block_kv_b='separate'``) as the loaded kv_b_proj if it is.
        A checkpoint resume overwrites it.
        """
        correction = getattr(draft_model, "hidden_correction", None)
        attention = getattr(correction, "attention", None)
        if attention is None:
            return
        prefix = "hidden_correction.attention."
        missing = (
            None if init_missing is None else {k for k in init_missing if k.startswith(prefix)}
        )
        if missing is not None and not missing:
            logger.info("[Rank 0] attentive correction: loaded from draft_init_path")
            return
        from angelspec.models.draft import attentive_correction

        layer = draft_model.config.attentive_correction_source_layer % len(draft_model.layers)
        source = draft_model.layers[layer].self_attn
        kv_in = "hidden half of kv_in" if attention.block_kv_in_embed else "kv_in"
        names = set() if missing is None else {k[len(prefix) :] for k in missing}
        kv_b = {n for n in names if n.startswith("block_kv_b_proj.")}
        gate = {n for n in names if n.startswith("block_")} - kv_b
        keys = {n for n in names if n.startswith(("kv_in.", "kv_a_layernorm.", "kv_norm."))}
        if missing is not None and names == keys | gate | kv_b:
            if keys:
                attentive_correction.init_block_keys_from_attention(attention, source)
                started = (
                    f"block keys as draft layer {layer}'s kv_a_proj_with_mqa ({kv_in})"
                    + (", gate at zero" if attention.block_gated else "")
                    + (", own kv_b as kv_b_proj" if attention.block_kv_b_separate else "")
                )
            else:
                parts = []
                if gate:
                    attentive_correction.init_block_gate_from_attention(attention, source)
                    parts.append(f"block gate at zero and block sink as draft layer {layer}'s")
                if kv_b:
                    attentive_correction.init_block_kv_b(attention)
                    parts.append("the block read's own kv_b as the loaded kv_b_proj")
                started = ", ".join(parts)
            logger.info(f"[Rank 0] attentive correction: loaded from draft_init_path, {started}")
            return
        attentive_correction.init_from_attention(attention, source)
        logger.info(
            f"[Rank 0] attentive correction starts as draft layer {layer}'s attention "
            "(hidden half of q_in, q_b, kv_b, sink"
            + (f", {kv_in} for the block keys" if attention.block_keys else "")
            + "), previous-token halves, o_proj"
            + (" and the block gate" if attention.block_gated else "")
            + " at zero"
            + ("; the block read's own kv_b as kv_b" if attention.block_kv_b_separate else "")
        )

    def _build_training_wrapper(self, draft_model):
        """Wrap ``draft_model`` in the training module (loss / forward plumbing).

        Subclasses (e.g. DFly) override this to swap in a wrapper that injects
        their extra behavior through the shared DFlash hooks.
        """
        usp = getattr(self.args, "attention_backend", None) == "usp"
        sp_size = int(getattr(self.args, "sp_ulysses_size", 1)) if usp else 1
        sp_rank = dist.get_rank(self.sp_group) if self.sp_group is not None else 0
        causal_attn = self.causal_attn
        self._log_usp_model_config(sp_size=sp_size, sp_rank=sp_rank, causal_attn=causal_attn)
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
            e2e_tv_vocab_chunk=self.e2e_tv_vocab_chunk,
            loss_checkpoint_chunks=getattr(self.args, "dflash_loss_checkpoint_chunks", False),
            path_selector_loss_alpha=self.path_selector_loss_alpha,
            path_selector_teacher_labels=self.path_selector_teacher_labels,
            path_selector_objective_weighting=self.path_selector_objective_weighting,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=self.sp_group,
            causal_attn=causal_attn,
            attn_diag_blocks=int(getattr(self.args, "dflash_attn_diag_blocks", 0) or 0),
            attn_oracle=getattr(self.args, "dflash_attn_oracle", "") or "",
            attn_oracle_layers=getattr(self.args, "dflash_attn_oracle_layers", None),
            hc_topk_diag=getattr(self.args, "dflash_hc_topk_diag", None),
            eval_accept_temperature=getattr(self.args, "dflash_eval_accept_temperature", None),
            attentive_correction_eval_off=bool(getattr(self.args, "dflash_acorr_eval_off", False)),
        )

    def _log_usp_model_config(self, *, sp_size: int, sp_rank: int, causal_attn: bool) -> None:
        """Emit a one-line USP enablement summary (rank 0)."""
        if getattr(self.args, "attention_backend", None) != "usp":
            return
        if dist.get_rank() != 0:
            return
        anchors_local = self.num_anchors // max(sp_size, 1)
        logger.info(
            f"[USP] draft wrapper: sp_size={sp_size} sp_rank={sp_rank} "
            f"usp_local_shard={bool(getattr(self.args, 'usp_local_shard', False))} "
            f"micro_batch_size={int(getattr(self.args, 'micro_batch_size', 1))} "
            f"num_anchors={self.num_anchors} -> anchors/rank={anchors_local} "
            f"block_size={self.block_size} causal_attn={causal_attn} "
            f"(context shard S/{sp_size}, gather full ctx K/V in attention)"
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

            from angelspec.training.position_config import configure_draft_positions

            configure_draft_positions(config, target_model_path)
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

            # A complete five-layer HYV4 MoE has 8.34B FFN parameters.  Building
            # rank 0 in fp32 only to copy bf16 target tensors and cast the whole
            # model afterwards needlessly peaks around 50 GB of host memory.
            # Construct this target-initialized variant directly in bf16; non-rank-0
            # construction is on meta and does not allocate either way.
            direct_bf16 = (
                dist.get_rank() == 0
                and getattr(config, "draft_mlp_type", "dense") == "moe"
                and getattr(config, "target_mlp_init_layers", None) is not None
            )
            previous_dtype = torch.get_default_dtype()
            if direct_bf16:
                torch.set_default_dtype(torch.bfloat16)
            try:
                draft_model = self._build_draft_model(config)
            finally:
                if direct_bf16:
                    torch.set_default_dtype(previous_dtype)

        if dist.get_rank() == 0:
            draft_model.load_embedding(
                target_model_path,
                embedding_key=getattr(self.args, "embedding_key", "model.embed_tokens.weight"),
            )
            self._init_draft_attention_from_target(draft_model, target_model_path)
            self._init_draft_mlp_from_target(draft_model, target_model_path)
            self._init_mtp_fusion_from_target(draft_model, target_model_path)
            init_missing = self._load_draft_init_weights(draft_model)
            self._init_attentive_correction(draft_model, init_missing)

        draft_model.freeze_embedding()
        draft_model = draft_model.to(torch.bfloat16)
        self._moe_aux_metrics: dict = {}
        if getattr(config, "moe_aux_free_bias_update", False):
            # Module.to(bf16) also casts floating buffers. Counts and their EMA
            # are controller state and must remain fp32.
            from angelspec.models.draft.mtp import Hy3MoE

            for module in draft_model.modules():
                if isinstance(module, Hy3MoE) and module.aux_free_bias_update:
                    module._expert_load_counts.data = module._expert_load_counts.data.float()
                    module.expert_load_ema.data = module.expert_load_ema.data.float()

        if getattr(self.args, "gradient_checkpointing", False):
            enable_dflash_activation_checkpointing(draft_model)
            logger.info(
                "DFlash gradient checkpointing on: %d decoder layers, non-reentrant",
                len(draft_model.layers),
            )

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] DFlash draft model: {trainable_count:,} trainable, "
            f"{frozen_count:,} frozen (embedding) parameters"
        )

        dflash_model = self._build_training_wrapper(draft_model)

        full_state = dflash_model.state_dict() if dist.get_rank() == 0 else {}

        # USP FULL_SHARD must use the HSDP mesh, not draft_dp. draft_dp is a
        # 1D slice of same-SP ranks, so wrapping on it creates one independent
        # FSDP replica per SP rank: grads never SUM across the sequence split
        # and the copies diverge after the first step. fsdp_mesh puts SP on the
        # HSDP replicate axis; trainer.mul_(sp_size) then turns that mean into
        # the SP-SUM the DFlash loss contract needs (same as MTP).
        dflash_model = apply_fsdp2(
            dflash_model,
            mesh=self.fsdp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            modules_to_shard=[
                *fused_expert_shard_units(draft_model),
                *draft_model.layers,
            ],
        )

        dflash_model = fsdp2_load_full_state_dict(
            dflash_model,
            full_state,
            self.fsdp_mesh,
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

        moe_scale = float(getattr(self.args, "moe_lr_scale", 1.0))
        router_scale = getattr(self.args, "moe_router_lr_scale", None)
        router_scale = moe_scale if router_scale is None else float(router_scale)
        self._moe_lr_scales = (moe_scale, router_scale)
        lr_scales = {}
        if self._moe_lr_scales != (1.0, 1.0):
            lr_scales = moe_lr_scales(self.draft_model, moe_scale, router_scale)
            if not lr_scales:
                raise ValueError("moe_lr_scale is set but the draft has no MoE FFN")

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
            muon_weight_decay=getattr(self.args, "muon_weight_decay", None),
            lr_scales=lr_scales,
            muon_per_expert=getattr(self.args, "muon_per_expert", True),
            muon_head_rows=(
                muon_head_rows(self.draft_model)
                if getattr(self.args, "muon_per_head", True)
                else None
            ),
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
                # The config always defines wsd_decay_style, as None when unset.
                wsd_decay_style=getattr(self.args, "wsd_decay_style", None)
                or getattr(self.args, "lr_wsd_decay_style", None)
                or "cosine",
            )

        self.lr_scheduler = self.optimizer.lr_scheduler

        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)

        self._init_target_lm_head(target_model_path)
        self._init_target_indexer(target_model_path)
        self._init_target_kv_latents(target_model_path)
        self._init_attn_diag_teachers(target_model_path)

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

    def _mtp_fusion_on(self) -> bool:
        """Whether the drafter fuses the target's final hidden into its block input."""
        draft_model = getattr(getattr(self, "dflash", None), "draft_model", None)
        return getattr(draft_model, "mtp_fusion", None) is not None

    def _init_target_lm_head(self, target_model_path: str) -> None:
        from angelspec.models.target.target_utils import TargetLMHead

        # KL/LK distillation needs the target's final RMSNorm to normalise the
        # (pre-norm) last_hidden_states from mooncake before the lm_head, and the MTP
        # fusion reads them through it. Skipped for the base CE path (load_norm=False →
        # norm=None → target_norm=None).
        load_norm = (
            self._distill_enabled
            or self.l1_loss_alpha > 0
            or self.dflash._extra_distill_needed()
            or (self.args.dflash_opd_enabled and self.args.dflash_opd_backend == "branch_requests")
            or self._mtp_fusion_on()
        )

        from angelspec.utils.hf_config_registry import register_extra_hf_configs

        register_extra_hf_configs()

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

    def _init_target_indexer(self, target_model_path: str) -> None:
        """Frozen target-layer indexers for drafters with ``target_index_layer`` set.

        One per distinct target layer: rank 0 reads the eight tensors each needs and
        broadcasts them, like the LM head. The wrapper only keeps references, so the
        weights stay out of FSDP and out of the drafter's checkpoints.
        """
        config = self.draft_model.config
        index_layer = getattr(config, "target_index_layer", None)
        if index_layer is None:
            return
        from angelspec.models.target.target_indexer import TargetIndexer

        topk = int(config.target_index_topk)
        union_layers = getattr(config, "target_index_union_layers", None)
        if union_layers is not None:
            layers = list(dict.fromkeys(int(x) for x in union_layers))
        else:
            layers = list(
                dict.fromkeys(
                    layer
                    for layer in target_layers_per_draft_layer(
                        index_layer, config.num_hidden_layers
                    )
                    if layer is not None
                )
            )
        build = (
            TargetIndexer.from_pretrained if dist.get_rank() == 0 else TargetIndexer.from_config
        )
        indexers = [build(target_model_path, layer, topk, device="cuda") for layer in layers]

        dist.barrier()
        for indexer in indexers:
            for param in indexer.parameters():
                dist.broadcast(param.data, src=0)

        self.target_indexers = indexers
        self.dflash.attach_target_indexers(indexers)
        logger.info(
            f"[Rank {self.dp_rank}] TargetIndexer (target layers {layers}, top-{topk}) "
            "initialized and synced"
        )

    def _init_target_kv_latents(self, target_model_path: str) -> None:
        """Frozen target-layer MLA latent projections for drafters with ``target_kv_layers``
        set, one per distinct target layer, loaded and synced like the indexers."""
        config = self.draft_model.config
        kv_layers = getattr(config, "target_kv_layers", None)
        if kv_layers is None:
            return
        from angelspec.models.target.target_kv_latent import TargetKVLatent

        layers = list(
            dict.fromkeys(target_layers_per_draft_layer(kv_layers, config.num_hidden_layers))
        )
        build = (
            TargetKVLatent.from_pretrained if dist.get_rank() == 0 else TargetKVLatent.from_config
        )
        modules = [build(target_model_path, layer, device="cuda") for layer in layers]

        dist.barrier()
        for module in modules:
            for param in module.parameters():
                dist.broadcast(param.data, src=0)

        self.target_kv_latents = modules
        self.dflash.attach_target_kv_latents(modules)
        logger.info(
            f"[Rank {self.dp_rank}] TargetKVLatent (target layers {layers}) initialized and synced"
        )

    def _init_attn_diag_teachers(self, target_model_path: str) -> None:
        """Frozen target attention copies for the eval-time attention diagnostic
        (``dflash_attn_diag_blocks``) or intervention (``dflash_attn_oracle``), one per
        target layer in ``target_kv_layers``, loaded and synced like the indexers."""
        if self.dflash.attn_diag_blocks <= 0 and not self.dflash.attn_oracle:
            return
        config = self.draft_model.config
        kv_layers = getattr(config, "target_kv_layers", None)
        if kv_layers is None:
            raise ValueError(
                "dflash_attn_diag_blocks / dflash_attn_oracle compare against "
                "target_kv_layers; set them"
            )
        from angelspec.models.target.target_attention_init import TargetAttentionTeacher

        layers = list(
            dict.fromkeys(target_layers_per_draft_layer(kv_layers, config.num_hidden_layers))
        )
        if dist.get_rank() == 0:
            modules = [
                TargetAttentionTeacher.from_pretrained(target_model_path, config, layer)
                for layer in layers
            ]
        else:
            modules = [TargetAttentionTeacher.from_config(config, layer) for layer in layers]

        dist.barrier()
        for module in modules:
            for param in module.parameters():
                dist.broadcast(param.data, src=0)

        self.dflash.attach_attn_diag_teachers(modules)
        if self.dflash.attn_oracle:
            logger.info(
                f"[Rank {self.dp_rank}] attention oracle '{self.dflash.attn_oracle}' in draft "
                f"layers {self.dflash.attn_oracle_layers or 'all'} from target layers {layers}"
            )
        else:
            logger.info(
                f"[Rank {self.dp_rank}] attention diagnostic: {self.dflash.attn_diag_blocks} "
                f"blocks per sample against target layers {layers}"
            )

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _split_hidden_states(
        self, hidden_states: torch.Tensor, last_hidden_states: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Split concatenated hidden states [B, seq_len, num_layers*D] into per-layer list.

        The target model concatenates hidden states from `num_target_layers` layers
        along the last dimension. We split them back into a list of [B, seq_len, D] tensors.

        When the last requested target layer is the target's final layer, vllm
        captures it into ``last_hidden_states`` instead (same pre-norm residual
        stream, just a different transport), so the concatenation holds one layer
        less and that tensor becomes the final context layer.
        """
        num_concatenated = self.num_target_layers
        if self.final_layer_from_last_hidden:
            if last_hidden_states is None:
                raise ValueError(
                    "target_layer_ids requests the target's final layer, which arrives as "
                    "last_hidden_states, but the batch has none. Check that the engine "
                    "stores it (store_last_hidden_states) and that the collator kept it."
                )
            num_concatenated -= 1
        per_layer_dim = hidden_states.shape[-1] // num_concatenated
        layers = list(hidden_states.split(per_layer_dim, dim=-1))
        if self.final_layer_from_last_hidden:
            layers.append(last_hidden_states)
        return layers

    def _forward(
        self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        device = torch.device("cuda")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        hidden_states = batch["hidden_states"]
        usp = self._defer_batch_h2d()
        sp_size = self._sp_size()
        sp_rank = self._sp_rank()
        padded_seq_len = (
            ((input_ids.shape[1] + sp_size - 1) // sp_size) * sp_size
            if usp
            else input_ids.shape[1]
        )
        sequence_sharded = bool(batch.pop("_usp_sequence_sharded", False))
        if usp and not sequence_sharded:
            hidden_states = slice_usp_sequence_shard(
                hidden_states,
                sp_rank=sp_rank,
                sp_size=sp_size,
                padded_seq_len=padded_seq_len,
            )
        hidden_states = hidden_states.to(device, non_blocking=True)

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
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(device, non_blocking=True)

        # Target final hidden states + norm — only needed for L1/KL/LK distillation
        # (None on the base CE path, in which case those loss terms are skipped).
        raw_last_hidden_states = batch.get("last_hidden_states")
        context_last_hidden_states = (
            raw_last_hidden_states if self.final_layer_from_last_hidden else None
        )
        if usp and context_last_hidden_states is not None and not sequence_sharded:
            context_last_hidden_states = slice_usp_sequence_shard(
                context_last_hidden_states,
                sp_rank=sp_rank,
                sp_size=sp_size,
                padded_seq_len=padded_seq_len,
            )
        if context_last_hidden_states is not None:
            context_last_hidden_states = context_last_hidden_states.to(device, non_blocking=True)

        injected_anchors = None
        injected_keep_mask = None
        teacher_hidden_at_labels = None
        last_hidden_states = None
        needs_teacher_hidden = (
            self._distill_enabled
            or self.l1_loss_alpha > 0
            or self.dflash._extra_distill_needed()
            or self._branch_opd_on()
        )
        if needs_teacher_hidden and raw_last_hidden_states is not None:
            if usp:
                injected_anchors, injected_keep_mask, local_anchors = self._sample_usp_anchors(
                    input_ids,
                    loss_mask,
                    ctx_doc_ids,
                    padded_seq_len,
                    position_ids=position_ids,
                )
                if sequence_sharded:
                    local_teacher_hidden = (
                        context_last_hidden_states
                        if context_last_hidden_states is not None
                        else raw_last_hidden_states.to(device, non_blocking=True)
                    )
                    teacher_hidden_at_labels = self._gather_sharded_teacher_rows(
                        local_teacher_hidden,
                        injected_anchors,
                        padded_seq_len,
                    )
                else:
                    teacher_hidden_at_labels = self._gather_teacher_rows(
                        raw_last_hidden_states,
                        local_anchors,
                    )
                teacher_hidden_at_labels = teacher_hidden_at_labels.to(
                    device,
                    non_blocking=True,
                )
            else:
                last_hidden_states = raw_last_hidden_states.to(device, non_blocking=True)
        # The MTP fusion reads the final hidden at every anchor's predecessor, so it needs
        # the whole sequence too (it rejects USP, where only teacher rows travel).
        fusion_on = self._mtp_fusion_on()
        if fusion_on and last_hidden_states is None and raw_last_hidden_states is not None:
            last_hidden_states = raw_last_hidden_states.to(device, non_blocking=True)
        target_norm = (
            getattr(self.target_lm_head, "norm", None)
            if needs_teacher_hidden or fusion_on
            else None
        )

        hidden_states_list = self._split_hidden_states(
            hidden_states,
            context_last_hidden_states,
        )
        del hidden_states

        opd_on = bool(self.score_engine) and getattr(self.args, "dflash_opd_enabled", False)
        # Warm the verify engine's prefix cache with this trunk while the draft
        # forward runs (no-op unless the branch backend is selected).
        self._prime_branch_opd(batch, input_ids)
        draft_kwargs = (
            {"opd_on_policy_prev": True}
            if opd_on and getattr(self.args, "dflash_opd_on_policy_prev", False)
            else {}
        )
        out = self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=self.target_lm_head_weight,
            last_hidden_states=last_hidden_states,
            teacher_hidden_at_labels=teacher_hidden_at_labels,
            target_norm=target_norm,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            position_ids=position_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
            context_pre_sharded=usp,
            return_draft=opd_on,
            **draft_kwargs,
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
            if self._branch_opd_on():
                opd_loss, opd_metrics = self._compute_opd_loss_branch(
                    input_ids, opd, last_hidden_states, target_norm
                )
            else:
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

    # Number of leading backward calls to probe; backward is where this model
    # peaks (FSDP unshards a whole MoE layer), so the first few carry the signal.
    _MEM_PROBE_CALLS = 3

    def _log_backward_memory(self, when: str) -> None:
        """Log allocator + driver memory around backward for the first few calls.

        Driver-free bytes matter independently of the allocator's own counters:
        the DSA sparse-MLA backward launches Triton kernels whose local memory is
        allocated by the driver, outside PyTorch's pool, and PyTorch does not
        return reserved memory on its own. That combination surfaces as
        "Triton Error [CUDA]: out of memory" at kernel launch rather than a
        torch.OutOfMemoryError, with no numbers attached — hence this probe.
        """
        n = getattr(self, "_mem_probe_calls", 0)
        if n >= self._MEM_PROBE_CALLS:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        gib = float(1024**3)
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "[mem] backward#%d %s: allocated=%.2f reserved=%.2f peak_alloc=%.2f "
            "peak_reserved=%.2f driver_free=%.2f/%.2f GiB",
            n,
            when,
            torch.cuda.memory_allocated() / gib,
            torch.cuda.memory_reserved() / gib,
            torch.cuda.max_memory_allocated() / gib,
            torch.cuda.max_memory_reserved() / gib,
            free / gib,
            total / gib,
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
        self._log_backward_memory("pre")
        scaled_loss.backward()
        self._log_backward_memory("post")
        self._mem_probe_calls = getattr(self, "_mem_probe_calls", 0) + 1
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

        grad_sync_mesh = getattr(self, "grad_sync_mesh", None)
        group = grad_sync_mesh.get_group() if grad_sync_mesh is not None else None
        reduce_kwargs = {"group": group} if group is not None else {}
        dist.all_reduce(loss_sum_pp, op=dist.ReduceOp.SUM, **reduce_kwargs)
        dist.all_reduce(correct_sum_pp, op=dist.ReduceOp.SUM, **reduce_kwargs)
        dist.all_reduce(count_pp, op=dist.ReduceOp.SUM, **reduce_kwargs)

        safe_count_pp = count_pp.clamp(min=1.0)
        avg_loss_pp = loss_sum_pp / safe_count_pp
        avg_acc_pp = correct_sum_pp / safe_count_pp
        return avg_loss_pp, avg_acc_pp, count_pp

    def _reduce_teacher_acc_pp(
        self, all_step_metrics: list[dict], *, count_key: str
    ) -> Optional[torch.Tensor]:
        """Count-weighted global mean of the teacher-argmax per-position accuracy.

        ``None`` when the model emitted no teacher logits (every distill term
        off). Presence is decided by config, not by data, so all ranks agree and
        the collective below stays symmetric.
        """
        return self._reduce_position_rate(
            all_step_metrics, key="teacher_acc_per_position", count_key=count_key
        )

    def _reduce_position_rate(
        self, all_step_metrics: list[dict], *, key: str, count_key: str
    ) -> Optional[torch.Tensor]:
        """Count-weighted global mean of the per-position rate ``key``; ``None`` when absent.

        Every forward emits ``key`` or none does (decided by config), so the
        collective stays symmetric across ranks.
        """
        vals = [m for m in all_step_metrics if key in m]
        if not vals:
            return None

        correct_sum_pp = torch.stack([m[key] * m[count_key] for m in vals]).sum(dim=0)
        count_pp = torch.stack([m[count_key] for m in vals]).sum(dim=0)

        group = self.grad_sync_mesh.get_group() if self.grad_sync_mesh is not None else None
        dist.all_reduce(correct_sum_pp, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(count_pp, op=dist.ReduceOp.SUM, group=group)
        return correct_sum_pp / count_pp.clamp(min=1.0)

    @property
    def _first_pred_slot(self) -> int:
        return getattr(getattr(self, "dflash", None), "first_pred_slot", 1)

    @staticmethod
    def _simulated_acc_len(pred_acc_pp: torch.Tensor) -> float:
        """Expected consecutively-accepted draft tokens under per-position
        independence: acc_0 + acc_0*acc_1 + ... + prod(acc_0..acc_{B-2})."""
        cumulative = 1.0
        total = 0.0
        for i in range(pred_acc_pp.shape[0]):
            cumulative *= pred_acc_pp[i].item()
            total += cumulative
        return total

    def _reduce_loss_components(self, all_step_metrics: list[dict], prefix: str) -> dict:
        """Reduce extra scalar loss components into ``{prefix}{key}`` global means."""
        out: dict = {}
        for key in self._extra_loss_component_keys:
            vals = [m[key] for m in all_step_metrics if key in m]
            if not vals:
                continue
            value = torch.stack([v.float() for v in vals]).mean()
            if dist.is_initialized() and dist.get_world_size() > 1:
                grad_sync_mesh = getattr(self, "grad_sync_mesh", None)
                group = grad_sync_mesh.get_group() if grad_sync_mesh is not None else None
                reduce_kwargs = {"group": group} if group is not None else {}
                dist.all_reduce(value, op=dist.ReduceOp.SUM, **reduce_kwargs)
                value = value / dist.get_world_size(group)
            out[f"{prefix}{key}"] = value.item()
        return out

    def _reduce_index_metrics(self, all_step_metrics: list[dict], prefix: str) -> dict:
        """DP-reduce the candidate-set metrics' sum/count scalars into ``{prefix}{key}``.

        Weighted by blocks rather than averaged per rank: an eval set with fewer samples
        than ranks leaves ranks with nothing to report, and a plain mean folded their
        zeros in (codespan_64k, 32 samples over 64 ranks, read half its real size).
        Every rank hands over the same keys, so the collective stays symmetric; a metric
        this run does not produce comes back with count 0 and is left out.
        """
        from angelspec.models.dflash import INDEX_METRIC_KEYS

        dicts = [m["_index"] for m in all_step_metrics if m.get("_index")]
        if not dicts:
            return {}
        flat = torch.stack(
            [
                torch.stack([d[key][part] for d in dicts]).float().sum()
                for key in INDEX_METRIC_KEYS
                for part in (0, 1)
            ]
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            grad_sync_mesh = getattr(self, "grad_sync_mesh", None)
            group = grad_sync_mesh.get_group() if grad_sync_mesh is not None else None
            dist.all_reduce(
                flat, op=dist.ReduceOp.SUM, **({"group": group} if group is not None else {})
            )
        out = {}
        for i, key in enumerate(INDEX_METRIC_KEYS):
            total, count = flat[2 * i], flat[2 * i + 1]
            if float(count) > 0:
                out[f"{prefix}{key}"] = (total / count).item()
        return out

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
        if self._eval_collator.usp_sequence_sharded and eval_mbs != 1:
            logger.warning(
                "Using eval_micro_batch_size=1 for sequence-sharded cached samples "
                "to preserve global shard offsets (requested %s)",
                eval_mbs,
            )
            eval_mbs = 1

        self.model.eval()
        all_metrics: list[dict] = []
        for i in range(0, len(self._eval_cache), eval_mbs):
            chunk = self._eval_cache[i : i + eval_mbs]
            batch = self._eval_collator(chunk)
            if not self._defer_batch_h2d():
                batch = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }
            all_metrics.append(self.eval_forward(batch))

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
        first = self._first_pred_slot
        pred_loss_pp = avg_loss_pp[first:]
        pred_acc_pp = avg_acc_pp[first:]
        pred_count_pp = count_pp[first:]

        simulated_acc_len = self._simulated_acc_len(pred_acc_pp)

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

        tgt_acc_pp = self._reduce_teacher_acc_pp(all_step_metrics, count_key="count_pp")
        if tgt_acc_pp is not None:
            pred_tgt_acc_pp = tgt_acc_pp[first:]
            _, avg_acc_tgt = self._compute_scalar_metrics(
                pred_loss_pp, pred_tgt_acc_pp, pred_count_pp
            )
            metrics["eval/avg_acc_tgt"] = avg_acc_tgt
            metrics["eval/simulated_acc_len_tgt"] = self._simulated_acc_len(pred_tgt_acc_pp)
            for i in range(pred_tgt_acc_pp.shape[0]):
                metrics[f"eval/acc_tgt_{i}"] = pred_tgt_acc_pp[i].item()
        recall_pp = self._reduce_position_rate(
            all_step_metrics, key="hc_topk_recall_per_position", count_key="count_pp"
        )
        if recall_pp is not None:
            for i, value in enumerate(recall_pp[first:].tolist()):
                metrics[f"eval/hc_topk_recall_{i}"] = value
        for name in ("sample_accept", "argmax_accept"):
            accept_pp = self._reduce_position_rate(
                all_step_metrics, key=f"{name}_per_position", count_key="count_pp"
            )
            if accept_pp is not None:
                metrics[f"eval/{name}_len"] = self._simulated_acc_len(accept_pp[first:])
                for i, value in enumerate(accept_pp[first:].tolist()):
                    metrics[f"eval/{name}_{i}"] = value

        metrics.update(self._reduce_loss_components(all_step_metrics, "eval/"))
        metrics.update(self._reduce_index_metrics(all_step_metrics, "eval/"))
        metrics.update(self._reduce_attn_diag(all_step_metrics))
        metrics.update(self._reduce_hc_topk_diag(all_step_metrics))

        if dist.get_rank() == 0:
            tgt_note = (
                f", sim_acc_len_tgt={metrics['eval/simulated_acc_len_tgt']:.2f}"
                if "eval/simulated_acc_len_tgt" in metrics
                else ""
            )
            for name in ("sample_accept_len", "argmax_accept_len"):
                if f"eval/{name}" in metrics:
                    tgt_note += f", {name}={metrics[f'eval/{name}']:.2f}"
            logger.info(
                f"eval: loss={weighted_avg_loss:.4f}, acc={avg_acc:.4f}, "
                f"sim_acc_len={simulated_acc_len:.2f}{tgt_note}"
            )

        return metrics

    def _reduce_attn_diag(self, all_step_metrics: list[dict]) -> dict:
        """Sum the attention diagnostic's statistics over batches and ranks, then average them.

        Every eval forward emits the same keys while the diagnostic is on, so the
        collective stays symmetric across ranks.
        """
        keys = sorted({key for m in all_step_metrics for key in m if key.startswith("attn_diag/")})
        if not keys:
            return {}
        sums = torch.stack(
            [
                torch.stack([m[key].double() for m in all_step_metrics if key in m]).sum()
                for key in keys
            ]
        )
        grad_sync_mesh = getattr(self, "grad_sync_mesh", None)
        group = grad_sync_mesh.get_group() if grad_sync_mesh is not None else None
        dist.all_reduce(
            sums, op=dist.ReduceOp.SUM, **({"group": group} if group is not None else {})
        )
        from angelspec.models.dflash_attn_diag import summarize_attn_diag

        summary = summarize_attn_diag(dict(zip(keys, sums.tolist())))
        return {f"eval/{key}": value for key, value in summary.items()}

    def _reduce_hc_topk_diag(self, all_step_metrics: list[dict]) -> dict:
        """Average the top-K hidden-correction diagnostic per slot over batches and ranks.

        Every eval forward emits the same ``hc_topk/<metric>/s<slot>`` sums while the
        diagnostic is on, so the collective stays symmetric across ranks. ``all`` is
        the count-weighted mean over slots.
        """
        keys = sorted({key for m in all_step_metrics for key in m if key.startswith("hc_topk/")})
        if not keys:
            return {}
        sums = torch.stack(
            [
                torch.stack([m[key].double() for m in all_step_metrics if key in m]).sum()
                for key in keys
            ]
        )
        grad_sync_mesh = getattr(self, "grad_sync_mesh", None)
        group = grad_sync_mesh.get_group() if grad_sync_mesh is not None else None
        dist.all_reduce(
            sums, op=dist.ReduceOp.SUM, **({"group": group} if group is not None else {})
        )
        totals = dict(zip(keys, sums.tolist()))
        names = sorted({key.split("/")[1] for key in keys} - {"count"})
        slots = sorted({key.split("/")[2] for key in keys}, key=lambda s: int(s[1:]))
        out: dict = {}
        for name in names:
            value_sum = count_sum = 0.0
            for slot in slots:
                count = totals.get(f"hc_topk/count/{slot}", 0.0)
                value = totals.get(f"hc_topk/{name}/{slot}", 0.0)
                out[f"eval/hc_topk/{name}/{slot}"] = value / count if count > 0 else float("nan")
                value_sum += value
                count_sum += count
            out[f"eval/hc_topk/{name}/all"] = (
                value_sum / count_sum if count_sum > 0 else float("nan")
            )
        return out

    def _aggregate_named_eval_metrics(self, all_step_metrics: list[dict]) -> dict:
        dropped = set(self._extra_loss_component_keys) - set(self._named_eval_component_keys)
        metrics = [
            {key: value for key, value in item.items() if key not in dropped}
            for item in all_step_metrics
        ]
        return self._aggregate_eval_metrics(metrics)

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
        self._opd_last_verify = {}
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
            "_opd_verify": dict(self._opd_last_verify),
            "_fwd_events": (evt_fwd_s, evt_fwd_e),
            "_bwd_events": (evt_bwd_s, evt_bwd_e),
            **loss_components,
        }

    @torch.no_grad()
    def _after_optimizer_step(self, step: int) -> None:
        """All-reduce MoE loads and apply the aux-loss-free expert-bias controller."""
        self._moe_aux_metrics = {}
        config = getattr(getattr(self, "draft_model", None), "config", None)
        if not getattr(config, "moe_aux_free_bias_update", False):
            return

        from angelspec.models.draft.mtp import Hy3MoE

        for name, module in self.draft_model.named_modules():
            if not isinstance(module, Hy3MoE) or not module.aux_free_bias_update:
                continue
            counts = module.take_expert_load_counts()
            if counts is None:
                continue
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            total = counts.sum()
            if total <= 0:
                continue
            fractions = counts / total
            ema = module.update_expert_load_ema(fractions)
            target = torch.full_like(ema, 1.0 / module.num_experts)
            direction = torch.sign(target - ema)
            # A common shift changes no top-k ordering. Centering avoids wasting
            # BF16 dynamic range on that null direction.
            direction.sub_(direction.mean())
            delta = direction * module.aux_free_bias_gamma
            self.optimizer.add_to_parameter(f"{name}.expert_bias", delta)

            layer = name.split(".")[1] if name.startswith("layers.") else name.replace(".", "_")
            prefix = f"train/moe_layer_{layer}"
            mean = fractions.mean().clamp_min(1e-12)
            entropy = -(
                fractions.clamp_min(1e-20) * fractions.clamp_min(1e-20).log()
            ).sum() / math.log(module.num_experts)
            self._moe_aux_metrics.update(
                {
                    f"{prefix}/used_experts": int((counts > 0).sum().item()),
                    f"{prefix}/max_share": fractions.max().item(),
                    f"{prefix}/load_cv": (fractions.std() / mean).item(),
                    f"{prefix}/entropy": entropy.item(),
                    f"{prefix}/ema_max_share": ema.max().item(),
                    f"{prefix}/bias_update_abs": delta.abs().mean().item(),
                }
            )

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
        # so the naming matches Eagle3 (acc_0 = first predicted token). Under
        # sample_from_anchor slot 0 is the first prediction and stays.
        first = self._first_pred_slot
        pred_loss_pp = avg_loss_pp[first:]
        pred_acc_pp = avg_acc_pp[first:]
        pred_count_pp = count_pp[first:]

        simulated_acc_len = self._simulated_acc_len(pred_acc_pp)

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
        moe_scale, router_scale = getattr(self, "_moe_lr_scales", (1.0, 1.0))
        if moe_scale != 1.0:
            metrics["train/lr_moe"] = self.optimizer.get_learning_rate(moe_scale)
        if router_scale != moe_scale:
            metrics["train/lr_moe_router"] = self.optimizer.get_learning_rate(router_scale)
        metrics.update(getattr(self, "_moe_aux_metrics", {}))

        for i in range(pred_loss_pp.shape[0]):
            metrics[f"train/ploss_{i}"] = pred_loss_pp[i].item()
            metrics[f"train/acc_{i}"] = pred_acc_pp[i].item()

        # ``*_tgt`` = scored against the target's argmax (the greedy serving
        # accept test) rather than the corpus token; the unsuffixed keys above
        # keep their historical ground-truth meaning.
        tgt_acc_pp = self._reduce_teacher_acc_pp(all_step_metrics, count_key="count_per_position")
        if tgt_acc_pp is not None:
            pred_tgt_acc_pp = tgt_acc_pp[first:]
            _, avg_acc_tgt = self._compute_scalar_metrics(
                pred_loss_pp, pred_tgt_acc_pp, pred_count_pp
            )
            metrics["train/avg_acc_tgt"] = avg_acc_tgt
            metrics["train/simulated_acc_len_tgt"] = self._simulated_acc_len(pred_tgt_acc_pp)
            for i in range(pred_tgt_acc_pp.shape[0]):
                metrics[f"train/acc_tgt_{i}"] = pred_tgt_acc_pp[i].item()
        recall_pp = self._reduce_position_rate(
            all_step_metrics, key="hc_topk_recall_per_position", count_key="count_per_position"
        )
        if recall_pp is not None:
            for i, value in enumerate(recall_pp[first:].tolist()):
                metrics[f"train/hc_topk_recall_{i}"] = value

        metrics.update(self._reduce_loss_components(all_step_metrics, "train/"))
        metrics.update(self._reduce_index_metrics(all_step_metrics, "train/"))

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
        if self._branch_opd_on():
            names = (
                "rpc_seconds",
                "warm_seconds",
                "prepare_time",
                "generate_time",
                "capture_time",
                "cached_tokens",
                "computed_tokens",
                "branches",
            )
            local = torch.tensor(
                [sum(m.get("_opd_verify", {}).get(k, 0) for m in all_step_metrics) for k in names]
                + [fwd_ms / 1000.0, bwd_ms / 1000.0],
                device="cuda",
                dtype=torch.float64,
            )
            total, maximum = local.clone(), local.clone()
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            for i, key in enumerate((*names, "forward_phase_s", "backward_phase_s")):
                metrics[f"opd_timing/{key}_mean"] = float(total[i] / dist.get_world_size())
                metrics[f"opd_timing/{key}_max"] = float(maximum[i])
            if dist.get_rank() == 0:
                logger.info(
                    "OPD_TIMING step=%s %s",
                    step + 1,
                    {k: v for k, v in metrics.items() if k.startswith("opd_timing/")},
                )

        # CUDA memory snapshot for USP A/B (allocated / reserved / peak since last reset).
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            metrics["perf/mem_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
            metrics["perf/mem_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
            metrics["perf/mem_peak_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)

        if dist.get_rank() == 0 and (step % 5 == 0 or step <= 5):
            mem_suffix = ""
            if "perf/mem_allocated_gb" in metrics:
                mem_suffix = (
                    f" mem_alloc={metrics['perf/mem_allocated_gb']:.2f}GiB"
                    f" mem_rsv={metrics['perf/mem_reserved_gb']:.2f}GiB"
                    f" mem_peak={metrics['perf/mem_peak_gb']:.2f}GiB"
                )
            usp_tag = " USP" if getattr(self.args, "attention_backend", None) == "usp" else ""
            logger.info(
                f"COMPUTE_BREAKDOWN step={step}:{usp_tag} "
                f"forward={fwd_ms:.1f}ms backward={bwd_ms:.1f}ms{mem_suffix}"
            )

        # Layer-gate read-out (gated_sum selection run only; all-rank collective).
        metrics.update(self._gate_metrics())

        # OPD metrics use global branch counts across ranks and micro-batches.
        opd_metrics = self._reduce_opd_metrics(all_step_metrics)
        metrics.update(opd_metrics)
        if opd_metrics and dist.get_rank() == 0:
            logger.info("OPD step=%d: %s", step, self._format_opd_metrics(opd_metrics))

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
