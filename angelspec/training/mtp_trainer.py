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

from angelspec import AutoDraftModel, AutoDraftModelConfig
from angelspec.models.draft.mtp import Hy3MoE
from angelspec.models.mtp import MTPModel
from angelspec.training import checkpoint
from angelspec.training.fsdp import (
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fused_expert_shard_units,
    mtp_linear_shard_units,
)
from angelspec.training.optimizer import BF16Optimizer, moe_lr_scales
from angelspec.training.trainer import Trainer
from angelspec.utils.distributed import get_gloo_group
from angelspec.utils.logging import logger
from angelspec.utils.metrics import means_from_token_totals, token_metric_totals
from angelspec.utils.source_metrics import (
    batch_source_ids,
    resolve_metric_sources,
    source_metric_log,
)
from angelspec.utils.tensor import padding, same_doc_next_mask
from angelspec.utils.usp import usp_chunk_size, usp_dp_average_factor


def _position_decay_weights(n: int, weights: Optional[list] = None, theta_decay=0.8) -> list:
    """Per-position TTT loss weights (defaults to 0.8**i)."""
    if weights is not None:
        if len(weights) != n:
            raise ValueError(f"ploss_weights length {len(weights)} does not match ttt_length {n}")
        return [float(w) for w in weights]
    return [theta_decay**i for i in range(n)]


def _reject_gradients(params: dict) -> None:
    """Fail on any gradient reaching parameters kept out of FSDP: those gradients
    would never be reduced, so ranks would silently drift apart."""
    for name, param in params.items():
        if param.requires_grad:

            def _raise(_grad, name=name):
                raise RuntimeError(f"{name} is kept out of FSDP and must not receive gradients")

            param.register_hook(_raise)


def _remap_legacy_fused_expert_key(key: str) -> str:
    """Map a legacy flat fused-expert FQN to the new ``FusedExpertProj`` naming.

    Older checkpoints stored the fused MoE expert weights as bare ``nn.Parameter``
    entries (e.g. ``...experts_gate_proj``); ``Hy3MoE`` now wraps them in a
    ``FusedExpertProj`` submodule, so the matching parameter is
    ``...experts_gate_proj.weight``. Reuses the authoritative name list from
    ``Hy3MoE._LEGACY_FUSED_NAMES`` to stay in sync with the ``load_state_dict``
    pre-hook that serves the non-DCP warm-start path.
    """
    name = key.rsplit(".", 1)[-1]
    if name in Hy3MoE._LEGACY_FUSED_NAMES:
        return key + ".weight"
    return key


def _dequant_fp8_block_weights(state: dict, block: tuple = (128, 128)) -> dict:
    """Fold block-FP8 tensors and their ``<key>_scale_inv`` scales (one per ``block`` tile
    of the last two dims, fused experts stacked in front) into bf16."""
    out = {}
    for key, value in state.items():
        if key.endswith("_scale_inv"):
            continue
        if value.dtype != torch.float8_e4m3fn:
            out[key] = value
            continue
        scale = state.get(key + "_scale_inv")
        if scale is None:
            raise ValueError(f"{key} is FP8 but the checkpoint has no {key}_scale_inv")
        rows, cols = value.shape[-2:]
        scale = scale.float().repeat_interleave(block[0], -2)[..., :rows, :]
        scale = scale.repeat_interleave(block[1], -1)[..., :cols]
        out[key] = (value.float() * scale).to(torch.bfloat16)
    return out


def align_mtp_inputs(
    raw_input_ids: torch.Tensor,
    raw_last_hidden: torch.Tensor,
    raw_loss_mask: torch.Tensor,
    *,
    draft_input_postnorm: bool = False,
    verifier_norm=None,
    ctx_doc_ids: torch.Tensor = None,
    chunk_size: int = 0,
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
    if chunk_size > 0 and verifier_norm is not None:
        from angelspec.models.ops.memory_chunks import normalize_shifted_hidden

        target_hidden_states = normalize_shifted_hidden(
            raw_last_hidden, verifier_norm, chunk_size, shift=True
        )
        draft_input = (
            normalize_shifted_hidden(raw_last_hidden, verifier_norm, chunk_size, shift=False)
            if draft_input_postnorm
            else raw_last_hidden
        )
    else:
        last_hidden_shifted = padding(raw_last_hidden, left=False)
        if verifier_norm is not None:
            with torch.no_grad():
                target_hidden_states = verifier_norm(last_hidden_shifted)
        else:
            target_hidden_states = last_hidden_shifted
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
        self._metric_sources = resolve_metric_sources(args)
        if self._metric_sources:
            logger.info("MTP token-level source metrics: %s", self._metric_sources)

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

        from angelspec.training.mtp_rope import configure_mtp_rope
        from angelspec.training.qwen35_mtp import (
            configure_qwen35_target,
            initialize_qwen35_mtp,
            is_qwen35_mtp,
        )

        configure_mtp_rope(draft_model_config, target_model_path)
        qwen35 = is_qwen35_mtp(draft_model_config)
        draft_opd = getattr(self.args, "mtp_draft_opd", False)
        if qwen35:
            configure_qwen35_target(self.args, draft_model_config, target_model_path)

        init_context = self._get_init_weight_context_manager()

        with init_context():
            draft_model = AutoDraftModel.from_config(
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
        # Two source layouts are supported:
        #   mtp_init_from_target_layer -> model.layers.<N>.*   (Hy3: MTP block stored as a
        #                                 decoder layer after the backbone; GQA attention)
        #   mtp_init_from_mtp_layer    -> model.mtp_layers.<N>.* (Hy4: dedicated DeepSeek-V3
        #                                 style MTP head; MLA + DSA indexer + fused experts)
        init_mtp_layer = getattr(draft_model_config, "mtp_init_from_mtp_layer", None)
        if init_mtp_layer is None:
            init_mtp_layer = getattr(self.args, "mtp_init_from_mtp_layer", None)
        init_layer = getattr(draft_model_config, "mtp_init_from_target_layer", None)
        if init_layer is None:
            init_layer = getattr(self.args, "mtp_init_from_target_layer", None)
        if draft_opd:
            if not self.args.load_path:
                from angelspec.training.mtp_opd_trainer import initialize_opd_weights

                initialize_opd_weights(draft_model, self.args.mtp_opd_init_model)
        elif qwen35:
            if dist.get_rank() == 0:
                source = initialize_qwen35_mtp(
                    draft_model, self.args, draft_model_config, target_model_path
                )
                logger.info("Qwen3.5 MTP initialization: %s", source)
        elif getattr(self.args, "offline_manifest_path", None):
            from angelspec.training.offline_mtp import initialize_offline_mtp

            initialize_offline_mtp(self, draft_model, draft_model_config, target_model_path)
        elif getattr(self.args, "mtp_draft_opd", False):
            # OPD loads a complete draft state below (or resumes its own DCP).
            pass
        elif init_mtp_layer is not None:
            self._load_mtp_from_target(
                draft_model, target_model_path, int(init_mtp_layer), source="mtp_layers"
            )
        elif init_layer is not None:
            self._load_mtp_from_target(
                draft_model, target_model_path, int(init_layer), source="layers"
            )

        if getattr(self.args, "mtp_draft_opd", False) and not getattr(
            self.args, "load_path", None
        ):
            from angelspec.training.mtp_opd_trainer import initialize_opd_weights

            initialize_opd_weights(draft_model, self.args.mtp_opd_init_model)

        dist.barrier(group=get_gloo_group())

        frozen_count = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        logger.info(
            f"[Rank {self.dp_rank}] MTP draft model: {trainable_count:,} trainable, {frozen_count:,} frozen parameters"
        )

        if draft_opd:
            from angelspec.models.mtp_opd import MTPOPDModel

            mtp_model = MTPOPDModel(
                draft_model,
                attention_backend=self.args.attention_backend,
                gradient_checkpointing=self.args.gradient_checkpointing,
                prefix_chunk_size=self.args.mtp_opd_prefix_chunk_size,
                fp8_indexer=self.args.mtp_opd_fp8_indexer,
            )
        else:
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
                argmax_chunk_size=getattr(self.args, "mtp_argmax_chunk_size", 0),
                ce_requires_grad=float(
                    getattr(draft_model_config, "mtp_ce_loss_scaling_factor", 1.0)
                )
                != 0.0,
                loss_checkpoint_chunks=getattr(self.args, "mtp_loss_checkpoint_chunks", False),
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

        # Keep fused expert projections as separate FSDP units so their expert
        # dimension shards cleanly.
        frozen_fp32 = draft_model.frozen_fp32_parameters()
        mtp_model = apply_fsdp2(
            mtp_model,
            mesh=self.fsdp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            # OPD reads projection weights directly; keep them in the root unit
            # whose forward unshards them for both sampling and gradient replay.
            modules_to_shard=[
                *fused_expert_shard_units(mtp_model),
                *([] if draft_opd else mtp_linear_shard_units(mtp_model)),
            ],
            ignored_params=set(frozen_fp32.values()),
        )

        mtp_model = fsdp2_load_full_state_dict(
            mtp_model,
            full_state,
            self.fsdp_mesh,
            cpu_offload=True if self.fsdp_cpu_offload else None,
            ignored_params=list(frozen_fp32.values()),
        )
        _reject_gradients(frozen_fp32)
        if getattr(self.args, "attention_backend", None) == "usp" and not draft_opd:
            from torch.distributed.fsdp import register_fsdp_forward_method

            register_fsdp_forward_method(mtp_model, "warmup_usp_flex")

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
        moe_scale = float(getattr(self.args, "moe_lr_scale", 1.0))
        router_scale = getattr(self.args, "moe_router_lr_scale", None)
        router_scale = moe_scale if router_scale is None else float(router_scale)
        lr_scales = {}
        if (moe_scale, router_scale) != (1.0, 1.0):
            lr_scales = moe_lr_scales(self.draft_model, moe_scale, router_scale)
            if not lr_scales:
                raise ValueError("moe_lr_scale is set but the draft has no MoE FFN")
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
            betas=(getattr(self.args, "adam_beta1", 0.9), getattr(self.args, "adam_beta2", 0.999)),
            optimizer_type=getattr(self.args, "optimizer_type", "adamw"),
            muon_momentum=getattr(self.args, "muon_momentum", 0.95),
            muon_nesterov=getattr(self.args, "muon_nesterov", True),
            muon_ns_steps=getattr(self.args, "muon_ns_steps", 5),
            muon_matched_adamw_rms=getattr(self.args, "muon_matched_adamw_rms", 0.2),
            muon_weight_decay=getattr(self.args, "muon_weight_decay", None),
            lr_scales=lr_scales,
            muon_per_expert=getattr(self.args, "muon_per_expert", True),
        )
        self.lr_scheduler = self.optimizer.lr_scheduler

        # Legacy fused-expert DCP keys only need remapping when the draft model
        # actually uses Hy3MoE layers; otherwise keep the default (no-remap) path.
        key_remap_fn = (
            _remap_legacy_fused_expert_key
            if any(isinstance(m, Hy3MoE) for m in self.draft_model.modules())
            else None
        )
        checkpoint_payload = checkpoint.load(self, key_remap_fn=key_remap_fn)
        if qwen35 and getattr(self.args, "load_path", None) and checkpoint_payload is None:
            raise RuntimeError("Requested Qwen3.5 training checkpoint could not be restored")
        checkpoint.finalize_load(self, checkpoint_payload)

        self._last_hs_prenorm = getattr(self.args, "last_hidden_states_prenorm", False)

        # Load the (frozen) target lm_head and tie it into the draft.
        self._init_target_lm_head(target_model_path)
        if self.target_lm_head is None:
            raise ValueError("target_lm_head is required for MTP but was None.")
        self.target_lm_head_weight = self.target_lm_head.lm_head.weight
        self.verifier_norm = self.target_lm_head.norm
        if getattr(self.args, "offline_manifest_path", None) and self.verifier_norm is None:
            raise ValueError("Offline MTP requires the real target final norm")
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
        if getattr(self.args, "attention_backend", None) == "usp" and not draft_opd:
            sp_size = getattr(self.args, "sp_ulysses_size", 1) * getattr(
                self.args, "sp_ring_size", 1
            )
            max_seq = getattr(self.args, "max_seq_length", None)
            if max_seq and sp_size > 1:
                shard_len = usp_chunk_size(max_seq, sp_size)
                logger.info(
                    f"[Rank {self.dp_rank}] USP flex warmup: shard_len={shard_len} "
                    f"(full_seq={max_seq}, sp_size={sp_size}) — SKIPPED (compiler.disable active)"
                )

        logger.info(f"[Rank {self.dp_rank}] MTP model initialized with FSDP2")
        return 0

    # CPT: load trained MTP layer from target checkpoint

    def _load_mtp_from_target(
        self, draft_model, target_model_path: str, layer_idx: int, source: str = "layers"
    ) -> None:
        """Warm-start the MTP draft from the target checkpoint's trained MTP head.

        ``source="layers"`` (Hy3): the MTP block is stored as a decoder layer right
        after the backbone under ``model.layers.<N>.*`` with GQA attention and
        per-expert ``mlp.experts.<e>.{gate,up,down}_proj`` tensors.

        ``source="mtp_layers"`` (Hy4): a dedicated DeepSeek-V3 style head under
        ``model.mtp_layers.<N>.*`` with MLA + DSA-indexer attention and FUSED
        ``mlp.experts.{gate_up_proj,down_proj}`` tensors, plus DeepSeek MoE naming
        (``mlp.gate.weight`` / ``mlp.gate.e_score_correction_bias`` /
        ``mlp.shared_experts.*``) that is remapped onto the draft's Hy3MoE names.

        Common mapping into MTPDraftModel:
            <prefix>.enorm/hnorm/eh_proj/final_layernorm      -> <same>
            <prefix>.input_layernorm/post_attention_layernorm -> midlayer.<same>
            <prefix>.self_attn.*                              -> midlayer.self_attn.*
        embed_tokens / lm_head are not in this layer (loaded/tied separately).
        """
        if dist.get_rank() != 0:
            return

        if source == "mtp_layers":
            prefixes = [f"model.mtp_layers.{layer_idx}."]
        else:
            # VL checkpoints nest the text tower under model.language_model.
            prefixes = [f"model.layers.{layer_idx}.", f"model.language_model.layers.{layer_idx}."]
        for prefix in prefixes:
            state = self._read_target_layer_weights(target_model_path, prefix)
            if state:
                break
        if not state:
            raise FileNotFoundError(
                f"MTP CPT init: no weights with prefixes {prefixes} found in {target_model_path}"
            )
        state = _dequant_fp8_block_weights(state, self._fp8_weight_block(target_model_path))

        fused_experts = any(key.endswith("mlp.experts.gate_up_proj") for key in state)
        if source == "mtp_layers" or fused_experts:
            # A decoder layer's hyper-connection mixers belong to the backbone's streams.
            state = {
                key: value
                for key, value in state.items()
                if not key[len(prefix) :].startswith(("hc_attn_layer.", "hc_mlp_layer."))
            }
            remapped = self._remap_mtp_layer_state(state, prefix)
        else:
            remapped = self._remap_decoder_layer_state(state, prefix)

        missing, unexpected = draft_model.load_state_dict(remapped, strict=False)
        # embed_tokens.weight + lm_head.weight are expected to be missing here.
        real_missing = [
            m for m in missing if not (m.startswith("embed_tokens") or m.startswith("lm_head"))
        ]
        if getattr(draft_model.config, "mtp_init_strict", False) and (real_missing or unexpected):
            raise RuntimeError(
                f"MTP CPT init from {prefix} does not cover the draft exactly: "
                f"missing={real_missing}, unexpected={unexpected}"
            )
        if real_missing:
            logger.warning(f"MTP CPT init: missing keys (will keep init values): {real_missing}")
        if unexpected:
            logger.warning(f"MTP CPT init: unexpected keys (ignored): {unexpected}")
        logger.info(
            f"[Rank 0] MTP CPT init from {prefix}: loaded {len(remapped)} tensors "
            f"({len(real_missing)} real-missing, {len(unexpected)} unexpected)"
        )

    @staticmethod
    def _fp8_weight_block(model_path: str) -> tuple:
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return (128, 128)
        with open(config_path) as f:
            config = json.load(f)
        quant = (
            config.get("quantization_config")
            or (config.get("text_config") or {}).get("quantization_config")
            or {}
        )
        return tuple(quant.get("weight_block_size") or (128, 128))

    @staticmethod
    def _remap_decoder_layer_state(state: dict, prefix: str) -> dict:
        """Per-expert MoE layer layout -> draft names, experts fused to [E, in, out].

        Hy3 already uses the draft's names (GQA attention, ``mlp.router.gate``,
        ``mlp.expert_bias``, ``mlp.shared_mlp``); DeepSeek-style checkpoints use
        ``mlp.gate``, ``mlp.gate.e_score_correction_bias``, ``mlp.shared_experts``
        and ``shared_head.norm`` for the final norm.
        """
        import re

        top_level = {"enorm", "hnorm", "eh_proj", "final_layernorm"}
        renames = {
            "mlp.gate.weight": "midlayer.mlp.router.gate.weight",
            "mlp.gate.e_score_correction_bias": "midlayer.mlp.expert_bias",
            "shared_head.norm.weight": "final_layernorm.weight",
        }
        expert_pat = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
        expert_slices = {"gate_proj": {}, "up_proj": {}, "down_proj": {}}

        remapped = {}
        for k, v in state.items():
            sub = k[len(prefix) :]
            m = expert_pat.match(sub)
            if m is not None:
                e_idx, proj = int(m.group(1)), m.group(2)
                expert_slices[proj][e_idx] = v
                continue
            if sub in renames:
                remapped[renames[sub]] = v
            elif sub.startswith("mlp.shared_experts."):
                remapped[f"midlayer.mlp.shared_mlp.{sub[len('mlp.shared_experts.'):]}"] = v
            elif sub.split(".")[0] in top_level:
                remapped[sub] = v
            else:
                remapped[f"midlayer.{sub}"] = v

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
        return remapped

    @staticmethod
    def _remap_mtp_layer_state(state: dict, prefix: str) -> dict:
        """Hy4 DeepSeek-V3 MTP head: MLA + DSA indexer + fused/DeepSeek-named MoE.

        DeepSeek MoE names -> draft Hy3MoE names:
            mlp.gate.weight                    -> midlayer.mlp.router.gate.weight
            mlp.gate.e_score_correction_bias   -> midlayer.mlp.expert_bias
            mlp.shared_experts.{gate,up,down}_proj -> midlayer.mlp.shared_mlp.{...}
            mlp.experts.gate_up_proj [E,2I,H]  -> split -> experts_{gate,up}_proj [E,H,I]
            mlp.experts.down_proj    [E,H,I]   -> transpose -> experts_down_proj [E,I,H]
        self_attn.* (MLA q/kv LoRA + linear_gate + learnable_sink_param + indexer.*)
        pass straight through to midlayer.self_attn.* (names already aligned).
        """
        top_level = {"enorm", "hnorm", "eh_proj", "final_layernorm"}
        remapped = {}
        for k, v in state.items():
            sub = k[len(prefix) :]
            if sub == "mlp.experts.gate_up_proj":
                # [E, 2I, H] with gate rows first, then up rows (both [out, in]).
                two_i = v.shape[1]
                inter = two_i // 2
                gate = v[:, :inter, :]  # [E, I, H] = [E, out, in]
                up = v[:, inter:, :]
                remapped["midlayer.mlp.experts_gate_proj"] = gate.transpose(1, 2).contiguous()
                remapped["midlayer.mlp.experts_up_proj"] = up.transpose(1, 2).contiguous()
            elif sub == "mlp.experts.down_proj":
                # [E, H, I] = [E, out, in] -> fused [E, in, out] = [E, I, H]
                remapped["midlayer.mlp.experts_down_proj"] = v.transpose(1, 2).contiguous()
            elif sub == "mlp.gate.weight":
                remapped["midlayer.mlp.router.gate.weight"] = v
            elif sub == "mlp.gate.e_score_correction_bias":
                remapped["midlayer.mlp.expert_bias"] = v
            elif sub.startswith("mlp.shared_experts."):
                tail = sub[len("mlp.shared_experts.") :]
                remapped[f"midlayer.mlp.shared_mlp.{tail}"] = v
            elif sub.split(".")[0] in top_level:
                remapped[sub] = v
            else:
                # input_layernorm / post_attention_layernorm / self_attn.* (incl. indexer.*)
                remapped[f"midlayer.{sub}"] = v
        return remapped

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
        from angelspec.utils.hf_config_registry import register_extra_hf_configs

        register_extra_hf_configs()

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

    def _forward(self, batch: dict, *, collect_source_metrics: bool = False):
        if getattr(self.args, "offline_manifest_path", None):
            batch = {
                key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
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
            chunk_size=getattr(self.args, "mtp_alignment_chunk_size", 0),
            ctx_doc_ids=ctx_doc_ids,
        )
        input_ids = input_ids.cuda()
        draft_input = draft_input.cuda()
        target_hidden_states = target_hidden_states.cuda()
        loss_mask = loss_mask.cuda()

        metric_kwargs = {}
        if collect_source_metrics:
            metric_kwargs = {
                "metric_source_ids": batch_source_ids(
                    batch, self._metric_sources, input_ids.device, doc_ids=ctx_doc_ids
                ),
                "metric_num_sources": len(self._metric_sources),
            }
        return self.model(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].cuda(),
            target_hidden_states=target_hidden_states,
            target_lm_head_weight=self.target_lm_head_weight,
            loss_mask=loss_mask,
            hidden_states=draft_input,
            position_ids=(
                batch.get("position_ids").cuda() if batch.get("position_ids") is not None else None
            ),
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_position_ids,
            **metric_kwargs,
        )

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
        source_totals = None
        if getattr(self, "_metric_sources", ()):
            output = self._forward(batch, collect_source_metrics=True)
            source_totals = output[-1]
            output = output[:-1]
        else:
            output = self._forward(batch)
        ce_losses, kl_losses, vlosses, acces, acc_counts, acces_gt, e2e_loss, vklosses = output
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
        if source_totals is not None:
            metrics["source_totals"] = source_totals.detach()
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
        source_names = getattr(self, "_metric_sources", ())
        if source_names:
            source_totals = torch.stack([m["source_totals"] for m in all_step_metrics]).sum(0)
            source_slots = totals.new_zeros(
                (len(source_names), *totals.shape), dtype=torch.float64
            )
            source_slots[:, :, 2:] = source_totals
            totals = torch.cat((totals.double().unsqueeze(0), source_slots), dim=0)
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
        split_metrics = {}
        if source_names:
            source_totals = totals[1:, :, 2:]
            totals = totals[0].float()
            split_metrics = source_metric_log(source_names, source_totals, totals)
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

        metrics.update(split_metrics)
        if dist.get_rank() == 0:
            logger.debug(f"step {step}: {metrics}")
        return metrics
