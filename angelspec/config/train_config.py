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

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from angelspec.config.inference_config import InferenceConfig
from angelspec.data.utils import is_local_data_path
from angelspec.utils.logging import logger


@dataclass
class DatasetConfig:
    chat_template: str = "llama3"
    defer_tokenization: bool = False
    drop_overlength: bool = (
        False  # drop (not truncate) samples whose token count exceeds max_seq_length-1
    )
    eval_data_path: Optional[str] = None
    eval_interval: int = 50
    eval_micro_batch_size: Optional[int] = None
    eval_prompt_key: Optional[str] = None
    last_turn_loss_only: Any = "auto"  # bool or "auto"
    min_loss_tokens: int = (
        0  # DFlash: skip sequences with < N supervised tokens (use 2*block_size)
    )
    prompt_key: str = "conversations"
    shuffle_dataset: bool = True
    train_data_path: str = ""


@dataclass
class DebugConfig:
    debug_inference_only: bool = False
    debug_train_only: bool = False
    enable_perf_metrics: bool = True
    max_dump_steps: int = 5
    memory_recorder: str = "torch"
    memory_snapshot_dir: str = "."
    memory_snapshot_num_steps: Optional[int] = None
    memory_snapshot_path: str = ""
    profile_dir_name: Optional[str] = "/tmp/angelspec_profiles"
    profile_step_end: int = 0
    profile_step_start: int = 0
    profile_target: list = field(default_factory=lambda: ["train_overall"])
    record_memory_history: bool = False
    save_debug_train_data: Optional[str] = None
    use_pytorch_profiler: bool = False


@dataclass
class LoggingConfig:
    report_to: str = "none"
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_dir: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_host: Optional[str] = None
    wandb_key: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_random_suffix: bool = True
    wandb_run_id: Optional[str] = None
    wandb_team: Optional[str] = None


@dataclass
class ModelConfig:
    draft_model_config: Optional[str] = None
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    norm_key: str = "model.norm.weight"
    target_model_backend: str = "sglang"
    target_model_path: str = ""
    trust_remote_code: bool = False


@dataclass
class TrainingConfig:
    attention_backend: str = "sdpa"
    colocate: bool = False
    continual_training: bool = False
    distributed_backend: str = "nccl"
    distributed_timeout_minutes: int = 10
    draft_accumulation_steps: int = 1
    fsdp_reduce_dtype: str = "float32"  # "float32" or "bfloat16"
    fsdp_strategy: str = "REPLICATE"
    # Controls which workload claims head-node GPUs first under PACK strategy.
    # "training_first" (default), "inference_first", or "custom".
    placement_strategy: str = "training_first"
    training_node_ips: Optional[list[str]] = None
    training_node_selectors: Optional[list[dict[str, str]]] = None
    compile_model: bool = False  # torch.compile the full training model
    sp_ring_size: int = 1
    sp_ulysses_size: int = 1
    # USP data delivery mode. False (Eagle3/SGLang default): the inference engine
    # pre-shards each sample's seq into per-rank Mooncake keys ({key}_usp{rank}) and
    # every SP rank loads only its slice. True (MTP pure-Ulysses): the engine writes
    # ONE unsharded sample; the controller fans the SAME full sample to every SP rank
    # and the model slices its own seq shard in-forward (MTPModel).
    usp_local_shard: bool = False

    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    load_path: Optional[str] = None
    lr_decay_style: str = "cosine"
    lr_wsd_decay_ratio: float = 0.2
    lr_wsd_decay_style: str = "cosine"
    lr_total_steps: Optional[int] = None
    max_concurrent_batches: int = 1
    max_grad_norm: float = 0.5
    max_seq_length: int = 8192
    min_lr: float = 0.0
    weight_decay: float = 0.0
    # AdamW betas. beta2 default 0.999; some DFlash recipes use 0.95, which tends
    # to be steadier for small drafts on short seqs.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    # Optimizer selection. "adamw" (default) keeps the existing fp32 master-weight
    # AdamW; "muon" updates >=2-D weight matrices via Momentum-Orthogonalized
    # Newton-Schulz (on the fp32 master copy) while norms/embeddings/LM head and
    # DFlare fusion logits stay on AdamW. Shared by DFlash/DFlare only unless the
    # other trainers thread it through. muon_matched_adamw_rms scales the per-matrix
    # LR to match AdamW's update RMS (MoonshotAI/Moonlight convention).
    optimizer_type: str = "adamw"
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_matched_adamw_rms: float = 0.2
    num_epochs: int = 10
    num_train_steps: Optional[int] = None
    micro_batch_size: int = 2
    prefetch_depth: int = 2  # 0 = disabled, >0 = async pre-fetch N batches ahead
    save_interval: int = 5000
    save_per_epoch: bool = False
    max_checkpoints: int = 0  # 0 = keep all, N > 0 = rotate and keep only N most recent
    seed: int = 0
    train_backend: str = "fsdp"
    train_env_vars: str = "{}"
    train_with_decode: bool = False
    training_num_gpus_per_node: int = 1
    training_num_nodes: int = 1
    ttt_length: int = 7
    # Per-position TTT loss weights. If unset, defaults to [0.8**i for i in range(ttt_length)].
    # Length must equal ttt_length when supplied.
    ploss_weights: Optional[list[float]] = None
    # Single-head MTP: feed the post-final-norm hidden (same tensor as the
    # teacher lm_head input) to the MTP block's hnorm, matching the vLLM serve
    # path. False = legacy pre-norm residual.
    mtp_draft_input_postnorm: bool = False
    # Single-head MTP on-policy TTT. When True, step>=1 embeds the draft's OWN
    # previous-step argmax (stop-grad) instead of the left-shifted ground-truth
    # token; step 0 stays teacher-forced and the CE/KL target is always
    # ground-truth. Trains the head on the same recurrent input distribution serve
    # uses → removes the train-vs-serve pos1+ gap. False (default) = teacher forcing.
    mtp_ttt_on_policy: bool = False
    # Half on-policy distillation for MTP via packed tree-forward scoring. When
    # True, the draft rolls out an on-policy block at each supervised anchor, a
    # single packed tree-forward has the target score every generated token, and
    # an OPD/KL term is added on the branch positions (block-level off-policy).
    # GPU-gated (needs the FlexAttention scoring engine). False (default) = off.
    mtp_opd_tree_forward: bool = False
    # Branch block length B (tokens generated per anchor). Must be <= ttt_length,
    # since the block is filled from the TTT per-step argmax.
    mtp_opd_block_size: int = 8
    # Cap on anchors (branches) per sequence; None = every supervised position.
    # Bounds teacher scoring cost (packed row length ~= trunk + K*B).
    mtp_opd_max_anchors: Optional[int] = None
    warmup_ratio: float = 0.015

    # WSD LR schedule parameters (used by DFlash trainer only)
    wsd_decay_ratio: float = 0.2
    wsd_decay_style: Optional[str] = None

    # DFlash-specific parameters (ignored for Eagle3 training)
    dflash_block_size: int = 16
    dflash_loss_decay_gamma: float = 7.0
    dflash_num_anchors: int = 512
    dflash_num_target_layers: int = 5
    # Compute draft logits (lm_head output) in fp32 before CE / argmax. Default
    # True: FSDP2 runs bf16 compute, and bf16 logit rounding flips small-margin
    # argmax decisions that both the CE gradient and the acc metric depend on. Set
    # False to reproduce old bf16-logit numerical trajectories.
    dflash_fp32_lm_head: bool = True
    # Unified DFlash loss architecture. Objective sets the per-position CE
    # weighting; the CE-side distillation terms (L1 / top-K KL / LK) stack on top
    # and are independent of (and combine additively with) the OPD term below.
    #   dflash_loss_objective: "decay" (exp-decay, default) or "dpace" (D-PACE
    #     continuation-value weighting; ignores dflash_loss_decay_gamma for CE).
    #   dflash_dpace_alpha: D-PACE confidence smoothing in [0, 1].
    #   dflash_ce_loss_alpha: scalar multiplier on the CE term. DEFAULT 1.0 —
    #     this is applied UNCONDITIONALLY, so a non-1.0 default (e.g. DSpark's
    #     0.1) would silently rescale every existing run. Set it explicitly in
    #     configs that want CE down-weighted (e.g. alongside L1).
    #   dflash_l1_loss_alpha: L1 (== 2*TV) distribution distillation weight;
    #     needs target last_hidden_states. 0 disables.
    #   dflash_kl_* / dflash_lk_*: convex-mix distillation vs the target's true
    #     last-layer logits, in [0, 1] against CE. LK precedes KL when both set;
    #     both need target last_hidden_states (+ final norm). 0 disables.
    dflash_loss_objective: str = "decay"
    dflash_dpace_alpha: float = 0.5
    dflash_ce_loss_alpha: float = 1.0
    dflash_l1_loss_alpha: float = 0.0
    dflash_kl_loss_weight: float = 0.0
    dflash_kl_temperature: float = 1.0
    dflash_kl_topk: int = 10
    dflash_kl_topk_renormalize: bool = True
    dflash_lk_loss_weight: float = 0.0
    dflash_lk_loss_type: str = "hybrid"  # "alpha" or "hybrid"
    dflash_lk_eta: float = 3.0
    dflash_lk_temperature: float = 1.0
    # Gated-sum layer-selection run only (fusion_type=gated_sum in the draft config).
    # Optional sparsity penalty weight on the layer gate: adds
    # `weight * H(softmax(gate))` to the loss to push the gate toward a peakier
    # ranking. Default 0 => no penalty (plan A: fixed-temperature softmax reads the
    # ranking on its own). Raise to O(1e-3) only if the gate stays too flat to
    # separate a top-k (plan B).
    dflash_gate_entropy_weight: float = 0.0

    # Half on-policy distillation (OPD, DFlash). When True the trainer, each step,
    # takes the draft's block proposals, packs a [trunk | branches] tree, has a
    # separate encoder-only score engine (see enable_opd_score_engine) score every
    # proposed token, and adds a KL(student‖teacher) term over the branches. The
    # base DFlash CE loss is unchanged. Requires a score engine handle.
    dflash_opd_enabled: bool = False
    # Weight of the OPD-KL term added to the base DFlash loss.
    dflash_opd_weight: float = 1.0
    # Cap on how many anchors OPD SCORES per sequence (evenly subsampled). The
    # base DFlash CE still uses all dflash_num_anchors; this only bounds the
    # packed tree-forward cost (scoring scales with anchors*block). None = all.
    dflash_opd_max_anchors: Optional[int] = None
    # Two-stream OPD loss: the scored proposal slots are
    # split per branch by greedy verify (draft-proposed token vs teacher argmax)
    # into an accepted prefix (response stream) and a rejected suffix (rejected-
    # draft stream), each scored with the k3 reverse-KL estimator on the proposed
    # token. The rejected stream is position-decayed by decay^(offset-1).
    dflash_opd_response_stream_weight: float = 1.0
    dflash_opd_rejected_stream_weight: float = 1.0
    dflash_opd_position_decay: float = 0.8
    dflash_opd_position_decay_enabled: bool = True
    dflash_opd_loss_max_clamp: Optional[float] = 10.0
    dflash_opd_logprob_min_clamp: Optional[float] = -10.0
    # Spawn the encoder-only score engine (VllmEngine with vllm_score_engine=True)
    # on its own role/GPU(s). Independent of dflash_opd_enabled so the engine can
    # be brought up / probed without touching the loss.
    enable_opd_score_engine: bool = False
    opd_score_engine_num_gpus: int = 1

    # Sequence packing (DFlash only). When True, the training-side collator packs
    # multiple samples into one fixed-length sequence (padded to max_seq_length)
    # with doc-aware block masks + doc-local RoPE, instead of pad-to-longest
    # micro-batches. NOTE: this switches the training recipe (fixed per-sequence
    # anchor budget → lower per-sample anchor coverage, length-weighted sample
    # importance, B=1 per step), so runs are NOT directly comparable to non-packed
    # runs and LR/step schedule must be recalibrated.
    dflash_packing: bool = False

    # Packing loss normalization (only meaningful with dflash_packing=True).
    # False (default): each packed row is weighted equally (loss / num_micro), and
    #   rows are consumed with a memory-lean look-ahead iterator (peak 1-2 rows).
    # True: each row is weighted by its supervised-token count (row_tokens /
    #   step_total_tokens). This needs the step's global token total, so ALL rows
    #   for the step are materialized first (peak = num_micro rows resident).
    dflash_packing_token_weighted_loss: bool = False

    # Sequence packing for single-head MTP. Same controller-driven row packing as
    # dflash_packing (pack_into_rows is length-based / model-agnostic), but the
    # training-side collator is MTPPackingCollator (packs last_hidden_states + emits
    # ctx_doc_ids / doc-local base_position_ids). Composes with USP local-shard
    # (usp_local_shard=true): each SP rank assembles the identical packed row and
    # slices its own seq shard in-forward. NOT compatible with USP sharded-data
    # pre-shard. Recipe caveat matches dflash_packing (not comparable to non-packed).
    mtp_packing: bool = False

    # Optional controller-side packing efficiency guard shared by DFlash and MTP.
    # 0 disables waiting (legacy behavior). When a candidate dispatch is below the
    # threshold, the controller waits for more inference supply, but flushes the
    # partial rows after packing_max_wait_seconds to avoid deadlock at train end.
    packing_min_fill_ratio: float = 0.0
    packing_max_wait_seconds: float = 5.0

    # DSpark-specific parameters (DFlash backbone + Markov/L1/confidence heads).
    # block_size is read from dflash_block_size (default 7 in DSpark configs).
    dspark_num_anchors: int = 512
    dspark_num_target_layers: int = 5
    dspark_loss_decay_gamma: float = 4.0
    dspark_ce_loss_alpha: float = 0.1
    dspark_l1_loss_alpha: float = 0.9
    dspark_confidence_head_alpha: float = 1.0


@dataclass
class DecodeConfig:
    """Config for train-with-decode mode (speculative decoding during training)."""

    cuda_graph_max_bs: Optional[int] = None
    max_new_tokens: int = 512
    min_new_tokens: int = 2
    stop_token_ids: Optional[list[int]] = None
    max_running_requests: Optional[int] = None
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_num_steps: Optional[int] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    weight_sync_enabled: bool = False
    weight_sync_interval: int = 500


@dataclass
class OnlineEvalConfig:
    """Config for online real spec-decode eval during training.

    Independent of ``train_with_decode``: periodically exports the current draft,
    (re)loads it into a spec-decode engine, drives a fixed eval prompt set, and
    logs the engine-native acceptance length/rate alongside the simulated eval
    metrics. See ``controller/online_eval.py``.
    """

    enabled: bool = False
    interval: int = 500
    dataset: Optional[str] = None
    prompt_key: str = "conversations"
    limit: int = 64
    num_spec_tokens: int = 3
    max_new_tokens: int = 256
    engine_mode: str = "shared"  # "shared" (reuse inference-segment GPUs) or "dedicated"
    engine_gpus: Optional[int] = None  # dedicated-mode: GPUs to carve out for eval
    gpus: Optional[str] = None  # explicit CUDA_VISIBLE_DEVICES for the eval serve
    port: int = 8130
    gpu_util: float = 0.85
    # MTP-specific: the served scaffold (dir with a dedicated spec-layer shard) and
    # its spec layer index (0 => infer from scaffold config num_hidden_layers).
    mtp_scaffold: Optional[str] = None
    mtp_spec_layer: int = 0
    # --- Persistent dedicated eval engine (stage 2) ---
    # num_gpus > 0 carves out an "eval" role placement group (see placement_group.py)
    # so the eval engine runs as a long-lived Ray actor, optionally pinned to its
    # own node(s). When 0, online eval falls back to the stage-1 in-process path.
    num_gpus: int = 0
    num_gpus_per_engine: int = 1
    num_gpus_per_node: int = 8
    node_ips: Optional[list[str]] = None
    node_selectors: Optional[list[dict]] = None
    speculative_method: str = "mtp"
    reasoning_parser: Optional[str] = None
    # Acceptance sampling: "argmax" (greedy token-equality) or "rs" (rejection
    # sampling; engaged by vLLM when temperature>0). rs accept-len is a sampled
    # trajectory, not comparable to argmax; seed makes a run reproducible.
    sampling_mode: str = "argmax"
    temperature: float = 1.0  # only used when sampling_mode == "rs"
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None
    # --- Multi-dataset eval (vendored under angelspec/data/eval_prompts) ---
    # When ``datasets`` is set, each named set is loaded from the vendored jsonl,
    # driven through the stage-2 eval engine, and reported per-dataset (+ mean).
    # When None, falls back to the single ``dataset`` jsonl path above.
    datasets: Optional[list[str]] = None
    per_dataset_limit: Optional[int] = None  # samples per dataset (None => reuse limit)
    dataset_sample_size: int = 1000  # sampling cap per dataset (seeded shuffle)
    dataset_seed: int = 42
    eval_prompts_dir: Optional[str] = None  # override built-in eval_prompts dir
    reasoning_effort: str = "no_think"  # chat_template_kwargs for chat generation


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    online_eval: OnlineEvalConfig = field(default_factory=OnlineEvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mooncake: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cache_dir: str = "./cache"
    cache_key: Optional[str] = None
    model_download_dir: Optional[str] = None
    output_dir: str = ""


_ALWAYS_LOCAL_PATH_KEYS = ("output_dir", "cache_dir", "model_download_dir")
_DATA_PATH_KEYS = ("dataset.train_data_path", "dataset.eval_data_path")


def _resolve_relative_paths(
    config: DictConfig,
    base_dir: str,
    *,
    skip_keys: frozenset[str] = frozenset(),
) -> None:
    """Resolve local relative paths in *config* against *base_dir* (in-place).

    Always-local keys (output_dir, cache_dir, …) are absolutized unconditionally.
    Data-path keys are only absolutized when ``is_local_data_path`` says they look
    like filesystem paths (as opposed to HF Hub dataset IDs).

    Keys listed in *skip_keys* are left untouched (useful for deferring
    CWD-relative keys when resolving a file-level config).
    """
    for dotted_key in (*_ALWAYS_LOCAL_PATH_KEYS, *_DATA_PATH_KEYS):
        if dotted_key in skip_keys:
            continue
        val = OmegaConf.select(config, dotted_key, default=None)
        if not (isinstance(val, str) and val):
            continue

        expanded = os.path.expanduser(val)
        if os.path.isabs(expanded):
            if expanded != val:
                OmegaConf.update(config, dotted_key, expanded)
            continue

        if dotted_key in _ALWAYS_LOCAL_PATH_KEYS or is_local_data_path(
            expanded, base_dir=base_dir
        ):
            OmegaConf.update(config, dotted_key, os.path.abspath(os.path.join(base_dir, expanded)))


def _validate_vllm_config(config: DictConfig) -> None:
    """Raise if the vllm backend is selected with unsupported feature flags."""
    if config.model.target_model_backend != "vllm":
        return
    unsupported_flags = {
        "inference.vllm.enable_multimodal": "enable_multimodal",
        "training.train_with_decode": "train_with_decode",
    }
    for key, label in unsupported_flags.items():
        if OmegaConf.select(config, key):
            raise NotImplementedError(f"{label} is not yet supported with the vllm backend!")


def _save_config_snapshot(config: DictConfig) -> None:
    """Save the resolved config to output_dir/config.yaml if output_dir is set."""
    output_dir = OmegaConf.select(config, "output_dir", default=None)
    if not output_dir:
        return
    dest = Path(output_dir) / "config.yaml"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, str(dest))
        logger.info(f"Saved resolved config to {dest}")
    except OSError as e:
        logger.warning(f"Failed to save config to {dest}: {e}")


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[list] = None,
    base_config: Optional[DictConfig] = None,
    save_snapshot: bool = False,
) -> DictConfig:
    schema = OmegaConf.structured(Config)

    configs_to_merge = [schema]

    if base_config is not None:
        configs_to_merge.append(base_config)

    if config_path is not None:
        file_config = OmegaConf.load(config_path)
        _resolve_relative_paths(
            file_config,
            os.path.dirname(os.path.abspath(config_path)),
            skip_keys=frozenset(_ALWAYS_LOCAL_PATH_KEYS),
        )
        configs_to_merge.append(file_config)

    if cli_args:
        cli_config = OmegaConf.from_dotlist(cli_args)
        configs_to_merge.append(cli_config)

    config = OmegaConf.merge(*configs_to_merge)
    _resolve_relative_paths(config, os.getcwd())

    _validate_vllm_config(config)

    if save_snapshot:
        _save_config_snapshot(config)

    return config


# Sub-sections whose fields receive a name prefix when flattened.
_PREFIXED_SECTIONS = {
    "decode": "decode_",
    "online_eval": "online_eval_",
    "mooncake": "mooncake_",
    "sglang": "sglang_",
    "vllm": "vllm_",
}


def config_to_flat_args(config: DictConfig) -> argparse.Namespace:
    flat: dict[str, Any] = {}

    def _add(key: str, val: Any, origin: str) -> None:
        if key in flat:
            raise ValueError(f"Duplicate config key '{key}' (from '{origin}')")
        flat[key] = val

    for section_name, section in config.items():
        if not isinstance(section, DictConfig):
            _add(section_name, section, section_name)
            continue

        prefix = _PREFIXED_SECTIONS.get(section_name, "")
        for key, val in section.items():
            # Nested sub-config (e.g. inference.sglang) — flatten with its
            # own prefix so consumers keep seeing ``sglang_tp_size`` etc.
            if isinstance(val, DictConfig) and key in _PREFIXED_SECTIONS:
                sub_prefix = _PREFIXED_SECTIONS[key]
                for sub_key, sub_val in val.items():
                    _add(
                        f"{sub_prefix}{sub_key}",
                        sub_val,
                        f"{section_name}.{key}.{sub_key}",
                    )
            else:
                _add(f"{prefix}{key}", val, f"{section_name}.{key}")

    # --- Computed / alias fields ---
    flat["world_size"] = flat["training_num_nodes"] * flat["training_num_gpus_per_node"]
    flat["rank"] = 0
    flat["dynamic_loss_mask"] = flat["defer_tokenization"] and not flat["train_with_decode"]
    flat["use_wandb"] = flat.get("use_wandb", False) or flat.get("report_to") == "wandb"
    flat["use_tensorboard"] = (
        flat.get("use_tensorboard", False) or flat.get("report_to") == "tensorboard"
    )
    flat["checkpoint_dir"] = (
        str(Path(flat["output_dir"]) / "checkpoints") if flat.get("output_dir") else None
    )
    if flat.get("continual_training") and not flat.get("load_path"):
        logger.warning("continual_training=True but no training.load_path was provided")

    if "last_hidden_states_prenorm" not in flat or flat["last_hidden_states_prenorm"] is None:
        flat["last_hidden_states_prenorm"] = flat.get("inference_engine_type") == "vllm"

    # MTP draft-input norm. When True, the post-final-norm hidden (the same tensor
    # fed to the teacher lm_head) is fed to the MTP block's hnorm. When False
    # (default), the raw pre-norm residual is fed to hnorm. Only affects step 0 of
    # the TTT loop; later steps already feed back the post-norm backbone output.
    if "mtp_draft_input_postnorm" not in flat or flat["mtp_draft_input_postnorm"] is None:
        flat["mtp_draft_input_postnorm"] = False

    # On-policy TTT. Missing/None → off (teacher forcing), so pre-existing configs
    # are unaffected.
    if "mtp_ttt_on_policy" not in flat or flat["mtp_ttt_on_policy"] is None:
        flat["mtp_ttt_on_policy"] = False

    return argparse.Namespace(**flat)


def save_config(config: DictConfig, path: str) -> None:
    OmegaConf.save(config, path)


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
