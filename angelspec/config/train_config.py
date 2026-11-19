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
    # "megatron_bin" forces the offline-packed loader; anything else -> conversation jsonl.
    data_format: Optional[str] = None
    # Weighted blend of Megatron sources: path to a text file whose lines are
    # "<weight> <prefix> <kind>" (kind "vlm" enables image-sidecar loading).
    # When set, overrides train_data_path (see load_megatron_bin_blend).
    bin_data_list: Optional[str] = None
    # Megatron .bin/.idx loader knobs (see angelspec/data/megatron_bin.py).
    megatron_max_bins: int = -1  # -1 = all bins; small value for smoke tests
    megatron_bin_start: int = 0  # first bin read; with megatron_max_bins selects a bin window
    megatron_sample_size: int = -1  # -1 = every conversation; N = seeded random subset of N
    megatron_sample_seed: int = 0
    megatron_loss_offset: int = 300000  # id+offset marks context (no-loss) tokens
    megatron_doc_sep_id: int = 120026  # per-document separator token
    megatron_pad_id: int = 120002  # pad token (excluded from loss)
    megatron_image_start_token_id: int = 120118  # one per image; slices VL image sidecar refs
    megatron_image_end_token_id: int = 120119
    megatron_video_start_token_id: int = 120122
    megatron_video_end_token_id: int = 120123
    megatron_image_mask_token_ids: list[int] = field(
        default_factory=lambda: [120118, 120119, 120120, 120121, 120122, 120123, 120124]
    )
    # Semicolon-separated "old_prefix:new_prefix" mappings for absolute patch paths.
    megatron_image_path_replacements: str = ""
    # Rewrite image_newline tokens inside image spans to the image placeholder, the
    # serving (vLLM / HF) layout when the target's vision tower emits the per-row
    # newline embedding itself. None: on for Hy4 flash targets unless their
    # vision_config.use_image_newline_token is false.
    megatron_image_newline_as_placeholder: Optional[bool] = None
    megatron_blend_unit: str = "tokens"  # "tokens" preserves fixed-bin weight semantics
    megatron_num_proc: int = 16
    drop_overlength: bool = (
        False  # drop (not truncate) samples whose token count exceeds max_seq_length-1
    )
    # Drop training samples shorter than this many tokens (0 keeps all), so training lengths
    # fall in [min_seq_length, max_seq_length - 1]. Eval sets are not filtered.
    min_seq_length: int = 0
    eval_data_path: Optional[str] = None
    eval_datasets: Optional[dict[str, str]] = None
    # Length cap for eval samples, defaulting to training.max_seq_length (e.g. train at 32k and
    # evaluate a 64k set). The inference engines and Mooncake buffers are sized for the longer one.
    eval_max_seq_length: Optional[int] = None
    eval_interval: int = 50
    eval_micro_batch_size: Optional[int] = None
    eval_prompt_key: Optional[str] = None
    eval_seed: int = 42
    # Seconds an eval may go without dispatching a batch before it is aborted.
    eval_idle_timeout_s: float = 300.0
    # When True (default), eval hidden states are generated once before training
    # (or loaded from disk) and kept in each trainer's CPU memory for the whole
    # run. Hidden states for long-context eval sets are large, so this can pin
    # tens of GB of host RAM until the process exits. Set False to skip the
    # disk/CPU cache and re-extract hidden states via the inference engine at
    # each eval interval, then drop them immediately afterwards.
    cache_eval_hidden_states: bool = True
    last_turn_loss_only: Any = "auto"  # bool or "auto"
    min_loss_tokens: int = (
        0  # DFlash: skip sequences with < N supervised tokens (use 2*block_size)
    )
    num_proc: int = 64
    prompt_key: str = "conversations"
    shuffle_dataset: bool = True
    train_data_path: str = ""
    # Stream the source dataset through a bounded in-flight window instead of
    # materializing every raw/tokenized sample at once (host-memory relief for
    # very large / long-context datasets). stream_buffer_size caps in-flight items.
    stream_dataset_loading: bool = False
    stream_buffer_size: int = 4096


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
    # SwanLab mirrors the wandb knobs; workspace is wandb's team and exp_name is
    # wandb's run name. angelspec/utils/swanlab.py reads all of these on every
    # run, so they must stay declared even when only wandb is enabled.
    use_swanlab: bool = False
    swanlab_dir: Optional[str] = None
    swanlab_group: Optional[str] = None
    swanlab_host: Optional[str] = None
    swanlab_key: Optional[str] = None
    swanlab_mode: Optional[str] = None
    swanlab_project: Optional[str] = None
    swanlab_exp_name: Optional[str] = None
    swanlab_random_suffix: bool = True
    swanlab_run_id: Optional[str] = None
    swanlab_workspace: Optional[str] = None


@dataclass
class ModelConfig:
    draft_model_config: Optional[str] = None
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    norm_key: str = "model.norm.weight"
    target_model_backend: Optional[str] = None
    target_model_path: str = ""
    trust_remote_code: bool = False


@dataclass
class TrainingConfig:
    attention_backend: str = "sdpa"
    colocate: bool = False
    continual_training: bool = False
    # Step of the load_path checkpoint at which the training data was switched. Resuming
    # exactly from it starts the new dataset at its first sample, counts num_epochs from
    # this step and re-plans the LR decay to reach min_lr at the new final step.
    data_switch_step: Optional[int] = None
    # Evaluate the initial (or load_path) draft once, then exit without training.
    eval_only: bool = False
    distributed_backend: str = "nccl"
    distributed_timeout_minutes: int = 10
    draft_accumulation_steps: int = 1
    fsdp_reduce_dtype: str = "float32"  # "float32" or "bfloat16"
    fsdp_strategy: str = "REPLICATE"
    fsdp_backward_prefetch: bool = True
    # FSDP (FULL_SHARD) param/grad/optimizer sharding extent. Two ways to set it:
    #   Preferred (explicit): hsdp_shard_dp = how many DP ranks to shard across
    #     (default = dp_size); hsdp_shard_sp = how many SP ranks to fold into the
    #     shard dim (default 1 = pure DP, SP replicated). Effective shard size =
    #     hsdp_shard_dp * hsdp_shard_sp. Folding SP (sp>1) requires the full DP dim
    #     sharded (hsdp_shard_dp == dp_size). Bigger shard = less mem/GPU, more comm.
    #   Legacy: hsdp_shard_size (raw int). <=dp_size shards pure DP; >dp_size shards
    #     full DP + an SP block. Ignored when hsdp_shard_dp/hsdp_shard_sp are set.
    # All None → conservative per-node default (<= dp_size, SP replicated).
    # Without USP only hsdp_shard_dp (or hsdp_shard_size) applies: a value below
    # the world size shards inside blocks of that many consecutive ranks and
    # replicates across them; unset shards over every rank.
    hsdp_shard_size: Optional[int] = None
    hsdp_shard_dp: Optional[int] = None
    hsdp_shard_sp: Optional[int] = None
    # Controls which workload claims head-node GPUs first under PACK strategy.
    # "training_first" (default), "inference_first", or "custom".
    placement_strategy: str = "training_first"
    training_node_ips: Optional[list[str]] = None
    training_node_selectors: Optional[list[dict[str, str]]] = None
    compile_model: bool = False  # torch.compile the full training model
    sp_ring_size: int = 1
    sp_ulysses_size: int = 1
    # USP data delivery mode. False: the inference engine pre-shards each sample's seq
    # into per-rank Mooncake keys ({key}_usp{rank}) and every SP rank loads only its
    # slice (SGLang). True: the engine writes ONE unsharded sample; the controller fans
    # the SAME full sample to every SP rank and the model slices its own seq shard
    # in-forward. None: True under USP with vLLM or offline data (both store whole
    # samples), False otherwise.
    usp_local_shard: Optional[bool] = None
    # One Mooncake read per SP group (the leader scatters to its peers) instead of one
    # per SP rank. None: on for online MTP training under USP with usp_local_shard.
    sp_leader_mooncake_fetch: Optional[bool] = None

    # None: on for MTP drafts, off otherwise.
    gradient_checkpointing: Optional[bool] = None
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
    # Decay for the AdamW optimizer: every param under optimizer_type="adamw", or the
    # non-matrix group (norms, gates, embeddings, LM head) under "muon".
    weight_decay: float = 0.0
    # Decay for the Muon group (>=2-D weight matrices). Unset inherits weight_decay,
    # which is the historical single-value behaviour. Set both to decay weight
    # matrices at the Muon convention of 0.1 without applying that to norms and gates.
    muon_weight_decay: Optional[float] = None
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
    # Muon on >2-D weights (fused MoE experts [E, in, out]). True orthogonalizes every
    # expert's [in, out] matrix on its own, on the rank holding it (FSDP2 shards experts
    # whole along dim 0), so nothing is gathered. False flattens each stack to one
    # [E, in*out] matrix, which gives every expert the same update norm however few
    # tokens reached it.
    muon_per_expert: bool = True
    # Muon per attention head for the matrices a draft's attention declares
    # (muon_head_rows()): each head's rows are orthogonalized on their own, as DeepSeek
    # trains query and key weights.
    muon_per_head: bool = True
    # LR multiplier for a DFlash or MTP draft's MoE FFNs: each layer's routed and
    # shared experts, router, and the post-attention norm feeding them, i.e. what
    # target_mlp_init_layers / the MTP warm start copies from the target. They follow
    # the shared warmup/decay curve times this factor (init, peak and min LR alike);
    # every other param keeps learning_rate. 1.0 keeps a single param group.
    moe_lr_scale: float = 1.0
    # Router-only override of moe_lr_scale; unset follows moe_lr_scale.
    moe_router_lr_scale: Optional[float] = None
    num_epochs: int = 10
    num_train_steps: Optional[int] = None
    micro_batch_size: int = 2
    prefetch_depth: int = 2  # 0 = disabled, >0 = async pre-fetch N batches ahead
    predispatch_next_step: bool = True
    fail_on_unavailable_mooncake: bool = False
    save_interval: int = 5000
    save_per_epoch: bool = False
    max_checkpoints: int = 0  # 0 = keep all, N > 0 = rotate and keep only N most recent
    # Leave each epoch-end checkpoint out of max_checkpoints rotation: save_per_epoch's,
    # and the final one when training stops on an epoch boundary. Their iterations are
    # listed in the checkpoint directory's epoch_checkpointed_iterations.txt.
    keep_epoch_checkpoints: bool = False
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
    # Sample independent MTP branches and score them with decoder-mode targets.
    mtp_draft_opd: bool = False
    # Model-only DCP initialization; load_path resumes a complete OPD checkpoint.
    mtp_opd_init_model: Optional[str] = None
    # Full-vocabulary overlap: -log(alpha), or 1 - cumulative product of alpha.
    mtp_opd_loss: str = "lk_alpha"
    mtp_opd_temperature: float = 1.0
    mtp_opd_prefix_chunk_size: int = 2048
    mtp_opd_fp8_indexer: bool = True
    mtp_opd_verify_max_num_seqs: int = 1024
    mtp_opd_verify_max_num_batched_tokens: int = 65536
    # Half on-policy distillation for MTP via packed tree-forward scoring. When
    # True, the draft rolls out an on-policy block at each supervised anchor, a
    # single packed tree-forward has the target score every generated token, and
    # an OPD/KL term is added on the branch positions (block-level off-policy).
    # GPU-gated (needs the FlexAttention scoring engine). False (default) = off.
    mtp_opd_tree_forward: bool = False
    # Drafts per anchor. Tree-forward requires <= ttt_length; draft-OPD requires equality.
    mtp_opd_block_size: int = 8
    # Cap on anchors per sequence. Draft-OPD requires a positive cap;
    # tree-forward also accepts None for every supervised position.
    mtp_opd_max_anchors: Optional[int] = None
    warmup_ratio: float = 0.015

    # WSD LR schedule parameters (used by DFlash trainer only)
    wsd_decay_ratio: float = 0.2
    wsd_decay_style: Optional[str] = None

    # DFlash-specific parameters (ignored for Eagle3 training)
    #
    # Warm-start the drafter's own weights from an HF-style directory (config.json
    # + safetensors) before FSDP shards it. This seeds weights only; load_path is
    # the DCP training checkpoint that also restores the step, optimizer and LR
    # state. Applied on rank 0 with strict=False, so
    # anything the directory does not carry keeps its init value.
    draft_init_path: Optional[str] = None
    dflash_block_size: int = 16
    dflash_loss_decay_gamma: float = 7.0
    dflash_num_anchors: int = 512
    # None = take num_target_layers from the draft config (see _validate_dflash_target_layers).
    dflash_num_target_layers: Optional[int] = None
    # Set automatically by train_entry, not meant to be configured by hand: when
    # target_layer_ids asks for the target's final layer, vllm delivers that one
    # through last_hidden_states (its reserved post-final-layer capture slot)
    # rather than inside the concatenated aux hidden states, so the trainer takes
    # the last context layer from there. See _validate_dflash_target_layers.
    dflash_final_layer_from_last_hidden: bool = False
    # Intra-block attention direction for the DFlash family (dflash / dfly /
    # dspark / dflare). Default False reproduces the legacy block-bidirectional
    # behavior (every draft token in a block sees all other draft tokens in the
    # same block). Set True to make intra-block attention causal (a draft token
    # only sees earlier draft tokens in its block). Context attention is causal
    # regardless (kv_idx < anchor_pos), independently of sequence parallelism.
    dflash_causal_attn: bool = False
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
    dflash_loss_checkpoint_chunks: bool = False
    # Eval only, for target_kv_layers drafters: compare each draft layer's attention with
    # its target layer's on this many blocks per sample, and count how many image tokens
    # the candidates keep (angelspec/models/dflash_attn_diag.py). 0 disables.
    dflash_attn_diag_blocks: int = 0
    # Eval only, for target_kv_layers drafters: replace the draft attention distribution
    # with the target's ("image": over the candidate image tokens, keeping their total
    # mass; "context": over all candidates; "identity": recompute only) in these draft
    # layers (None: all), to measure what matching the target's attention is worth.
    dflash_attn_oracle: str = ""
    dflash_attn_oracle_layers: Optional[list[int]] = None
    # Eval only, for drafters with attentive_correction: leave its attention out (the SwiGLU
    # correction stays), to measure what it adds.
    dflash_acorr_eval_off: bool = False
    # Eval only, for drafters with enable_hidden_correction: compare the serving-time
    # top-K form of the correction (correct only the K highest uncorrected logits and
    # pick among them, which skips the per-slot full-vocabulary LM head) with the full
    # correction, for each K listed. None disables.
    dflash_hc_topk_diag: Optional[list[int]] = None
    # Eval only: speculative-sampling acceptance per position with the target sampling at
    # this temperature, p and q the target's and the drafter's distributions there:
    # sum_v min(p_v, q_v) when the drafter samples its token (eval/sample_accept_<i>), and
    # p of its argmax when it proposes that, as vLLM's DFlash does (eval/argmax_accept_<i>);
    # each also summed over prefix products (eval/*_accept_len). Needs the teacher logits,
    # so a distill term must be on. None disables.
    dflash_eval_accept_temperature: Optional[float] = None
    dfly_context_chunk_size: int = 0
    dfly_context_checkpoint: bool = False
    mtp_alignment_chunk_size: int = 0
    dflash_ce_loss_alpha: float = 1.0
    dflash_l1_loss_alpha: float = 0.0
    dflash_kl_loss_weight: float = 0.0
    dflash_kl_topk: int = 10
    dflash_lk_loss_weight: float = 0.0
    dflash_lk_loss_type: str = "hybrid"  # "alpha" or "hybrid"
    dflash_lk_eta: float = 3.0
    # End-to-end multi-step TV loss (γ-step MTP; γ=block_size). Independent term
    # added on top of the total loss (not mutually exclusive with KL/LK). Needs
    # target last_hidden_states. 0 disables.
    dflash_e2e_tv_loss_weight: float = 0.0
    # Vocab-dim chunk size for the streaming softmax+min inside the e2e-TV loss.
    # 0 (default) falls back to env var ``ANGELSPEC_DFLASH_E2E_TV_VOCAB_CHUNK``,
    # then the model default (8192). Smaller values reduce peak GPU memory (for
    # large vocab / long-seq training that OOMs in ``_compute_e2e_tv_loss``) at
    # the cost of more kernel launches.
    dflash_e2e_tv_vocab_chunk: int = 0
    # Path selector (enable_path_selector in the draft config): weight of its
    # K-way cross-entropy over the unary top-k, teacher-forcing the predecessor
    # token. Independent term added on top of the total, like e2e-TV. Ignored
    # unless the draft config actually builds the selector head, so the default
    # is the "on when the head exists" weight rather than 0.
    dflash_path_selector_loss_alpha: float = 1.0
    # Which token the selector must rank first. True (default): follow whatever the
    # unary head is trained against, i.e. switch to the target's argmax whenever the
    # corpus CE stops driving it — either displaced by KL/LK (dflash_kl_loss_weight /
    # dflash_lk_loss_weight) or dropped by dflash_ce_loss_alpha=0, as in a pure
    # e2e-TV recipe — since that is also the token acceptance is tested against at
    # serving time. CE + e2e-TV keeps the corpus token, because e2e-TV is added on
    # top of the CE rather than replacing it.
    # False restores the pre-fix behaviour (always the corpus token), for A/B only.
    dflash_path_selector_teacher_labels: bool = True
    # Whether the selector's K-way CE rides the objective weights (dflash_loss_objective:
    # decay / dpace) like the unary terms. D-PACE's continuation value answers "which
    # block position is worth fixing", which is a question about capacity allocation —
    # but the selector is two token-indexed codebooks plus one projection, shared across
    # positions and teacher-forced on ground-truth predecessors everywhere, so the
    # weighting mostly skews which bigrams it sees. Under dpace_alpha=0.5 with a
    # low-confidence drafter that skew reaches ~100x between block slot 1 and 7.
    # False weighs every supervised position equally (denominator switches to the
    # unweighted count to match). True keeps the historical behaviour.
    dflash_path_selector_objective_weighting: bool = True
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
    # Decoder branch scoring retains multimodal prompts and MRoPE positions.
    dflash_opd_backend: str = "packed_tree"
    dflash_opd_loss: str = "two_stream_k3"
    dflash_opd_vocab_chunk: int = 64
    dflash_opd_accept_temperature: float = 0.9
    dflash_opd_accept_top_k: int = 0
    dflash_opd_accept_top_p: float = 1.0
    dflash_opd_relax_temperature: float = 1.0
    dflash_opd_on_policy_prev: bool = False
    dflash_opd_verify_max_num_seqs: int = 1024
    dflash_opd_verify_max_num_batched_tokens: int = 65536
    dflash_opd_gt_shortcut: bool = True
    dflash_opd_teacher_parity_steps: int = 0
    dflash_opd_teacher_parity_branches: int = 8
    dflash_opd_teacher_parity_tol: float = 0.001
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
    mtp_source_metrics: bool = False
    mtp_metric_sources: Optional[list[str]] = None
    mtp_argmax_chunk_size: int = 0
    mtp_loss_checkpoint_chunks: bool = False
    packing_balance_dp: bool = False

    # Optional controller-side packing efficiency guard shared by DFlash and MTP.
    # 0 disables waiting (legacy behavior). When a candidate dispatch is below the
    # threshold, the controller waits for more inference supply, but flushes the
    # partial rows after packing_max_wait_seconds to avoid deadlock at train end.
    packing_min_fill_ratio: float = 0.0
    packing_max_wait_seconds: float = 0.0

    # DSpark-specific parameters (DFlash backbone + Markov/L1/confidence heads).
    # block_size is read from dflash_block_size. The other dspark_* knobs default to
    # None, which takes the matching dflash_* value (an unset dflash_num_target_layers
    # is the draft config's); the DSpark paper recipe is gamma 4.0, ce 0.1, l1 0.9.
    dspark_num_anchors: Optional[int] = None
    dspark_num_target_layers: Optional[int] = None
    dspark_loss_decay_gamma: Optional[float] = None
    dspark_ce_loss_alpha: Optional[float] = None
    dspark_l1_loss_alpha: Optional[float] = None
    dspark_confidence_head_alpha: float = 1.0
    # Detach the confidence head's input ([hidden; Markov embedding]) so its BCE
    # trains the head alone and the drafter trains on the DFlash loss only.
    dspark_confidence_detach: bool = False


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
class OfflineConfig:
    manifest_path: Optional[str] = None
    tokenized_path: Optional[str] = None
    frozen_target_path: Optional[str] = None
    format: str = "auto"
    verify_hashes: bool = True
    eval_samples: int = 4
    # JSON {"rows": [[sample_id, ...], ...]} in global row order (row p -> DP rank p % dp).
    row_plan_path: Optional[str] = None


@dataclass
class OnlineDumpConfig:
    """Lossless online-training capture into the native offline format."""

    enabled: bool = False
    # Capture without training: only the inference side of the run is started.
    dump_only: bool = False
    dump_path: Optional[str] = None
    frozen_target_path: Optional[str] = None
    validation_fraction: float = 0.05
    capture_epochs: int = 1
    # Copied samples waiting for a file writer; size it to writer_threads samples.
    max_pending_bytes: int | str = 32 * 1024**3
    # Dump actors, each copying the samples routed to it (equal samples to the same
    # one). Unset puts one on every inference node other than the driver's.
    writers: Optional[int] = None
    # Parallel file writers and Mooncake readers inside each dump actor.
    writer_threads: int = 16
    reader_threads: int = 8
    # Dispatched batches the writer may still be copying before dispatch pauses.
    # Unset, one optimizer step (draft_accumulation_steps): a step's batches are
    # copied while that step trains.
    max_inflight_batches: Optional[int] = None
    # Publication re-reads every file when true; otherwise it checks sizes and file metadata.
    verify_on_publish: bool = False
    # Ray node IP for every writer. Unset or "auto" spreads them over the inference
    # nodes other than the driver's.
    writer_node_ip: Optional[str] = None
    writer_env_vars: dict[str, str] = field(default_factory=dict)
    strict: bool = True


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    online_eval: OnlineEvalConfig = field(default_factory=OnlineEvalConfig)
    online_dump: OnlineDumpConfig = field(default_factory=OnlineDumpConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    mooncake: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cache_dir: str = "./cache"
    cache_key: Optional[str] = None
    model_download_dir: Optional[str] = None
    output_dir: str = ""


_ALWAYS_LOCAL_PATH_KEYS = ("output_dir", "cache_dir", "model_download_dir")
_DATA_PATH_KEYS = (
    "dataset.train_data_path",
    "dataset.eval_data_path",
    "offline.manifest_path",
    "offline.tokenized_path",
    "offline.frozen_target_path",
    "offline.row_plan_path",
    "online_dump.dump_path",
    "online_dump.frozen_target_path",
)


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
    named = OmegaConf.select(config, "dataset.eval_datasets", default=None)
    if isinstance(named, DictConfig):
        for name, path in list(named.items()):
            if not isinstance(path, str) or not path:
                continue
            expanded = os.path.expanduser(path)
            if os.path.isabs(expanded):
                named[name] = expanded
            elif is_local_data_path(expanded, base_dir=base_dir):
                named[name] = os.path.abspath(os.path.join(base_dir, expanded))
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
    if config.inference.inference_engine_type != "vllm":
        return
    unsupported_flags = {
        "training.train_with_decode": "train_with_decode",
    }
    for key, label in unsupported_flags.items():
        if OmegaConf.select(config, key):
            raise NotImplementedError(f"{label} is not yet supported with the vllm backend!")


def _validate_eval_max_seq_length(config: DictConfig) -> None:
    eval_len = config.dataset.eval_max_seq_length
    if eval_len is None:
        return
    if eval_len < 2:
        raise ValueError(f"dataset.eval_max_seq_length must be >= 2, got {eval_len}")
    engine_len = OmegaConf.select(config, "inference.vllm.max_model_len", default=None)
    if (
        config.inference.inference_engine_type == "vllm"
        and engine_len is not None
        and engine_len < eval_len
    ):
        raise ValueError(
            f"inference.vllm.max_model_len={engine_len} is shorter than "
            f"dataset.eval_max_seq_length={eval_len}; longer eval samples would be skipped"
        )


def _validate_online_dump_config(config: DictConfig) -> None:
    if config.online_dump.dump_only and not config.online_dump.enabled:
        raise ValueError("online_dump.dump_only requires online_dump.enabled=true")
    if not config.online_dump.enabled:
        return
    if config.offline.manifest_path:
        raise ValueError("online_dump.enabled cannot be combined with offline.manifest_path")
    if not config.online_dump.dump_path or not config.online_dump.frozen_target_path:
        raise ValueError(
            "online_dump.enabled requires online_dump.dump_path and "
            "online_dump.frozen_target_path"
        )
    if (
        Path(config.online_dump.dump_path).resolve()
        == Path(config.online_dump.frozen_target_path).resolve()
    ):
        raise ValueError("online_dump.dump_path and frozen_target_path must be different")
    if not 0 < config.online_dump.validation_fraction < 1:
        raise ValueError("online_dump.validation_fraction must be between 0 and 1")
    if config.online_dump.capture_epochs < 1:
        raise ValueError("online_dump.capture_epochs must be positive")
    pending_bytes = config.online_dump.max_pending_bytes
    if isinstance(pending_bytes, str):
        from angelspec.config.mooncake_config import MooncakeConfig

        pending_bytes = MooncakeConfig.parse_size(pending_bytes)
        config.online_dump.max_pending_bytes = pending_bytes
    if pending_bytes < 1:
        raise ValueError("online_dump.max_pending_bytes must be positive")
    if (
        min(
            1 if config.online_dump.writers is None else config.online_dump.writers,
            config.online_dump.writer_threads,
            config.online_dump.reader_threads,
        )
        < 1
    ):
        raise ValueError("online_dump.writers, writer_threads and reader_threads must be positive")
    inflight = config.online_dump.max_inflight_batches
    if inflight is not None and inflight < 1:
        raise ValueError("online_dump.max_inflight_batches must be positive")
    if not config.online_dump.strict:
        raise ValueError("Online dump v1 is lossless-only; set online_dump.strict=true")
    if config.dataset.defer_tokenization:
        raise ValueError("Online dump v1 requires dataset.defer_tokenization=false")
    if config.training.train_with_decode:
        raise ValueError("Online dump v1 does not support training.train_with_decode")
    if config.inference.inference_engine_type not in ("vllm", "sgl"):
        raise ValueError("Online dump v1 requires inference_engine_type=vllm or sgl")
    multimodal = (
        config.inference.vllm.enable_multimodal
        if config.inference.inference_engine_type == "vllm"
        else config.inference.sglang.enable_multimodal
    )
    if multimodal and config.inference.inference_engine_type != "vllm":
        raise ValueError("Online VL dump currently requires the vllm backend")


def _validate_min_seq_length(config: DictConfig) -> None:
    min_len = config.dataset.min_seq_length
    if min_len < 0:
        raise ValueError(f"dataset.min_seq_length must be >= 0, got {min_len}")
    max_len = config.training.max_seq_length
    if min_len >= max_len:
        raise ValueError(
            f"dataset.min_seq_length={min_len} leaves no training samples: training keeps "
            f"at most training.max_seq_length - 1 = {max_len - 1} tokens"
        )


def _validate_keep_epoch_checkpoints(config: DictConfig) -> None:
    training = config.training
    if training.keep_epoch_checkpoints and not training.save_per_epoch:
        raise ValueError(
            "training.keep_epoch_checkpoints keeps the checkpoints training.save_per_epoch "
            "writes at each epoch end; set save_per_epoch"
        )


def _normalize_legacy_backend(config: DictConfig) -> None:
    legacy = OmegaConf.select(config, "model.target_model_backend", default=None)
    current = OmegaConf.select(config, "inference.inference_engine_type", default=None)
    if not legacy:
        return
    legacy = {"sglang": "sgl"}.get(str(legacy), str(legacy))
    if current is None:
        OmegaConf.update(config, "inference.inference_engine_type", legacy, force_add=True)
    elif current != legacy:
        logger.warning(
            "Ignoring deprecated model.target_model_backend=%s because "
            "inference.inference_engine_type=%s is set",
            legacy,
            current,
        )


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
        _normalize_legacy_backend(file_config)
        _resolve_relative_paths(
            file_config,
            os.path.dirname(os.path.abspath(config_path)),
            skip_keys=frozenset(_ALWAYS_LOCAL_PATH_KEYS),
        )
        configs_to_merge.append(file_config)

    if cli_args:
        cli_config = OmegaConf.from_dotlist(cli_args)
        _normalize_legacy_backend(cli_config)
        configs_to_merge.append(cli_config)

    config = OmegaConf.merge(*configs_to_merge)
    _resolve_relative_paths(config, os.getcwd())

    _validate_vllm_config(config)
    _validate_eval_max_seq_length(config)
    _validate_online_dump_config(config)
    _validate_min_seq_length(config)
    _validate_keep_epoch_checkpoints(config)
    from angelspec.config.eval_config import named_eval_datasets

    named_eval_datasets(
        argparse.Namespace(
            eval_datasets=config.dataset.eval_datasets,
            eval_data_path=config.dataset.eval_data_path,
            offline_manifest_path=config.offline.manifest_path,
        )
    )

    if save_snapshot:
        _save_config_snapshot(config)

    return config


# Sub-sections whose fields receive a name prefix when flattened.
_PREFIXED_SECTIONS = {
    "offline": "offline_",
    "decode": "decode_",
    "online_dump": "online_dump_",
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
    flat["use_swanlab"] = flat.get("use_swanlab", False) or flat.get("report_to") == "swanlab"
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
