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

"""DFlash draft model: block-diffusion drafter with dual-source KV injection.

Architecture overview:
  - W_proj projects concatenated multi-layer target hidden states into context features
  - Each decoder layer uses dual-source KV: context KV (from target) + draft KV (from draft)
  - Bidirectional attention within each block; no inter-block attention
  - Shared embedding and LM head from target model (frozen)
  - Optional TreeFlash hidden-states correction before the LM head
    (``enable_hidden_correction``), applied by the training wrapper; with
    ``attentive_correction`` it also reads a draft layer's context again, its query
    knowing the previous token (see :mod:`angelspec.models.draft.attentive_correction`), optionally
    to the top-k uncorrected logits only (``enable_hidden_correction_topk``)
  - Optional path selector reranking the unary top-k (``enable_path_selector``,
    see :mod:`angelspec.models.draft.path_selector`), also driven by the wrapper
  - Optional MTP fusion of the target's final hidden at the last accepted token into
    the block input (``mtp_fusion``, see :class:`MTPFusion`), also added by the wrapper
  - ``layer_arch`` picks the decoder-layer body; ``draft_mlp_type`` picks its
    dense SwiGLU or target-style full MoE FFN, and ``attention_type`` its attention
    block from :mod:`angelspec.models.draft.attention_registry` (GQA, or hy4's gated MLA +
    DSA indexer)
"""

import json
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

from angelspec.models.draft.attention_registry import (
    build_draft_attention,
    draft_attention_types,
)
from angelspec.models.draft.path_selector import build_path_selector
from angelspec.models.ops.attention_mask import block_mask_sdpa
from angelspec.models.ops.mrope import validate_mrope_parameters
from angelspec.models.ops.ulysses import all_gather_seq
from angelspec.utils.distributed import get_sp_ulysses_group
from angelspec.utils.logging import logger

# One-shot USP attention proof log (shared across layers / ranks via class flag).
_USP_ATTN_LOGGED = False

#: Decoder-layer bodies. Orthogonal to the context wiring (which is what
#: distinguishes DFlash / DFlare / DFly), so a new body can be added here without
#: touching the context wiring. ``"dflash"`` is the dense SwiGLU layer.
LAYER_ARCHES = ("dflash",)
DRAFT_MLP_TYPES = ("dense", "moe")
INDEXER_TYPES = ("full", "shared")
DSA_BACKENDS = ("auto", "torch", "tilelang")
#: Attention types whose context is an MLA latent in the target's own space:
#: target_kv_layers hands them the target's latents, target_attn_init_layers copies the
#: target's attention into them, and dsa_backend "tilelang" runs them on the sparse-MLA
#: kernel, with these RoPE widths next to the 512-wide latent.
TARGET_MLA_ROPE_DIMS = {"mla": 64}
_TARGET_MLA_TYPES = " or ".join(repr(name) for name in TARGET_MLA_ROPE_DIMS)
#: Drafters built on DFlashDraftModel's own backbone (DFlashTrainer and its DSpark subclass).
DFLASH_BACKBONE_MODEL_TYPES = ("dflash", "dspark", "dfly2")
MTP_FUSION_MODES = ("slot0", "slot1", "broadcast")
MTP_FUSION_INITS = ("target", "random")


class DFlashConfig(PretrainedConfig):
    """Configuration for DFlash draft model.

    ``attention_type`` picks the attention block (see ``attention_registry``), so the
    dense body can run a target family's MLA in place of GQA. The MLA field names
    follow ``MTPConfig`` and the Hy4 flash target config, so a target's attention
    fields can be copied over as-is and only shrunk to drafter scale.
    """

    model_type = "dflash"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        vocab_size: int = 152064,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        rope_theta: Optional[float] = None,
        num_target_layers: int = 5,
        target_hidden_size: int = 4096,
        target_num_hidden_layers: int = 36,
        target_layer_ids: Optional[List[int]] = None,
        mask_token_id: int = 151669,
        tie_word_embeddings: bool = False,
        model_arch: str = "dflash",
        fusion_type: str = "concat_fc",
        gate_temperature: float = 1.0,
        enable_hidden_correction: bool = False,
        hidden_correction_intermediate_size: Optional[int] = None,
        enable_hidden_correction_topk: bool = False,
        hidden_correction_topk: int = 64,
        # The hidden correction also attends to the context draft layer
        # attentive_correction_source_layer reads (the target's latents there), its query
        # knowing the previous token. See models/draft/attentive_correction.
        attentive_correction: bool = False,
        attentive_correction_source_layer: int = -1,
        # Its keys also include the slot's own block up to the slot (the drafted prefix):
        # in the context's softmax ("shared") or a softmax of their own added through a
        # zero-started tanh gate ("gated").
        attentive_correction_block_keys: bool = False,
        attentive_correction_block_mode: str = "shared",
        # The gated block read's kv_b (key and value up-projections from the latent):
        # the context read's ("shared") or a copy of its own ("separate").
        attentive_correction_block_kv_b: str = "shared",
        # What kv_in projects each block slot's latent from: its hidden and the token before
        # it ("hidden_embed"), or its hidden alone ("hidden"), which leaves the block's keys
        # and values independent of the drafted tokens, like the context's.
        attentive_correction_block_kv_in: str = "hidden_embed",
        enable_path_selector: bool = False,
        path_selector_rank: int = 256,
        path_selector_top_k: int = 16,
        # Sliding window over the context: each draft block reads only the last
        # context_local_window context tokens before its anchor (None = all of them).
        # A list holds one window per draft layer.
        context_local_window: Optional[Union[int, List[Optional[int]]]] = None,
        # Context candidates from the target's own DSA indexer: each draft block also
        # reads the target_index_topk positions target layer target_index_layer scores
        # highest for the last accepted token (anchor - 1). See models/target/target_indexer.
        # A list holds one target layer per draft layer, each draft layer reading its
        # own target layer's candidates; None there leaves a draft layer its window alone.
        target_index_layer: Optional[Union[int, List[Optional[int]]]] = None,
        target_index_topk: Optional[int] = None,
        # Target KV reuse: draft layer i attends to target layer target_kv_layers[i]'s
        # own MLA latent, recomputed from the captured hidden, in place of a latent
        # projected from the fused context feature; the drafter then builds no context
        # projection. An int gives every draft layer the same target layer. See
        # models/target/target_kv_latent.
        target_kv_layers: Optional[Union[int, List[int]]] = None,
        # Score these target indexers at the shared anchor-1 and give every draft layer
        # the set-union of their top-k (plus context_local_window).
        # Needs target_index_layer / target_index_topk (which draft layers read the union).
        target_index_union_layers: Optional[List[int]] = None,
        # Cross-layer context KV: draft layer i attends over the context latent that draft
        # layer context_kv_source_layers[i] projects, instead of projecting its own; its
        # draft-block keys and kv_b_proj stay its own. [0, 0, 0, 0, 0] has every layer read
        # layer 0's, so only layer 0 projects (and, served, caches) the context.
        context_kv_source_layers: Optional[List[int]] = None,
        # Draft layer i's input_layernorm and attention start out as target layer
        # target_attn_init_layers[i]'s (copied when a run starts from scratch, then
        # trained); an int gives every draft layer the same target layer. See
        # models/target/target_attention_init.
        target_attn_init_layers: Optional[Union[int, List[int]]] = None,
        # Decoder FFN body. "dense" is the original DFlash SwiGLU. "moe" is the
        # target-style routed MoE from draft.mtp.Hy3MoE. With
        # target_mlp_init_layers, each draft layer copies the corresponding target
        # layer's post-attention norm, router, all routed experts and shared expert,
        # then trains every copied parameter normally.
        draft_mlp_type: str = "dense",
        target_mlp_init_layers: Optional[Union[int, List[int]]] = None,
        # Which target experts become the drafter's num_experts routed experts (their
        # router rows and correction biases follow): None copies all of them, so
        # num_experts must equal the target's; "bias" takes the num_experts lowest
        # target correction biases; "random" a draw seeded by the draft layer index;
        # or one list of expert ids per draft layer. A subset may route fewer experts
        # per token (num_experts_per_tok) than the target.
        target_mlp_init_experts: Optional[Union[str, List[List[int]]]] = None,
        # A dense FFN can start from target experts instead: draft layer i's DFlashMLP
        # is target layer target_mlp_dense_init_layers[i]'s shared expert(s) followed
        # by k routed experts along the intermediate dim (intermediate_size must be
        # (shared + k) * expert width), each routed expert's down projection scaled by
        # target_mlp_dense_init_routed_scale (unset: the target's routed scaling
        # factor / k, the mean weight route_norm gives a selected expert), and the
        # post-attention norm is copied too. target_mlp_dense_init_experts picks the
        # k experts: "bias" takes the ones with the lowest target expert-correction
        # bias (pushed down most by the target's balancer, i.e. its most popular),
        # "random" a seeded draw, or one list of expert ids per draft layer.
        target_mlp_dense_init_layers: Optional[Union[int, List[int]]] = None,
        target_mlp_dense_init_experts: Union[str, List[List[int]]] = "bias",
        target_mlp_dense_init_routed_scale: Optional[float] = None,
        num_experts: int = 256,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 768,
        moe_router_use_sigmoid: bool = True,
        moe_router_enable_expert_bias: bool = True,
        route_norm: bool = True,
        router_scaling_factor: float = 2.827,
        swiglu_limit: float = 0.0,
        moe_skip_expert_init: bool = False,
        moe_aux_free_bias_update: bool = False,
        moe_aux_free_bias_beta: float = 0.9,
        moe_aux_free_bias_gamma: float = 1e-4,
        # The target's MTP layer opens by fusing its final hidden at a position with the
        # next token's embedding; at the last accepted token, whose next token is the
        # anchor, it predicts the drafter's first token. The training wrapper adds that
        # fusion, through a zero-initialized gate, to the anchor slot's input ("slot0"), to
        # the first predicted slot's ("slot1") or to every slot's ("broadcast").
        # mtp_fusion_init="target" starts it from the target's MTP layer, "random" from
        # scratch. See MTPFusion.
        mtp_fusion: Optional[str] = None,
        mtp_fusion_init: str = "target",
        layer_arch: str = "dflash",
        # Attention block, read on every layer_arch, from attention_registry: "gqa" is
        # DFlashAttention, "mla" hy4's gated MLA (+ optional DSA indexer), both over the
        # same dual-source KV.
        attention_type: str = "gqa",
        # gated MLA (only read when attention_type="mla")
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        gated_mla: bool = False,
        gating_type: str = "elementwise",
        learnable_sink: bool = False,
        learnable_sink_init: float = 0.0,
        # RMSNorm after o_proj, as Hy4 flash's attention ends (its sandwich_norm's
        # post_proj_norm); needed to start from such a target's attention.
        attn_post_proj_norm: bool = False,
        # DSA indexer (only read when attention_type="mla")
        index_topk: Optional[int] = None,
        index_head_dim: int = 128,
        index_n_heads: int = 16,
        dsa_backend: str = "torch",
        dsa_fp8_indexer: bool = False,
        indexer_types: Optional[List[str]] = None,
        image_start_token_id: Optional[int] = None,
        image_end_token_id: Optional[int] = None,
        image_newline_token_id: Optional[int] = None,
        **kwargs,
    ):
        rope_parameters = kwargs.get("rope_parameters")
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        # hy4-style configs nest rope_theta under rope_parameters; read that
        # spelling too, so such a config can be copied over without silently
        # falling back to the 10k default. A flat rope_theta, when given, wins.
        if rope_theta is None:
            nested_theta = rope_parameters.get("rope_theta")
            rope_theta = float(nested_theta) if nested_theta is not None else 10000.0

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_target_layers = num_target_layers
        self.target_hidden_size = target_hidden_size
        self.target_num_hidden_layers = target_num_hidden_layers
        self.target_layer_ids = target_layer_ids
        self.mask_token_id = mask_token_id
        # Selects the draft model class within the shared DFlash training stack:
        # "dflash" -> DFlashDraftModel, "dflare" -> DFlareDraftModel. DFlare reuses
        # DFlashConfig wholesale (see dflare.py header), so the architecture choice
        # rides on this field rather than a separate config class.
        self.model_arch = model_arch
        # Context fusion mechanism: "concat_fc" = DFlash's concat -> Linear (default,
        # unchanged); "gated_sum" = per-candidate RMSNorm -> softmax gate -> weighted
        # sum for learnable layer selection (DFlashGatedDraftModel). gate_temperature
        # only applies to gated_sum.
        self.fusion_type = fusion_type
        self.gate_temperature = gate_temperature
        # TreeFlash hidden-states correction (see HiddenStatesCorrection). Off by
        # default, so a plain DFlash config is unchanged; DFly turns it on.
        self.enable_hidden_correction = enable_hidden_correction
        self.hidden_correction_intermediate_size = hidden_correction_intermediate_size
        # The correction's serving form, trained as such: only the hidden_correction_topk
        # highest uncorrected logits take their corrected value, the rest keep the
        # uncorrected one. Serving then runs the full-vocabulary LM head once per block,
        # in parallel, and the predecessor-dependent part on those candidates only,
        # instead of a full LM head per slot. Applied by the training wrapper.
        self.enable_hidden_correction_topk = enable_hidden_correction_topk
        self.hidden_correction_topk = hidden_correction_topk
        if enable_hidden_correction_topk:
            if not enable_hidden_correction:
                raise ValueError("enable_hidden_correction_topk needs enable_hidden_correction")
            if not 1 <= hidden_correction_topk <= vocab_size:
                raise ValueError(
                    f"hidden_correction_topk must be in [1, vocab_size={vocab_size}], "
                    f"got {hidden_correction_topk}"
                )
        # Path selector (see .path_selector): predecessor-conditioned rerank of
        # the unary top-k, which is what gives a block-parallel drafter an edge
        # between consecutive draft positions. Off by default, so a plain DFlash
        # config is unchanged.
        self.enable_path_selector = enable_path_selector
        self.path_selector_rank = path_selector_rank
        self.path_selector_top_k = path_selector_top_k
        if enable_path_selector:
            if path_selector_rank < 1:
                raise ValueError(f"path_selector_rank must be >= 1, got {path_selector_rank}")
            # 2 is the smallest set with something to rerank; above vocab_size the
            # top-k does not exist. Serving reranks before any draft-vocab mapping,
            # so the bound is the full vocabulary.
            if not 2 <= path_selector_top_k <= vocab_size:
                raise ValueError(
                    f"path_selector_top_k must be in [2, vocab_size={vocab_size}], "
                    f"got {path_selector_top_k}"
                )

        # Counted back from each block's anchor, so a block's queries share one
        # window. Deliberately not ``sliding_window``: vLLM reads that field as a
        # per-token window on the drafter's own attention and KV cache.
        per_layer_windows = isinstance(context_local_window, (list, tuple))
        if per_layer_windows:
            if len(context_local_window) != num_hidden_layers:
                raise ValueError(
                    f"context_local_window lists {len(context_local_window)} windows for "
                    f"{num_hidden_layers} draft layers"
                )
            context_local_window = list(context_local_window)
        for window in context_local_window if per_layer_windows else [context_local_window]:
            if window is not None and (
                isinstance(window, bool) or not isinstance(window, int) or window < 1
            ):
                raise ValueError(
                    f"context_local_window must be a positive int or None, got {window!r}"
                )
        self.context_local_window = context_local_window

        # With candidates on, context_local_window is the always-visible part next to
        # them (a block sees their union), not a cap on the whole context.
        if (target_index_layer is None) != (target_index_topk is None):
            raise ValueError("target_index_layer and target_index_topk must be set together")

        def check_target_layers(name, value, allow_none=False):
            # vLLM captures target_layer_id i as the input of layer i + 1, and the
            # target's last layer arrives after the head instead: reading layer L's
            # input (for its indexer or its latent) needs id L - 1 captured and below
            # that last layer.
            captured = (
                target_layer_ids
                if target_layer_ids is not None
                else build_target_layer_ids(num_target_layers, target_num_hidden_layers)
            )
            last_id = target_num_hidden_layers - 1 if target_num_hidden_layers else None
            per_layer = isinstance(value, (list, tuple))
            if per_layer and len(value) != num_hidden_layers:
                raise ValueError(
                    f"{name} lists {len(value)} target layers for {num_hidden_layers} draft layers"
                )
            for layer in value if per_layer else [value]:
                if layer is None and allow_none and per_layer:
                    continue
                if isinstance(layer, bool) or not isinstance(layer, int) or layer < 1:
                    raise ValueError(f"{name} must be a positive int, got {layer!r}")
                input_id = layer - 1
                if input_id not in captured or (last_id is not None and input_id >= last_id):
                    raise ValueError(
                        f"{name}={layer} reads its input from target_layer_id {input_id}, "
                        f"which must be in target_layer_ids={list(captured)} and below the "
                        f"target's last layer ({last_id})"
                    )

        if target_index_layer is not None:
            check_target_layers("target_index_layer", target_index_layer, allow_none=True)
            if isinstance(target_index_layer, (list, tuple)):
                if all(layer is None for layer in target_index_layer):
                    raise ValueError("target_index_layer names no target layer; leave it unset")
                # The sparse kernel reads a fixed-width key list per query, so a draft layer
                # without candidates needs a window to bound it.
                windows = windows_per_draft_layer(context_local_window, num_hidden_layers)
                for i, (layer, window) in enumerate(zip(target_index_layer, windows)):
                    if layer is None and window is None:
                        raise ValueError(
                            f"draft layer {i} has neither a target_index_layer nor a "
                            "context_local_window; give it a window"
                        )
            if (
                isinstance(target_index_topk, bool)
                or not isinstance(target_index_topk, int)
                or target_index_topk < 1
            ):
                raise ValueError(
                    f"target_index_topk must be a positive int, got {target_index_topk!r}"
                )
        if target_kv_layers is not None:
            check_target_layers("target_kv_layers", target_kv_layers)
            if attention_type not in TARGET_MLA_ROPE_DIMS:
                raise ValueError(
                    "target_kv_layers reuses the target's MLA latent; set "
                    f"attention_type={_TARGET_MLA_TYPES}"
                )
            if index_topk is not None:
                raise ValueError(
                    "with target_kv_layers there is no drafter-side context for the drafter's "
                    "own DSA indexer to score; drop index_topk (target_index_layer picks the "
                    "context instead)"
                )
            # DFly configs subclass this one but build their own context path; DSpark
            # only adds heads on top of the DFlash backbone.
            if (
                type(self).model_type not in DFLASH_BACKBONE_MODEL_TYPES
                or model_arch != "dflash"
                or fusion_type != "concat_fc"
            ):
                raise ValueError(
                    "target_kv_layers is wired through the DFlash backbone only (model_type in "
                    f"{DFLASH_BACKBONE_MODEL_TYPES}, model_arch 'dflash', fusion_type "
                    f"'concat_fc'), got {type(self).model_type!r} / {model_arch!r} / "
                    f"{fusion_type!r}"
                )
        self.target_index_layer = (
            list(target_index_layer)
            if isinstance(target_index_layer, (list, tuple))
            else target_index_layer
        )
        self.target_index_topk = target_index_topk
        self.target_kv_layers = (
            list(target_kv_layers)
            if isinstance(target_kv_layers, (list, tuple))
            else target_kv_layers
        )
        if target_index_union_layers is not None:
            if target_index_layer is None:
                raise ValueError(
                    "target_index_union_layers needs target_index_layer / target_index_topk "
                    "(which draft layers read the union)"
                )
            if (
                not isinstance(target_index_union_layers, (list, tuple))
                or not target_index_union_layers
            ):
                raise ValueError(
                    "target_index_union_layers must be a non-empty list of target layers, "
                    f"got {target_index_union_layers!r}"
                )
            # Flat list of indexers to score (not one entry per draft layer).
            for layer in target_index_union_layers:
                check_target_layers("target_index_union_layers", int(layer))
            self.target_index_union_layers = [int(layer) for layer in target_index_union_layers]
        else:
            self.target_index_union_layers = None
        if context_kv_source_layers is not None:
            sources = context_kv_source_layers
            if not isinstance(sources, (list, tuple)) or len(sources) != num_hidden_layers:
                raise ValueError(
                    f"context_kv_source_layers must list one draft layer per draft layer "
                    f"({num_hidden_layers}), got {sources!r}"
                )
            for i, source in enumerate(sources):
                # Layers run in order, so a layer can only read a latent already projected.
                if isinstance(source, bool) or not isinstance(source, int) or not 0 <= source <= i:
                    raise ValueError(
                        f"context_kv_source_layers[{i}] must be a draft layer in [0, {i}], "
                        f"got {source!r}"
                    )
                if sources[source] != source:
                    raise ValueError(
                        f"context_kv_source_layers[{i}] = {source} reads a layer that does not "
                        "project its own context latent; point it at that layer's source"
                    )
            if attention_type != "mla":
                raise ValueError(
                    "context_kv_source_layers shares MLA context latents; "
                    "set attention_type='mla'"
                )
            if target_kv_layers is not None:
                raise ValueError(
                    "target_kv_layers already replaces every layer's context latent; drop "
                    "context_kv_source_layers"
                )
            if index_topk is not None:
                raise ValueError(
                    "a layer reading another's context latent has no context of its own for "
                    "the drafter's DSA indexer to score; drop index_topk (target_index_layer "
                    "picks the context instead)"
                )
            if type(self).model_type not in DFLASH_BACKBONE_MODEL_TYPES or model_arch != "dflash":
                raise ValueError(
                    "context_kv_source_layers is wired through DFlashDraftModel.forward only "
                    f"(model_type in {DFLASH_BACKBONE_MODEL_TYPES}, model_arch 'dflash'), got "
                    f"{type(self).model_type!r} / {model_arch!r}"
                )
            context_kv_source_layers = list(context_kv_source_layers)
        self.context_kv_source_layers = context_kv_source_layers

        if target_attn_init_layers is not None:
            per_layer = isinstance(target_attn_init_layers, (list, tuple))
            if per_layer and len(target_attn_init_layers) != num_hidden_layers:
                raise ValueError(
                    f"target_attn_init_layers lists {len(target_attn_init_layers)} target layers "
                    f"for {num_hidden_layers} draft layers"
                )
            for layer in target_attn_init_layers if per_layer else [target_attn_init_layers]:
                if (
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or layer < 0
                    or (target_num_hidden_layers and layer >= target_num_hidden_layers)
                ):
                    raise ValueError(
                        "target_attn_init_layers entries must be target layers in "
                        f"[0, {target_num_hidden_layers}), got {layer!r}"
                    )
            if attention_type not in TARGET_MLA_ROPE_DIMS:
                raise ValueError(
                    "target_attn_init_layers copies the target's MLA; set "
                    f"attention_type={_TARGET_MLA_TYPES}"
                )
            # DFlashTrainer.init_model applies it to the DFlash backbone's layers
            # (DSparkTrainer defers to it); DFly's layers are not wired for it.
            if type(self).model_type not in DFLASH_BACKBONE_MODEL_TYPES:
                raise ValueError(
                    "target_attn_init_layers is applied for the DFlash backbone only (model_type "
                    f"in {DFLASH_BACKBONE_MODEL_TYPES}), got model_type {type(self).model_type!r}"
                )
        self.target_attn_init_layers = (
            list(target_attn_init_layers)
            if isinstance(target_attn_init_layers, (list, tuple))
            else target_attn_init_layers
        )

        if draft_mlp_type not in DRAFT_MLP_TYPES:
            raise ValueError(
                f"draft_mlp_type must be one of {DRAFT_MLP_TYPES}, got {draft_mlp_type!r}"
            )
        if draft_mlp_type == "moe":
            fields = {
                "num_experts": num_experts,
                "num_shared_experts": num_shared_experts,
                "num_experts_per_tok": num_experts_per_tok,
                "moe_intermediate_size": moe_intermediate_size,
            }
            for name, value in fields.items():
                minimum = 0 if name == "num_shared_experts" else 1
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
            if num_experts_per_tok > num_experts:
                raise ValueError(
                    f"num_experts_per_tok ({num_experts_per_tok}) exceeds "
                    f"num_experts ({num_experts})"
                )
            if not math.isfinite(router_scaling_factor) or router_scaling_factor <= 0:
                raise ValueError(
                    "router_scaling_factor must be finite and positive, got "
                    f"{router_scaling_factor!r}"
                )
            if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
                raise ValueError(f"swiglu_limit must be finite and >= 0, got {swiglu_limit!r}")
            if moe_aux_free_bias_update and not moe_router_enable_expert_bias:
                raise ValueError("moe_aux_free_bias_update needs moe_router_enable_expert_bias")
            if not (math.isfinite(moe_aux_free_bias_beta) and 0 <= moe_aux_free_bias_beta < 1):
                raise ValueError(
                    "moe_aux_free_bias_beta must be finite and in [0, 1), got "
                    f"{moe_aux_free_bias_beta!r}"
                )
            if not (math.isfinite(moe_aux_free_bias_gamma) and moe_aux_free_bias_gamma > 0):
                raise ValueError(
                    "moe_aux_free_bias_gamma must be finite and positive, got "
                    f"{moe_aux_free_bias_gamma!r}"
                )

        def check_mlp_init_layers(name, value):
            per_layer = isinstance(value, (list, tuple))
            if per_layer and len(value) != num_hidden_layers:
                raise ValueError(
                    f"{name} lists {len(value)} target layers for {num_hidden_layers} draft layers"
                )
            for layer in value if per_layer else [value]:
                if (
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or layer < 0
                    or (target_num_hidden_layers and layer >= target_num_hidden_layers)
                ):
                    raise ValueError(
                        f"{name} entries must be target layers in "
                        f"[0, {target_num_hidden_layers}), got {layer!r}"
                    )
            # DFlashTrainer.init_model applies it to the DFlash backbone's layers
            # (DSparkTrainer defers to it).
            if (
                type(self).model_type not in DFLASH_BACKBONE_MODEL_TYPES
                or model_arch not in ("dflash", "dspark")
                or layer_arch != "dflash"
            ):
                raise ValueError(
                    f"{name} is applied to the DFlash backbone's dflash layers (model_type in "
                    f"{DFLASH_BACKBONE_MODEL_TYPES}), got "
                    f"{type(self).model_type!r} / {model_arch!r} / {layer_arch!r}"
                )

        def check_expert_choice(name, value, count=None):
            if isinstance(value, str):
                if value not in ("bias", "random"):
                    raise ValueError(
                        f"{name} must be 'bias', 'random' or per-layer expert lists, got {value!r}"
                    )
                return
            if not (
                isinstance(value, (list, tuple))
                and len(value) == num_hidden_layers
                and all(isinstance(ids, (list, tuple)) and ids for ids in value)
                and len({len(ids) for ids in value}) == 1
                and (count is None or len(value[0]) == count)
                and all(
                    isinstance(e, int) and not isinstance(e, bool) and e >= 0
                    for ids in value
                    for e in ids
                )
                and all(len(set(ids)) == len(ids) for ids in value)
            ):
                needed = "the same number of" if count is None else f"{count}"
                raise ValueError(
                    f"{name} lists must give each of the {num_hidden_layers} draft layers "
                    f"{needed} distinct non-negative expert ids, got {value!r}"
                )

        if target_mlp_init_layers is not None:
            if draft_mlp_type != "moe":
                raise ValueError(
                    "target_mlp_init_layers copies a target MoE; set draft_mlp_type='moe'"
                )
            check_mlp_init_layers("target_mlp_init_layers", target_mlp_init_layers)
            if target_mlp_init_experts is not None:
                check_expert_choice(
                    "target_mlp_init_experts", target_mlp_init_experts, num_experts
                )
        elif target_mlp_init_experts is not None:
            raise ValueError("target_mlp_init_experts needs target_mlp_init_layers")
        dense_experts = target_mlp_dense_init_experts
        dense_scale = target_mlp_dense_init_routed_scale
        if target_mlp_dense_init_layers is not None:
            if draft_mlp_type != "dense":
                raise ValueError(
                    "target_mlp_dense_init_layers builds a dense FFN from target experts; "
                    "set draft_mlp_type='dense'"
                )
            check_mlp_init_layers("target_mlp_dense_init_layers", target_mlp_dense_init_layers)
            check_expert_choice("target_mlp_dense_init_experts", dense_experts)
            if dense_scale is not None and (
                isinstance(dense_scale, bool) or not math.isfinite(dense_scale) or dense_scale <= 0
            ):
                raise ValueError(
                    "target_mlp_dense_init_routed_scale must be finite and positive, got "
                    f"{dense_scale!r}"
                )
        if moe_skip_expert_init and target_mlp_init_layers is None:
            raise ValueError(
                "moe_skip_expert_init leaves expert tensors uninitialized; set "
                "target_mlp_init_layers so the trainer overwrites all of them"
            )
        self.draft_mlp_type = str(draft_mlp_type)
        self.target_mlp_init_layers = (
            list(target_mlp_init_layers)
            if isinstance(target_mlp_init_layers, (list, tuple))
            else target_mlp_init_layers
        )
        self.target_mlp_init_experts = (
            target_mlp_init_experts
            if target_mlp_init_experts is None or isinstance(target_mlp_init_experts, str)
            else [list(ids) for ids in target_mlp_init_experts]
        )
        self.target_mlp_dense_init_layers = (
            list(target_mlp_dense_init_layers)
            if isinstance(target_mlp_dense_init_layers, (list, tuple))
            else target_mlp_dense_init_layers
        )
        self.target_mlp_dense_init_experts = (
            dense_experts
            if isinstance(dense_experts, str)
            else [list(ids) for ids in dense_experts]
        )
        self.target_mlp_dense_init_routed_scale = (
            None if dense_scale is None else float(dense_scale)
        )
        self.num_experts = int(num_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.moe_router_use_sigmoid = bool(moe_router_use_sigmoid)
        self.moe_router_enable_expert_bias = bool(moe_router_enable_expert_bias)
        self.route_norm = bool(route_norm)
        self.router_scaling_factor = float(router_scaling_factor)
        self.swiglu_limit = float(swiglu_limit)
        self.moe_skip_expert_init = bool(moe_skip_expert_init)
        self.moe_aux_free_bias_update = bool(moe_aux_free_bias_update)
        self.moe_aux_free_bias_beta = float(moe_aux_free_bias_beta)
        self.moe_aux_free_bias_gamma = float(moe_aux_free_bias_gamma)

        if mtp_fusion is not None and mtp_fusion not in MTP_FUSION_MODES:
            raise ValueError(
                f"mtp_fusion must be one of {MTP_FUSION_MODES} or None, got {mtp_fusion!r}"
            )
        if mtp_fusion_init not in MTP_FUSION_INITS:
            raise ValueError(
                f"mtp_fusion_init must be one of {MTP_FUSION_INITS}, got {mtp_fusion_init!r}"
            )
        if mtp_fusion is not None:
            if target_hidden_size != hidden_size:
                raise ValueError(
                    "mtp_fusion fuses the target's final hidden into the block input, so "
                    f"target_hidden_size ({target_hidden_size}) must equal hidden_size "
                    f"({hidden_size})"
                )
            # DFlashTrainer.init_model initializes it (DSparkTrainer defers to it); DFly's
            # trainer does not.
            if type(self).model_type not in DFLASH_BACKBONE_MODEL_TYPES:
                raise ValueError(
                    "mtp_fusion is applied for the drafters DFlashTrainer builds (model_type in "
                    f"{DFLASH_BACKBONE_MODEL_TYPES}), got model_type {type(self).model_type!r}"
                )
        self.mtp_fusion = mtp_fusion
        self.mtp_fusion_init = mtp_fusion_init

        if attentive_correction:
            if not enable_hidden_correction:
                raise ValueError(
                    "attentive_correction adds to the hidden correction; set "
                    "enable_hidden_correction"
                )
            if target_kv_layers is None:
                raise ValueError(
                    "attentive_correction reads the target latents a draft layer reads; set "
                    "target_kv_layers"
                )
            if q_lora_rank is None:
                raise ValueError("attentive_correction projects its query through q_lora_rank")
            if (
                isinstance(attentive_correction_source_layer, bool)
                or not isinstance(attentive_correction_source_layer, int)
                or not -num_hidden_layers <= attentive_correction_source_layer < num_hidden_layers
            ):
                raise ValueError(
                    "attentive_correction_source_layer must be a draft layer index in "
                    f"[{-num_hidden_layers}, {num_hidden_layers}), got "
                    f"{attentive_correction_source_layer!r}"
                )
        if attentive_correction_block_keys and not attentive_correction:
            raise ValueError(
                "attentive_correction_block_keys extends the attentive correction's keys; set "
                "attentive_correction"
            )
        if attentive_correction_block_mode not in ("shared", "gated"):
            raise ValueError(
                "attentive_correction_block_mode must be 'shared' or 'gated', got "
                f"{attentive_correction_block_mode!r}"
            )
        if attentive_correction_block_mode != "shared" and not attentive_correction_block_keys:
            raise ValueError(
                "attentive_correction_block_mode places the block keys; set "
                "attentive_correction_block_keys"
            )
        if attentive_correction_block_kv_b not in ("shared", "separate"):
            raise ValueError(
                "attentive_correction_block_kv_b must be 'shared' or 'separate', got "
                f"{attentive_correction_block_kv_b!r}"
            )
        if attentive_correction_block_kv_b != "shared" and (
            not attentive_correction_block_keys or attentive_correction_block_mode != "gated"
        ):
            raise ValueError(
                "attentive_correction_block_kv_b='separate' is the gated block read's own kv_b; "
                "set attentive_correction_block_keys and attentive_correction_block_mode='gated'"
            )
        if attentive_correction_block_kv_in not in ("hidden_embed", "hidden"):
            raise ValueError(
                "attentive_correction_block_kv_in must be 'hidden_embed' or 'hidden', got "
                f"{attentive_correction_block_kv_in!r}"
            )
        if (
            attentive_correction_block_kv_in != "hidden_embed"
            and not attentive_correction_block_keys
        ):
            raise ValueError(
                "attentive_correction_block_kv_in is what the block keys are projected from; set "
                "attentive_correction_block_keys"
            )
        self.attentive_correction = bool(attentive_correction)
        self.attentive_correction_source_layer = int(attentive_correction_source_layer)
        self.attentive_correction_block_keys = bool(attentive_correction_block_keys)
        self.attentive_correction_block_mode = str(attentive_correction_block_mode)
        self.attentive_correction_block_kv_b = str(attentive_correction_block_kv_b)
        self.attentive_correction_block_kv_in = str(attentive_correction_block_kv_in)

        if layer_arch not in LAYER_ARCHES:
            raise ValueError(f"layer_arch must be one of {LAYER_ARCHES}, got {layer_arch!r}")
        self.layer_arch = layer_arch

        if attention_type not in draft_attention_types():
            raise ValueError(
                f"attention_type must be one of {draft_attention_types()}, got {attention_type!r}"
            )
        self.attention_type = attention_type
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.gated_mla = gated_mla
        if gating_type not in ("elementwise", "headwise"):
            raise ValueError(
                f"gating_type must be 'elementwise' or 'headwise', got {gating_type!r}"
            )
        self.gating_type = gating_type
        self.learnable_sink = learnable_sink
        self.learnable_sink_init = learnable_sink_init
        if attn_post_proj_norm and attention_type != "mla":
            raise ValueError("attn_post_proj_norm is hy4's MLA norm; set attention_type='mla'")
        self.attn_post_proj_norm = bool(attn_post_proj_norm)

        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        if dsa_backend not in DSA_BACKENDS:
            raise ValueError(f"dsa_backend must be one of {DSA_BACKENDS}, got {dsa_backend!r}")
        self.dsa_backend = dsa_backend
        self.dsa_fp8_indexer = dsa_fp8_indexer
        self.indexer_types = indexer_types

        # M-RoPE is switched on the way the Hy4 flash target and vLLM's uses_mrope
        # read it: an mrope_section inside rope_parameters. Without one the
        # drafter keeps plain 1D RoPE.
        mrope_section = rope_parameters.get("mrope_section")
        self.mrope_section = list(mrope_section) if mrope_section is not None else None
        self.mrope_interleaved = self.mrope_section is not None and bool(
            rope_parameters.get("mrope_interleaved", False)
        )
        self.position_provider = kwargs.get("position_provider")
        if (
            self.mrope_section is not None
            and self.position_provider is None
            and all(
                value is not None
                for value in (image_start_token_id, image_end_token_id, image_newline_token_id)
            )
        ):
            self.position_provider = "hy_v4_flash"
        self.position_recipe_version = kwargs.get("position_recipe_version", 1)
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_newline_token_id = image_newline_token_id
        if self.mrope_section is not None:
            self.rope_parameters = dict(rope_parameters)
            nested_theta = rope_parameters.get("rope_theta", self.rope_theta)
            if float(nested_theta) != self.rope_theta:
                raise ValueError("Flat rope_theta conflicts with MRoPE rope_parameters")
            self.rope_parameters["rope_theta"] = self.rope_theta
            validate_mrope_parameters(qk_rope_head_dim, self.rope_parameters)
            if attention_type != "mla":
                raise ValueError(
                    "M-RoPE is wired through hy4's MLA attention; set attention_type='mla'"
                )
            missing = [
                name
                for name in (
                    "image_start_token_id",
                    "image_end_token_id",
                    "image_newline_token_id",
                )
                if getattr(self, name) is None
            ]
            if missing and self.position_provider != "provided":
                raise ValueError(f"Hy4 MRoPE position preparation requires {', '.join(missing)}")

        # hy4 ships qk_head_dim explicitly; it is derived, so only cross-check it.
        qk_head_dim = kwargs.get("qk_head_dim")
        if qk_head_dim is not None and int(qk_head_dim) != qk_nope_head_dim + qk_rope_head_dim:
            raise ValueError(
                f"qk_head_dim={qk_head_dim} disagrees with qk_nope_head_dim + "
                f"qk_rope_head_dim = {qk_nope_head_dim + qk_rope_head_dim}"
            )

        if index_topk is not None:
            if attention_type != "mla":
                raise ValueError("index_topk (DSA) requires attention_type='mla'")
            if q_lora_rank is None:
                raise ValueError(
                    "index_topk (DSA) scores from the LoRA q residual; set q_lora_rank"
                )
        # Only an explicit "tilelang" is held to the kernels' constraints; "auto"
        # means "use them where they fit", and falls back to torch where they do
        # not (see hy4_mla.resolve_dsa_backend).
        if dsa_backend == "tilelang":
            # Same constraints the MTP head enforces: the sparse-MLA kernel runs
            # the absorbed form with d_v pinned to 512, and it is a *sparse*
            # kernel — it only ever reads a key list, from the drafter's own DSA
            # indexer or from the target indexer's candidates.
            if attention_type not in TARGET_MLA_ROPE_DIMS:
                raise ValueError(
                    f"dsa_backend='tilelang' requires attention_type={_TARGET_MLA_TYPES}"
                )
            if index_topk is None and target_index_layer is None:
                raise ValueError(
                    "dsa_backend='tilelang' reads keys from an index list; set index_topk "
                    "(the drafter's DSA indexer) or target_index_layer (the target's)"
                )
            if index_topk is not None and target_index_layer is not None:
                raise ValueError(
                    "with dsa_backend='tilelang', index_topk and target_index_layer would "
                    "both pick the context keys; set one"
                )
            rope = TARGET_MLA_ROPE_DIMS[attention_type]
            if kv_lora_rank != 512 or qk_rope_head_dim != rope:
                raise ValueError(
                    f"tilelang sparse MLA with attention_type={attention_type!r} requires "
                    f"kv_lora_rank=512 and qk_rope_head_dim={rope}"
                )
        if indexer_types is not None:
            unknown = {str(t) for t in indexer_types} - set(INDEXER_TYPES)
            if unknown:
                raise ValueError(
                    f"indexer_types entries must be in {INDEXER_TYPES}, got {unknown}"
                )
            if len(indexer_types) < self.num_hidden_layers:
                raise ValueError(
                    f"indexer_types has {len(indexer_types)} entries for "
                    f"{self.num_hidden_layers} layers"
                )
            schedule = [str(t) for t in indexer_types[: self.num_hidden_layers]]
            # "shared" borrows the most recent "full" layer's indexer, so the
            # schedule has to open with one.
            if schedule[0] == "shared":
                raise ValueError(
                    "indexer_types[0] is 'shared', but no earlier layer owns an indexer to borrow"
                )
            if "shared" in schedule and (attention_type != "mla" or index_topk is None):
                raise ValueError(
                    "indexer_types asks layers to share a DSA indexer, but this config "
                    "builds none: sharing needs attention_type='mla' and index_topk"
                )


class DFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DFlashRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # +20 buffer avoids cache rebuild if sequences slightly exceed the configured limit.
        self._set_cos_sin_cache(max_position_embeddings + 20, self.inv_freq.device, torch.float32)

    def _apply(self, fn, recurse=True):
        # A model-wide .to(bfloat16) must not round the frequencies: a longer
        # sequence rebuilds the table from them, and bf16 positions and angles are
        # wrong past a few hundred tokens.
        inv_freq = self.inv_freq
        super()._apply(fn, recurse)
        if not inv_freq.is_meta:
            self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# NOTE: Not called in DFlashAttention.forward() (RoPE is applied inline there),
# but kept as a standard utility.
def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class DFlashAttention(nn.Module):
    """Dual-source KV attention for DFlash.

    K/V come from two sources concatenated along the sequence dimension:
      1. Context KV: projected from target model's context features (via shared W_k/W_v)
      2. Draft KV: projected from draft model's own hidden states (via same W_k/W_v)

    Q comes only from the draft model's hidden states.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Q-norm and K-norm (Qwen3 architecture requirement)
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = DFlashRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
        )
        # Lazy USP group (resolved on first USP forward after init_usp_groups).
        self._usp_group = None
        self._usp_size = 1
        self._usp_rank = 0

    def _ensure_usp(self) -> None:
        if self._usp_group is None:
            self._usp_group = get_sp_ulysses_group()
            if self._usp_group is not None:
                self._usp_size = dist.get_world_size(self._usp_group)
                self._usp_rank = dist.get_rank(self._usp_group)
            else:
                self._usp_size = 1
                self._usp_rank = 0

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        full_ctx_len: Optional[int] = None,
        sparse_kv_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with dual-source KV.

        Args:
            draft_hidden: [B, draft_len, D] — hidden states of draft tokens
            context_hidden: [B, ctx_len, D] — context features from target
                (local seq shard under USP; ``full_ctx_len`` is the global length)
            draft_position_ids: [B, draft_len] — position IDs for draft tokens
            context_position_ids: [B, ctx_len] — position IDs for context tokens
            block_mask: FlexAttention BlockMask for block-causal attention
            full_ctx_len: global context length under USP (None => non-USP)
            sparse_kv_indices: must be None; only hy4's MLA has a kernel that reads keys
                from index lists.
        """
        if sparse_kv_indices is not None:
            raise ValueError(
                "GQA attends through flex and its block_mask; sparse_kv_indices is for "
                "hy4 MLA on the tilelang backend"
            )
        bsz, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]
        usp = full_ctx_len is not None and full_ctx_len != ctx_len
        if usp:
            self._ensure_usp()
            usp = self._usp_size > 1
            if usp and full_ctx_len != ctx_len * self._usp_size:
                raise ValueError(
                    f"USP context shard mismatch: local={ctx_len}, "
                    f"full_ctx_len={full_ctx_len}, sp={self._usp_size}"
                )

        # Q only from draft
        q = self.q_proj(draft_hidden)
        q = q.view(bsz, draft_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, num_heads, draft_len, head_dim]

        # K/V from both context and draft (shared projections)
        k_ctx = self.k_proj(context_hidden)
        v_ctx = self.v_proj(context_hidden)
        k_draft = self.k_proj(draft_hidden)
        v_draft = self.v_proj(draft_hidden)

        if usp:
            # Context is seq-sharded for memory; draft (anchor-strided) differs per
            # rank, so we cannot Ulysses-head-shard draft Q. Instead: RoPE on the
            # local context shard, all-gather context K/V along seq, then the same
            # GQA flex path as non-USP. Peak attention KV matches a full forward,
            # but the multi-layer target hidden states stay S/sp resident.
            k_ctx = self.k_norm(k_ctx.view(bsz, ctx_len, self.num_kv_heads, self.head_dim))
            k_ctx = k_ctx.transpose(1, 2)
            v_ctx = v_ctx.view(bsz, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k_draft = self.k_norm(
                k_draft.view(bsz, draft_len, self.num_kv_heads, self.head_dim)
            ).transpose(1, 2)
            v_draft = v_draft.view(bsz, draft_len, self.num_kv_heads, self.head_dim).transpose(
                1, 2
            )

            rope_len = max(
                int(full_ctx_len),
                int(draft_position_ids.max().item()) + 1 if draft_position_ids.numel() else 1,
                int(context_position_ids.max().item()) + 1 if context_position_ids.numel() else 1,
            )
            cos, sin = self.rotary_emb(q, seq_len=rope_len)
            cos = cos.to(q.device)
            sin = sin.to(q.device)
            cos_s = cos.squeeze(1).squeeze(0)
            sin_s = sin.squeeze(1).squeeze(0)
            q = (q * cos_s[draft_position_ids].unsqueeze(1)) + (
                _rotate_half(q) * sin_s[draft_position_ids].unsqueeze(1)
            )
            k_ctx = (k_ctx * cos_s[context_position_ids].unsqueeze(1)) + (
                _rotate_half(k_ctx) * sin_s[context_position_ids].unsqueeze(1)
            )
            k_draft = (k_draft * cos_s[draft_position_ids].unsqueeze(1)) + (
                _rotate_half(k_draft) * sin_s[draft_position_ids].unsqueeze(1)
            )

            k_local_len = k_ctx.shape[2]
            k_ctx = all_gather_seq(k_ctx, self._usp_group)
            v_ctx = all_gather_seq(v_ctx, self._usp_group)
            global _USP_ATTN_LOGGED
            if not _USP_ATTN_LOGGED and dist.get_rank() == 0:
                _USP_ATTN_LOGGED = True
                logger.info(
                    f"[USP] attention gather: sp_rank={self._usp_rank}/{self._usp_size} "
                    f"ctx_kv local_seq={k_local_len} -> gathered_seq={k_ctx.shape[2]} "
                    f"draft_kv_local_seq={k_draft.shape[2]} "
                    f"q_heads={q.shape[1]} (draft Q stays local; context seq-sharded)"
                )

            k = torch.cat([k_ctx, k_draft], dim=2)
            v = torch.cat([v_ctx, v_draft], dim=2)

            if block_mask is not None and q.is_cuda:
                from angelspec.models.ops.flex_attention import (
                    compile_friendly_flex_attention,
                )

                attn_output = compile_friendly_flex_attention(
                    query=q,
                    key=k,
                    value=v,
                    block_mask=block_mask,
                    enable_gqa=True,
                )
            else:
                k = _repeat_kv(k, self.num_kv_groups)
                v = _repeat_kv(v, self.num_kv_groups)
                attn_output = block_mask_sdpa(q, k, v, block_mask, is_causal=False, dropout_p=0.0)
        else:
            # Concatenate K and V along sequence dimension BEFORE normalization
            # Concat → K-norm → RoPE
            k = torch.cat([k_ctx, k_draft], dim=1)  # [B, ctx+draft, kv_dim]
            v = torch.cat([v_ctx, v_draft], dim=1)

            total_len = ctx_len + draft_len
            k = k.view(bsz, total_len, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k).transpose(1, 2)  # [B, num_kv_heads, ctx+draft, head_dim]
            v = v.view(bsz, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # RoPE with concatenated position IDs (context + draft)
            full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
            cos, sin = self.rotary_emb(q, seq_len=total_len)
            cos = cos.to(q.device)
            sin = sin.to(q.device)

            # Apply RoPE to Q (using draft positions only — last draft_len of full_position_ids)
            cos_q = cos.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
            sin_q = sin.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
            q = (q * cos_q) + (_rotate_half(q) * sin_q)

            # Apply RoPE to K (using full concatenated positions)
            cos_k = cos.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(1)
            sin_k = sin.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(1)
            k = (k * cos_k) + (_rotate_half(k) * sin_k)

            if block_mask is not None and q.is_cuda:
                from angelspec.models.ops.flex_attention import (
                    compile_friendly_flex_attention,
                )

                # Use enable_gqa=True to let FlexAttention handle GQA internally
                # instead of materializing expanded KV via _repeat_kv
                attn_output = compile_friendly_flex_attention(
                    query=q,
                    key=k,
                    value=v,
                    block_mask=block_mask,
                    enable_gqa=True,
                )
            else:
                k = _repeat_kv(k, self.num_kv_groups)
                v = _repeat_kv(v, self.num_kv_groups)
                attn_output = block_mask_sdpa(q, k, v, block_mask, is_causal=False, dropout_p=0.0)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, draft_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def build_gqa_attention(config: PretrainedConfig, layer_idx: int) -> nn.Module:
    return DFlashAttention(config)


def build_dflash_self_attn(config: PretrainedConfig, layer_idx: int) -> nn.Module:
    """Attention block for one dense decoder layer, per ``config.attention_type``."""
    return build_draft_attention(config, layer_idx)


class DFlashDecoderLayer(nn.Module):
    """Single transformer decoder layer for DFlash draft model.

    ``config.attention_type`` picks the attention block from ``attention_registry`` —
    DFlash's GQA (default) or a target family's MLA — so the dense body is not tied to
    GQA. ``self_attn`` can be passed in instead,
    which is how the drafter hands out DSA indexers shared across layers (hy4's
    ``indexer_types``).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int = 0,
        self_attn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        if self_attn is None:
            self_attn = build_dflash_self_attn(config, self.layer_idx)
        self.self_attn = self_attn
        if getattr(config, "draft_mlp_type", "dense") == "moe":
            from angelspec.models.draft.mtp import Hy3MoE

            self.mlp = Hy3MoE(config)
            if self.mlp.aux_free_bias_update:
                # The drafter is cast to bf16, too coarse for the controller's steps; they
                # go through the optimizer's fp32 master (BF16Optimizer.add_to_parameter),
                # which only holds trainable parameters. It never receives a gradient.
                self.mlp.expert_bias.requires_grad_(True)
        else:
            self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        full_ctx_len: Optional[int] = None,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        context_latent: Optional[torch.Tensor] = None,
        return_context_latent: bool = False,
    ):
        """``return_context_latent`` (hy4's MLA only) also returns the context latent the
        attention projected, for the layers ``context_kv_source_layers`` points here."""
        # Only the TARGET_MLA_ROPE_DIMS attention blocks take context_latent.
        latent_kwargs = {} if context_latent is None else {"context_latent": context_latent}
        if return_context_latent:
            latent_kwargs["return_context_latent"] = True
        attention_kwargs = dict(
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            full_ctx_len=full_ctx_len,
            sparse_kv_indices=sparse_kv_indices,
            **latent_kwargs,
        )
        residual = draft_hidden
        draft_hidden = self.input_layernorm(draft_hidden)
        draft_hidden = self.self_attn(draft_hidden=draft_hidden, **attention_kwargs)
        if return_context_latent:
            draft_hidden, projected_latent = draft_hidden
        draft_hidden = residual + draft_hidden

        residual = draft_hidden
        draft_hidden = self.post_attention_layernorm(draft_hidden)
        draft_hidden = self.mlp(draft_hidden)
        draft_hidden = residual + draft_hidden

        return (draft_hidden, projected_latent) if return_context_latent else draft_hidden


def build_target_layer_ids(num_target_layers: int, num_hidden_layers: int) -> List[int]:
    """Compute uniformly spaced layer IDs from the target model.

    Uses evenly-spaced layer IDs:
      start = 1, end = num_hidden_layers - 3, span = end - start
      For num_target_layers=5 and num_hidden_layers=36:
        start=1, end=33, span=32
        → [1, 9, 17, 25, 33]

    Note: num_target_layers here is the number of layers
    to capture. num_hidden_layers is the
    total number of target model decoder layers.
    """
    if num_target_layers == 1:
        return [num_hidden_layers // 2]
    start = 1
    end = num_hidden_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_target_layers - 1))) for i in range(num_target_layers)
    ]


def target_layers_per_draft_layer(
    value: Optional[Union[int, List[Optional[int]]]], num_draft_layers: int
) -> Optional[List[Optional[int]]]:
    """``target_index_layer`` / ``target_kv_layers`` as one target layer per draft layer:
    a list as given (``None`` for a draft layer without one), an int repeated, ``None``
    when unset."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [None if layer is None else int(layer) for layer in value]
    return [int(value)] * num_draft_layers


def windows_per_draft_layer(
    value: Optional[Union[int, List[Optional[int]]]], num_draft_layers: int
) -> List[Optional[int]]:
    """``context_local_window`` as one window per draft layer: a list as given, else the
    int or ``None`` repeated."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * num_draft_layers


class HiddenStatesCorrection(nn.Module):
    """TreeFlash hidden-states correction (formula (1)), residual + zero-init::

        h'_{t+i} = h_{t+i} + SwiGLU( norm(h_{t+i}) :: norm(e_{t+i-1}) )

    ``e_{t+i-1}`` is the previous (teacher-forced) token embedding, ``::`` is
    feature-dim concat. The output projection is zero-initialized, so the
    correction starts at 0 and the model degenerates back to plain DFlash.

    With an ``attention`` (:class:`~angelspec.models.draft.attentive_correction.
    AttentiveCorrection`, ``attentive_correction`` in the config) the correction also
    reads a draft layer's context again from the same two normalized inputs, when the
    caller passes that context.
    """

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        attention: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embed_size = int(embed_size)
        self.intermediate_size = int(intermediate_size)
        self.attention = attention

        # Normalize each stream independently before concat so the hidden state
        # and the differently-scaled token embedding contribute comparably.
        self.hidden_norm = DFlashRMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.embed_norm = DFlashRMSNorm(self.embed_size, eps=rms_norm_eps)

        in_features = self.hidden_size + self.embed_size
        self.gate_proj = nn.Linear(in_features, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Residual zero-init: correction starts at 0 -> exactly recovers DFlash.
        nn.init.zeros_(self.down_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embeds: torch.Tensor,
        context: Optional[dict] = None,
        attention_stats: Optional[dict] = None,
    ) -> torch.Tensor:
        """Apply the residual correction.

        Args:
            hidden_states: ``[B, ..., hidden_size]`` drafter output hidden states.
            prev_token_embeds: ``[B, ..., embed_size]`` previous-token embeddings
                (same leading shape as ``hidden_states``).
            context: what ``attention`` reads for these slots (its ``position_ids``,
                ``latent`` and ``indices`` arguments); ``None`` leaves it out.
            attention_stats: filled, when given and the attention ran, with the squared
                norms of its output and of ``hidden_states`` over the slots it read for.

        Returns:
            Corrected hidden states, same shape/dtype as ``hidden_states``.
        """
        h_norm = self.hidden_norm(hidden_states)
        e_norm = self.embed_norm(prev_token_embeds.to(hidden_states.dtype))
        gate_in = torch.cat([h_norm, e_norm], dim=-1)
        delta = self.down_proj(F.silu(self.gate_proj(gate_in)) * self.up_proj(gate_in))
        delta = delta.to(hidden_states.dtype)
        if self.attention is not None and context is not None:
            read, alive = self.attention(h_norm, e_norm, **context, stats=attention_stats)
            delta = delta + read.to(delta.dtype)
            if attention_stats is not None:
                with torch.no_grad():
                    live = alive.unsqueeze(-1).float()
                    attention_stats["attention"] = (read.float().pow(2) * live).sum()
                    attention_stats["hidden"] = (hidden_states.float().pow(2) * live).sum()

        return hidden_states + delta


def build_hidden_correction(config) -> Optional[nn.Module]:
    """Build the hidden-states correction module, or ``None`` if disabled.

    The previous-token embedding dim equals the draft ``hidden_size`` (the
    drafter reuses the target token embedding). The SwiGLU intermediate width
    defaults to ``hidden_size`` unless ``hidden_correction_intermediate_size``
    is set.
    """
    if not bool(getattr(config, "enable_hidden_correction", False)):
        return None

    hidden_size = int(config.hidden_size)
    intermediate = getattr(config, "hidden_correction_intermediate_size", None)
    intermediate = int(intermediate) if intermediate else hidden_size
    attention = None
    if bool(getattr(config, "attentive_correction", False)):
        from angelspec.models.draft.attentive_correction import AttentiveCorrection

        attention = AttentiveCorrection(config)
    return HiddenStatesCorrection(
        hidden_size=hidden_size,
        embed_size=hidden_size,
        intermediate_size=intermediate,
        rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        attention=attention,
    )


class MTPFusion(nn.Module):
    """The opening of the target's MTP layer, added to the drafter's block input::

        f = eh_proj( enorm(E(x_a)) :: hnorm(h_{a-1}) )

    hy4's MTP layer fuses the target's final hidden at a position with the embedding of the
    token after it, then runs a decoder layer on ``f`` to predict the token after that. At
    the last accepted token ``a - 1`` and the anchor ``x_a``, that prediction is the
    drafter's first (``x_{a+1}``, block slot 1), and ``h_{a-1}`` is the one target state no
    other drafter input carries: the target's last-layer output, which feeds only its LM
    head and MTP layer. ``h`` is taken after the target's final norm, as vLLM hands it to
    the MTP layer; ``::`` is feature-dim concat. Names and shapes match the target's
    ``mtp_layers.0`` (see ``models.target.target_mtp_fusion``).

    ``f`` reaches the block through per-channel gates, zero-initialized so that adding the
    module leaves a drafter's output unchanged until they open. ``mode`` picks where:
    ``"slot0"`` adds it to the anchor slot (where ``E(x_a)`` sits), ``"slot1"`` to the first
    predicted slot (whose token the MTP layer predicts from ``f``), each through ``gate``;
    ``"broadcast"`` adds it to every slot, through ``gate`` on the anchor slot and
    ``mask_gate`` shared by the mask slots. None of the three depends on the block size.
    """

    def __init__(self, hidden_size: int, mode: str, rms_norm_eps: float = 1e-6):
        super().__init__()
        if mode not in MTP_FUSION_MODES:
            raise ValueError(f"mode must be one of {MTP_FUSION_MODES}, got {mode!r}")
        self.mode = mode
        # The block slot ``gate`` acts on.
        self.gate_slot = 1 if mode == "slot1" else 0
        self.enorm = DFlashRMSNorm(hidden_size, eps=rms_norm_eps)
        self.hnorm = DFlashRMSNorm(hidden_size, eps=rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        # 1-D like norm weights, so BF16Optimizer keeps them on AdamW: Muon would
        # orthogonalize a [slots, D] gate table as if it were a linear map.
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.mask_gate = nn.Parameter(torch.zeros(hidden_size)) if mode == "broadcast" else None

    def fuse(self, anchor_embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """``f`` from ``[..., D]`` anchor embeddings and the target's (normed) final hiddens."""
        return self.eh_proj(
            torch.cat(
                [self.enorm(anchor_embeds), self.hnorm(prev_hidden.to(anchor_embeds.dtype))],
                dim=-1,
            )
        )

    def slot_gates(self, block_size: int) -> torch.Tensor:
        """``[block_size, D]``: each block slot's gate, 0 where nothing is added."""
        if self.mask_gate is not None:
            return torch.cat([self.gate[None], self.mask_gate[None].expand(block_size - 1, -1)])
        gates = [torch.zeros_like(self.gate)] * block_size
        gates[self.gate_slot] = self.gate
        return torch.stack(gates)

    def forward(
        self, anchor_embeds: torch.Tensor, prev_hidden: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """``[..., block_size, D]``: what to add to each block's input."""
        fused = self.fuse(anchor_embeds, prev_hidden)
        return self.slot_gates(block_size).to(fused.dtype) * fused.unsqueeze(-2)


def build_mtp_fusion(config) -> Optional[nn.Module]:
    """Build the MTP fusion (``mtp_fusion``), or ``None`` if disabled."""
    mode = getattr(config, "mtp_fusion", None)
    if mode is None:
        return None
    return MTPFusion(
        hidden_size=int(config.hidden_size),
        mode=mode,
        rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
    )


def build_draft_layers(config, num_layers: int) -> Tuple[nn.ModuleList, Optional[nn.ModuleList]]:
    """Decoder layers for ``config.layer_arch``, plus any shared DSA indexers.

    Returns ``(layers, indexers)``. ``indexers`` is ``None`` unless a DSA schedule
    shares them: those are the indexers hy4's ``indexer_types`` schedule shares
    between layers, for the *drafter* to own rather than the layer that opens
    them — ``DFlashTrainer`` shards each decoder layer as its own FSDP unit, so
    an indexer living in layer 0 is resharded before a borrowing layer can read
    it. Owned at the root it belongs to the root FSDP unit, which stays
    unsharded for the whole step. ``None`` when no layer shares (each owns its
    own, in place). See :func:`~.hy4_mla.build_hy4_self_attns`.

    Every body keeps the dual-source KV contract (draft Q, ``context ++ draft``
    K/V), so the context wiring above it is free to be DFlash's, DFlare's or
    DFly's.
    """
    # Every layer builds its own attention block in place, which is what keeps a
    # GQA config's init unchanged: the MLA path below goes through
    # build_hy4_self_attns to get indexer sharing, and that constructs all the
    # attentions up front, drawing from the RNG in a different order.
    if str(getattr(config, "attention_type", "gqa")) != "mla":
        layers = nn.ModuleList([DFlashDecoderLayer(config, i) for i in range(num_layers)])
        return layers, None

    from angelspec.models.draft.hy4_mla import build_hy4_self_attns

    self_attns, shared_indexers = build_hy4_self_attns(config, num_layers)
    layers = nn.ModuleList(
        [DFlashDecoderLayer(config, i, self_attn=self_attns[i]) for i in range(num_layers)]
    )
    return layers, shared_indexers


class DFlashDraftModel(PreTrainedModel):
    """DFlash draft model with dual-source KV injection.

    Trainable parameters:
      - W_proj: Linear(num_target_layers * target_hidden_size, hidden_size)
      - proj_norm: RMSNorm after projection (neither built with ``target_kv_layers``)
      - N decoder layers (each with attention + FFN), per ``layer_arch``
      - hidden_correction: TreeFlash correction head, only with
        ``enable_hidden_correction`` (applied before the LM head by the
        ``DFlashModel`` training wrapper)
      - path_selector: predecessor-conditioned top-k reranker, only with
        ``enable_path_selector`` (also driven by the training wrapper, which
        supplies the unary logits it reranks)
      - mtp_fusion: the target MTP layer's input fusion, only with ``mtp_fusion``
        (added to the block input by the training wrapper, which holds the anchors
        and the target's final hidden states)

    Frozen (from target):
      - embed_tokens: token embedding
      - LM head is external (loaded separately in trainer)
    """

    config_class = DFlashConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.layer_arch = str(getattr(config, "layer_arch", "dflash"))

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        num_target_layers = getattr(config, "num_target_layers", 5)
        self.num_target_layers = num_target_layers
        self.mask_token_id = getattr(config, "mask_token_id", 151669)

        # Target layer IDs for hidden state extraction
        target_num_hidden = getattr(config, "target_num_hidden_layers", 36)
        self.target_layer_ids = getattr(config, "target_layer_ids", None)
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(num_target_layers, target_num_hidden)

        # Context feature projection: concat(multi-layer hidden) → hidden_size. A drafter
        # reading the target's own latents (target_kv_layers) builds none: it would get
        # no gradient, which DDP-style replication rejects.
        if getattr(config, "target_kv_layers", None) is None:
            proj_input_dim = num_target_layers * target_hidden_size
            self.context_proj = nn.Linear(proj_input_dim, self.hidden_size, bias=False)
            self.context_norm = DFlashRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Token embedding (shared from target, frozen)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers, self.indexers = self._build_layers(config)
        self._context_kv_sources = getattr(config, "context_kv_source_layers", None)
        # Layers whose context latent later layers read; they hand it on as they project it.
        self._context_kv_lenders = {
            source
            for index, source in enumerate(self._context_kv_sources or [])
            if source != index
        }

        self.final_norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # TreeFlash correction (formula (1)); ``None`` unless enabled. Applied by
        # the training wrapper right before the LM head, not inside forward().
        self.hidden_correction = build_hidden_correction(config)

        # Path selector; ``None`` unless enabled. Like the correction above it is
        # driven by the training wrapper (it needs the unary logits), not forward().
        self.path_selector = build_path_selector(config)

        # MTP fusion; ``None`` unless enabled. The training wrapper adds it to the block
        # input (see DFlashModel._draft_backbone), not forward().
        self.mtp_fusion = build_mtp_fusion(config)

    def _build_layers(self, config) -> Tuple[nn.ModuleList, Optional[nn.ModuleList]]:
        """Decoder layers + shared DSA indexers. Subclass seam.

        DFlare overrides it to build its own layers; DFly comes back to this one.
        Called once from ``__init__``, so a subclass swaps the layer stack instead
        of building and discarding one.
        """
        return build_draft_layers(config, self.num_layers)

    def extract_context_feature(self, all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Extract and project context features from target hidden states.

        Args:
            all_hidden_states: list of [B, seq_len, D] tensors from target layers

        Returns:
            context_feature: [B, seq_len, hidden_size]
        """
        concatenated = torch.cat(all_hidden_states, dim=-1).to(self.context_proj.weight.dtype)
        projected = self.context_proj(concatenated)
        return self.context_norm(projected)

    def forward(
        self,
        draft_input_ids: Optional[torch.Tensor],
        context_feature: Optional[torch.Tensor],
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
        full_ctx_len: Optional[int] = None,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        context_latents: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through draft model.

        Args:
            draft_input_ids: [B, draft_len] — token IDs (anchor + MASK tokens).
                Ignored if noise_embedding is provided.
            context_feature: [B, ctx_len, D] — projected context from target
                (local seq shard under USP when ``full_ctx_len`` is set); ``None``
                with ``context_latents``
            draft_position_ids: [B, draft_len]
            context_position_ids: [B, ctx_len]
            block_mask: FlexAttention BlockMask, or a list with one per layer
            noise_embedding: [B, draft_len, D] — pre-computed embeddings (from training wrapper)
            full_ctx_len: global context length for USP Ulysses attention
            sparse_kv_indices: [B, draft_len, K] int32 — the keys each draft token may
                read, as indices into ``[context | draft]`` (``-1`` unused), or a list
                with one per layer. Replaces ``block_mask`` (then None) when every
                layer runs the TileLang sparse kernel; see
                ``_create_dflash_sparse_kv_indices``.
            context_latents: one ``[B, ctx_len, kv_lora_rank + qk_rope_head_dim]``
                context latent per layer, in place of the ones each layer projects
                from ``context_feature`` (``target_kv_layers``)

        Returns:
            hidden_states: [B, draft_len, D] — pre-norm hidden states
        """
        if noise_embedding is not None:
            draft_hidden = noise_embedding
        else:
            draft_hidden = self.embed_tokens(draft_input_ids)
        if context_feature is not None:
            draft_hidden = draft_hidden.to(context_feature.dtype)

        lent_latents = {}
        for index, layer in enumerate(self.layers):
            layer_kwargs = {}
            if context_latents is not None:
                layer_kwargs["context_latent"] = context_latents[index]
            elif index in self._context_kv_lenders:
                layer_kwargs["return_context_latent"] = True
            elif self._context_kv_sources is not None and self._context_kv_sources[index] != index:
                layer_kwargs["context_latent"] = lent_latents[self._context_kv_sources[index]]
            out = layer(
                draft_hidden=draft_hidden,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask[index] if isinstance(block_mask, list) else block_mask,
                full_ctx_len=full_ctx_len,
                sparse_kv_indices=(
                    sparse_kv_indices[index]
                    if isinstance(sparse_kv_indices, list)
                    else sparse_kv_indices
                ),
                **layer_kwargs,
            )
            if index in self._context_kv_lenders:
                draft_hidden, lent_latents[index] = out
            else:
                draft_hidden = out

        if getattr(self.config, "confidence_on_prenorm_hidden", False):
            self.confidence_hidden = draft_hidden
        return self.final_norm(draft_hidden)

    def freeze_embedding(self) -> None:
        self.embed_tokens.weight.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """Load embedding weights from target model checkpoint."""
        if os.path.exists(model_path):
            glob_path = os.path.join(model_path, "*.index.json")
            import glob as glob_mod

            index_json_path = glob_mod.glob(glob_path)

            if len(index_json_path) == 0:
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    with safe_open(safetensors_path, framework="pt") as f:
                        self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
                    return
                pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    state_dict = torch.load(
                        pytorch_model_path, map_location="cpu", weights_only=True
                    )
                    self.embed_tokens.weight.copy_(state_dict[embedding_key])
                    return
                raise FileNotFoundError(
                    f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
                )
            index_json_path = index_json_path[0]
            with open(index_json_path, "r") as f:
                index_json = json.load(f)
            ckpt_file = index_json["weight_map"][embedding_key]
            if ckpt_file.endswith(".safetensors"):
                with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
                    self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
            else:
                state_dict = torch.load(
                    os.path.join(model_path, ckpt_file), map_location="cpu", weights_only=True
                )
                self.embed_tokens.weight.copy_(state_dict[embedding_key])
        else:
            local_cache_path = snapshot_download(repo_id=model_path)
            self.load_embedding(local_cache_path, embedding_key)
