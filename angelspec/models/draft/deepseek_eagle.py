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

"""DeepSeek MLA (Multi-head Latent Attention) Eagle3 draft model for training."""

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from angelspec.models.draft.base import Eagle3DraftModel

# TODO: Extract shared components into a common module to reduce duplication:
# - LlamaMLP, LlamaRMSNorm, RoPE classes → angelspec/models/draft/modules.py
# - _init_rope() is near-identical to LlamaAttention._init_rope (~60 lines)
# - DecoderLayer.forward() is line-for-line identical to LlamaDecoderLayer.forward()
# - embed_input_ids/project_hidden_states/compute_logits/backbone are identical
#   to LlamaForCausalLMEagle3 and could live in Eagle3DraftModel base class
# - Suffix attention loop could be batched (einsum instead of Python loop) for
#   both this file and llama3_eagle.py
from angelspec.models.draft.llama3_eagle import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaYarnRotaryEmbedding,
    yarn_get_mscale,
)
from angelspec.models.ops.flex_attention import (
    compile_friendly_flex_attention,
    eagle3_block_mask,
)
from angelspec.utils.logging import logger, print_with_rank


def _rope_config_get(rope_scaling, key, default=None):
    """Get a value from rope_scaling config (dict or object)."""
    if rope_scaling is None:
        return default
    if isinstance(rope_scaling, dict):
        return rope_scaling.get(key, default)
    return getattr(rope_scaling, key, default)


# ── Interleaved RoPE (DeepSeek convention) ────────────────────────────────
#
# DeepSeek MLA uses interleaved-pair rotation where consecutive dimension
# pairs (0,1), (2,3), ... are rotated together.  This differs from the
# neox-style rotation used in Llama/HF-transformers (first-half vs second-half).
#
# We reuse the existing LlamaRotaryEmbedding classes (which produce neox-layout
# cos/sin caches) and convert neox→interleaved on the fly in the apply function.
# Conversion: neox [θ0,..,θ31, θ0,..,θ31] → interleaved [θ0,θ0, θ1,θ1, ..]
# is just cos[..., :half].repeat_interleave(2, dim=-1).


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotation: pairs dims (0,1), (2,3), ..."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


@torch.compile(dynamic=True)
def _apply_rotary_pos_emb_interleaved(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply RoPE with interleaved rotation, accepting neox-layout cos/sin."""
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    half = cos.shape[-1] // 2
    cos = cos[..., :half].repeat_interleave(2, dim=-1)
    sin = sin[..., :half].repeat_interleave(2, dim=-1)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


class DeepSeekMLAAttention(nn.Module):
    """MLA attention (SDPA backend) for Eagle3 draft model training.

    Implements the MLA forward path from DeepSeek-V2/V3:
      Q: down_proj -> layernorm -> up_proj (optional LoRA compression)
      KV: down_proj -> split(compressed, k_rope) -> layernorm -> up_proj -> split(k_nope, value)
      RoPE applied only to qk_rope_head_dim dimensions.

    Supports both cached (EAGLE3 suffix pattern) and non-cached attention paths.
    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings

        # Eagle3: attention input is cat(input_emb, hidden_states) = hidden_size * 2
        input_dim = self.hidden_size * 2

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(input_dim, self.q_lora_rank, bias=False)
            self.q_a_layernorm = LlamaRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(input_dim, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            input_dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = LlamaRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection: v_head_dim per head (NOT qk_head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        # RoPE on qk_rope_head_dim only
        self._init_rope()

        # Softmax scale with optional YaRN mscale
        self.softmax_scale = self._compute_softmax_scale()

    def _init_rope(self):
        """Initialize rotary embeddings with qk_rope_head_dim as the dimension."""
        rope_dim = self.qk_rope_head_dim
        rope_scaling = self.config.rope_scaling
        rope_theta = getattr(self.config, "rope_theta", 10000)
        rget = partial(_rope_config_get, rope_scaling)
        scaling_type = rget("rope_type", rget("type"))

        if rope_scaling is None or scaling_type == "default":
            self.rotary_emb = LlamaRotaryEmbedding(
                rope_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=rope_theta,
            )
        else:
            scaling_factor = rget("factor")

            if scaling_type in (None, "default"):
                self.rotary_emb = LlamaRotaryEmbedding(
                    rope_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=rope_theta,
                )
            elif scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    rope_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    rope_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "llama3":
                self.rotary_emb = LlamaRotaryEmbedding(
                    rope_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=rope_theta,
                    scaling_factor=scaling_factor if scaling_factor is not None else 1.0,
                    low_freq_factor=rget("low_freq_factor"),
                    high_freq_factor=rget("high_freq_factor"),
                    orig_max_position=rget("original_max_position_embeddings"),
                )
            elif scaling_type == "yarn":
                self.rotary_emb = LlamaYarnRotaryEmbedding(
                    rope_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    original_max_position_embeddings=rget("original_max_position_embeddings"),
                    scaling_factor=scaling_factor,
                    beta_fast=rget("beta_fast"),
                    beta_slow=rget("beta_slow"),
                    mscale=rget("mscale"),
                    mscale_all_dim=rget("mscale_all_dim"),
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _compute_softmax_scale(self) -> float:
        """Compute softmax scale, incorporating YaRN mscale if applicable."""
        rope_scaling = self.config.rope_scaling
        if rope_scaling is not None:
            rget = partial(_rope_config_get, rope_scaling)
            scaling_type = rget("rope_type", rget("type"))
            if scaling_type == "yarn":
                factor = rget("factor", 1.0)
                mscale_all_dim = rget("mscale_all_dim", 0)
                mscale = yarn_get_mscale(factor, mscale_all_dim)
                return (mscale * mscale) / math.sqrt(self.qk_head_dim)
        return 1.0 / math.sqrt(self.qk_head_dim)

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden_states to Q, K_nope, K_rope_raw, V.

        Returns:
            query: [B, H, S, qk_head_dim]
            k_nope: [B, H, S, qk_nope_head_dim]
            k_rope_raw: [B, 1, S, qk_rope_head_dim]  (single head, before RoPE)
            value: [B, H, S, v_head_dim]
        """
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)

        # KV down projection + split
        kv_combined = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_rope_raw = torch.split(
            kv_combined, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Reshape: k_nope [B,S,H,D] -> [B,H,S,D], value same
        k_nope = k_nope.transpose(1, 2)
        value = value.transpose(1, 2)

        # k_rope: [B, S, rope_dim] -> [B, 1, S, rope_dim]
        k_rope_raw = k_rope_raw.unsqueeze(1)

        return q, k_nope, k_rope_raw, value

    def _apply_rope_and_assemble(
        self,
        query_states: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope_raw: torch.Tensor,
        position_ids: torch.Tensor,
        lck: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to rope dims and assemble full Q, K.

        Returns:
            query_states: [B, H, S, qk_head_dim]
            key_states: [B, H, S, qk_head_dim]
        """
        bsz, num_heads, q_len, _ = query_states.shape

        q_nope = query_states[..., : self.qk_nope_head_dim]
        q_rope = query_states[..., self.qk_nope_head_dim :]

        cos, sin = self.rotary_emb(q_rope, seq_len=q_len + lck)
        cos, sin = cos.to(q_rope.device), sin.to(q_rope.device)
        q_rope, k_rope = _apply_rotary_pos_emb_interleaved(
            q_rope, k_rope_raw, cos, sin, position_ids + lck
        )

        # Expand k_rope from [B, 1, S, rope_dim] to [B, H, S, rope_dim]
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)

        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)

        return query_states, key_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states, k_nope, k_rope_raw, value_states = self._project_qkv(hidden_states)

        if not use_cache:
            query_states, key_states = self._apply_rope_and_assemble(
                query_states, k_nope, k_rope_raw, position_ids, lck=0
            )

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=0.0,
                scale=self.softmax_scale,
            )

        else:
            # Cached path with EAGLE3 suffix attention pattern
            # cache_keys shape: [bsz, num_heads, num_cached, seq_len, qk_head_dim]
            # cache_values shape: [bsz, num_heads, num_cached, seq_len, v_head_dim]
            lck = 0 if cache_keys is None else cache_keys.shape[2]

            query_states, key_states = self._apply_rope_and_assemble(
                query_states, k_nope, k_rope_raw, position_ids, lck=lck
            )

            # Append to 5D tensor cache (K and V have different last dims)
            if cache_keys is None:
                cache_keys = key_states.unsqueeze(2)
                cache_values = value_states.unsqueeze(2)
            else:
                cache_keys = torch.cat([cache_keys, key_states.unsqueeze(2)], dim=2)
                cache_values = torch.cat([cache_values, value_states.unsqueeze(2)], dim=2)

            lck = cache_keys.shape[2]
            k0 = cache_keys[:, :, 0]
            v0 = cache_values[:, :, 0]

            # Causal attention on k0
            attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) * self.softmax_scale
            attn_weights = attn_weights + attention_mask

            # Suffix diagonal attention on k1..kN
            for i in range(1, lck):
                ki = cache_keys[:, :, i]
                attn_weightsi = (query_states * ki).sum(-1) * self.softmax_scale
                attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

            # Upcast to fp32 for softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_weights0 = attn_weights[..., :q_len]
            attn_output = torch.matmul(attn_weights0, v0)

            for i in range(1, lck):
                vi = cache_values[:, :, i]
                attn_weightsi = attn_weights[..., q_len + i - 1]
                attn_outputi = attn_weightsi[..., None] * vi
                attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, cache_keys, cache_values


class DeepSeekMLAFlexAttention(DeepSeekMLAAttention):
    """MLA attention with flex_attention backend.

    Cache is concatenated along the seq dimension (not 5D):
      cache_keys:   [B, H, total_seq, qk_head_dim]
      cache_values: [B, H, total_seq, v_head_dim]

    EAGLE3 mask pattern is handled by eagle3_block_mask (analytical when shapes
    align to BLOCK_SIZE/Q_LEN, create_block_mask fallback otherwise).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states, k_nope, k_rope_raw, value_states = self._project_qkv(hidden_states)

        # cache_keys shape: [bsz, num_heads, past_seq_len, qk_head_dim] (concatenated along seq)
        past_seen_tokens = cache_keys.shape[2] if cache_keys is not None else 0
        lck = past_seen_tokens // q_len

        query_states, key_states = self._apply_rope_and_assemble(
            query_states, k_nope, k_rope_raw, position_ids, lck=lck
        )

        if cache_keys is not None:
            key_cache = torch.cat([cache_keys, key_states], dim=2)
            value_cache = torch.cat([cache_values, value_states], dim=2)
        else:
            key_cache = key_states
            value_cache = value_states

        flex_attention_func = flex_attention if q_len <= 128 else compile_friendly_flex_attention

        block_mask = eagle3_block_mask(
            Q_LEN=q_len,
            KV_LEN=key_cache.shape[-2],
            B=bsz,
            H=1,  # Rely on broadcast
            device=query_states.device,
            lck=lck,
        )

        attn_output = flex_attention_func(
            query=query_states,
            key=key_cache.contiguous(),
            value=value_cache.contiguous(),
            block_mask=block_mask,
            scale=self.softmax_scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, key_cache, value_cache


class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size

        if attention_backend == "sdpa":
            self.self_attn = DeepSeekMLAAttention(config=config)
        elif attention_backend == "flex_attention":
            print_with_rank("Using flex attention on MLA draft model training!")
            self.self_attn = DeepSeekMLAFlexAttention(config=config)
        else:
            raise ValueError(
                f"DeepSeekDecoderLayer supports 'sdpa' and 'flex_attention' backends, got '{attention_backend}'"
            )

        self.attention_backend = attention_backend
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        # Eagle3: concatenate input embedding and hidden states
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        hidden_states, cache_keys, cache_values = self.self_attn(
            hidden_states=hidden_states,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache_keys, cache_values


class Eagle3DeepseekV2ForCausalLM(Eagle3DraftModel):
    """Eagle3 draft model using DeepSeek MLA attention."""

    config_class = DeepseekV3Config

    def __init__(self, config: DeepseekV3Config, attention_backend: str = "sdpa") -> None:
        super().__init__(config)

        self.target_vocab_size = config.vocab_size
        self.vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = DeepSeekDecoderLayer(config, attention_backend=attention_backend)

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.fc = nn.Linear(
            target_hidden_size * self.num_aux_hidden_states, config.hidden_size, bias=False
        )
        use_fc_norm = getattr(config, "fc_norm", None)
        if use_fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    LlamaRMSNorm(target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(self.num_aux_hidden_states)
                ]
            )
        else:
            self.fc_norm = None

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_output = getattr(config, "norm_output", False)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        if self.vocab_size != self.target_vocab_size:
            self.register_buffer("t2d", torch.ones(self.target_vocab_size, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(self.vocab_size, dtype=torch.int64))

        logger.info(
            f"Eagle3DeepseekV2ForCausalLM: hidden_size={config.hidden_size}, "
            f"num_heads={config.num_attention_heads}, "
            f"kv_lora_rank={config.kv_lora_rank}, "
            f"q_lora_rank={config.q_lora_rank}, "
            f"qk_nope={config.qk_nope_head_dim}, qk_rope={config.qk_rope_head_dim}, "
            f"v_head={config.v_head_dim}, "
            f"vocab={self.vocab_size}/{self.target_vocab_size}"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected_size = self.fc.in_features
        if hidden_states.size(-1) != expected_size:
            raise ValueError(
                f"Target hidden states size mismatch: {hidden_states.size(-1)} != expected: {expected_size}"
            )
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
