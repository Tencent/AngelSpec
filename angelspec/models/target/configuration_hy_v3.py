"""HYV3 (Hunyuan-3 / ``hy_v3``) HuggingFace config class.

transformers does not ship an ``hy_v3`` config, and Hy3 checkpoints declare
``model_type="hy_v3"`` with no ``auto_map``. Without a registered config class,
``AutoConfig.from_pretrained`` raises before vLLM or AngelSpec can build the
model. vLLM keeps its own ``hy_v3`` config internally; this mirror exists purely
so ``transformers.AutoConfig`` (used on the AngelSpec side to read target
metadata) resolves the same ``model_type``.

Registered into ``transformers.AutoConfig`` by
``angelspec.inference.engine.vllm_runtime_patch.register_extra_hf_configs``.
"""

from typing import Any

from transformers import PretrainedConfig


class HYV3Config(PretrainedConfig):

    model_type = "hy_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=120832,
        hidden_size=4096,
        intermediate_size=13312,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.006,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        rope_parameters: dict[str, Any] | None = None,
        qk_norm=True,
        tie_word_embeddings=False,
        enable_attention_fp32_softmax=False,
        enable_lm_head_fp32=True,
        num_experts=192,
        num_experts_per_tok=8,
        num_shared_experts=1,
        expert_hidden_dim=1536,
        moe_router_enable_expert_bias=True,
        moe_router_use_sigmoid=True,
        route_norm=True,
        router_scaling_factor=None,
        use_grouped_mm=False,
        enable_moe_fp32_combine=False,
        first_k_dense_replace=1,
        output_router_logits=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        rope_theta = kwargs.pop("rope_theta", 11158840.0)
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.rope_parameters = rope_parameters
        self.qk_norm = qk_norm
        self.tie_word_embeddings = tie_word_embeddings
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.enable_attention_fp32_softmax = enable_attention_fp32_softmax

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_router_use_sigmoid = moe_router_use_sigmoid
        self.route_norm = route_norm
        self.use_grouped_mm = use_grouped_mm
        self.router_scaling_factor = router_scaling_factor
        self.enable_moe_fp32_combine = enable_moe_fp32_combine

        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits

        if eos_token_id is not None and isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HYV3Config"]
