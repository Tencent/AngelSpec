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

import copy
import json
import logging
import warnings

import torch
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)


def _copy_config_value(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return copy.deepcopy(value)


def _normalize_rope_scaling(rope_scaling):
    if rope_scaling is None:
        return None

    normalized = _copy_config_value(rope_scaling)
    if not isinstance(normalized, dict):
        return normalized

    scaling_type = normalized.get("rope_type", normalized.get("type"))
    if scaling_type == "yarn":
        yarn_defaults = {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
        }
        for key, default in yarn_defaults.items():
            if normalized.get(key) is None:
                normalized[key] = default

    return normalized


def generate_draft_model_config(
    target_model_path: str, template_config_path: str = None, cache_dir: str = None
):
    """
    Auto-generate draft model config based on target model parameters.

    When template_config_path is provided, uses it as a base and overrides with
    target model params. Otherwise, builds the config entirely from the target model.

    Args:
        target_model_path (str): Path to the target model
        template_config_path (str, optional): Template config file path
        cache_dir (str, optional): Cache directory

    Returns:
        dict: Generated draft model config dictionary
    """
    target_config = AutoConfig.from_pretrained(
        target_model_path, cache_dir=cache_dir, trust_remote_code=True
    )

    text_config = getattr(target_config, "text_config", target_config)

    # VLMs often resize embeddings after init, making config.vocab_size stale.
    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path, cache_dir=cache_dir, trust_remote_code=True
    )
    tokenizer_vocab_size = len(tokenizer)
    if tokenizer_vocab_size > text_config.vocab_size:
        logger.warning(
            f"Tokenizer vocab size ({tokenizer_vocab_size}) > config vocab_size "
            f"({text_config.vocab_size}). Using tokenizer vocab size."
        )
        text_config.vocab_size = tokenizer_vocab_size

    if template_config_path is not None:
        with open(template_config_path, "r") as f:
            draft_config = json.load(f)
    else:
        warnings.warn(
            "No template config provided for draft model. "
            "Auto-generating config entirely from target model. "
            "Consider providing a template via draft_model_config for full control."
        )
        draft_config = {
            "architectures": ["LlamaForCausalLMEagle3"],
        }

    draft_config["model_type"] = "llama"

    param_mappings = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        "rope_theta": "rope_theta",
        "rope_scaling": "rope_scaling",
        "rms_norm_eps": "rms_norm_eps",
        "hidden_act": "hidden_act",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "torch_dtype": "torch_dtype",
    }

    for target_param, draft_param in param_mappings.items():
        if hasattr(text_config, target_param):
            value = getattr(text_config, target_param)
        elif hasattr(target_config, target_param):
            value = getattr(target_config, target_param)
        else:
            continue
        if target_param == "torch_dtype" and isinstance(value, torch.dtype):
            value = str(value).replace("torch.", "")
        else:
            value = _copy_config_value(value)
        if target_param == "rope_scaling":
            value = _normalize_rope_scaling(value)
        draft_config[draft_param] = value

    draft_config["num_hidden_layers"] = 1
    draft_config["tie_word_embeddings"] = False
    draft_config["use_cache"] = True

    if "draft_vocab_size" not in draft_config:
        draft_config["draft_vocab_size"] = draft_config.get("vocab_size")

    return draft_config
