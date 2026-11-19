import copy
import json
from dataclasses import asdict
from pathlib import Path

from angelspec.data.positions import PositionRecipe, position_recipe_from_config
from angelspec.models.ops.mrope import validate_mrope_parameters

HY4_FLASH_MODEL_TYPE = "hy_v4_flash"


def _mtp_attention_type(layer_types):
    if layer_types and "full_attention" in layer_types:
        return "full_attention"
    return layer_types[-1] if layer_types else "full_attention"


def flat_target_rope(text):
    """RoPE parameters of the target's MTP block as one flat dict.

    Hy4 flash nests RoPE per layer type (``{"deepseek_sparse_attention": {...},
    "sliding_attention": {...}}``); the served MTP block uses the entry of its own
    attention type.
    """
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    nested = {k: v for k, v in rope.items() if isinstance(v, dict)}
    if not nested or rope.get("mrope_section"):
        return rope
    layer_type = _mtp_attention_type(text.get("layer_types"))
    if layer_type not in nested:
        raise ValueError(
            f"Target rope_parameters has no entry for MTP attention type {layer_type!r}"
        )
    flat = {k: v for k, v in rope.items() if not isinstance(v, dict)}
    flat.update(nested[layer_type])
    return flat


def configure_draft_positions(config, target_model_path):
    if getattr(config, "attention_type", None) != "mla":
        return config
    is_mtp = getattr(config, "model_type", None) == "mtp"
    supplied = getattr(config, "rope_parameters", None) or {}
    if not is_mtp and not supplied.get("mrope_section"):
        return config
    if isinstance(target_model_path, dict):
        target = target_model_path
    else:
        path = Path(target_model_path) / "config.json"
        if path.is_file():
            target = json.loads(path.read_text())
        else:
            from transformers import AutoConfig

            target = AutoConfig.from_pretrained(
                target_model_path, trust_remote_code=True
            ).to_dict()
    text = target.get("text_config", target)
    target_rope = flat_target_rope(text)
    if is_mtp and not target_rope.get("mrope_section"):
        if supplied.get("mrope_section"):
            raise ValueError("Native MTP MRoPE differs from the target serving configuration")
        return config
    rope = copy.deepcopy(target_rope if is_mtp else supplied)
    rope.setdefault("rope_theta", text.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
    rope.setdefault("rope_type", rope.get("type", "default"))
    if is_mtp:
        if int(text.get("qk_rope_head_dim", config.qk_rope_head_dim)) != config.qk_rope_head_dim:
            raise ValueError("MTP and target qk_rope_head_dim differ")
        for key in ("mrope_section", "mrope_interleaved", "rope_theta", "rope_type"):
            if key in supplied and supplied[key] != rope.get(
                key, False if key == "mrope_interleaved" else None
            ):
                raise ValueError(f"MTP {key} differs from target")
    validate_mrope_parameters(config.qk_rope_head_dim, rope)
    if not is_mtp and float(getattr(config, "rope_theta", rope["rope_theta"])) != float(
        rope["rope_theta"]
    ):
        raise ValueError("Draft rope_theta conflicts with the exported MRoPE configuration")
    provider = getattr(config, "position_provider", None)
    if provider is None:
        if target.get("model_type") != HY4_FLASH_MODEL_TYPE:
            raise ValueError(
                "Unknown MRoPE model family; configure a position provider explicitly"
            )
        provider = HY4_FLASH_MODEL_TYPE
    expected_fingerprint = getattr(config, "position_recipe_fingerprint", None)
    if provider == HY4_FLASH_MODEL_TYPE:
        if target.get("model_type") != HY4_FLASH_MODEL_TYPE:
            raise ValueError("the hy_v4_flash position provider requires a Hy4 flash target")
        vision = target.get("vision_config", {})
        for key in ("cat_extra_token", "video_spatial_compression", "video_temporal_compression"):
            if vision.get(key, False):
                raise ValueError(f"Hy4 token position reconstruction does not support {key}")
        # Hy4 flash configs may carry ``use_image_newline_token: null``, the default (on).
        if vision.get("use_image_newline_token") is False:
            raise ValueError("Hy4 token position reconstruction requires image newline tokens")
        for key in ("temporal_patch_size", "spatial_patch_size"):
            if vision.get(key, 1) != 1:
                raise ValueError(f"Hy4 token position reconstruction requires {key}=1")
        for key, default in asdict(PositionRecipe()).items():
            if key.endswith("token_id"):
                value = target.get(key, default)
                old = getattr(config, key, None)
                if old is not None and old != value:
                    raise ValueError(f"Draft {key} differs from target")
                setattr(config, key, value)
    config.rope_parameters = rope
    config.rope_theta = float(rope["rope_theta"])
    config.position_provider = provider
    if not hasattr(config, "position_recipe_version"):
        config.position_recipe_version = 1
    recipe = position_recipe_from_config(config)
    if expected_fingerprint is not None and expected_fingerprint != recipe.identity:
        raise ValueError("Saved position recipe differs from the current configuration")
    config.position_recipe_fingerprint = recipe.identity
    return config
