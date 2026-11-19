import pytest
from omegaconf import OmegaConf

from angelspec.config.train_config import (
    Config,
    _normalize_legacy_backend,
    _validate_vllm_config,
)


def test_vllm_decode_validation_uses_inference_backend():
    config = OmegaConf.structured(Config)
    config.inference.inference_engine_type = "vllm"
    config.training.train_with_decode = True
    with pytest.raises(NotImplementedError, match="train_with_decode"):
        _validate_vllm_config(config)


def test_legacy_target_model_backend_is_migrated():
    config = OmegaConf.create({"model": {"target_model_backend": "sglang"}})
    _normalize_legacy_backend(config)
    assert config.inference.inference_engine_type == "sgl"
