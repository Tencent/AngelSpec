import glob
import os
from types import SimpleNamespace

import pytest

from angelspec.config.utils import generate_draft_model_config
from angelspec.models.draft.auto import AutoDraftModelConfig
from angelspec.train_entry import _validate_dflash_target_layers


def test_generate_draft_model_config_preserves_rope_fields(monkeypatch):
    rope_scaling = {
        "type": "yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    text_config = SimpleNamespace(
        vocab_size=32000,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        max_position_embeddings=262144,
        rope_theta=5000000,
        rope_scaling=rope_scaling,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )
    target_config = SimpleNamespace(text_config=text_config)

    class DummyTokenizer:
        def __len__(self):
            return 32000

    monkeypatch.setattr(
        "angelspec.config.utils.AutoConfig.from_pretrained",
        lambda *args, **kwargs: target_config,
    )
    monkeypatch.setattr(
        "angelspec.config.utils.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    draft_config = generate_draft_model_config("dummy-model")

    assert draft_config["max_position_embeddings"] == 262144
    assert draft_config["rope_theta"] == 5000000
    assert draft_config["rope_scaling"] == rope_scaling


def test_generate_draft_model_config_copies_rope_scaling(monkeypatch):
    rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192}
    text_config = SimpleNamespace(
        vocab_size=32000,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=8192,
        max_position_embeddings=65536,
        rope_theta=1000000,
        rope_scaling=rope_scaling,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )
    target_config = SimpleNamespace(text_config=text_config)

    class DummyTokenizer:
        def __len__(self):
            return 32000

    monkeypatch.setattr(
        "angelspec.config.utils.AutoConfig.from_pretrained",
        lambda *args, **kwargs: target_config,
    )
    monkeypatch.setattr(
        "angelspec.config.utils.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    draft_config = generate_draft_model_config("dummy-model")
    rope_scaling["factor"] = 999.0

    assert draft_config["rope_scaling"]["factor"] == 8.0


def test_generate_draft_model_config_fills_yarn_defaults(monkeypatch):
    rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192}
    text_config = SimpleNamespace(
        vocab_size=32000,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=8192,
        max_position_embeddings=65536,
        rope_theta=1000000,
        rope_scaling=rope_scaling,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )
    target_config = SimpleNamespace(text_config=text_config)

    class DummyTokenizer:
        def __len__(self):
            return 32000

    monkeypatch.setattr(
        "angelspec.config.utils.AutoConfig.from_pretrained",
        lambda *args, **kwargs: target_config,
    )
    monkeypatch.setattr(
        "angelspec.config.utils.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    draft_config = generate_draft_model_config("dummy-model")

    assert draft_config["rope_scaling"]["beta_fast"] == 32.0
    assert draft_config["rope_scaling"]["beta_slow"] == 1.0
    assert draft_config["rope_scaling"]["mscale"] == 1.0
    assert draft_config["rope_scaling"]["mscale_all_dim"] == 0.0


# ----------------------------------------------------------------------
# Aux target-layer request (engine capture ↔ trainer split ↔ draft projection)
# ----------------------------------------------------------------------

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "angelspec", "config")


def _args(cfg, num_target_layers=None, layer_ids=None, engine="vllm"):
    count = num_target_layers if num_target_layers is not None else cfg.num_target_layers
    return SimpleNamespace(
        inference_engine_type=engine,
        # DSpark configs subclass DFlashConfig but are split by their own knob.
        dflash_num_target_layers=count,
        dspark_num_target_layers=count,
        aux_hidden_states_layers=(
            layer_ids if layer_ids is not None else list(cfg.target_layer_ids)
        ),
    )


@pytest.mark.parametrize("engine", ["vllm", "sgl"])
@pytest.mark.parametrize(
    "path", sorted(glob.glob(os.path.join(CONFIG_DIR, "d*_draft_config.json")))
)
def test_packaged_dflash_configs_are_self_consistent(path, engine):
    cfg = AutoDraftModelConfig.from_file(path)
    if getattr(cfg, "target_layer_ids", None) is None:
        pytest.skip("no explicit target_layer_ids")
    _validate_dflash_target_layers(_args(cfg, engine=engine), cfg)


def test_dspark_target_layer_count_falls_back_to_dflash():
    cfg = AutoDraftModelConfig.from_file(
        os.path.join(CONFIG_DIR, "dspark_qwen3_8b_draft_config.json")
    )
    args = _args(cfg)
    args.dspark_num_target_layers = None
    _validate_dflash_target_layers(args, cfg)
    args.dflash_num_target_layers = cfg.num_target_layers + 1
    with pytest.raises(ValueError, match="training.dflash_num_target_layers="):
        _validate_dflash_target_layers(args, cfg)
