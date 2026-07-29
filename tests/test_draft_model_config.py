from types import SimpleNamespace

from angelspec.config.utils import generate_draft_model_config


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
