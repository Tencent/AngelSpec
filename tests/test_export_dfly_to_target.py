import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from angelspec.models.draft.auto import AutoDraftModel, AutoDraftModelConfig
from tools.export_dfly_to_target import (
    PREFIX,
    SHARD,
    dfly_fields,
    export,
    is_fp8_projection,
    quantize_fp8_blocks,
    serving_key,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "angelspec", "config")

_LAYER_TENSORS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.linear_gate.weight",
    "self_attn.post_proj_norm.weight",
    "self_attn.learnable_sink_param",
)
_GLOBAL_TENSORS = (
    "final_norm.weight",
    "hidden_correction.hidden_norm.weight",
    "hidden_correction.embed_norm.weight",
    "hidden_correction.gate_proj.weight",
    "hidden_correction.up_proj.weight",
    "hidden_correction.down_proj.weight",
    "mtp_fusion.gate",
    "mtp_fusion.enorm.weight",
    "mtp_fusion.hnorm.weight",
    "mtp_fusion.eh_proj.weight",
    "confidence_head.proj.weight",
    "confidence_head.proj.bias",
)


def test_hy_v4_flash_drafter_maps_onto_the_served_layout():
    config = AutoDraftModelConfig.from_file(
        os.path.join(CONFIG_DIR, "dfly2_hy_v4_flash_draft_config.json")
    )
    with torch.device("meta"):
        model = AutoDraftModel.from_config(config)
    served, fp8 = set(), set()
    for key in model.state_dict():
        if key == "embed_tokens.weight":
            continue
        served.add(serving_key(key))
        if is_fp8_projection(key):
            fp8.add(serving_key(key))
    expected = {PREFIX + name for name in _GLOBAL_TENSORS}
    expected |= {f"{PREFIX}{i}.{name}" for i in range(5) for name in _LAYER_TENSORS}
    assert served == expected
    projections = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    projections += ("self_attn.q_b_proj", "self_attn.o_proj", "self_attn.linear_gate")
    assert fp8 == {f"{PREFIX}{i}.{p}.weight" for i in range(5) for p in projections}


def test_block_fp8_round_trips_within_e4m3_precision():
    torch.manual_seed(0)
    weight = torch.randn(300, 200) * torch.logspace(-3, 1, 200)
    weight[:128, :128] = 0
    quantized, scale = quantize_fp8_blocks(weight, (128, 128), torch.bfloat16)
    assert quantized.dtype == torch.float8_e4m3fn and scale.shape == (3, 2)
    expanded = scale.float().repeat_interleave(128, 0)[:300].repeat_interleave(128, 1)[:, :200]
    restored = quantized.float() * expanded
    assert bool(torch.isfinite(restored).all())
    assert float(restored[:128, :128].abs().max()) == 0.0
    error = (restored - weight).abs() / weight.abs().clamp_min(1e-6)
    assert float(error[weight.abs() > expanded * 16].max()) < 0.07


def _write_target(root, drafter_shard=False):
    os.makedirs(root)
    config = {
        "model_type": "hy_v4_flash",
        "text_config": {
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["model.layers.0.mlp.gate_proj", f"{PREFIX}9.stale"],
            }
        },
        "dfly_num_hidden_layers": 9,
    }
    tensors = {
        "model.embed_tokens.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4).to(torch.float8_e4m3fn),
        "model.layers.0.self_attn.o_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.bfloat16),
        f"{PREFIX}9.stale.weight": torch.zeros(16, 16, dtype=torch.bfloat16),
    }
    files = (
        ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        if drafter_shard
        else ("model-00001-of-00001.safetensors",) * 2
    )
    weight_map = {key: files[key.startswith(PREFIX)] for key in tensors}
    for file_name in set(files):
        save_file(
            {key: t for key, t in tensors.items() if weight_map[key] == file_name},
            os.path.join(root, file_name),
        )
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(root, "tokenizer.json"), "w") as f:
        f.write("{}")
    return total


def _write_draft(root, **overrides):
    os.makedirs(root)
    torch.manual_seed(1)
    state = {
        "embed_tokens.weight": torch.randn(8, 4),
        "layers.0.input_layernorm.weight": torch.randn(4),
        "layers.0.self_attn.q_b_proj.weight": torch.randn(256, 200),
        "layers.0.self_attn.q_a_proj.weight": torch.randn(100, 64),
        "final_norm.weight": torch.randn(4),
        "mtp_fusion.gate": torch.randn(4),
        "mtp_fusion.eh_proj.weight": torch.randn(4, 8),
    }
    save_file(state, os.path.join(root, "model.safetensors"))
    config = {
        "model_type": "dfly2",
        "num_hidden_layers": 1,
        "mask_token_id": 7,
        "mtp_fusion": "slot0",
        "enable_hidden_correction": False,
        "context_local_window": None,
        "target_index_layer": [3],
        "target_index_topk": 64,
        "target_kv_layers": [3],
        **overrides,
    }
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config, f)
    return state


def _read(path):
    with safe_open(path, framework="pt") as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def _assert_each_tensor_served_once(out):
    """vLLM reads every tensor of every shard in the directory."""
    seen = {}
    for name in sorted(os.listdir(out)):
        if name.endswith(".safetensors"):
            for key in _read(os.path.join(out, name)):
                assert key not in seen, f"{key} in {seen.get(key)} and {name}"
                seen[key] = name
    with open(os.path.join(out, "model.safetensors.index.json")) as f:
        assert seen == json.load(f)["weight_map"]


def test_export_writes_the_served_checkpoint(tmp_path):
    target, draft, out = (str(tmp_path / name) for name in ("target", "draft", "out"))
    total = _write_target(target)
    state = _write_draft(draft)
    export(target, draft, out)

    shard = _read(os.path.join(out, SHARD))
    q_b = f"{PREFIX}0.self_attn.q_b_proj.weight"
    assert shard[q_b].dtype == torch.float8_e4m3fn
    assert shard[q_b + "_scale_inv"].dtype == torch.bfloat16
    assert shard[q_b + "_scale_inv"].shape == (2, 2)
    restored = (
        shard[q_b].float()
        * shard[q_b + "_scale_inv"]
        .float()
        .repeat_interleave(128, 0)
        .repeat_interleave(128, 1)[:, :200]
    )
    torch.testing.assert_close(
        restored, state["layers.0.self_attn.q_b_proj.weight"], rtol=0.07, atol=1e-2
    )
    assert shard[f"{PREFIX}0.self_attn.q_a_proj.weight"].dtype == torch.bfloat16
    assert shard[f"{PREFIX}mtp_fusion.gate"].dtype == torch.bfloat16
    assert not any("embed_tokens" in key for key in shard)

    with open(os.path.join(out, "model.safetensors.index.json")) as f:
        index = json.load(f)
    assert f"{PREFIX}9.stale.weight" not in index["weight_map"]
    assert all(index["weight_map"][key] == SHARD for key in shard)
    added = sum(t.numel() * t.element_size() for t in shard.values())
    assert index["metadata"]["total_size"] == total - 16 * 16 * 2 + added

    with open(os.path.join(out, "config.json")) as f:
        config = json.load(f)
    assert config["dfly_num_hidden_layers"] == 1 and config["dfly_mask_token_id"] == 7
    assert config["dfly_mtp_fusion"] == "slot0" and config["dfly_target_kv_layers"] == [3]
    assert config["dfly_context_local_window"] is None
    assert (config["dfly_enable_ihc"], config["dfly_mlp_type"], config["dfly_attn_causal"]) == (
        False,
        "dense",
        True,
    )
    assert "dfly_block_size" not in config
    assert config["text_config"]["quantization_config"]["modules_to_not_convert"] == [
        "model.layers.0.mlp.gate_proj",
        f"{PREFIX}0.self_attn.q_a_proj",
        f"{PREFIX}mtp_fusion.eh_proj",
    ]
    assert os.path.islink(os.path.join(out, "tokenizer.json"))
    rewritten = os.path.join(out, "model-00001-of-00001.safetensors")
    assert not os.path.islink(rewritten)
    assert not any(key.startswith(PREFIX) for key in _read(rewritten))
    _assert_each_tensor_served_once(out)


def test_export_drops_the_shard_of_the_drafter_the_target_served(tmp_path):
    target, draft, out = (str(tmp_path / name) for name in ("target", "draft", "out"))
    _write_target(target, drafter_shard=True)
    _write_draft(draft)
    export(target, draft, out)
    assert os.path.islink(os.path.join(out, "model-00001-of-00002.safetensors"))
    assert not os.path.lexists(os.path.join(out, "model-00002-of-00002.safetensors"))
    _assert_each_tensor_served_once(out)


def test_bf16_export_leaves_the_projections_unquantized(tmp_path):
    target, draft, out = (str(tmp_path / name) for name in ("target", "draft", "out"))
    _write_target(target)
    _write_draft(draft, model_type="dspark")
    export(target, draft, out, fp8=False)
    shard = _read(os.path.join(out, SHARD))
    assert shard[f"{PREFIX}0.self_attn.q_b_proj.weight"].dtype == torch.bfloat16
    assert not any(key.endswith("_scale_inv") for key in shard)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_type": "qwen3", "model_arch": "dfly", "target_kv_layers": None},
        {"model_type": "dspark", "target_kv_layers": None},
    ],
)
def test_export_rejects_drafters_vllm_cannot_serve_as_dfly2(tmp_path, overrides):
    target, draft, out = (str(tmp_path / name) for name in ("target", "draft", "out"))
    _write_target(target)
    _write_draft(draft, **overrides)
    with pytest.raises(ValueError, match="not a DFly2 drafter"):
        export(target, draft, out)
    assert not os.path.exists(out)


def test_block_size_is_written_only_when_given():
    config = {"num_hidden_layers": 5, "mask_token_id": 9, "target_kv_layers": [1]}
    assert dfly_fields(config, block_size=8)["dfly_block_size"] == 8
    assert "dfly_block_size" not in dfly_fields(config)


@pytest.mark.skipif(
    not os.environ.get("ANGELSPEC_HY4_TARGET"),
    reason="set ANGELSPEC_HY4_TARGET to a Hy4 checkpoint",
)
def test_hy4_draft_config_exports_the_released_dfly_fields():
    with open(os.path.join(os.environ["ANGELSPEC_HY4_TARGET"], "config.json")) as f:
        released = {key: value for key, value in json.load(f).items() if key.startswith("dfly_")}
    with open(os.path.join(CONFIG_DIR, "dfly2_hy_v4_draft_config.json")) as f:
        draft = json.load(f)
    assert dfly_fields(draft, block_size=released["dfly_block_size"]) == released


@pytest.mark.skipif(
    not os.environ.get("ANGELSPEC_HY4_TARGET"),
    reason="set ANGELSPEC_HY4_TARGET to a Hy4 checkpoint",
)
def test_hy4_draft_config_matches_the_released_drafter():
    target = os.environ["ANGELSPEC_HY4_TARGET"]
    config = AutoDraftModelConfig.from_file(
        os.path.join(CONFIG_DIR, "dfly2_hy_v4_draft_config.json")
    )
    with torch.device("meta"):
        model = AutoDraftModel.from_config(config)
    ours = {
        serving_key(key): tuple(value.shape)
        for key, value in model.state_dict().items()
        if key not in ("embed_tokens.weight", "lm_head.weight")
    }
    with open(os.path.join(target, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    released = {}
    for key, file_name in weight_map.items():
        if key.startswith(PREFIX):
            with safe_open(os.path.join(target, file_name), framework="pt") as f:
                released[key] = tuple(f.get_slice(key).get_shape())
    assert ours == released
