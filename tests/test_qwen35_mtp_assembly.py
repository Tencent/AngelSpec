"""tools/qwen35_mtp.py: a donor's MTP head replacing the one a Qwen3.5 checkpoint ships."""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from qwen35_mtp import assemble, checkpoint_index, expected_shapes  # noqa: E402

CONFIG = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 4,
        "head_dim": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 6,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "mtp_num_hidden_layers": 1,
        "rms_norm_eps": 1e-6,
        "attn_output_gate": True,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
    },
    "vision_config": {"hidden_size": 4},
}


def _head():
    shapes = expected_shapes(CONFIG["text_config"], 1, False)
    return {key: torch.randn(shape) for key, shape in shapes.items()}


def _checkpoint(path, tensors):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(CONFIG))
    (path / "tokenizer.json").write_text('{"test": true}')
    shards = {
        "model-00001.safetensors": {k: v for k, v in tensors.items() if "visual" not in k},
        "model-00002.safetensors": {k: v for k, v in tensors.items() if "visual" in k},
    }
    for name, payload in shards.items():
        save_file(payload, path / name, metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {key: name for name, payload in shards.items() for key in payload},
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_replace_swaps_the_head_and_leaves_the_sources_alone(tmp_path):
    backbone = {
        "model.language_model.embed_tokens.weight": torch.randn(16, 4),
        "model.language_model.layers.0.weight": torch.randn(4, 4),
        "model.visual.weight": torch.randn(4, 4),
    }
    new_head = _head()
    base, donor, out = tmp_path / "base", tmp_path / "donor", tmp_path / "out"
    _checkpoint(base, {**backbone, **_head()})
    _checkpoint(donor, {**backbone, **new_head})
    before = {p.name: p.read_bytes() for p in base.iterdir()}
    assemble(base, donor, out, "replace")
    assert before == {p.name: p.read_bytes() for p in base.iterdir()}
    assert not any(p.is_symlink() for p in out.rglob("*"))
    index, _ = checkpoint_index(out)
    loaded = {}
    for shard in set(index["weight_map"].values()):
        loaded.update(load_file(out / shard))
    assert set(loaded) == set(backbone) | set(new_head)
    for key, value in {**backbone, **new_head}.items():
        assert torch.equal(value, loaded[key]), key
    assert (out / "tokenizer.json").read_bytes() == (base / "tokenizer.json").read_bytes()
