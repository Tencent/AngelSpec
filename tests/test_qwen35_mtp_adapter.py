"""Qwen3.5's native MTP layer as a drafter: against transformers' decoder layer, through
an export round trip, and trained through the TTT wrapper."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel
from angelspec.training.qwen35_mtp import (
    draft_to_native,
    initialize_qwen35_mtp,
    native_to_draft,
    validate_qwen35_target,
)


def config():
    return MTPConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=53,
        mlp_type="dense",
        target_hidden_size=32,
        rms_norm_type="qwen3_5",
        attn_output_gate=True,
        partial_rotary_factor=0.5,
        max_position_embeddings=128,
        rope_theta=10000,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
    )


def target_config(cfg):
    return {
        "model_type": "qwen3_5",
        "text_config": {
            **{
                name: getattr(cfg, name)
                for name in (
                    "hidden_size",
                    "intermediate_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "vocab_size",
                    "rms_norm_eps",
                )
            },
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 4,
            "attn_output_gate": True,
            "tie_word_embeddings": True,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "rope_parameters": {
                "rope_theta": cfg.rope_theta,
                "rope_type": "default",
                "partial_rotary_factor": cfg.partial_rotary_factor,
            },
        },
    }


def reference_forward(model, embedding, hidden, mask, positions, layer=None):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    cfg = model.config
    if layer is None:
        target = Qwen3_5TextConfig(**target_config(cfg)["text_config"])
        target._attn_implementation = "eager"
        layer = Qwen3_5DecoderLayer(target, layer_idx=3).to(embedding.dtype)
        layer.load_state_dict(model.midlayer.state_dict(), strict=True)
    values = model.eh_proj(torch.cat([model.enorm(embedding), model.hnorm(hidden)], -1))
    dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    frequency = positions.float().unsqueeze(-1) * inv_freq
    angles = torch.cat([frequency, frequency], -1)
    result = layer(
        values, (angles.cos().to(values.dtype), angles.sin().to(values.dtype)), attention_mask=mask
    )
    return model.final_layernorm(result), layer


def test_native_hf_layer_forward_and_gradient():
    torch.manual_seed(4)
    cfg = config()
    model = MTPDraftModel(cfg, attention_backend="sdpa")
    reference = copy.deepcopy(model)
    inputs = torch.randn(2, 7, 32, requires_grad=True)
    inputs_ref = inputs.detach().clone().requires_grad_()
    hidden = torch.randn_like(inputs)
    positions = torch.arange(7).expand(2, -1)
    mask = torch.triu(torch.full((1, 1, 7, 7), float("-inf")), diagonal=1).expand(2, -1, -1, -1)
    actual, _, _ = model.backbone(inputs, hidden, mask, positions, use_cache=False)
    expected, layer = reference_forward(reference, inputs_ref, hidden, mask, positions)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(inputs.grad, inputs_ref.grad, rtol=1e-4, atol=1e-6)
    for name, left in model.midlayer.named_parameters():
        right = dict(layer.named_parameters())[name]
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)


def test_moe_layer_matches_hf_through_native_weights():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
    )

    torch.manual_seed(6)
    moe = dict(
        mlp_type="moe",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_router_use_sigmoid=False,
        moe_router_enable_expert_bias=False,
        router_scaling_factor=1.0,
        shared_expert_gate=True,
    )
    cfg = MTPConfig(**{**config().to_dict(), **moe})
    model = MTPDraftModel(cfg, attention_backend="sdpa")
    text = {k: v for k, v in target_config(cfg)["text_config"].items() if k != "intermediate_size"}
    text.update(
        model_type="qwen3_5_moe_text",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )
    validate_qwen35_target(cfg, {"text_config": text})
    hf = Qwen3_5MoeTextConfig(**text)
    hf._attn_implementation = "eager"
    layer = Qwen3_5MoeDecoderLayer(hf, layer_idx=3)

    native = draft_to_native(model.state_dict())
    draft = native_to_draft(native)
    for name, value in model.state_dict().items():
        if name != "embed_tokens.weight":
            assert torch.equal(draft[name], value), name
    prefix = "mtp.layers.0."
    state = {k[len(prefix) :]: v for k, v in native.items() if k.startswith(prefix)}
    experts = [
        state.pop(f"mlp.experts.{e}.{p}_proj.weight")
        for e in range(4)
        for p in ("gate", "up", "down")
    ]
    state["mlp.experts.gate_up_proj"] = torch.stack(
        [torch.cat(experts[3 * e : 3 * e + 2]) for e in range(4)]
    )
    state["mlp.experts.down_proj"] = torch.stack([experts[3 * e + 2] for e in range(4)])
    layer.load_state_dict(state, strict=True)

    embedding, hidden = torch.randn(2, 7, 32), torch.randn(2, 7, 32)
    positions = torch.arange(7).expand(2, -1)
    mask = torch.triu(torch.full((1, 1, 7, 7), float("-inf")), diagonal=1).expand(2, -1, -1, -1)
    actual, _, _ = model.backbone(embedding, hidden, mask, positions, use_cache=False)
    expected, _ = reference_forward(model, embedding, hidden, mask, positions, layer=layer)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


def test_export_round_trip(tmp_path):
    from angelspec.training.qwen35_mtp import read_weights
    from tools.export_qwen35_mtp import export

    cfg = config()
    draft = MTPDraftModel(cfg)
    source = tmp_path / "draft"
    source.mkdir()
    cfg.save_pretrained(source)
    torch.save(draft.state_dict(), source / "pytorch_model.bin")
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps(target_config(cfg)))
    destination = export(source, target, tmp_path / "export")
    actual = read_weights(destination, prefix="mtp.")
    assert set(actual) == set(draft_to_native(draft.state_dict()))
    for key, value in draft_to_native(draft.state_dict()).items():
        assert torch.equal(actual[key], value.to(torch.bfloat16))
    restored = MTPDraftModel(cfg)
    initialize_qwen35_mtp(restored, SimpleNamespace(draft_init_path=str(destination)), cfg, target)
    for key, value in native_to_draft(actual).items():
        assert torch.equal(restored.state_dict()[key], value.float())
    with pytest.raises(FileExistsError):
        export(source, target, destination)


def test_qwen35_ttt_loss_checkpoint_and_packing():
    from angelspec.models.mtp import MTPModel

    torch.manual_seed(5)
    cfg = config()
    draft = MTPDraftModel(cfg, attention_backend="sdpa")
    draft.set_lm_head_weight(torch.randn(53, 32))
    model = MTPModel(
        draft,
        length=3,
        attention_backend="sdpa",
        gradient_checkpointing=True,
        on_policy=True,
        argmax_chunk_size=2,
        loss_checkpoint_chunks=True,
    )
    ids = torch.randint(1, 53, (1, 8))
    result = model(
        input_ids=ids,
        hidden_states=torch.randn(1, 8, 32),
        target_hidden_states=torch.randn(1, 8, 32),
        target_lm_head_weight=draft.lm_head_weight,
        attention_mask=torch.ones_like(ids),
        loss_mask=torch.ones_like(ids),
        ctx_doc_ids=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        base_position_ids=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
    )
    loss = sum(result[0])
    loss.backward()
    assert torch.isfinite(loss)
    assert draft.midlayer.self_attn.q_proj.weight.grad is not None
