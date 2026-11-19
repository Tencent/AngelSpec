import ast
import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from angelspec.data.utils import DataCollatorWithPadding, MTPPackingCollator
from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel, MTPMLAAttention
from angelspec.models.mtp import MTPModel
from angelspec.models.ops.mrope import MRotaryEmbedding, advance_mtp_positions
from angelspec.training.mtp_rope import configure_mtp_rope

ROPE = {
    "rope_theta": 1e7,
    "rope_type": "default",
    "mrope_interleaved": True,
    "mrope_section": [12, 10, 10],
}


def config():
    return MTPConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=128,
        rope_parameters=ROPE,
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        attention_type="mla",
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=64,
        v_head_dim=8,
        gated_mla=True,
        learnable_sink=True,
        index_topk=3,
        index_head_dim=80,
        index_n_heads=2,
        target_hidden_size=32,
    )


def positions():
    return torch.tensor(
        [[[0, 1, 2, 2, 2, 2, 4, 5], [0, 1, 2, 2, 3, 3, 4, 5], [0, 1, 2, 3, 2, 3, 4, 5]]]
    )


def item(pos):
    n = pos.shape[-1]
    return {
        "input_ids": torch.arange(n)[None],
        "loss_mask": torch.ones(1, n, dtype=torch.long),
        "last_hidden_states": torch.randn(1, n, 32),
        "position_ids": pos,
    }


def test_hy4_interleaved_frequencies_and_text_equivalence():
    rope = MRotaryEmbedding(64, ROPE)
    pos = positions()
    ref = torch.zeros(1, 4, 8, 8)
    cos, sin = rope(ref, pos)
    inv = 1 / (1e7 ** (torch.arange(0, 64, 2).float() / 64))
    selected = []
    for j in range(32):
        axis = 1 if j % 3 == 1 and j < 30 else 2 if j % 3 == 2 and j < 30 else 0
        selected.append(pos[:, axis] * inv[j])
    phase = torch.stack(selected, -1).repeat(1, 1, 2)
    torch.testing.assert_close(cos, phase.cos(), rtol=0, atol=0)
    torch.testing.assert_close(sin, phase.sin(), rtol=0, atol=0)
    text = torch.arange(8)[None]
    c1, s1 = rope(ref, text)
    c3, s3 = rope(ref, text[:, None].expand(-1, 3, -1))
    assert torch.equal(c1, c3) and torch.equal(s1, s3)
    assert not torch.equal(cos, c1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mrope_keeps_fp32_frequencies_after_dtype_move(dtype):
    rope = MRotaryEmbedding(64, ROPE).to(dtype=dtype)
    assert rope.inv_freq.dtype == torch.float32
    expected = MRotaryEmbedding(64, ROPE)(torch.zeros(1, 8, 4, 8), positions())
    actual = rope(torch.zeros(1, 8, 4, 8, dtype=dtype), positions())
    assert actual[0].dtype == torch.float32
    assert torch.equal(actual[0], expected[0])


def test_contiguous_sections():
    rope = MRotaryEmbedding(64, {**ROPE, "mrope_interleaved": False})
    p = positions()
    cos, _ = rope(torch.zeros(1, 8, 4, 8), p)
    phase = p.float()[..., None] * rope.inv_freq
    phase = torch.cat([phase[:, 0, :, :12], phase[:, 1, :, 12:22], phase[:, 2, :, 22:]], -1)
    assert torch.equal(cos, phase.repeat(1, 1, 2).cos())


def test_config_inherits_target_and_rejects_conflicts(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "hy_v4_flash", "rope_parameters": ROPE, "qk_rope_head_dim": 64})
    )
    cfg = config()
    cfg.rope_parameters = None
    configure_mtp_rope(cfg, tmp_path)
    assert cfg.rope_parameters == ROPE
    assert MTPConfig(**cfg.to_dict()).rope_parameters == ROPE
    cfg.rope_parameters["rope_theta"] = 1e4
    with pytest.raises(ValueError, match="rope_theta differs"):
        configure_mtp_rope(cfg, tmp_path)


def test_packing_preserves_three_axes_and_document_resets():
    p = positions()
    batch = MTPPackingCollator(20)([item(p), item(p)])
    assert batch["position_ids"].shape == (1, 3, 20)
    assert torch.equal(batch["position_ids"][..., :8], p)
    assert torch.equal(batch["position_ids"][..., 8:16], p)
    assert batch["position_ids"][..., 16:].count_nonzero() == 0
    assert torch.equal(batch["base_position_ids"][0, :16], torch.arange(8).repeat(2))
    bad = item(p)
    del bad["position_ids"]
    with pytest.raises(ValueError, match="prepared position_ids"):
        MTPPackingCollator(20)([item(p), bad])


def test_padding_preserves_three_axes():
    batch = DataCollatorWithPadding()([item(positions()), item(positions()[..., :5])])
    assert batch["position_ids"].shape == (2, 3, 256)
    assert torch.equal(batch["position_ids"][0, :, :8], positions()[0])
    assert batch["position_ids"][1, :, 5:].count_nonzero() == 0


def test_cached_positions_match_vllm_text_decode_convention():
    p = positions()
    assert torch.equal(advance_mtp_positions(p, 0), p)
    for step in range(1, 5):
        assert torch.equal(advance_mtp_positions(p, step), p[:, :1].expand(-1, 3, -1) + step)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_ttt_mrope_five_steps_backward(checkpoint):
    torch.manual_seed(3)
    cfg = config()
    draft = MTPDraftModel(cfg)
    weight = torch.randn(64, 32)
    draft.set_lm_head_weight(weight)
    model = MTPModel(draft, length=5, gradient_checkpointing=checkpoint)
    seen = []
    original = draft.midlayer.self_attn._cos_sin

    def record(ref, pos, n):
        seen.append(pos.detach().clone())
        return original(ref, pos, n)

    kwargs = dict(
        input_ids=torch.arange(8)[None] + 1,
        attention_mask=torch.ones(1, 8),
        target_hidden_states=torch.randn(1, 8, 32),
        target_lm_head_weight=weight,
        loss_mask=torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]),
        hidden_states=torch.randn(1, 8, 32),
        position_ids=positions(),
    )
    with mock.patch.object(draft.midlayer.self_attn, "_cos_sin", side_effect=record):
        out = model(**kwargs)
        assert len(seen) == 5
        for step, actual in enumerate(seen):
            assert torch.equal(actual, advance_mtp_positions(positions(), step))
        sum(out[0]).backward()
    assert draft.eh_proj.weight.grad is not None
    assert torch.isfinite(draft.eh_proj.weight.grad).all()
    kwargs["position_ids"] = None
    with pytest.raises(ValueError, match="prepare positions"):
        model(**kwargs)


def test_mrope_indexer_and_absorbed_attention_share_cos_sin():
    attn = MTPMLAAttention(config())
    hidden = torch.randn(1, 8, 32)
    q_nope, q_pe, k_nope, k_pe, _, _ = attn._project(hidden)
    cos, sin = attn._cos_sin(q_nope, positions(), 8)
    keys = attn.indexer.project_keys(hidden, cos, sin)
    text_cos, text_sin = attn._cos_sin(q_nope, torch.arange(8)[None], 8)
    text_keys = attn.indexer.project_keys(hidden, text_cos, text_sin)
    assert not torch.allclose(keys, text_keys)
    q1, k1 = attn._assemble(q_nope, q_pe, k_nope, k_pe, cos, sin)
    qn, qp, kv, kp, _ = attn._project_absorbed(hidden)
    c2, s2 = attn._cos_sin(qn, positions(), 8)
    q2, k2, _ = attn._assemble_absorbed(qn, qp, kv, kp, c2, s2)
    torch.testing.assert_close(q1[..., -64:].transpose(1, 2), q2[..., -64:])
    torch.testing.assert_close(k1[:, :1, :, -64:].transpose(1, 2), k2[..., -64:])


def test_local_vllm_hy4_formula_matches():
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None:
        pytest.skip("vLLM not installed")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "model_executor/layers/rotary_embedding/mrope.py"
    )
    tree = ast.parse(path.read_text())
    node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "apply_interleaved_rope"
        ),
        None,
    )
    if node is None:
        pytest.skip("vLLM interleaved helper unavailable")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    rope = MRotaryEmbedding(64, ROPE)
    p = positions()
    phase = p.float()[..., None] * rope.inv_freq
    expected = ns["apply_interleaved_rope"](phase[0].cos(), ROPE["mrope_section"])
    actual, _ = rope(torch.zeros(1, 8, 4, 8), p)
    torch.testing.assert_close(actual[0, :, :32], expected, rtol=0, atol=0)


def test_tilelang_mrope_uses_three_axes_and_decode_steps():
    import angelspec.models.draft.mtp as mod

    attn = MTPMLAAttention(config())
    attn.dsa_backend = "tilelang"
    attn.fp8_indexer = False
    selected = torch.zeros(1, 8, 3, dtype=torch.int32)
    index = (selected, torch.zeros(1, 8, dtype=torch.bool))
    seen = []
    original = attn._cos_sin

    def capture(ref, pos, n):
        seen.append(pos.clone())
        return original(ref, pos, n)

    def sparse(q, kv, indices, **kwargs):
        return q[..., :16].float(), None

    keys = values = None
    with (
        mock.patch.object(attn.indexer, "sparse_indices", return_value=index),
        mock.patch.object(mod, "tilelang_sparse_mla", side_effect=sparse),
        mock.patch.object(attn, "_cos_sin", side_effect=capture),
    ):
        for step in range(5):
            out, keys, values, index = attn(
                torch.randn(1, 8, 32),
                cache_keys=keys,
                cache_values=values,
                attention_mask=torch.ones(1, 8),
                position_ids=positions(),
                use_cache=True,
                precomputed_index=index if step else None,
                return_index=True,
            )
            assert torch.equal(seen[-1], advance_mtp_positions(positions(), step))
            assert out.shape == (1, 8, 32)
            assert keys.shape[1] == (step + 1) * 8


def test_mrope_initial_token_mask_only_applies_to_first_step():
    draft = MTPDraftModel(config())
    p = positions()
    p[:, 0, 1] = 0
    embeddings = torch.randn(1, 8, 32)
    seen = []
    handle = draft.enorm.register_forward_pre_hook(
        lambda m, inputs: seen.append(inputs[0].clone())
    )
    try:
        _, keys, values, index = draft.backbone(
            embeddings, torch.randn(1, 8, 32), torch.ones(1, 8), p, return_index=True
        )
        assert seen[0][:, 0].count_nonzero() == 0
        assert torch.equal(seen[0][:, 1], embeddings[:, 1])
        draft.backbone(
            embeddings,
            torch.randn(1, 8, 32),
            torch.ones(1, 8),
            p,
            cache_keys=keys,
            cache_values=values,
            precomputed_index=index,
        )
        assert torch.equal(seen[1], embeddings)
    finally:
        handle.remove()


def test_mrope_rebuilds_nonpersistent_buffer_after_to_empty():
    rope = MRotaryEmbedding(64, ROPE)
    expected = rope.inv_freq.clone()
    rope.to_empty(device="cpu")
    assert torch.equal(rope.inv_freq, expected)
    assert not rope.state_dict()


def test_missing_positions_not_hidden_by_packing_base_positions():
    draft = MTPDraftModel(config())
    weight = torch.randn(64, 32)
    draft.set_lm_head_weight(weight)
    model = MTPModel(draft, length=1)
    kwargs = dict(
        input_ids=torch.arange(8)[None],
        attention_mask=torch.ones(1, 8),
        target_hidden_states=torch.randn(1, 8, 32),
        target_lm_head_weight=weight,
        loss_mask=torch.ones(1, 8),
        hidden_states=torch.randn(1, 8, 32),
        base_position_ids=torch.arange(8)[None],
    )
    with pytest.raises(ValueError, match="prepared position_ids"):
        model(**kwargs)


def test_mtp_rope_rejects_unknown_scaling_and_bad_sections():
    with pytest.raises(ValueError, match="rope_type"):
        MRotaryEmbedding(64, {**ROPE, "rope_type": "yarn"})
    with pytest.raises(ValueError, match="mrope_section"):
        MRotaryEmbedding(64, {**ROPE, "mrope_section": [12, 10, 9]})


def test_mla_attention_matches_one_dimensional_text_path():
    torch.manual_seed(11)
    cfg = config()
    mr = MTPMLAAttention(cfg)
    cfg.rope_parameters = None
    one = MTPMLAAttention(cfg)
    one.load_state_dict(mr.state_dict())
    h = torch.randn(1, 8, 32)
    p = torch.arange(8)[None]
    args = dict(
        hidden_states=h, attention_mask=torch.ones(1, 8), use_cache=True, return_index=True
    )
    a = mr(**args, position_ids=p[:, None].expand(-1, 3, -1))
    b = one(**args, position_ids=p)
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    assert torch.equal(a[-1][0], b[-1][0])
