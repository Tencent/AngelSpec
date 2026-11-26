"""DeepSeek-V4.1 target reuse and a dsv41_mla DFly2 drafter, against vLLM's selection and
kernels, the checkpoint, and a training step through the wrapper."""

import os

import pytest
import torch
import torch.nn as nn

import angelspec.models.target.dsv41_target as dsv41_target
from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.models.draft.dsv41_attentive_correction import init_from_attention
from angelspec.models.draft.dsv41_mla import (
    DSV41DraftAttention,
    apply_rope,
    rope_cos_sin,
)
from angelspec.models.target.dsv41_target import (
    DSV41TargetReuse,
    index_logits,
    kv_record,
    rms_norm,
    select_candidate_blocks,
)

H, V = 64, 128
TARGET_PATH = os.environ.get("DSV41_TARGET_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
YARN = dict(original_seq_len=65536, factor=16.0, beta_fast=32.0, beta_slow=1.0)


def _target_config(**kw):
    """Layers 2-3 compress 2:1 (source 2), 4-7 read layer 4's 1:1 KV; index sources 2, 4,
    6, layer 4 proposing 3 blocks of 4 to layer 6."""
    text = dict(
        model_type="deepseek_v41_text",
        hidden_size=H,
        num_hidden_layers=8,
        num_attention_heads=2,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=8,
        compress_ratios=[0, 0, 2, 2, 1, 1, 1, 1, 0],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4, 6],
        candidate_source_layer_id=4,
        candidate_topk_blocks=3,
        candidate_block_size=4,
        sliding_window=6,
        rms_norm_eps=1e-6,
        compress_rope_theta=160000,
        rope_theta=10000,
        rope_scaling=dict(
            rope_type="yarn",
            factor=16,
            beta_fast=32,
            beta_slow=1,
            original_max_position_embeddings=65536,
        ),
    )
    text.update(kw)
    return {"model_type": "deepseek_v41", "text_config": text}


def _reuse(kv_layers=(5, 6), index_layers=(5, 6), topk=8, **target):
    """In bf16, as from_pretrained loads it."""
    torch.manual_seed(0)
    reuse = DSV41TargetReuse(_target_config(**target), kv_layers, index_layers, topk)
    with torch.no_grad():
        for name, param in reuse.named_parameters():
            if name.endswith("norm") and not name.endswith("attn_norm"):
                param.uniform_(0.5, 1.5)
    return reuse.to(torch.bfloat16)


def _inputs(reuse, length, seed=1):
    """Attention inputs of every layer the reuse reads: attn_norm'd, as captured."""
    g = torch.Generator().manual_seed(seed)
    return {
        layer: rms_norm(torch.randn(1, length, H, generator=g) * 3, part.attn_norm, reuse.eps)
        for layer, part in ((int(k), p) for k, p in reuse.parts.items())
    }


def _reference_picks(reuse, layer, hidden, anchors, doc_ids):
    """vLLM's picks for anchor - 1, one row at a time: the index source's best over
    [doc_start, anchor), past the candidate source only in its best blocks (newest always in)."""
    source = reuse.index_source[layer]
    candidate = reuse.candidate_source[source]
    _, keys, scale = reuse._source(
        reuse._text_kv_source[source], hidden[reuse._text_kv_source[source]], doc_ids
    )
    length = hidden[source].shape[1]
    starts = dsv41_target.document_starts(doc_ids, 1, length, "cpu")[0]
    positions = torch.arange(length) - starts
    cos, sin = rope_cos_sin(positions, reuse.rope_dim, reuse.rope_theta, **reuse.rope_yarn)
    size = reuse.candidate_block_size

    def scores(at, r, start, end):
        q, w = reuse._queries(at, hidden[at][0, r : r + 1], cos[r : r + 1], sin[r : r + 1])
        return index_logits(q, w, keys[0], scale[0], torch.tensor([start]), torch.tensor([end]))[0]

    picks = []
    for anchor in anchors[0].tolist():
        start = int(starts[anchor])
        if anchor == start:
            picks.append(set())
            continue
        r = anchor - 1
        allowed = list(range(start, anchor))
        if candidate is not None:
            block_scores = {}
            cand = scores(candidate, r, start, anchor)
            for s in allowed:
                b = (s - start) // size
                block_scores[b] = max(block_scores.get(b, float("-inf")), float(cand[s]))
            block_scores[(anchor - 1 - start) // size] = float("inf")
            kept = sorted(block_scores, key=lambda b: -block_scores[b])[: reuse.candidate_blocks]
            allowed = [s for s in allowed if (s - start) // size in kept]
        own = scores(source, r, start, anchor)
        ranked = sorted(allowed, key=lambda s: -float(own[s]))
        picks.append(set(ranked[: reuse.topk]))
    return picks


@pytest.mark.parametrize("packed", [False, True])
def test_picks_follow_vllms_two_level_selection(packed):
    reuse = _reuse()
    length = 64
    hidden = _inputs(reuse, length)
    doc_ids = torch.tensor([[0] * 30 + [1] * 34]) if packed else None
    anchors = torch.tensor([[1, 5, 12, 20, 29, 30, 31, 45, 63]])
    for layer in (5, 6):
        got = reuse.candidates(layer, hidden, anchors, doc_ids)
        want = _reference_picks(reuse, layer, hidden, anchors, doc_ids)
        assert [set(row[row >= 0].tolist()) for row in got[0]] == want, layer
    # Layer 6 picks within layer 4's blocks; past 12 visible keys that leaves out some.
    full = _reuse(candidate_topk_blocks=64)
    unrestricted = full.candidates(6, hidden, anchors, doc_ids)
    restricted = reuse.candidates(6, hidden, anchors, doc_ids)
    assert not torch.equal(unrestricted, restricted)


def test_the_context_is_the_window_then_the_kv_sources_global_latents():
    reuse = _reuse()
    hidden = _inputs(reuse, 20)
    latent = reuse.context_latent(5, hidden, None)
    assert latent.shape == (1, 40, 512) and latent.dtype == torch.bfloat16
    cos, sin = rope_cos_sin(torch.arange(20)[None], 64, 160000.0, **YARN)
    torch.testing.assert_close(
        latent[:, :20], reuse.window_latents(5, hidden[5], cos, sin), rtol=0, atol=0
    )
    part = reuse.parts["4"]
    score = hidden[4].float() @ part.compressor_wkv.weight.float().t()
    want = kv_record(rms_norm(score, part.compressor_norm, reuse.eps), cos, sin)
    torch.testing.assert_close(latent[:, 20:], want, rtol=0, atol=0)


def _draft_config(**kw):
    fields = dict(
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=1024,
        num_target_layers=3,
        target_hidden_size=H,
        target_num_hidden_layers=8,
        target_layer_ids=[3, 4, 5],
        mask_token_id=V - 1,
        attention_type="dsv41_mla",
        num_attention_heads=2,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        rope_theta=160000.0,
        original_seq_len=65536,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        context_local_window=6,
        target_index_layer=[5, 6],
        target_index_topk=8,
        target_kv_layers=[5, 6],
    )
    fields.update(kw)
    return DFlashConfig(**fields)


def test_dfly2_trains_on_the_targets_kv_and_picks():
    torch.manual_seed(0)
    config = _draft_config(
        dsa_backend="grouped",
        hc_mult=4,
        target_hc_init_layers=[5, 6],
        enable_hidden_correction=True,
        attentive_correction=True,
    )
    draft = DFlashDraftModel(config).float()
    draft.freeze_embedding()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        # As target_hc_init_layers leaves them: mixes that tell the copies apart.
        for layer in draft.layers:
            for name in dsv41_target.HC_PARAMS:
                param = getattr(layer, name)
                param.copy_(torch.randn(param.shape, generator=g) * 0.3)
        correction = draft.hidden_correction
        init_from_attention(correction.attention, draft.layers[-1].self_attn)
        for weight in (correction.attention.wo_b.weight, correction.down_proj.weight):
            weight.copy_(torch.randn(weight.shape, generator=g) * 0.05)
    wrapper = DFlashModel(draft_model=draft, block_size=4, num_anchors=3, causal_attn=False)
    reuse = _reuse()
    wrapper.attach_target_indexers(reuse.indexers())
    wrapper.attach_target_kv_latents(reuse.kv_modules())
    hidden = _inputs(reuse, 48, seed=2)
    loss = wrapper(
        input_ids=torch.randint(0, 100, (1, 48), generator=g),
        hidden_states_list=[hidden[4], hidden[5], hidden[6]],
        loss_mask=torch.ones(1, 48),
        lm_head_weight=torch.randn(V, H, generator=g),
        injected_anchors=torch.tensor([[7, 21, 40]]),
    )[0]
    loss.backward()
    assert torch.isfinite(loss)
    missing = [n for n, p in wrapper.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    for layer in draft.layers:
        for name in dsv41_target.HC_PARAMS:
            assert float(getattr(layer, name).grad.abs().sum()) > 0, name


def test_the_capture_hands_over_the_attention_input(monkeypatch):
    dsv41 = pytest.importorskip("vllm.models.deepseek_v41.nvidia.model")
    from angelspec.inference.engine import vllm_runtime_patch as patch

    cls = dsv41.DeepseekV4DecoderLayer

    def fake_forward(self, x, positions, *, capture_previous_aux=False):
        out = self.attn(positions, x * 2, None)
        return out, None, None, None, None, (x.mean(0) if capture_previous_aux else None)

    monkeypatch.setattr(cls, "forward", fake_forward)
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_angelspec_attn_input", False, raising=False)
    assert patch.apply_dsv41_attention_input_capture()

    class Attn(nn.Module):
        def forward(self, positions, x, _):
            return x + 1

    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.attn = Attn()
    layer.use_sequence_parallel = False
    layer._angelspec_layer_index = 5
    x, positions = torch.randn(3, 4), torch.arange(3)
    monkeypatch.setenv(patch.DSV41_ATTN_INPUT_AUX_ENV, "5,9")
    torch.testing.assert_close(layer(x, positions, capture_previous_aux=True)[-1], x * 2)
    assert layer(x, positions)[-1] is None
    monkeypatch.setenv(patch.DSV41_ATTN_INPUT_AUX_ENV, "9")
    torch.testing.assert_close(layer(x, positions, capture_previous_aux=True)[-1], x.mean(0))
    monkeypatch.setenv(patch.DSV41_ATTN_INPUT_AUX_ENV, "5")
    layer.use_sequence_parallel = True
    with pytest.raises(NotImplementedError):
        layer(x, positions, capture_previous_aux=True)


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def vllm_rope():
    pytest.importorskip("vllm")
    if not os.path.isdir(TARGET_PATH):
        pytest.skip("no DeepSeek-V4.1 checkpoint")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
    from vllm.transformers_utils.config import get_config

    text = get_config(TARGET_PATH, trust_remote_code=True)
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        return build_deepseek_v4_rope(
            text,
            head_dim=512,
            rope_head_dim=64,
            max_position_embeddings=text.max_position_embeddings,
            compress_ratio=1,
        )


@_cuda
def test_rope_table_is_vllms(vllm_rope):
    positions = torch.arange(140000, device="cuda")
    cos, sin = rope_cos_sin(positions, 64, 160000.0, **YARN)
    assert torch.equal(torch.cat([cos, sin], -1), vllm_rope.cos_sin_cache[:140000])


def _decode_record(cache, slots, block):
    page = cache.view(cache.shape[0], -1)[slots // block]
    row, rows = slots % block, torch.arange(len(slots), device=slots.device)
    values = page[:, : block * 576].view(-1, block, 576)[rows, row]
    scales = page[:, block * 576 : block * 584].view(-1, block, 8)[rows, row]
    nope = values[:, :448].view(torch.float8_e4m3fn).float().view(-1, 7, 64)
    nope = nope * torch.exp2(scales[:, :7].float() - 127).unsqueeze(-1)
    return torch.cat(
        [nope.view(-1, 448), values[:, 448:].contiguous().view(torch.bfloat16).float()], -1
    )


@_cuda
def test_arithmetic_is_vllms(vllm_rope):
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks as vllm_select,
    )
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        fused_save_compress_norm,
        rope_quant_insert,
    )
    from vllm.models.deepseek_v41.common.ops.indexer_k_store import (
        indexer_k_norm_rope_store,
    )
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    torch.manual_seed(0)
    dev, tokens, block, eps = "cuda", 2048, 64, 1e-20
    table = vllm_rope.cos_sin_cache
    positions = torch.randint(0, 140000, (tokens,), device=dev)
    cos, sin = rope_cos_sin(positions, 64, 160000.0, **YARN)
    slots = torch.randperm(tokens, device=dev)
    pages = tokens // block

    def mismatch(a, b):
        return float((a.float() != b.float()).float().mean())

    # Layer 20's compressor and the KV records (global through Triton, window through C++).
    x = (torch.randn(tokens, 5120, device=dev) * 2).to(torch.bfloat16)
    wkv = (torch.randn(512, 5120, device=dev) * 0.02).to(torch.bfloat16)
    norm = (torch.rand(512, device=dev) + 0.5).to(torch.bfloat16)
    score = torch.mm(x, wkv.T, out_dtype=torch.float32)
    assert torch.equal(dsv41_target._mm_fp32(x, wkv), score)
    latent = torch.empty(tokens, 512, device=dev, dtype=torch.bfloat16)
    fused_save_compress_norm(score, positions, None, slots, None, None, norm, eps, 1, latent)
    assert mismatch(rms_norm(score, norm, eps), latent) < 1e-4
    cache = torch.zeros(pages, block, 584, dtype=torch.uint8, device=dev)
    rope_quant_insert(latent, positions, table, cache, slots, 1)
    assert mismatch(kv_record(latent, cos, sin), _decode_record(cache, slots, block)) < 1e-4
    window = torch.zeros_like(cache)
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(
        latent, window, slots, positions, table, block, None, False
    )
    assert mismatch(kv_record(latent, cos, sin), _decode_record(window, slots, block)) < 1e-4

    k_pre = (torch.randn(tokens, 128, device=dev) * 0.7).to(torch.bfloat16)
    k_norm = (torch.rand(128, device=dev) + 0.5).to(torch.bfloat16)
    k_cache = torch.zeros(pages, block, 132, dtype=torch.uint8, device=dev)
    indexer_k_norm_rope_store(k_pre, positions, table, k_norm, eps, k_cache, slots, 1, False)
    page = k_cache.view(pages, -1)[slots // block]
    rows, row = torch.arange(tokens, device=dev), slots % block
    k_vals = page[:, : block * 128].view(-1, block, 128)[rows, row].view(torch.float8_e4m3fn)
    k_scale = page[:, block * 128 : block * 132].contiguous().view(-1, block, 4)[rows, row]
    k_scale = k_scale.contiguous().view(torch.float32).squeeze(-1)
    keys, scale = dsv41_target.index_key_fp8(apply_rope(rms_norm(k_pre, k_norm, eps), cos, sin))
    assert torch.equal(keys.float(), k_vals.float()) and torch.equal(scale, k_scale)

    q = (torch.randn(tokens, 32, 128, device=dev) * 1.5).to(torch.bfloat16)
    w = torch.randn(tokens, 32, device=dev).to(torch.bfloat16)
    q8, w_out = fused_indexer_q_rope_quant(
        positions, q, table, w, 128**-0.5, 32**-0.5, use_fp4=False, weights_out_dtype=torch.float32
    )
    ours_q, q_scale = dsv41_target.index_query_fp8(
        apply_rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2)).float()
    )
    assert mismatch(ours_q, q8) < 1e-5
    assert torch.equal(w.float() * q_scale * (128**-0.5 * 32**-0.5), w_out)

    starts = torch.randint(0, 300, (256,), device=dev)
    ends = starts + torch.randint(1, tokens - 300, (256,), device=dev)
    spans = (ends - starts).tolist()
    packed = fp8_fp4_mqa_logits(
        (q8[:256].contiguous(), None),
        (k_vals, k_scale),
        w_out[:256].contiguous(),
        starts.int(),
        ends.int(),
        max(spans),
    )
    ref = torch.full((256, tokens), float("-inf"), device=dev)
    for row, (start, span) in enumerate(zip(starts.tolist(), spans)):
        ref[row, start : start + span] = packed[row, :span]
    ours = index_logits(q8[:256], w_out[:256], k_vals, k_scale, starts, ends)
    assert torch.equal(ours, ref)
    out = torch.empty(256, 64, dtype=torch.int32, device=dev)
    vllm_select(ref, starts.int(), ends.int(), 64, 8, out)
    n_blocks = -(-tokens // 8)
    want = torch.zeros(256, n_blocks + 1, dtype=torch.bool, device=dev)
    want.scatter_(1, torch.where(out >= 0, out.long(), n_blocks), True)
    assert torch.equal(select_candidate_blocks(ref, starts, ends, 8, 64), want[:, :n_blocks])


@pytest.mark.skipif(not os.path.isdir(TARGET_PATH), reason="no DeepSeek-V4.1 checkpoint")
def test_attention_and_mixes_copy_the_checkpoint():
    config = _draft_config(
        hidden_size=5120,
        num_attention_heads=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        rms_norm_eps=1e-20,
        target_hidden_size=5120,
    )
    layer = nn.Module()
    layer.input_layernorm = nn.Module()
    layer.input_layernorm.weight = nn.Parameter(torch.zeros(5120))
    layer.self_attn = DSV41DraftAttention(config)
    mix, width = 6 * 4, 4 * 5120
    for kind in ("attn", "ffn"):
        setattr(layer, f"hc_{kind}_fn", nn.Parameter(torch.zeros(mix, width)))
        setattr(layer, f"hc_{kind}_base", nn.Parameter(torch.zeros(mix)))
        setattr(layer, f"hc_{kind}_scale", nn.Parameter(torch.zeros(3)))
    draft = nn.Module()
    draft.layers = nn.ModuleList([layer])
    dsv41_target.init_attention_from_dsv41_target(draft, TARGET_PATH, [28])
    dsv41_target.init_hc_from_dsv41_target(draft, TARGET_PATH, [28])
    reader = dsv41_target._TargetReader(TARGET_PATH)
    assert torch.equal(
        layer.input_layernorm.weight, reader.tensor("layers.28.attn_norm.weight").float()
    )
    assert torch.equal(layer.self_attn.attn_sink, reader.tensor("layers.28.attn.attn_sink"))
    assert torch.equal(layer.self_attn.wo_a.weight, reader.dense("layers.28.attn.wo_a"))
    for name in dsv41_target.HC_PARAMS:
        assert torch.equal(getattr(layer, name), reader.tensor(f"layers.28.{name}")), name
