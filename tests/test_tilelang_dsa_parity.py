"""TileLang DSA backend vs torch backend numerical parity (GPU-only).

Guards the hy4 long-context MTP path: the tilelang cached forward (lightning indexer +
sparse MLA) must agree with the torch reference (MTPIndexer chunked-topk +
_sparse_cached_attn / _dense_block0) on identical inputs. The two paths interpret
``attention_mask`` differently -- tilelang reads it as a validity mask (causality comes
from build_contiguous_causal_ranges), the torch path reads an additive causal mask -- so
each is fed its own form.
"""

import pytest
import torch

from angelspec.models.draft.mtp import MTPConfig, MTPMLAAttention
from angelspec.models.ops.tilelang_dsa import tilelang_dsa_available

pytestmark = pytest.mark.skipif(
    not tilelang_dsa_available(), reason="TileLang DSA kernels require CUDA and tilelang"
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _make_pair(index_topk, **overrides):
    base = dict(
        hidden_size=512,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
        learnable_sink_init=0.0,
        attention_type="mla",
        index_topk=index_topk,
        index_head_dim=128,
        index_n_heads=8,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        **overrides,
    )
    torch_attn = (
        MTPMLAAttention(MTPConfig(dsa_backend="torch", **base), attention_backend="sdpa")
        .to(DEVICE, DTYPE)
        .eval()
    )
    fast_attn = (
        MTPMLAAttention(MTPConfig(dsa_backend="tilelang", **base), attention_backend="sdpa")
        .to(DEVICE, DTYPE)
        .eval()
    )
    fast_attn.load_state_dict(torch_attn.state_dict())
    # randomize the learnable sink so the sink fusion is actually exercised
    with torch.no_grad():
        s = torch.randn(base["num_attention_heads"], device=DEVICE)
        torch_attn.learnable_sink_param.copy_(s)
        fast_attn.learnable_sink_param.copy_(s)
    return torch_attn, fast_attn


def _additive_causal(seq):
    m = torch.zeros(1, 1, seq, seq, device=DEVICE, dtype=DTYPE)
    fut = torch.triu(torch.ones(seq, seq, device=DEVICE, dtype=torch.bool), 1)
    return m.masked_fill(fut[None, None], torch.finfo(DTYPE).min)


def _agree(a, b, tag, cos_min=0.99, rel_max=5e-2):
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / (a.norm() + 1e-6)).item()
    print(f"  [{tag}] cos={cos:.5f} rel_fro={rel:.4f}")
    assert cos >= cos_min and rel <= rel_max, f"{tag}: cos={cos} rel={rel}"


def test_parity_full_causal():
    """seq <= index_topk: both paths reduce to full causal attention + sink."""
    torch.manual_seed(0)
    torch_attn, fast_attn = _make_pair(index_topk=256)
    seq = 128
    h = torch.randn(1, seq, 512, device=DEVICE, dtype=DTYPE)
    pos = torch.arange(seq, device=DEVICE).unsqueeze(0)

    out_t, _, _ = torch_attn(
        h, attention_mask=_additive_causal(seq), position_ids=pos, use_cache=True
    )
    valid = torch.ones(1, seq, device=DEVICE, dtype=DTYPE)
    out_f, _, _ = fast_attn(h, attention_mask=valid, position_ids=pos, use_cache=True)
    _agree(out_t, out_f, "full_causal")


def test_parity_sparse_shared_index():
    """seq > index_topk: compare the sparse attention output on rows with a full
    top-k causal window (row >= topk). The two indexers select identical keys
    there (see test_indexer_selection_overlap), so any diff is the sparse_mla
    kernel vs the torch gather-softmax reference. Short rows (< topk) are excluded:
    torch pads them with real positions + an all_open dense fallback while tilelang
    pads with -1, so their handling legitimately differs."""
    torch.manual_seed(1)
    torch_attn, fast_attn = _make_pair(index_topk=64)
    seq = 256
    topk = 64
    h = torch.randn(1, seq, 512, device=DEVICE, dtype=DTYPE)
    pos = torch.arange(seq, device=DEVICE).unsqueeze(0)

    out_t, _, _ = torch_attn(
        h, attention_mask=_additive_causal(seq), position_ids=pos, use_cache=True
    )
    valid = torch.ones(1, seq, device=DEVICE, dtype=DTYPE)
    out_f, _, _ = fast_attn(h, attention_mask=valid, position_ids=pos, use_cache=True)
    _agree(out_t[:, topk:], out_f[:, topk:], "sparse_full_window")


def test_indexer_selection_overlap():
    """The TileLang lightning indexer's top-k must overlap the torch MTPIndexer top-k."""
    torch.manual_seed(2)
    torch_attn, fast_attn = _make_pair(index_topk=64)
    seq = 256
    h = torch.randn(1, seq, 512, device=DEVICE, dtype=DTYPE)
    pos = torch.arange(seq, device=DEVICE).unsqueeze(0)

    _, _, _, idx_t = torch_attn(
        h,
        attention_mask=_additive_causal(seq),
        position_ids=pos,
        use_cache=True,
        return_index=True,
    )
    _, _, _, idx_f = fast_attn(
        h,
        attention_mask=torch.ones(1, seq, device=DEVICE, dtype=DTYPE),
        position_ids=pos,
        use_cache=True,
        return_index=True,
    )
    ti, fi = idx_t[0], idx_f[0]  # [b, s, topk] int32
    topk = ti.shape[-1]
    inter = 0
    total = 0
    for row in range(topk, seq):  # rows with a full topk causal window
        a = set(ti[0, row].tolist()) - {-1}
        b = set(fi[0, row].tolist()) - {-1}
        inter += len(a & b)
        total += len(a)
    frac = inter / max(total, 1)
    print(f"  [indexer_overlap] mean top-k overlap={frac:.4f}")
    assert frac >= 0.9, f"indexer selection overlap too low: {frac}"


def test_parity_ttt_chain():
    """Multi-step cached (TTT) parity with shared index across steps."""
    torch.manual_seed(3)
    torch_attn, fast_attn = _make_pair(index_topk=256)
    seq = 96
    pos = torch.arange(seq, device=DEVICE).unsqueeze(0)
    hs = [torch.randn(1, seq, 512, device=DEVICE, dtype=DTYPE) for _ in range(3)]

    def run(attn, mask):
        out, ck, cv, idx = attn(
            hs[0], attention_mask=mask, position_ids=pos, use_cache=True, return_index=True
        )
        outs = [out]
        for step in range(1, 3):
            out, ck, cv = attn(
                hs[step],
                cache_keys=ck,
                cache_values=cv,
                attention_mask=mask,
                position_ids=pos,
                use_cache=True,
                precomputed_index=idx,
            )
            outs.append(out)
        return outs

    outs_t = run(torch_attn, _additive_causal(seq))
    outs_f = run(fast_attn, torch.ones(1, seq, device=DEVICE, dtype=DTYPE))
    for step, (a, b) in enumerate(zip(outs_t, outs_f)):
        _agree(a, b, f"ttt_step{step}")


def test_parity_ttt_chain_unshared_index():
    """Multi-step cached (TTT) parity with dsa_index_share=false.

    Step 0 returns the indexer keys; later steps re-select top-k against those
    keys. Both backends must agree on every step's output.
    """
    torch.manual_seed(4)
    torch_attn, fast_attn = _make_pair(index_topk=64, dsa_index_share=False)
    seq = 256
    pos = torch.arange(seq, device=DEVICE).unsqueeze(0)
    hs = [torch.randn(1, seq, 512, device=DEVICE, dtype=DTYPE) for _ in range(3)]

    def run(attn, mask):
        out, ck, cv, keys = attn(
            hs[0], attention_mask=mask, position_ids=pos, use_cache=True, return_index=True
        )
        assert keys.shape == (1, seq, attn.indexer.head_dim)
        outs = [out]
        for step in range(1, 3):
            out, ck, cv = attn(
                hs[step],
                cache_keys=ck,
                cache_values=cv,
                attention_mask=mask,
                position_ids=pos,
                use_cache=True,
                precomputed_index=keys,
            )
            outs.append(out)
        return outs

    outs_t = run(torch_attn, _additive_causal(seq))
    outs_f = run(fast_attn, torch.ones(1, seq, device=DEVICE, dtype=DTYPE))
    for step, (a, b) in enumerate(zip(outs_t, outs_f)):
        _agree(a[:, 64:], b[:, 64:], f"ttt_unshared_step{step}")
