import math

import torch

import angelspec.models.draft.mtp as mtp_module
from angelspec.models.draft.mtp import MTPConfig, MTPMLAAttention
from angelspec.models.ops.sparse_indices import (
    append_diagonal_kv_indices,
    build_contiguous_causal_ranges,
)
from angelspec.models.ops.sparse_mla import torch_lightning_indexer


def test_build_contiguous_causal_ranges_plain():
    starts, ends = build_contiguous_causal_ranges(6, query_offset=2, query_len=3)
    torch.testing.assert_close(starts, torch.tensor([[0, 0, 0]], dtype=torch.int32))
    torch.testing.assert_close(ends, torch.tensor([[3, 4, 5]], dtype=torch.int32))


def test_build_contiguous_causal_ranges_packed_and_padding():
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    starts, ends = build_contiguous_causal_ranges(
        8,
        doc_ids=doc_ids,
        valid_mask=valid,
    )
    torch.testing.assert_close(
        starts,
        torch.tensor([[0, 0, 0, 3, 3, 3, 6, 6]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        ends,
        torch.tensor([[1, 2, 3, 4, 5, 6, 6, 6]], dtype=torch.int32),
    )


def test_append_diagonal_kv_indices():
    base = torch.tensor(
        [
            [
                [0, 1, -1, -1],
                [0, 1, 2, -1],
                [0, 1, 2, 3],
            ]
        ],
        dtype=torch.int32,
    )
    out = append_diagonal_kv_indices(
        base,
        seq_len=3,
        num_cached_steps=3,
        alignment=8,
    )
    assert out.shape == (1, 3, 1, 8)
    torch.testing.assert_close(out[0, :, 0, 4], torch.tensor([3, 4, 5], dtype=torch.int32))
    torch.testing.assert_close(out[0, :, 0, 5], torch.tensor([6, 7, 8], dtype=torch.int32))
    torch.testing.assert_close(out[..., 6:], torch.full_like(out[..., 6:], -1))


def test_torch_lightning_indexer_scores_within_each_range():
    torch.manual_seed(5)
    q, k, w = torch.randn(7, 3, 8), torch.randn(20, 8), torch.randn(7, 3)
    starts = torch.tensor([0, 0, 4, 10, 2, 0, 19])
    ends = torch.tensor([0, 3, 20, 12, 18, 20, 20])
    scores, indices = torch_lightning_indexer(q, k, w, starts, ends, 5)
    logits = (torch.einsum("qhd,kd->qhk", q, k).relu() * w[..., None]).sum(1)
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        picked = indices[row][indices[row] >= 0].long()
        assert len(picked) == min(5, end - start)
        assert bool(((picked >= start) & (picked < end)).all())
        if len(picked):
            expected = logits[row, start:end].topk(len(picked)).values
            torch.testing.assert_close(scores[row, : len(picked)], expected)
            torch.testing.assert_close(logits[row, picked], expected)
        assert bool((scores[row, len(picked) :] == -torch.inf).all())


def test_absorbed_mla_matches_expanded_qk_and_value():
    torch.manual_seed(7)
    config = MTPConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=16,
        qk_rope_head_dim=64,
        v_head_dim=16,
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
        attention_type="mla",
        index_topk=4,
        index_head_dim=128,
        index_n_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
    )
    attention = MTPMLAAttention(config).double()
    hidden = torch.randn(1, 4, 32, dtype=torch.float64)
    position_ids = torch.arange(4).unsqueeze(0)

    q_nope, q_pe, k_nope, k_pe, values, _ = attention._project(hidden)
    cos, sin = attention._cos_sin(q_nope, position_ids, 4)
    q_expanded, k_expanded = attention._assemble(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        cos,
        sin,
    )

    q_nope_a, q_pe_a, kv_compressed, k_pe_a, _ = attention._project_absorbed(hidden)
    q_absorbed, kv_absorbed, w_vc = attention._assemble_absorbed(
        q_nope_a,
        q_pe_a,
        kv_compressed,
        k_pe_a,
        cos,
        sin,
    )

    logits_expanded = torch.einsum("bhsd,bhtd->bhst", q_expanded, k_expanded)
    logits_absorbed = torch.einsum(
        "bshd,btd->bhst",
        q_absorbed,
        kv_absorbed[:, :, 0],
    )
    torch.testing.assert_close(logits_absorbed, logits_expanded, rtol=1e-10, atol=1e-10)

    probs = torch.softmax(torch.randn(1, 2, 4, 4, dtype=torch.float64), dim=-1)
    expanded_out = torch.einsum("bhst,bhtd->bhsd", probs, values).transpose(1, 2)
    latent_out = torch.einsum("bhst,btm->bshm", probs, kv_compressed)
    absorbed_out = torch.einsum("bshm,hdm->bshd", latent_out, w_vc)
    torch.testing.assert_close(absorbed_out, expanded_out, rtol=1e-10, atol=1e-10)


def _reference_sparse_mla(q, kv, indices, *, scaling, sink=None, **kwargs):
    batch, seq_len, heads, _ = q.shape
    topk = indices.shape[-1]
    safe = indices[:, :, 0].long().clamp_min(0)
    batch_idx = torch.arange(batch).view(batch, 1, 1).expand(-1, seq_len, topk)
    gathered = kv[batch_idx, safe, 0]
    q_main, q_tail = q[..., :512], q[..., 512:]
    k_main, k_tail = gathered[..., :512], gathered[..., 512:]
    logits = (
        torch.einsum("bshd,bskd->bshk", q_main.float(), k_main.float())
        + torch.einsum("bshd,bskd->bshk", q_tail.float(), k_tail.float())
    ) * scaling
    logits = logits.masked_fill(indices[:, :, 0, None] < 0, float("-inf"))
    if sink is not None:
        logits = torch.cat(
            [logits, sink.float().view(1, 1, heads, 1).expand(batch, seq_len, -1, -1)],
            dim=-1,
        )
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("bshk,bskd->bshd", probs[..., :topk], k_main.float())
    lse = torch.logsumexp(logits[..., :topk], dim=-1) * math.log2(math.e)
    return out.to(q.dtype), lse


def test_tilelang_cached_path_contract(monkeypatch):
    torch.manual_seed(11)
    config = MTPConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=16,
        qk_rope_head_dim=64,
        v_head_dim=16,
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
        attention_type="mla",
        index_topk=4,
        index_head_dim=128,
        index_n_heads=2,
        dsa_backend="tilelang",
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
    )
    attention = MTPMLAAttention(config).bfloat16()
    monkeypatch.setattr(mtp_module, "tilelang_sparse_mla", _reference_sparse_mla)

    def fake_indices(hidden_states, *args, **kwargs):
        seq_len = hidden_states.shape[1]
        indices = torch.full(
            (hidden_states.shape[0], seq_len, config.index_topk),
            -1,
            dtype=torch.int32,
        )
        for i in range(seq_len):
            indices[:, i, : i + 1] = torch.arange(i + 1, dtype=torch.int32)
        return indices, (indices < 0).any(-1, keepdim=True)

    monkeypatch.setattr(attention.indexer, "sparse_indices", fake_indices)
    position_ids = torch.arange(4).unsqueeze(0)
    mask = torch.ones(1, 4)
    hidden0 = torch.randn(1, 4, 32, dtype=torch.bfloat16, requires_grad=True)
    out0, cache0, values0, index = attention(
        hidden0,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=True,
        return_index=True,
    )
    assert out0.shape == (1, 4, 32)
    assert cache0.shape == (1, 4, 1, 576)
    assert values0.numel() == 0
    assert index[0].shape == (1, 4, 4)

    hidden1 = torch.randn(1, 4, 32, dtype=torch.bfloat16, requires_grad=True)
    out1, cache1, values1 = attention(
        hidden1,
        cache_keys=cache0,
        cache_values=values0,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=True,
        precomputed_index=index,
    )
    assert out1.shape == (1, 4, 32)
    assert cache1.shape == (1, 8, 1, 576)
    assert values1.numel() == 0
    (out0.float().sum() + out1.float().sum()).backward()
    assert hidden0.grad is not None
    assert hidden1.grad is not None
