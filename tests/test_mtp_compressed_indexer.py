"""Hy4 flash compressed (two-level) indexer inside the tilelang MTP attention (GPU-only).

A small random layer with ``group_topk=16`` / ``ratio=4`` (64 candidates) makes
group selection active at a few hundred tokens. These tests cover how the attention
uses the selection: document packing and the TTT chain that reuses the step-0 index.
"""

import pytest
import torch

from angelspec.models.draft.mtp import MTPConfig, MTPMLAAttention
from angelspec.models.ops.tilelang_dsa import tilelang_dsa_available

pytestmark = pytest.mark.skipif(
    not tilelang_dsa_available(), reason="needs CUDA and the TileLang kernels"
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 512
CANDIDATES = 64


def config(**overrides):
    base = dict(
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
        learnable_sink_init=0.0,
        sandwich_norm=True,
        attention_type="mla",
        dsa_backend="tilelang",
        index_topk=CANDIDATES,
        index_head_dim=128,
        index_n_heads=8,
        use_compressed_indexer=True,
        compressed_indexer_compress_ratio=4,
        compressed_indexer_num_attention_heads=16,
        compressed_indexer_head_dim=128,
        compressed_indexer_group_topk=CANDIDATES // 4,
        compressed_indexer_candidate_topk=CANDIDATES,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        max_position_embeddings=8192,
        rope_theta=10000.0,
    )
    base.update(overrides)
    return MTPConfig(**base)


def compressed_attention(seed):
    torch.manual_seed(seed)
    attn = MTPMLAAttention(config(), attention_backend="sdpa").to(DEVICE, DTYPE).eval()
    with torch.no_grad():
        attn.indexer.intra_group_ape.normal_()
        attn.learnable_sink_param.normal_()
    return attn


def ones(seq):
    return torch.ones(1, seq, device=DEVICE, dtype=DTYPE)


def assert_close(a, b, tag):
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / b.norm()).item()
    assert cos > 0.999 and rel < 2e-2, (tag, cos, rel)


@torch.no_grad()
def test_packed_documents_select_and_attend_like_separate_documents():
    attn = compressed_attention(seed=0)
    # 202 is not a multiple of the group size, so the second document's groups do
    # not line up with groups of the packed sequence; both are long enough for the
    # 15 recalled groups to be a real choice.
    lengths = [202, 150]
    total = sum(lengths)
    hidden = torch.randn(1, total, HIDDEN, device=DEVICE, dtype=DTYPE)
    pos = torch.cat([torch.arange(n) for n in lengths]).to(DEVICE)[None]
    doc = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)]).to(DEVICE)[None]

    packed_out, _, _, (packed_index, _) = attn(
        hidden,
        attention_mask=ones(total),
        position_ids=pos,
        use_cache=True,
        ctx_doc_ids=doc,
        return_index=True,
    )
    start = 0
    for n in lengths:
        out, _, _, (index, _) = attn(
            hidden[:, start : start + n],
            attention_mask=ones(n),
            position_ids=pos[:, start : start + n],
            use_cache=True,
            return_index=True,
        )
        shifted = torch.where(index >= 0, index + start, index)
        assert torch.equal(packed_index[:, start : start + n], shifted)
        assert_close(packed_out[:, start : start + n], out, f"document at {start}")
        start += n
    # Selection was a real choice: the last row of each document sees only part of it.
    for end, n in ((lengths[0], lengths[0]), (total, lengths[1])):
        assert (packed_index[0, end - 1] >= 0).sum() < n


@torch.no_grad()
def test_ttt_chain_matches_dense_attention_when_the_index_covers_the_context():
    fast = compressed_attention(seed=1)
    # torch backend with index_topk above the sequence: plain dense causal TTT.
    dense = (
        MTPMLAAttention(
            config(use_compressed_indexer=False, dsa_backend="torch", index_topk=256),
            attention_backend="sdpa",
        )
        .to(DEVICE, DTYPE)
        .eval()
    )
    missing, unexpected = dense.load_state_dict(fast.state_dict(), strict=False)
    assert all(k.startswith("indexer.") for k in missing)
    assert all(k.startswith("compressed_indexer.") for k in unexpected)

    seq = 60  # fewer complete groups than recalled groups: every key is a candidate
    pos = torch.arange(seq, device=DEVICE)[None]
    steps = [torch.randn(1, seq, HIDDEN, device=DEVICE, dtype=DTYPE) for _ in range(3)]
    causal = torch.zeros(1, 1, seq, seq, device=DEVICE, dtype=DTYPE).masked_fill(
        torch.ones(seq, seq, device=DEVICE, dtype=torch.bool).triu(1), torch.finfo(DTYPE).min
    )

    def chain(attn, mask):
        out, keys, values, index = attn(
            steps[0], attention_mask=mask, position_ids=pos, use_cache=True, return_index=True
        )
        outs = [out]
        for hidden in steps[1:]:
            out, keys, values = attn(
                hidden,
                cache_keys=keys,
                cache_values=values,
                attention_mask=mask,
                position_ids=pos,
                use_cache=True,
                precomputed_index=index,
            )
            outs.append(out)
        return outs, index

    fast_outs, (index, _) = chain(fast, ones(seq))
    dense_outs, _ = chain(dense, causal)
    for row in range(seq):
        assert set(index[0, row].tolist()) - {-1} == set(range(row + 1))
    for step, (a, b) in enumerate(zip(fast_outs, dense_outs)):
        assert_close(a, b, f"ttt step {step}")


def recorded_selections(attn):
    calls = []
    select = attn.indexer.sparse_indices

    def recorded(*args, **kwargs):
        selection = select(*args, **kwargs)
        calls.append((kwargs["precomputed_keys"], selection))
        return selection

    attn.indexer.sparse_indices = recorded
    return calls


@torch.no_grad()
def test_unshared_ttt_steps_select_their_own_groups_against_the_step0_keys():
    shared = compressed_attention(seed=2)
    unshared = MTPMLAAttention(config(dsa_index_share=False), attention_backend="sdpa")
    unshared = unshared.to(DEVICE, DTYPE).eval()
    unshared.load_state_dict(shared.state_dict())
    shared_calls, unshared_calls = recorded_selections(shared), recorded_selections(unshared)
    seq = 300
    args = dict(
        attention_mask=ones(seq),
        position_ids=torch.arange(seq, device=DEVICE)[None],
        use_cache=True,
    )
    steps = [torch.randn(1, seq, HIDDEN, device=DEVICE, dtype=DTYPE) for _ in range(3)]

    out, ck, cv, keys0 = unshared(steps[0], return_index=True, **args)
    ref, rk, rv, index0 = shared(steps[0], return_index=True, **args)
    assert_close(out, ref, "step 0")
    # Unshared hands later steps the step-0 key inputs instead of the selection.
    assert keys0.shape[:2] == (1, seq)
    assert torch.equal(unshared_calls[0][1][0], index0[0])

    changed = False
    for step in steps[1:]:
        out, ck, cv = unshared(
            step, cache_keys=ck, cache_values=cv, precomputed_index=keys0, **args
        )
        selection = unshared_calls[-1][1]
        # The shared path applying this step's own selection reproduces the step.
        ref, rk, rv = shared(
            step, cache_keys=rk, cache_values=rv, precomputed_index=selection, **args
        )
        assert_close(out, ref, "own selection")
        changed |= not torch.equal(selection[0], index0[0])
    assert all(keys is keys0 for keys, _ in unshared_calls)
    assert len(unshared_calls) == 3 and len(shared_calls) == 1
    assert changed
