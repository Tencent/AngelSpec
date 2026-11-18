"""The TileLang DSA kernels against their torch references (GPU-only): sparse MLA outputs
and q/kv/sink gradients, with -1 slots, rows with no key, head counts off the kernel's
64-head blocks and NoPE latents; lightning-indexer selections, counts and scores."""

import pytest
import torch

from angelspec.models.ops.sparse_mla import torch_lightning_indexer, torch_sparse_mla
from angelspec.models.ops.tilelang_dsa import (
    tilelang_dsa_available,
    tilelang_lightning_indexer,
    tilelang_sparse_mla,
)

pytestmark = pytest.mark.skipif(
    not tilelang_dsa_available(), reason="TileLang DSA kernels require CUDA and tilelang"
)

DEVICE = "cuda"
EMPTY_ROWS = 2


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _indices(batch, rows, keys, width, generator):
    """Distinct keys in random slots, ``-1`` elsewhere; the first rows list none."""
    indices = torch.full((batch, rows, width), -1, dtype=torch.int32)
    count = int(min(width, keys) * 0.7)
    for b in range(batch):
        for s in range(EMPTY_ROWS, rows):
            slots = torch.randperm(width, generator=generator)[:count]
            indices[b, s, slots] = torch.randperm(keys, generator=generator)[:count].int()
    return indices.to(DEVICE)


@pytest.mark.parametrize(
    "heads,dim,width,with_sink",
    [
        (4, 576, 100, True),
        (40, 576, 100, True),
        (64, 576, 64, False),
        (128, 576, 100, True),
        (16, 512, 130, True),
        (64, 512, 7, False),
    ],
)
def test_sparse_mla_matches_the_torch_reference(heads, dim, width, with_sink):
    g = torch.Generator().manual_seed(heads + dim + width)
    batch, rows, keys = 2, 48, 160
    q = (torch.randn(batch, rows, heads, dim, generator=g) * 0.5).to(DEVICE, torch.bfloat16)
    kv = (torch.randn(batch, keys, 1, dim, generator=g) * 0.5).to(DEVICE, torch.bfloat16)
    sink = torch.randn(heads, generator=g).to(DEVICE) if with_sink else None
    indices = _indices(batch, rows, keys, width, g).unsqueeze(2)
    grad = torch.randn(batch, rows, heads, 512, generator=g).to(DEVICE, torch.bfloat16)

    def run(fn, q, kv, sink, **kw):
        q, kv = q.clone().requires_grad_(), kv.clone().requires_grad_()
        sink = sink.clone().requires_grad_() if sink is not None else None
        out = fn(q, kv, indices, scaling=dim**-0.5, sink=sink, **kw)[0]
        out.backward(grad.to(out.dtype))
        return out, q.grad, kv.grad, sink.grad if sink is not None else None

    got = run(tilelang_sparse_mla, q, kv, sink)
    want = run(torch_sparse_mla, q.float(), kv.float(), sink, d_v=512)
    for name, a, b in zip(("out", "dq", "dkv", "dsink"), got, want):
        if b is None:
            continue
        assert torch.isfinite(a).all(), name
        assert _rel(a, b) < 2e-2, f"{name}: {_rel(a, b)}"
    assert float(got[0][:, :EMPTY_ROWS].detach().abs().max()) == 0.0


@pytest.mark.parametrize(
    "queries,keys,heads,dim,topk,chunk",
    [
        (100, 1000, 32, 128, 64, 0),
        (257, 3000, 32, 128, 2048, 64),
        (64, 500, 64, 128, 100, None),
        (50, 40, 8, 64, 64, 7),
    ],
)
def test_lightning_indexer_matches_the_torch_reference(queries, keys, heads, dim, topk, chunk):
    g = torch.Generator().manual_seed(queries + keys + topk)
    q = torch.randn(queries, heads, dim, generator=g).to(DEVICE, torch.bfloat16)
    k = torch.randn(keys, dim, generator=g).to(DEVICE, torch.bfloat16)
    w = torch.randn(queries, heads, generator=g).to(DEVICE)
    ends = torch.randint(0, keys + 1, (queries,), generator=g)
    starts = (ends - torch.randint(0, keys + 1, (queries,), generator=g)).clamp_min(0)
    starts, ends = starts.to(DEVICE), ends.to(DEVICE)

    scores, indices = tilelang_lightning_indexer(
        q, k, w, starts, ends, topk, query_chunk_size=chunk
    )
    ref_scores, ref_indices = torch_lightning_indexer(q, k, w, starts, ends, topk)
    valid = indices >= 0
    assert torch.equal(valid.sum(-1), (ends - starts).clamp(max=topk))
    assert bool(((indices >= starts[:, None]) & (indices < ends[:, None]) | ~valid).all())
    for row in range(queries):
        ours = set(indices[row].tolist()) - {-1}
        theirs = set(ref_indices[row].tolist()) - {-1}
        # fp32 summation order may swap the keys tied at the top-k boundary.
        assert len(ours ^ theirs) <= 2, row
    torch.testing.assert_close(
        scores.masked_fill(~valid, 0), ref_scores.masked_fill(~valid, 0), rtol=1e-4, atol=1e-4
    )
