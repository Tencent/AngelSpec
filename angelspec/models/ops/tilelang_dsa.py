import math
import os
from functools import lru_cache

import torch
import torch.nn.functional as F

# The sparse-MLA kernel reads values from a 512-wide latent (d_v) and scores q . k over it
# plus 64 RoPE channels.
DSA_LATENT_DIM = 512
DSA_ROPE_DIM = 64

#: The kernels read index lists in blocks; a list's width must be a multiple of this.
DSA_INDEX_ALIGNMENT = 64

#: dtype the DSA kernels are compiled for.
DSA_KERNEL_DTYPE = torch.bfloat16

_INDEXER_KEY_BLOCK = 128


@lru_cache(maxsize=1)
def _kernels():
    from angelspec.models.ops import tilelang_dsa_kernels

    return tilelang_dsa_kernels


def tilelang_dsa_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _kernels()
    except Exception:
        return False
    return True


def _indexer_query_chunk(query_len: int, key_len: int, requested: int | None) -> int:
    if requested is None:
        raw = os.environ.get("ANGELSPEC_DSA_INDEXER_QUERY_CHUNK", "auto")
        if raw == "auto":
            requested = max(1, min(2048, (128 << 20) // max(1, key_len * 4)))
        else:
            try:
                requested = int(raw)
            except ValueError as exc:
                raise ValueError(
                    "ANGELSPEC_DSA_INDEXER_QUERY_CHUNK must be auto or a nonnegative integer"
                ) from exc
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValueError("query_chunk_size must be a nonnegative integer or None")
    return query_len if requested == 0 else min(query_len, requested)


@torch.no_grad()
def tilelang_lightning_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    key_starts: torch.Tensor,
    key_ends: torch.Tensor,
    topk: int,
    *,
    query_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``topk`` keys of ``sum_h weights[q, h] * relu(index_q[q, h] . index_k[k])`` over
    ``key_starts[q] <= k < key_ends[q]``: fp32 scores and int32 indices, ``[Q, topk]`` each,
    ``-inf``/``-1`` past a row's valid keys.

    ``None`` uses ANGELSPEC_DSA_INDEXER_QUERY_CHUNK (default ``auto``: at most 2048 rows and
    approximately 128 MiB of FP32 logits per call). ``0`` scores every row in one call;
    positive values set a row limit.
    """
    if index_q.ndim != 3 or index_k.ndim != 2:
        raise ValueError("Expected index_q [Q,H,D] and index_k [K,D]")
    query_len, heads, dim = index_q.shape
    if dim != index_k.shape[1]:
        raise ValueError("Query/key indexer dimensions differ")
    if weights.shape not in (index_q.shape[:2], (*index_q.shape[:2], 1)):
        raise ValueError("Indexer weights must have shape [Q,H] or [Q,H,1]")
    if key_starts.shape != (query_len,) or key_ends.shape != (query_len,):
        raise ValueError("Key ranges must contain one start/end per query")
    if heads > 128:
        raise ValueError(f"the indexer kernel scores at most 128 heads, got {heads}")
    topk = int(topk)
    key_len = index_k.shape[0]
    chunk = _indexer_query_chunk(query_len, key_len, query_chunk_size)
    device = index_q.device
    scores = torch.full((query_len, topk), -torch.inf, device=device, dtype=torch.float32)
    indices = torch.full((query_len, topk), -1, device=device, dtype=torch.int32)
    if query_len == 0 or key_len == 0 or topk == 0:
        return scores, indices

    index_q = index_q.to(DSA_KERNEL_DTYPE).contiguous()
    index_k = F.pad(index_k.to(DSA_KERNEL_DTYPE), (0, 0, 0, _INDEXER_KEY_BLOCK)).contiguous()
    weights = weights.reshape(query_len, heads).float().contiguous()
    key_starts = key_starts.to(torch.int32).contiguous()
    key_ends = key_ends.to(torch.int32).contiguous()
    columns = torch.arange(index_k.shape[0], device=device, dtype=torch.int32)
    rows_per_block = 128 // heads
    width = min(topk, key_len)

    for start in range(0, query_len, chunk):
        end = min(start + chunk, query_len)
        rows = end - start
        q, w = index_q[start:end], weights[start:end]
        starts, ends = key_starts[start:end], key_ends[start:end]
        pad = -rows % rows_per_block
        if pad:
            # Padding rows repeat the last range so they do not widen the block's key span.
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            w = F.pad(w, (0, 0, 0, pad))
            starts = torch.cat([starts, starts[-1:].expand(pad)])
            ends = torch.cat([ends, ends[-1:].expand(pad)])
        logits = _kernels().lightning_indexer_logits(
            q.reshape(-1, dim), index_k, w, starts.contiguous(), ends.contiguous()
        )[:rows]
        outside = (columns < starts[:rows, None]) | (columns >= ends[:rows, None])
        logits = logits.masked_fill_(outside, -torch.inf)[:, :key_len]
        top_scores, top_indices = logits.topk(width, dim=-1)
        scores[start:end, :width] = top_scores
        indices[start:end, :width] = top_indices.to(torch.int32).masked_fill_(
            top_scores == -torch.inf, -1
        )
    return scores, indices


class _SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, sink, scaling):
        kernels = _kernels()
        heads = q.shape[2]
        sink_log2 = torch.full(
            (kernels.padded_heads(heads),), -torch.inf, device=q.device, dtype=torch.float32
        )
        if sink is not None:
            sink_log2[:heads] = sink.detach().float() * kernels.LOG2E
        out, lse = kernels.sparse_mla_fwd(
            q,
            kv,
            indices,
            sink_log2,
            heads,
            DSA_LATENT_DIM,
            q.shape[-1] - DSA_LATENT_DIM,
            indices.shape[-1],
            scaling,
        )
        ctx.save_for_backward(q, kv, indices, out, lse, sink_log2)
        ctx.scaling = scaling
        ctx.sink_dtype = None if sink is None else sink.dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        q, kv, indices, out, lse, sink_log2 = ctx.saved_tensors
        kernels = _kernels()
        heads = q.shape[2]
        dout = dout.to(out.dtype).contiguous()
        delta = kernels.sparse_mla_bwd_delta(out, dout)
        pad = kernels.padded_heads(heads) - heads
        dkv = torch.zeros(kv.shape, device=kv.device, dtype=torch.float32)
        dq = kernels.sparse_mla_bwd(
            q,
            kv,
            dout,
            indices,
            F.pad(lse, (0, pad), value=math.inf).contiguous(),
            F.pad(delta, (0, pad)).contiguous(),
            dkv,
            heads,
            DSA_LATENT_DIM,
            q.shape[-1] - DSA_LATENT_DIM,
            indices.shape[-1],
            ctx.scaling,
        )
        dsink = None
        if ctx.sink_dtype is not None and ctx.needs_input_grad[3]:
            sink_prob = torch.exp2(sink_log2[:heads] - lse)
            dsink = -(sink_prob * delta).sum((0, 1)).to(ctx.sink_dtype)
        return dq, dkv.to(kv.dtype), None, dsink, None


def tilelang_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float,
    sink: torch.Tensor | None = None,
):
    """``q`` ``[B, S, H, D]`` against ``kv`` ``[B, T, 1, D]`` at ``indices`` ``[B, S, 1, K]``
    (``-1`` for no key), bf16. ``D`` is the 512-wide latent plus 64 RoPE channels, or the
    latent alone for a NoPE latent, which goes in zero-padded: the zeros add nothing to
    q . k. ``sink`` is a per-head logit in the softmax denominator. A row with no key comes
    back zero. Returns ``([B, S, H, 512], None)``, as ``torch_sparse_mla`` does."""
    if q.shape[-1] == DSA_LATENT_DIM:
        q = F.pad(q, (0, DSA_ROPE_DIM))
        kv = F.pad(kv, (0, DSA_ROPE_DIM))
    if q.shape[-1] != DSA_LATENT_DIM + DSA_ROPE_DIM:
        raise ValueError(
            f"sparse MLA takes a {DSA_LATENT_DIM}-wide latent with or without "
            f"{DSA_ROPE_DIM} RoPE channels, got {q.shape[-1]}"
        )
    if kv.dim() == 3:
        kv = kv.unsqueeze(2)
    if indices.dim() == 3:
        indices = indices.unsqueeze(2)
    if kv.shape[2] != 1 or indices.shape[2] != 1:
        raise ValueError("sparse MLA reads one shared latent KV head (kv_group=1)")
    indices = indices.to(torch.int32)
    width = indices.shape[-1]
    pad = max(DSA_INDEX_ALIGNMENT, -(-width // DSA_INDEX_ALIGNMENT) * DSA_INDEX_ALIGNMENT) - width
    if pad:
        indices = F.pad(indices, (0, pad), value=-1)
    out = _SparseMLA.apply(
        q.contiguous(), kv.contiguous(), indices.contiguous(), sink, float(scaling)
    )
    return out, None
