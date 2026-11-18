import os

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def _query_chunk(width: int, dim: int, heads: int) -> int:
    budget = max(1, int(os.environ.get("ANGELSPEC_MTP_SPARSE_CHUNK_MB", "512"))) << 20
    # fp32 gathered keys plus fp32 logits and probabilities per query row.
    per_query = width * dim * 4 + heads * width * 4 * 2
    return max(1, budget // per_query)


def _sparse_mla_rows(q, kv, indices, scaling, d_v, sink):
    bsz, rows, heads, dim = q.shape
    width = indices.shape[-1]
    valid = indices >= 0
    flat = indices.clamp_min(0).long().reshape(bsz, rows * width, 1).expand(-1, -1, kv.shape[-1])
    keys = torch.gather(kv, 1, flat).view(bsz, rows, width, -1).float()
    logits = torch.matmul(q.float(), keys.transpose(-1, -2)) * scaling
    logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
    if sink is not None:
        sink_logits = sink.float().view(1, 1, heads, 1).expand(bsz, rows, heads, 1)
        logits = torch.cat([logits, sink_logits], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
    if sink is not None:
        probs = probs[..., :width]
    return torch.matmul(probs, keys[..., :d_v]).to(q.dtype)


def torch_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float,
    d_v: int,
    sink: torch.Tensor | None = None,
):
    """Sparse MLA over latent KV, the semantics of ``tilelang_sparse_mla``.

    q ``[b, s, h, d]``, kv ``[b, t, 1, d]`` (or ``[b, t, d]``), indices ``[b, s, 1, w]``
    (or ``[b, s, w]``) into kv's token dim, -1 for no key. Values are ``kv[..., :d_v]``.
    Returns ``([b, s, h, d_v], None)``.
    """
    if kv.dim() == 4:
        kv = kv.squeeze(2)
    bsz, seq_len, heads, dim = q.shape
    indices = indices.reshape(bsz, seq_len, -1)
    chunk = _query_chunk(indices.shape[-1], dim, heads)
    use_checkpoint = torch.is_grad_enabled() and os.environ.get(
        "ANGELSPEC_MTP_SPARSE_CHUNK_CHECKPOINT", "1"
    ) not in {"0", "false", "False"}
    outputs = []
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        args = (q[:, start:end], kv, indices[:, start:end].detach(), scaling, d_v, sink)
        if use_checkpoint:
            outputs.append(
                torch_checkpoint(
                    _sparse_mla_rows, *args, use_reentrant=False, preserve_rng_state=False
                )
            )
        else:
            outputs.append(_sparse_mla_rows(*args))
    return torch.cat(outputs, dim=1), None


@torch.no_grad()
def torch_lightning_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    key_starts: torch.Tensor,
    key_ends: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``tilelang_lightning_indexer`` in torch, query rows chunked to bound the
    ``[rows, heads, keys]`` fp32 scores."""
    query_len, heads, _ = index_q.shape
    key_len = index_k.shape[0]
    topk = int(topk)
    device = index_q.device
    scores = torch.full((query_len, topk), -torch.inf, device=device, dtype=torch.float32)
    indices = torch.full((query_len, topk), -1, device=device, dtype=torch.int32)
    if query_len == 0 or key_len == 0 or topk == 0:
        return scores, indices
    width = min(topk, key_len)
    keys = index_k.float()
    weights = weights.reshape(query_len, heads).float()
    columns = torch.arange(key_len, device=device)
    chunk = max(1, (256 << 20) // (heads * key_len * 4))
    for start in range(0, query_len, chunk):
        end = min(start + chunk, query_len)
        logits = torch.einsum("qhd,kd->qhk", index_q[start:end].float(), keys).relu_()
        logits = torch.einsum("qhk,qh->qk", logits, weights[start:end])
        outside = (columns < key_starts[start:end, None]) | (columns >= key_ends[start:end, None])
        top_scores, top_indices = logits.masked_fill_(outside, -torch.inf).topk(width, dim=-1)
        scores[start:end, :width] = top_scores
        indices[start:end, :width] = top_indices.to(torch.int32).masked_fill_(
            top_scores == -torch.inf, -1
        )
    return scores, indices
