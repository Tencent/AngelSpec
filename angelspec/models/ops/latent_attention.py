"""Multi-query attention over one latent that is both key and value (DeepSeek-V4.1, absorbed
MLA): flex on CUDA under a BlockMask, elsewhere dense scores, for short contexts only."""

from typing import Optional

import torch
import torch.distributed as dist

#: Backward tiles for a 16-bit 512-wide latent; flex's defaults for that width are several times
#: slower. They need ~227 KB of shared memory.
LATENT_512_FLEX_KERNEL_OPTIONS = {
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 64,
    "bwd_BLOCK_M2": 64,
    "bwd_BLOCK_N2": 32,
    "bwd_num_stages": 2,
}


def _flex_kernel_options(query: torch.Tensor) -> Optional[dict]:
    if (
        query.shape[-1] == 512
        and query.element_size() == 2
        and torch.cuda.get_device_capability(query.device) >= (9, 0)
    ):
        return LATENT_512_FLEX_KERNEL_OPTIONS
    return None


def gather_context_latent(latent: torch.Tensor, full_ctx_len: Optional[int]) -> torch.Tensor:
    if full_ctx_len is None or int(full_ctx_len) == latent.shape[1]:
        return latent
    from angelspec.models.ops.ulysses import all_gather_seq
    from angelspec.utils.distributed import get_sp_ulysses_group

    group = get_sp_ulysses_group()
    if group is None or int(full_ctx_len) != latent.shape[1] * dist.get_world_size(group):
        raise ValueError(
            f"USP context shard mismatch: local={latent.shape[1]}, full_ctx_len={full_ctx_len}"
        )
    return all_gather_seq(latent.unsqueeze(1), group).squeeze(1)


def latent_attention(
    query: torch.Tensor,
    latent: torch.Tensor,
    block_mask,
    scale: float,
    sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The sink is an extra key per head scoring ``sink``, value 0; no visible key gives zero."""
    if block_mask is not None and query.is_cuda:
        from angelspec.models.ops.flex_attention import compile_friendly_flex_attention

        kv = latent.unsqueeze(1)
        out, lse = compile_friendly_flex_attention(
            query,
            kv,
            kv,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=True,
            return_lse=True,
            kernel_options=_flex_kernel_options(query),
        )
        if sink is not None:
            rescale = torch.sigmoid(lse - sink.float().view(1, -1, 1)).unsqueeze(-1)
            out = out * rescale.to(out.dtype)
        return out

    bsz, _, q_len, _ = query.shape
    logits = torch.einsum("bhqd,bkd->bhqk", query.float(), latent.float()) * scale
    visible = None
    if block_mask is not None:
        from torch.nn.attention.flex_attention import create_mask

        visible = create_mask(
            block_mask.mask_mod, bsz, 1, q_len, latent.shape[1], device=query.device
        )
        logits = logits.masked_fill(~visible, float("-inf"))
    if sink is not None:
        sink_logits = sink.float().view(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
        logits = torch.cat([logits, sink_logits], dim=-1)
    probs = torch.nan_to_num(logits.softmax(dim=-1))
    if sink is not None:
        probs = probs[..., :-1]
    if visible is not None:
        probs = probs * visible.any(-1, keepdim=True)
    return torch.einsum("bhqk,bkd->bhqd", probs, latent.float()).to(query.dtype)


def flex_sparse_latent_attention(
    q: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float,
    sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Listed keys as a flex block mask: cost follows the context span covered, not list length."""
    from angelspec.models.ops.flex_attention import compile_friendly_create_block_mask

    bsz, rows, _ = indices.shape
    length = latent.shape[1]
    groups, tables = [], []
    for keys in indices:
        unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
        table = torch.zeros(unique.shape[0], length + 1, dtype=torch.bool, device=keys.device)
        table.scatter_(1, torch.where(unique >= 0, unique.long(), length), True)
        tables.append(table[:, :length])
        groups.append(inverse)
    table = torch.zeros(
        bsz, max(t.shape[0] for t in tables), length, dtype=torch.bool, device=indices.device
    )
    for b, rows_table in enumerate(tables):
        table[b, : rows_table.shape[0]] = rows_table
    groups = torch.stack(groups)

    def listed_keys_mask_mod(b, h, q_idx, kv_idx):
        return table[b, groups[b, q_idx], kv_idx]

    block_mask = compile_friendly_create_block_mask(
        listed_keys_mask_mod, bsz, None, rows, length, q.device, _compile=q.is_cuda
    )
    out = latent_attention(q.transpose(1, 2), latent, block_mask, scaling, sink=sink)
    return out.transpose(1, 2)


#: fp32 scratch per checkpointed chunk of key lists in grouped_sparse_latent_attention.
GROUPED_CHUNK_BYTES = 1 << 30


class _GatherRows(torch.autograd.Function):
    """``source[index]`` (``-1`` reads row 0) with a backward free of atomics, which overlapping,
    heavily padded key lists serialize on."""

    @staticmethod
    def forward(ctx, source, index, unique):
        ctx.save_for_backward(index)
        ctx.shape, ctx.unique = source.shape, unique
        return source[index.clamp(min=0)]

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        flat = index.reshape(-1)
        grad = grad.reshape(flat.shape[0], -1)
        live = (flat >= 0).nonzero().squeeze(-1)
        rows = flat[live]
        if ctx.unique:
            out = grad.new_zeros(ctx.shape[0], grad.shape[1])
            out[rows] = grad[live]
        else:
            order = torch.argsort(rows)
            counts = torch.bincount(rows, minlength=ctx.shape[0])
            crow = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
            ones = torch.ones(live.numel(), dtype=grad.dtype, device=grad.device)
            matrix = torch.sparse_csr_tensor(
                crow, live[order], ones, size=(ctx.shape[0], flat.shape[0])
            )
            out = matrix @ grad
        return out.view(ctx.shape), None, None


def _read_groups(
    q: torch.Tensor,
    latent: torch.Tensor,
    table: torch.Tensor,
    lists: torch.Tensor,
    sink: Optional[torch.Tensor],
    scaling: float,
) -> torch.Tensor:
    queries = _GatherRows.apply(q, table, True)
    keys = _GatherRows.apply(latent, lists.long(), False)
    # Scores in fp32 as the kernels keep them; the read back at the inputs' precision.
    logits = torch.einsum("gwhd,gkd->gwhk", queries.float(), keys.float()) * scaling
    logits = logits.masked_fill((lists < 0)[:, None, None, :], float("-inf"))
    total = torch.logsumexp(logits, -1, keepdim=True)
    if sink is not None:
        total = torch.logaddexp(total, sink.float().view(1, 1, -1, 1))
    read = torch.einsum("gwhk,gkd->gwhd", torch.exp(logits - total).to(keys.dtype), keys)
    return read[table >= 0].to(q.dtype)


def grouped_sparse_latent_attention(
    q: torch.Tensor,
    latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float,
    sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Kernel-free, chunked and recomputed in backward; cost follows the lists' length."""
    from torch.utils.checkpoint import checkpoint

    heads, dim = q.shape[-2:]
    rows = []
    for q_row, latent_row, keys in zip(q, latent, indices):
        live = (keys >= 0).any(-1).nonzero().squeeze(-1)
        if not live.numel():
            rows.append(torch.zeros_like(q_row))
            continue
        lists, group = torch.unique(keys[live], dim=0, return_inverse=True)
        counts = torch.bincount(group, minlength=lists.shape[0])
        order = torch.argsort(group, stable=True)
        slot = (
            torch.arange(order.numel(), device=q.device)
            - (counts.cumsum(0) - counts)[group[order]]
        )
        table = torch.full(
            (lists.shape[0], int(counts.max())), -1, dtype=torch.long, device=q.device
        )
        table[group[order], slot] = live[order]
        width, n_keys = table.shape[1], lists.shape[1]
        per_list = 4 * (2 * width * heads * n_keys + n_keys * dim + width * heads * dim)
        chunk = max(1, GROUPED_CHUNK_BYTES // per_list)
        # Gathered in fp32: the overlapping lists' gradients would contend on bf16 atomics.
        latent_row = latent_row.float()
        reads = [
            checkpoint(
                _read_groups,
                q_row,
                latent_row,
                table[start : start + chunk],
                lists[start : start + chunk],
                sink,
                scaling,
                use_reentrant=False,
            )
            for start in range(0, lists.shape[0], chunk)
        ]
        used = table >= 0
        rows.append(torch.zeros_like(q_row).index_put((table[used],), torch.cat(reads)))
    return torch.stack(rows)
