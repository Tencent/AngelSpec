"""Ulysses sequence-parallel primitives (torch-native all-to-all / all-gather).

Shared by MTP (seq-sharded QKV) and DFlash-family (seq-sharded context KV +
head-sharded draft Q). Differentiable via custom autograd Functions.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


class _SeqAllToAll(torch.autograd.Function):
    """Ulysses all-to-all over a [b, H, S, d] tensor: swap head and seq dims."""

    @staticmethod
    def forward(ctx, x, group, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all_4d(x, group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad):
        g = _all_to_all_4d(grad, ctx.group, ctx.gather_dim, ctx.scatter_dim)
        return g, None, None, None


def _all_to_all_4d(x, group, scatter_dim, gather_dim):
    """Ulysses all-to-all for a 4D [b, H, S, d] tensor."""
    if os.environ.get("ANGELSPEC_USP_COLLECTIVE_DEBUG") == "1":
        print(
            f"[USP-A2A rank{dist.get_rank()}] all_to_all scatter={scatter_dim} "
            f"gather={gather_dim} shape={tuple(x.shape)}",
            flush=True,
        )
    world = dist.get_world_size(group)
    if world == 1:
        return x
    b, H, S, d = x.shape
    if scatter_dim == 1 and gather_dim == 2:
        # [b, H, S_local, d] -> [b, H_local, S, d]
        assert H % world == 0, f"num_heads {H} not divisible by sp world {world}"
        Hl = H // world
        x = x.reshape(b, world, Hl, S, d).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        return out.permute(1, 2, 0, 3, 4).contiguous().reshape(b, Hl, world * S, d)
    if scatter_dim == 2 and gather_dim == 1:
        # [b, H_local, S, d] -> [b, H, S_local, d]
        assert S % world == 0, f"seq {S} not divisible by sp world {world}"
        Sl = S // world
        Hl = H
        x = x.reshape(b, Hl, world, Sl, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        return out.permute(1, 0, 2, 3, 4).contiguous().reshape(b, world * Hl, Sl, d)
    raise ValueError(
        f"unsupported (scatter_dim={scatter_dim}, gather_dim={gather_dim}); "
        "expected (1,2) or (2,1)"
    )


def ulysses_all_to_all(x, group, scatter_dim, gather_dim):
    """Differentiable Ulysses all-to-all wrapper."""
    return _SeqAllToAll.apply(x, group, scatter_dim, gather_dim)


class _AllGatherHeads(torch.autograd.Function):
    """All-gather along the head dim: [B, H_local, S, d] -> [B, H_full, S, d].

    Each rank owns a disjoint head shard (same S). Backward returns the local
    head slice of the upstream gradient (no cross-rank reduction).
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        world = dist.get_world_size(group)
        ctx.world = world
        if world == 1:
            return x
        xs = x.contiguous()
        chunks = [torch.empty_like(xs) for _ in range(world)]
        dist.all_gather(chunks, xs, group=group)
        return torch.cat(chunks, dim=1)

    @staticmethod
    def backward(ctx, grad):
        if ctx.world == 1:
            return grad, None
        rank = dist.get_rank(ctx.group)
        Hl = grad.shape[1] // ctx.world
        return grad[:, rank * Hl : (rank + 1) * Hl].contiguous(), None


def all_gather_heads(x, group):
    """Differentiable all-gather on head dim (dim=1)."""
    return _AllGatherHeads.apply(x, group)


class _AllGatherSeq(torch.autograd.Function):
    """All-gather along the sequence dim: [B, H, S_local, d] -> [B, H, S_full, d].

    Rank ``r`` owns seq slice ``[r*S_local:(r+1)*S_local]``. Every rank's local
    queries may attend to the full gathered KV, so backward must **sum** the
    per-rank grads for each seq shard (reduce-scatter), not merely slice the
    local rank's grad tensor.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        world = dist.get_world_size(group)
        ctx.world = world
        if world == 1:
            return x
        xs = x.contiguous()
        if dist.get_backend(group) == "gloo":
            chunks = [torch.empty_like(xs) for _ in range(world)]
            dist.all_gather(chunks, xs, group=group)
            return torch.cat(chunks, dim=2)
        gather_input = xs.movedim(2, 0).contiguous()
        gather_output = torch.empty(
            (world * gather_input.shape[0],) + gather_input.shape[1:],
            dtype=x.dtype,
            device=x.device,
        )
        dist.all_gather_into_tensor(gather_output, gather_input, group=group)
        return gather_output.movedim(0, 2).contiguous()

    @staticmethod
    def backward(ctx, grad):
        if ctx.world == 1:
            return grad, None
        S_full = grad.shape[2]
        assert S_full % ctx.world == 0
        Sl = S_full // ctx.world
        if dist.get_backend(ctx.group) == "gloo":
            g = grad.contiguous().clone()
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=ctx.group)
            rank = dist.get_rank(ctx.group)
            return g[:, :, rank * Sl : (rank + 1) * Sl].contiguous(), None
        reduce_input = grad.movedim(2, 0).contiguous()
        reduce_output = torch.empty(
            (Sl, grad.shape[0], grad.shape[1], grad.shape[3]),
            dtype=grad.dtype,
            device=grad.device,
        )
        dist.reduce_scatter_tensor(
            reduce_output,
            reduce_input,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )
        return reduce_output.movedim(0, 2).contiguous(), None


def all_gather_seq(x, group):
    """Differentiable all-gather on sequence dim (dim=2)."""
    return _AllGatherSeq.apply(x, group)


@torch.no_grad()
def gather_seq_rows(x: torch.Tensor, positions: torch.Tensor, group) -> torch.Tensor:
    """Rows of a sequence-sharded ``[B, S_local, D]`` tensor at global ``positions``.

    Rank ``r`` holds rows ``[r*S_local:(r+1)*S_local]``. ``positions`` is ``[B, N]``
    and may differ per rank, ``N`` may not. Each rank fills in the rows it holds for
    every rank's request and one sum hands each rank its own, so a row comes back
    exactly as its owner stored it. Returns ``[B, N, D]``; not differentiable.
    """
    world = dist.get_world_size(group)
    if world == 1:
        return x.gather(1, positions.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    rank = dist.get_rank(group)
    shard = x.shape[1]
    requests = [torch.empty_like(positions) for _ in range(world)]
    dist.all_gather(requests, positions.contiguous(), group=group)
    local = torch.stack(requests) - rank * shard  # [world, B, N]
    held = (local >= 0) & (local < shard)
    rows = (
        x.unsqueeze(0)
        .expand(world, -1, -1, -1)
        .gather(2, local.clamp(0, shard - 1).unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
    )
    rows = torch.where(held.unsqueeze(-1), rows, torch.zeros((), dtype=x.dtype, device=x.device))
    dist.all_reduce(rows, op=dist.ReduceOp.SUM, group=group)
    return rows[rank]
