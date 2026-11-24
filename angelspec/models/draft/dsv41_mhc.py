"""DeepSeek-V4.1's manifold-constrained hyper-connections (mHC): ``hc_mult`` residual copies,
collapsed by ``pre``, refilled by ``post`` and mixed by the doubly stochastic ``comb``."""

import torch
import torch.nn.functional as F


def hc_split_sinkhorn(mixes, scale, base, hc_mult: int, iters: int, eps: float):
    """Same arithmetic as DeepSeek's ``hc_split_sinkhorn`` kernel."""
    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    comb = (mixes[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]).unflatten(
        -1, (hc_mult, hc_mult)
    )
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(
    h: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    iters: int,
    eps: float,
    norm_eps: float,
):
    x = h.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, fn.float()) * rsqrt
    return hc_split_sinkhorn(mixes, scale.float(), base.float(), h.shape[-2], iters, eps)


def hc_pre(h: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...k,...kd->...d", pre_mix.float(), h.float()).to(h.dtype)


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
    """Copy k gets ``post_k * x`` plus the residual copies mixed through column k of comb."""
    mixed = torch.einsum("...jk,...jd->...kd", comb, residual.float())
    return (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)
