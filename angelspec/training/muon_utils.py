"""Muon (MomentUm Orthogonalized by Newton-schulz) primitives.

Pure-tensor helpers following KellerJordan/Muon and MoonshotAI/Moonlight. No
optimizer state or distributed-training framework assumptions live here — see
``optimizer.BF16Optimizer`` for how these are wired into angelspec's fp32
master-weight training loop.

The Newton-Schulz orthogonalization runs on a FULL 2-D matrix. Under FSDP2 the
gradient/master-weight is a sharded ``DTensor``; ``zeropower_via_newtonschulz5``
gathers it with ``full_tensor()`` before iterating and returns a full (replicated)
tensor. Resharding back to the parameter's own placement is the caller's job.
"""

import math

import torch


def adjust_lr_for_muon(lr: float, param_shape, matched_adamw_rms: float = 0.2) -> float:
    """Scale the base LR per matrix so Muon's update magnitude matches AdamW.

    Follows MoonshotAI/Moonlight: ``lr * sqrt(max(A, B)) * matched_adamw_rms``
    where ``(A, B)`` are the first two dims of the weight. This keeps a single
    scheduler LR usable for both the Muon and AdamW parameter groups.
    """
    A, B = param_shape[:2]
    return lr * math.sqrt(max(A, B)) * matched_adamw_rms


def zeropower_via_newtonschulz5(
    grad: torch.Tensor, steps: int = 5, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Newton-Schulz iteration computing the zeroth power / orthogonalization of ``grad``.

    Quintic iteration with coefficients tuned to maximize the slope at zero (see
    KellerJordan/Muon). Produces roughly ``U V^T`` (the orthogonal factor of the
    SVD ``grad = U S V^T``) up to a noisy diagonal, which does not hurt Muon.

    ``grad`` may be a plain tensor OR an FSDP2 ``DTensor``: in the DTensor case it
    is gathered to a full tensor, orthogonalized, and returned as a full tensor
    (NOT resharded). The NS iterations run in ``dtype`` (bf16 by default) for
    speed; the input's own dtype governs the returned dtype only through the
    caller (this function returns ``dtype``).

    Args:
        grad: 2-D (or higher, flattened by the caller) gradient matrix.
        steps: number of NS iterations (5 is almost always enough).
        dtype: working precision for the iteration.

    Returns:
        Orthogonalized matrix, same shape as ``grad``, dtype ``dtype``.
    """
    from torch.distributed.tensor import DTensor

    if isinstance(grad, DTensor):
        grad = grad.full_tensor()

    assert grad.ndim >= 2, grad.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = grad.to(dtype)
    # Orthogonalize on the smaller dimension.
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    # Ensure spectral norm is at most 1.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X


__all__ = ["adjust_lr_for_muon", "zeropower_via_newtonschulz5"]
