"""Compare independently captured teacher rows under the training distribution."""

import torch
import torch.nn.functional as F

from angelspec.models.ops.dfly_opd_loss import tempered_target_probs


def teacher_parity_metrics(reference, actual, head, temperature=0.9, top_k=0, top_p=1.0):
    if reference.shape != actual.shape or reference.numel() == 0:
        raise ValueError("Teacher parity requires nonempty aligned hidden rows")
    ref, got = reference.float(), actual.float()
    if not bool(torch.isfinite(ref).all() and torch.isfinite(got).all()):
        raise ValueError("Nonfinite teacher parity inputs")
    diff = got - ref
    ref_logits = F.linear(reference.to(head.dtype), head).float()
    got_logits = F.linear(actual.to(head.dtype), head).float()
    tv = 0.5 * (
        tempered_target_probs(ref_logits, temperature, top_k, top_p)
        - tempered_target_probs(got_logits, temperature, top_k, top_p)
    ).abs().sum(-1)
    return {
        "hidden_relative_l2": float(diff.norm() / ref.norm().clamp_min(1e-8)),
        "hidden_max_abs": float(diff.abs().max()),
        "logit_mean_abs": float((got_logits - ref_logits).abs().mean()),
        "logit_max_abs": float((got_logits - ref_logits).abs().max()),
        "argmax_disagreement": float(
            (got_logits.argmax(-1) != ref_logits.argmax(-1)).float().mean()
        ),
        "tv_mean": float(tv.mean()),
        "tv_max": float(tv.max()),
    }


def assert_teacher_parity(metrics, tolerance=0.001):
    import math

    limits = {
        "hidden_relative_l2": 0.01,
        "tv_mean": tolerance,
        "tv_max": 0.01,
        "argmax_disagreement": 0.001,
    }
    failed = {
        k: (metrics[k], v)
        for k, v in limits.items()
        if not math.isfinite(metrics[k]) or metrics[k] > v
    }
    if failed:
        raise RuntimeError(
            f"Independent prefill/verify teacher parity failed: {failed}; {metrics}"
        )
