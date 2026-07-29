from __future__ import annotations

import torch


def token_metric_totals(
    metrics: list[dict],
    *,
    ce_key: str,
) -> torch.Tensor:
    """Build per-step [ce, kl, correct, correct_gt, count] token sums."""
    totals = []
    for item in metrics:
        count = item["acc_counts"].float()
        totals.append(
            torch.stack(
                [
                    item[ce_key].float() * count,
                    item.get("kl", torch.zeros_like(count)).float() * count,
                    item["acces"].float() * count,
                    item["acces_gt"].float() * count,
                    count,
                ],
                dim=-1,
            )
        )
    return torch.stack(totals, dim=0).sum(dim=0)


def means_from_token_totals(totals: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Convert per-step sums to means; zero-count steps remain finite zeros."""
    count = totals[:, 4]
    denom = count.clamp_min(1.0)
    return (
        totals[:, 0] / denom,
        totals[:, 1] / denom,
        totals[:, 2] / denom,
        totals[:, 3] / denom,
        count,
    )


def token_weighted_loss_scale(
    row_tokens: torch.Tensor,
    global_step_tokens: torch.Tensor,
    dp_world_size: int,
) -> torch.Tensor:
    """Scale a row so a later DP-AVG yields the global supervised-token mean."""
    if dp_world_size <= 0:
        raise ValueError(f"dp_world_size must be positive, got {dp_world_size}")
    return row_tokens.float() / global_step_tokens.float().clamp_min(1.0) * dp_world_size
