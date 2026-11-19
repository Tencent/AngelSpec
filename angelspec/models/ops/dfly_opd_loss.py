"""Acceptance objectives for branch-request draft OPD."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def opd_greedy_metric_keys(width: int) -> tuple[str, ...]:
    """Metric keys emitted by ``opd_greedy_accept_loss_from_hs`` for ``width`` slots."""
    keys = ["opd/branches"]
    for name in ("alpha", "prefix", "hit", "hit_prefix"):
        keys.extend(f"opd/{name}_sum_{i}" for i in range(width))
    return tuple(keys)


def opd_greedy_accept_loss_from_hs(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    block_size: int,
    *,
    vocab_chunk: int = 64,
    logprob_min_clamp: float | None = -10.0,
):
    """Expected GREEDY accept length as the OPD objective."""
    width = int(block_size) - 1
    if width <= 0:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    keys = opd_greedy_metric_keys(width)
    M = int(student_hidden.shape[0])
    if M == 0:
        z = student_hidden.new_zeros(())
        return z, {k: z.detach().clone() for k in keys}
    if M % width != 0:
        raise ValueError(
            f"OPD scored-slot count {M} not divisible by (block_size-1)={width}; "
            "layout invariant (B-1 scored slots per branch) violated."
        )
    chunk = int(vocab_chunk) if int(vocab_chunk) > 0 else M
    chunk = min(chunk, M)

    def rows(s, t):
        """(log a, hit) for one row chunk; recomputed in backward."""
        s_logits = F.linear(s, lm_head_weight).float()
        t_star = F.linear(t, lm_head_weight).float().argmax(-1)
        log_a = F.log_softmax(s_logits, dim=-1).gather(-1, t_star.unsqueeze(-1)).squeeze(-1)
        hit = (s_logits.argmax(-1) == t_star).to(log_a.dtype)
        return log_a, hit

    log_a_chunks, hit_chunks = [], []
    for s, t in zip(student_hidden.split(chunk), teacher_hidden.split(chunk)):
        if s.requires_grad:
            log_a, hit = checkpoint(rows, s, t, use_reentrant=False)
        else:
            log_a, hit = rows(s, t)
        log_a_chunks.append(log_a)
        hit_chunks.append(hit.detach())
    # log q <= 0 analytically; clamp_max only removes fp32 roundoff. The floor
    # keeps a hopeless early position from zeroing the whole tail's gradient.
    log_a = torch.cat(log_a_chunks).view(-1, width).clamp_max(0.0)
    if logprob_min_clamp is not None:
        log_a = log_a.clamp_min(float(logprob_min_clamp))
    prefix_log = log_a.cumsum(dim=-1)
    # -expm1(sum log a) == 1 - prod a, accurate near 1.
    opd_loss = (-torch.expm1(prefix_log)).mean()

    with torch.no_grad():
        branches = log_a.shape[0]
        alpha = log_a.exp()
        prefix = prefix_log.exp()
        hit = torch.cat(hit_chunks).view(-1, width)
        hit_prefix = hit.cumprod(dim=-1)
        metrics = {"opd/branches": log_a.new_tensor(float(branches))}
        for name, value in (
            ("alpha", alpha),
            ("prefix", prefix),
            ("hit", hit),
            ("hit_prefix", hit_prefix),
        ):
            column = value.sum(dim=0)
            for i in range(width):
                metrics[f"opd/{name}_sum_{i}"] = column[i]
    return opd_loss, metrics


def tempered_target_probs(
    logits: torch.Tensor,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """The target distribution vLLM's rejection sampler tests draft tokens against."""
    scaled = logits.float() / float(temperature)
    vocab = scaled.shape[-1]
    use_k = 0 < int(top_k) < vocab
    use_p = float(top_p) < 1.0
    if use_k or use_p:
        ordered, index = scaled.sort(dim=-1)
        if use_k:
            kth = ordered[..., vocab - int(top_k)].unsqueeze(-1)
            ordered = ordered.masked_fill(ordered < kth, float("-inf"))
        if use_p:
            drop = ordered.softmax(dim=-1).cumsum(dim=-1) <= 1.0 - float(top_p)
            drop[..., -1] = False
            ordered = ordered.masked_fill(drop, float("-inf"))
        scaled = torch.empty_like(scaled).scatter_(-1, index, ordered)
    return scaled.softmax(dim=-1)


def opd_tempered_metric_keys(num_slots: int) -> tuple[str, ...]:
    """Metric keys emitted by ``opd_tempered_accept_loss_from_hs`` for ``num_slots``."""
    keys = ["opd/branches"]
    for name in ("accept", "prefix", "soft", "best", "hit", "hit_prefix"):
        keys.extend(f"opd/{name}_sum_{i}" for i in range(num_slots))
    return tuple(keys)


def opd_tempered_accept_loss_from_hs(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    proposal_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    slots_per_branch: int,
    *,
    temperature: float = 0.9,
    top_k: int = 0,
    top_p: float = 1.0,
    relax_temperature: float = 1.0,
    vocab_chunk: int = 64,
):
    """Expected accept length of a greedy drafter under a sampling target."""
    slots = int(slots_per_branch)
    if slots < 1:
        raise ValueError(f"slots_per_branch must be >= 1, got {slots_per_branch}")
    if not temperature > 0 or not relax_temperature > 0:
        raise ValueError(
            f"temperature and relax_temperature must be > 0, got "
            f"{temperature} and {relax_temperature}"
        )
    keys = opd_tempered_metric_keys(slots)
    M = int(student_hidden.shape[0])
    if M == 0:
        z = student_hidden.new_zeros(())
        return z, {k: z.detach().clone() for k in keys}
    if M % slots != 0 or proposal_ids.numel() != M or teacher_hidden.shape[0] != M:
        raise ValueError(
            f"OPD rows misaligned: {M} student, {teacher_hidden.shape[0]} teacher, "
            f"{proposal_ids.numel()} proposals; expected a multiple of "
            f"slots_per_branch={slots}"
        )

    teacher_hidden = teacher_hidden.detach()
    proposal_ids = proposal_ids.reshape(-1)
    chunk = int(vocab_chunk) if int(vocab_chunk) > 0 else M
    chunk = min(chunk, M)

    def rows(s, t, x):
        """(relaxed a, exact a, best a, greedy hit) for one row chunk."""
        p = tempered_target_probs(F.linear(t, lm_head_weight), temperature, top_k, top_p)
        z = F.linear(s, lm_head_weight).float()
        soft = (torch.softmax(z / relax_temperature, dim=-1) * p).sum(dim=-1)
        exact = p.gather(-1, x.unsqueeze(-1)).squeeze(-1)
        best, top = p.max(dim=-1)
        return soft, exact, best, (x == top).to(exact.dtype)

    parts: list[list[torch.Tensor]] = [[], [], [], []]
    for s, t, x in zip(
        student_hidden.split(chunk), teacher_hidden.split(chunk), proposal_ids.split(chunk)
    ):
        out = checkpoint(rows, s, t, x, use_reentrant=False) if s.requires_grad else rows(s, t, x)
        parts[0].append(out[0])
        for i in range(1, 4):
            parts[i].append(out[i].detach())
    a_soft, a, a_best, hit = (torch.cat(p).view(-1, slots) for p in parts)

    with torch.no_grad():
        prefix = a.cumprod(dim=-1)
        reach = torch.cat([torch.ones_like(prefix[:, :1]), prefix[:, :-1]], dim=-1)
        worth = torch.ones_like(a)
        for j in range(slots - 2, -1, -1):
            worth[:, j] = 1.0 + a[:, j + 1] * worth[:, j + 1]
        weight = reach * worth

    exact_loss = 1.0 - prefix.sum(dim=-1).mean() / slots
    surrogate = -(weight * a_soft).sum(dim=-1).mean() / slots
    opd_loss = exact_loss + (surrogate - surrogate.detach())

    with torch.no_grad():
        metrics = {"opd/branches": a.new_tensor(float(a.shape[0]))}
        for name, value in (
            ("accept", a),
            ("prefix", prefix),
            ("soft", a_soft.detach()),
            ("best", a_best),
            ("hit", hit),
            ("hit_prefix", hit.cumprod(dim=-1)),
        ):
            column = value.sum(dim=0)
            for i in range(slots):
                metrics[f"opd/{name}_sum_{i}"] = column[i]
    return opd_loss, metrics
