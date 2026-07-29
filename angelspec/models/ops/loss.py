# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import torch
import torch.nn.functional as F


def _forward_kl_from_logits(logits: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.float()
    return torch.logsumexp(logits_f32, dim=-1) - (target_p * logits_f32).sum(-1)


def _softmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.float()
    return torch.exp(logits_f32 - torch.logsumexp(logits_f32, dim=-1, keepdim=True))


@torch.compile(dynamic=None)
def compiled_sum_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    norm_eps,
):
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == tp.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


@torch.compile(dynamic=None)
def compiled_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    norm_eps,
):
    """torch.compile'd index_select + RMSNorm + lm_head + Forward KL loss.

    Takes full (B*T, ...) flat tensors and performs index_select inside the
    compiled graph so the compiler can fuse the gather with subsequent ops.

    Args:
        prenorm_hidden_states_flat: (B*T, H) — flattened draft hidden states
        target_p_flat: (B*T, V_out) — flattened target probs (detached)
        valid_idx: (N,) int64 — indices of non-masked positions
        norm_weight: (H,)
        lm_head_weight: (V_out, H) — draft lm_head weight
        norm_eps: float
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)  # (N, V_out)

    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == tp.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


@torch.compile(dynamic=None)
def compiled_sum_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    target_lm_head_weight,
    norm_eps,
):
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


@torch.compile(dynamic=None)
def compiled_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    target_lm_head_weight,
    norm_eps,
):
    """torch.compile'd index_select + target softmax + RMSNorm + lm_head + Forward KL loss.

    Like compiled_forward_kl_loss but takes full (B*T, ...) flat tensors and
    performs index_select inside the compiled graph.  This lets the compiler
    fuse the gather with subsequent ops, avoiding a separate (N, V_full) copy
    outside the compiled region.

    Used for the non-pruning (LazyTarget) path where V_full is large.
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # Target probs (detached weights → no grad flows through target)
    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)

    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


def _ce_from_logits(logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy against hard target tokens (sum semantics)."""
    logits_f32 = logits.float()
    logZ = torch.logsumexp(logits_f32, dim=-1)
    tgt_logit = logits_f32.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
    return logZ - tgt_logit


def _kl_full(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Full-vocab forward KL KL(teacher || student) per position (sum semantics)."""
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    t_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    t_p = t_logp.exp()
    return (t_p * (t_logp - s_logp)).sum(-1)


def _kl_topk_variant_a(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, topk: int
) -> torch.Tensor:
    """Variant A top-k: renormalise BOTH within the teacher top-k subspace.

    Take teacher top-K indices, gather student logits there, then
    log_softmax/softmax independently inside that K-subspace before computing KL.
    """
    t_f32 = teacher_logits.float()
    s_f32 = student_logits.float()
    topk_idx = t_f32.topk(topk, dim=-1).indices
    t_sub = t_f32.gather(-1, topk_idx)
    s_sub = s_f32.gather(-1, topk_idx)
    t_logp = F.log_softmax(t_sub, dim=-1)
    s_logp = F.log_softmax(s_sub, dim=-1)
    t_p = t_logp.exp()
    return (t_p * (t_logp - s_logp)).sum(-1)


def _kl_variant_b(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, topk: int
) -> torch.Tensor:
    """Variant B: full-vocab softmax, but accumulate KL only on teacher top-k.

    KL = Σ_{i∈topk(teacher)} p_T(i)·[log p_T(i) − log p_S(i)] with full-vocab
    softmax for both (penalises probability mass leaking outside the top-k).
    """
    t_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    topk_idx = t_logp.topk(topk, dim=-1).indices
    t_logp_k = t_logp.gather(-1, topk_idx)
    s_logp_k = s_logp.gather(-1, topk_idx)
    t_p_k = t_logp_k.exp()
    return (t_p_k * (t_logp_k - s_logp_k)).sum(-1)


def lk_tv_kl_per_pos(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    form: str,
    eta: float = 3.0,
    topk=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position single-step distillation term + TV distance.

        TV = 0.5·Σ|p−q|,  KL = Σ p·(log p − log q),  λ = exp(−eta·sg(1−TV))
        form="tv" → TV;  "kl" → KL;  "lk" → λ·KL + (1−λ)·TV

    ``student_logits`` is grad-carrying, ``teacher_logits`` is the (detached) teacher.
    ``topk`` restricts both to the teacher top-k subspace (renormalised there); None =
    full vocab. Returns (ell, tv), both [N].
    """
    t_f32 = teacher_logits.float()
    s_f32 = student_logits.float()
    if topk is not None:
        topk_idx = t_f32.topk(int(topk), dim=-1).indices
        t_f32 = t_f32.gather(-1, topk_idx)
        s_f32 = s_f32.gather(-1, topk_idx)
    s_logp = F.log_softmax(s_f32, dim=-1)
    q = s_logp.exp()
    p = F.softmax(t_f32, dim=-1)
    tv = 0.5 * (p - q).abs().sum(dim=-1)
    if form == "tv":
        return tv, tv
    eps = 1e-9
    kl = (p * (p.clamp_min(eps).log() - s_logp)).sum(dim=-1)
    if form == "lk":
        lam = torch.exp(-eta * (1.0 - tv).detach().clamp(0.0, 1.0))
        return lam * kl + (1.0 - lam) * tv, tv
    if form == "kl":
        return kl, tv
    raise ValueError(f"Unknown single-step form={form!r}; expected 'tv'/'kl'/'lk'.")


def mtp_loss_from_hs(
    post_norm_hidden_flat: torch.Tensor,
    target_hidden_states_flat: torch.Tensor,
    valid_idx: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    *,
    use_kl: bool = False,
    kl_topk=None,
    kl_variant: str = "a",
    gt_labels_flat: torch.Tensor = None,
    ce_use_ground_truth: bool = False,
    single_step_form: str = None,
    single_step_eta: float = 3.0,
    return_per_pos: bool = False,
):
    """CE + (optional) KL loss for the single-head MTP draft.

    The draft hidden states are ALREADY post-final_layernorm (the MTP backbone
    applies final_layernorm), so we project them straight through the tied
    lm_head — no extra RMSNorm here (unlike the Eagle3 fused path).

    Args:
        post_norm_hidden_flat: (B*T, H) draft hidden states after final_layernorm.
        target_hidden_states_flat: (B*T, H_t) target last hidden states.
        valid_idx: (N,) indices of non-masked positions.
        lm_head_weight: (V, H) tied/frozen student head (== target head).
        target_lm_head_weight: (V, H_t) frozen target head (teacher).
        use_kl: whether to compute the KL term.
        kl_topk: teacher top-k subspace size (None = full vocab).
        kl_variant: "a" (subspace renorm) or "b" (full softmax, top-k accumulate).
        gt_labels_flat: (B*T,) ground-truth next-token ids (data hard labels).
            Required when ce_use_ground_truth=True.
        ce_use_ground_truth: when True the CE target is the data's ground-truth
            token instead of the target model's own argmax (self-distillation,
            whose train-time acc is an over-optimistic proxy for serve acceptance).
            KL teacher is always the target head.
        single_step_form: when set ("tv"/"kl"/"lk") the ``kl_sum`` slot carries this
            single-step distillation term (via ``lk_tv_kl_per_pos``) instead of the
            plain forward-KL; ``use_kl`` is ignored.
        single_step_eta: LK schedule decay rate (form="lk" only).
        return_per_pos: also return per-position ``ell`` and ``alpha`` (= 1 − TV) on
            the full flat grid (``post_norm_hidden_flat.shape[0]``); requires
            ``single_step_form``.

    Returns:
        (ce_sum, kl_sum, correct, count, correct_gt) — all sum-reduced over valid
        positions; the caller divides by ``count`` to recover the mean (keep this
        sum-then-divide path to avoid a sum/mean gradient-scale pitfall).
        ``correct`` = draft argmax == target argmax (self-distill acc);
        ``correct_gt`` = draft argmax == ground-truth token (real-acceptance proxy).
        With ``return_per_pos`` two extra tensors (ell_full, alpha_full), each [B*T],
        are appended (non-valid positions filled ell=0 / alpha=1).
    """
    hs_all = post_norm_hidden_flat.index_select(0, valid_idx)
    ths_all = target_hidden_states_flat.index_select(0, valid_idx)
    gt_all = gt_labels_flat.index_select(0, valid_idx) if gt_labels_flat is not None else None

    if ce_use_ground_truth and gt_all is None:
        raise ValueError("ce_use_ground_truth=True requires gt_labels_flat")
    if return_per_pos and single_step_form is None:
        raise ValueError("return_per_pos=True requires single_step_form")

    # Chunk over valid rows so the full-vocab logits [N, V] never materialise at
    # once (at 128k a 32k-row shard × 120832 vocab is ~15 GB fp32 and OOMs). All
    # returned quantities are position-wise sums, so summing per chunk is exact.
    # CHUNK is tunable via ANGELSPEC_MTP_LOSS_CHUNK (0/unset => no chunking).
    n = hs_all.shape[0]
    chunk = int(os.environ.get("ANGELSPEC_MTP_LOSS_CHUNK", "0") or 0)
    if chunk <= 0 or chunk >= n:
        chunk = n if n > 0 else 1

    ce_sum = hs_all.new_zeros(())
    kl_sum = hs_all.new_zeros(())
    correct_sum = hs_all.new_zeros((), dtype=torch.float32)
    correct_gt_sum = hs_all.new_zeros((), dtype=torch.float32)
    count = torch.tensor(float(n), device=hs_all.device)

    # Per-position ell/alpha on the valid subset; scattered to the full grid below.
    ell_valid = hs_all.new_zeros(n) if return_per_pos else None
    alpha_valid = hs_all.new_zeros(n) if return_per_pos else None

    for start in range(0, n, chunk):
        hs = hs_all[start : start + chunk]
        ths = ths_all[start : start + chunk]

        target_logits = F.linear(ths, target_lm_head_weight)
        target_tokens = target_logits.argmax(-1)
        student_logits = F.linear(hs, lm_head_weight)
        student_tokens = student_logits.argmax(-1)

        gt_tokens = gt_all[start : start + chunk] if gt_all is not None else None
        ce_target_tokens = gt_tokens if ce_use_ground_truth else target_tokens

        ce_tok = _ce_from_logits(student_logits, ce_target_tokens)
        correct_sum = correct_sum + (student_tokens == target_tokens).float().sum()
        if gt_tokens is not None:
            correct_gt_sum = correct_gt_sum + (student_tokens == gt_tokens).float().sum()

        if single_step_form is not None:
            ell_tok, tv_tok = lk_tv_kl_per_pos(
                student_logits,
                target_logits,
                single_step_form,
                eta=single_step_eta,
                topk=kl_topk,
            )
            kl_sum = kl_sum + ell_tok.sum()
            if return_per_pos:
                ell_valid[start : start + chunk] = ell_tok
                alpha_valid[start : start + chunk] = 1.0 - tv_tok
        elif use_kl:
            if kl_topk is None:
                kl_tok = _kl_full(student_logits, target_logits)
            elif kl_variant == "b":
                kl_tok = _kl_variant_b(student_logits, target_logits, int(kl_topk))
            else:
                kl_tok = _kl_topk_variant_a(student_logits, target_logits, int(kl_topk))
            kl_sum = kl_sum + kl_tok.sum()

        ce_sum = ce_sum + ce_tok.sum()

    if return_per_pos:
        m = post_norm_hidden_flat.shape[0]
        ell_full = hs_all.new_zeros(m)
        alpha_full = hs_all.new_ones(m)
        ell_full = ell_full.index_copy(0, valid_idx, ell_valid)
        alpha_full = alpha_full.index_copy(0, valid_idx, alpha_valid)
        return ce_sum, kl_sum, correct_sum, count, correct_gt_sum, ell_full, alpha_full

    return ce_sum, kl_sum, correct_sum, count, correct_gt_sum


# Metric keys emitted by opd_two_stream_kl_from_hs; a module constant so the
# trainer can build zero-filled dicts and keep DP all_reduce collectives
# symmetric across ranks (every opd-on rank must reduce the identical key set).
_OPD_METRIC_KEYS = (
    "opd/resp_kl_sum",
    "opd/rej_kl_wsum",
    "opd/accepted_cnt",
    "opd/rejected_cnt",
    "opd/rej_eff_w_sum",
    "opd/clamp_frac_sum",
    "opd/scored_cnt",
)


def opd_two_stream_kl_from_hs(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    proposed_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    block_size: int,
    *,
    response_stream_weight: float = 1.0,
    rejected_stream_weight: float = 1.0,
    position_decay: float = 0.8,
    position_decay_enabled: bool = True,
    loss_max_clamp: float | None = 10.0,
    logprob_min_clamp: float | None = -10.0,
):
    """Two-stream reverse-KL(k3) OPD loss over the DFlash proposal tree.

    Two KL terms (response + rejected-draft stream) over the half-OPD proposal tree.
    The M scored slots are branch-major × within-branch, EXACTLY ``block_size-1``
    per branch (see ``tree_layout.build_dflash_opd_layout``), so a reshape to
    ``(num_branches, block_size-1)`` recovers per-branch rows whose columns run in
    ascending block offset ``1..block_size-1``.

    Greedy verify per branch (matches the serving rule
    ``candidates == target_predict``): accept the prefix where the draft-proposed
    token equals the teacher argmax; the suffix from the first mismatch is
    "rejected".
      - accepted slots -> response stream: reverse-KL(k3) on the proposed token.
      - rejected slots -> rejected-draft stream: same k3, weighted by
        ``decay^(offset-1)`` (offset = block-start position, 1-based).

    Both student and teacher project through the SAME frozen target ``lm_head``;
    gradient flows only through ``student_hidden`` (teacher is detached upstream).

    Returns:
        (opd_loss, metrics) where ``metrics`` values are un-averaged sum/count
        scalars (detached) so the trainer can DP-reduce them correctly.
    """
    width = int(block_size) - 1
    if width <= 0:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    M = int(student_hidden.shape[0])
    device = student_hidden.device
    if M == 0:
        z = student_hidden.new_zeros(())
        return z, {k: z.detach().clone() for k in _OPD_METRIC_KEYS}
    if M % width != 0:
        raise ValueError(
            f"OPD scored-slot count {M} not divisible by (block_size-1)={width}; "
            "layout invariant (B-1 scored slots per branch) violated."
        )
    if not (0.0 < position_decay <= 1.0):
        raise ValueError(f"position_decay must be in (0, 1], got {position_decay}")

    # Per-slot scalar logprobs (student grad-carrying) + teacher argmax, chunked
    # over M so the [C, V] fp32 logits never materialise all at once (Qwen3 V~151k
    # => ~1.2 MB/slot for the two fp32 logits). Upcast to fp32 before log_softmax,
    # matching _ce_from_logits / _kl_full.
    chunk = int(os.environ.get("ANGELSPEC_MTP_LOSS_CHUNK", "0") or 0)
    if chunk <= 0 or chunk >= M:
        chunk = M if M > 0 else 1
    s_logp_chunks, t_logp_chunks, t_argmax_chunks = [], [], []
    for start in range(0, M, chunk):
        sl = slice(start, start + chunk)
        s_logits = F.linear(student_hidden[sl], lm_head_weight).float()
        t_logits = F.linear(teacher_hidden[sl], lm_head_weight).float()
        ids = proposed_ids[sl].unsqueeze(-1)
        s_logp_chunks.append(F.log_softmax(s_logits, dim=-1).gather(-1, ids).squeeze(-1))
        t_logp_chunks.append(F.log_softmax(t_logits, dim=-1).gather(-1, ids).squeeze(-1))
        t_argmax_chunks.append(t_logits.argmax(-1))
    student_logp = torch.cat(s_logp_chunks)  # (M,) grad
    teacher_logp = torch.cat(t_logp_chunks)  # (M,) detached
    teacher_argmax = torch.cat(t_argmax_chunks)  # (M,) detached

    if logprob_min_clamp is not None:
        student_logp = student_logp.clamp_min(logprob_min_clamp)
        teacher_logp = teacher_logp.clamp_min(logprob_min_clamp)

    # k3 reverse-KL estimator (Schulman low-variance): exp(Δ) - Δ - 1, Δ = t - s.
    delta = (teacher_logp - student_logp).clamp(-20.0, 20.0)
    k3 = torch.exp(delta) - delta - 1.0
    if loss_max_clamp is not None:
        clamp_frac_sum = (k3.detach().abs() > loss_max_clamp).float().sum()
        k3 = k3.clamp(min=-loss_max_clamp, max=loss_max_clamp)
    else:
        clamp_frac_sum = k3.new_zeros(())

    # Greedy verify per branch: accepted = matched prefix (cumprod), rest rejected.
    proposed = proposed_ids.reshape(-1, width)
    t_argmax = teacher_argmax.reshape(-1, width)
    match = (proposed == t_argmax).long()
    accepted = torch.cumprod(match, dim=1).bool().reshape(-1)  # (M,)
    rejected = ~accepted

    # Rejected-draft position decay: weight = decay^(offset-1), offset = col+1.
    if position_decay_enabled:
        exps = torch.arange(width, device=device, dtype=torch.float32)  # 0..width-1 == offset-1
        rej_w_row = torch.pow(torch.tensor(float(position_decay), device=device), exps)
        rej_w = rej_w_row.unsqueeze(0).expand(proposed.shape[0], -1).reshape(-1)
    else:
        rej_w = torch.ones(M, device=device)
    rej_w = rej_w * rejected.float()

    resp_mask = accepted.float()
    resp_sum = (k3 * resp_mask).sum()
    resp_cnt = resp_mask.sum()
    rej_sum = (k3 * rej_w).sum()  # position-weighted
    rej_cnt = rejected.float().sum()
    rej_eff = rej_w.sum()  # effective (decayed) count

    # Local (per-micro-batch) normalisation, consistent with the existing
    # kl / cnt convention; DDP averages gradients across ranks.
    denom = response_stream_weight * resp_cnt + rejected_stream_weight * rej_eff
    if float(denom) <= 0.0:
        opd_loss = (resp_sum + rej_sum) * 0.0
    else:
        opd_loss = (response_stream_weight * resp_sum + rejected_stream_weight * rej_sum) / denom

    metrics = {
        "opd/resp_kl_sum": resp_sum.detach(),
        "opd/rej_kl_wsum": rej_sum.detach(),
        "opd/accepted_cnt": resp_cnt.detach(),
        "opd/rejected_cnt": rej_cnt.detach(),
        "opd/rej_eff_w_sum": rej_eff.detach(),
        "opd/clamp_frac_sum": clamp_frac_sum.detach(),
        "opd/scored_cnt": (resp_cnt + rej_cnt).detach(),
    }
    return opd_loss, metrics
