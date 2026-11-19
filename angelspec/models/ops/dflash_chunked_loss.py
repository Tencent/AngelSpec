from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_dflash_loss(
    model,
    hidden,
    targets,
    previous,
    keep,
    label_valid,
    label_indices,
    loss_mask,
    head,
    n_blocks,
    anchors,
    teacher,
    teacher_at_labels,
    target_norm,
    chunk,
    return_draft,
):
    from angelspec.models.dflash import (
        DFlashModel,
        _dpace_position_weights,
        _hc_topk_hits,
    )

    batch, _, width = hidden.shape
    block = model.block_size
    per_chunk = max(1, chunk // max(1, batch * block))
    hidden_blocks = hidden.reshape(batch, n_blocks, block, width)
    mask = keep.unsqueeze(-1).expand(-1, -1, block).float() * label_valid.float()
    first = model.first_pred_slot
    mask = mask * (torch.arange(block, device=hidden.device).view(1, 1, -1) >= first).float()
    mask = mask * torch.gather(loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, label_indices)
    available = teacher is not None or teacher_at_labels is not None
    lk_active = model.lk_loss_weight > 0 and available
    kl_active = not lk_active and model.kl_loss_weight > 0 and available
    tv_active = model.e2e_tv_loss_weight > 0 and available
    want_teacher = lk_active or kl_active or tv_active
    if model.l1_loss_alpha > 0 and not available:
        raise ValueError("L1 distillation requires teacher hidden states")
    selector = model._path_selector() is not None and model.path_selector_loss_alpha != 0
    corpus_ce = model.ce_loss_alpha > 0 and not (lk_active or kl_active)

    def compute(h, target, prev, binary, teacher_h, weight, start):
        nb = h.shape[1]
        hflat = h.reshape(batch, nb * block, width)
        logits = model._compute_draft_logits(hflat, weight, prev, nb, block_start=start)
        candidates = model._hc_topk_candidates
        if model.fp32_lm_head:
            logits = logits.float()
        flat = logits.reshape(-1, logits.shape[-1])
        with torch.set_grad_enabled(torch.is_grad_enabled() and model.ce_loss_alpha != 0):
            ce = F.cross_entropy(flat, target.reshape(-1), reduction="none").reshape(
                batch, nb, block
            )
        objective = binary
        if (
            model.loss_objective == "decay"
            and model.loss_decay_gamma is not None
            and model.loss_decay_gamma > 0
        ):
            positions = torch.arange(block, device=h.device).view(1, 1, -1)
            objective = binary * torch.exp(
                -(positions - first).clamp(min=0).float() / model.loss_decay_gamma
            )
        elif model.loss_objective == "dpace":
            with torch.no_grad():
                weights = torch.ones_like(binary)
                if block > first:
                    weights[..., first:] = _dpace_position_weights(
                        torch.exp(-ce[..., first:].float()), model.dpace_alpha
                    )
                objective = binary * weights
        teacher_logits = None
        if want_teacher or model.l1_loss_alpha > 0:
            with torch.no_grad():
                th = target_norm(teacher_h) if target_norm is not None else teacher_h
                teacher_logits = F.linear(th.to(weight.dtype), weight).reshape_as(flat)
        l1 = torch.zeros_like(ce)
        if model.l1_loss_alpha > 0:
            l1 = model._compute_l1_loss(flat, teacher_logits).reshape_as(ce)
        zero = flat.new_zeros(())
        base_num = ((model.ce_loss_alpha * ce + model.l1_loss_alpha * l1) * objective).sum()
        kl_num, lk_num, tv_num, accept_num, blocks = zero, zero, zero, zero, zero
        if lk_active:
            term = model._compute_lk_loss(flat, teacher_logits, model.lk_loss_type, model.lk_eta)
            lk_num = (term * objective.reshape(-1).float()).sum()
        elif kl_active:
            term = model._compute_topk_kl_loss_variant_b(flat, teacher_logits, model.kl_topk)
            kl_num = (term * objective.reshape(-1).float()).sum()
        if tv_active:
            blocks = (binary.sum(-1) > 0).float().sum()
            tv, accepted = DFlashModel._compute_e2e_tv_loss(
                SimpleNamespace(sp_size=1),
                logits.reshape(batch, nb, block, -1),
                teacher_logits.reshape(batch, nb, block, -1),
                binary,
                vocab_chunk=model.e2e_tv_vocab_chunk,
            )
            tv_num, accept_num = tv * blocks, accepted * blocks
        teacher_ids = teacher_logits.argmax(-1) if want_teacher else None
        hc_hits = (
            _hc_topk_hits(candidates, teacher_ids if teacher_ids is not None else target, binary)
            if candidates is not None
            else flat.new_zeros(block)
        )
        sel_num, recall, sel_correct, serving = zero, zero, zero, zero
        if selector:
            labels = target.reshape(batch, -1)
            if model.path_selector_teacher_labels and not corpus_ce and teacher_ids is not None:
                labels = teacher_ids.reshape(batch, -1)
            sel_num, recall, sel_correct, serving = model._path_selector_sums(
                unary_logits=logits,
                hidden=hflat,
                prev_token_ids=prev.reshape(batch, -1),
                label_ids=labels,
                objective_weights=(
                    objective if model.path_selector_objective_weighting else binary
                ).reshape(batch, -1),
                binary_mask=binary.reshape(batch, -1),
            )
        with torch.no_grad():
            predicted = model._greedy_draft_ids(logits, hflat, prev.reshape(batch, -1)).reshape_as(
                target
            )
            correct = ((predicted == target) & (binary > 0.5)).float().sum((0, 1))
            teacher_correct = (
                ((predicted == teacher_ids.reshape_as(target)) & (binary > 0.5))
                .float()
                .sum((0, 1))
                if teacher_ids is not None
                else torch.zeros_like(correct)
            )
            ce_num = (ce * objective).sum()
            l1_num = (l1 * objective.float()).sum()
            ce_position = (ce * binary).sum((0, 1))
            proposals = flat.argmax(-1).reshape_as(target) if return_draft else target.new_empty(0)
        return (
            base_num,
            kl_num,
            lk_num,
            tv_num,
            sel_num,
            objective.sum(),
            binary.sum(),
            blocks,
            ce_num,
            l1_num,
            accept_num,
            recall.detach(),
            sel_correct.detach(),
            serving.detach(),
            hc_hits,
            correct,
            teacher_correct,
            ce_position,
            binary.sum((0, 1)),
            proposals,
        )

    totals = None
    proposal_parts = []
    for start in range(0, n_blocks, per_chunk):
        end = min(start + per_chunk, n_blocks)
        th = None
        if want_teacher or model.l1_loss_alpha > 0:
            if teacher_at_labels is not None:
                th = teacher_at_labels.reshape(batch, n_blocks, block, -1)[:, start:end]
            else:
                indices = (label_indices[:, start:end] - 1).clamp(min=0).reshape(batch, -1)
                th = teacher.gather(1, indices.unsqueeze(-1).expand(-1, -1, teacher.shape[-1]))
                th = th.reshape(batch, end - start, block, -1)
        args = (
            hidden_blocks[:, start:end],
            targets[:, start:end],
            previous[:, start:end],
            mask[:, start:end],
            th,
            head,
            start,
        )
        if torch.is_grad_enabled():
            values = checkpoint(compute, *args, use_reentrant=False)
        else:
            values = compute(*args)
        totals = (
            list(values[:-1]) if totals is None else [a + b for a, b in zip(totals, values[:-1])]
        )
        if return_draft:
            proposal_parts.append(values[-1])
    if totals is None:
        raise ValueError("Chunked loss requires at least one block")
    (
        base,
        kl,
        lk,
        tv,
        sel,
        weighted,
        binary,
        blocks,
        ce,
        l1,
        accepted,
        recall,
        sel_correct,
        serving,
        hc_hits,
        correct,
        teacher_correct,
        ce_pos,
        count_pos,
    ) = totals
    counts = torch.stack([weighted.detach(), binary.detach(), blocks.detach()])
    if model.sp_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=model.sp_group)
    denom = counts[0].clamp(min=1e-6)
    base, kl, lk = (
        base / denom,
        (kl / denom.float()).to(base.dtype),
        (lk / denom.float()).to(base.dtype),
    )
    loss = base
    if lk_active or kl_active:
        mix = max(0.0, min(1.0, model.lk_loss_weight if lk_active else model.kl_loss_weight))
        loss = mix * (lk if lk_active else kl) + (1 - mix) * base
    tv = tv / counts[2].clamp(min=1)
    loss = loss + model.e2e_tv_loss_weight * tv
    components = {
        "ce_loss": (ce / weighted.clamp(min=1e-6)).detach(),
        "kl_loss": kl.detach(),
        "lk_loss": lk.detach(),
        "e2e_tv_loss": tv.detach(),
    }
    if model.l1_loss_alpha > 0:
        components["l1_loss"] = (l1 / weighted.clamp(min=1e-6)).detach()
    if want_teacher:
        components["teacher_acc_per_position"] = teacher_correct / count_pos.clamp(min=1)
    if tv_active:
        components["e2e_accept_len"] = (accepted / blocks.clamp(min=1)).detach()
    if model.hidden_correction_topk:
        components["hc_topk_recall"] = (hc_hits.sum() / binary.clamp(min=1e-6)).detach()
        components["hc_topk_recall_per_position"] = (hc_hits / count_pos.clamp(min=1)).detach()
    if selector:
        selector_loss = sel / (
            denom if model.path_selector_objective_weighting else counts[1].clamp(min=1e-6)
        )
        loss = loss + model.path_selector_loss_alpha * selector_loss
        components.update(
            selector_loss=selector_loss.detach(),
            selector_recall=(recall / binary.clamp(min=1e-6)).detach(),
            selector_acc=(sel_correct / serving.clamp(min=1e-6)).detach(),
        )
    if model.gate_entropy_weight > 0 and hasattr(model.draft_model, "gate_entropy"):
        loss = loss + model.gate_entropy_weight * model.draft_model.gate_entropy()
    if model.sp_size > 1:
        loss = loss + hidden.sum() * 0
    result = (
        loss,
        correct.sum() / binary.clamp(min=1),
        ce_pos / count_pos.clamp(min=1),
        correct / count_pos.clamp(min=1),
        count_pos,
        components,
    )
    if return_draft:
        return (
            *result,
            {
                "draft_hidden": hidden,
                "proposals": torch.cat(proposal_parts, dim=1),
                "anchor_positions": anchors,
                "block_keep_mask": keep,
            },
        )
    return result
