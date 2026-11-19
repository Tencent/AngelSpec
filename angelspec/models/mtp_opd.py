"""On-policy, independent MTP branches over one shared Hy4 flash prefix.

Anchor a predicts token a+1. Prefix column j consumes token j+1 and target
post-norm hidden j. Branches use only columns <a and their own generated chain.
The compressed indexer is recomputed on that logical sequence at EVERY step.
"""

from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from angelspec.models.draft.mtp import (
    _apply_rope_split,
    _interleaved_to_split_rope,
    _mtp_seq_all_gather,
)
from angelspec.models.ops.tilelang_dsa import (
    tilelang_lightning_indexer,
    tilelang_sparse_mla,
)
from angelspec.utils.distributed import get_draft_sp_group


def log_probs(hidden, head):
    # Sampling and replay intentionally share FP32 projection AND normalization.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.log_softmax(F.linear(hidden.float(), head.float()), dim=-1)


def mtp_opd_loss(student, teacher, head, *, loss_type="lk_alpha", chunk_size=64):
    """Full-vocabulary overlap losses on [anchor, draft_step, hidden] chains.

    alpha = sum_v min(p_teacher, q_draft). LK uses -log(alpha); e2e-TV uses
    1 - cumprod(alpha), differentiating through all preceding draft steps.
    Projection chunks are checkpointed before joining the per-position overlaps,
    so chain gradients cross chunk boundaries without retaining full logits.
    The caller masks padding and normalizes by valid anchors * draft steps.
    """
    if loss_type not in ("lk_alpha", "e2e_tv"):
        raise ValueError(f"Unsupported MTP OPD loss: {loss_type!r}")
    if student.ndim != 3 or teacher.shape != student.shape or student.shape[1] < 1:
        raise ValueError("OPD requires matching [anchors, steps, hidden] tensors with steps >= 1")
    if head.ndim != 2 or head.shape[1] != student.shape[-1] or head.shape[0] < 1:
        raise ValueError("OPD head must have shape [vocabulary, hidden]")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("OPD loss chunk_size must be a positive integer")
    shape = student.shape[:-1]
    if student.shape[0] == 0:
        empty = student.float().sum(-1) * 0.0
        return empty, empty.detach()
    head = head.detach()
    student, teacher = student.flatten(0, -2), teacher.detach().flatten(0, -2)

    def rows(s, t):
        q, p = log_probs(s, head), log_probs(t, head)
        return torch.logsumexp(torch.minimum(p, q), dim=-1)

    chunks = []
    for s, t in zip(student.split(chunk_size), teacher.split(chunk_size)):
        log_alpha = checkpoint(rows, s, t, use_reentrant=False) if s.requires_grad else rows(s, t)
        chunks.append(log_alpha)
    log_alpha = torch.cat(chunks).view(shape)
    if loss_type == "lk_alpha":
        losses = -log_alpha
    else:
        # log(alpha) <= 0 analytically; clamp only positive FP32 roundoff.
        # -expm1(sum(log(alpha))) is 1 - cumprod(alpha), also accurate near 1.
        log_alpha = log_alpha.clamp_max(0.0)
        losses = -torch.expm1(log_alpha.cumsum(dim=-1))
    return losses, log_alpha.detach().exp()


class UnrollOutput(NamedTuple):
    hidden: torch.Tensor
    drafts: torch.Tensor


class BranchIndexer:
    """Share complete prefix groups; pool groups crossing each anchor separately."""

    @torch.no_grad()
    def __init__(self, indexer, prefix_keys, use_fp8=False):
        self.indexer = indexer
        self.prefix = prefix_keys
        self.length = prefix_keys.shape[1]
        self.use_fp8 = use_fp8
        ix = indexer
        keys, cos, sin = prefix_keys.split([ix.key_input_dim, ix.rope_dim, ix.rope_dim], -1)
        cu = torch.tensor([0, self.length], dtype=torch.int32, device=keys.device)
        pooled, _, starts = ix._pool(keys[0], cu)
        self.pooled = ix._rope_last(pooled[None, :, None], cos[:, starts], sin[:, starts])[0, :, 0]
        if use_fp8 and len(self.pooled):
            from angelspec.models.ops.opd_indexer import quantize_keys

            self.quantized = quantize_keys(self.pooled)

    @torch.no_grad()
    def select(self, xn, q_resid, cos, sin, anchors, chain_keys):
        ix, r = self.indexer, self.indexer.ratio
        n = anchors.numel()
        step = chain_keys.shape[1] if chain_keys is not None else 0
        q = ix.query_projection(q_resid).view(1, n, ix.n_heads, ix.head_dim)
        q = ix._rope_last(q, cos, sin)[0].to(torch.bfloat16)
        weights = ix.head_weight_projection(xn)[0].float() * (ix.n_heads**-0.5 * ix.head_dim**-0.5)
        current = (anchors + step - 1) // r
        stable = torch.minimum(anchors // r, current)
        k = ix.group_topk - 1
        if self.pooled.shape[0]:
            if self.use_fp8:
                from angelspec.inference.engine.opd_vllm import stable_coarse_topk
                from angelspec.models.ops.opd_indexer import fp8_coarse_logits

                logits = fp8_coarse_logits(q, self.quantized, weights, stable)
                groups = stable_coarse_topk(logits, stable, k)
                scores = logits.gather(-1, groups.long().clamp_min(0))
            else:
                scores, groups = tilelang_lightning_indexer(
                    q,
                    self.pooled,
                    weights,
                    torch.zeros_like(stable),
                    stable,
                    k,
                )
            scores = scores.masked_fill(groups < 0, -torch.inf)
        else:
            groups = anchors.new_full((n, k), -1)
            scores = weights.new_full((n, k), -torch.inf)

        # At most ceil((r-1 + step)/r) new complete groups. None contains
        # future ground truth: gather logical positions from this branch only.
        if step:
            extra_count = (r - 1 + step) // r
            extra_groups = anchors[:, None] // r + torch.arange(extra_count, device=anchors.device)
            logical = extra_groups[..., None] * r + torch.arange(r, device=anchors.device)
            from_prefix = logical < anchors[:, None, None]
            prefix = self.prefix[0, logical.clamp(max=self.length - 1)]
            chain_index = (logical - anchors[:, None, None]).clamp(0, step - 1)
            branch = chain_keys[torch.arange(n, device=anchors.device)[:, None, None], chain_index]
            inputs = torch.where(from_prefix[..., None], prefix, branch)
            tk, kc, ks = inputs.split([ix.key_input_dim, ix.rope_dim, ix.rope_dim], -1)
            values = F.linear(tk.float(), ix.key_value_projection.weight.float())
            gates = (
                F.linear(tk.float(), ix.key_gate_projection.weight.float())
                + ix.intra_group_ape.float()
            )
            pooled = (values * gates.softmax(dim=-2)).sum(-2)
            pooled = F.rms_norm(
                pooled, (ix.head_dim,), ix.key_norm.weight.float(), ix.key_norm.eps
            )
            pooled = ix._rope_last(
                pooled.to(torch.bfloat16).reshape(1, -1, 1, ix.head_dim),
                kc[:, :, 0].reshape(1, -1, ix.rope_dim),
                ks[:, :, 0].reshape(1, -1, ix.rope_dim),
            ).reshape(n, extra_count, ix.head_dim)
            if self.use_fp8:
                from angelspec.models.ops.opd_indexer import (
                    fp8_coarse_logits,
                    quantize_keys,
                )

                extra_logits = fp8_coarse_logits(q, quantize_keys(pooled.flatten(0, 1)), weights)
                columns = torch.arange(n, device=q.device)[:, None] * extra_count + torch.arange(
                    extra_count, device=q.device
                )
                extra_scores = extra_logits.gather(-1, columns)
            else:
                extra_scores = (
                    torch.einsum("nhd,ngd->nhg", q.float(), pooled.float()).relu()
                    * weights[..., None]
                ).sum(1)
            extra_scores.masked_fill_(extra_groups >= current[:, None], -torch.inf)
            all_scores = torch.cat([scores, extra_scores], -1)
            all_groups = torch.cat([groups.long(), extra_groups], -1)
            if self.use_fp8:
                # Canonical group order before stable score sorting, including
                # groups that became complete across this branch's anchor.
                group_order = all_groups.argsort(dim=-1, stable=True)
                all_groups = all_groups.gather(-1, group_order)
                all_scores = all_scores.gather(-1, group_order)
                order = all_scores.argsort(dim=-1, descending=True, stable=True)[:, :k]
                top_scores = all_scores.gather(-1, order)
            else:
                top_scores, order = all_scores.topk(k, dim=-1)
            groups = all_groups.gather(-1, order).masked_fill(~top_scores.isfinite(), -1)

        recalled = (groups[..., None] * r + torch.arange(r, device=anchors.device)).flatten(1)
        recalled = recalled.masked_fill(groups.repeat_interleave(r, -1) < 0, -1)
        tail = current[:, None] * r + torch.arange(r, device=anchors.device)
        tail = tail.masked_fill(tail >= (anchors + step)[:, None], -1)
        logical = torch.cat([recalled, tail], -1)
        # Prefix [0,L); chain is stored step-major [L+s*N+anchor].
        physical = torch.where(
            logical < anchors[:, None],
            logical,
            self.length
            + (logical - anchors[:, None]) * n
            + torch.arange(n, device=anchors.device)[:, None],
        )
        return physical.sort(-1).values.to(torch.int32)[None, :, None]


class MTPOPDModel(nn.Module):
    def __init__(
        self,
        draft_model,
        gradient_checkpointing=True,
        prefix_chunk_size=2048,
        fp8_indexer=False,
        attention_backend="sdpa",
    ):
        super().__init__()
        self.draft_model = draft_model
        self.gradient_checkpointing = gradient_checkpointing
        self.prefix_chunk_size = prefix_chunk_size
        self.fp8_indexer = fp8_indexer
        a = draft_model.midlayer.self_attn
        if not a.use_compressed_indexer or a.dsa_backend != "tilelang":
            raise ValueError("draft-OPD currently requires Hy4 flash compressed tilelang MLA")
        self.sp_group = get_draft_sp_group() if attention_backend == "usp" else None
        self.sp_size = dist.get_world_size(self.sp_group) if self.sp_group is not None else 1
        self.sp_rank = dist.get_rank(self.sp_group) if self.sp_group is not None else 0

    def _fuse(self, ids, hidden, pos, mask_zero=False):
        d = self.draft_model
        emb = d.embed_input_ids(ids)
        if mask_zero:
            zero = (pos == 0).all(1) if pos.ndim == 3 else pos == 0
            emb = emb.masked_fill(zero[..., None], 0)
        return d.eh_proj(torch.cat([d.enorm(emb), d.hnorm(hidden)], -1))

    def _prefix(self, ids, hidden, positions):
        a = self.draft_model.midlayer.self_attn
        x = self._fuse(ids, hidden, positions, mask_zero=True)
        xn = self.draft_model.midlayer.input_layernorm(x)
        cos, sin = a._cos_sin(xn, positions, ids.shape[-1])
        kv, pe = a.kv_a_proj_with_mqa(xn).split([a.kv_lora_rank, a.qk_rope_head_dim], -1)
        kv = a.kv_a_layernorm(kv)
        pe = _apply_rope_split(_interleaved_to_split_rope(pe[:, None]), cos, sin).transpose(1, 2)
        return torch.cat([kv[:, :, None], pe], -1), a.indexer.project_keys(xn, cos, sin)

    def _step(self, residual, positions, prefix, chain, selector, anchors, chain_index):
        d, a = self.draft_model, self.draft_model.midlayer.self_attn
        xn = d.midlayer.input_layernorm(residual)
        qn, qp, kv, kp, qr = a._project_absorbed(xn)
        cos, sin = a._cos_sin(xn, positions, prefix.shape[1] + len(chain))
        q, kv, wv = a._assemble_absorbed(qn, qp, kv, kp, cos, sin)
        if chain:
            # Placeholder marks that this step is past the initial prefix column.
            chain = [*chain[:-1], kv]
            chain_index = [*chain_index, a.indexer.project_keys(xn, cos, sin)[0]]
        indices = selector.select(
            xn,
            qr,
            cos,
            sin,
            anchors,
            torch.stack(chain_index, 1) if chain_index else None,
        )
        all_kv = torch.cat([prefix, *chain], 1) if chain else prefix
        out = tilelang_sparse_mla(
            q.to(torch.bfloat16),
            all_kv.to(torch.bfloat16),
            indices,
            scaling=a.scaling,
            sink=a.learnable_sink_param if a.learnable_sink else None,
        )[0]
        out = torch.einsum("bnhm,hdm->bnhd", out.to(wv.dtype), wv)
        if a.gated_mla:
            gate = a.linear_gate(xn).view(1, -1, a.num_heads, a.gate_projection_size)
            out = out * gate.sigmoid()
        x = residual + a._output_proj(out.flatten(2))
        x = x + d.midlayer.mlp(d.midlayer.post_attention_layernorm(x))
        return d.final_layernorm(x), kv, (chain_index[-1] if chain_index else None)

    def forward(
        self,
        input_ids,
        target_hidden,
        anchors,
        num_steps=5,
        drafts=None,
        position_ids=None,
        generator=None,
        sample_chunk=64,
    ):
        if input_ids.ndim != 1 or target_hidden.ndim != 2 or anchors.numel() == 0:
            raise ValueError("One unpacked conversation and nonempty anchors required")
        if bool(((anchors < 1) | (anchors > len(input_ids) - 2)).any()):
            raise ValueError("Anchor outside [1, sequence_length-2]")
        a = self.draft_model.midlayer.self_attn
        positions = position_ids
        if positions is None:
            if a.uses_mrope:
                raise ValueError("MRoPE positions required for Hy4 flash OPD")
            positions = torch.arange(len(input_ids), device=input_ids.device)[None]
        length = anchors.max().clone()
        if self.sp_size > 1:
            dist.all_reduce(length, op=dist.ReduceOp.MAX, group=self.sp_group)
        length = int(length)
        shard = (length + self.sp_size - 1) // self.sp_size
        start = shard * self.sp_rank
        # Equal projection/chunk counts on every rank, including padding.
        cols = torch.arange(start, start + shard, device=input_ids.device).clamp(max=length - 1)
        kvs, iks = [], []
        for cc in cols.split(self.prefix_chunk_size):
            args = (input_ids[cc + 1][None], target_hidden[cc][None], positions[..., cc])
            out = (
                checkpoint(self._prefix, *args, use_reentrant=False)
                if self.gradient_checkpointing and torch.is_grad_enabled()
                else self._prefix(*args)
            )
            kvs.append(out[0])
            iks.append(out[1])
        prefix, ik = torch.cat(kvs, 1), torch.cat(iks, 1)
        if self.sp_size > 1:
            prefix = _mtp_seq_all_gather(prefix, self.sp_group)
            ik = _mtp_seq_all_gather(ik, self.sp_group)
        prefix, ik = prefix[:, :length], ik[:, :length]
        selector = BranchIndexer(a.indexer, ik, use_fp8=self.fp8_indexer)
        base_pos = positions[..., anchors - 1]
        residual = self._fuse(
            input_ids[anchors][None], target_hidden[anchors - 1][None], base_pos, mask_zero=True
        )
        chain, chain_index, hiddens, tokens = [], [], [], []
        for step in range(num_steps):
            if step:
                residual = self._fuse(tokens[-1][None], hiddens[-1], base_pos + step)
            args = (
                residual,
                base_pos + step,
                prefix,
                [*chain, None] if step else [],
                selector,
                anchors,
                list(chain_index),
            )
            out = (
                checkpoint(self._step, *args, use_reentrant=False)
                if self.gradient_checkpointing and torch.is_grad_enabled()
                else self._step(*args)
            )
            hidden, kv, index = out
            hiddens.append(hidden)
            if step:
                chain.append(kv)
                chain_index.append(index)
            if drafts is not None:
                token = drafts[:, step]
            else:
                with torch.no_grad():
                    token = torch.cat(
                        [
                            torch.multinomial(
                                log_probs(h, self.draft_model.lm_head_weight).exp(),
                                1,
                                generator=generator,
                            ).squeeze(-1)
                            for h in hidden[0].split(sample_chunk)
                        ]
                    )
            tokens.append(token)
        return UnrollOutput(torch.stack(hiddens, 2)[0], torch.stack(tokens, 1))
