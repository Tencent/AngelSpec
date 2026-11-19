"""Eval-time comparison of a drafter's attention with the target's.

Enabled by ``training.dflash_attn_diag_blocks``.

For drafters that read the target's own MLA latents (``DFlashConfig.target_kv_layers``), draft
layer ``i`` and target layer ``L_i`` score the same context candidates in the same latent space.
Forward hooks on the draft attention modules recompute, for a few blocks per sample, the draft's
attention probabilities over its key list (the block's context candidates, then the block's own
keys) and target layer ``L_i``'s attention over the same context candidates at the query that
predicts the same token: position ``anchor + j - 1`` for draft slot ``j``, from a frozen copy of
that target layer (:class:`~angelspec.models.target.target_attention_init.TargetAttentionTeacher`).
They also count how many of the image tokens before each anchor are among its block's
candidates. The hooks run inside each draft layer's forward, where FSDP has the layer's
parameters unsharded.

Per draft layer, from the sums :func:`summarize_attn_diag` turns into means:

- ``idx_img_coverage``: image tokens among the candidates / image tokens before the anchor
  (``_cut``: only blocks whose context outgrew the indexer's top-k).
- ``draft_{ctx,block,sink}_mass``: where the draft's attention goes.
- ``img_mass_{draft,target}``: share of the attention over context candidates on image tokens.
- ``kl_ctx``: KL(target || draft) over the context candidates, per head, then averaged.
- ``top16_overlap``: shared top-16 positions of the head-averaged distributions.
- ``img_kl`` / ``img_top8_overlap``: the same within the image candidates (rows with >= 16).
- ``selfcheck_relerr``: the recomputed attention output against the kernel's, which pins the
  recomputed probabilities to the ones the drafter actually used.

:class:`AttnOracle` (``training.dflash_attn_oracle``) intervenes instead of observing: it replaces
the drafter's attention with one that spreads the drafter's own mass on its image (or all context)
candidates the way the target does, so the eval's acceptance shows what attending like the target
would be worth.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from angelspec.models.draft.mtp import _apply_rope_split, _interleaved_to_split_rope
from angelspec.models.target.target_indexer import document_starts

_ROW_CHUNK = 32
_ORACLE_CHUNK = 64
_COVERAGE_STATS = (
    "blocks",
    "blocks_img",
    "img_visible",
    "img_selected",
    "blocks_full",
    "blocks_cut",
    "img_visible_cut",
    "img_selected_cut",
)
_ATTENTION_STATS = (
    "rows",
    "ctx_mass",
    "block_mass",
    "sink_mass",
    "kl",
    "top16",
    "img_rows",
    "img_mass_draft",
    "img_mass_target",
    "img16_rows",
    "img_kl",
    "img_top8",
    "selfcheck_err",
)


def image_token_mask(
    input_ids: torch.Tensor, image_token_ids: Optional[Tuple[int, int, int]]
) -> torch.Tensor:
    """``[B, L]`` bool: inside an image span, ``<image_start>`` and ``<image_end>`` included."""
    if image_token_ids is None:
        return torch.zeros_like(input_ids, dtype=torch.bool)
    start, end = image_token_ids[0], image_token_ids[1]
    opened = (input_ids == start).cumsum(-1)
    closed = (input_ids == end).cumsum(-1)
    return ((opened - closed) > 0) | (input_ids == end)


def sink_softmax(logits: torch.Tensor, sink: Optional[torch.Tensor]) -> torch.Tensor:
    """Softmax over the last axis with an optional per-head sink in the denominator.

    ``logits`` is ``[R, H, K]`` with ``-inf`` at unused keys; ``sink`` is ``[H]``. Same
    construction as ``Hy4DraftMLAAttention._softmax_with_sink``.
    """
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    if sink is not None:
        lse = torch.logaddexp(lse, sink.to(logits.dtype).view(1, -1, 1))
    return torch.exp(logits - lse)


def _top_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """``[R]``: the share of ``b``'s top-``k`` keys that are also in ``a``'s (rows ``[R, K]``)."""
    k = min(k, a.shape[-1])
    marked = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, a.topk(k, dim=-1).indices, True)
    return marked.gather(-1, b.topk(k, dim=-1).indices).float().mean(-1)


def _kl(target: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """``[R]``: KL(target || draft) per head over the last axis, averaged over heads."""
    return (torch.xlogy(target, target) - target * draft.clamp(min=1e-12).log()).sum(-1).mean(-1)


def attention_row_stats(
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    is_ctx: torch.Tensor,
    is_img: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Per-row comparison of one chunk of rows; sums over the rows that have context keys.

    Args:
        draft_probs: ``[R, H, K]`` the draft's attention over its key list (sink included,
            so rows sum to at most 1).
        target_logits: ``[R, H, K]`` the target's scores, ``-inf`` outside the context keys.
        is_ctx: ``[R, K]`` keys that are context candidates.
        is_img: ``[R, K]`` keys that are image tokens.
    """
    ctx = is_ctx[:, None, :].to(draft_probs.dtype)
    has_ctx = is_ctx.any(-1)
    ctx_mass = (draft_probs * ctx).sum(-1)  # [R, H]
    block_mass = (draft_probs * (1 - ctx)).sum(-1)
    draft = draft_probs * ctx / ctx_mass.clamp(min=1e-12).unsqueeze(-1)
    target = torch.softmax(target_logits.masked_fill(~has_ctx[:, None, None], 0.0), dim=-1)
    target = target * ctx

    img = (is_img & is_ctx)[:, None, :].to(draft.dtype)
    n_img = (is_img & is_ctx).sum(-1)
    img_rows = has_ctx & (n_img > 0)
    img16 = has_ctx & (n_img >= 16)
    draft_img = draft * img
    target_img = target * img
    draft_in = draft_img / draft_img.sum(-1, keepdim=True).clamp(min=1e-12)
    target_in = target_img / target_img.sum(-1, keepdim=True).clamp(min=1e-12)

    keep = has_ctx.to(draft.dtype)
    stats = {
        "rows": keep.sum(),
        "ctx_mass": (ctx_mass.mean(-1) * keep).sum(),
        "block_mass": (block_mass.mean(-1) * keep).sum(),
        "sink_mass": ((1 - ctx_mass - block_mass).mean(-1) * keep).sum(),
        "kl": (_kl(target, draft) * keep).sum(),
        "top16": (_top_overlap(target.mean(1), draft.mean(1), 16) * keep).sum(),
        "img_rows": img_rows.sum().to(draft.dtype),
        "img_mass_draft": (draft_img.sum(-1).mean(-1) * img_rows).sum(),
        "img_mass_target": (target_img.sum(-1).mean(-1) * img_rows).sum(),
        "img16_rows": img16.sum().to(draft.dtype),
        "img_kl": (_kl(target_in, draft_in) * img16).sum(),
        "img_top8": (_top_overlap(target_in.mean(1), draft_in.mean(1), 8) * img16).sum(),
    }
    return stats


def oracle_mix(
    probs: torch.Tensor, target_logits: torch.Tensor, sel: torch.Tensor
) -> torch.Tensor:
    """``probs`` with the drafter's mass on the ``sel`` keys spread as the target spreads its own.

    ``probs`` / ``target_logits`` are ``[R, H, K]``; ``sel`` is ``[R, K]`` and must lie within the
    keys ``target_logits`` scores. Rows without ``sel`` keys come back unchanged.
    """
    has = sel.any(-1)
    logits = target_logits.masked_fill(~sel[:, None, :], float("-inf"))
    target = torch.softmax(logits.masked_fill(~has[:, None, None], 0.0), dim=-1)
    mass = (probs * sel[:, None, :]).sum(-1, keepdim=True)
    return torch.where(sel[:, None, :], mass * target, probs)


def _layer_inputs(attn: nn.Module, kwargs: dict) -> Optional[SimpleNamespace]:
    """One draft attention call's keys, latents and positions, from its forward hook's kwargs."""
    keys = kwargs.get("sparse_kv_indices")
    latent_ctx = kwargs.get("context_latent")
    if keys is None or latent_ctx is None:
        return None
    draft_hidden = kwargs["draft_hidden"]
    draft_pos = kwargs["draft_position_ids"]
    ctx_pos = kwargs["context_position_ids"]
    ctx_len = latent_ctx.shape[1]
    rope_len = max(
        ctx_len + draft_hidden.shape[1],
        int(draft_pos.max().item()) + 1,
        int(ctx_pos.max().item()) + 1,
    )
    cos, sin = attn._cos_sin(draft_hidden, draft_pos, rope_len)
    latent_draft = attn._project_kv_latent(draft_hidden, cos, sin)
    return SimpleNamespace(
        keys=keys,
        latent=torch.cat([latent_ctx.to(latent_draft.dtype), latent_draft], dim=1),
        draft_hidden=draft_hidden,
        draft_pos=draft_pos,
        ctx_pos=ctx_pos,
        ctx_len=ctx_len,
        rope_len=rope_len,
    )


def _chunk_scores(attn, teacher, layer, target_hidden, anchors, block_size, b, chunk):
    """For draft rows ``chunk`` of batch row ``b``: their keys, the gathered latents, the drafter's
    attention probabilities and the target layer's scores (``-inf`` outside the context keys)."""
    key = layer.keys[b, chunk].long()
    valid = key >= 0
    lat = layer.latent[b, key.clamp(min=0)].float()  # [R, K, kv_lora + rope]

    q_nope, q_pe, _ = attn._project_q(layer.draft_hidden[b : b + 1, chunk])
    cos_q, sin_q = attn._cos_sin(q_nope, layer.draft_pos[b : b + 1, ..., chunk], layer.rope_len)
    q_pe = _apply_rope_split(_interleaved_to_split_rope(q_pe), cos_q, sin_q)
    q = attn._absorb_q(q_nope, q_pe)[0].float()  # [R, H, kv_lora + rope]
    sink = attn.learnable_sink_param.float() if attn.learnable_sink else None
    logits = torch.einsum("rhm,rkm->rhk", q, lat) * attn.scaling
    probs = sink_softmax(logits.masked_fill(~valid[:, None, :], float("-inf")), sink)

    query = (anchors[b, chunk // block_size] + chunk % block_size - 1).clamp(min=0)
    normed = teacher.input_layernorm(
        target_hidden[b, query].to(teacher.input_layernorm.weight.dtype)
    ).unsqueeze(0)
    t_nope, t_pe, _ = teacher.self_attn._project_q(normed)
    pos = layer.ctx_pos[b : b + 1, ..., query]
    cos_t, sin_t = teacher.self_attn._cos_sin(t_nope.float(), pos, layer.rope_len)
    t_pe = _apply_rope_split(_interleaved_to_split_rope(t_pe.float()), cos_t, sin_t)
    t_q = teacher.self_attn._absorb_q(t_nope, t_pe.to(t_nope.dtype))[0].float()
    is_ctx = valid & (key < layer.ctx_len)
    t_logits = torch.einsum("rhm,rkm->rhk", t_q, lat) * teacher.self_attn.scaling
    return key, lat, probs, t_logits.masked_fill(~is_ctx[:, None, :], float("-inf")), is_ctx


def _attention_output(attn, layer, b, chunk, probs, lat):
    """The drafter's attention output for rows ``chunk`` under ``probs``, as its kernel does."""
    _, w_vc = attn._kv_b_weights()
    mixed = torch.einsum("rhk,rkm->rhm", probs, lat[..., : attn.kv_lora_rank])
    out = torch.einsum("rhm,hdm->rhd", mixed.to(w_vc.dtype), w_vc)
    return attn._output_tail(out.unsqueeze(0), layer.draft_hidden[b : b + 1, chunk])[0]


class AttnDiag:
    """Hooks one forward of a ``target_kv_layers`` drafter and sums the statistics above."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        teachers: Sequence[nn.Module],
        names: Sequence[str],
        target_hidden: Sequence[torch.Tensor],
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
        block_size: int,
        num_blocks: int,
        index_topk: Optional[int],
        image_token_ids: Optional[Tuple[int, int, int]],
    ):
        self.layers = list(layers)
        self.teachers = list(teachers)
        self.names = list(names)
        self.target_hidden = list(target_hidden)
        self.block_size = int(block_size)
        self.index_topk = index_topk
        self.anchors = anchor_positions.long()
        self.keep = block_keep_mask.bool()
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        self.is_img = image_token_mask(input_ids, image_token_ids)
        cum_img = torch.cat(
            [torch.zeros(bsz, 1, dtype=torch.long, device=device), self.is_img.long().cumsum(-1)],
            dim=-1,
        )
        anchors = self.anchors.clamp(0, seq_len - 1)
        doc_start = document_starts(ctx_doc_ids, bsz, seq_len, device).gather(1, anchors)
        self.visible = (self.anchors - doc_start).clamp(min=0)
        self.visible_img = cum_img.gather(1, anchors) - cum_img.gather(1, doc_start)

        # Up to num_blocks evenly spaced live blocks per row, every slot whose label is supervised.
        self.rows: List[torch.Tensor] = []
        offsets = torch.arange(1, self.block_size, device=device)
        for b in range(bsz):
            live = self.keep[b].nonzero(as_tuple=True)[0]
            if len(live) > num_blocks:
                live = live[
                    torch.linspace(0, len(live) - 1, num_blocks, device=device).round().long()
                ]
            labels = self.anchors[b, live].unsqueeze(-1) + offsets
            ok = labels < seq_len
            ok &= (
                loss_mask[b].gather(0, labels.clamp(max=seq_len - 1).flatten()).view_as(labels) > 0
            )
            rows = live.unsqueeze(-1) * self.block_size + offsets
            self.rows.append(rows[ok])

        zero = torch.zeros((), dtype=torch.float64, device=device)
        self.sums: Dict[str, torch.Tensor] = {
            f"attn_diag/{name}/{stat}": zero.clone()
            for name in self.names
            for stat in (*_COVERAGE_STATS, *_ATTENTION_STATS)
        }

    def _add(self, name: str, stats: Dict[str, torch.Tensor]) -> None:
        for stat, value in stats.items():
            self.sums[f"attn_diag/{name}/{stat}"] += value.double()

    @contextmanager
    def hooks(self):
        handles = [
            layer.self_attn.register_forward_hook(self._hook(i), with_kwargs=True)
            for i, layer in enumerate(self.layers)
        ]
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def _hook(self, index: int):
        def hook(module, args, kwargs, output):
            with torch.no_grad():
                self._observe(index, module, kwargs, output)

        return hook

    def _observe(self, index: int, attn: nn.Module, kwargs: dict, output: torch.Tensor) -> None:
        layer = _layer_inputs(attn, kwargs)
        if layer is None:
            return
        name, teacher = self.names[index], self.teachers[index]
        for b in range(layer.draft_hidden.shape[0]):
            self._add(name, self._coverage(b, layer.keys[b], layer.ctx_len))
            rows = self.rows[b]
            # A zero-loss sample (eval padding) has no rows; split() would still hand out
            # one empty chunk.
            if rows.numel() == 0:
                continue
            for chunk in rows.split(_ROW_CHUNK):
                key, lat, probs, t_logits, is_ctx = _chunk_scores(
                    attn,
                    teacher,
                    layer,
                    self.target_hidden[index],
                    self.anchors,
                    self.block_size,
                    b,
                    chunk,
                )
                recomputed = _attention_output(attn, layer, b, chunk, probs, lat)
                actual = output[b, chunk].float()
                err = (recomputed.float() - actual).norm(dim=-1) / actual.norm(dim=-1).clamp(
                    min=1e-6
                )
                is_img = self.is_img[b, key.clamp(0, layer.ctx_len - 1)] & is_ctx
                stats = attention_row_stats(probs, t_logits, is_ctx, is_img)
                stats["selfcheck_err"] = (err * is_ctx.any(-1)).sum()
                self._add(name, stats)

    def _coverage(self, b: int, keys: torch.Tensor, ctx_len: int) -> Dict[str, torch.Tensor]:
        """Image tokens among each live block's context candidates (slot 0's key row)."""
        live = self.keep[b].nonzero(as_tuple=True)[0]
        key = keys[live * self.block_size].long()
        is_ctx = (key >= 0) & (key < ctx_len)
        selected = (self.is_img[b, key.clamp(0, ctx_len - 1)] & is_ctx).sum(-1)
        visible = self.visible_img[b, live]
        has_img = visible > 0
        cut = (
            (self.visible[b, live] > self.index_topk)
            if self.index_topk
            else torch.zeros_like(has_img)
        )
        counts = {
            "blocks": torch.ones_like(visible),
            "blocks_img": has_img,
            "img_visible": visible,
            "img_selected": selected,
            "blocks_full": has_img & (selected >= visible),
            "blocks_cut": cut,
            "img_visible_cut": visible * cut,
            "img_selected_cut": selected * cut,
        }
        return {stat: count.double().sum() for stat, count in counts.items()}

    def stats(self) -> Dict[str, torch.Tensor]:
        return dict(self.sums)


class AttnOracle:
    """Hooks one forward and replaces chosen draft layers' attention (``dflash_attn_oracle``).

    ``"image"`` spreads the drafter's attention mass on its image candidates the way target layer
    ``L_i`` spreads its own over them at the query that predicts the same token, ``"context"`` does
    the same over all context candidates, and ``"identity"`` recomputes the attention unchanged, a
    control for the recomputation. The drafter keeps its own mass on its block keys and the sink.
    Every live row is replaced, so the eval's acceptance is what the drafter, as trained, would
    reach attending like the target.
    """

    MODES = ("identity", "image", "context")

    def __init__(
        self,
        mode: str,
        layers: Sequence[nn.Module],
        teachers: Sequence[nn.Module],
        target_hidden: Sequence[torch.Tensor],
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        block_size: int,
        image_token_ids: Optional[Tuple[int, int, int]],
        layer_indices: Optional[Sequence[int]] = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"dflash_attn_oracle must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.layers = list(layers)
        self.teachers = list(teachers)
        self.target_hidden = list(target_hidden)
        self.block_size = int(block_size)
        self.anchors = anchor_positions.long()
        indices = range(len(self.layers)) if layer_indices is None else layer_indices
        self.layer_indices = [int(i) for i in indices]
        if any(not 0 <= i < len(self.layers) for i in self.layer_indices):
            raise ValueError(
                f"dflash_attn_oracle_layers {self.layer_indices} outside the drafter's layers"
            )
        self.is_img = image_token_mask(input_ids, image_token_ids)
        offsets = torch.arange(self.block_size, device=input_ids.device)
        keep = block_keep_mask.bool()
        self.rows = [
            (keep[b].nonzero(as_tuple=True)[0].unsqueeze(-1) * self.block_size + offsets).flatten()
            for b in range(input_ids.shape[0])
        ]

    @contextmanager
    def hooks(self):
        handles = [
            self.layers[i].self_attn.register_forward_hook(self._hook(i), with_kwargs=True)
            for i in self.layer_indices
        ]
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def _hook(self, index: int):
        def hook(module, args, kwargs, output):
            with torch.no_grad():
                return self._replace(index, module, kwargs, output)

        return hook

    def _replace(self, index: int, attn: nn.Module, kwargs: dict, output: torch.Tensor):
        layer = _layer_inputs(attn, kwargs)
        if layer is None:
            return None
        new = output.clone()
        for b, rows in enumerate(self.rows):
            if rows.numel() == 0:
                continue
            for chunk in rows.split(_ORACLE_CHUNK):
                key, lat, probs, t_logits, is_ctx = _chunk_scores(
                    attn,
                    self.teachers[index],
                    layer,
                    self.target_hidden[index],
                    self.anchors,
                    self.block_size,
                    b,
                    chunk,
                )
                if self.mode != "identity":
                    sel = is_ctx
                    if self.mode == "image":
                        sel = sel & self.is_img[b, key.clamp(0, layer.ctx_len - 1)]
                    probs = oracle_mix(probs, t_logits, sel)
                new[b, chunk] = _attention_output(attn, layer, b, chunk, probs, lat).to(new.dtype)
        return new

    def stats(self) -> Dict[str, torch.Tensor]:
        return {}


def summarize_attn_diag(sums: Dict[str, float]) -> Dict[str, float]:
    """Summed statistics (keys ``attn_diag/<layer>/<stat>``) -> per-layer means."""
    out: Dict[str, float] = {}
    layers = sorted({key.split("/")[1] for key in sums if key.startswith("attn_diag/")})
    ratios = {
        "idx_img_coverage": ("img_selected", "img_visible"),
        "idx_img_coverage_cut": ("img_selected_cut", "img_visible_cut"),
        "idx_full_cover_share": ("blocks_full", "blocks_img"),
        "idx_cut_share": ("blocks_cut", "blocks"),
        "draft_ctx_mass": ("ctx_mass", "rows"),
        "draft_block_mass": ("block_mass", "rows"),
        "draft_sink_mass": ("sink_mass", "rows"),
        "img_mass_draft": ("img_mass_draft", "img_rows"),
        "img_mass_target": ("img_mass_target", "img_rows"),
        "kl_ctx": ("kl", "rows"),
        "top16_overlap": ("top16", "rows"),
        "img_kl": ("img_kl", "img16_rows"),
        "img_top8_overlap": ("img_top8", "img16_rows"),
        "selfcheck_relerr": ("selfcheck_err", "rows"),
    }
    for layer in layers:
        got = {
            key.split("/")[2]: float(value)
            for key, value in sums.items()
            if key.startswith(f"attn_diag/{layer}/")
        }
        for metric, (num, den) in ratios.items():
            den_sum = got.get(den, 0.0)
            out[f"attn_diag/{layer}/{metric}"] = (
                got.get(num, 0.0) / den_sum if den_sum > 0 else float("nan")
            )
        out[f"attn_diag/{layer}/rows"] = got.get("rows", 0.0)
        out[f"attn_diag/{layer}/img_rows"] = got.get("img_rows", 0.0)
    return out
