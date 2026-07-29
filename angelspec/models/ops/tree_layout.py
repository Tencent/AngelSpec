"""Trunk-collapse tree layout for the packed tree-forward scoring path.

Given a GT trunk and the draft's on-policy block per anchor, this builds ONE
packed row::

    [ trunk (GT prefix, doc 0) | branch_0 | branch_1 | ... | branch_{K-1} ]

and every tensor the downstream needs: doc ids + anchor ids for the tree mask
(``tree_mask.create_tree_mask_mod``), absolute RoPE positions for the cache-less
prefill, and the ``score_index`` slots whose hidden -> lm_head yields the teacher
distribution for each generated branch token.

Score semantics (the single source of off-by-one truth — do NOT change):
``branch_k`` is a block ``[g_0 .. g_{B-1}]`` hanging off trunk anchor ``a_k``.
The predecessor slot that predicts ``g_m`` is

    m == 0  -> the trunk anchor slot ``a_k``      (conditioning prefix == GT)
    m >= 1  -> the slot of ``g_{m-1}`` in branch  (prefix already diverged)

so ``g_0`` is scored off the (recomputed) trunk and ``g_{m>=1}`` off the branch,
all in one packed forward. A branch token that happens to equal the GT token
AFTER a deviation is still scored — its conditioning prefix is the draft prefix,
not the GT one.

This file only builds the layout for a SINGLE trunk (one GT sequence == one
packed row), matching ``tree_mask.py``'s ``d_kv == 0`` == "the trunk" assumption.
Cross-sequence bin-packing is a later, controller-level concern.
"""

from dataclasses import dataclass

import torch


@dataclass
class TreeLayout:
    packed_ids: torch.Tensor  # (N,) long  [trunk | branch_0 | ... | branch_{K-1}]
    doc_ids: torch.Tensor  # (N,) long  0=trunk, k+1=branch_k, -1=padding
    anchor_of: torch.Tensor  # (N,) long  branch token's trunk anchor idx; -1 for trunk/pad
    positions: torch.Tensor  # (N,) long  absolute RoPE positions
    score_index: torch.Tensor  # (M,) long  predecessor slot feeding teacher dist per scored token
    score_target_ids: (
        torch.Tensor
    )  # (M,) long  the g_{k,m} predicted at each score_index (same order)
    cu_seqlens: torch.Tensor  # (2,) long  [0, N] — whole row is a single request


def build_tree_layout(
    trunk_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    branch_tokens: torch.Tensor,
    max_position: int | None = None,
    pad_to: int | None = None,
) -> TreeLayout:
    """Build the packed trunk+branch layout for one GT sequence.

    Args:
        trunk_ids: (L,) long — the GT token sequence (doc 0, the trunk).
        anchor_positions: (K,) long — trunk positions each carrying a branch,
            ascending. Each ``a_k`` is the inclusive upper bound of the trunk the
            branch may attend to (see ``tree_mask``).
        branch_tokens: (K, B) long — the draft's on-policy block per anchor;
            row k is ``[g_{k,0} .. g_{k,B-1}]``.
        max_position: if given, assert ``positions.max() < max_position`` so a
            branch never silently runs past the target's RoPE table (-> NaN).
        pad_to: if given, right-pad the row to this fixed length N (torch.compile
            bucket). Padding is ``doc_id = -1`` (invisible both ways in the mask),
            ``packed_ids = 0``, ``positions = 0``, ``anchor_of = -1``. Never
            padded into ``score_index``.

    Returns:
        A :class:`TreeLayout`. Ordering of ``score_index`` / ``score_target_ids``
        is branch-major then block-major; the trainer gathers student hidden in
        that same order.
    """
    if trunk_ids.dim() != 1:
        raise ValueError(f"trunk_ids must be 1-D (L,), got {tuple(trunk_ids.shape)}")
    if branch_tokens.dim() != 2:
        raise ValueError(f"branch_tokens must be 2-D (K, B), got {tuple(branch_tokens.shape)}")
    if anchor_positions.dim() != 1 or anchor_positions.shape[0] != branch_tokens.shape[0]:
        raise ValueError(
            "anchor_positions must be (K,) matching branch_tokens' K; "
            f"got {tuple(anchor_positions.shape)} vs K={branch_tokens.shape[0]}"
        )

    device = trunk_ids.device
    L = int(trunk_ids.shape[0])
    K, B = int(branch_tokens.shape[0]), int(branch_tokens.shape[1])
    anchors = anchor_positions.tolist()
    if any(not (0 <= a < L) for a in anchors):
        raise ValueError(f"anchor_positions out of trunk range [0, {L}): {anchors}")

    # --- trunk segment (doc 0, plain 0..L-1) ---
    seg_ids = [trunk_ids.to(torch.long)]
    seg_doc = [torch.zeros(L, dtype=torch.long, device=device)]
    seg_anchor = [torch.full((L,), -1, dtype=torch.long, device=device)]
    seg_pos = [torch.arange(L, dtype=torch.long, device=device)]

    score_idx: list[int] = []
    score_tgt: list[int] = []
    offset = L  # flat start of the next branch segment

    # --- branch segments (doc k+1); block offset m -> absolute pos a_k + 1 + m ---
    for k in range(K):
        a = anchors[k]
        blk = branch_tokens[k].to(torch.long)
        seg_ids.append(blk)
        seg_doc.append(torch.full((B,), k + 1, dtype=torch.long, device=device))
        seg_anchor.append(torch.full((B,), a, dtype=torch.long, device=device))
        seg_pos.append(a + 1 + torch.arange(B, dtype=torch.long, device=device))

        # predecessor slot for g_{k,m}: m==0 -> trunk anchor slot a; m>=1 -> branch g_{m-1}.
        for m in range(B):
            score_idx.append(a if m == 0 else offset + (m - 1))
            score_tgt.append(int(blk[m]))
        offset += B

    packed_ids = torch.cat(seg_ids)
    doc_ids = torch.cat(seg_doc)
    anchor_of = torch.cat(seg_anchor)
    positions = torch.cat(seg_pos)
    N = int(packed_ids.shape[0])  # == L + K * B

    if pad_to is not None:
        if pad_to < N:
            raise ValueError(f"pad_to={pad_to} < packed length {N}")
        pad = pad_to - N
        if pad:
            packed_ids = torch.cat([packed_ids, torch.zeros(pad, dtype=torch.long, device=device)])
            doc_ids = torch.cat([doc_ids, torch.full((pad,), -1, dtype=torch.long, device=device)])
            anchor_of = torch.cat(
                [anchor_of, torch.full((pad,), -1, dtype=torch.long, device=device)]
            )
            positions = torch.cat([positions, torch.zeros(pad, dtype=torch.long, device=device)])
        N = pad_to

    if max_position is not None and N and int(positions.max()) >= max_position:
        raise ValueError(
            f"branch positions exceed RoPE bound: max {int(positions.max())} >= {max_position}"
        )

    return TreeLayout(
        packed_ids=packed_ids,
        doc_ids=doc_ids,
        anchor_of=anchor_of,
        positions=positions,
        score_index=torch.tensor(score_idx, dtype=torch.long, device=device),
        score_target_ids=torch.tensor(score_tgt, dtype=torch.long, device=device),
        cu_seqlens=torch.tensor([0, N], dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Rollout -> layout bridge (pure tensor; the trainer glue that feeds the builder
# from the MTP TTT rollout, one row per sequence).
# ---------------------------------------------------------------------------
def select_anchor_positions(
    loss_mask: torch.Tensor,
    max_anchors: int | None = None,
) -> torch.Tensor:
    """Pick the trunk positions to hang branches off — the supervised tokens.

    Args:
        loss_mask: (L,) 0/1 — 1 marks a supervised (assistant) trunk token.
        max_anchors: if given and there are more supervised positions than this,
            evenly subsample down to ``max_anchors`` (keeps them spread across the
            sequence rather than clustering at the front).

    Returns:
        (K,) long, ascending trunk positions. Empty if nothing is supervised.
    """
    if loss_mask.dim() != 1:
        raise ValueError(f"loss_mask must be 1-D (L,), got {tuple(loss_mask.shape)}")
    anchors = torch.nonzero(loss_mask > 0, as_tuple=False).flatten()
    if max_anchors is not None and anchors.numel() > max_anchors > 0:
        # Evenly spaced pick over the supervised indices (endpoints included).
        sel = torch.linspace(0, anchors.numel() - 1, steps=max_anchors).round().long()
        anchors = anchors[sel.unique()]
    return anchors.to(torch.long)


def collect_blocks(
    per_step_argmax: torch.Tensor,
    anchor_positions: torch.Tensor,
) -> torch.Tensor:
    """Gather each anchor's on-policy block from the TTT per-step argmax.

    In the parallel MTP TTT loop, step ``m`` produces an argmax token for EVERY
    position; on-policy step ``m>=1`` already consumed its own step-(m-1) argmax.
    So the block the draft would generate from anchor ``a`` is exactly the column
    ``a`` read down the steps: ``g_{a,m} = per_step_argmax[m, a]``.

    Args:
        per_step_argmax: (B, L) long — row m = the step-m argmax over positions.
        anchor_positions: (K,) long — trunk positions selected as anchors.

    Returns:
        (K, B) long — ``branch_tokens[k, m] = per_step_argmax[m, anchor_k]``.
    """
    if per_step_argmax.dim() != 2:
        raise ValueError(f"per_step_argmax must be 2-D (B, L), got {tuple(per_step_argmax.shape)}")
    return (
        per_step_argmax.index_select(1, anchor_positions.to(torch.long))
        .transpose(0, 1)
        .contiguous()
    )


def build_tree_layout_batch(
    trunk_ids: torch.Tensor,
    per_step_argmax: torch.Tensor,
    loss_mask: torch.Tensor,
    max_anchors: int | None = None,
    max_position: int | None = None,
    pad_to: int | None = None,
) -> list[TreeLayout]:
    """Build one :class:`TreeLayout` per sequence in a training micro-batch.

    Bridges the MTP TTT rollout to the packed builder: for each sequence, select
    supervised anchors, gather their on-policy blocks, and build the layout. One
    packed row per sequence (single trunk) — matching the score_packed forward's
    single-request contract; the trainer dispatches / reads them back in order.

    Args:
        trunk_ids: (Bsz, L) long — GT sequences.
        per_step_argmax: (Bsz, B, L) long — draft TTT per-step argmax per sequence.
        loss_mask: (Bsz, L) 0/1 — supervised positions per sequence.
        max_anchors / max_position / pad_to: forwarded (see the single builders).

    Returns:
        list of length ``Bsz``; a sequence with no supervised token yields a
        trunk-only layout (empty ``score_index``).
    """
    if not (trunk_ids.dim() == 2 and per_step_argmax.dim() == 3 and loss_mask.dim() == 2):
        raise ValueError(
            "expected trunk_ids (Bsz,L), per_step_argmax (Bsz,B,L), loss_mask (Bsz,L); got "
            f"{tuple(trunk_ids.shape)}, {tuple(per_step_argmax.shape)}, {tuple(loss_mask.shape)}"
        )
    layouts: list[TreeLayout] = []
    for b in range(trunk_ids.shape[0]):
        anchors = select_anchor_positions(loss_mask[b], max_anchors)
        blocks = collect_blocks(per_step_argmax[b], anchors)  # (K, B), K may be 0
        layouts.append(
            build_tree_layout(
                trunk_ids[b],
                anchors,
                blocks,
                max_position=max_position,
                pad_to=pad_to,
            )
        )
    return layouts


def build_dflash_opd_layout(
    trunk_ids: torch.Tensor,
    proposals: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_size: int,
    max_position: int | None = None,
    max_anchors: int | None = None,
) -> tuple["TreeLayout", torch.Tensor]:
    """DFlash proposal → OPD tree layout + aligned student draft-hidden slots.

    DFlash's block layout is ``[anchor_slot(0), pred@1 .. pred@{B-1}]``: slot 0 is
    the anchor input, slots 1..B-1 predict absolute positions ``anchor+1..anchor+B-1``.
    So the branch is the B-1 proposals at slots 1..B-1, and the student for branch
    token j (position anchor+1+j) is the draft hidden at block slot j+1.

    ``max_anchors`` caps how many valid anchors are scored (evenly subsampled) —
    the OPD scoring cost scales with anchors*block, so this bounds the packed-tree
    size without touching the base DFlash loss (which still uses all anchors).

    Args:
        trunk_ids: (L,) long — the GT trunk.
        proposals: (n_blocks, block_size) long — draft argmax per within-block slot.
        anchor_positions: (n_blocks,) long.
        block_keep_mask: (n_blocks,) bool — valid anchors.
        block_size: B.

    Returns:
        (layout, student_slots) where ``layout`` is the packed tree over the valid
        anchors' branches (block length B-1), and ``student_slots`` (M,) indexes the
        draft hidden's flat ``[n_blocks*block_size]`` axis at the branch slots,
        ordered branch-major × within-branch to match ``layout.score_index`` /
        ``layout.score_target_ids`` (M = n_valid * (B-1)).
    """
    if proposals.dim() != 2 or proposals.shape[1] != block_size:
        raise ValueError(
            f"proposals must be (n_blocks, {block_size}), got {tuple(proposals.shape)}"
        )
    device = trunk_ids.device
    keep = torch.nonzero(block_keep_mask.bool(), as_tuple=False).flatten()  # valid block indices
    if max_anchors is not None and keep.numel() > max_anchors > 0:
        sel = torch.linspace(0, keep.numel() - 1, steps=max_anchors).round().long().unique()
        keep = keep[sel]
    anchors = anchor_positions.index_select(0, keep)
    branch_tokens = proposals.index_select(0, keep)[:, 1:block_size]  # (n_valid, B-1)
    layout = build_tree_layout(trunk_ids, anchors, branch_tokens, max_position=max_position)
    # Student slots: valid block k contributes flat draft-hidden slots
    # k*B+1 .. k*B+(B-1), in valid-block order — matches build_tree_layout's
    # branch-major × within-branch score_index ordering.
    within = torch.arange(1, block_size, device=device)
    student_slots = (keep.view(-1, 1) * block_size + within.view(1, -1)).reshape(-1)
    return layout, student_slots


@dataclass
class BatchTreeLayout:
    packed_ids: torch.Tensor  # (N,)  all sequences' [trunk | branches] concatenated
    doc_ids: torch.Tensor  # (N,)  globally-unique per (sequence, trunk|branch); -1 pad
    anchor_of: torch.Tensor  # (N,)  GLOBAL flat anchor idx for branch tokens; -1 else
    positions: torch.Tensor  # (N,)  per-sequence absolute RoPE positions
    trunk_doc_of: torch.Tensor  # (N,)  each token's sequence's trunk doc id
    cu_seqlens: torch.Tensor  # (S+1,) per-sequence request boundaries
    score_index: torch.Tensor  # (M,)  GLOBAL flat predecessor slots
    score_target_ids: torch.Tensor  # (M,)
    student_seq: torch.Tensor  # (M,)  which input sequence (0..S-1) each scored token is from
    student_slot: torch.Tensor  # (M,)  flat draft-hidden slot within that sequence


def build_dflash_opd_batch_layout(
    trunk_ids_list,
    proposals_list,
    anchor_positions_list,
    block_keep_mask_list,
    block_size: int,
    max_position: int | None = None,
    max_anchors: int | None = None,
) -> BatchTreeLayout:
    """Pack MANY sequences' DFlash proposal trees into ONE forward (batched OPD).

    Each sequence contributes a ``[trunk | branch_0 | ...]`` segment; segments are
    concatenated and kept mutually invisible via globally-unique doc ids +
    per-token ``trunk_doc_of`` (see ``create_tree_mask_mod``). This replaces the
    per-sequence score_packed RPC (one big forward instead of N serial ones).

    Args: per-sequence lists (length S) of the same tensors ``build_dflash_opd_layout``
    takes. Returns a :class:`BatchTreeLayout`; ``student_seq``/``student_slot`` map
    each scored token back to ``draft_hidden[seq][slot]`` for the KL student.
    """
    S = len(trunk_ids_list)
    device = trunk_ids_list[0].device if S else torch.device("cpu")
    ids, docs, anchors, poss, tdocs = [], [], [], [], []
    sidx, stgt, sseq, sslot = [], [], [], []
    cu = [0]
    doc_cursor = 0
    tok_offset = 0
    for s in range(S):
        layout, student_slots = build_dflash_opd_layout(
            trunk_ids_list[s],
            proposals_list[s],
            anchor_positions_list[s],
            block_keep_mask_list[s],
            block_size,
            max_position=max_position,
            max_anchors=max_anchors,
        )
        n = int(layout.packed_ids.shape[0])
        # branches are local docs 1..K → distinct docs = K+1 (trunk 0 + K branches).
        k = int(layout.doc_ids.max().item()) if layout.doc_ids.numel() else 0
        ids.append(layout.packed_ids)
        docs.append(doc_cursor + layout.doc_ids)  # trunk→cursor, branch j→cursor+j
        anchors.append(
            torch.where(layout.anchor_of >= 0, layout.anchor_of + tok_offset, layout.anchor_of)
        )
        poss.append(layout.positions)
        tdocs.append(torch.full((n,), doc_cursor, dtype=torch.long, device=device))
        m = int(layout.score_index.shape[0])
        if m:
            sidx.append(layout.score_index + tok_offset)
            stgt.append(layout.score_target_ids)
            sseq.append(torch.full((m,), s, dtype=torch.long, device=device))
            sslot.append(student_slots)
        doc_cursor += k + 1
        tok_offset += n
        cu.append(tok_offset)

    def _cat(xs, dtype):
        return torch.cat(xs) if xs else torch.empty((0,), dtype=dtype, device=device)

    return BatchTreeLayout(
        packed_ids=torch.cat(ids),
        doc_ids=torch.cat(docs),
        anchor_of=torch.cat(anchors),
        positions=torch.cat(poss),
        trunk_doc_of=torch.cat(tdocs),
        cu_seqlens=torch.tensor(cu, dtype=torch.long, device=device),
        score_index=_cat(sidx, torch.long),
        score_target_ids=_cat(stgt, torch.long),
        student_seq=_cat(sseq, torch.long),
        student_slot=_cat(sslot, torch.long),
    )
