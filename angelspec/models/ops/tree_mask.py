"""Tree attention mask for the packed tree-forward scoring path.

The scoring forward packs one logical sequence as::

    [ trunk (GT prefix) | branch_0 | branch_1 | ... | branch_{k-1} ]

and runs a single cache-less prefill over it. Each ``branch_i`` is a block the
draft generated on-policy hanging off trunk anchor position ``anchor_i``. All Q
and KV live in the SAME packed row (encoder-only self-attention), so physical
causality is NOT usable — a later branch sits physically after an earlier one
yet must not see it. Visibility is doc-structured instead:

    causal  ∧  ( kv in own doc  ∨  (kv in trunk[doc 0] ∧ kv_pos ≤ anchor_of[q]) )

i.e. a branch token attends to (a) the trunk up to and including its anchor and
(b) its own branch causally — nothing else. Trunk tokens attend to the trunk
causally (plain prefill). Padding (doc_id < 0) is invisible both ways.

The predicate matches ``models/dflash.py::_create_dflash_mask_mod`` in style so
it composes with ``compile_friendly_create_block_mask`` (build the BlockMask on
CUDA with ``_compile=True`` to stay O(num_blocks); on CPU it returns None, so
tests materialise the predicate into a bool grid directly).
"""

from typing import Callable

import torch


def create_tree_mask_mod(
    doc_ids: torch.Tensor,
    anchor_of: torch.Tensor,
    trunk_doc_of: torch.Tensor | None = None,
) -> Callable:
    """Build a FlexAttention ``mask_mod`` for packed trunk+branch scoring.

    Supports both a single sequence's tree (one trunk) and MANY sequences packed
    into one row (batched scoring). In the multi-sequence case each token carries
    its own sequence's trunk doc id via ``trunk_doc_of`` and cross-sequence
    attention falls out (a token only ever sees its own branch doc or its own
    trunk doc), so no separate request isolation is needed.

    Args:
        doc_ids: (B, L) long. GLOBALLY unique per (sequence, trunk|branch): a
            sequence's trunk is one id, each of its branches another id; ``< 0`` =
            padding (invisible). Within a doc, physical order == causal order.
        anchor_of: (B, L) long. For a branch token, the GLOBAL flat index of its
            trunk anchor (inclusive upper bound of the visible trunk). Unused for
            trunk / padding (set to any value, e.g. -1).
        trunk_doc_of: (B, L) long or None. For each token, the doc id of ITS
            sequence's trunk. None = single sequence whose trunk doc id is ``0``
            (backward-compatible with the original single-tree layout).

    Returns:
        ``tree_mask_mod(b, h, q_idx, kv_idx) -> bool`` — vectorised over
        broadcast q_idx/kv_idx, so it works both inside flex_attention and when
        materialised into a [Q, KV] grid for CPU tests.
    """

    def tree_mask_mod(b, h, q_idx, kv_idx):
        d_q = doc_ids[b, q_idx]
        d_kv = doc_ids[b, kv_idx]
        anchor = anchor_of[b, q_idx]
        t_q = 0 if trunk_doc_of is None else trunk_doc_of[b, q_idx]

        both_valid = (d_q >= 0) & (d_kv >= 0)
        kv_is_my_trunk = d_kv == t_q  # kv belongs to q's own sequence's trunk
        is_trunk_q = d_q == t_q  # q is a trunk token of its sequence
        is_branch_q = both_valid & (~is_trunk_q)

        # Trunk query: plain causal within its own trunk.
        trunk_vis = is_trunk_q & kv_is_my_trunk & (kv_idx <= q_idx)

        # Branch query: (a) its own trunk up to and including its anchor, and
        # (b) its own branch causally. Never other branches or other sequences.
        see_trunk = is_branch_q & kv_is_my_trunk & (kv_idx <= anchor)
        see_own = is_branch_q & (d_kv == d_q) & (kv_idx <= q_idx)

        return both_valid & (trunk_vis | see_trunk | see_own)

    tree_mask_mod.__name__ = "tree_mask_mod"
    return tree_mask_mod
