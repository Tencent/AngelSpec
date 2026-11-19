"""Path selector: predecessor-conditioned reranking of the drafter's top-k tokens.

A block-parallel drafter scores every position independently, so nothing ties the
token it proposes at position ``i+1`` to the token it proposes at position ``i``.
The path selector adds that missing edge: after the unary draft logits nominate
``path_selector_top_k`` candidates per position, it rescores each candidate as a
transition from the position's predecessor token::

    unary(candidate)
      + dot(predecessor_codebook(previous) * hidden_projection(hidden),
            successor_codebook(candidate))

At inference the walk starts from the verified anchor and conditions each position
on the token selected at the preceding one, so one parallel drafter forward still
covers the whole block — only the small selector walks the candidate path.
Training teacher-forces the previous token instead (see ``DFlashModel``).

The transition score is a rank-``path_selector_rank`` factorization of a full
bigram table, kept as two free full-vocabulary codebooks. Architecture and
checkpoint contract follow Z Lab's DFlash2 (MIT-licensed) selector:
https://github.com/z-lab/dflash/blob/07ebd93/dflash/model.py
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def selector_training_candidates(
    candidate_ids: torch.Tensor,  # [..., top_k]
    target_ids: torch.Tensor,  # [...]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the K-way selector label set, injecting a missing target at rank K.

    The unary top-k is what the selector reranks at serving time, so training
    scores exactly that set. When the hard target falls outside it there is no
    valid K-way label, so the weakest candidate is replaced by the target; every
    position then has a well-defined label without growing the selector to the
    full vocabulary.

    Returns ``(training_candidate_ids, target_positions, contains_target)``.
    ``contains_target`` marks the positions whose target was already in the
    original unary top-k — i.e. the ones the selector could actually recover at
    serving time — so metrics can exclude the injected ones.
    """
    top_k = candidate_ids.shape[-1]
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.to(torch.int64).argmax(dim=-1)
    target_positions = torch.where(contains_target, target_positions, top_k - 1)

    training_candidate_ids = candidate_ids.clone()
    training_candidate_ids[..., -1] = torch.where(
        contains_target, training_candidate_ids[..., -1], target_ids
    )
    return training_candidate_ids, target_positions, contains_target


class PathSelector(nn.Module):
    """Bilinear (predecessor, hidden, candidate) scorer applied after unary top-k."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"path_selector_rank must be >= 1, got {rank}")
        if top_k < 2:
            # A 1-way rerank has nothing to choose between, and its K-way CE is
            # identically zero, so the selector would train on no signal at all.
            raise ValueError(f"path_selector_top_k must be >= 2, got {top_k}")
        if top_k > vocab_size:
            raise ValueError(
                f"path_selector_top_k ({top_k}) cannot exceed vocab_size ({vocab_size})"
            )
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        nn.init.normal_(self.predecessor_codebook, std=initializer_range)
        nn.init.normal_(self.successor_codebook, std=initializer_range)
        nn.init.normal_(self.hidden_projection.weight, std=initializer_range)

    def context(self, hidden_states: torch.Tensor, predecessor_ids: torch.Tensor) -> torch.Tensor:
        """Elementwise interaction of the predecessor token and the hidden state.

        Args:
            hidden_states: ``[..., hidden_size]`` drafter hidden states.
            predecessor_ids: ``[...]`` token preceding each position.

        Returns:
            ``[..., rank]`` context vector.
        """
        predecessor = self.predecessor_codebook[predecessor_ids.long()]
        projected_hidden = self.hidden_projection(hidden_states)
        return predecessor * projected_hidden.to(predecessor.dtype)

    def transition_scores(
        self,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score the given candidate IDs as transitions. Returns ``[..., num_candidates]``."""
        context = self.context(hidden_states, predecessor_ids)
        successors = self.successor_codebook[candidate_ids.long()]
        return (context.unsqueeze(-2) * successors).sum(dim=-1)

    def score_candidates(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add transition scores to the unary logits of a candidate subset.

        Args:
            unary_logits: ``[..., vocab_size]`` draft logits.
            hidden_states: ``[..., hidden_size]`` drafter hidden states.
            predecessor_ids: ``[...]`` token preceding each position.
            candidate_ids: ``[..., num_candidates]`` candidates to score.

        Returns:
            ``[..., num_candidates]`` edge scores, in ``unary_logits``' dtype.
        """
        unary_scores = unary_logits.gather(-1, candidate_ids)
        transition = self.transition_scores(hidden_states, predecessor_ids, candidate_ids)
        return unary_scores + transition.to(unary_scores.dtype)

    def select(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take the unary top-k and return ``(candidate_ids, edge_scores)``.

        This is the serving-time entry point: the caller walks a block by feeding
        each position's argmax back as the next position's ``predecessor_ids``.
        """
        candidate_ids = unary_logits.topk(top_k or self.top_k, dim=-1).indices
        scores = self.score_candidates(unary_logits, hidden_states, predecessor_ids, candidate_ids)
        return candidate_ids, scores


def build_path_selector(config) -> Optional[nn.Module]:
    """Build the path selector, or ``None`` when ``enable_path_selector`` is off.

    The selector reranks candidates BEFORE any draft-to-target vocabulary
    mapping, so its codebooks are sized by the full ``vocab_size``.
    """
    if not bool(getattr(config, "enable_path_selector", False)):
        return None

    return PathSelector(
        vocab_size=int(config.vocab_size),
        hidden_size=int(config.hidden_size),
        rank=int(getattr(config, "path_selector_rank", 256)),
        top_k=int(getattr(config, "path_selector_top_k", 16)),
        initializer_range=float(getattr(config, "initializer_range", 0.02)),
    )
