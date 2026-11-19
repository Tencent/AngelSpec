"""Hy4 flash compressed-group index helpers, and their agreement with vLLM's IndexPool."""

import importlib.util

import pytest
import torch

from angelspec.models.ops.sparse_indices import (
    build_contiguous_causal_ranges,
    build_forced_tail_candidates,
    compressed_group_token_offsets,
    compressed_middle_group_ranges,
    compute_compressed_cu_seqlens,
    expand_compressed_groups_to_tokens,
)

RATIO = 4


def test_group_layout_of_packed_documents():
    cu = torch.tensor([0, 10, 13, 21], dtype=torch.int32)
    assert compute_compressed_cu_seqlens(cu, RATIO).tolist() == [0, 2, 2, 4]
    assert compressed_group_token_offsets(cu, RATIO).tolist() == [0, 4, 13, 17]
    groups = torch.tensor([[1, -1], [2, 3]])
    assert expand_compressed_groups_to_tokens(groups, cu, RATIO).tolist() == [
        [4, 5, 6, 7, -1, -1, -1, -1],
        [13, 14, 15, 16, 17, 18, 19, 20],
    ]


def test_no_complete_group_expands_to_nothing():
    cu = torch.tensor([0, 3], dtype=torch.int32)
    groups = torch.full((2, 2), -1)
    assert (expand_compressed_groups_to_tokens(groups, cu, RATIO) == -1).all()


def test_ranges_and_tail_follow_the_query_position():
    starts = torch.tensor([0, 0, 0, 10, 10])
    ends = torch.tensor([1, 4, 7, 15, 10])  # the last row is padding
    doc_groups = torch.tensor([0, 0, 0, 2, 2])
    lo, hi = compressed_middle_group_ranges(starts, ends, doc_groups, RATIO)
    assert lo.tolist() == [0, 0, 0, 2, 2] and hi.tolist() == [0, 0, 1, 3, 2]
    assert build_forced_tail_candidates(starts, ends, RATIO).tolist() == [
        [0, -1, -1, -1],
        [0, 1, 2, 3],
        [4, 5, 6, -1],
        [14, -1, -1, -1],
        [-1, -1, -1, -1],
    ]


@pytest.mark.skipif(
    importlib.util.find_spec("vllm.models.hy_v4_flash") is None, reason="needs vLLM's Hy4 flash"
)
@pytest.mark.parametrize("budget", [3, 20])
def test_candidates_match_vllm_index_pool(budget):
    from vllm.models.hy_v4_flash.nvidia.indexpool_ops import (
        expand_groups,
        select_groups,
    )

    length = 37
    g = torch.Generator().manual_seed(budget)
    logits = torch.randn(length, length // RATIO, generator=g)
    positions = torch.arange(length)
    served = expand_groups(select_groups(logits, positions // RATIO, budget), positions, RATIO)

    cu = torch.tensor([0, length], dtype=torch.int32)
    starts, ends = (x[0].long() for x in build_contiguous_causal_ranges(length))
    lo, hi = compressed_middle_group_ranges(starts, ends, torch.zeros_like(starts), RATIO)
    columns = torch.arange(logits.shape[1])
    scores = logits.masked_fill((columns < lo[:, None]) | (columns >= hi[:, None]), -torch.inf)
    picked = scores.topk(min(budget, logits.shape[1]), dim=-1)
    groups = torch.where(picked.values > -torch.inf, picked.indices, -1)
    trained = torch.cat(
        [
            expand_compressed_groups_to_tokens(groups, cu, RATIO),
            build_forced_tail_candidates(starts, ends, RATIO),
        ],
        dim=-1,
    )
    for row in range(length):
        assert set(trained[row].tolist()) - {-1} == set(served[row].tolist()) - {-1}, row
