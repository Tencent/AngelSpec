import os
import types
import unittest
from unittest import mock

import torch

from angelspec.models.ops.sparse_mla import torch_lightning_indexer
from angelspec.models.ops.tilelang_dsa import (
    _indexer_query_chunk,
    tilelang_lightning_indexer,
)
from angelspec.utils.env import get_angelspec_env_vars


def fake_kernels(calls):
    """The logits kernel's contract: ``[rows, keys]`` fp32 scores, defined only inside each
    row's ``[start, end)``. Outside it the fake writes huge scores, which top-k would pick
    if the wrapper did not mask them."""

    def lightning_indexer_logits(index_q, index_k, weights, starts, ends):
        rows, heads = weights.shape
        calls.append((index_q, index_k, weights, starts, ends))
        q = index_q.view(rows, heads, -1).float()
        logits = (torch.einsum("qhd,kd->qhk", q, index_k.float()).relu() * weights[..., None]).sum(
            1
        )
        columns = torch.arange(index_k.shape[0])
        inside = (columns >= starts[:, None]) & (columns < ends[:, None])
        return torch.where(inside, logits, torch.full_like(logits, 1e9))

    return types.SimpleNamespace(lightning_indexer_logits=lightning_indexer_logits)


class TestIndexerQueryChunk(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.q = torch.randn(7, 2, 4).bfloat16()
        self.k = torch.randn(11, 4).bfloat16()
        self.weights = torch.rand(7, 2)
        self.starts = torch.tensor([0, 0, 3, 3, 3, 8, 8])
        self.ends = torch.tensor([2, 4, 5, 6, 8, 8, 11])
        self.calls = []
        patcher = mock.patch(
            "angelspec.models.ops.tilelang_dsa._kernels", return_value=fake_kernels(self.calls)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def call(self, **kwargs):
        return tilelang_lightning_indexer(
            self.q, self.k, self.weights, self.starts, self.ends, kwargs.pop("topk", 4), **kwargs
        )

    def test_chunks_match_the_torch_reference(self):
        for chunk in (1, 2, 3, 7, 50):
            for topk in (4, 16):
                with self.subTest(chunk=chunk, topk=topk):
                    self.calls.clear()
                    scores, indices = self.call(query_chunk_size=chunk, topk=topk)
                    ref_scores, ref_indices = torch_lightning_indexer(
                        self.q, self.k, self.weights, self.starts, self.ends, topk
                    )
                    torch.testing.assert_close(scores, ref_scores)
                    torch.testing.assert_close(indices, ref_indices)
                    self.assertEqual(indices.dtype, torch.int32)
                    self.assertTrue(bool((indices[5] == -1).all()))
                    self.assertEqual(len(self.calls), -(-7 // chunk))

    def test_rows_are_padded_to_the_kernel_block_with_the_last_range(self):
        self.call(query_chunk_size=3)
        rows_per_block = 128 // self.q.shape[1]
        for index_q, index_k, weights, starts, _ in self.calls:
            self.assertEqual(weights.shape[0] % rows_per_block, 0)
            self.assertEqual(index_q.shape[0], weights.shape[0] * self.q.shape[1])
            self.assertTrue(all(t.is_contiguous() for t in (index_q, index_k, weights)))
            self.assertEqual(starts.dtype, torch.int32)
            self.assertEqual(index_k.shape[0], 11 + 128)
        last_starts, last_ends = self.calls[-1][3], self.calls[-1][4]
        self.assertTrue(bool((last_starts[1:] == self.starts[-1]).all()))
        self.assertTrue(bool((last_ends[1:] == self.ends[-1]).all()))

    def test_auto_budget_disable_explicit_and_environment(self):
        with mock.patch.dict(os.environ, {"ANGELSPEC_DSA_INDEXER_QUERY_CHUNK": "auto"}):
            self.assertEqual(_indexer_query_chunk(131072, 131072, None), 256)
            self.assertEqual(_indexer_query_chunk(4096, 100, None), 2048)
            self.assertEqual(_indexer_query_chunk(7, 11, None), 7)
        for raw, expected in (("0", 7), ("2", 2), ("100", 7)):
            with mock.patch.dict(os.environ, {"ANGELSPEC_DSA_INDEXER_QUERY_CHUNK": raw}):
                self.assertEqual(_indexer_query_chunk(7, 11, None), expected)
                self.assertEqual(_indexer_query_chunk(7, 11, 3), 3)
                self.assertEqual(
                    get_angelspec_env_vars()["ANGELSPEC_DSA_INDEXER_QUERY_CHUNK"], raw
                )
        for raw in ("bad", "-1", "1.5"):
            with mock.patch.dict(os.environ, {"ANGELSPEC_DSA_INDEXER_QUERY_CHUNK": raw}):
                with self.assertRaises(ValueError):
                    _indexer_query_chunk(7, 11, None)
        for value in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                self.call(query_chunk_size=value)

    def test_noncontiguous_inputs_and_singleton_weights(self):
        reference = self.call(query_chunk_size=0)
        self.q = self.q.transpose(1, 2).contiguous().transpose(1, 2)
        self.k = self.k.t().contiguous().t()
        self.weights = self.weights.unsqueeze(-1)
        actual = self.call(query_chunk_size=2)
        torch.testing.assert_close(actual[0], reference[0])
        torch.testing.assert_close(actual[1], reference[1])

    def test_empty_queries_skip_the_kernel(self):
        self.q, self.weights = self.q[:0], self.weights[:0]
        self.starts, self.ends = self.starts[:0], self.ends[:0]
        scores, indices = self.call(query_chunk_size=2)
        self.assertEqual(scores.shape, (0, 4))
        self.assertEqual(indices.shape, (0, 4))
        self.assertEqual(self.calls, [])

    def test_shape_errors_do_not_launch_the_kernel(self):
        self.starts = self.starts[:-1]
        with self.assertRaisesRegex(ValueError, "Key ranges"):
            self.call(query_chunk_size=2)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
