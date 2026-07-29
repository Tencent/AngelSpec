import unittest

import torch

from angelspec.controller.training_controller import (
    packed_rows_fill_ratio,
    should_wait_for_packing_fill,
    validate_packing_candidates,
)
from angelspec.data.utils import DFlashPackingCollator, MTPPackingCollator
from angelspec.utils.metrics import (
    means_from_token_totals,
    token_metric_totals,
    token_weighted_loss_scale,
)
from angelspec.utils.usp import (
    usp_chunk_size,
    usp_dp_average_factor,
    validate_dflash_usp_layout,
    validate_mtp_usp_layout,
)


class TestMTPUSPConfigSafety(unittest.TestCase):
    def test_requires_local_shard(self):
        with self.assertRaisesRegex(ValueError, "usp_local_shard=true"):
            validate_mtp_usp_layout(
                attention_backend="usp",
                usp_local_shard=False,
                sp_ring_size=1,
            )

    def test_rejects_ring_attention(self):
        with self.assertRaisesRegex(NotImplementedError, "sp_ring_size"):
            validate_mtp_usp_layout(
                attention_backend="usp",
                usp_local_shard=True,
                sp_ring_size=2,
            )

    def test_valid_pure_ulysses_layout(self):
        validate_mtp_usp_layout(
            attention_backend="usp",
            usp_local_shard=True,
            sp_ring_size=1,
        )

    def test_non_usp_layout_is_unchanged(self):
        validate_mtp_usp_layout(
            attention_backend="sdpa",
            usp_local_shard=False,
            sp_ring_size=4,
        )

    def test_chunk_size_uses_ceil_division(self):
        self.assertEqual(usp_chunk_size(9, 4), 3)
        self.assertEqual(usp_chunk_size(8, 4), 2)

    def test_flat_usp_group_uses_dp_average_factor(self):
        self.assertEqual(usp_dp_average_factor(group_world_size=16, sp_size=8), 2)
        self.assertEqual(usp_dp_average_factor(group_world_size=8, sp_size=8), 1)
        with self.assertRaises(ValueError):
            usp_dp_average_factor(group_world_size=10, sp_size=4)

    def test_dflash_family_rejects_usp(self):
        with self.assertRaisesRegex(NotImplementedError, "do not currently support"):
            validate_dflash_usp_layout(attention_backend="usp")
        validate_dflash_usp_layout(attention_backend="fa")


class TestPackingCollatorSafety(unittest.TestCase):
    @staticmethod
    def _mtp_sample(seq_len: int, hidden_len: int, loss_len: int):
        return {
            "input_ids": torch.arange(seq_len).unsqueeze(0),
            "last_hidden_states": torch.randn(1, hidden_len, 8),
            "loss_mask": torch.ones(1, loss_len),
        }

    @staticmethod
    def _dflash_sample(seq_len: int, hidden_len: int, loss_len: int):
        return {
            "input_ids": torch.arange(seq_len).unsqueeze(0),
            "hidden_states": torch.randn(1, hidden_len, 8),
            "loss_mask": torch.ones(1, loss_len),
        }

    def test_mtp_truncates_overallocated_hidden_and_loss_mask(self):
        batch = MTPPackingCollator(8)([self._mtp_sample(5, 7, 8)])
        self.assertEqual(tuple(batch["last_hidden_states"].shape), (1, 8, 8))
        self.assertEqual(batch["attention_mask"][0].tolist(), [1] * 5 + [0] * 3)
        self.assertEqual(batch["loss_mask"][0].tolist(), [1] * 5 + [0] * 3)

    def test_dflash_truncates_overallocated_hidden_and_loss_mask(self):
        batch = DFlashPackingCollator(8)([self._dflash_sample(5, 7, 8)])
        self.assertEqual(tuple(batch["hidden_states"].shape), (1, 8, 8))
        self.assertEqual(batch["attention_mask"][0].tolist(), [1] * 5 + [0] * 3)
        self.assertEqual(batch["loss_mask"][0].tolist(), [1] * 5 + [0] * 3)

    def test_mtp_rejects_short_hidden(self):
        with self.assertRaisesRegex(ValueError, "shorter than input_ids"):
            MTPPackingCollator(8)([self._mtp_sample(5, 4, 5)])

    def test_dflash_rejects_short_hidden(self):
        with self.assertRaisesRegex(ValueError, "shorter than input_ids"):
            DFlashPackingCollator(8)([self._dflash_sample(5, 4, 5)])

    def test_rejects_short_loss_mask(self):
        with self.assertRaisesRegex(ValueError, "loss_mask length"):
            MTPPackingCollator(8)([self._mtp_sample(5, 5, 4)])

    def test_rejects_mixed_hidden_presence(self):
        with_hidden = self._mtp_sample(3, 3, 3)
        without_hidden = {
            "input_ids": torch.arange(2).unsqueeze(0),
            "loss_mask": torch.ones(1, 2),
        }
        with self.assertRaisesRegex(ValueError, "either all samples or none"):
            MTPPackingCollator(8)([with_hidden, without_hidden])


class TestPackingControllerSafety(unittest.TestCase):
    def test_rejects_oversized_sample_instead_of_waiting_forever(self):
        with self.assertRaisesRegex(ValueError, "max_seq_length=8"):
            validate_packing_candidates([("ok", 5), ("too_long", 9)], max_seq=8)

    def test_accepts_packable_candidates(self):
        validate_packing_candidates([("a", 1), ("b", 8)], max_seq=8)

    def test_fill_ratio_counts_all_dispatched_rows(self):
        rank_rows = [[[0, 1]], [[2, 3]]]
        self.assertAlmostEqual(
            packed_rows_fill_ratio(rank_rows, [100, 100, 100, 50], max_seq=250),
            0.7,
        )

    def test_low_fill_waits_then_flushes_on_timeout(self):
        self.assertTrue(
            should_wait_for_packing_fill(
                fill_ratio=0.7,
                min_fill_ratio=0.8,
                waited_seconds=2.0,
                max_wait_seconds=5.0,
            )
        )
        self.assertFalse(
            should_wait_for_packing_fill(
                fill_ratio=0.7,
                min_fill_ratio=0.8,
                waited_seconds=5.0,
                max_wait_seconds=5.0,
            )
        )


class TestTokenWeightedMetrics(unittest.TestCase):
    def test_uneven_counts_are_weighted_by_tokens(self):
        metrics = [
            {
                "ce": torch.tensor([1.0]),
                "kl": torch.tensor([0.5]),
                "acces": torch.tensor([0.9]),
                "acces_gt": torch.tensor([0.8]),
                "acc_counts": torch.tensor([100.0]),
            },
            {
                "ce": torch.tensor([3.0]),
                "kl": torch.tensor([1.5]),
                "acces": torch.tensor([0.1]),
                "acces_gt": torch.tensor([0.2]),
                "acc_counts": torch.tensor([10.0]),
            },
        ]
        totals = token_metric_totals(metrics, ce_key="ce")
        ce, kl, acc, acc_gt, count = means_from_token_totals(totals)
        self.assertAlmostEqual(float(ce[0]), 130.0 / 110.0, places=6)
        self.assertAlmostEqual(float(kl[0]), 65.0 / 110.0, places=6)
        self.assertAlmostEqual(float(acc[0]), 91.0 / 110.0, places=6)
        self.assertAlmostEqual(float(acc_gt[0]), 82.0 / 110.0, places=6)
        self.assertEqual(float(count[0]), 110.0)

    def test_empty_shard_has_zero_weight(self):
        metrics = [
            {
                "ce": torch.tensor([2.0]),
                "acces": torch.tensor([0.75]),
                "acces_gt": torch.tensor([0.5]),
                "acc_counts": torch.tensor([4.0]),
            },
            {
                "ce": torch.tensor([0.0]),
                "acces": torch.tensor([0.0]),
                "acces_gt": torch.tensor([0.0]),
                "acc_counts": torch.tensor([0.0]),
            },
        ]
        totals = token_metric_totals(metrics, ce_key="ce")
        ce, _, acc, acc_gt, count = means_from_token_totals(totals)
        self.assertEqual(float(ce[0]), 2.0)
        self.assertEqual(float(acc[0]), 0.75)
        self.assertEqual(float(acc_gt[0]), 0.5)
        self.assertEqual(float(count[0]), 4.0)

    def test_dp_average_scales_reconstruct_global_token_mean(self):
        # rank0 has 100 tokens, rank1 has 10. DDP averages the two rank grads.
        global_tokens = torch.tensor(110.0)
        scale0 = token_weighted_loss_scale(torch.tensor(100.0), global_tokens, 2)
        scale1 = token_weighted_loss_scale(torch.tensor(10.0), global_tokens, 2)
        loss0 = torch.tensor(2.0)
        loss1 = torch.tensor(5.0)
        after_dp_avg = (loss0 * scale0 + loss1 * scale1) / 2
        expected = (loss0 * 100 + loss1 * 10) / 110
        self.assertAlmostEqual(float(after_dp_avg), float(expected), places=7)

    def test_token_weighting_handles_empty_rank(self):
        global_tokens = torch.tensor(110.0)
        scales = [
            token_weighted_loss_scale(torch.tensor(100.0), global_tokens, 3),
            token_weighted_loss_scale(torch.tensor(10.0), global_tokens, 3),
            token_weighted_loss_scale(torch.tensor(0.0), global_tokens, 3),
        ]
        losses = [torch.tensor(2.0), torch.tensor(5.0), torch.tensor(99.0)]
        after_dp_avg = sum(loss * scale for loss, scale in zip(losses, scales)) / 3
        expected = (losses[0] * 100 + losses[1] * 10) / 110
        self.assertAlmostEqual(float(after_dp_avg), float(expected), places=7)


if __name__ == "__main__":
    unittest.main()
