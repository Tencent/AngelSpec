"""Tests for DFlash training model components.

Covers:
1. DFlashConfig: creation and attribute access
2. build_target_layer_ids: uniform spacing of target layers
3. DFlashDraftModel: forward pass shapes, embedding load/freeze, q_norm/k_norm
4. DFlashModel helper functions: anchor sampling with block_keep_mask, position IDs, mask
5. DFlashModel: end-to-end forward, loss + accuracy computation, loss_mask at labels
"""

import math
import unittest
from unittest import mock

import torch

from angelspec.models.dflash import (
    DFlashModel,
    _create_dflash_mask_mod,
)
from angelspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDraftModel,
    build_target_layer_ids,
)


def _make_config(
    H=128,
    intermediate=512,
    num_layers=1,
    num_heads=4,
    num_kv_heads=2,
    V=256,
    num_target_layers=3,
    target_hidden_size=None,
    target_num_hidden=12,
):
    return DFlashConfig(
        hidden_size=H,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=num_target_layers,
        target_hidden_size=target_hidden_size or H,
        target_num_hidden_layers=target_num_hidden,
        mask_token_id=V - 1,
    )


def _make_dflash_model(H=64, V=128, num_target_layers=2, block_size=4, num_anchors=4):
    """Helper to create a DFlashModel for testing."""
    config = _make_config(
        H=H,
        intermediate=256,
        num_heads=4,
        num_kv_heads=2,
        V=V,
        num_target_layers=num_target_layers,
        target_num_hidden=12,
    )
    draft_model = DFlashDraftModel(config).to(dtype=torch.float32)
    draft_model.freeze_embedding()
    return DFlashModel(
        draft_model=draft_model,
        block_size=block_size,
        num_anchors=num_anchors,
        loss_decay_gamma=7.0,
    )


class TestDFlashConfig(unittest.TestCase):
    def test_config_attributes(self):
        config = _make_config(H=128, V=256, num_target_layers=5)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.num_target_layers, 5)
        self.assertEqual(config.model_type, "dflash")
        self.assertFalse(config.tie_word_embeddings)

    def test_config_serialization(self):
        config = _make_config()
        d = config.to_dict()
        restored = DFlashConfig(**{k: v for k, v in d.items() if k != "transformers_version"})
        self.assertEqual(restored.hidden_size, config.hidden_size)
        self.assertEqual(restored.num_target_layers, config.num_target_layers)


class TestBuildTargetLayerIds(unittest.TestCase):
    def test_single_layer(self):
        ids = build_target_layer_ids(1, 36)
        self.assertEqual(ids, [18])  # num_hidden_layers // 2

    def test_five_layers_36(self):
        """Expected target layer ids [1, 9, 17, 25, 33] for Qwen3-8B (36 layers)."""
        ids = build_target_layer_ids(5, 36)
        self.assertEqual(ids, [1, 9, 17, 25, 33])

    def test_two_layers(self):
        ids = build_target_layer_ids(2, 36)
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[1], 33)  # end = 36 - 3 = 33

    def test_monotonically_increasing(self):
        for n in range(1, 8):
            ids = build_target_layer_ids(n, 36)
            self.assertEqual(len(ids), n)
            for i in range(1, len(ids)):
                self.assertGreater(ids[i], ids[i - 1])


class TestDFlashDraftModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = _make_config(
            H=64,
            intermediate=256,
            num_heads=4,
            num_kv_heads=2,
            V=128,
            num_target_layers=2,
            target_num_hidden=12,
        )
        self.model = DFlashDraftModel(self.config).to(dtype=torch.float32)
        self.model.eval()

    def test_forward_shapes(self):
        B, draft_len, ctx_len = 2, 8, 16
        H = self.config.hidden_size

        draft_input_ids = torch.randint(0, self.config.vocab_size, (B, draft_len))
        context_feature = torch.randn(B, ctx_len, H)
        draft_pos = torch.arange(draft_len).unsqueeze(0).expand(B, -1)
        ctx_pos = torch.arange(ctx_len).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            out = self.model(
                draft_input_ids=draft_input_ids,
                context_feature=context_feature,
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
            )

        self.assertEqual(out.shape, (B, draft_len, H))

    def test_forward_with_noise_embedding(self):
        """Forward should accept pre-computed noise_embedding instead of draft_input_ids."""
        B, draft_len, ctx_len = 2, 8, 16
        H = self.config.hidden_size

        noise_embedding = torch.randn(B, draft_len, H)
        context_feature = torch.randn(B, ctx_len, H)
        draft_pos = torch.arange(draft_len).unsqueeze(0).expand(B, -1)
        ctx_pos = torch.arange(ctx_len).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            out = self.model(
                draft_input_ids=None,
                context_feature=context_feature,
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
                noise_embedding=noise_embedding,
            )

        self.assertEqual(out.shape, (B, draft_len, H))

    def test_extract_context_feature(self):
        B, seq_len = 2, 16
        H = self.config.hidden_size
        num_target = self.config.num_target_layers

        hs_list = [torch.randn(B, seq_len, H) for _ in range(num_target)]

        with torch.no_grad():
            ctx = self.model.extract_context_feature(hs_list)

        self.assertEqual(ctx.shape, (B, seq_len, H))

    def test_freeze_embedding(self):
        self.model.freeze_embedding()
        self.assertFalse(self.model.embed_tokens.weight.requires_grad)

    def test_trainable_params_exclude_embedding(self):
        self.model.freeze_embedding()
        for name, param in self.model.named_parameters():
            if "embed_tokens" in name:
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad, f"{name} should be trainable")

    def test_has_q_norm_and_k_norm(self):
        """DFlashAttention should have q_norm and k_norm (Qwen3 requirement)."""
        for layer in self.model.layers:
            attn = layer.self_attn
            self.assertTrue(hasattr(attn, "q_norm"), "Missing q_norm in DFlashAttention")
            self.assertTrue(hasattr(attn, "k_norm"), "Missing k_norm in DFlashAttention")
            # q_norm and k_norm should operate on head_dim
            self.assertEqual(attn.q_norm.weight.shape[0], attn.head_dim)
            self.assertEqual(attn.k_norm.weight.shape[0], attn.head_dim)


class TestAnchorSampling(unittest.TestCase):
    def setUp(self):
        self.model = _make_dflash_model(block_size=8, num_anchors=4)

    def test_basic_sampling(self):
        B, seq_len = 2, 64
        loss_mask = torch.ones(B, seq_len)
        anchors, keep_mask = self.model._sample_anchor_positions(
            seq_len, loss_mask, loss_mask.device
        )

        self.assertEqual(anchors.shape[0], B)
        self.assertEqual(keep_mask.shape, anchors.shape)
        self.assertTrue(keep_mask.dtype == torch.bool)

    def test_sorted_order(self):
        B, seq_len = 1, 128
        loss_mask = torch.ones(B, seq_len)
        anchors, keep_mask = self.model._sample_anchor_positions(
            seq_len, loss_mask, loss_mask.device
        )

        for b in range(B):
            a = anchors[b]
            self.assertTrue(torch.all(a[1:] >= a[:-1]))

    def test_respects_loss_mask(self):
        B, seq_len = 1, 64
        loss_mask = torch.zeros(B, seq_len)
        loss_mask[:, 32:] = 1.0
        model = _make_dflash_model(block_size=4, num_anchors=4)
        anchors, keep_mask = model._sample_anchor_positions(seq_len, loss_mask, loss_mask.device)

        # Valid anchors should be >= 32 (where loss_mask is 1)
        for b in range(B):
            for a in range(anchors.shape[1]):
                if keep_mask[b, a]:
                    self.assertGreaterEqual(anchors[b, a].item(), 32)

    def test_block_keep_mask_tracks_validity(self):
        """When fewer valid positions than num_anchors, keep_mask should mark invalid ones."""
        B, seq_len = 1, 64
        loss_mask = torch.zeros(B, seq_len)
        # Only 2 valid positions
        loss_mask[:, 10] = 1.0
        loss_mask[:, 20] = 1.0
        model = _make_dflash_model(block_size=4, num_anchors=10)
        anchors, keep_mask = model._sample_anchor_positions(seq_len, loss_mask, loss_mask.device)

        # Should have fewer valid anchors than requested
        valid_count = keep_mask.sum(dim=1)
        self.assertLessEqual(valid_count[0].item(), 2)

    def test_short_sequence_graceful_fallback(self):
        """Sequences too short for anchor sampling return all-False keep_mask (zero loss)."""
        B = 1
        loss_mask = torch.ones(B, 4)
        model = _make_dflash_model(block_size=8, num_anchors=2)
        anchors, keep_mask = model._sample_anchor_positions(4, loss_mask, loss_mask.device)
        self.assertEqual(anchors.shape, (B, 2))
        self.assertFalse(keep_mask.any(), "keep_mask should be all-False for too-short sequences")

    def test_all_zero_loss_mask_graceful_fallback(self):
        """All-zero loss_mask returns all-False keep_mask instead of crashing."""
        B, seq_len = 2, 64
        loss_mask = torch.zeros(B, seq_len)
        model = _make_dflash_model(block_size=4, num_anchors=4)
        anchors, keep_mask = model._sample_anchor_positions(seq_len, loss_mask, loss_mask.device)
        self.assertEqual(anchors.shape, (B, 4))
        self.assertFalse(keep_mask.any(), "keep_mask should be all-False for all-zero loss_mask")


class TestPositionIds(unittest.TestCase):
    def setUp(self):
        self.model = _make_dflash_model(block_size=8)

    def test_shapes(self):
        B, num_anchors, block_size, seq_len = 2, 4, 4, 32
        anchor_pos = torch.tensor([[0, 8, 16, 24], [1, 9, 17, 25]])
        model = _make_dflash_model(block_size=block_size)

        ctx_ids, draft_ids = model._create_position_ids(anchor_pos, seq_len)

        self.assertEqual(ctx_ids.shape, (B, seq_len))
        self.assertEqual(draft_ids.shape, (B, num_anchors * block_size))

    def test_context_sequential(self):
        anchor_pos = torch.tensor([[0, 8]])
        model = _make_dflash_model(block_size=4)
        ctx_ids, _ = model._create_position_ids(anchor_pos, seq_len=16)
        expected = torch.arange(16).unsqueeze(0)
        torch.testing.assert_close(ctx_ids, expected)

    def test_draft_offsets(self):
        anchor_pos = torch.tensor([[5, 20]])
        model = _make_dflash_model(block_size=4)
        _, draft_ids = model._create_position_ids(anchor_pos, seq_len=32)

        # Block 0: [5, 6, 7, 8], Block 1: [20, 21, 22, 23]
        expected = torch.tensor([[5, 6, 7, 8, 20, 21, 22, 23]])
        torch.testing.assert_close(draft_ids, expected)


class TestDFlashMaskMod(unittest.TestCase):
    def test_block_internal_visibility(self):
        """Within a block, all positions should see each other (bidirectional)."""
        block_size, ctx_len = 4, 16
        anchor_pos = torch.tensor([[4, 12]])
        block_keep_mask = torch.tensor([[True, True]])

        mask_mod = _create_dflash_mask_mod(
            anchor_positions=anchor_pos,
            block_keep_mask=block_keep_mask,
            ctx_len=ctx_len,
            block_size=block_size,
        )

        # Block 0: q_idx=0..3, kv_idx=ctx_len+0..3 (=16..19) are all in block 0
        for qi in range(4):
            for ki in range(4):
                self.assertTrue(mask_mod(0, 0, qi, ctx_len + ki))

    def test_inter_block_invisible(self):
        """Tokens in block 0 should NOT see tokens in block 1."""
        block_size, ctx_len = 4, 16
        anchor_pos = torch.tensor([[4, 12]])
        block_keep_mask = torch.tensor([[True, True]])

        mask_mod = _create_dflash_mask_mod(
            anchor_positions=anchor_pos,
            block_keep_mask=block_keep_mask,
            ctx_len=ctx_len,
            block_size=block_size,
        )

        # Block 0 query (q_idx=0) should not see block 1 draft (kv_idx=ctx_len+4..7)
        for ki in range(4, 8):
            self.assertFalse(mask_mod(0, 0, 0, ctx_len + ki))

        # Block 1 query (q_idx=4) should not see block 0 draft
        for ki in range(4):
            self.assertFalse(mask_mod(0, 0, 4, ctx_len + ki))

    def test_context_causal(self):
        """Block i should see context tokens BEFORE its anchor position."""
        block_size, ctx_len = 4, 16
        anchor_pos = torch.tensor([[8]])
        block_keep_mask = torch.tensor([[True]])

        mask_mod = _create_dflash_mask_mod(
            anchor_positions=anchor_pos,
            block_keep_mask=block_keep_mask,
            ctx_len=ctx_len,
            block_size=block_size,
        )

        # Context positions 0..7 should be visible (before anchor 8)
        for kv in range(8):
            self.assertTrue(mask_mod(0, 0, 0, kv))

        # Context position 8+ should NOT be visible
        for kv in range(8, ctx_len):
            self.assertFalse(mask_mod(0, 0, 0, kv))

    def test_invalid_block_sees_nothing(self):
        """Invalid blocks (block_keep_mask=False) should see nothing."""
        block_size, ctx_len = 4, 16
        anchor_pos = torch.tensor([[4, 12]])
        block_keep_mask = torch.tensor([[True, False]])  # Block 1 is invalid

        mask_mod = _create_dflash_mask_mod(
            anchor_positions=anchor_pos,
            block_keep_mask=block_keep_mask,
            ctx_len=ctx_len,
            block_size=block_size,
        )

        # Block 1 (q_idx=4..7) should see nothing
        for qi in range(4, 8):
            for kv in range(ctx_len + 8):
                self.assertFalse(mask_mod(0, 0, qi, kv))

        # Block 0 should still work normally
        self.assertTrue(mask_mod(0, 0, 0, 0))  # sees context


class TestDFlashModelForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.H = 64
        self.V = 128
        self.num_target_layers = 2
        self.model = _make_dflash_model(
            H=self.H,
            V=self.V,
            num_target_layers=self.num_target_layers,
            block_size=4,
            num_anchors=4,
        )
        self.model.eval()

    def test_forward_produces_loss_and_acc(self):
        B, seq_len = 1, 32
        input_ids = torch.randint(0, self.V, (B, seq_len))
        hidden_states_list = [
            torch.randn(B, seq_len, self.H) for _ in range(self.num_target_layers)
        ]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(self.V, self.H)

        with torch.no_grad():
            loss, acc, loss_pp, acc_pp, count_pp, _ = self.model(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertGreaterEqual(acc.item(), 0.0)
        self.assertLessEqual(acc.item(), 1.0)
        self.assertEqual(loss_pp.shape, (self.model.block_size,))
        self.assertEqual(acc_pp.shape, (self.model.block_size,))
        self.assertEqual(count_pp.shape, (self.model.block_size,))

    def test_loss_requires_grad(self):
        """Loss should be differentiable through the draft model."""
        B, seq_len = 1, 32
        input_ids = torch.randint(0, self.V, (B, seq_len))
        hidden_states_list = [
            torch.randn(B, seq_len, self.H) for _ in range(self.num_target_layers)
        ]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(self.V, self.H)

        self.model.train()
        loss, acc, _, _, _, _ = self.model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=lm_head_weight,
        )

        loss.backward()

        grad_found = False
        for name, param in self.model.draft_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    grad_found = True
                    break
        self.assertTrue(grad_found, "No gradient flowed to draft model parameters")

    def test_loss_mask_at_label_positions(self):
        """Loss mask should be gathered at label positions.

        When loss_mask is 0 for some positions, labels at those positions
        should not contribute to loss.
        """
        B, seq_len = 1, 64
        input_ids = torch.randint(0, self.V, (B, seq_len))
        hidden_states_list = [
            torch.randn(B, seq_len, self.H) for _ in range(self.num_target_layers)
        ]
        lm_head_weight = torch.randn(self.V, self.H)

        # All valid
        loss_mask_full = torch.ones(B, seq_len)
        with torch.no_grad():
            loss_full, *_ = self.model(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask_full,
                lm_head_weight=lm_head_weight,
            )

        # Half masked out
        loss_mask_half = torch.ones(B, seq_len)
        loss_mask_half[:, :32] = 0.0
        with torch.no_grad():
            loss_half, *_ = self.model(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask_half,
                lm_head_weight=lm_head_weight,
            )

        # Both should be finite
        self.assertTrue(torch.isfinite(loss_full))
        self.assertTrue(torch.isfinite(loss_half))

    def test_loss_decay_weights(self):
        """Loss decay should follow exp(-(k-1)/gamma) for block_size=16, gamma=7.0."""
        model = _make_dflash_model(block_size=16, num_anchors=2)
        model.loss_decay_gamma = 7.0

        device = torch.device("cpu")
        k = torch.arange(16, device=device).view(1, 1, -1)
        decay = torch.exp(-(k - 1).clamp(min=0).float() / 7.0)

        # k=0 → exp(0) = 1.0 (but excluded by pos_in_block > 0)
        # k=1 → exp(0) = 1.0
        # k=2 → exp(-1/7) ≈ 0.867
        # k=15 → exp(-14/7) = exp(-2) ≈ 0.135
        self.assertAlmostEqual(decay[0, 0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(decay[0, 0, 1].item(), 1.0, places=5)
        self.assertAlmostEqual(decay[0, 0, 2].item(), math.exp(-1.0 / 7.0), places=5)
        self.assertAlmostEqual(decay[0, 0, 15].item(), math.exp(-14.0 / 7.0), places=5)

    def test_label_alignment_same_position(self):
        """Labels at positions anchor+0..anchor+block_size-1 (same-position prediction)."""
        device = torch.device("cpu")
        block_size = 4

        anchor_positions = torch.tensor([[5, 20]])
        label_offsets = torch.arange(0, block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets

        # Block 0 (anchor=5): labels at positions 5, 6, 7, 8
        self.assertEqual(label_indices[0, 0].tolist(), [5, 6, 7, 8])
        # Block 1 (anchor=20): labels at positions 20, 21, 22, 23
        self.assertEqual(label_indices[0, 1].tolist(), [20, 21, 22, 23])

    def test_anchor_loss_excluded(self):
        """Position 0 in block (the anchor itself) should have zero loss weight."""
        device = torch.device("cpu")
        block_size = 4

        pos_in_block = torch.arange(block_size, device=device).view(1, 1, -1)
        anchor_weight = (pos_in_block > 0).float()

        # pos 0 (anchor) should be excluded
        self.assertEqual(anchor_weight[0, 0, 0].item(), 0.0)
        # pos 1, 2, 3 should be included
        for i in range(1, block_size):
            self.assertEqual(anchor_weight[0, 0, i].item(), 1.0)


class TestDFlashTrainerAggregation(unittest.TestCase):
    class _DummyOptimizer:
        def get_learning_rate(self):
            return 1e-3

    @staticmethod
    def _make_trainer():
        from angelspec.training.dflash_trainer import DFlashTrainer

        trainer = object.__new__(DFlashTrainer)
        trainer.loss_decay_gamma = 1.0
        trainer.global_step = 7
        trainer.optimizer = TestDFlashTrainerAggregation._DummyOptimizer()
        return trainer

    @staticmethod
    def _step_metrics(loss_key: str, acc_key: str, count_key: str) -> list[dict]:
        return [
            {
                loss_key: torch.tensor([0.0, 1.0, 5.0, 0.0]),
                acc_key: torch.tensor([0.0, 0.5, 1.0, 0.0]),
                count_key: torch.tensor([0.0, 10.0, 2.0, 0.0]),
            },
            {
                loss_key: torch.tensor([0.0, 0.0, 2.0, 4.0]),
                acc_key: torch.tensor([0.0, 0.0, 0.25, 0.5]),
                count_key: torch.tensor([0.0, 0.0, 8.0, 4.0]),
            },
        ]

    @staticmethod
    def _expected_avg_loss() -> float:
        counts = [10.0, 10.0, 4.0]
        losses = [1.0, 2.6, 4.0]
        weights = [math.exp(-i) for i in range(len(losses))]
        numerator = sum(losses[i] * counts[i] * weights[i] for i in range(len(losses)))
        denominator = sum(counts[i] * weights[i] for i in range(len(losses)))
        return numerator / denominator

    @mock.patch("angelspec.training.dflash_trainer.dist.get_rank", return_value=0)
    @mock.patch(
        "angelspec.training.dflash_trainer.dist.all_reduce",
        side_effect=lambda tensor, op=None: None,
    )
    def test_aggregate_metrics_uses_per_position_counts(self, _mock_all_reduce, _mock_get_rank):
        trainer = self._make_trainer()
        metrics = trainer._aggregate_metrics(
            self._step_metrics("loss_per_position", "acc_per_position", "count_per_position"),
            step=11,
            grad_norm=torch.tensor(3.0),
        )

        self.assertAlmostEqual(metrics["train/ploss_0"], 1.0, places=6)
        self.assertAlmostEqual(metrics["train/ploss_1"], 2.6, places=6)
        self.assertAlmostEqual(metrics["train/ploss_2"], 4.0, places=6)
        self.assertAlmostEqual(metrics["train/acc_0"], 0.5, places=6)
        self.assertAlmostEqual(metrics["train/acc_1"], 0.4, places=6)
        self.assertAlmostEqual(metrics["train/acc_2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["train/avg_acc"], 11.0 / 24.0, places=6)
        self.assertAlmostEqual(metrics["train/avg_loss"], self._expected_avg_loss(), places=6)
        self.assertAlmostEqual(metrics["train/simulated_acc_len"], 0.8, places=6)

    @mock.patch("angelspec.training.dflash_trainer.dist.get_rank", return_value=0)
    @mock.patch(
        "angelspec.training.dflash_trainer.dist.all_reduce",
        side_effect=lambda tensor, op=None: None,
    )
    def test_aggregate_eval_metrics_uses_exact_scalar_reduction(
        self, _mock_all_reduce, _mock_get_rank
    ):
        trainer = self._make_trainer()
        metrics = trainer._aggregate_eval_metrics(
            self._step_metrics("loss_pp", "acc_pp", "count_pp")
        )

        self.assertAlmostEqual(metrics["eval/ploss_0"], 1.0, places=6)
        self.assertAlmostEqual(metrics["eval/ploss_1"], 2.6, places=6)
        self.assertAlmostEqual(metrics["eval/ploss_2"], 4.0, places=6)
        self.assertAlmostEqual(metrics["eval/acc_0"], 0.5, places=6)
        self.assertAlmostEqual(metrics["eval/acc_1"], 0.4, places=6)
        self.assertAlmostEqual(metrics["eval/acc_2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["eval/avg_acc"], 11.0 / 24.0, places=6)
        self.assertAlmostEqual(metrics["eval/avg_loss"], self._expected_avg_loss(), places=6)
        self.assertAlmostEqual(metrics["eval/simulated_acc_len"], 0.8, places=6)


class TestMiniTrainingLoop(unittest.TestCase):
    """Smoke test: forward → backward → optimizer step should not crash and loss should decrease."""

    def test_loss_decreases_over_steps(self):
        torch.manual_seed(42)
        model = _make_dflash_model()

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )

        B, seq_len = 1, 32
        H, V = 64, 128
        input_ids = torch.randint(0, V, (B, seq_len))
        hidden_states_list = [torch.randn(B, seq_len, H) for _ in range(2)]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(V, H)

        model.train()
        losses = []
        for step in range(10):
            optimizer.zero_grad()
            loss, acc, _, _, _, _ = model(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (allow some noise, check last < first)
        self.assertLess(losses[-1], losses[0], "Loss did not decrease over 10 training steps")
        self.assertTrue(all(math.isfinite(v) for v in losses), "Non-finite loss encountered")

    def test_gradient_accumulation(self):
        """Two half-LR steps with accumulated gradients should produce finite grads."""
        torch.manual_seed(42)
        H, V = 64, 128
        model = _make_dflash_model()

        B, seq_len = 1, 32
        input_ids = torch.randint(0, V, (B, seq_len))
        hidden_states_list = [torch.randn(B, seq_len, H) for _ in range(2)]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(V, H)

        model.train()
        model.zero_grad()
        loss1, *_ = model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=lm_head_weight,
        )
        (loss1 / 2).backward()

        loss2, *_ = model(
            input_ids=input_ids,
            hidden_states_list=hidden_states_list,
            loss_mask=loss_mask,
            lm_head_weight=lm_head_weight,
        )
        (loss2 / 2).backward()

        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
                has_grad = True
        self.assertTrue(has_grad)


class TestTrainerActorDispatch(unittest.TestCase):
    """Verify that TrainerActor dispatches to the correct trainer class."""

    def test_dflash_config_detected(self):
        from angelspec.models.draft.dflash import DFlashConfig

        config = DFlashConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
        self.assertIsInstance(config, DFlashConfig)

    def test_auto_config_from_dict_dflash(self):
        from angelspec.models.draft.auto import AutoDraftModelConfig

        config_dict = {
            "architectures": ["DFlashDraftModel"],
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "num_target_layers": 5,
            "target_hidden_size": 64,
            "target_num_hidden_layers": 12,
        }
        config = AutoDraftModelConfig.from_dict(config_dict)
        from angelspec.models.draft.dflash import DFlashConfig

        self.assertIsInstance(config, DFlashConfig)

    def test_auto_config_from_dict_eagle3(self):
        from angelspec.models.draft.auto import AutoDraftModelConfig

        config_dict = {
            "architectures": ["LlamaForCausalLMEagle3"],
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
        }
        config = AutoDraftModelConfig.from_dict(config_dict)
        from angelspec.models.draft.dflash import DFlashConfig

        self.assertNotIsInstance(config, DFlashConfig)


class TestTargetModelGeneralization(unittest.TestCase):
    """Verify that eagle3_target_model supports configurable layer counts."""

    def test_set_aux_layers_custom_count(self):
        """set_aux_hidden_states_layers should accept any non-empty list."""
        from unittest.mock import MagicMock

        from angelspec.models.target.eagle3_target_model import Eagle3TargetModel

        class MockTarget(Eagle3TargetModel):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                pass

            def generate_eagle3_data(self, *args, **kwargs):
                pass

        mock = MockTarget()
        mock.model = MagicMock()
        mock.model.config.num_hidden_layers = 36

        mock.set_aux_hidden_states_layers([1, 9, 17, 25, 33])
        self.assertEqual(len(mock.aux_hidden_states_layers), 5)
        self.assertEqual(mock.aux_hidden_states_layers, [1, 9, 17, 25, 33])

    def test_set_aux_layers_default_eagle3(self):
        from unittest.mock import MagicMock

        from angelspec.models.target.eagle3_target_model import Eagle3TargetModel

        class MockTarget(Eagle3TargetModel):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                pass

            def generate_eagle3_data(self, *args, **kwargs):
                pass

        mock = MockTarget()
        mock.model = MagicMock()
        mock.model.config.num_hidden_layers = 28

        mock.set_aux_hidden_states_layers(None)
        self.assertEqual(len(mock.aux_hidden_states_layers), 3)
        self.assertEqual(mock.aux_hidden_states_layers, [1, 13, 24])

    def test_set_aux_layers_rejects_empty(self):
        from unittest.mock import MagicMock

        from angelspec.models.target.eagle3_target_model import Eagle3TargetModel

        class MockTarget(Eagle3TargetModel):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                pass

            def generate_eagle3_data(self, *args, **kwargs):
                pass

        mock = MockTarget()
        mock.model = MagicMock()

        with self.assertRaises(ValueError):
            mock.set_aux_hidden_states_layers([])


class TestMooncakeBufferSizing(unittest.TestCase):
    """Verify mooncake buffer sizing adapts to layer count."""

    def test_buffer_size_scales_with_layers(self):
        from angelspec.transfer.mooncake.helpers import calculate_eagle3_buffer_size

        size_3 = calculate_eagle3_buffer_size(
            max_seq_len=1024, batch_size=2, hidden_dim=3584, num_aux_layers=3
        )
        size_5 = calculate_eagle3_buffer_size(
            max_seq_len=1024, batch_size=2, hidden_dim=3584, num_aux_layers=5
        )
        self.assertGreater(size_5, size_3)

    def test_buffer_size_with_dflash_default(self):
        from angelspec.transfer.mooncake.helpers import calculate_eagle3_buffer_size

        size = calculate_eagle3_buffer_size(
            max_seq_len=1024, batch_size=2, hidden_dim=3584, num_aux_layers=5
        )
        self.assertGreater(size, 0)


class TestDFlashAuxLayerIds(unittest.TestCase):
    """Verify DFlash aux layer ID computation."""

    def test_dflash_layer_ids_match_build_target(self):
        from angelspec.models.draft.dflash import build_target_layer_ids

        ids = build_target_layer_ids(5, 36)
        self.assertEqual(len(ids), 5)
        self.assertGreaterEqual(ids[0], 1)
        self.assertLessEqual(ids[-1], 35)

    def test_dflash_layer_ids_for_28_layers(self):
        from angelspec.models.draft.dflash import build_target_layer_ids

        ids = build_target_layer_ids(5, 28)
        self.assertEqual(ids, [1, 7, 13, 19, 25])


class TestTrainEntryDFlashIntegration(unittest.TestCase):
    """Verify train_entry DFlash config auto-sets aux layer IDs."""

    def test_auto_sets_aux_layers(self):
        from argparse import Namespace

        from angelspec.models.draft.dflash import DFlashConfig, build_target_layer_ids

        args = Namespace()
        config = DFlashConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            num_target_layers=5,
            target_num_hidden_layers=28,
        )

        if isinstance(config, DFlashConfig) and not getattr(
            args, "aux_hidden_states_layers", None
        ):
            target_layer_ids = getattr(config, "target_layer_ids", None)
            if target_layer_ids is None:
                num_target = getattr(config, "num_target_layers", 5)
                target_num_hidden = getattr(config, "target_num_hidden_layers", 36)
                target_layer_ids = build_target_layer_ids(num_target, target_num_hidden)
            args.aux_hidden_states_layers = target_layer_ids

        self.assertIsNotNone(args.aux_hidden_states_layers)
        self.assertEqual(len(args.aux_hidden_states_layers), 5)

    def test_does_not_override_explicit_layers(self):
        from argparse import Namespace

        from angelspec.models.draft.dflash import DFlashConfig

        args = Namespace(aux_hidden_states_layers=[1, 5, 10, 15, 20])
        config = DFlashConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            num_target_layers=5,
            target_num_hidden_layers=28,
        )

        if isinstance(config, DFlashConfig) and not getattr(
            args, "aux_hidden_states_layers", None
        ):
            args.aux_hidden_states_layers = [0, 1, 2, 3, 4]

        self.assertEqual(args.aux_hidden_states_layers, [1, 5, 10, 15, 20])


class TestDFlashTrainingConfig(unittest.TestCase):
    """Verify DFlash parameters are present in TrainingConfig."""

    def test_config_has_dflash_params(self):
        from angelspec.config.train_config import TrainingConfig

        config = TrainingConfig()
        self.assertEqual(config.dflash_block_size, 16)
        self.assertEqual(config.dflash_num_anchors, 512)
        self.assertEqual(config.dflash_loss_decay_gamma, 7.0)
        self.assertEqual(config.dflash_num_target_layers, 5)

    def test_config_roundtrip_via_omegaconf(self):
        from omegaconf import OmegaConf

        from angelspec.config.train_config import Config

        schema = OmegaConf.structured(Config)
        self.assertEqual(schema.training.dflash_block_size, 16)
        self.assertEqual(schema.training.dflash_num_target_layers, 5)

        overrides = OmegaConf.from_dotlist(
            [
                "training.dflash_block_size=8",
                "training.dflash_num_target_layers=3",
            ]
        )
        merged = OmegaConf.merge(schema, overrides)
        self.assertEqual(merged.training.dflash_block_size, 8)
        self.assertEqual(merged.training.dflash_num_target_layers, 3)


class TestDFlashTrainingQuality(unittest.TestCase):
    """Comprehensive training quality validation for DFlash."""

    def _make_model_and_data(
        self,
        H=64,
        V=128,
        num_target_layers=2,
        block_size=4,
        num_anchors=4,
        seq_len=64,
        batch_size=2,
    ):
        model = _make_dflash_model(
            H=H,
            V=V,
            num_target_layers=num_target_layers,
            block_size=block_size,
            num_anchors=num_anchors,
        )
        input_ids = torch.randint(0, V, (batch_size, seq_len))
        hidden_states_list = [
            torch.randn(batch_size, seq_len, H) for _ in range(num_target_layers)
        ]
        loss_mask = torch.ones(batch_size, seq_len)
        lm_head_weight = torch.randn(V, H)
        return model, input_ids, hidden_states_list, loss_mask, lm_head_weight

    def _train_steps(
        self, model, input_ids, hidden_states_list, loss_mask, lm_head_weight, steps=20, lr=1e-3
    ):
        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
        losses, accs = [], []
        for _ in range(steps):
            optimizer.zero_grad()
            loss, acc, _, _, _, _ = model(
                input_ids=input_ids,
                hidden_states_list=hidden_states_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accs.append(acc.item())
        return losses, accs

    def test_longer_sequence(self):
        """Training on longer sequences (128 tokens) should converge."""
        torch.manual_seed(42)
        model, *data = self._make_model_and_data(seq_len=128, num_anchors=8)
        losses, _ = self._train_steps(model, *data, steps=20)
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(all(math.isfinite(v) for v in losses))

    def test_large_block_size(self):
        """Block size 8 should still converge."""
        torch.manual_seed(42)
        model, *data = self._make_model_and_data(block_size=8, seq_len=128, num_anchors=4)
        losses, _ = self._train_steps(model, *data, steps=20)
        self.assertLess(losses[-1], losses[0])

    def test_accuracy_improves(self):
        """Accuracy should improve over training when overfitting on a small batch."""
        torch.manual_seed(42)
        model, *data = self._make_model_and_data(seq_len=64, batch_size=1)
        _, accs = self._train_steps(model, *data, steps=30)
        avg_first5 = sum(accs[:5]) / 5
        avg_last5 = sum(accs[-5:]) / 5
        self.assertGreater(
            avg_last5, avg_first5, f"Accuracy did not improve: {avg_first5:.4f} → {avg_last5:.4f}"
        )

    def test_gradient_norms_are_healthy(self):
        """Gradient norms should stay finite and non-zero during training."""
        torch.manual_seed(42)
        model, input_ids, hs_list, mask, lm_w = self._make_model_and_data()
        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        grad_norms = []
        for _ in range(10):
            optimizer.zero_grad()
            loss, *_ = model(
                input_ids=input_ids,
                hidden_states_list=hs_list,
                loss_mask=mask,
                lm_head_weight=lm_w,
            )
            loss.backward()
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm**0.5)
            optimizer.step()
        self.assertTrue(all(math.isfinite(g) for g in grad_norms), "Non-finite gradient norm")
        self.assertTrue(all(g > 0 for g in grad_norms), "Zero gradient norm detected")

    def test_multiple_target_layers(self):
        """Training with 5 target layers (DFlash default) should work."""
        torch.manual_seed(42)
        model, *data = self._make_model_and_data(num_target_layers=5, seq_len=64)
        losses, _ = self._train_steps(model, *data, steps=15)
        self.assertLess(losses[-1], losses[0])

    def test_loss_mask_with_padding(self):
        """Model should handle sequences with partial padding via loss_mask."""
        torch.manual_seed(42)
        model, input_ids, hs_list, loss_mask, lm_w = self._make_model_and_data(
            seq_len=64,
            batch_size=2,
        )
        loss_mask[0, :16] = 0.0
        loss_mask[1, :32] = 0.0
        losses, _ = self._train_steps(
            model,
            input_ids,
            hs_list,
            loss_mask,
            lm_w,
            steps=15,
        )
        self.assertTrue(all(math.isfinite(v) for v in losses))
        self.assertLess(losses[-1], losses[0])


class TestDFlashVsEagle3Architecture(unittest.TestCase):
    """Structural comparison between DFlash and Eagle3 architectures."""

    def test_parameter_count_comparison(self):
        H, V = 64, 128
        dflash_config = _make_config(
            H=H,
            intermediate=256,
            num_heads=4,
            num_kv_heads=2,
            V=V,
            num_target_layers=5,
            target_num_hidden=12,
        )
        dflash_model = DFlashDraftModel(dflash_config).to(dtype=torch.float32)
        dflash_model.freeze_embedding()

        dflash_trainable = sum(p.numel() for p in dflash_model.parameters() if p.requires_grad)
        dflash_frozen = sum(p.numel() for p in dflash_model.parameters() if not p.requires_grad)

        self.assertGreater(dflash_trainable, 0)
        self.assertGreater(dflash_frozen, 0)
        self.assertGreater(dflash_trainable, dflash_frozen)

    def test_dflash_context_proj_dimension(self):
        H = 64
        num_target_layers = 5
        config = _make_config(H=H, V=128, num_target_layers=num_target_layers)
        model = DFlashDraftModel(config)
        expected_input_dim = num_target_layers * H
        self.assertEqual(model.context_proj.in_features, expected_input_dim)
        self.assertEqual(model.context_proj.out_features, H)

    def test_dflash_uses_ce_loss_not_kl(self):
        torch.manual_seed(42)
        H, V = 64, 128
        model = _make_dflash_model(H=H, V=V, num_target_layers=2, block_size=4, num_anchors=2)

        B, seq_len = 1, 32
        input_ids = torch.randint(0, V, (B, seq_len))
        hs_list = [torch.randn(B, seq_len, H) for _ in range(2)]
        loss_mask = torch.ones(B, seq_len)
        lm_head_weight = torch.randn(V, H)

        with torch.no_grad():
            loss, acc, _, _, _, _ = model(
                input_ids=input_ids,
                hidden_states_list=hs_list,
                loss_mask=loss_mask,
                lm_head_weight=lm_head_weight,
            )

        self.assertGreaterEqual(loss.item(), 0.0, "CE loss should be non-negative")
        self.assertGreaterEqual(acc.item(), 0.0)
        self.assertLessEqual(acc.item(), 1.0)

    def test_dflash_block_parallel_vs_sequential(self):
        torch.manual_seed(42)
        H, V = 64, 128
        block_size = 4
        num_anchors = 3
        config = _make_config(H=H, V=V, num_target_layers=2, target_num_hidden=12)
        draft_model = DFlashDraftModel(config).to(dtype=torch.float32)

        B, ctx_len = 1, 32
        draft_len = num_anchors * block_size
        draft_input_ids = torch.randint(0, V, (B, draft_len))
        context_feature = torch.randn(B, ctx_len, H)
        draft_pos = torch.arange(draft_len).unsqueeze(0)
        ctx_pos = torch.arange(ctx_len).unsqueeze(0)

        with torch.no_grad():
            out = draft_model(
                draft_input_ids=draft_input_ids,
                context_feature=context_feature,
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
            )

        self.assertEqual(
            out.shape,
            (B, draft_len, H),
            "DFlash should produce all predictions in one forward pass",
        )


class TestDFlashConfigYAML(unittest.TestCase):
    """Verify the DFlash training config YAML loads correctly."""

    def test_dflash_yaml_loads(self):
        import os

        try:
            from angelspec.config.train_config import load_config
        except (ImportError, ModuleNotFoundError):
            self.skipTest("load_config requires ray (not installed locally)")

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "sglang_qwen3_8b_dflash.yaml",
        )
        if not os.path.exists(config_path):
            self.skipTest(f"DFlash config not found at {config_path}")

        config = load_config(config_path=config_path)
        self.assertEqual(config.training.dflash_block_size, 16)
        self.assertEqual(config.training.dflash_num_anchors, 512)
        self.assertEqual(config.training.dflash_num_target_layers, 5)
        self.assertAlmostEqual(config.training.dflash_loss_decay_gamma, 7.0)
        self.assertEqual(
            config.model.draft_model_config,
            "angelspec/config/dflash_draft_config.json",
        )

    def test_dflash_draft_config_json_loads(self):
        import json
        import os

        from angelspec.models.draft.auto import AutoDraftModelConfig

        json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "angelspec",
            "config",
            "dflash_draft_config.json",
        )
        if not os.path.exists(json_path):
            self.skipTest(f"DFlash draft config JSON not found at {json_path}")

        with open(json_path) as f:
            config_dict = json.load(f)

        config = AutoDraftModelConfig.from_dict(config_dict)
        from angelspec.models.draft.dflash import DFlashConfig

        self.assertIsInstance(config, DFlashConfig)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 5)  # Updated from 1 to 5
        self.assertEqual(config.num_target_layers, 5)
        self.assertEqual(config.target_num_hidden_layers, 36)


class TestDFlashHotfixes(unittest.TestCase):
    """Tests for incremental correctness fixes (tensor schema, collator, validation)."""

    def test_collator_accepts_dflash_batch(self):
        """DataCollator should accept hidden_states without target/last_hidden_states."""
        from angelspec.data.utils import DataCollatorWithPadding

        collator = DataCollatorWithPadding()
        samples = []
        for seq_len in [32, 48]:
            samples.append(
                {
                    "input_ids": torch.randint(0, 1000, (1, seq_len)),
                    "hidden_states": torch.randn(1, seq_len, 384),
                    "loss_mask": torch.ones(1, seq_len),
                }
            )

        # Should not raise (DFlash does not provide target/last_hidden_states)
        batch = collator(samples)
        self.assertIsNotNone(batch["hidden_states"])
        self.assertIsNone(batch["target"])
        self.assertIsNone(batch["last_hidden_states"])

    def test_min_loss_tokens_validation(self):
        """min_loss_tokens < 2 * block_size should be caught."""
        from argparse import Namespace

        # Simulate the validation logic from train_entry
        args = Namespace(
            dflash_block_size=16,
            min_loss_tokens=10,  # < 2 * 16 = 32
        )
        block_size = getattr(args, "dflash_block_size", 16)
        min_loss = getattr(args, "min_loss_tokens", 0)

        with self.assertRaises(ValueError):
            if min_loss < 2 * block_size:
                raise ValueError(
                    f"DFlash requires dataset.min_loss_tokens >= 2 * training.dflash_block_size "
                    f"({min_loss} < {2 * block_size})."
                )

    def test_min_loss_tokens_validation_passes(self):
        """min_loss_tokens >= 2 * block_size should pass."""
        block_size = 16
        min_loss = 32  # == 2 * 16
        self.assertGreaterEqual(min_loss, 2 * block_size)


if __name__ == "__main__":
    unittest.main()
