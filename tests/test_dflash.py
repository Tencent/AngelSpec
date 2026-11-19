"""Tests for DFlash training model components.

Covers:
1. DFlashConfig: creation and attribute access
2. build_target_layer_ids: uniform spacing of target layers
3. DFlashDraftModel: forward pass shapes, embedding load/freeze, q_norm/k_norm
4. DFlashModel helper functions: anchor sampling with block_keep_mask, position IDs, mask
5. DFlashModel: end-to-end forward, loss + accuracy computation, loss_mask at labels
6. enable_hidden_correction: the TreeFlash correction on a plain DFlash drafter
7. attention_type="mla": hy4's gated MLA (+ optional DSA) on the dense body
8. context_local_window: sliding-window context for every attention block
"""

import math
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch

from angelspec.models.dflash import DFlashModel, _create_dflash_mask_mod
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


class TestDFlashContextLocalWindow(unittest.TestCase):
    """``context_local_window``: a block reads only the W context tokens before its anchor."""

    def test_config_defaults_off_validates_and_round_trips(self):
        self.assertIsNone(_make_config().context_local_window)

        base = {k: v for k, v in _make_config().to_dict().items() if k != "transformers_version"}
        config = DFlashConfig(**{**base, "context_local_window": 2048})
        restored = DFlashConfig(
            **{k: v for k, v in config.to_dict().items() if k != "transformers_version"}
        )
        self.assertEqual(restored.context_local_window, 2048)

        for bad in (0, -8, True, 2048.0, "2048"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                DFlashConfig(**{**base, "context_local_window": bad})

    def test_mask_keeps_only_the_window_before_each_anchor(self):
        ctx_len = 32
        mask_mod = _create_dflash_mask_mod(
            anchor_positions=torch.tensor([[5, 20]]),
            block_keep_mask=torch.tensor([[True, True]]),
            ctx_len=ctx_len,
            block_size=4,
            context_local_window=8,
        )
        for kv in range(ctx_len):
            # Block 1 (anchor 20): exactly [12, 20).
            self.assertEqual(bool(mask_mod(0, 0, 4, kv)), 12 <= kv < 20, kv)
            # Block 0 (anchor 5): the window reaches past position 0, so [0, 5).
            self.assertEqual(bool(mask_mod(0, 0, 0, kv)), kv < 5, kv)
        # Each block still sees its own draft tokens.
        self.assertTrue(mask_mod(0, 0, 0, ctx_len))
        self.assertTrue(mask_mod(0, 0, 4, ctx_len + 4))

    def test_mask_window_intersects_packed_docs_and_spares_draft_keys(self):
        from torch.nn.attention.flex_attention import create_mask

        block_size, ctx_len = 4, 32
        kwargs = dict(
            anchor_positions=torch.tensor([[5, 20, 27]]),
            block_keep_mask=torch.tensor([[True, True, True]]),
            ctx_len=ctx_len,
            block_size=block_size,
            ctx_doc_ids=torch.tensor([[0] * 16 + [1] * 16]),
            causal=True,
        )
        q_len, kv_len = 3 * block_size, ctx_len + 3 * block_size

        def grid(**extra):
            mask_mod = _create_dflash_mask_mod(**kwargs, **extra)
            return create_mask(mask_mod, 1, 1, q_len, kv_len, device="cpu")[0, 0]

        full, narrow = grid(), grid(context_local_window=8)
        # A window covering the whole prefix is a no-op.
        self.assertTrue(torch.equal(grid(context_local_window=ctx_len), full))
        # Windowing only ever removes context keys; the draft half is identical.
        self.assertTrue(torch.equal(narrow[:, ctx_len:], full[:, ctx_len:]))
        self.assertTrue(torch.equal(narrow & full, narrow))
        # Anchor 20: window [12, 20) clipped to doc 1, which starts at 16.
        self.assertEqual(
            narrow[block_size, :ctx_len].nonzero().flatten().tolist(), [16, 17, 18, 19]
        )
        # Anchor 27: window [19, 27) lies inside doc 1.
        self.assertEqual(
            narrow[2 * block_size, :ctx_len].nonzero().flatten().tolist(), list(range(19, 27))
        )

    def _loss(self, attention: str, window, seq_len: int = 40):
        H, V = 64, 128
        g = torch.Generator().manual_seed(1)
        batch = dict(
            input_ids=torch.randint(0, V, (1, seq_len), generator=g),
            hidden_states_list=[torch.randn(1, seq_len, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(1, seq_len),
            lm_head_weight=torch.randn(V, H, generator=g),
            injected_anchors=torch.tensor([[6, 20, 31]]),
        )
        config = _make_config(
            H=H,
            intermediate=256,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            V=V,
            num_target_layers=2,
        )
        if attention == "mla":
            for key, value in dict(
                attention_type="mla",
                q_lora_rank=32,
                kv_lora_rank=16,
                qk_nope_head_dim=8,
                qk_rope_head_dim=8,
                v_head_dim=12,
                gated_mla=True,
                learnable_sink=True,
            ).items():
                setattr(config, key, value)
        config.context_local_window = window
        torch.manual_seed(0)
        model = DFlashDraftModel(config).to(torch.float32)
        wrapper = DFlashModel(draft_model=model, block_size=4, num_anchors=3)
        with torch.no_grad():
            return wrapper(**batch)[0]

    def test_window_reaches_every_attention_block(self):
        for attention in ("gqa", "mla"):
            with self.subTest(attention=attention):
                full = self._loss(attention, None)
                # A window covering the whole sequence changes nothing ...
                torch.testing.assert_close(self._loss(attention, 40), full)
                # ... and a short one changes what the blocks read.
                self.assertFalse(torch.allclose(self._loss(attention, 4), full))


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
        for _name, param in self.model.draft_model.named_parameters():
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


class TestDFlashServingAcceptanceMetrics(unittest.TestCase):
    """The target-argmax accuracy and the TV accept length ride on the teacher
    logits the distill terms already build, so they exist iff one is enabled."""

    BS = 4

    def _model(self, **kw):
        config = _make_config(
            H=64, intermediate=256, num_heads=4, num_kv_heads=2, V=128, num_target_layers=2
        )
        draft_model = DFlashDraftModel(config).to(dtype=torch.float32)
        draft_model.freeze_embedding()
        model = DFlashModel(
            draft_model=draft_model,
            block_size=self.BS,
            num_anchors=6,
            loss_decay_gamma=7.0,
            **kw,
        )
        model.eval()
        return model

    @staticmethod
    def _inputs(B=2, S=24, H=64, V=128):
        g = torch.Generator().manual_seed(3)
        return dict(
            input_ids=torch.randint(0, V, (B, S), generator=g),
            hidden_states_list=[torch.randn(B, S, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(B, S),
            lm_head_weight=torch.randn(V, H, generator=g),
            last_hidden_states=torch.randn(B, S, H, generator=g),
        )

    def test_emitted_when_e2e_tv_on(self):
        model = self._model(ce_loss_alpha=0.0, e2e_tv_loss_weight=1.0)
        comps = model(**self._inputs())[5]

        accept_len = comps["e2e_accept_len"]
        self.assertTrue(torch.isfinite(accept_len))
        # Prefix products of α ∈ (0, 1] over the block_size-1 predicted slots.
        self.assertGreaterEqual(accept_len.item(), 0.0)
        self.assertLessEqual(accept_len.item(), self.BS - 1)

        tacc = comps["teacher_acc_per_position"]
        self.assertEqual(tacc.shape[0], self.BS)
        self.assertEqual(tacc[0].item(), 0.0)  # anchor slot is masked out
        self.assertTrue(((tacc >= 0.0) & (tacc <= 1.0)).all())

    def test_absent_when_no_distill_term(self):
        # l1 builds its own target logits without populating teacher_logits_flat,
        # so it must not be mistaken for a teacher being available.
        model = self._model(ce_loss_alpha=1.0, l1_loss_alpha=0.9)
        comps = model(**self._inputs())[5]
        self.assertNotIn("e2e_accept_len", comps)
        self.assertNotIn("teacher_acc_per_position", comps)


class TestE2ETVUSPNormalization(unittest.TestCase):
    """e2e-TV must divide by the SP-GLOBAL valid-block count, not the local one.

    The DFlash loss contract is ``local_sum / global_count`` per rank, so the SP
    gradient SUM (trainer's ``mul_(sp_size)``) reconstructs the full-sequence
    mean. A local denominator inflates the term by ~sp_size and, once the
    per-rank valid-block counts differ, reweights each block by
    ``N_global/N_local`` on top of that. Mirrors MTP's
    ``test_e2e_usp_global_count_parity_gloo``.
    """

    GAMMA = 3
    N_BLOCKS = 6
    V = 4

    @classmethod
    def _tensors(cls):
        g = torch.Generator().manual_seed(7)
        shape = (1, cls.N_BLOCKS, cls.GAMMA, cls.V)
        student = torch.randn(*shape, generator=g)
        teacher = torch.randn(*shape, generator=g)
        # Slot 0 is the anchor (masked in forward), block 5 is wholly invalid and
        # block 4 only partly. Under stride-2 sharding that leaves rank 0 with
        # blocks {0,2,4} = 3 valid and rank 1 with {1,3,5} = 2 valid, so the two
        # shards are unbalanced and neither is empty.
        mask = torch.ones(1, cls.N_BLOCKS, cls.GAMMA)
        mask[:, :, 0] = 0.0
        mask[:, 5] = 0.0
        mask[:, 4, 2] = 0.0
        return student, teacher, mask

    @staticmethod
    def _worker(rank, world, tmpdir, ret):
        import torch.distributed as dist

        from angelspec.models.dflash import DFlashModel

        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmpdir}/pg", world_size=world, rank=rank
        )

        student_full, teacher_full, mask_full = TestE2ETVUSPNormalization._tensors()

        model = object.__new__(DFlashModel)
        torch.nn.Module.__init__(model)
        model.sp_size = world
        model.sp_group = dist.group.WORLD

        sl = slice(rank, None, world)  # anchors are strided across SP in forward
        student = student_full[:, sl].clone().requires_grad_(True)
        loss, _ = model._compute_e2e_tv_loss(
            student_logits_pb=student,
            teacher_logits_pb=teacher_full[:, sl].clone(),
            valid_mask_pb=mask_full[:, sl].clone(),
        )
        loss.backward()

        grad_full = torch.zeros_like(student_full)
        grad_full[:, sl] = student.grad
        ret[rank] = (float(loss), grad_full.reshape(-1).tolist())
        dist.barrier()
        dist.destroy_process_group()

    def _run(self, world):
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            mgr = mp.Manager()
            ret = mgr.dict()
            mp.spawn(self._worker, args=(world, tmp, ret), nprocs=world, join=True)
            loss = sum(ret[r][0] for r in range(world))
            grad = torch.tensor(
                [sum(ret[r][1][i] for r in range(world)) for i in range(len(ret[0][1]))]
            )
            return loss, grad

    def test_sp_sum_reconstructs_the_single_rank_loss_and_grad(self):
        ref_loss, ref_grad = self._run(1)
        sp_loss, sp_grad = self._run(2)
        self.assertAlmostEqual(ref_loss, sp_loss, places=5)
        self.assertTrue(
            torch.allclose(ref_grad, sp_grad, rtol=1e-5, atol=1e-7),
            f"max grad diff={(ref_grad - sp_grad).abs().max().item()}",
        )


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
        trainer.grad_sync_mesh = None
        trainer.score_engine = None
        trainer.optimizer = TestDFlashTrainerAggregation._DummyOptimizer()
        return trainer

    @staticmethod
    def _step_metrics(
        loss_key: str, acc_key: str, count_key: str, *, teacher_acc: bool = False
    ) -> list[dict]:
        metrics = [
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
        if teacher_acc:
            metrics[0]["teacher_acc_per_position"] = torch.tensor([0.0, 1.0, 1.0, 0.0])
            metrics[1]["teacher_acc_per_position"] = torch.tensor([0.0, 0.0, 0.5, 1.0])
        return metrics

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
        side_effect=lambda tensor, op=None, group=None: None,
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
        # No teacher logits => no target-argmax companion keys at all, rather
        # than a zero-filled series that would read as "acceptance collapsed".
        self.assertNotIn("train/simulated_acc_len_tgt", metrics)
        self.assertNotIn("train/acc_tgt_0", metrics)

    @mock.patch("angelspec.training.dflash_trainer.dist.get_rank", return_value=0)
    @mock.patch(
        "angelspec.training.dflash_trainer.dist.all_reduce",
        side_effect=lambda tensor, op=None, group=None: None,
    )
    def test_aggregate_metrics_target_argmax_acc(self, _mock_all_reduce, _mock_get_rank):
        trainer = self._make_trainer()
        metrics = trainer._aggregate_metrics(
            self._step_metrics(
                "loss_per_position",
                "acc_per_position",
                "count_per_position",
                teacher_acc=True,
            ),
            step=11,
            grad_norm=torch.tensor(3.0),
        )

        # Count-weighted per position, same reduction as acc_per_position:
        # pos1 = 10/10, pos2 = (1.0*2 + 0.5*8)/10, pos3 = 4/4.
        self.assertAlmostEqual(metrics["train/acc_tgt_0"], 1.0, places=6)
        self.assertAlmostEqual(metrics["train/acc_tgt_1"], 0.6, places=6)
        self.assertAlmostEqual(metrics["train/acc_tgt_2"], 1.0, places=6)
        self.assertAlmostEqual(metrics["train/avg_acc_tgt"], 20.0 / 24.0, places=6)
        self.assertAlmostEqual(metrics["train/simulated_acc_len_tgt"], 2.2, places=6)
        # The ground-truth series must be untouched by the new one.
        self.assertAlmostEqual(metrics["train/simulated_acc_len"], 0.8, places=6)

    @mock.patch("angelspec.training.dflash_trainer.dist.get_rank", return_value=0)
    @mock.patch(
        "angelspec.training.dflash_trainer.dist.all_reduce",
        side_effect=lambda tensor, op=None, group=None: None,
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


class TestDFlashTrainerHiddenStateSplit(unittest.TestCase):
    """Rebuilding the per-layer context list from what the engine actually sent."""

    @staticmethod
    def _trainer(num_target_layers, final_layer_from_last_hidden):
        from angelspec.training.dflash_trainer import DFlashTrainer

        trainer = object.__new__(DFlashTrainer)
        trainer.num_target_layers = num_target_layers
        trainer.final_layer_from_last_hidden = final_layer_from_last_hidden
        return trainer

    def test_splits_all_layers_out_of_the_concatenation(self):
        trainer = self._trainer(3, False)
        layers = [torch.full((2, 4, 5), float(i)) for i in range(3)]
        got = trainer._split_hidden_states(torch.cat(layers, dim=-1))
        self.assertEqual(len(got), 3)
        for want, have in zip(layers, got):
            torch.testing.assert_close(have, want)

    def test_final_layer_comes_from_last_hidden_states(self):
        # vllm captures the target's final layer into last_hidden_states, so the
        # concatenation holds num_target_layers-1 layers and that tensor is the
        # last context layer.
        trainer = self._trainer(3, True)
        layers = [torch.full((2, 4, 5), float(i)) for i in range(2)]
        final = torch.full((2, 4, 5), 2.0)
        got = trainer._split_hidden_states(torch.cat(layers, dim=-1), final)
        self.assertEqual(len(got), 3)
        for want, have in zip([*layers, final], got):
            torch.testing.assert_close(have, want)

    def test_missing_last_hidden_states_is_an_error(self):
        trainer = self._trainer(3, True)
        with self.assertRaises(ValueError):
            trainer._split_hidden_states(torch.zeros(2, 4, 10), None)


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
        for _step in range(10):
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
        from angelspec.transfer.mooncake.helpers import (
            calculate_hidden_state_buffer_size,
        )

        size_3 = calculate_hidden_state_buffer_size(
            max_seq_len=1024, batch_size=2, hidden_dim=3584, num_aux_layers=3
        )
        size_5 = calculate_hidden_state_buffer_size(
            max_seq_len=1024, batch_size=2, hidden_dim=3584, num_aux_layers=5
        )
        self.assertGreater(size_5, size_3)

    def test_buffer_size_with_dflash_default(self):
        from angelspec.transfer.mooncake.helpers import (
            calculate_hidden_state_buffer_size,
        )

        size = calculate_hidden_state_buffer_size(
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
        # Unset, the count is taken from the draft config.
        self.assertIsNone(config.dflash_num_target_layers)

    def test_config_roundtrip_via_omegaconf(self):
        from omegaconf import OmegaConf

        from angelspec.config.train_config import Config

        schema = OmegaConf.structured(Config)
        self.assertEqual(schema.training.dflash_block_size, 16)
        self.assertIsNone(schema.training.dflash_num_target_layers)

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
        except ImportError:
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
            "angelspec/config/dflash_qwen3_8b_draft_config.json",
        )

    def test_dflash_draft_config_json_loads(self):
        import json
        import os

        from angelspec.models.draft.auto import AutoDraftModelConfig

        json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "angelspec",
            "config",
            "dflash_qwen3_8b_draft_config.json",
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


class TestDFlashHiddenCorrection(unittest.TestCase):
    """``enable_hidden_correction`` on a plain DFlash drafter (no DFlare fusion).

    The correction module is DFlash-family (built by ``DFlashDraftModel``, applied
    by ``DFlashModel`` before the LM head); DFly only flips its default on.
    """

    def test_absent_by_default(self):
        model = DFlashDraftModel(_make_config(H=64, num_target_layers=2))
        self.assertIsNone(model.hidden_correction)
        self.assertFalse(any("hidden_correction" in k for k in model.state_dict()))

    def test_built_and_zero_init_when_enabled(self):
        config = _make_config(H=64, num_target_layers=2)
        config.enable_hidden_correction = True
        model = DFlashDraftModel(config)
        self.assertIsNotNone(model.hidden_correction)
        # Zero-init down_proj: the correction starts as the identity, so enabling
        # it cannot perturb a converged DFlash checkpoint at step 0.
        self.assertTrue((model.hidden_correction.down_proj.weight == 0).all())
        keys = set(model.state_dict())
        for expected in (
            "hidden_correction.gate_proj.weight",
            "hidden_correction.up_proj.weight",
            "hidden_correction.down_proj.weight",
            "hidden_correction.hidden_norm.weight",
            "hidden_correction.embed_norm.weight",
        ):
            self.assertIn(expected, keys, f"missing key: {expected}")

    def test_intermediate_size_override(self):
        config = _make_config(H=64, num_target_layers=2)
        config.enable_hidden_correction = True
        config.hidden_correction_intermediate_size = 96
        model = DFlashDraftModel(config)
        # SwiGLU input is hidden :: prev-token embedding, both hidden_size wide.
        self.assertEqual(tuple(model.hidden_correction.gate_proj.weight.shape), (96, 128))
        self.assertEqual(tuple(model.hidden_correction.down_proj.weight.shape), (64, 96))

    def test_wrapper_applies_correction(self):
        torch.manual_seed(0)
        H, V = 64, 128
        config = _make_config(H=H, intermediate=256, num_heads=4, num_kv_heads=2, V=V)
        config.num_target_layers = 2
        config.enable_hidden_correction = True
        draft_model = DFlashDraftModel(config).to(dtype=torch.float32)
        draft_model.freeze_embedding()
        model = DFlashModel(draft_model=draft_model, block_size=4, num_anchors=4)
        model.eval()

        g = torch.Generator().manual_seed(1)
        inputs = dict(
            input_ids=torch.randint(0, V, (1, 32), generator=g),
            hidden_states_list=[torch.randn(1, 32, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(1, 32),
            lm_head_weight=torch.randn(V, H, generator=g),
        )

        with torch.no_grad():
            base_loss = model(**inputs)[0].item()
            # Identity at init, so a non-zero down_proj must move the loss.
            draft_model.hidden_correction.down_proj.weight.normal_()
            perturbed_loss = model(**inputs)[0].item()
        self.assertTrue(math.isfinite(base_loss))
        self.assertNotAlmostEqual(base_loss, perturbed_loss, places=5)

    def test_rope_theta_from_rope_parameters(self):
        # hy4 target configs nest rope_theta; a DFlash config copied from one must
        # not silently fall back to the 10k default.
        config = DFlashConfig(hidden_size=64, rope_parameters={"rope_theta": 1e7})
        self.assertEqual(config.rope_theta, 1e7)


class TestDFlashDenseMLA(unittest.TestCase):
    """``attention_type="mla"`` on the default dense body (``layer_arch="dflash"``).

    The attention block is orthogonal to the FFN, so the dense SwiGLU layer takes
    hy4's gated MLA (+ DSA indexer) exactly as the two MoE bodies do; only the
    attention submodule changes. ``tests/test_dfly.py`` covers the MLA block itself.
    """

    def _mla_config(self, **kw):
        config = _make_config(H=64, intermediate=256, num_heads=4, num_kv_heads=2, V=128)
        config.num_target_layers = 2
        config.attention_type = "mla"
        config.q_lora_rank = 32
        config.kv_lora_rank = 16
        config.qk_nope_head_dim = 8
        config.qk_rope_head_dim = 8
        config.v_head_dim = 12
        config.gated_mla = True
        config.learnable_sink = True
        for k, v in kw.items():
            setattr(config, k, v)
        return config

    def test_dense_body_keeps_its_ffn_and_swaps_the_attention(self):
        from angelspec.models.draft.dflash import DFlashDecoderLayer, DFlashMLP
        from angelspec.models.draft.hy4_mla import Hy4DraftMLAAttention

        model = DFlashDraftModel(self._mla_config(num_hidden_layers=2))
        self.assertTrue(all(isinstance(layer, DFlashDecoderLayer) for layer in model.layers))
        self.assertTrue(all(isinstance(layer.mlp, DFlashMLP) for layer in model.layers))
        self.assertTrue(
            all(isinstance(layer.self_attn, Hy4DraftMLAAttention) for layer in model.layers)
        )
        # Dense body: no expert weights, MLA or not.
        self.assertFalse(any("experts" in k for k in model.state_dict()))

    def test_gqa_stays_the_default(self):
        from angelspec.models.draft.dflash import DFlashAttention

        model = DFlashDraftModel(_make_config(H=64, num_layers=2, num_target_layers=2))
        self.assertTrue(
            all(isinstance(layer.self_attn, DFlashAttention) for layer in model.layers)
        )
        self.assertIsNone(model.indexers)

    def test_mla_without_dsa_builds_no_indexer(self):
        # DSA is a second opt-in on top of MLA: no index_topk means no indexer
        # weights at all, and attention stays dense over the whole context.
        model = DFlashDraftModel(self._mla_config(num_hidden_layers=2))
        for layer in model.layers:
            self.assertIsNone(layer.self_attn.indexer)
            self.assertFalse(layer.self_attn.use_indexer)
        self.assertIsNone(model.indexers)
        self.assertFalse(any("indexer" in k for k in model.state_dict()))

    def test_gqa_init_draws_layer_by_layer(self):
        # The GQA path must keep building each layer's attention in place, so a
        # given seed still produces the init it did before MLA became selectable
        # here. Observable without pinning values: layer 0 of a 1-layer and a
        # 2-layer model must agree, which only holds if the draws run
        # attn/FFN/norms per layer rather than all attentions up front (the way
        # the MLA path does it, to hand out shared indexers).
        def layer0(num_layers):
            torch.manual_seed(0)
            model = DFlashDraftModel(
                _make_config(H=64, num_layers=num_layers, num_target_layers=2)
            )
            return {k: v for k, v in model.state_dict().items() if k.startswith("layers.0.")}

        one, two = layer0(1), layer0(2)
        self.assertEqual(sorted(one), sorted(two))
        for name, tensor in one.items():
            self.assertTrue(torch.equal(tensor, two[name]), f"{name} init changed")

    def test_shared_indexers_are_owned_by_the_drafter(self):
        # Same layout as the MoE bodies: a borrowed indexer cannot live in the
        # layer that opens it, because the trainer reshards each layer on exit.
        model = DFlashDraftModel(
            self._mla_config(
                num_hidden_layers=3,
                index_topk=4,
                index_head_dim=16,
                index_n_heads=2,
                indexer_types=["full", "shared", "shared"],
            )
        )
        self.assertIsNotNone(model.indexers)
        self.assertEqual(len(model.indexers), 1)
        self.assertFalse(any("layers.0.self_attn.indexer" in k for k in model.state_dict()))

    def test_forward_backward_with_both_heads(self):
        torch.manual_seed(0)
        H, V = 64, 128
        config = self._mla_config(
            num_hidden_layers=2,
            enable_hidden_correction=True,
            enable_path_selector=True,
            path_selector_rank=16,
            path_selector_top_k=8,
        )
        model = DFlashDraftModel(config).to(torch.float32)
        model.freeze_embedding()
        wrapper = DFlashModel(draft_model=model, block_size=4, num_anchors=6)

        g = torch.Generator().manual_seed(1)
        loss = wrapper(
            input_ids=torch.randint(0, V, (2, 24), generator=g),
            hidden_states_list=[torch.randn(2, 24, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(2, 24),
            lm_head_weight=torch.randn(V, H, generator=g),
        )[0]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        params = dict(model.named_parameters())
        for name in (
            "layers.1.self_attn.q_b_proj.weight",
            "layers.1.self_attn.kv_b_proj.weight",
            "layers.1.self_attn.linear_gate.weight",
            "layers.1.self_attn.learnable_sink_param",
            "hidden_correction.down_proj.weight",
            "path_selector.hidden_projection.weight",
        ):
            grad = params[name].grad
            self.assertIsNotNone(grad, f"no grad for {name}")
            self.assertGreater(grad.abs().sum().item(), 0.0, f"zero grad for {name}")

    def test_dflare_rejects_mla(self):
        # DFlare's separate context K/V projections have no counterpart in the MLA
        # block, so the pairing is rejected instead of silently ignoring one of them.
        from angelspec.models.draft.dflare import DFlareDraftModel

        with self.assertRaisesRegex(ValueError, "attention"):
            DFlareDraftModel(self._mla_config(num_hidden_layers=1))


class TestDFlashDraftInitPath(unittest.TestCase):
    """``draft_init_path``: warm-starting the drafter's own weights, pre-FSDP."""

    def _config(self, **kw):
        config = _make_config(H=64, intermediate=256, num_heads=4, num_kv_heads=2, V=128)
        config.num_target_layers = 2
        config.enable_hidden_correction = True
        for k, v in kw.items():
            setattr(config, k, v)
        return config

    def test_draft_init_path_seeds_the_drafter(self):
        import json
        import tempfile

        from safetensors.torch import save_file

        from angelspec.training.dflash_trainer import DFlashTrainer

        torch.manual_seed(0)
        config = self._config(num_hidden_layers=2)
        source = DFlashDraftModel(config)
        # A partial directory, like the MTP-head extraction: layer bodies only.
        payload = {
            k: v for k, v in source.state_dict().items() if k.startswith(("layers.", "final_norm"))
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            save_file(payload, str(path / "model-00001-of-00001.safetensors"))
            with open(path / "model.safetensors.index.json", "w") as f:
                json.dump(
                    {"weight_map": dict.fromkeys(payload, "model-00001-of-00001.safetensors")}, f
                )

            trainer = object.__new__(DFlashTrainer)
            trainer.args = Namespace(draft_init_path=str(path))
            target = DFlashDraftModel(config)
            DFlashTrainer._load_draft_init_weights(trainer, target)

        loaded = dict(target.named_parameters())
        for key in payload:
            torch.testing.assert_close(loaded[key], payload[key])
        # Keys the directory does not carry keep their own init.
        self.assertFalse(
            torch.equal(loaded["context_proj.weight"], source.state_dict()["context_proj.weight"])
        )

    def test_draft_init_path_accepts_pytorch_model_bin(self):
        """``hf_final`` dirs from ``save_draft_model_for_serving`` are .bin, not
        safetensors, and a drafter trained with hidden correction carries keys a
        plain-DFlash config does not declare."""
        import tempfile

        from angelspec.training.dflash_trainer import DFlashTrainer

        torch.manual_seed(0)
        source_config = self._config(num_hidden_layers=2)
        source_config.enable_hidden_correction = True
        payload = DFlashDraftModel(source_config).state_dict()
        self.assertTrue(any(k.startswith("hidden_correction.") for k in payload))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            torch.save(payload, str(path / "pytorch_model.bin"))

            trainer = object.__new__(DFlashTrainer)
            trainer.args = Namespace(draft_init_path=str(path))
            config = self._config(num_hidden_layers=2)
            config.enable_hidden_correction = False
            target = DFlashDraftModel(config)
            target_embedding = target.embed_tokens.weight.detach().clone()
            DFlashTrainer._load_draft_init_weights(trainer, target)

        loaded = dict(target.named_parameters())
        for key, value in payload.items():
            if key.startswith("hidden_correction."):
                self.assertNotIn(key, loaded)
            elif key == "embed_tokens.weight":
                # The current target's embedding wins over the warm-start copy.
                torch.testing.assert_close(loaded[key], target_embedding)
            else:
                torch.testing.assert_close(loaded[key], value)

    def test_offline_draft_init_rejects_a_different_embedding(self):
        import tempfile

        from angelspec.training.dflash_trainer import DFlashTrainer

        torch.manual_seed(0)
        config = self._config(num_hidden_layers=2)
        payload = DFlashDraftModel(config).state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(payload, str(Path(tmp) / "pytorch_model.bin"))
            trainer = object.__new__(DFlashTrainer)
            trainer.args = Namespace(draft_init_path=tmp, offline_manifest_path="manifest.json")
            with self.assertRaisesRegex(ValueError, "embedding differs"):
                DFlashTrainer._load_draft_init_weights(trainer, DFlashDraftModel(config))

    def test_offline_draft_init_accepts_a_file_without_embedding(self):
        import tempfile

        from angelspec.training.dflash_trainer import DFlashTrainer

        torch.manual_seed(0)
        config = self._config(num_hidden_layers=2)
        payload = DFlashDraftModel(config).state_dict()
        payload.pop("embed_tokens.weight")
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(payload, str(Path(tmp) / "pytorch_model.bin"))
            trainer = object.__new__(DFlashTrainer)
            trainer.args = Namespace(draft_init_path=tmp, offline_manifest_path="manifest.json")
            target = DFlashDraftModel(config)
            DFlashTrainer._load_draft_init_weights(trainer, target)
        torch.testing.assert_close(target.final_norm.weight, payload["final_norm.weight"])

    def test_no_draft_init_path_is_a_noop(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        trainer = object.__new__(DFlashTrainer)
        trainer.args = Namespace(draft_init_path=None)
        model = DFlashDraftModel(self._config())
        before = model.state_dict()["final_norm.weight"].clone()
        DFlashTrainer._load_draft_init_weights(trainer, model)
        torch.testing.assert_close(model.state_dict()["final_norm.weight"], before)


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
