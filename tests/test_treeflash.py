"""Tests for the TreeFlash / DSpark structure overlay.

Covers the invariants that make the merge safe:
1. HiddenStatesCorrection is zero-initialized → identity at init (degenerates to
   DFlash), and non-trivial once the down-proj is perturbed.
2. PositionAdaptiveAlpha: alpha in (0, alpha_max), monotone ramp init, smooth_loss.
3. VanillaMarkov with pos_adaptive=False is byte-identical to the plain bigram
   bias (== the pre-merge head), and pos_adaptive=True scales the bias per slot.
4. TreeFlash dispatch + build + tiny forward through the DSparkModel wrapper.
5. Checkpoint-compat guard: the DSpark / TreeFlash state_dict carries the
   expected head/correction parameter names.
"""

import unittest

import torch

from angelspec.models.draft.auto import AutoEagle3DraftModel
from angelspec.models.draft.dspark import (
    DSparkConfig,
    DSparkDraftModel,
    HiddenStatesCorrection,
    PositionAdaptiveAlpha,
    VanillaMarkov,
)
from angelspec.models.draft.treeflash_dspark_dflare import TreeflashDSparkDFlareDraftModel
from angelspec.models.dspark import DSparkModel

H, V, BS = 64, 128, 4


def _config(model_arch="dflash", **kw):
    base = dict(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        mask_token_id=V - 1,
        markov_rank=16,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        enable_hidden_correction=True,
        block_size=BS,
        model_arch=model_arch,
    )
    base.update(kw)
    return DSparkConfig(**base)


class TestHiddenStatesCorrection(unittest.TestCase):
    def test_zero_init_is_identity(self):
        torch.manual_seed(0)
        hc = HiddenStatesCorrection(hidden_size=H, embed_size=H, intermediate_size=H)
        self.assertTrue((hc.down_proj.weight == 0).all(), "down_proj must be zero-init")
        h = torch.randn(2, 12, H)
        e = torch.randn(2, 12, H)
        self.assertTrue(torch.allclose(hc(h, e), h), "correction must be identity at init")

    def test_nonzero_after_perturb(self):
        torch.manual_seed(0)
        hc = HiddenStatesCorrection(hidden_size=H, embed_size=H, intermediate_size=H)
        with torch.no_grad():
            hc.down_proj.weight.normal_()
        h = torch.randn(2, 12, H)
        e = torch.randn(2, 12, H)
        self.assertFalse(torch.allclose(hc(h, e), h))

    def test_pos_adaptive_shape_and_identity(self):
        # With zero-init down_proj, identity holds regardless of pos-adaptive alpha.
        hc = HiddenStatesCorrection(
            hidden_size=H, embed_size=H, intermediate_size=H, pos_adaptive=True, block_size=BS
        )
        self.assertIsNotNone(hc.pos_alpha)
        h = torch.randn(2, 3 * BS, H)  # n_pos multiple of block_size
        e = torch.randn(2, 3 * BS, H)
        self.assertTrue(torch.allclose(hc(h, e), h))

    def test_pos_count_not_multiple_raises(self):
        hc = HiddenStatesCorrection(
            hidden_size=H, embed_size=H, intermediate_size=H, pos_adaptive=True, block_size=BS
        )
        with torch.no_grad():
            hc.down_proj.weight.normal_()  # make delta nonzero so the reshape runs
        with self.assertRaises(ValueError):
            hc(torch.randn(2, BS + 1, H), torch.randn(2, BS + 1, H))


class TestPositionAdaptiveAlpha(unittest.TestCase):
    def test_alpha_range_and_ramp(self):
        pa = PositionAdaptiveAlpha(block_size=BS, alpha_max=0.8, alpha_start=0.1, alpha_end=0.5)
        a = pa.alpha()
        self.assertEqual(a.shape, (BS,))
        self.assertTrue((a > 0).all() and (a <= 0.8 + 1e-6).all())
        self.assertLess(a[0].item(), a[-1].item())  # monotone ramp at init

    def test_smooth_loss_toggle(self):
        self.assertIsNone(PositionAdaptiveAlpha(block_size=BS, smooth_lambda=0.0).smooth_loss())
        reg = PositionAdaptiveAlpha(block_size=BS, smooth_lambda=0.1).smooth_loss()
        self.assertIsNotNone(reg)
        self.assertGreaterEqual(reg.item(), 0.0)

    def test_requires_positive_block_size(self):
        with self.assertRaises(ValueError):
            PositionAdaptiveAlpha(block_size=None)


class TestVanillaMarkovEquivalence(unittest.TestCase):
    def _bias_inputs(self):
        torch.manual_seed(0)
        base = torch.randn(2, 3, BS, V)
        tokens = torch.randint(0, V, (2, 3, BS))
        return base, tokens

    def test_pos_adaptive_off_is_plain_bigram_bias(self):
        # pos_adaptive=False must reproduce the plain ``base + bias`` behaviour of
        # the pre-merge VanillaMarkov (no per-slot scaling, no extra params).
        m = VanillaMarkov(vocab_size=V, markov_rank=16, pos_adaptive=False)
        self.assertIsNone(m.pos_alpha)
        # no pos_alpha parameters in the state_dict
        self.assertFalse(any("pos_alpha" in k for k in m.state_dict()))
        base, tokens = self._bias_inputs()
        expected = base + m.compute_step_bias(tokens)
        self.assertTrue(torch.allclose(m.apply_block_logits(base, token_ids=tokens), expected))

    def test_pos_adaptive_on_scales_bias(self):
        m = VanillaMarkov(vocab_size=V, markov_rank=16, pos_adaptive=True, block_size=BS)
        self.assertIsNotNone(m.pos_alpha)
        base, tokens = self._bias_inputs()
        alpha = m.pos_alpha.alpha()
        expected = base + m.compute_step_bias(tokens) * alpha.view(1, 1, -1, 1)
        self.assertTrue(torch.allclose(m.apply_block_logits(base, token_ids=tokens), expected))


class TestTreeFlashDispatchAndForward(unittest.TestCase):
    def test_auto_dispatch(self):
        tf = AutoEagle3DraftModel.from_config(_config(model_arch="dflare"))
        ds = AutoEagle3DraftModel.from_config(_config(model_arch="dflash"))
        self.assertIsInstance(tf, TreeflashDSparkDFlareDraftModel)
        self.assertIsInstance(ds, DSparkDraftModel)

    def test_treeflash_carries_heads(self):
        tf = AutoEagle3DraftModel.from_config(_config(model_arch="dflare"))
        self.assertIsNotNone(tf.markov_head)
        self.assertIsNotNone(tf.hidden_correction)
        self.assertIsNotNone(tf.confidence_head)

    def test_treeflash_forward_six_tuple(self):
        torch.manual_seed(0)
        tf = AutoEagle3DraftModel.from_config(_config(model_arch="dflare")).to(torch.float32)
        tf.freeze_embedding()
        m = DSparkModel(
            draft_model=tf,
            block_size=BS,
            num_anchors=6,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            confidence_head_alpha=1.0,
        )
        m.eval()
        B, S = 2, 24
        g = torch.Generator().manual_seed(1)
        out = m(
            input_ids=torch.randint(0, V, (B, S), generator=g),
            hidden_states_list=[torch.randn(B, S, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(B, S),
            lm_head_weight=torch.randn(V, H, generator=g),
            last_hidden_states=torch.randn(B, S, H, generator=g),
        )
        self.assertEqual(len(out), 6)
        loss, _, lpp, _, cpp, comps = out
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(lpp.shape[0], BS)
        self.assertEqual(cpp[0].item(), 0.0)  # DFlash convention: slot 0 masked
        self.assertEqual(
            set(comps), {"ce_loss", "kl_loss", "lk_loss", "l1_loss", "confidence_loss"}
        )


class TestCheckpointParamNames(unittest.TestCase):
    """Guard the head/correction parameter names so a trained DSpark /
    TreeFlash checkpoint keeps loading."""

    def test_treeflash_state_dict_keys(self):
        tf = AutoEagle3DraftModel.from_config(_config(model_arch="dflare"))
        keys = set(tf.state_dict())
        for expected in (
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
            "hidden_correction.gate_proj.weight",
            "hidden_correction.up_proj.weight",
            "hidden_correction.down_proj.weight",
            "hidden_correction.hidden_norm.weight",
            "hidden_correction.embed_norm.weight",
            "confidence_head.proj.weight",
        ):
            self.assertIn(expected, keys, f"missing checkpoint key: {expected}")

    def test_dspark_no_correction_when_disabled(self):
        ds = DSparkDraftModel(_config(model_arch="dflash", enable_hidden_correction=False))
        self.assertIsNone(ds.hidden_correction)
        self.assertFalse(any("hidden_correction" in k for k in ds.state_dict()))


if __name__ == "__main__":
    unittest.main()
