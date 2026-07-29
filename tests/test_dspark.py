"""Tests for DSpark (DFlash backbone + Markov / confidence heads + L1 distillation).

Pins the DSpark wiring so future refactors can't silently break the objective:

1. DSparkConfig / DSparkDraftModel: head construction, subclass relationship.
2. forward returns the 6-tuple with detached per-component losses.
3. Loss-wiring invariants:
   - internal identity: combined loss == ce_a*ce + l1_a*l1 + cf_a*conf  (so the
     logged loss_components are trustworthy)
   - all-masked batch -> loss 0
   - gradients reach markov + confidence + backbone; embedding stays frozen
   - next-token convention: every within-block slot is supervised (B predictions)
4. Markov / confidence head unit math.
5. Algorithm dispatch (DSparkConfig resolves from the JSON and is checked before
   DFlashConfig since it subclasses it).
"""

import unittest

import torch

from angelspec.models.draft.auto import AutoDraftModelConfig
from angelspec.models.draft.dflash import DFlashConfig
from angelspec.models.draft.dspark import (
    AcceptRatePredictor,
    DSparkConfig,
    DSparkDraftModel,
    VanillaMarkov,
)
from angelspec.models.dspark import DSparkModel

CE_A, L1_A, CF_A = 0.1, 0.9, 1.0


def _make_dspark_config(
    H=64,
    V=128,
    num_target_layers=2,
    markov_rank=16,
    enable_confidence_head=True,
    confidence_head_with_markov=True,
):
    return DSparkConfig(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=num_target_layers,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        mask_token_id=V - 1,
        markov_rank=markov_rank,
        markov_head_type="vanilla",
        enable_confidence_head=enable_confidence_head,
        confidence_head_with_markov=confidence_head_with_markov,
    )


def _make_dspark_model(block_size=4, num_anchors=6, **cfg_kw):
    config = _make_dspark_config(**cfg_kw)
    draft = DSparkDraftModel(config).to(dtype=torch.float32)
    draft.freeze_embedding()
    return DSparkModel(
        draft_model=draft,
        block_size=block_size,
        num_anchors=num_anchors,
        loss_decay_gamma=4.0,
        ce_loss_alpha=CE_A,
        l1_loss_alpha=L1_A,
        confidence_head_alpha=CF_A,
    )


def _batch(B=2, S=24, H=64, V=128, num_target_layers=2, all_masked=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, V, (B, S), generator=g)
    hidden_states_list = [torch.randn(B, S, H, generator=g) for _ in range(num_target_layers)]
    loss_mask = torch.zeros(B, S) if all_masked else torch.ones(B, S)
    if not all_masked:
        loss_mask[:, :2] = 0  # prompt
    lm_head_weight = torch.randn(V, H, generator=g)
    last_hidden_states = torch.randn(B, S, H, generator=g)
    return dict(
        input_ids=input_ids,
        hidden_states_list=hidden_states_list,
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        last_hidden_states=last_hidden_states,
    )


class TestDSparkConfig(unittest.TestCase):
    def test_subclasses_dflash_and_attrs(self):
        cfg = _make_dspark_config(markov_rank=32)
        self.assertIsInstance(cfg, DFlashConfig)  # ordering hazard: check DSpark first
        self.assertEqual(cfg.model_type, "dspark")
        self.assertEqual(cfg.markov_rank, 32)
        self.assertTrue(cfg.enable_confidence_head)

    def test_draft_model_heads(self):
        cfg = _make_dspark_config(H=64, markov_rank=16)
        m = DSparkDraftModel(cfg)
        self.assertIsInstance(m.markov_head, VanillaMarkov)
        self.assertIsInstance(m.confidence_head, AcceptRatePredictor)
        # confidence input = hidden + markov_rank when fused
        self.assertEqual(m.confidence_head.proj.in_features, 64 + 16)

    def test_no_heads(self):
        cfg = _make_dspark_config(
            markov_rank=0, enable_confidence_head=False, confidence_head_with_markov=False
        )
        m = DSparkDraftModel(cfg)
        self.assertIsNone(m.markov_head)
        self.assertIsNone(m.confidence_head)


class TestDSparkForward(unittest.TestCase):
    def test_returns_six_tuple_with_detached_components(self):
        m = _make_dspark_model()
        out = m(**_batch())
        self.assertEqual(len(out), 6)
        loss, acc, lpp, app, cpp, comps = out
        self.assertEqual(
            set(comps), {"ce_loss", "kl_loss", "lk_loss", "l1_loss", "confidence_loss"}
        )
        for v in comps.values():
            self.assertTrue(torch.isfinite(v).all())
            self.assertFalse(v.requires_grad)  # detached for logging
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(lpp.shape[0], m.block_size)

    def test_internal_loss_identity(self):
        # At world_size==1 (no process group), the combined loss must equal the
        # alpha-weighted sum of the logged components — so the components are a
        # faithful decomposition of what's actually optimized.
        m = _make_dspark_model()
        loss, _, _, _, _, comps = m(**_batch(seed=1))
        recomputed = (
            CE_A * comps["ce_loss"] + L1_A * comps["l1_loss"] + CF_A * comps["confidence_loss"]
        )
        self.assertTrue(
            torch.allclose(loss, recomputed, atol=1e-4), f"{loss.item()} vs {recomputed.item()}"
        )

    def test_all_masked_is_zero(self):
        m = _make_dspark_model()
        loss, _, _, _, _, comps = m(**_batch(all_masked=True))
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        for v in comps.values():
            self.assertAlmostEqual(v.item(), 0.0, places=5)

    def test_next_token_convention_slot0_masked(self):
        # DSpark now reuses the unified DFlashModel.forward: slot 0 is the masked
        # anchor (count 0) and slots 1..B-1 are the supervised predictions. With a
        # long fully supervised sequence every predicted slot accumulates tokens.
        m = _make_dspark_model(block_size=4, num_anchors=8)
        b = _batch(B=2, S=40)
        b["loss_mask"] = torch.ones(2, 40)
        _, _, _, _, count_per_position, _ = m(**b)
        self.assertEqual(count_per_position.shape[0], 4)
        self.assertEqual(count_per_position[0].item(), 0.0)  # anchor slot masked
        self.assertTrue(
            (count_per_position[1:] > 0).all(),
            f"some predicted slot unsupervised: {count_per_position.tolist()}",
        )

    def test_grad_flow_and_frozen_embedding(self):
        m = _make_dspark_model()
        loss, *_ = m(**_batch(seed=2))
        loss.backward()
        draft = m.draft_model
        self.assertIsNotNone(draft.markov_head.markov_w2.weight.grad)
        self.assertGreater(draft.markov_head.markov_w2.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(draft.confidence_head.proj.weight.grad)
        self.assertGreater(draft.confidence_head.proj.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(draft.context_proj.weight.grad)
        self.assertIsNone(draft.embed_tokens.weight.grad)  # frozen

    def test_ce_only_without_target(self):
        # ce-only (l1=0, no confidence) must run without last_hidden_states.
        m = _make_dspark_model(
            markov_rank=16, enable_confidence_head=False, confidence_head_with_markov=False
        )
        m.l1_loss_alpha = 0.0
        m.ce_loss_alpha = 1.0
        m.confidence_head_alpha = 0.0
        b = _batch()
        b["last_hidden_states"] = None
        loss, *_ = m(**b)
        self.assertTrue(torch.isfinite(loss))


class TestHeadMath(unittest.TestCase):
    def test_vanilla_markov_is_bigram_bias(self):
        torch.manual_seed(0)
        mk = VanillaMarkov(vocab_size=50, markov_rank=8)
        base = torch.randn(2, 3, 4, 50)
        prev = torch.randint(0, 50, (2, 3, 4))
        out = mk.apply_block_logits(base, token_ids=prev)
        expected = base + mk.markov_w2(mk.markov_w1(prev))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_confidence_head_is_linear(self):
        torch.manual_seed(0)
        head = AcceptRatePredictor(20)
        feats = torch.randn(2, 3, 4, 20)
        out = head(feats)
        expected = head.proj(feats).squeeze(-1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))
        self.assertEqual(out.shape, (2, 3, 4))


class TestDispatch(unittest.TestCase):
    def test_json_resolves_to_dspark_config(self):
        cfg = AutoDraftModelConfig.from_dict(
            {
                "architectures": ["DSparkDraftModel"],
                "model_type": "dspark",
                "hidden_size": 64,
                "vocab_size": 128,
                "num_hidden_layers": 1,
                "num_target_layers": 2,
                "markov_rank": 16,
                "enable_confidence_head": True,
            }
        )
        self.assertIsInstance(cfg, DSparkConfig)
        # Subclass of DFlashConfig -> any isinstance(DFlashConfig) dispatch must
        # test DSparkConfig first (trainer_actor / train_entry rely on this).
        self.assertIsInstance(cfg, DFlashConfig)


if __name__ == "__main__":
    unittest.main()
