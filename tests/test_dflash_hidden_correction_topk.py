"""Hidden correction restricted to the top-k uncorrected logits (``enable_hidden_correction_topk``).

Covers:
1. Config gating: off by default with K = 64, fields carried through ``to_dict``, needs the
   correction itself, K bounded by the vocabulary, validated only when enabled; the training
   wrapper reads K from the drafter's config
2. ``_compute_draft_logits``: the K highest uncorrected logits take their corrected value and
   the rest keep the uncorrected one; K = vocabulary is the full correction; the argmax is the
   best corrected candidate unless the K+1-th uncorrected logit beats it
3. Gradients: logits outside the top-k reach the backbone but not the correction
4. End to end through ``DFlashModel``: the loss is computed on the top-k distribution
   (K = vocabulary reproduces the full correction, a small K changes it), also with the
   slot0 MTP fusion it is first trained with, and the chunked projection gives the same loss
5. ``hc_topk_recall`` and its per-position form: logged only with the restriction, 1 at
   K = vocabulary and growing with K, the per-position values average back to the overall
   one, the same on every logits path, reduced count-weighted into ``hc_topk_recall_<i>``,
   and kept by named eval sets
6. ``hc_prev_top1_rate`` / ``hc_acc_prev_top1`` / ``hc_acc_prev_other``: slot k >= 2 is
   flagged when its previous token is slot k-1's uncorrected argmax; logged only with the
   restriction, the two accuracies weighted by the rate add back to the accuracy over slots
   >= 2 and split exactly along the flags; kept by named eval sets
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel, DFlashRMSNorm

H, V = 64, 128


def _config(**kw):
    kw.setdefault("enable_hidden_correction", True)
    return DFlashConfig(
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
        **kw,
    )


def _wrapper(k=None):
    """Same weights for every ``k``; a non-zero correction so the restriction matters."""
    kw = {} if k is None else dict(enable_hidden_correction_topk=True, hidden_correction_topk=k)
    torch.manual_seed(0)
    draft = DFlashDraftModel(_config(**kw)).to(dtype=torch.float32)
    draft.freeze_embedding()
    nn.init.normal_(draft.hidden_correction.down_proj.weight, std=0.5)
    return DFlashModel(draft_model=draft, block_size=4, num_anchors=4)


def _logit_inputs(nb=3, bs=4):
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(1, nb * bs, H, generator=g),
        torch.randn(V, H, generator=g),
        torch.randint(0, V, (1, nb, bs), generator=g),
        nb,
    )


def _forward_inputs():
    g = torch.Generator().manual_seed(2)
    return dict(
        input_ids=torch.randint(0, V, (1, 32), generator=g),
        hidden_states_list=[torch.randn(1, 32, H, generator=g) for _ in range(2)],
        loss_mask=torch.ones(1, 32),
        lm_head_weight=torch.randn(V, H, generator=g),
    )


def _loss(model, inputs):
    model.eval()
    torch.manual_seed(7)
    with torch.no_grad():
        return model(**inputs)[0]


class TestConfigGating(unittest.TestCase):
    def test_off_by_default(self):
        cfg = DFlashConfig()
        self.assertFalse(cfg.enable_hidden_correction_topk)
        self.assertEqual(cfg.hidden_correction_topk, 64)
        self.assertEqual(_wrapper().hidden_correction_topk, 0)

    def test_fields_carried_and_serializable(self):
        cfg = _config(enable_hidden_correction_topk=True, hidden_correction_topk=16)
        d = cfg.to_dict()
        restored = DFlashConfig(**{k: v for k, v in d.items() if k != "transformers_version"})
        self.assertTrue(restored.enable_hidden_correction_topk)
        self.assertEqual(restored.hidden_correction_topk, 16)
        self.assertEqual(_wrapper(16).hidden_correction_topk, 16)

    def test_needs_the_correction(self):
        with self.assertRaises(ValueError):
            _config(enable_hidden_correction=False, enable_hidden_correction_topk=True)

    def test_k_bounded_by_vocab(self):
        for k in (0, V + 1):
            with self.assertRaises(ValueError):
                _config(enable_hidden_correction_topk=True, hidden_correction_topk=k)

    def test_validation_only_when_enabled(self):
        self.assertFalse(_config(hidden_correction_topk=0).enable_hidden_correction_topk)

    def test_rejected_by_a_wrapper_with_its_own_logits(self):
        # DSpark-style subclasses compute their own logits and would silently skip the
        # restriction the config asks for.
        class OwnLogits(DFlashModel):
            def _compute_draft_logits(
                self, draft_hidden, lm_head_weight, prev_token_ids, n_blocks
            ):
                return F.linear(draft_hidden, lm_head_weight)

        config = _config(enable_hidden_correction_topk=True)
        with self.assertRaises(ValueError):
            OwnLogits(draft_model=DFlashDraftModel(config), block_size=4, num_anchors=4)
        OwnLogits(draft_model=DFlashDraftModel(_config()), block_size=4, num_anchors=4)


class TestTopkLogits(unittest.TestCase):
    def _references(self, model, hidden, weight, prev):
        base = F.linear(hidden, weight)
        full = F.linear(model._apply_hidden_correction(hidden, prev), weight)
        return base, full

    def test_off_is_the_full_correction(self):
        model, (hidden, weight, prev, nb) = _wrapper(), _logit_inputs()
        _, full = self._references(model, hidden, weight, prev)
        with torch.no_grad():
            torch.testing.assert_close(model._compute_draft_logits(hidden, weight, prev, nb), full)

    def test_corrected_inside_the_topk_uncorrected_outside(self):
        model, (hidden, weight, prev, nb) = _wrapper(8), _logit_inputs()
        with torch.no_grad():
            base, full = self._references(model, hidden, weight, prev)
            out = model._compute_draft_logits(hidden, weight, prev, nb)
        inside = torch.zeros_like(base, dtype=torch.bool).scatter(-1, base.topk(8).indices, True)
        self.assertTrue((out[inside] != base[inside]).any())
        torch.testing.assert_close(out[inside], full[inside])
        torch.testing.assert_close(out[~inside], base[~inside])

    def test_k_vocab_is_the_full_correction(self):
        model, (hidden, weight, prev, nb) = _wrapper(V), _logit_inputs()
        with torch.no_grad():
            _, full = self._references(model, hidden, weight, prev)
            torch.testing.assert_close(model._compute_draft_logits(hidden, weight, prev, nb), full)

    def test_runner_up_wins_when_the_correction_sinks_every_candidate(self):
        # Corrected logits = -9 x uncorrected: every top-k candidate drops far below the
        # K+1-th uncorrected logit, which serving's argmax then picks.
        k = 8
        hidden, weight, prev, nb = _logit_inputs()
        fake = SimpleNamespace(
            draft_model=SimpleNamespace(),
            hidden_correction_topk=k,
            _apply_hidden_correction=lambda h, p, block_start=0: -9 * h,
            _prev_top1=lambda logits, prev: None,
        )
        out = DFlashModel._compute_draft_logits(fake, hidden, weight, prev, nb)
        runner_up = F.linear(hidden, weight).topk(k + 1).indices[..., k]
        torch.testing.assert_close(out.argmax(-1), runner_up)


class TestGradients(unittest.TestCase):
    def test_outside_reaches_the_backbone_only(self):
        model, (hidden, weight, prev, nb) = _wrapper(8), _logit_inputs()
        hidden.requires_grad_(True)
        out = model._compute_draft_logits(hidden, weight, prev, nb)
        inside = torch.zeros_like(out, dtype=torch.bool).scatter(
            -1, F.linear(hidden, weight).topk(8).indices, True
        )
        out[~inside].sum().backward()
        self.assertTrue(hidden.grad.abs().sum() > 0)
        for p in model.draft_model.hidden_correction.parameters():
            self.assertTrue(p.grad is None or (p.grad == 0).all())

        out = model._compute_draft_logits(hidden, weight, prev, nb)
        out[inside].sum().backward()
        self.assertTrue(model.draft_model.hidden_correction.down_proj.weight.grad.abs().sum() > 0)


class TestEndToEnd(unittest.TestCase):
    def test_loss_uses_the_topk_distribution(self):
        inputs = _forward_inputs()
        full = _loss(_wrapper(), inputs)
        self.assertTrue(torch.isfinite(full))
        torch.testing.assert_close(_loss(_wrapper(V), inputs), full)
        self.assertFalse(torch.allclose(_loss(_wrapper(4), inputs), full))

    def test_with_slot0_mtp_fusion(self):
        # The fusion feeds the block input and the restriction the logits, so with both on
        # K = vocabulary still reproduces the full correction.
        g = torch.Generator().manual_seed(3)
        inputs = dict(
            _forward_inputs(),
            last_hidden_states=torch.randn(1, 32, H, generator=g),
            target_norm=DFlashRMSNorm(H, eps=1e-6),
        )

        def loss(k):
            kw = dict(mtp_fusion="slot0", mtp_fusion_init="random")
            if k is not None:
                kw.update(enable_hidden_correction_topk=True, hidden_correction_topk=k)
            torch.manual_seed(0)
            draft = DFlashDraftModel(_config(**kw)).to(dtype=torch.float32)
            draft.freeze_embedding()
            nn.init.normal_(draft.hidden_correction.down_proj.weight, std=0.5)
            with torch.no_grad():
                draft.mtp_fusion.gate.fill_(0.5)
            return _loss(DFlashModel(draft_model=draft, block_size=4, num_anchors=4), inputs)

        full = loss(None)
        self.assertTrue(torch.isfinite(full))
        torch.testing.assert_close(loss(V), full)
        self.assertFalse(torch.allclose(loss(4), full))

    def test_chunked_projection_matches_full_path(self):
        inputs, model = _forward_inputs(), _wrapper(4)
        losses = []
        for chunk in ("0", "8"):
            with mock.patch.dict("os.environ", {"ANGELSPEC_DFLASH_LOSS_CHUNK": chunk}):
                losses.append(_loss(model, inputs))
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-4, atol=1e-4)


class TestRecallMetric(unittest.TestCase):
    def _forward(self, model, inputs, chunk="0"):
        model.eval()
        torch.manual_seed(7)
        with (
            mock.patch.dict("os.environ", {"ANGELSPEC_DFLASH_LOSS_CHUNK": chunk}),
            torch.no_grad(),
        ):
            return model(**inputs)

    def _components(self, model, inputs, chunk="0"):
        return self._forward(model, inputs, chunk)[5]

    def test_logged_with_the_restriction_and_growing_with_k(self):
        inputs = _forward_inputs()
        off = self._components(_wrapper(), inputs)
        self.assertNotIn("hc_topk_recall", off)
        self.assertNotIn("hc_topk_recall_per_position", off)
        recalls = [
            self._components(_wrapper(k), inputs)["hc_topk_recall"].item() for k in (1, 4, 16, V)
        ]
        self.assertLess(recalls[0], 1.0)
        self.assertEqual(recalls, sorted(recalls))
        self.assertEqual(recalls[-1], 1.0)

    def test_per_position_averages_back_to_the_overall_recall(self):
        inputs = _forward_inputs()
        *_, count_pp, comps = self._forward(_wrapper(4), inputs)
        per_position = comps["hc_topk_recall_per_position"]
        self.assertEqual(per_position.shape, count_pp.shape)
        torch.testing.assert_close(
            (per_position * count_pp).sum() / count_pp.sum(), comps["hc_topk_recall"]
        )
        full_vocab = self._components(_wrapper(V), inputs)["hc_topk_recall_per_position"]
        torch.testing.assert_close(full_vocab[count_pp > 0], torch.ones(int((count_pp > 0).sum())))

    def test_same_on_every_logits_path(self):
        inputs = _forward_inputs()
        full = self._components(_wrapper(4), inputs)
        decay = self._components(_wrapper(4), inputs, chunk="8")
        checkpointed_model = _wrapper(4)
        checkpointed_model.loss_checkpoint_chunks = True
        checkpointed = self._components(checkpointed_model, inputs, chunk="8")
        for key in ("hc_topk_recall", "hc_topk_recall_per_position"):
            torch.testing.assert_close(decay[key], full[key])
            torch.testing.assert_close(checkpointed[key], full[key])

    @mock.patch("angelspec.training.dflash_trainer.dist.get_rank", return_value=0)
    @mock.patch(
        "angelspec.training.dflash_trainer.dist.all_reduce",
        side_effect=lambda tensor, op=None, group=None: None,
    )
    def test_eval_aggregation_reports_each_position(self, _all_reduce, _get_rank):
        from angelspec.training.dflash_trainer import DFlashTrainer

        trainer = object.__new__(DFlashTrainer)
        trainer.grad_sync_mesh = None
        trainer.loss_decay_gamma = 1.0
        steps = [
            {
                "loss_pp": torch.tensor([0.0, 1.0, 1.0]),
                "acc_pp": torch.tensor([0.0, 0.5, 0.5]),
                "count_pp": torch.tensor([0.0, 2.0, 4.0]),
                "hc_topk_recall": torch.tensor(0.5),
                "hc_topk_recall_per_position": torch.tensor([0.0, 1.0, 0.25]),
            },
            {
                "loss_pp": torch.tensor([0.0, 1.0, 1.0]),
                "acc_pp": torch.tensor([0.0, 0.5, 0.5]),
                "count_pp": torch.tensor([0.0, 6.0, 4.0]),
                "hc_topk_recall": torch.tensor(0.7),
                "hc_topk_recall_per_position": torch.tensor([0.0, 0.5, 0.75]),
            },
        ]
        metrics = trainer._aggregate_eval_metrics(steps)
        self.assertAlmostEqual(metrics["eval/hc_topk_recall_0"], (2 * 1.0 + 6 * 0.5) / 8)
        self.assertAlmostEqual(metrics["eval/hc_topk_recall_1"], (4 * 0.25 + 4 * 0.75) / 8)
        self.assertAlmostEqual(metrics["eval/hc_topk_recall"], 0.6, places=6)

    def test_named_eval_sets_keep_it(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        trainer = object.__new__(DFlashTrainer)
        trainer._aggregate_eval_metrics = lambda metrics: metrics
        value = torch.tensor(0.5)
        step = {"hc_topk_recall": value, "lk_loss": value, "loss_pp": value}
        out = DFlashTrainer._aggregate_named_eval_metrics(trainer, [step])
        self.assertEqual(set(out[0]), {"hc_topk_recall", "loss_pp"})


class TestPrevMetrics(unittest.TestCase):
    KEYS = ("hc_prev_top1_rate", "hc_acc_prev_top1", "hc_acc_prev_other")

    def _forward(self, model, inputs):
        model.eval()
        torch.manual_seed(7)
        with mock.patch.dict("os.environ", {"ANGELSPEC_DFLASH_LOSS_CHUNK": "0"}), torch.no_grad():
            return model(**inputs)

    def test_flags_a_previous_token_that_tops_the_previous_slot(self):
        nb, bs = 2, 4
        logits = torch.randn(1, nb * bs, V, generator=torch.Generator().manual_seed(5))
        top = logits.view(1, nb, bs, V).argmax(-1)
        prev = torch.zeros(1, nb, bs, dtype=torch.long)
        prev[..., 1:] = (top[..., :-1] + 1) % V
        prev[0, 0, 2] = top[0, 0, 1]
        prev[0, 1, 3] = top[0, 1, 2]
        # Slot 1's previous token is the anchor, not drawn from the anchor slot's logits.
        prev[0, 1, 1] = top[0, 1, 0]
        expected = torch.zeros(1, nb, bs, dtype=torch.bool)
        expected[0, 0, 2] = expected[0, 1, 3] = True
        torch.testing.assert_close(_wrapper(4)._prev_top1(logits, prev), expected.view(1, -1))

    def test_logged_with_the_restriction_and_adding_back_to_the_later_accuracy(self):
        inputs = _forward_inputs()
        self.assertFalse(set(self.KEYS) & set(self._forward(_wrapper(), inputs)[5]))
        _, _, _, acc_pp, count_pp, comps = self._forward(_wrapper(4), inputs)
        rate, acc_top1, acc_other = (float(comps[key]) for key in self.KEYS)
        for value in (rate, acc_top1, acc_other):
            self.assertTrue(0.0 <= value <= 1.0)
        later = (acc_pp[2:] * count_pp[2:]).sum() / count_pp[2:].sum()
        self.assertAlmostEqual(rate * acc_top1 + (1 - rate) * acc_other, float(later), places=5)

    def test_split_along_the_flags(self):
        def slot2_only(self, logits, prev_token_ids):
            flags = torch.zeros_like(prev_token_ids, dtype=torch.bool)
            flags[:, :, 2] = True
            return flags.view(prev_token_ids.size(0), -1)

        with mock.patch.object(DFlashModel, "_prev_top1", slot2_only):
            _, _, _, acc_pp, count_pp, comps = self._forward(_wrapper(4), _forward_inputs())
        self.assertGreater(float(count_pp[2:].min()), 0.0)
        self.assertAlmostEqual(
            float(comps["hc_prev_top1_rate"]), float(count_pp[2] / count_pp[2:].sum()), places=6
        )
        self.assertAlmostEqual(float(comps["hc_acc_prev_top1"]), float(acc_pp[2]), places=6)
        self.assertAlmostEqual(float(comps["hc_acc_prev_other"]), float(acc_pp[3]), places=6)

    def test_named_eval_sets_keep_them(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        trainer = object.__new__(DFlashTrainer)
        trainer._aggregate_eval_metrics = lambda metrics: metrics
        value = torch.tensor(0.5)
        step = {**{key: value for key in self.KEYS}, "lk_loss": value, "loss_pp": value}
        out = DFlashTrainer._aggregate_named_eval_metrics(trainer, [step])
        self.assertEqual(set(out[0]), {*self.KEYS, "loss_pp"})


if __name__ == "__main__":
    unittest.main()
