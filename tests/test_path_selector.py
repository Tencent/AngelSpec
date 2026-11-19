"""Tests for the path selector (predecessor-conditioned rerank of the unary top-k).

The selector is a DFlash-family drafter head gated by ``enable_path_selector``,
so it rides the shared DFlash config / drafter / training-wrapper path rather
than owning an arch of its own. Covers:

1. Config gating: off by default, fields carried through, validation.
2. Drafter wiring: head built only when enabled, codebooks sized by vocab.
3. Selector math: score == unary + bilinear transition; ``select`` takes the
   unary top-k; the candidate label set injects a missing target at rank K.
4. Loss wiring through ``DFlashModel``: selector keys appear only when the head
   is on, the logged components are a faithful decomposition of the total,
   masked batches stay at 0, and gradients reach every selector parameter.
   ``acc_per_position`` / ``teacher_acc_per_position`` use the selector pick,
   so progress-bar ``acc_len`` / ``acc_len_tgt`` match the serving path.
5. Chunked-projection parity: ANGELSPEC_DFLASH_LOSS_CHUNK must not change the
   selector term (it is position-local, so it accumulates per chunk).
6. Optimizer grouping: the token-indexed codebooks stay off Muon, while the
   selector's genuine linear map stays on it.
7. The shipped hy4 JSON configs resolve with the selector enabled.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.models.draft.path_selector import (
    PathSelector,
    build_path_selector,
    selector_training_candidates,
)
from angelspec.training.optimizer import _is_muon_matrix

SEL_A = 0.5


def _make_config(
    H=64,
    V=128,
    num_target_layers=2,
    enable_path_selector=True,
    path_selector_rank=8,
    path_selector_top_k=4,
):
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
        num_target_layers=num_target_layers,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        mask_token_id=V - 1,
        enable_path_selector=enable_path_selector,
        path_selector_rank=path_selector_rank,
        path_selector_top_k=path_selector_top_k,
    )


def _make_model(
    block_size=4,
    num_anchors=6,
    selector_alpha=SEL_A,
    objective_weighting=True,
    **cfg_kw,
):
    config = _make_config(**cfg_kw)
    draft = DFlashDraftModel(config).to(dtype=torch.float32)
    draft.freeze_embedding()
    return DFlashModel(
        draft_model=draft,
        block_size=block_size,
        num_anchors=num_anchors,
        loss_decay_gamma=4.0,
        path_selector_loss_alpha=selector_alpha,
        path_selector_objective_weighting=objective_weighting,
    )


def _selector_reference(m, batch, seed, *, apply_decay):
    """Recompute the selector term from scratch: top-k -> label injection -> K-way CE.

    ``apply_decay`` mirrors ``path_selector_objective_weighting``: True folds in the
    decay position weights, False weighs every supervised slot equally.
    """
    captured = {}
    real_backbone = m._draft_backbone

    def spy(*args, **kwargs):
        out = real_backbone(*args, **kwargs)
        captured["draft_hidden"], captured["anchors"], captured["keep"], captured["nb"] = out
        return out

    with mock.patch.object(m, "_draft_backbone", side_effect=spy):
        torch.manual_seed(seed)
        comps = m(**batch)[5]

    bsz, seq_len = batch["input_ids"].shape
    bs, nb = m.block_size, captured["nb"]
    anchors, keep = captured["anchors"], captured["keep"]

    offsets = torch.arange(bs).view(1, 1, -1)
    label_idx = anchors.unsqueeze(-1) + offsets
    valid_label = label_idx < seq_len
    safe_idx = label_idx.clamp(max=seq_len - 1)
    target_ids = torch.gather(batch["input_ids"].unsqueeze(1).expand(-1, nb, -1), 2, safe_idx)
    anchor_tokens = torch.gather(batch["input_ids"], 1, anchors.clamp(0, seq_len - 1))
    prev_ids = torch.cat([anchor_tokens.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)

    weight = keep.unsqueeze(-1).expand(-1, -1, bs).float() * valid_label.float()
    weight = weight * (torch.arange(bs).view(1, 1, -1) > 0).float()
    weight = weight * torch.gather(batch["loss_mask"].unsqueeze(1).expand(-1, nb, -1), 2, safe_idx)
    if apply_decay:
        k = torch.arange(bs).view(1, 1, -1)
        weight = weight * torch.exp(-(k - 1).clamp(min=0).float() / m.loss_decay_gamma)

    hidden = captured["draft_hidden"]
    logits = m._compute_draft_logits(hidden, batch["lm_head_weight"], prev_ids, nb).float()

    sel = m.draft_model.path_selector
    cand = logits.topk(sel.top_k, dim=-1).indices
    ids, pos, _ = selector_training_candidates(cand, target_ids.view(bsz, -1))
    scores = sel.score_candidates(logits, hidden, prev_ids.view(bsz, -1), ids)
    ce = F.cross_entropy(scores.flatten(0, -2).float(), pos.flatten(), reduction="none").view_as(
        pos
    )
    expected = (ce * weight.view(bsz, -1)).sum() / weight.sum().clamp(min=1e-6)
    return comps, expected


def _batch(B=2, S=24, H=64, V=128, num_target_layers=2, all_masked=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, V, (B, S), generator=g)
    hidden_states_list = [torch.randn(B, S, H, generator=g) for _ in range(num_target_layers)]
    loss_mask = torch.zeros(B, S) if all_masked else torch.ones(B, S)
    if not all_masked:
        loss_mask[:, :2] = 0  # prompt
    return dict(
        input_ids=input_ids,
        hidden_states_list=hidden_states_list,
        loss_mask=loss_mask,
        lm_head_weight=torch.randn(V, H, generator=g),
    )


def _batch_with_teacher(B=2, S=24, H=64, V=128, num_target_layers=2, seed=0):
    """Batch whose teacher logits argmax to one token that no corpus label uses.

    ``last_hidden_states`` is the first basis vector, so teacher logit ``v`` reduces
    to ``lm_head_weight[v, 0]``; spiking a single row pins the argmax to
    ``teacher_token`` while ``input_ids`` stay in the lower half of the vocab. That
    makes the two candidate label sources disjoint and trivially distinguishable.
    """
    g = torch.Generator().manual_seed(seed)
    teacher_token = V - 2
    lm_head_weight = torch.randn(V, H, generator=g)
    lm_head_weight[teacher_token, 0] = 100.0
    last_hidden_states = torch.zeros(B, S, H)
    last_hidden_states[..., 0] = 1.0
    loss_mask = torch.ones(B, S)
    loss_mask[:, :2] = 0  # prompt
    batch = dict(
        input_ids=torch.randint(0, V // 2, (B, S), generator=g),
        hidden_states_list=[torch.randn(B, S, H, generator=g) for _ in range(num_target_layers)],
        loss_mask=loss_mask,
        lm_head_weight=lm_head_weight,
        last_hidden_states=last_hidden_states,
    )
    return batch, teacher_token


class TestConfigGating(unittest.TestCase):
    def test_off_by_default(self):
        cfg = DFlashConfig()
        self.assertFalse(cfg.enable_path_selector)
        self.assertIsNone(build_path_selector(cfg))

    def test_fields_carried_and_serializable(self):
        cfg = _make_config(path_selector_rank=32, path_selector_top_k=8)
        self.assertTrue(cfg.enable_path_selector)
        self.assertEqual(cfg.path_selector_rank, 32)
        self.assertEqual(cfg.path_selector_top_k, 8)
        d = cfg.to_dict()
        restored = DFlashConfig(**{k: v for k, v in d.items() if k != "transformers_version"})
        self.assertTrue(restored.enable_path_selector)
        self.assertEqual(restored.path_selector_top_k, 8)

    def test_top_k_above_vocab_rejected(self):
        # Serving reranks before any draft-vocab mapping, so vocab_size is the bound.
        with self.assertRaises(ValueError):
            _make_config(V=16, path_selector_top_k=32)

    def test_degenerate_top_k_rejected(self):
        # A 1-way rerank has a identically-zero K-way CE: no training signal.
        with self.assertRaises(ValueError):
            _make_config(path_selector_top_k=1)

    def test_invalid_rank_rejected(self):
        with self.assertRaises(ValueError):
            _make_config(path_selector_rank=0)

    def test_validation_only_when_enabled(self):
        # A disabled selector must not police fields nothing will read.
        cfg = _make_config(enable_path_selector=False, path_selector_top_k=1)
        self.assertFalse(cfg.enable_path_selector)


class TestDrafterWiring(unittest.TestCase):
    def test_head_built_when_enabled(self):
        draft = DFlashDraftModel(_make_config(H=64, V=128, path_selector_rank=8))
        sel = draft.path_selector
        self.assertIsInstance(sel, PathSelector)
        self.assertEqual(sel.predecessor_codebook.shape, (128, 8))
        self.assertEqual(sel.successor_codebook.shape, (128, 8))
        self.assertEqual(sel.hidden_projection.in_features, 64)
        self.assertEqual(sel.hidden_projection.out_features, 8)

    def test_absent_when_disabled(self):
        draft = DFlashDraftModel(_make_config(enable_path_selector=False))
        self.assertIsNone(draft.path_selector)
        self.assertFalse(
            [k for k in draft.state_dict() if "path_selector" in k],
            "disabled selector must not leak keys into the checkpoint",
        )

    def test_state_dict_keys(self):
        draft = DFlashDraftModel(_make_config())
        keys = set(draft.state_dict())
        for expected in (
            "path_selector.predecessor_codebook",
            "path_selector.successor_codebook",
            "path_selector.hidden_projection.weight",
        ):
            self.assertIn(expected, keys)


class TestSelectorMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_score_is_unary_plus_bilinear_transition(self):
        sel = PathSelector(vocab_size=20, hidden_size=6, rank=4, top_k=3)
        unary = torch.randn(2, 5, 20)
        hidden = torch.randn(2, 5, 6)
        prev = torch.randint(0, 20, (2, 5))
        cand = torch.randint(0, 20, (2, 5, 3))

        got = sel.score_candidates(unary, hidden, prev, cand)

        ctx = sel.predecessor_codebook[prev] * sel.hidden_projection(hidden)
        transition = (ctx.unsqueeze(-2) * sel.successor_codebook[cand]).sum(-1)
        self.assertTrue(torch.allclose(got, unary.gather(-1, cand) + transition, atol=1e-5))

    def test_select_uses_unary_top_k(self):
        sel = PathSelector(vocab_size=20, hidden_size=6, rank=4, top_k=3)
        unary = torch.randn(2, 5, 20)
        hidden = torch.randn(2, 5, 6)
        prev = torch.randint(0, 20, (2, 5))

        cand, scores = sel.select(unary, hidden, prev)

        self.assertEqual(cand.shape, (2, 5, 3))
        self.assertTrue(torch.equal(cand, unary.topk(3, dim=-1).indices))
        self.assertTrue(
            torch.allclose(scores, sel.score_candidates(unary, hidden, prev, cand), atol=1e-6)
        )

    def test_transition_depends_on_predecessor(self):
        # The whole point of the head: the same hidden state under a different
        # predecessor must score candidates differently.
        sel = PathSelector(vocab_size=20, hidden_size=6, rank=4, top_k=3)
        hidden = torch.randn(1, 1, 6)
        cand = torch.tensor([[[0, 1, 2]]])
        a = sel.transition_scores(hidden, torch.tensor([[3]]), cand)
        b = sel.transition_scores(hidden, torch.tensor([[7]]), cand)
        self.assertFalse(torch.allclose(a, b))

    def test_training_candidates_keep_present_target(self):
        cand = torch.tensor([[5, 6, 7, 8]])
        target = torch.tensor([7])
        ids, pos, contains = selector_training_candidates(cand, target)
        self.assertTrue(torch.equal(ids, cand))  # untouched
        self.assertTrue(contains.item())
        self.assertEqual(pos.item(), 2)

    def test_training_candidates_inject_missing_target_at_rank_k(self):
        cand = torch.tensor([[5, 6, 7, 8]])
        target = torch.tensor([99])
        ids, pos, contains = selector_training_candidates(cand, target)
        self.assertFalse(contains.item())
        self.assertEqual(pos.item(), 3)  # last slot
        self.assertEqual(ids[0, 3].item(), 99)  # weakest candidate replaced
        self.assertTrue(torch.equal(ids[0, :3], cand[0, :3]))  # stronger ones kept

    def test_training_candidates_does_not_mutate_input(self):
        cand = torch.tensor([[5, 6, 7, 8]])
        original = cand.clone()
        selector_training_candidates(cand, torch.tensor([99]))
        self.assertTrue(torch.equal(cand, original))

    def test_label_indexes_the_target_token(self):
        # target_positions must point at the target inside training_candidate_ids;
        # that identity is what makes the K-way CE a correct objective.
        torch.manual_seed(1)
        cand = torch.randint(0, 50, (4, 6, 5))
        target = torch.randint(0, 50, (4, 6))
        ids, pos, _ = selector_training_candidates(cand, target)
        picked = ids.gather(-1, pos.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.equal(picked, target))


class TestLossWiring(unittest.TestCase):
    def test_components_present_and_detached(self):
        m = _make_model()
        loss, _, lpp, _, _, comps = m(**_batch())
        self.assertLessEqual({"selector_loss", "selector_recall", "selector_acc"}, set(comps))
        for v in comps.values():
            self.assertTrue(torch.isfinite(v).all())
            self.assertFalse(v.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(lpp.shape[0], m.block_size)

    def test_no_components_when_head_absent(self):
        m = _make_model(enable_path_selector=False)
        _, _, _, _, _, comps = m(**_batch())
        self.assertNotIn("selector_loss", comps)

    def test_no_components_when_alpha_zero(self):
        m = _make_model(selector_alpha=0.0)
        _, _, _, _, _, comps = m(**_batch())
        self.assertNotIn("selector_loss", comps)

    def test_loss_is_ce_plus_weighted_selector(self):
        # The logged components must decompose the loss that is actually
        # optimized, otherwise the training curves are unreadable.
        m = _make_model()
        loss, _, _, _, _, comps = m(**_batch(seed=1))
        recomputed = comps["ce_loss"] + SEL_A * comps["selector_loss"]
        self.assertTrue(
            torch.allclose(loss, recomputed, atol=1e-4),
            f"{loss.item()} vs {recomputed.item()}",
        )

    def test_alpha_scales_only_the_selector_term(self):
        torch.manual_seed(0)
        m = _make_model(selector_alpha=1.0)
        b = _batch(seed=3)
        torch.manual_seed(7)
        loss_one, _, _, _, _, comps_one = m(**b)
        m.path_selector_loss_alpha = 2.0
        torch.manual_seed(7)  # same anchor sampling
        loss_two, _, _, _, _, comps_two = m(**b)
        self.assertTrue(torch.allclose(comps_one["ce_loss"], comps_two["ce_loss"], atol=1e-5))
        self.assertTrue(torch.allclose(loss_two - loss_one, comps_one["selector_loss"], atol=1e-4))

    def test_all_masked_is_zero_not_nan(self):
        # Zero supervised tokens leaves the denominator at 0; the term must read 0.
        m = _make_model()
        loss, _, _, _, _, comps = m(**_batch(all_masked=True))
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        self.assertAlmostEqual(comps["selector_loss"].item(), 0.0, places=5)

    def test_grad_reaches_every_selector_parameter(self):
        m = _make_model()
        loss, *_ = m(**_batch(seed=2))
        loss.backward()
        sel = m.draft_model.path_selector
        for name, param in sel.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no grad")
            self.assertGreater(param.grad.abs().sum().item(), 0.0, f"{name} grad is all zero")
        # Backbone still trains, embedding stays frozen.
        self.assertIsNotNone(m.draft_model.context_proj.weight.grad)
        self.assertIsNone(m.draft_model.embed_tokens.weight.grad)

    def test_alpha_zero_reproduces_plain_dflash_gradient(self):
        # Enabling the head must not perturb an existing recipe while its weight
        # is 0: same model, same batch, selector detached vs merely unweighted.
        b = _batch(seed=5)
        m = _make_model(selector_alpha=0.0)

        torch.manual_seed(11)
        loss, *_ = m(**b)
        loss.backward()
        with_head = m.draft_model.context_proj.weight.grad.clone()

        m.zero_grad(set_to_none=True)
        head = m.draft_model.path_selector
        m.draft_model.path_selector = None  # exercise the no-selector code path
        try:
            torch.manual_seed(11)  # identical anchor sampling
            loss, *_ = m(**b)
            loss.backward()
        finally:
            m.draft_model.path_selector = head
        without_head = m.draft_model.context_proj.weight.grad.clone()

        self.assertTrue(torch.allclose(with_head, without_head, atol=1e-6))

    def test_metrics_are_probabilities(self):
        m = _make_model()
        _, _, _, _, _, comps = m(**_batch(seed=4))
        for key in ("selector_recall", "selector_acc"):
            value = comps[key].item()
            self.assertGreaterEqual(value, 0.0, key)
            self.assertLessEqual(value, 1.0, key)


class TestGreedyDraftIds(unittest.TestCase):
    """Progress-bar acc_len / acc_len_tgt must score the serving draft token."""

    def test_falls_back_to_unary_argmax_without_selector(self):
        m = _make_model(enable_path_selector=False)
        unary = torch.tensor([[[0.0, 4.0, 1.0, 0.5]]])  # argmax = 1
        hidden = torch.zeros(1, 1, 64)
        prev = torch.zeros(1, 1, dtype=torch.long)
        self.assertEqual(m._greedy_draft_ids(unary, hidden, prev).item(), 1)

    def test_uses_selector_rerank_not_unary_argmax(self):
        m = _make_model(path_selector_top_k=4)
        # Unary argmax is token 0; token 1 is still inside the top-k.
        unary = torch.zeros(1, 1, 128)
        unary[0, 0, 0] = 10.0
        unary[0, 0, 1] = 9.0
        hidden = torch.zeros(1, 1, 64)
        prev = torch.zeros(1, 1, dtype=torch.long)

        def prefer_token_one(unary_logits, hidden_states, predecessor_ids, top_k=None):
            cand = unary_logits.topk(top_k or 4, dim=-1).indices
            return cand, (cand == 1).to(unary_logits.dtype)

        with mock.patch.object(m.draft_model.path_selector, "select", prefer_token_one):
            pred = m._greedy_draft_ids(unary, hidden, prev)
        self.assertEqual(pred.item(), 1)

    def test_forward_acc_follows_greedy_draft_ids(self):
        m = _make_model()
        batch = _batch(seed=6)
        # Constant corpus tokens so the label at every supervised slot is 7.
        batch["input_ids"] = torch.full_like(batch["input_ids"], 7)

        def always(token):
            def _fn(unary_logits, hidden, prev):
                return torch.full(
                    unary_logits.shape[:-1],
                    token,
                    dtype=torch.long,
                    device=unary_logits.device,
                )

            return _fn

        torch.manual_seed(3)
        with mock.patch.object(m, "_greedy_draft_ids", side_effect=always(7)):
            acc_hit = m(**batch)[3]
        torch.manual_seed(3)
        with mock.patch.object(m, "_greedy_draft_ids", side_effect=always(0)):
            acc_miss = m(**batch)[3]
        self.assertTrue(torch.allclose(acc_hit[1:], torch.ones_like(acc_hit[1:])))
        self.assertTrue(torch.allclose(acc_miss[1:], torch.zeros_like(acc_miss[1:])))

        def unary_only(unary_logits, hidden, prev):
            return unary_logits.argmax(dim=-1)

        torch.manual_seed(3)
        with mock.patch.object(m, "_greedy_draft_ids", side_effect=unary_only):
            acc_unary = m(**batch)[3]
        head = m.draft_model.path_selector
        m.draft_model.path_selector = None
        try:
            torch.manual_seed(3)
            acc_no_head = m(**batch)[3]
        finally:
            m.draft_model.path_selector = head
        self.assertTrue(torch.allclose(acc_unary, acc_no_head, atol=1e-5))


class TestSelectorObjectiveWeighting(unittest.TestCase):
    """``path_selector_objective_weighting`` decouples the selector CE from decay/D-PACE.

    The continuation-value weighting answers "which block position is worth
    fixing", which is a capacity-allocation question. The selector is two
    token-indexed codebooks plus one projection, shared across positions, so the
    weighting mostly skews the bigram statistics it trains on — hence the opt-out.
    """

    def test_defaults_to_on(self):
        self.assertTrue(_make_model().path_selector_objective_weighting)

    def test_unweighted_matches_an_independent_reference(self):
        m = _make_model(block_size=4, num_anchors=6, selector_alpha=1.0, objective_weighting=False)
        comps, expected = _selector_reference(m, _batch(seed=8), 17, apply_decay=False)
        self.assertTrue(
            torch.allclose(comps["selector_loss"], expected, atol=1e-4),
            f"{comps['selector_loss'].item()} vs {expected.item()}",
        )

    def test_weighted_still_matches_the_decay_reference(self):
        m = _make_model(block_size=4, num_anchors=6, selector_alpha=1.0, objective_weighting=True)
        comps, expected = _selector_reference(m, _batch(seed=8), 17, apply_decay=True)
        self.assertTrue(
            torch.allclose(comps["selector_loss"], expected, atol=1e-4),
            f"{comps['selector_loss'].item()} vs {expected.item()}",
        )

    def test_flag_moves_the_selector_term_only(self):
        # Same weights, same anchors: the unary CE must not notice the flag, and
        # the selector term must.
        b = _batch(seed=8)
        on = _make_model(block_size=4, num_anchors=6, selector_alpha=1.0)
        off = _make_model(block_size=4, num_anchors=6, selector_alpha=1.0)
        off.load_state_dict(on.state_dict())
        off.path_selector_objective_weighting = False

        torch.manual_seed(17)
        comps_on = on(**b)[5]
        torch.manual_seed(17)
        comps_off = off(**b)[5]

        self.assertTrue(torch.allclose(comps_on["ce_loss"], comps_off["ce_loss"], atol=1e-5))
        self.assertFalse(
            torch.allclose(comps_on["selector_loss"], comps_off["selector_loss"], atol=1e-4),
            "objective weighting did not reach the selector term",
        )
        for key in ("selector_recall", "selector_acc"):
            # Diagnostics ride the binary mask either way, so they must not move.
            self.assertTrue(torch.allclose(comps_on[key], comps_off[key], atol=1e-6), key)

    def test_chunked_projection_matches_full_path_when_off(self):
        b = _batch(seed=6)
        m = _make_model(block_size=4, num_anchors=6, objective_weighting=False)

        results = []
        for chunk in ("0", "8"):  # 0 => full path, 8 rows => 2 blocks/chunk
            with mock.patch.dict("os.environ", {"ANGELSPEC_DFLASH_LOSS_CHUNK": chunk}):
                torch.manual_seed(13)
                loss, _, _, _, _, comps = m(**b)
            results.append((loss, comps))

        (full_loss, full_comps), (chunked_loss, chunked_comps) = results
        self.assertIn("selector_loss", chunked_comps, "chunked path dropped the selector term")
        self.assertTrue(
            torch.allclose(
                full_comps["selector_loss"], chunked_comps["selector_loss"], rtol=1e-4, atol=1e-4
            ),
            f"{full_comps['selector_loss'].item()} vs {chunked_comps['selector_loss'].item()}",
        )
        self.assertTrue(torch.allclose(full_loss, chunked_loss, rtol=1e-4, atol=1e-4))

    def test_all_masked_is_zero_not_nan(self):
        m = _make_model(objective_weighting=False)
        loss, _, _, _, _, comps = m(**_batch(all_masked=True))
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(comps["selector_loss"].item(), 0.0, places=5)


class TestSelectorLabelSource(unittest.TestCase):
    """The K-way CE must rank the token the unary head is trained toward.

    KL/LK replace the corpus CE and ``ce_loss_alpha=0`` drops it, so from then on
    the drafter is aligned to the target's distribution and acceptance is tested
    against the target's token; labelling the selector with the corpus token there
    would train it to rank a token the verifier rejects wherever the two disagree.
    """

    def _run(self, *, lk=0.0, e2e_tv=0.0, ce_alpha=1.0, teacher_labels=True):
        batch, teacher_token = _batch_with_teacher()
        m = _make_model()
        m.lk_loss_weight = lk
        m.e2e_tv_loss_weight = e2e_tv
        m.ce_loss_alpha = ce_alpha
        m.path_selector_teacher_labels = teacher_labels

        captured = {}
        original = m._path_selector_sums

        def spy(**kwargs):
            captured["label_ids"] = kwargs["label_ids"].clone()
            return original(**kwargs)

        with mock.patch.object(m, "_path_selector_sums", spy):
            torch.manual_seed(11)  # fixed anchor sampling
            _, _, _, _, _, comps = m(**batch)
        return captured["label_ids"], comps, batch, teacher_token

    def test_lk_distillation_switches_to_the_target_argmax(self):
        labels, _, _, teacher_token = self._run(lk=1.0)
        self.assertTrue((labels == teacher_token).all())

    def test_plain_ce_keeps_the_corpus_token(self):
        labels, _, batch, teacher_token = self._run()
        self.assertTrue(torch.isin(labels, batch["input_ids"]).all())
        self.assertFalse((labels == teacher_token).any())

    def test_e2e_tv_alone_keeps_the_corpus_token(self):
        # e2e-TV is added on top of the CE instead of replacing it, so the tokens
        # themselves are still trained against the corpus.
        labels, _, batch, teacher_token = self._run(e2e_tv=1.0)
        self.assertTrue(torch.isin(labels, batch["input_ids"]).all())
        self.assertFalse((labels == teacher_token).any())

    def test_e2e_tv_without_corpus_ce_switches_to_the_target_argmax(self):
        # Pure e2e-TV recipe: ce_loss_alpha=0 leaves nothing pulling the unary head
        # toward the corpus, so the corpus token is no longer a label the verifier
        # would accept.
        labels, _, _, teacher_token = self._run(e2e_tv=1.0, ce_alpha=0.0)
        self.assertTrue((labels == teacher_token).all())

    def test_zero_ce_without_a_teacher_term_falls_back_to_the_corpus_token(self):
        # Degenerate config: no teacher logits exist to label against, so the
        # switch must stay off rather than dereference a missing teacher.
        labels, _, batch, teacher_token = self._run(ce_alpha=0.0)
        self.assertTrue(torch.isin(labels, batch["input_ids"]).all())
        self.assertFalse((labels == teacher_token).any())

    def test_flag_off_restores_the_corpus_token_under_distillation(self):
        labels, _, batch, teacher_token = self._run(lk=1.0, teacher_labels=False)
        self.assertTrue(torch.isin(labels, batch["input_ids"]).all())
        self.assertFalse((labels == teacher_token).any())

    def test_label_source_changes_the_optimized_term(self):
        _, on, _, _ = self._run(lk=1.0)
        _, off, _, _ = self._run(lk=1.0, teacher_labels=False)
        self.assertFalse(
            torch.allclose(on["selector_loss"], off["selector_loss"], atol=1e-4),
            "label source did not reach the selector term",
        )


class TestChunkedParity(unittest.TestCase):
    def test_chunked_projection_matches_full_path(self):
        # One model, one batch, same anchor sampling: only the projection chunking
        # differs, and the selector must be invariant to it.
        b = _batch(seed=6)
        m = _make_model(block_size=4, num_anchors=6)

        results = []
        for chunk in ("0", "8"):  # 0 => full path, 8 rows => 2 blocks/chunk
            with mock.patch.dict("os.environ", {"ANGELSPEC_DFLASH_LOSS_CHUNK": chunk}):
                torch.manual_seed(13)
                loss, _, _, _, _, comps = m(**b)
            results.append((loss, comps))

        (full_loss, full_comps), (chunked_loss, chunked_comps) = results
        self.assertIn("selector_loss", chunked_comps, "chunked path dropped the selector term")
        for key in ("ce_loss", "selector_loss"):
            self.assertTrue(
                torch.allclose(full_comps[key], chunked_comps[key], rtol=1e-4, atol=1e-4),
                f"{key}: {full_comps[key].item()} vs {chunked_comps[key].item()}",
            )
        self.assertTrue(
            torch.allclose(full_loss, chunked_loss, rtol=1e-4, atol=1e-4),
            f"{full_loss.item()} vs {chunked_loss.item()}",
        )


class TestSelectorObjective(unittest.TestCase):
    def test_matches_an_independent_reference(self):
        # Recompute the selector term from scratch (top-k -> label injection ->
        # decay-weighted K-way CE) and require the wrapper to agree, pinning the
        # objective rather than just its plumbing.
        m = _make_model(block_size=4, num_anchors=6, selector_alpha=1.0)
        b = _batch(seed=8)

        captured = {}
        real_backbone = m._draft_backbone

        def spy(*args, **kwargs):
            out = real_backbone(*args, **kwargs)
            captured["draft_hidden"], captured["anchors"], captured["keep"], captured["nb"] = out
            return out

        with mock.patch.object(m, "_draft_backbone", side_effect=spy):
            torch.manual_seed(17)
            _, _, _, _, _, comps = m(**b)

        bsz, seq_len = b["input_ids"].shape
        bs, nb = m.block_size, captured["nb"]
        anchors, keep = captured["anchors"], captured["keep"]

        offsets = torch.arange(bs).view(1, 1, -1)
        label_idx = anchors.unsqueeze(-1) + offsets
        valid_label = label_idx < seq_len
        safe_idx = label_idx.clamp(max=seq_len - 1)
        target_ids = torch.gather(b["input_ids"].unsqueeze(1).expand(-1, nb, -1), 2, safe_idx)
        anchor_tokens = torch.gather(b["input_ids"], 1, anchors.clamp(0, seq_len - 1))
        prev_ids = torch.cat([anchor_tokens.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)

        weight = keep.unsqueeze(-1).expand(-1, -1, bs).float() * valid_label.float()
        weight = weight * (torch.arange(bs).view(1, 1, -1) > 0).float()
        weight = weight * torch.gather(b["loss_mask"].unsqueeze(1).expand(-1, nb, -1), 2, safe_idx)
        k = torch.arange(bs).view(1, 1, -1)
        weight = weight * torch.exp(-(k - 1).clamp(min=0).float() / m.loss_decay_gamma)

        hidden = captured["draft_hidden"]
        logits = m._compute_draft_logits(hidden, b["lm_head_weight"], prev_ids, nb).float()

        sel = m.draft_model.path_selector
        cand = logits.topk(sel.top_k, dim=-1).indices
        ids, pos, _ = selector_training_candidates(cand, target_ids.view(bsz, -1))
        scores = sel.score_candidates(logits, hidden, prev_ids.view(bsz, -1), ids)
        ce = F.cross_entropy(
            scores.flatten(0, -2).float(), pos.flatten(), reduction="none"
        ).view_as(pos)
        expected = (ce * weight.view(bsz, -1)).sum() / weight.sum().clamp(min=1e-6)

        self.assertTrue(
            torch.allclose(comps["selector_loss"], expected, atol=1e-4),
            f"{comps['selector_loss'].item()} vs {expected.item()}",
        )


class TestMuonGrouping(unittest.TestCase):
    """The codebooks are token-indexed lookup tables, so Muon must not touch them.

    They are non-degenerate 2-D params, so nothing crashes if they leak into the
    Muon group -- they would just be silently orthogonalized (and all-gathered)
    every step. That makes this split worth pinning explicitly.
    """

    def _split(self):
        draft = DFlashDraftModel(_make_config()).to(dtype=torch.float32)
        draft.freeze_embedding()
        muon, adamw = [], []
        for name, param in draft.named_parameters():
            if not param.requires_grad:
                continue
            (muon if _is_muon_matrix(name, param) else adamw).append(name)
        return muon, adamw

    def test_codebooks_go_to_adamw(self):
        muon, adamw = self._split()
        for name in (
            "path_selector.predecessor_codebook",
            "path_selector.successor_codebook",
        ):
            self.assertIn(name, adamw)
            self.assertNotIn(name, muon)

    def test_hidden_projection_stays_on_muon(self):
        # This one IS a genuine linear map (hidden -> rank), not a lookup table,
        # so excluding the codebooks must not over-reach and catch it too.
        muon, adamw = self._split()
        self.assertIn("path_selector.hidden_projection.weight", muon)
        self.assertNotIn("path_selector.hidden_projection.weight", adamw)

    def test_backbone_matrices_unaffected(self):
        muon, _ = self._split()
        self.assertIn("context_proj.weight", muon)
        self.assertTrue(any(n.endswith("self_attn.q_proj.weight") for n in muon))


if __name__ == "__main__":
    unittest.main()
