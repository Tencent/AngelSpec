"""Tests for single-head MTP draft + TTT training components.

Covers:
1. MTPConfig / MTPDraftModel: construction, forward shapes, frozen lm_head/embed
2. mtp_loss_from_hs: CE matches cross_entropy; KL full == top-k(V) variant a;
   variant b runs; sum semantics
3. ★2-step TTT (MTPModel): KV cache concat grows 1→2 and step-1 block is not
   overwritten; position_ids advance by the cached-step offset; the left-shift
   (padding) feeds the next predicted token; cached vs uncached step-1 match
4. CPT key remap: model.layers.<N>.* loads into MTPDraftModel (strict on the
   MTP submodules, embed/lm_head excluded)
"""

import unittest

import torch
import torch.nn.functional as F

from angelspec.models.draft.mtp import Hy3MoE, MTPConfig, MTPDraftModel
from angelspec.models.mtp import MTPModel
from angelspec.models.ops.loss import mtp_loss_from_hs


def _tiny_config(H=32, V=64, experts=4, top_k=2):
    return MTPConfig(
        hidden_size=H,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        qk_norm=True,
        vocab_size=V,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        rope_theta=10000.0,
        num_experts=experts,
        num_shared_experts=1,
        num_experts_per_tok=top_k,
        moe_intermediate_size=32,
        route_norm=True,
        router_scaling_factor=1.5,
        target_hidden_size=H,
        tie_lm_head=True,
    )


def _causal_mask(S, dtype=torch.float32):
    return torch.triu(torch.full((S, S), float("-inf"), dtype=dtype), diagonal=1)[None, None]


class TestMTPConfig(unittest.TestCase):
    def test_defaults_hy3(self):
        cfg = MTPConfig()
        self.assertEqual(cfg.vocab_size, 120832)
        self.assertEqual(cfg.num_experts, 192)
        self.assertEqual(cfg.hidden_size, 4096)
        self.assertTrue(cfg.qk_norm)

    def test_serialization_roundtrip(self):
        cfg = _tiny_config()
        d = cfg.to_dict()
        restored = MTPConfig(**{k: v for k, v in d.items() if k != "transformers_version"})
        self.assertEqual(restored.num_experts, cfg.num_experts)
        self.assertEqual(restored.vocab_size, cfg.vocab_size)


class TestMTPDraftModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = _tiny_config()
        self.model = MTPDraftModel(self.cfg).to(torch.float32)
        # lm_head is tied to the (frozen) target head by the trainer; do the same
        # here so compute_logits has a weight to use.
        self.model.set_lm_head_weight(torch.randn(self.cfg.vocab_size, self.cfg.hidden_size))

    def test_backbone_and_logits_shapes(self):
        B, S, H = 2, 6, self.cfg.hidden_size
        ids = torch.randint(0, self.cfg.vocab_size, (B, S))
        hs = torch.randn(B, S, H)
        emb = self.model.embed_input_ids(ids)
        pos = torch.arange(S).unsqueeze(0).expand(B, S)
        out, ck, cv = self.model.backbone(
            emb, self.model.project_hidden_states(hs), _causal_mask(S), pos, use_cache=True
        )
        self.assertEqual(tuple(out.shape), (B, S, H))
        # cache: [bsz, num_heads, num_cached_steps, seq_len, head_dim]
        self.assertEqual(ck.shape[2], 1)
        self.assertEqual(cv.shape[2], 1)
        logits = self.model.compute_logits(out)
        self.assertEqual(tuple(logits.shape), (B, S, self.cfg.vocab_size))

    def test_lm_head_and_embedding_frozen(self):
        # lm_head is a plain attribute (tied to target, never trained, invisible
        # to FSDP — not a Parameter or buffer); embedding frozen after freeze.
        self.model.freeze_embedding()
        param_names = {n for n, _ in self.model.named_parameters()}
        buffer_names = {n for n, _ in self.model.named_buffers()}
        self.assertNotIn("lm_head_weight", param_names)  # not a Parameter
        self.assertNotIn("lm_head_weight", buffer_names)  # not a buffer either
        self.assertFalse(self.model.embed_tokens.weight.requires_grad)
        # MTP block params are trainable
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        self.assertTrue(any("eh_proj" in n for n in trainable))
        self.assertTrue(any("midlayer" in n for n in trainable))
        self.assertFalse(any("lm_head" in n for n in trainable))
        self.assertFalse(any("embed_tokens" in n for n in trainable))

    def test_set_lm_head_weight_ties_and_freezes(self):
        W = torch.randn(self.cfg.vocab_size, self.cfg.hidden_size)
        self.model.set_lm_head_weight(W)
        # stored as a buffer (no grad, not a Parameter), sharing the given tensor
        self.assertNotIn("lm_head_weight", {n for n, _ in self.model.named_parameters()})
        self.assertTrue(torch.equal(self.model.lm_head_weight, W))

    def test_position_zero_embedding_masked(self):
        # hy_v3 masks the position-0 token embedding inside backbone.
        B, S, H = 1, 4, self.cfg.hidden_size
        ids = torch.randint(1, self.cfg.vocab_size, (B, S))
        emb = self.model.embed_input_ids(ids)
        hs = torch.randn(B, S, H)
        pos = torch.arange(S).unsqueeze(0)
        # Monkeypatch enorm to capture its input.
        captured = {}
        orig = self.model.enorm.forward

        def spy(x):
            captured["x"] = x.detach().clone()
            return orig(x)

        self.model.enorm.forward = spy
        self.model.backbone(emb, hs, _causal_mask(S), pos, use_cache=True)
        # position 0 row should be zeroed before enorm
        self.assertTrue(torch.allclose(captured["x"][:, 0], torch.zeros(H)))
        self.assertFalse(torch.allclose(captured["x"][:, 1], torch.zeros(H)))


class TestMTPLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.N, self.H, self.V = 24, 16, 40
        self.hs = torch.randn(self.N, self.H)
        self.ths = torch.randn(self.N, self.H)
        self.W = torch.randn(self.V, self.H)
        self.vidx = torch.arange(self.N)

    def test_ce_matches_cross_entropy(self):
        ce, kl, corr, cnt, corr_gt = mtp_loss_from_hs(self.hs, self.ths, self.vidx, self.W, self.W)
        ref_tokens = (self.ths @ self.W.T).argmax(-1)
        ref = F.cross_entropy(self.hs @ self.W.T, ref_tokens, reduction="sum")
        self.assertTrue(torch.allclose(ce, ref, atol=1e-4))
        self.assertEqual(float(kl), 0.0)
        self.assertEqual(float(cnt), self.N)

    def test_kl_full_equals_topk_full_variant_a(self):
        _, kl_full, *_ = mtp_loss_from_hs(
            self.hs, self.ths, self.vidx, self.W, self.W, use_kl=True, kl_topk=None
        )
        _, kl_topk, *_ = mtp_loss_from_hs(
            self.hs,
            self.ths,
            self.vidx,
            self.W,
            self.W,
            use_kl=True,
            kl_topk=self.V,
            kl_variant="a",
        )
        self.assertTrue(torch.allclose(kl_full, kl_topk, atol=1e-3))

    def test_kl_variant_b_runs_and_nonneg(self):
        _, kl_b, *_ = mtp_loss_from_hs(
            self.hs, self.ths, self.vidx, self.W, self.W, use_kl=True, kl_topk=5, kl_variant="b"
        )
        self.assertGreaterEqual(float(kl_b), -1e-4)

    def test_masked_subset_only(self):
        vidx = torch.tensor([0, 5, 10])
        ce, _, _, cnt, _ = mtp_loss_from_hs(self.hs, self.ths, vidx, self.W, self.W)
        self.assertEqual(float(cnt), 3)

    def test_ce_ground_truth_target(self):
        # CE against ground-truth tokens instead of target argmax.
        gt = torch.randint(0, self.V, (self.N,))
        ce, _, _, _, corr_gt = mtp_loss_from_hs(
            self.hs,
            self.ths,
            self.vidx,
            self.W,
            self.W,
            gt_labels_flat=gt,
            ce_use_ground_truth=True,
        )
        ref = F.cross_entropy(self.hs @ self.W.T, gt, reduction="sum")
        self.assertTrue(torch.allclose(ce, ref, atol=1e-4))
        ref_corr_gt = ((self.hs @ self.W.T).argmax(-1) == gt).float().sum()
        self.assertTrue(torch.allclose(corr_gt, ref_corr_gt))

    def test_single_step_form_fills_kl_slot(self):
        # single_step_form routes the tv/kl/lk term through the kl_sum slot; the CE
        # slot and count are unchanged from the legacy call.
        ce0, _, _, cnt0, _ = mtp_loss_from_hs(self.hs, self.ths, self.vidx, self.W, self.W)
        for form in ("tv", "kl", "lk"):
            ce, distill, _, cnt, _ = mtp_loss_from_hs(
                self.hs, self.ths, self.vidx, self.W, self.W, single_step_form=form
            )
            self.assertTrue(torch.allclose(ce, ce0, atol=1e-5))
            self.assertEqual(float(cnt), float(cnt0))
            self.assertGreaterEqual(float(distill), -1e-4)

    def test_lk_matches_dflash_hybrid(self):
        from angelspec.models.dflash import DFlashModel
        from angelspec.models.ops.loss import lk_tv_kl_per_pos

        s = self.hs @ self.W.T
        t = self.ths @ self.W.T
        ell, _ = lk_tv_kl_per_pos(s, t, form="lk", eta=3.0)
        ref = DFlashModel._compute_lk_loss(None, s, t, loss_type="hybrid", eta=3.0)
        self.assertTrue(torch.allclose(ell, ref, atol=1e-5))

    def test_lk_eta_zero_equals_kl(self):
        from angelspec.models.ops.loss import lk_tv_kl_per_pos

        s = self.hs @ self.W.T
        t = self.ths @ self.W.T
        lk0, _ = lk_tv_kl_per_pos(s, t, form="lk", eta=0.0)
        kl, _ = lk_tv_kl_per_pos(s, t, form="kl")
        self.assertTrue(torch.allclose(lk0, kl, atol=1e-6))

    def test_per_pos_scatter_shapes(self):
        vidx = torch.tensor([0, 5, 10])
        out = mtp_loss_from_hs(
            self.hs,
            self.ths,
            vidx,
            self.W,
            self.W,
            single_step_form="tv",
            return_per_pos=True,
        )
        self.assertEqual(len(out), 7)
        ell_full, alpha_full = out[5], out[6]
        self.assertEqual(tuple(ell_full.shape), (self.N,))
        self.assertEqual(tuple(alpha_full.shape), (self.N,))
        # non-valid positions: ell=0, alpha=1
        mask = torch.ones(self.N, dtype=torch.bool)
        mask[vidx] = False
        self.assertTrue(torch.allclose(ell_full[mask], torch.zeros(mask.sum())))
        self.assertTrue(torch.allclose(alpha_full[mask], torch.ones(mask.sum())))


class TestMTPE2ELoss(unittest.TestCase):
    """e2e acceptance-chain coupling: direct objective + W_m surrogate."""

    def _model(self, length, e2e_direct):
        m = MTPModel.__new__(MTPModel)
        m.e2e_weighting = True
        m.e2e_direct = e2e_direct
        m._usp_size = 1
        m._usp_group = None
        return m

    def test_direct_matches_manual(self):
        torch.manual_seed(0)
        gamma, M = 3, 4
        alpha = [torch.rand(M) for _ in range(gamma)]
        ell = [torch.rand(M) for _ in range(gamma)]
        mask = [torch.ones(M) for _ in range(gamma)]
        loss = self._model(gamma, True)._e2e_loss(ell, alpha, mask)
        # 1 - (1/γ) Σ_j Π_{i<=j} α_i, averaged over positions
        stack = torch.stack(alpha, 0)
        expected = (1.0 - torch.cumprod(stack, 0).sum(0) / gamma).mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_direct_zero_when_alpha_one(self):
        gamma, M = 4, 3
        alpha = [torch.ones(M) for _ in range(gamma)]
        ell = [torch.rand(M) for _ in range(gamma)]
        mask = [torch.ones(M) for _ in range(gamma)]
        loss = self._model(gamma, True)._e2e_loss(ell, alpha, mask)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_mask_intersection_drops_chain(self):
        gamma, M = 2, 3
        alpha = [torch.tensor([0.2, 0.5, 0.9]) for _ in range(gamma)]
        ell = [torch.tensor([1.0, 1.0, 1.0]) for _ in range(gamma)]
        # position 1 invalid at step 1 -> its chain excluded
        mask = [torch.ones(M), torch.tensor([1.0, 0.0, 1.0])]
        loss = self._model(gamma, True)._e2e_loss(ell, alpha, mask)
        valid_all = torch.stack(mask, 0).prod(0)
        per_pos = 1.0 - torch.cumprod(torch.stack(alpha, 0), 0).sum(0) / gamma
        expected = (per_pos * valid_all).sum() / valid_all.sum()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_wm_matches_manual(self):
        torch.manual_seed(2)
        gamma, M = 3, 5
        alpha = [torch.rand(M) for _ in range(gamma)]
        ell = [torch.rand(M) for _ in range(gamma)]
        mask = [torch.ones(M) for _ in range(gamma)]
        loss = self._model(gamma, False)._e2e_loss(ell, alpha, mask)
        a = torch.stack(alpha, 0)
        pre = [torch.ones(M)]
        for m in range(1, gamma):
            pre.append(pre[m - 1] * a[m - 1])
        S = [None] * gamma
        S[gamma - 1] = torch.ones(M)
        for m in range(gamma - 2, -1, -1):
            S[m] = 1.0 + a[m + 1] * S[m + 1]
        W = torch.stack([(pre[m] * S[m]) / gamma for m in range(gamma)], 0)
        expected = (W * torch.stack(ell, 0)).sum(0).mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_direct_differentiable_through_alpha(self):
        gamma, M = 3, 4
        alpha = [torch.rand(M, dtype=torch.double, requires_grad=True) for _ in range(gamma)]
        mask = [torch.ones(M, dtype=torch.double) for _ in range(gamma)]
        model = self._model(gamma, True)
        torch.autograd.gradcheck(
            lambda *a: model._e2e_loss(list(a), list(a), mask),
            tuple(alpha),
        )

    def test_wm_weights_detached(self):
        # W_m path: gradient flows only through ell, not alpha.
        gamma, M = 3, 4
        alpha = [torch.rand(M, requires_grad=True) for _ in range(gamma)]
        ell = [torch.rand(M, requires_grad=True) for _ in range(gamma)]
        mask = [torch.ones(M) for _ in range(gamma)]
        loss = self._model(gamma, False)._e2e_loss(ell, alpha, mask)
        loss.backward()
        for a in alpha:
            self.assertIsNone(a.grad)
        for e in ell:
            self.assertIsNotNone(e.grad)


class TestMTPTwoStepTTT(unittest.TestCase):
    """The shift / KV-cache alignment is the easiest place to introduce bugs."""

    def setUp(self):
        torch.manual_seed(2)
        self.cfg = _tiny_config()
        self.draft = MTPDraftModel(self.cfg).to(torch.float32).eval()
        # Tie a deterministic lm_head.
        W = torch.randn(self.cfg.vocab_size, self.cfg.hidden_size)
        self.draft.set_lm_head_weight(W)
        self.model = MTPModel(
            self.draft, length=2, attention_backend="sdpa", gradient_checkpointing=False
        ).eval()

    def _run(self):
        B, S, H = 2, 5, self.cfg.hidden_size
        input_ids = torch.randint(1, self.cfg.vocab_size, (B, S))
        hidden = torch.randn(B, S, H)
        target_hidden = torch.randn(B, S, H)
        target_W = self.draft.lm_head_weight
        loss_mask = torch.ones(B, S)
        return input_ids, hidden, target_hidden, target_W, loss_mask

    def test_kv_cache_grows_and_step1_preserved(self):
        """Instrument backbone to capture per-step cache; assert growth 1→2 and
        that step-1's cached K/V block is untouched at step 2."""
        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        captures = []
        orig_backbone = self.draft.backbone

        def spy(
            input_embeds,
            hidden_states,
            attention_mask,
            position_ids,
            cache_keys=None,
            cache_values=None,
            use_cache=True,
            ctx_doc_ids=None,
        ):
            out, ck, cv = orig_backbone(
                input_embeds,
                hidden_states,
                attention_mask,
                position_ids,
                cache_keys,
                cache_values,
                use_cache,
            )
            captures.append(
                {
                    "in_cache_steps": 0 if cache_keys is None else cache_keys.shape[2],
                    "out_cache_steps": ck.shape[2],
                    "out_ck": ck.detach().clone(),
                    "in_pos": position_ids.detach().clone(),
                }
            )
            return out, ck, cv

        self.draft.backbone = spy
        with torch.no_grad():
            self.model(
                input_ids=input_ids,
                attention_mask=torch.ones(input_ids.shape),
                target_hidden_states=target_hidden,
                target_lm_head_weight=target_W,
                loss_mask=loss_mask,
                hidden_states=hidden,
            )
        self.assertEqual(len(captures), 2)
        # step 0: no incoming cache, produces 1 cached block
        self.assertEqual(captures[0]["in_cache_steps"], 0)
        self.assertEqual(captures[0]["out_cache_steps"], 1)
        # step 1: receives 1 cached block, produces 2
        self.assertEqual(captures[1]["in_cache_steps"], 1)
        self.assertEqual(captures[1]["out_cache_steps"], 2)
        # the first cached block must be byte-identical after step 1 appended a new one
        step0_block = captures[0]["out_ck"][:, :, 0]
        step1_first_block = captures[1]["out_ck"][:, :, 0]
        self.assertTrue(torch.equal(step0_block, step1_first_block))

    def test_position_ids_offset_by_cached_steps(self):
        """RoPE positions advance by +lck per cached step (the EAGLE chain),
        so step 1 effectively sees positions shifted by 1 vs step 0."""
        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        B, S = input_ids.shape
        # The MTPModel builds contiguous base position_ids [0..S-1]; inside the
        # cached attention, query positions are base + lck. Assert the base is
        # contiguous and identical across steps (the offset is applied in attn).
        pos_seen = []
        orig = self.draft.backbone

        def spy(
            input_embeds,
            hidden_states,
            attention_mask,
            position_ids,
            cache_keys=None,
            cache_values=None,
            use_cache=True,
            ctx_doc_ids=None,
        ):
            pos_seen.append(position_ids.detach().clone())
            return orig(
                input_embeds,
                hidden_states,
                attention_mask,
                position_ids,
                cache_keys,
                cache_values,
                use_cache,
                ctx_doc_ids,
            )

        self.draft.backbone = spy
        with torch.no_grad():
            self.model(
                input_ids=input_ids,
                attention_mask=torch.ones(input_ids.shape),
                target_hidden_states=target_hidden,
                target_lm_head_weight=target_W,
                loss_mask=loss_mask,
                hidden_states=hidden,
            )
        # MTPModel builds a shared (1, S) contiguous base; the per-step +lck
        # offset is applied inside the cached attention, not here.
        expected = torch.arange(S).unsqueeze(0)
        for p in pos_seen:
            self.assertTrue(torch.equal(p, expected))

    def test_left_shift_feeds_next_token(self):
        """Step 2's input_ids must be step 1's input shifted left by one."""
        from angelspec.utils.tensor import padding

        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        seen_ids = []
        orig_embed = self.draft.embed_input_ids

        def spy(ids):
            seen_ids.append(ids.detach().clone())
            return orig_embed(ids)

        self.draft.embed_input_ids = spy
        with torch.no_grad():
            self.model(
                input_ids=input_ids.clamp(min=0, max=self.cfg.vocab_size - 1),
                attention_mask=torch.ones(input_ids.shape),
                target_hidden_states=target_hidden,
                target_lm_head_weight=target_W,
                loss_mask=loss_mask,
                hidden_states=hidden,
            )
        self.assertEqual(len(seen_ids), 2)
        clamped = input_ids.clamp(min=0, max=self.cfg.vocab_size - 1)
        self.assertTrue(torch.equal(seen_ids[0], clamped))
        self.assertTrue(torch.equal(seen_ids[1], padding(clamped, left=False)))

    def test_forward_returns_per_step_losses(self):
        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        ce, kl, vl, acc, cnt, acc_gt, e2e, vkl = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones(input_ids.shape),
            target_hidden_states=target_hidden,
            target_lm_head_weight=target_W,
            loss_mask=loss_mask,
            hidden_states=hidden,
        )
        self.assertEqual(len(ce), 2)
        self.assertEqual(len(kl), 2)
        self.assertEqual(len(acc), 2)
        self.assertEqual(len(acc_gt), 2)
        self.assertIsNone(e2e)  # e2e off by default
        for c in ce:
            self.assertTrue(torch.isfinite(c).all())

    def test_forward_e2e_direct_backward(self):
        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        model = MTPModel(
            self.draft,
            length=2,
            attention_backend="sdpa",
            gradient_checkpointing=False,
            single_step_form="tv",
            e2e_weighting=True,
            e2e_direct=True,
        ).train()
        ce, kl, vl, acc, cnt, acc_gt, e2e, vkl = model(
            input_ids=input_ids,
            attention_mask=torch.ones(input_ids.shape),
            target_hidden_states=target_hidden,
            target_lm_head_weight=target_W,
            loss_mask=loss_mask,
            hidden_states=hidden.requires_grad_(True),
        )
        self.assertIsNotNone(e2e)
        self.assertEqual(e2e.dim(), 0)
        self.assertTrue(torch.isfinite(e2e))
        e2e.backward()

    def test_forward_e2e_wm_runs(self):
        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        model = MTPModel(
            self.draft,
            length=2,
            attention_backend="sdpa",
            gradient_checkpointing=False,
            single_step_form="lk",
            e2e_weighting=True,
            e2e_direct=False,
        ).train()
        *_, e2e, _ = model(
            input_ids=input_ids,
            attention_mask=torch.ones(input_ids.shape),
            target_hidden_states=target_hidden,
            target_lm_head_weight=target_W,
            loss_mask=loss_mask,
            hidden_states=hidden,
        )
        self.assertIsNotNone(e2e)
        self.assertTrue(torch.isfinite(e2e))


class TestMTPOnPolicyTTT(unittest.TestCase):
    """On-policy TTT (EAGLE-3 faithful default): step>=1 embeds the
    draft's own previous-step argmax (stop-grad); step 0 stays teacher-forced;
    the CE target chain is ground-truth and must never be polluted by argmax."""

    def setUp(self):
        torch.manual_seed(3)
        self.cfg = _tiny_config()
        self.draft = MTPDraftModel(self.cfg).to(torch.float32).eval()
        W = torch.randn(self.cfg.vocab_size, self.cfg.hidden_size)
        self.draft.set_lm_head_weight(W)

    def _model(self, on_policy):
        return MTPModel(
            self.draft,
            length=3,
            attention_backend="sdpa",
            gradient_checkpointing=False,
            on_policy=on_policy,
        ).eval()

    def _run(self):
        B, S, H = 2, 5, self.cfg.hidden_size
        input_ids = torch.randint(1, self.cfg.vocab_size, (B, S))
        hidden = torch.randn(B, S, H)
        target_hidden = torch.randn(B, S, H)
        loss_mask = torch.ones(B, S)
        return input_ids, hidden, target_hidden, self.draft.lm_head_weight, loss_mask

    def _spy_embed(self):
        seen = []
        orig = self.draft.embed_input_ids

        def spy(ids):
            seen.append(ids.detach().clone())
            return orig(ids)

        self.draft.embed_input_ids = spy
        return seen

    def test_on_policy_feeds_prev_argmax(self):
        """step>=1 embeds the previous step's own argmax, NOT the shifted gt."""
        from angelspec.utils.tensor import padding

        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        clamped = input_ids.clamp(min=0, max=self.cfg.vocab_size - 1)

        # Capture each step's backbone output so we can recompute its argmax.
        outs = []
        orig_bb = self.draft.backbone

        def bb_spy(
            input_embeds,
            hidden_states,
            attention_mask,
            position_ids,
            cache_keys=None,
            cache_values=None,
            use_cache=True,
            ctx_doc_ids=None,
        ):
            out, ck, cv = orig_bb(
                input_embeds,
                hidden_states,
                attention_mask,
                position_ids,
                cache_keys,
                cache_values,
                use_cache,
            )
            outs.append(out.detach().clone())
            return out, ck, cv

        self.draft.backbone = bb_spy
        seen = self._spy_embed()
        model = self._model(on_policy=True)
        with torch.no_grad():
            model(
                input_ids=clamped,
                attention_mask=torch.ones(input_ids.shape),
                target_hidden_states=target_hidden,
                target_lm_head_weight=target_W,
                loss_mask=loss_mask,
                hidden_states=hidden,
            )
        self.assertEqual(len(seen), 3)
        # step 0 is teacher-forced (embeds the original ground-truth).
        self.assertTrue(torch.equal(seen[0], clamped))
        # step 1 embeds step 0's own argmax (NOT the shifted ground-truth).
        expected_pred0 = self.draft.compute_logits(outs[0]).argmax(-1)
        self.assertTrue(torch.equal(seen[1], expected_pred0))
        self.assertFalse(torch.equal(seen[1], padding(clamped, left=False)))
        # step 2 embeds step 1's own argmax.
        expected_pred1 = self.draft.compute_logits(outs[1]).argmax(-1)
        self.assertTrue(torch.equal(seen[2], expected_pred1))

    def test_on_policy_gt_labels_unpolluted(self):
        """CE target chain stays ground-truth even under on-policy embedding."""
        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        clamped = input_ids.clamp(min=0, max=self.cfg.vocab_size - 1)

        seen_gt = []
        orig_step_loss = MTPModel._step_loss

        def step_loss_spy(self_, *args, gt_labels=None, **kwargs):
            seen_gt.append(gt_labels.detach().clone())
            return orig_step_loss(self_, *args, gt_labels=gt_labels, **kwargs)

        model = self._model(on_policy=True)
        model._step_loss = step_loss_spy.__get__(model, MTPModel)
        with torch.no_grad():
            model(
                input_ids=clamped,
                attention_mask=torch.ones(input_ids.shape),
                target_hidden_states=target_hidden,
                target_lm_head_weight=target_W,
                loss_mask=loss_mask,
                hidden_states=hidden,
            )
        # gt_labels at step k must be the ground-truth chain shifted left k+1 times,
        # derived purely from the original input_ids (never argmax).
        from angelspec.utils.tensor import padding

        gt = clamped
        for k in range(3):
            self.assertTrue(torch.equal(seen_gt[k], padding(gt, left=False)))
            gt = padding(gt, left=False)

    def test_off_policy_matches_teacher_forcing(self):
        """on_policy=False reproduces the classic left-shifted ground-truth feed."""
        from angelspec.utils.tensor import padding

        input_ids, hidden, target_hidden, target_W, loss_mask = self._run()
        clamped = input_ids.clamp(min=0, max=self.cfg.vocab_size - 1)
        seen = self._spy_embed()
        model = self._model(on_policy=False)
        with torch.no_grad():
            model(
                input_ids=clamped,
                attention_mask=torch.ones(input_ids.shape),
                target_hidden_states=target_hidden,
                target_lm_head_weight=target_W,
                loss_mask=loss_mask,
                hidden_states=hidden,
            )
        self.assertEqual(len(seen), 3)
        expected = clamped
        for k in range(3):
            self.assertTrue(torch.equal(seen[k], expected))
            expected = padding(expected, left=False)


class TestMTPRouter(unittest.TestCase):
    """Lock in hy_v3 routing: sigmoid scores, top-k on (scores + expert_bias),
    weights from UN-biased scores, route_norm renorm, scaling on routed only,
    shared expert added UNSCALED.

    Reference mirrors vllm.../fused_moe grouped_topk for the single-group hy_v3
    case (num_expert_group=1, topk_group=1 → grouping is a no-op).
    """

    def setUp(self):
        torch.manual_seed(7)
        self.cfg = _tiny_config(H=16, V=32, experts=8, top_k=3)
        self.cfg.router_scaling_factor = 2.5
        self.cfg.route_norm = True
        self.moe = Hy3MoE(self.cfg).to(torch.float32).eval()
        # randomise bias so the bias-vs-weight distinction is exercised
        with torch.no_grad():
            self.moe.expert_bias.copy_(torch.randn(self.cfg.num_experts))

    def _reference(self, x):
        cfg = self.cfg
        # The gate is kept in fp32 (vLLM GateLinear semantics); mirror that.
        logits = self.moe.router.gate(x.float())
        scores = logits.sigmoid()
        scored = scores + self.moe.expert_bias.float().unsqueeze(0)
        topk_idx = torch.topk(scored, cfg.num_experts_per_tok, dim=-1).indices
        w = torch.gather(scores, 1, topk_idx)  # un-biased scores
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        w = (w * cfg.router_scaling_factor).to(x.dtype)
        out = torch.zeros_like(x)
        # Compute each routed expert from the FUSED [E, in, out] weights.
        act = self.moe.act_fn
        for t in range(x.shape[0]):
            for s in range(cfg.num_experts_per_tok):
                e = int(topk_idx[t, s])
                xt = x[t : t + 1]
                g = xt @ self.moe.experts_gate_proj[e]
                u = xt @ self.moe.experts_up_proj[e]
                ye = (act(g) * u) @ self.moe.experts_down_proj[e]
                out[t] += ye.squeeze(0) * w[t, s]
        out = out + self.moe.shared_mlp(x)
        return out, topk_idx

    def test_routed_output_matches_reference(self):
        x = torch.randn(1, 5, self.cfg.hidden_size)
        got = self.moe(x).squeeze(0)
        ref, _ = self._reference(x.squeeze(0))
        self.assertTrue(torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max().item())

    def test_topk_uses_biased_scores_but_weights_unbiased(self):
        # Construct a case where bias changes WHICH experts are picked: a large
        # negative bias on the otherwise-top expert should drop it from top-k,
        # while the surviving weights still come from un-biased scores.
        x = torch.randn(1, 4, self.cfg.hidden_size)
        _, topk_idx = self._reference(x.squeeze(0))
        # bias all but expert 0 strongly positive so they dominate selection
        with torch.no_grad():
            b = torch.full((self.cfg.num_experts,), 5.0)
            b[0] = -5.0
            self.moe.expert_bias.copy_(b)
        got = self.moe(x).squeeze(0)
        ref, ref_idx = self._reference(x.squeeze(0))
        self.assertTrue(torch.allclose(got, ref, atol=1e-5))
        # expert 0 should now rarely (ideally never) be selected
        self.assertEqual(int((ref_idx == 0).sum()), 0)

    def test_scaling_applies_to_routed_not_shared(self):
        x = torch.randn(1, 3, self.cfg.hidden_size)
        # With scaling=1 vs scaling=k, the DIFFERENCE must equal (k-1)*routed,
        # i.e. shared expert output is unaffected by scaling.
        self.cfg.router_scaling_factor = 1.0
        self.moe.router_scaling_factor = 1.0
        out1 = self.moe(x)
        self.moe.router_scaling_factor = 3.0
        out3 = self.moe(x)
        shared = self.moe.shared_mlp(x.squeeze(0))
        routed1 = out1.squeeze(0) - shared
        routed3 = out3.squeeze(0) - shared
        self.assertTrue(torch.allclose(routed3, routed1 * 3.0, atol=1e-5))

    def test_grouped_matches_reference_path(self):
        # The default grouped (_grouped_mm) path and the ANGELSPEC_MTP_MOE_REFERENCE
        # per-expert loop must produce identical routed outputs (same routing,
        # same expert math, just batched vs serial).
        import os

        x = torch.randn(2, 6, self.cfg.hidden_size)
        grouped = self.moe(x)  # default path
        os.environ["ANGELSPEC_MTP_MOE_REFERENCE"] = "1"
        try:
            reference = self.moe(x)
        finally:
            os.environ.pop("ANGELSPEC_MTP_MOE_REFERENCE", None)
        self.assertTrue(
            torch.allclose(grouped, reference, atol=1e-5),
            (grouped - reference).abs().max().item(),
        )

    def test_grouped_path_backward(self):
        # _grouped_mm must propagate gradients to the fused expert weights.
        x = torch.randn(2, 5, self.cfg.hidden_size, requires_grad=True)
        out = self.moe(x)
        out.pow(2).sum().backward()
        self.assertIsNotNone(self.moe.experts_gate_proj.grad)
        self.assertIsNotNone(self.moe.experts_down_proj.grad)
        self.assertTrue(torch.isfinite(self.moe.experts_gate_proj.grad).all())
        self.assertIsNotNone(x.grad)


class TestMTPCPTRemap(unittest.TestCase):
    """Verify the CPT key remap matches MTPDraftModel's submodule names.

    We synthesise a checkpoint-style state dict for model.layers.<N>.* using the
    model's own structure, then check the trainer's remap loads it strictly into
    the MTP submodules (embed/lm_head excluded).
    """

    def test_remap_loads_into_draft(self):
        cfg = _tiny_config()
        model = MTPDraftModel(cfg).to(torch.float32)

        # Build a fake target-layer state dict with the checkpoint naming:
        #   model.layers.<N>.{enorm,hnorm,eh_proj,final_layernorm}
        #   model.layers.<N>.{input_layernorm,post_attention_layernorm}
        #   model.layers.<N>.self_attn.*
        #   model.layers.<N>.mlp.{router.gate,expert_bias,shared_mlp.*}
        #   model.layers.<N>.mlp.experts.<e>.{gate,up,down}_proj.weight  (per-expert)
        N = 80
        prefix = f"model.layers.{N}."
        top_level = {"enorm", "hnorm", "eh_proj", "final_layernorm"}
        E, H, inter = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        ckpt = {}
        # non-expert params straight from the model structure
        for name, p in model.named_parameters():
            if name.startswith("embed_tokens") or name.startswith("lm_head"):
                continue
            if name.startswith("midlayer.mlp.experts_"):
                continue  # fused — emit per-expert keys below instead
            if name.startswith("midlayer."):
                ckpt[prefix + name[len("midlayer.") :]] = torch.randn_like(p)
            else:
                top = name.split(".")[0]
                if top in top_level:
                    ckpt[prefix + name] = torch.randn_like(p)
        # per-expert checkpoint weights ([out, in] Linear layout)
        for e in range(E):
            ckpt[f"{prefix}mlp.experts.{e}.gate_proj.weight"] = torch.randn(inter, H)
            ckpt[f"{prefix}mlp.experts.{e}.up_proj.weight"] = torch.randn(inter, H)
            ckpt[f"{prefix}mlp.experts.{e}.down_proj.weight"] = torch.randn(H, inter)

        # Apply the same remap logic the trainer uses (without dist/cuda).
        import re

        expert_pat = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
        expert_slices = {"gate_proj": {}, "up_proj": {}, "down_proj": {}}
        remapped = {}
        for k, v in ckpt.items():
            sub = k[len(prefix) :]
            m = expert_pat.match(sub)
            if m is not None:
                expert_slices[m.group(2)][int(m.group(1))] = v
                continue
            top = sub.split(".")[0]
            if top in top_level:
                remapped[sub] = v
            else:
                remapped[f"midlayer.{sub}"] = v
        for proj, fused in (
            ("gate_proj", "experts_gate_proj"),
            ("up_proj", "experts_up_proj"),
            ("down_proj", "experts_down_proj"),
        ):
            slices = expert_slices[proj]
            n_e = max(slices) + 1
            remapped[f"midlayer.mlp.{fused}"] = torch.stack(
                [slices[e].t().contiguous() for e in range(n_e)], dim=0
            )

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        real_missing = [
            m for m in missing if not (m.startswith("embed_tokens") or m.startswith("lm_head"))
        ]
        self.assertEqual(real_missing, [])
        self.assertEqual(list(unexpected), [])
        # eh_proj loaded; fused experts loaded with correct [E, in, out] layout.
        self.assertTrue(torch.equal(model.eh_proj.weight, remapped["eh_proj.weight"]))
        moe = model.midlayer.mlp
        self.assertEqual(tuple(moe.experts_gate_proj.shape), (E, H, inter))
        self.assertEqual(tuple(moe.experts_down_proj.shape), (E, inter, H))
        # spot-check expert 0 gate slice equals the transposed checkpoint weight
        self.assertTrue(
            torch.equal(
                moe.experts_gate_proj[0],
                ckpt[f"{prefix}mlp.experts.0.gate_proj.weight"].t(),
            )
        )


class TestMTPServeAlignment(unittest.TestCase):
    """The train/serve off-by-one fix: align_mtp_inputs must reproduce the vLLM
    serve MTP convention — at position j the draft sees embed(token[j+1]) paired
    with the UN-shifted hidden raw_h[j], while the teacher hidden stays shifted
    (raw_h[j+1]) so its argmax matches the left-shifted ground-truth label.
    """

    def _raw(self):
        # Distinct per-position hidden so shifts are detectable: raw_h[j] = j+1.
        B, S, H = 1, 6, 4
        raw_ids = torch.arange(S).view(B, S)  # token[j] = j
        raw_h = (torch.arange(S).float() + 1.0).view(B, S, 1).expand(B, S, H).contiguous()
        raw_mask = torch.ones(B, S)  # default all-supervised
        return raw_ids, raw_h, raw_mask

    def test_draft_hidden_unshifted_teacher_shifted(self):
        from angelspec.training.mtp_trainer import align_mtp_inputs

        raw_ids, raw_h, raw_mask = self._raw()
        input_ids, draft_input, target_hidden, _ = align_mtp_inputs(
            raw_ids, raw_h, raw_mask, draft_input_postnorm=False, verifier_norm=None
        )
        S = raw_ids.shape[1]

        # embed source is left-shifted: input_ids[j] == token[j+1] (last is padded 0).
        self.assertTrue(torch.equal(input_ids[0, : S - 1], torch.arange(1, S)))
        self.assertEqual(int(input_ids[0, -1]), 0)

        # draft hidden is UN-shifted: draft_input[j] == raw_h[j] == j+1 (the fix).
        self.assertTrue(torch.equal(draft_input[0, :, 0], torch.arange(S).float() + 1.0))

        # teacher hidden is left-shifted: target_hidden[j] == raw_h[j+1] == j+2
        # (last position padded 0).
        self.assertTrue(torch.equal(target_hidden[0, : S - 1, 0], torch.arange(2, S + 1).float()))
        self.assertEqual(float(target_hidden[0, -1, 0]), 0.0)

        # The whole point: at position j, draft hidden (raw_h[j]) is exactly ONE
        # token BEHIND the teacher hidden (raw_h[j+1]) — different shifts.
        self.assertTrue(torch.equal(draft_input[0, 1:, 0], target_hidden[0, :-1, 0]))

    def test_loss_mask_shifted_twice_to_predicted_token(self):
        # loss_mask must gate on the PREDICTED token (token[j+2] at step 0), i.e.
        # left-shifted twice vs the raw input-token-viewed mask. Verifies the
        # second off-by-N fix.
        from angelspec.training.mtp_trainer import align_mtp_inputs
        from angelspec.utils.tensor import padding

        S = 10
        raw_ids = torch.arange(S).view(1, S)
        raw_h = torch.zeros(1, S, 4)
        # response = tokens 3..7
        raw_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]).float()
        _, _, _, loss_mask = align_mtp_inputs(
            raw_ids, raw_h, raw_mask, draft_input_postnorm=False, verifier_norm=None
        )
        # step 0 position j predicts token[j+2]; loss_mask[j] must equal raw_mask[j+2].
        expected = padding(padding(raw_mask, left=False), left=False)
        self.assertTrue(torch.equal(loss_mask, expected))
        # explicit: position 1 predicts token 3 (response) -> supervised.
        self.assertEqual(int(loss_mask[0, 1]), 1)
        # position 6 predicts token 8 (not response) -> not supervised.
        self.assertEqual(int(loss_mask[0, 6]), 0)

    def test_postnorm_applies_norm_to_unshifted_draft(self):
        from angelspec.training.mtp_trainer import align_mtp_inputs

        raw_ids, raw_h, raw_mask = self._raw()
        # verifier_norm doubles its input so we can detect it ran + on which tensor.
        norm = lambda x: x * 2.0  # noqa: E731
        _, draft_input, target_hidden, _ = align_mtp_inputs(
            raw_ids, raw_h, raw_mask, draft_input_postnorm=True, verifier_norm=norm
        )
        S = raw_ids.shape[1]
        # draft hidden = norm(UN-shifted raw_h): 2*(j+1).
        self.assertTrue(torch.equal(draft_input[0, :, 0], (torch.arange(S).float() + 1.0) * 2.0))
        # teacher hidden = norm(shifted raw_h): 2*(j+2) for j<S-1.
        self.assertTrue(
            torch.equal(target_hidden[0, : S - 1, 0], (torch.arange(2, S + 1).float()) * 2.0)
        )

    def test_off_policy_default_no_norm_matches_legacy_teacher(self):
        # With no verifier_norm and postnorm off, the teacher hidden is just the
        # left-shifted raw hidden (unchanged from legacy), while draft is the fix.
        from angelspec.training.mtp_trainer import align_mtp_inputs
        from angelspec.utils.tensor import padding

        raw_ids, raw_h, raw_mask = self._raw()
        _, draft_input, target_hidden, _ = align_mtp_inputs(
            raw_ids, raw_h, raw_mask, draft_input_postnorm=False, verifier_norm=None
        )
        # teacher == legacy left-shifted hidden; draft == raw (un-shifted).
        self.assertTrue(torch.equal(target_hidden, padding(raw_h, left=False)))
        self.assertTrue(torch.equal(draft_input, raw_h))


class TestMTPFlashCachedMerge(unittest.TestCase):
    """The `fa` cached-attention merge must (a) match the eager sdpa cache path
    numerically (fwd + grad) and (b) pass fp64 gradcheck on its merge algebra.

    Runs on CPU (use_flex=False → dense block-0 fallback; flex has no CPU backward),
    exercising the SAME plain-autograd merge (block-0 + closed-form chain LSE merge)
    the GPU flex path uses — the algebra is nailed here in fp64. The GPU flex block-0
    kernel is validated separately by the on-hardware fa-vs-sdpa parity check.
    """

    def _eager_cache_attn(self, q, cache_k, cache_v, add_mask, scale):
        """Reproduce MTPAttention's eager cache-branch attention exactly.

        q: [b,H,s,d]; cache_*: [b,H,nc,s,d]; add_mask: [b,1,s,s] additive causal.
        """
        b, H, s, d = q.shape
        nc = cache_k.shape[2]
        k0 = cache_k[:, :, 0]
        v0 = cache_v[:, :, 0]
        aw = torch.matmul(q, k0.transpose(2, 3)) * scale + add_mask
        for i in range(1, nc):
            ki = cache_k[:, :, i]
            awi = (q * ki).sum(-1) * scale
            aw = torch.cat((aw, awi[..., None]), dim=-1)
        aw = torch.softmax(aw, dim=-1)
        out = torch.matmul(aw[..., :s], v0)
        for i in range(1, nc):
            vi = cache_v[:, :, i]
            out = out + aw[..., s + i - 1][..., None] * vi
        return out

    def _rand_inputs(self, b=2, H=3, s=5, d=4, nc=2, dtype=torch.float64):
        torch.manual_seed(11)
        q = torch.randn(b, H, s, d, dtype=dtype)
        cache_k = torch.randn(b, H, nc, s, d, dtype=dtype)
        cache_v = torch.randn(b, H, nc, s, d, dtype=dtype)
        return q, cache_k, cache_v

    def test_forward_matches_eager(self):
        from angelspec.models.draft.mtp import _mtp_cached_merge

        q, ck, cv = self._rand_inputs()
        b, H, s, d = q.shape
        scale = 1.0 / (d**0.5)
        valid = torch.ones(b, s, dtype=torch.bool)
        add_mask = torch.triu(torch.full((s, s), float("-inf"), dtype=q.dtype), diagonal=1)[
            None, None
        ]

        got = _mtp_cached_merge(q, ck, cv, valid, scale, use_flex=False)
        ref = self._eager_cache_attn(q, ck, cv, add_mask, scale)
        self.assertTrue(torch.allclose(got, ref, atol=1e-9), (got - ref).abs().max().item())

    def test_backward_matches_eager(self):
        from angelspec.models.draft.mtp import _mtp_cached_merge

        q, ck, cv = self._rand_inputs()
        b, H, s, d = q.shape
        scale = 1.0 / (d**0.5)
        valid = torch.ones(b, s, dtype=torch.bool)
        add_mask = torch.triu(torch.full((s, s), float("-inf"), dtype=q.dtype), diagonal=1)[
            None, None
        ]
        grad_seed = torch.randn(b, H, s, d, dtype=q.dtype)

        # our merge
        q1 = q.clone().requires_grad_(True)
        ck1 = ck.clone().requires_grad_(True)
        cv1 = cv.clone().requires_grad_(True)
        out1 = _mtp_cached_merge(q1, ck1, cv1, valid, scale, use_flex=False)
        out1.backward(grad_seed)

        # autograd reference through the eager path
        q2 = q.clone().requires_grad_(True)
        ck2 = ck.clone().requires_grad_(True)
        cv2 = cv.clone().requires_grad_(True)
        out2 = self._eager_cache_attn(q2, ck2, cv2, add_mask, scale)
        out2.backward(grad_seed)

        for name, a, ref in (
            ("dq", q1.grad, q2.grad),
            ("dck", ck1.grad, ck2.grad),
            ("dcv", cv1.grad, cv2.grad),
        ):
            self.assertTrue(
                torch.allclose(a, ref, atol=1e-8),
                f"{name} max diff {(a - ref).abs().max().item()}",
            )

    def test_gradcheck_double(self):
        from angelspec.models.draft.mtp import _mtp_cached_merge

        q, ck, cv = self._rand_inputs(b=1, H=2, s=4, d=3, nc=3)
        b, H, s, d = q.shape
        scale = 1.0 / (d**0.5)
        valid = torch.ones(b, s, dtype=torch.bool)
        q = q.requires_grad_(True)
        ck = ck.requires_grad_(True)
        cv = cv.requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, k, v: _mtp_cached_merge(a, k, v, valid, scale, use_flex=False),
                (q, ck, cv),
                atol=1e-6,
                rtol=1e-4,
            )
        )

    def test_fa_backend_matches_sdpa_via_model(self):
        """End-to-end through MTPAttention: fa (dense fallback) == sdpa cache path
        for the SAME weights, forward + grads. Run in fp32 (the sdpa path's
        _make_causal_mask uses finfo.min which overflows fp64's default-fp32
        torch.full); the fp64 gradcheck above already pins the Function algebra."""
        torch.manual_seed(5)
        cfg = _tiny_config()
        B, S, H = 2, 6, cfg.hidden_size
        dt = torch.float32

        draft_sdpa = MTPDraftModel(cfg, attention_backend="sdpa").to(dt)
        draft_fa = MTPDraftModel(cfg, attention_backend="fa").to(dt)
        draft_fa.load_state_dict(draft_sdpa.state_dict())
        W = torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=dt)
        draft_sdpa.set_lm_head_weight(W)
        draft_fa.set_lm_head_weight(W)

        ids = torch.randint(1, cfg.vocab_size, (B, S))
        hs = torch.randn(B, S, H, dtype=dt)
        th = torch.randn(B, S, H, dtype=dt)
        lm = torch.ones(B, S)
        am = torch.ones(B, S)

        def run(draft):
            model = MTPModel(draft, length=2, attention_backend=draft.attention_backend)
            model.train()
            ce, *_ = model(
                input_ids=ids,
                attention_mask=am,
                target_hidden_states=th,
                target_lm_head_weight=W,
                loss_mask=lm,
                hidden_states=hs,
            )
            loss = sum(ce)
            loss.backward()
            return float(loss), draft

        l_sdpa, d_sdpa = run(draft_sdpa)
        l_fa, d_fa = run(draft_fa)
        self.assertAlmostEqual(l_sdpa, l_fa, places=4)
        g_sdpa = dict(d_sdpa.named_parameters())
        for n, p in d_fa.named_parameters():
            if p.grad is None:
                continue
            ref = g_sdpa[n].grad
            if ref is None:
                continue
            self.assertTrue(
                torch.allclose(p.grad, ref, atol=1e-4, rtol=1e-3),
                f"{n} grad max diff {(p.grad - ref).abs().max().item()}",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "flex block-0 needs CUDA")
    def test_fa_flex_matches_sdpa_gpu(self):
        """On GPU (bf16), the fa backend using the REAL compiled flex block-0 must
        match the sdpa eager cache path on loss + grads (loose bf16 tol). Validates
        the flex wiring (out0+lse0) and merge on hardware. head_dim must be 128
        (flex-friendly); tiny head_dim (e.g. 8/16) hits NoValidChoicesError."""
        torch.manual_seed(5)
        cfg = _tiny_config()
        cfg.head_dim = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.hidden_size = cfg.head_dim * cfg.num_attention_heads
        cfg.target_hidden_size = cfg.hidden_size
        dev, dt = "cuda", torch.bfloat16
        B, S, H = 2, 130, cfg.hidden_size

        draft_sdpa = MTPDraftModel(cfg, attention_backend="sdpa").to(dev, dt)
        draft_fa = MTPDraftModel(cfg, attention_backend="fa").to(dev, dt)
        draft_fa.load_state_dict(draft_sdpa.state_dict())
        W = torch.randn(cfg.vocab_size, cfg.hidden_size, device=dev, dtype=dt)
        draft_sdpa.set_lm_head_weight(W)
        draft_fa.set_lm_head_weight(W)

        ids = torch.randint(1, cfg.vocab_size, (B, S), device=dev)
        hs = torch.randn(B, S, H, device=dev, dtype=dt)
        th = torch.randn(B, S, H, device=dev, dtype=dt)
        lm = torch.ones(B, S, device=dev)
        am = torch.ones(B, S, device=dev)

        def run(draft):
            model = MTPModel(draft, length=2, attention_backend=draft.attention_backend)
            model.train()
            ce, *_ = model(
                input_ids=ids,
                attention_mask=am,
                target_hidden_states=th,
                target_lm_head_weight=W,
                loss_mask=lm,
                hidden_states=hs,
            )
            loss = sum(ce)
            loss.backward()
            return float(loss), draft

        l_sdpa, d_sdpa = run(draft_sdpa)
        l_fa, d_fa = run(draft_fa)
        self.assertTrue(
            abs(l_sdpa - l_fa) < 0.05 * max(1.0, abs(l_sdpa)),
            f"loss mismatch sdpa={l_sdpa} fa={l_fa}",
        )
        g_sdpa = dict(d_sdpa.named_parameters())
        for n, p in d_fa.named_parameters():
            if p.grad is None or g_sdpa[n].grad is None:
                continue
            ref = g_sdpa[n].grad
            denom = ref.abs().max().clamp_min(1e-3)
            self.assertTrue(
                (p.grad - ref).abs().max() < 0.05 * denom,
                f"{n} grad rel diff {((p.grad - ref).abs().max() / denom).item()}",
            )


class TestMTPUlyssesAllToAll(unittest.TestCase):
    """Ulysses sequence-parallel all-to-all (Step B, torch-native, no yunchang).

    The layout math (_mtp_all_to_all_4d) must be an exact seq<->head reshuffle,
    and running it across N gloo ranks then merging must reproduce the SAME
    attention output a single card computes over the full sequence with the `fa`
    backend. CPU + gloo, no GPU/flex needed.
    """

    def test_all_to_all_roundtrip_single_process(self):
        # world=1 is identity; verify the reshape/permute is a no-op then.
        from angelspec.models.draft.mtp import _mtp_all_to_all_4d

        class _FakeGroup:
            pass

        import torch.distributed as dist

        orig = dist.get_world_size
        dist.get_world_size = lambda group=None: 1
        try:
            x = torch.randn(2, 8, 5, 4)
            y = _mtp_all_to_all_4d(x, _FakeGroup(), 1, 2)
            self.assertTrue(torch.equal(x, y))
        finally:
            dist.get_world_size = orig

    @staticmethod
    def _e2e_global_count_worker(rank, world, tmpdir, ret):
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmpdir}/pg", world_size=world, rank=rank
        )

        model = object.__new__(MTPModel)
        torch.nn.Module.__init__(model)
        model.e2e_weighting = True
        model.e2e_direct = True
        model._usp_size = world
        model._usp_group = dist.group.WORLD

        gamma, n = 2, 6
        shard = n // world
        start, end = rank * shard, (rank + 1) * shard
        alpha_full = torch.tensor(
            [
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                [0.85, 0.75, 0.65, 0.55, 0.45, 0.35],
            ],
            dtype=torch.double,
        )
        # world=2 gives rank1 an empty valid-chain shard.
        valid_full = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.double)
        alpha_local = alpha_full[:, start:end].clone().requires_grad_(True)
        ell_local = torch.ones_like(alpha_local, requires_grad=True)
        masks = [valid_full[start:end].clone() for _ in range(gamma)]

        loss = model._e2e_loss(
            [ell_local[i] for i in range(gamma)],
            [alpha_local[i] for i in range(gamma)],
            masks,
        )
        loss.backward()

        grad_global = torch.zeros_like(alpha_full)
        grad_global[:, start:end] = alpha_local.grad
        ret[rank] = (float(loss), grad_global.reshape(-1).tolist())
        dist.barrier()
        dist.destroy_process_group()

    def test_e2e_usp_global_count_parity_gloo(self):
        import tempfile

        import torch.multiprocessing as mp

        def run(world):
            with tempfile.TemporaryDirectory() as tmp:
                mgr = mp.Manager()
                ret = mgr.dict()
                mp.spawn(
                    self._e2e_global_count_worker,
                    args=(world, tmp, ret),
                    nprocs=world,
                    join=True,
                )
                loss = sum(ret[r][0] for r in range(world))
                grad = torch.tensor(
                    [sum(ret[r][1][i] for r in range(world)) for i in range(len(ret[0][1]))],
                    dtype=torch.double,
                )
                return loss, grad

        ref_loss, ref_grad = run(1)
        usp_loss, usp_grad = run(2)
        self.assertAlmostEqual(ref_loss, usp_loss, places=12)
        self.assertTrue(
            torch.allclose(ref_grad, usp_grad, rtol=1e-10, atol=1e-12),
            f"max e2e grad diff={(ref_grad - usp_grad).abs().max().item()}",
        )

    @staticmethod
    def _sp_sum_dp_avg_worker(rank, world, tmpdir, ret):
        import torch.distributed as dist

        from angelspec.utils.usp import usp_dp_average_factor

        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmpdir}/pg", world_size=world, rank=rank
        )
        # Two DP replicas, each split over two SP ranks:
        # DP0 full grad = 1 + 3 = 4; DP1 full grad = 10 + 14 = 24.
        local = torch.tensor([1.0, 3.0, 10.0, 14.0], dtype=torch.double)[rank]
        reduced = local.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= usp_dp_average_factor(world, sp_size=2)
        ret[rank] = float(reduced)
        dist.barrier()
        dist.destroy_process_group()

    def test_usp_grad_reduction_is_sp_sum_dp_avg_gloo(self):
        import tempfile

        import torch.multiprocessing as mp

        world = 4
        with tempfile.TemporaryDirectory() as tmp:
            mgr = mp.Manager()
            ret = mgr.dict()
            mp.spawn(
                self._sp_sum_dp_avg_worker,
                args=(world, tmp, ret),
                nprocs=world,
                join=True,
            )
            # Average of the two reconstructed DP gradients: (4 + 24) / 2 = 14.
            for rank in range(world):
                self.assertAlmostEqual(ret[rank], 14.0, places=12)

    @staticmethod
    def _worker(rank, world, tmpdir, seed, B, H, KV, D, S, ret):
        import torch.distributed as dist

        from angelspec.models.draft.mtp import MTPAttention, MTPConfig

        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmpdir}/pg", world_size=world, rank=rank
        )
        from angelspec.utils import distributed as dmod

        # Pure-Ulysses group == the whole world.
        dmod._SP_ULYSSES_GROUP = dist.group.WORLD

        torch.manual_seed(seed)
        cfg = MTPConfig(
            hidden_size=H * D,
            intermediate_size=64,
            num_attention_heads=H,
            num_key_value_heads=KV,
            head_dim=D,
            qk_norm=True,
            vocab_size=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=512,
            rope_theta=10000.0,
            num_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            route_norm=True,
            router_scaling_factor=1.5,
            target_hidden_size=H * D,
            tie_lm_head=True,
        )
        # Same weights on every rank (deterministic seed).
        attn_usp = MTPAttention(cfg, attention_backend="usp")
        attn_fa = MTPAttention(cfg, attention_backend="fa")
        attn_fa.load_state_dict(attn_usp.state_dict())
        attn_usp = attn_usp.double()
        attn_fa = attn_fa.double()

        torch.manual_seed(1234)
        full_hs = torch.randn(B, S, H * D, dtype=torch.double)
        pos_full = torch.arange(S).unsqueeze(0)

        # Single-card fa reference: full sequence, one TTT step (use_cache path).
        ref, _, _ = attn_fa(
            full_hs,
            cache_keys=None,
            cache_values=None,
            attention_mask=torch.ones(B, S, dtype=torch.bool),
            position_ids=pos_full,
            use_cache=True,
        )

        # USP: this rank sees only its seq shard; global positions; the module
        # all-to-alls internally to attend the full sequence.
        shard = S // world
        sl = slice(rank * shard, (rank + 1) * shard)
        local_hs = full_hs[:, sl, :].contiguous()
        pos_local = torch.arange(rank * shard, (rank + 1) * shard).unsqueeze(0)
        out, _, _ = attn_usp(
            local_hs,
            cache_keys=None,
            cache_values=None,
            attention_mask=torch.ones(B, shard, dtype=torch.bool),
            position_ids=pos_local,
            use_cache=True,
        )
        # Compare this rank's shard of the reference.
        ref_shard = ref[:, sl, :]
        max_diff = (out - ref_shard).abs().max().item()
        ret[rank] = max_diff
        dist.barrier()
        dist.destroy_process_group()

    def test_usp_attention_matches_fa_gloo(self):
        import tempfile

        import torch.multiprocessing as mp

        world = 2
        B, H, KV, D = 1, 4, 2, 8  # H divisible by world; small head_dim -> dense block-0 on CPU
        S = 8
        with tempfile.TemporaryDirectory() as tmp:
            mgr = mp.Manager()
            ret = mgr.dict()
            mp.spawn(
                self._worker,
                args=(world, tmp, 7, B, H, KV, D, S, ret),
                nprocs=world,
                join=True,
            )
            for r in range(world):
                self.assertIn(r, ret, f"rank {r} did not report")
                self.assertLess(ret[r], 1e-6, f"rank {r} usp vs fa max diff {ret[r]}")

    @staticmethod
    def _model_worker(rank, world, tmpdir, ret):
        import torch.distributed as dist

        from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel
        from angelspec.models.mtp import MTPModel

        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmpdir}/pg", world_size=world, rank=rank
        )
        from angelspec.utils import distributed as dmod

        dmod._SP_ULYSSES_GROUP = dist.group.WORLD
        dmod._DRAFT_SP_GROUP = dist.group.WORLD

        torch.manual_seed(3)
        H, D, KV = 4, 8, 2
        cfg = MTPConfig(
            hidden_size=H * D,
            intermediate_size=64,
            num_attention_heads=H,
            num_key_value_heads=KV,
            head_dim=D,
            qk_norm=True,
            vocab_size=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=512,
            rope_theta=10000.0,
            num_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            route_norm=True,
            router_scaling_factor=1.5,
            target_hidden_size=H * D,
            tie_lm_head=True,
        )
        # fp32 (not fp64): the MoE path uses torch._grouped_mm which only supports
        # fp16/bf16/fp32. CPU flex has no backward anyway, so _mtp_cached_merge takes
        # its dense block-0 fallback — the USP all-to-all + merge algebra is what we're
        # checking here, already pinned in fp64 by the attention-only test above.
        draft = MTPDraftModel(cfg, attention_backend="usp").float()
        W = torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32)
        draft.set_lm_head_weight(W)
        model = MTPModel(draft, length=2, attention_backend="usp")
        model.train()

        B, S = 1, 8
        torch.manual_seed(1234)  # identical FULL sample on every rank
        ids = torch.randint(1, cfg.vocab_size, (B, S))
        hs = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)
        th = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)
        lm = torch.ones(B, S)
        am = torch.ones(B, S)
        ce, *_ = model(
            input_ids=ids,
            attention_mask=am,
            target_hidden_states=th,
            target_lm_head_weight=W,
            loss_mask=lm,
            hidden_states=hs,
        )
        loss = sum(ce)
        loss.backward()
        # Each rank's ce is (LOCAL ce_sum / GLOBAL token count), so per-rank losses
        # differ; their SUM is the global mean. Report loss + a grad-norm so the
        # driver can check finiteness and that every param got a gradient (the
        # zero-token-shard connect-graph trick must keep FSDP's reduction fed).
        gnorm = sum(
            (p.grad.double() ** 2).sum() for p in model.parameters() if p.grad is not None
        ).sqrt()
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        ret[rank] = (float(loss), float(gnorm), int(n_grad), int(n_trainable))
        dist.barrier()
        dist.destroy_process_group()

    def test_usp_mtpmodel_loss_consistent_gloo(self):
        """MTPModel.forward under usp: seq slicing + global-count loss all-reduce
        runs a full TTT loop + backward across ranks without collective hangs, with
        finite loss/grads. Most trainable params get a gradient (a few toy MoE
        experts may be unrouted with only 8 tokens — not a bug)."""
        import tempfile

        import torch.multiprocessing as mp

        world = 2
        with tempfile.TemporaryDirectory() as tmp:
            mgr = mp.Manager()
            ret = mgr.dict()
            mp.spawn(self._model_worker, args=(world, tmp, ret), nprocs=world, join=True)
            self.assertEqual(set(ret.keys()), set(range(world)))
            for r in range(world):
                loss, gnorm, n_grad, n_trainable = ret[r]
                self.assertTrue(loss == loss and abs(loss) < 1e6, f"rank{r} bad loss {loss}")
                self.assertTrue(gnorm == gnorm and 0 < gnorm < 1e6, f"rank{r} bad gnorm {gnorm}")
                # backward reached the bulk of the head (attn + fusion + router +
                # most experts); a handful of unrouted toy experts is acceptable.
                self.assertGreater(
                    n_grad,
                    0.6 * n_trainable,
                    f"rank{r} only {n_grad}/{n_trainable} params got grad",
                )

    @staticmethod
    def _packing_parity_worker(rank, world, tmpdir, ret, doc_lens):
        """One USP rank over a PACKED multi-doc sample. Returns the per-step LOSS
        value ce[k] (= local ce_sum_k / GLOBAL token count under USP). Summed across
        ranks this reconstructs the single-card global-mean loss — the property that
        makes the post-backward grad SUM equal the full-sequence gradient. (A LOCAL-
        count normalization would instead make Σ_r ce[k] = Σ_r ce_sum_r/c_local_r,
        which does NOT equal the single-card mean → this test would fail.)"""
        import torch.distributed as dist

        from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel
        from angelspec.models.mtp import MTPModel

        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmpdir}/pg", world_size=world, rank=rank
        )
        from angelspec.utils import distributed as dmod

        dmod._SP_ULYSSES_GROUP = dist.group.WORLD
        dmod._DRAFT_SP_GROUP = dist.group.WORLD

        torch.manual_seed(3)  # identical weights across every world size
        H, D, KV = 4, 8, 2
        cfg = MTPConfig(
            hidden_size=H * D,
            intermediate_size=64,
            num_attention_heads=H,
            num_key_value_heads=KV,
            head_dim=D,
            qk_norm=True,
            vocab_size=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=512,
            rope_theta=10000.0,
            num_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            route_norm=True,
            router_scaling_factor=1.5,
            target_hidden_size=H * D,
            tie_lm_head=True,
        )
        draft = MTPDraftModel(cfg, attention_backend="usp").float()
        W = torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32)
        draft.set_lm_head_weight(W)
        model = MTPModel(draft, length=2, attention_backend="usp")
        model.train()

        B, S = 1, sum(doc_lens)
        torch.manual_seed(1234)  # identical full sample across every world size
        ids = torch.randint(1, cfg.vocab_size, (B, S))
        hs = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)
        th = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)
        lm = torch.ones(B, S)
        am = torch.ones(B, S)
        ctx_doc_ids = torch.cat(
            [torch.full((1, L), d, dtype=torch.long) for d, L in enumerate(doc_lens)], dim=1
        )
        base_pos = torch.cat(
            [torch.arange(L, dtype=torch.long).unsqueeze(0) for L in doc_lens], dim=1
        )
        ce, kl, vloss, acc, cnt, *_ = model(
            input_ids=ids,
            attention_mask=am,
            target_hidden_states=th,
            target_lm_head_weight=W,
            loss_mask=lm,
            hidden_states=hs,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_pos,
        )
        # Per-step loss value (global-count normalized under USP). Σ_r ce[k] == single-card ce[k].
        ret[rank] = [float(ce[k]) for k in range(len(ce))]
        dist.barrier()
        dist.destroy_process_group()

    def _run_packing_parity(self, world, doc_lens):
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            mgr = mp.Manager()
            ret = mgr.dict()
            mp.spawn(
                self._packing_parity_worker,
                args=(world, tmp, ret, doc_lens),
                nprocs=world,
                join=True,
            )
            nsteps = len(ret[0])
            return [sum(ret[r][k] for r in range(world)) for k in range(nsteps)]

    def test_usp_packing_loss_parity_gloo(self):
        """Packed multi-doc sample, N-card USP vs 1-card, per-step LOSS value summed
        across ranks. Under global-count normalization Σ_r ce[k] == single-card ce[k].
        Doc boundaries ALIGNED to the SP shard cut → exact parity (sharding + doc-gate
        + doc-local RoPE math is shard-invariant). Docs that STRADDLE a shard cut must
        also have exact parity: TTT labels, masks, and teacher-hidden windows are shifted
        on the full sequence before slicing. This locks both cross-shard target handling
        and USP global-count loss normalization."""
        # S=8, world=2 → shard=4. Aligned docs [4,4] land exactly on the cut.
        ref = self._run_packing_parity(1, [4, 4])
        aligned = self._run_packing_parity(2, [4, 4])
        for k, (a, b) in enumerate(zip(ref, aligned)):
            self.assertAlmostEqual(a, b, places=3, msg=f"aligned step{k}: 1-card={a} 2-card={b}")
        # Straddling: docs [3,5] → doc boundary at 3, shard cut at 4 (mid doc-1).
        ref_s = self._run_packing_parity(1, [3, 5])
        strad = self._run_packing_parity(2, [3, 5])
        resid = sum(abs(a - b) for a, b in zip(ref_s, strad)) / (sum(abs(a) for a in ref_s) + 1e-9)
        print(
            f"[packing×USP] cross-shard cut-doc loss residual (S=8, world=2, docs=[3,5]) = {resid:.4%}"
        )
        self.assertLess(resid, 1e-5, f"cut-doc residual unexpectedly large: {resid}")

    @staticmethod
    def _grad_parity_worker(rank, world, tmpdir, ret):
        """One USP rank: packed doc-gated sample with doc boundaries ALIGNED to the SP
        shard cut (so no cross-shard TTT-shift contamination), forward + backward on
        the loss. Returns the flat per-parameter gradient. Summed across ranks (as the
        trainer's _usp_manual_grad_allreduce does) it must equal the single-card
        gradient — the core guarantee of global-count normalization. LOCAL-count would
        inflate the summed grad by ~world size (rel residual ≈ 100%, not a few %)."""
        import torch.distributed as dist

        from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel
        from angelspec.models.mtp import MTPModel

        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmpdir}/pg", world_size=world, rank=rank
        )
        from angelspec.utils import distributed as dmod

        dmod._SP_ULYSSES_GROUP = dist.group.WORLD
        dmod._DRAFT_SP_GROUP = dist.group.WORLD

        torch.manual_seed(3)  # identical weights across every world size
        H, D, KV = 4, 8, 2
        cfg = MTPConfig(
            hidden_size=H * D,
            intermediate_size=64,
            num_attention_heads=H,
            num_key_value_heads=KV,
            head_dim=D,
            qk_norm=True,
            vocab_size=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=512,
            rope_theta=10000.0,
            num_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            route_norm=True,
            router_scaling_factor=1.5,
            target_hidden_size=H * D,
            tie_lm_head=True,
        )
        draft = MTPDraftModel(cfg, attention_backend="usp").float()
        W = torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32)
        draft.set_lm_head_weight(W)
        model = MTPModel(draft, length=2, attention_backend="usp")
        model.train()

        # Two docs [4,4]; world=2 shard cut at 4 lands exactly on the doc boundary →
        # no straddle, so the ONLY grad difference from single-card is the (fixed)
        # normalization, not boundary contamination.
        doc_lens = [4, 4]
        B, S = 1, sum(doc_lens)
        torch.manual_seed(1234)  # identical full sample across every world size
        ids = torch.randint(1, cfg.vocab_size, (B, S))
        hs = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)
        th = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)
        lm = torch.ones(B, S)
        am = torch.ones(B, S)
        ctx_doc_ids = torch.cat(
            [torch.full((1, L), d, dtype=torch.long) for d, L in enumerate(doc_lens)], dim=1
        )
        base_pos = torch.cat(
            [torch.arange(L, dtype=torch.long).unsqueeze(0) for L in doc_lens], dim=1
        )
        ce, *_ = model(
            input_ids=ids,
            attention_mask=am,
            target_hidden_states=th,
            target_lm_head_weight=W,
            loss_mask=lm,
            hidden_states=hs,
            ctx_doc_ids=ctx_doc_ids,
            base_position_ids=base_pos,
        )
        model.zero_grad(set_to_none=True)
        sum(ce).backward()
        flat = [
            float(g)
            for p in model.parameters()
            if p.requires_grad
            for g in (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        ]
        ret[rank] = flat
        dist.barrier()
        dist.destroy_process_group()

    def test_usp_global_count_grad_parity_gloo(self):
        """The fixed quantity, directly: N-card USP per-shard grads SUMMED == 1-card
        grad (doc boundaries aligned to the shard cut). LOCAL-count normalization would
        make the summed grad ~world× too large (rel ≈ 100%)."""
        import tempfile

        import torch.multiprocessing as mp

        def run(world):
            with tempfile.TemporaryDirectory() as tmp:
                mgr = mp.Manager()
                ret = mgr.dict()
                mp.spawn(self._grad_parity_worker, args=(world, tmp, ret), nprocs=world, join=True)
                n = len(ret[0])
                return [sum(ret[r][i] for r in range(world)) for i in range(n)]

        ref = torch.tensor(run(1))
        summed = torch.tensor(run(2))
        rel = (summed - ref).norm() / (ref.norm() + 1e-9)
        print(f"[USP grad parity] ||Σ_r g_r - g_1card|| / ||g_1card|| = {rel:.4%}")
        # Aligned docs → summed grad matches single-card to fp32/all-to-all rounding,
        # NOT the ~world× (rel≈100%) a local-count normalization would produce.
        self.assertLess(rel, 0.02, f"summed USP grad deviates from single-card: rel={rel}")


if __name__ == "__main__":
    unittest.main()
