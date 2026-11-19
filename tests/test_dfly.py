"""Tests for the dfly (DFlareV2) draft model.

dfly is a DFlash-family drafter (its own ``DFlyConfig`` / ``"Qwen3DFlyModel"`` →
``DFlyTrainer`` / ``DFlyModel``) so the hidden-states correction runs through the
wrapper hook. It is a ``DFlareDraftModel`` whose layers are restored to DFlash
shared-KV layers, with the DFlash FC ``context_proj`` re-added and the DFlare
per-layer fusion applied as a residual. It has no dependency on DSpark.

Covers:
1. auto/dispatch: ``DFlyConfig`` → ``DFlyDraftModel`` (and a plain ``DSparkConfig``
   still → ``DSparkDraftModel`` — no cross-routing).
2. Structure: DFlash shared-KV layers (no separate ``k_proj_target``), the
   re-added ``context_proj``, the inherited ``layer_fusion_weights`` / ``context_norm``.
3. hidden_correction: present, zero-init identity, and NO markov / confidence head.
4. ``target_hidden_size != hidden_size`` raises.
5. Tiny forward through the ``DFlyModel`` wrapper: 6-tuple, finite loss, slot-0
   masked, ``loss_components`` keys; and the correction actually runs (perturbing
   ``down_proj`` changes the loss).
6. state_dict carries the expected backbone/correction keys and none for the
   (absent) markov / confidence heads.
7. ``attention_type="mla"``: hy4's gated MLA over DFly's dual-source keys —
   shapes, mask semantics (a masked context token cannot reach the output; a
   dropped block reads zero), the learnable sink, the DSA top-k gate and its
   indexers (per-layer, or shared across a run of layers via hy4's
   ``indexer_types``), plus forward/backward with DSA on.
"""

import unittest
from unittest import mock

import torch

import angelspec.models.draft.hy4_mla as hy4_mla
from angelspec.models.dfly import DFlyModel
from angelspec.models.draft.auto import AutoDraftModel
from angelspec.models.draft.dflash import DFlashAttention
from angelspec.models.draft.dfly import DFlyConfig, DFlyDraftModel
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.draft.hy4_mla import (
    Hy4DraftMLAAttention,
    keep_mask_to_indices,
    keep_mask_to_ranges,
    materialize_keep_mask,
)
from angelspec.models.ops.sparse_mla import torch_lightning_indexer

H, V, BS = 64, 128, 4


def _base(**kw):
    base = dict(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=2,
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
        # dfly carries no markov / confidence head (matches the shipped config).
        markov_rank=0,
        enable_confidence_head=False,
        confidence_head_with_markov=False,
        block_size=BS,
    )
    base.update(kw)
    return base


def _config(**kw):
    kw.setdefault("enable_hidden_correction", True)
    return DFlyConfig(**_base(**kw))


class TestDflyDispatchAndStructure(unittest.TestCase):
    def test_auto_dispatch(self):
        fly = AutoDraftModel.from_config(_config())
        self.assertIsInstance(fly, DFlyDraftModel)
        # A plain DSpark config still builds the DSpark drafter (no cross-routing).
        ds = AutoDraftModel.from_config(DSparkConfig(**_base()))
        self.assertIsInstance(ds, DSparkDraftModel)
        self.assertNotIsInstance(ds, DFlyDraftModel)

    def test_shared_kv_layers(self):
        # dfly restores DFlash shared-KV layers: k_proj/v_proj present, and NOT
        # DFlare's separate context projections (k_proj_target / v_proj_target).
        fly = DFlyDraftModel(_config())
        self.assertEqual(len(fly.layers), fly.num_layers)
        for layer in fly.layers:
            self.assertTrue(hasattr(layer.self_attn, "k_proj"))
            self.assertTrue(hasattr(layer.self_attn, "v_proj"))
            self.assertFalse(hasattr(layer.self_attn, "k_proj_target"))
            self.assertFalse(hasattr(layer.self_attn, "v_proj_target"))

    def test_context_proj_and_fusion_wired(self):
        cfg = _config()
        fly = DFlyDraftModel(cfg)
        # FC context projection re-added (DFlare.__init__ had deleted it).
        self.assertTrue(hasattr(fly, "context_proj"))
        self.assertEqual(
            tuple(fly.context_proj.weight.shape),
            (H, cfg.num_target_layers * H),
        )
        # DFlare fusion + norm inherited.
        self.assertEqual(
            tuple(fly.layer_fusion_weights.shape),
            (cfg.num_hidden_layers, cfg.num_target_layers),
        )
        self.assertTrue(hasattr(fly, "context_norm"))

    def test_hidden_correction_present_and_identity(self):
        fly = DFlyDraftModel(_config())
        self.assertIsNotNone(fly.hidden_correction)
        self.assertTrue(
            (fly.hidden_correction.down_proj.weight == 0).all(),
            "down_proj must be zero-init (identity at init)",
        )
        # dfly builds no markov / confidence head.
        self.assertIsNone(getattr(fly, "markov_head", None))
        self.assertIsNone(getattr(fly, "confidence_head", None))

    def test_hidden_correction_none_when_disabled(self):
        fly = DFlyDraftModel(_config(enable_hidden_correction=False))
        self.assertIsNone(fly.hidden_correction)
        self.assertFalse(any("hidden_correction" in k for k in fly.state_dict()))

    def test_target_hidden_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            DFlyDraftModel(_config(target_hidden_size=H + 8))


class TestDflyForward(unittest.TestCase):
    def _build_wrapper(self):
        torch.manual_seed(0)
        fly = AutoDraftModel.from_config(_config()).to(torch.float32)
        fly.freeze_embedding()
        m = DFlyModel(
            draft_model=fly,
            block_size=BS,
            num_anchors=6,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
        )
        m.eval()
        return m

    def _inputs(self):
        B, S = 2, 24
        g = torch.Generator().manual_seed(1)
        return dict(
            input_ids=torch.randint(0, V, (B, S), generator=g),
            hidden_states_list=[torch.randn(B, S, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(B, S),
            lm_head_weight=torch.randn(V, H, generator=g),
            last_hidden_states=torch.randn(B, S, H, generator=g),
        )

    def test_forward_six_tuple(self):
        m = self._build_wrapper()
        out = m(**self._inputs())
        self.assertEqual(len(out), 6)
        loss, _, lpp, _, cpp, comps = out
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(lpp.shape[0], BS)
        self.assertEqual(cpp[0].item(), 0.0)  # DFlash convention: slot 0 masked
        # DFly rides the plain DFlash loss (no confidence head).
        self.assertEqual(set(comps), {"ce_loss", "kl_loss", "lk_loss", "l1_loss", "e2e_tv_loss"})

    def test_correction_actually_runs(self):
        # With zero-init down_proj the correction is identity; perturbing it must
        # change the loss, proving the wrapper hook applies hidden_correction.
        m = self._build_wrapper()
        inputs = self._inputs()
        base_loss = m(**inputs)[0].item()
        with torch.no_grad():
            m.draft_model.hidden_correction.down_proj.weight.normal_()
        perturbed_loss = m(**inputs)[0].item()
        self.assertNotAlmostEqual(base_loss, perturbed_loss, places=5)


class TestDflyCheckpointParamNames(unittest.TestCase):
    def test_state_dict_keys(self):
        fly = DFlyDraftModel(_config())
        keys = set(fly.state_dict())
        for expected in (
            "context_proj.weight",
            "layer_fusion_weights",
            "context_norm.weight",
            "hidden_correction.gate_proj.weight",
            "hidden_correction.up_proj.weight",
            "hidden_correction.down_proj.weight",
            "hidden_correction.hidden_norm.weight",
            "hidden_correction.embed_norm.weight",
        ):
            self.assertIn(expected, keys, f"missing key: {expected}")
        # No markov / confidence head params (dfly builds neither).
        self.assertFalse(any("markov_head" in k for k in keys))
        self.assertFalse(any("confidence_head" in k for k in keys))


def _mla_kwargs(**kw):
    """The gated-MLA field defaults, shared by the hy4 and MTP layer bodies."""
    kw.setdefault("attention_type", "mla")
    kw.setdefault("q_lora_rank", 32)
    kw.setdefault("kv_lora_rank", 16)
    kw.setdefault("qk_nope_head_dim", 8)
    kw.setdefault("qk_rope_head_dim", 8)
    kw.setdefault("v_head_dim", 12)
    kw.setdefault("gated_mla", True)
    kw.setdefault("learnable_sink", True)
    return kw


def _mla_config(**kw):
    return _config(**_mla_kwargs(**kw))


def _mla_attn(cfg):
    """A standalone dual-source MLA block (no surrounding layer/iHC)."""
    return Hy4DraftMLAAttention(cfg)


def _shared_indexer_config(**kw):
    """An MLA + DSA config whose ``indexer_types`` shares indexers across layers."""
    kw.setdefault("num_hidden_layers", 4)
    kw.setdefault("index_topk", 4)
    kw.setdefault("index_head_dim", 16)
    kw.setdefault("index_n_heads", 2)
    kw.setdefault("indexer_types", ["full", "shared", "shared", "full"])
    return _mla_config(**kw)


def _dual_source_inputs(cfg, ctx_len=16, draft_len=4, seed=0):
    """``(draft_hidden, context_hidden, draft_pos, ctx_pos)`` for one attention call."""
    g = torch.Generator().manual_seed(seed)
    h = cfg.hidden_size
    draft = torch.randn(1, draft_len, h, generator=g)
    context = torch.randn(1, ctx_len, h, generator=g)
    ctx_pos = torch.arange(ctx_len).unsqueeze(0)
    draft_pos = torch.arange(ctx_len, ctx_len + draft_len).unsqueeze(0)
    return draft, context, draft_pos, ctx_pos


def _causal_block_mask(ctx_len, draft_len, block_size, anchor):
    """A one-block DFly training mask: context < anchor, plus intra-block causal."""
    from torch.nn.attention.flex_attention import create_block_mask

    from angelspec.models.dflash import _create_dflash_mask_mod

    mask_mod = _create_dflash_mask_mod(
        anchor_positions=torch.tensor([[anchor]]),
        block_keep_mask=torch.tensor([[True]]),
        ctx_len=ctx_len,
        block_size=block_size,
        causal=True,
    )
    return create_block_mask(
        mask_mod, B=1, H=None, Q_LEN=draft_len, KV_LEN=ctx_len + draft_len, device="cpu"
    )


class TestDflyHy4MLAConfig(unittest.TestCase):
    """Config plumbing for the hy4 attention path (MLA + DSA + hy4 spellings)."""

    def test_rope_theta_read_from_rope_parameters(self):
        # hy4 nests it; a flat rope_theta, when given, still wins.
        base = _base()
        base.pop("rope_theta")
        nested = {"rope_theta": 1e7, "rope_type": "default"}
        self.assertEqual(DFlyConfig(rope_parameters=nested, **base).rope_theta, 1e7)
        self.assertEqual(
            DFlyConfig(rope_parameters=nested, rope_theta=5e5, **base).rope_theta, 5e5
        )

    def test_rejected_settings(self):
        # DSA scores from the LoRA q residual.
        with self.assertRaises(ValueError):
            _mla_config(q_lora_rank=None, index_topk=4)
        # DSA is an MLA-only feature.
        with self.assertRaises(ValueError):
            _config(index_topk=4)
        # qk_head_dim is derived, so a contradictory value is a config bug.
        with self.assertRaises(ValueError):
            _mla_config(qk_head_dim=999)
        # A schedule shorter than the model leaves layers unaccounted for.
        with self.assertRaises(ValueError):
            _mla_config(
                num_hidden_layers=3,
                index_topk=4,
                index_head_dim=16,
                indexer_types=["full", "full"],
            )
        # "shared" borrows the most recent "full" layer's indexer, so a schedule
        # cannot open with it.
        with self.assertRaises(ValueError):
            _shared_indexer_config(indexer_types=["shared", "full", "full", "full"])
        # Sharing an indexer needs there to be one to share.
        with self.assertRaises(ValueError):
            _shared_indexer_config(index_topk=None)

    def test_gqa_remains_the_default(self):
        fly = DFlyDraftModel(_config())
        self.assertTrue(all(isinstance(la.self_attn, DFlashAttention) for la in fly.layers))
        self.assertFalse(any("kv_a_proj_with_mqa" in k for k in fly.state_dict()))


class TestDflyHy4MLAStructure(unittest.TestCase):
    def test_layers_use_mla_with_expected_shapes(self):
        cfg = _mla_config(num_hidden_layers=2)
        fly = AutoDraftModel.from_config(cfg)
        attn = fly.layers[0].self_attn
        self.assertIsInstance(attn, Hy4DraftMLAAttention)
        heads, qk = cfg.num_attention_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.assertEqual(tuple(attn.q_a_proj.weight.shape), (cfg.q_lora_rank, cfg.hidden_size))
        self.assertEqual(tuple(attn.q_b_proj.weight.shape), (heads * qk, cfg.q_lora_rank))
        self.assertEqual(
            tuple(attn.kv_a_proj_with_mqa.weight.shape),
            (cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
        )
        self.assertEqual(
            tuple(attn.kv_b_proj.weight.shape),
            (heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank),
        )
        self.assertEqual(
            tuple(attn.o_proj.weight.shape), (cfg.hidden_size, heads * cfg.v_head_dim)
        )
        self.assertEqual(tuple(attn.learnable_sink_param.shape), (heads,))
        self.assertEqual(
            tuple(attn.linear_gate.weight.shape), (heads * cfg.v_head_dim, cfg.hidden_size)
        )
        # No DSA unless index_topk asks for it.
        self.assertFalse(attn.use_indexer)

    def test_headwise_gate_is_one_scalar_per_head(self):
        attn = DFlyDraftModel(_mla_config(gating_type="headwise")).layers[0].self_attn
        self.assertEqual(attn.gate_projection_size, 1)

    def test_every_layer_owns_an_indexer_without_a_sharing_schedule(self):
        for indexer_types in (None, ["full", "full"]):
            cfg = _mla_config(
                num_hidden_layers=2,
                index_topk=4,
                index_head_dim=16,
                index_n_heads=2,
                indexer_types=indexer_types,
            )
            fly = DFlyDraftModel(cfg)
            self.assertIsNone(fly.indexers)
            self.assertIsNot(fly.layers[0].self_attn.indexer, fly.layers[1].self_attn.indexer)
            # hy4's own naming, so a pretrained self_attn.indexer still loads.
            self.assertIn("layers.0.self_attn.indexer.wk.weight", fly.state_dict())

    def test_shared_schedule_hands_out_one_indexer_per_run(self):
        cfg = _shared_indexer_config()
        fly = DFlyDraftModel(cfg)
        # ["full", "shared", "shared", "full"] -> two indexers, layers 0-2 on the
        # first and layer 3 on the second.
        self.assertEqual(len(fly.indexers), 2)
        borrowed = [layer.self_attn.dsa_indexer for layer in fly.layers]
        self.assertIs(borrowed[0], fly.indexers[0])
        self.assertIs(borrowed[1], fly.indexers[0])
        self.assertIs(borrowed[2], fly.indexers[0])
        self.assertIs(borrowed[3], fly.indexers[1])

    def test_shared_indexers_are_owned_by_the_drafter_not_a_layer(self):
        # Where they live is what makes them usable: DFlashTrainer shards every
        # decoder layer as its own FSDP unit, so an indexer owned by layer 0 is
        # resharded back to a DTensor before layer 1 borrows it. Owned at the root
        # they belong to the root FSDP unit, which is not resharded mid-step.
        fly = DFlyDraftModel(_shared_indexer_config())
        keys = set(fly.state_dict())
        self.assertIn("indexers.0.wk.weight", keys)
        self.assertFalse(any(".self_attn.indexer." in k for k in keys))
        layer_params = {id(p) for p in fly.layers.parameters()}
        for indexer in fly.indexers:
            for param in indexer.parameters():
                self.assertNotIn(id(param), layer_params)

    def test_no_parameter_is_attached_at_two_paths(self):
        # FSDP's loader (DCP set_model_state_dict) deduplicates shared parameters
        # and then loads with strict=True, so any parameter reachable by two names
        # breaks model init. Guard the whole hy4 MLA + DSA model, not just the
        # indexer that first tripped it — including the shared-indexer schedule,
        # whose borrowers hold the module outside nn.Module's registry for exactly
        # this reason.
        for cfg in (
            _mla_config(num_hidden_layers=3, index_topk=4, index_head_dim=16, index_n_heads=2),
            _shared_indexer_config(),
        ):
            with torch.device("meta"):
                fly = DFlyDraftModel(cfg)
            every_path = [n for n, _ in fly.named_parameters(remove_duplicate=False)]
            deduped = [n for n, _ in fly.named_parameters()]
            self.assertEqual(
                sorted(set(every_path) - set(deduped)), [], "parameter reachable by two names"
            )


class TestDflyHy4MLAAttention(unittest.TestCase):
    """The dual-source MLA math: mask semantics, sink, DSA gate."""

    def test_respects_the_training_mask(self):
        # The mask is materialised from the BlockMask's own mask_mod, so a context
        # position the training mask forbids must not reach the output at all.
        torch.manual_seed(0)
        cfg = _mla_config()
        attn = _mla_attn(cfg)
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg)
        ctx_len, draft_len, anchor = context.shape[1], draft.shape[1], 8
        bm = _causal_block_mask(ctx_len, draft_len, draft_len, anchor)

        base = attn(draft, context, draft_pos, ctx_pos, block_mask=bm)
        perturbed_ctx = context.clone()
        perturbed_ctx[:, anchor + 2] += 5.0  # beyond the anchor => masked out
        after = attn(draft, perturbed_ctx, draft_pos, ctx_pos, block_mask=bm)
        torch.testing.assert_close(base, after)

        visible = context.clone()
        visible[:, anchor - 1] += 5.0  # before the anchor => visible
        self.assertFalse(
            torch.allclose(base, attn(draft, visible, draft_pos, ctx_pos, block_mask=bm))
        )

    def test_dropped_block_gets_zero_output(self):
        # block_keep_mask=False masks a whole query row; FlexAttention returns 0
        # for those, and the additive-mask path must agree instead of emitting NaN.
        from torch.nn.attention.flex_attention import create_block_mask

        from angelspec.models.dflash import _create_dflash_mask_mod

        torch.manual_seed(0)
        cfg = _mla_config()
        attn = _mla_attn(cfg)
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg)
        ctx_len, draft_len = context.shape[1], draft.shape[1]
        mask_mod = _create_dflash_mask_mod(
            anchor_positions=torch.tensor([[8]]),
            block_keep_mask=torch.tensor([[False]]),
            ctx_len=ctx_len,
            block_size=draft_len,
            causal=True,
        )
        bm = create_block_mask(
            mask_mod, B=1, H=None, Q_LEN=draft_len, KV_LEN=ctx_len + draft_len, device="cpu"
        )
        out = attn(draft, context, draft_pos, ctx_pos, block_mask=bm)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_learnable_sink_drains_probability_mass(self):
        # The sink is one extra key with value 0: raising it shrinks the output.
        torch.manual_seed(0)
        cfg = _mla_config()
        attn = _mla_attn(cfg)
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg)
        bm = _causal_block_mask(context.shape[1], draft.shape[1], draft.shape[1], 8)

        with torch.no_grad():
            attn.learnable_sink_param.fill_(-30.0)  # sink ~ off
            small_sink = attn(draft, context, draft_pos, ctx_pos, block_mask=bm)
            attn.learnable_sink_param.fill_(30.0)  # sink takes almost all mass
            big_sink = attn(draft, context, draft_pos, ctx_pos, block_mask=bm)
        self.assertLess(big_sink.abs().max().item(), small_sink.abs().max().item())
        # o_proj has no bias, so a fully drained softmax collapses the output to 0.
        self.assertLess(big_sink.abs().max().item(), 1e-4)

    def test_dsa_keeps_at_most_topk_context_keys(self):
        torch.manual_seed(0)
        topk = 4
        cfg = _mla_config(index_topk=topk, index_head_dim=16, index_n_heads=2)
        attn = _mla_attn(cfg)
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg, ctx_len=16)
        ctx_len, draft_len = context.shape[1], draft.shape[1]
        # Anchor at the end so every context position is otherwise visible: the
        # only thing that can narrow the row is the top-k gate.
        bm = _causal_block_mask(ctx_len, draft_len, draft_len, ctx_len)
        kv_len = ctx_len + draft_len

        q_nope, _, q_resid = attn._project_q(draft)
        cos_q, sin_q = attn._cos_sin(q_nope, draft_pos, kv_len)
        cos_c, sin_c = attn._cos_sin(q_nope, ctx_pos, kv_len)
        # Only the context is scored; the draft block is always kept whole.
        index_keys = attn.indexer.project_keys(context, cos_c, sin_c)
        base_keep = materialize_keep_mask(bm, 1, draft_len, kv_len, draft.device)
        keep = attn._dsa_keep(draft, q_resid, index_keys, cos_q, sin_q, base_keep, ctx_len)

        ctx_kept = keep[0, 0, :, :ctx_len].sum(-1)
        self.assertTrue((ctx_kept <= topk).all(), f"kept more than {topk}: {ctx_kept.tolist()}")
        self.assertTrue((ctx_kept > 0).all())
        # The draft block itself is never sparsified.
        self.assertTrue(keep[0, 0, :, ctx_len:].all())

    def test_dsa_is_a_noop_when_topk_covers_every_key(self):
        torch.manual_seed(0)
        cfg_dsa = _mla_config(index_topk=64, index_head_dim=16, index_n_heads=2)
        cfg_dense = _mla_config()
        dsa = _mla_attn(cfg_dsa)
        dense = _mla_attn(cfg_dense)
        dense.load_state_dict(
            {k: v for k, v in dsa.state_dict().items() if not k.startswith("indexer.")}
        )
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg_dsa, ctx_len=16)
        bm = _causal_block_mask(context.shape[1], draft.shape[1], draft.shape[1], 8)
        torch.testing.assert_close(
            dsa(draft, context, draft_pos, ctx_pos, block_mask=bm),
            dense(draft, context, draft_pos, ctx_pos, block_mask=bm),
        )


def _sparse_config(**kw):
    """An MLA config the TileLang kernels accept: they pin the latent/rope widths."""
    kw.setdefault("dsa_backend", "tilelang")
    kw.setdefault("kv_lora_rank", 512)
    kw.setdefault("qk_rope_head_dim", 64)
    kw.setdefault("index_topk", 64)
    kw.setdefault("index_head_dim", 128)
    kw.setdefault("index_n_heads", 2)
    return _mla_config(**kw)


def _patch_sparse_kernels():
    """Run the tilelang path on reference kernels (TileLang compiles for CUDA only).

    ``_reference_sparse_mla`` is the double ``tests/test_sparse_dsa_adapter.py``
    already pins the MTP head against, reused here on purpose: both backends are
    then checked against one statement of the kernel contract. ``DSA_KERNEL_DTYPE``
    is relaxed to fp32 because bf16 is a kernel requirement, not part of the math, and
    keeping it would put a rounding floor under every parity assertion.
    """
    from tests.test_sparse_dsa_adapter import _reference_sparse_mla

    return (
        mock.patch.object(hy4_mla, "tilelang_sparse_mla", _reference_sparse_mla),
        mock.patch.object(hy4_mla, "tilelang_lightning_indexer", torch_lightning_indexer),
        mock.patch.object(hy4_mla, "DSA_KERNEL_DTYPE", torch.float32),
    )


class TestDflyHy4TileLangConfig(unittest.TestCase):
    """``dsa_backend="tilelang"``: the kernel's constraints, surfaced at config time."""

    def test_kernel_shape_constraints(self):
        # The sparse-MLA kernel runs the absorbed form with d_v pinned to 512.
        with self.assertRaises(ValueError):
            _sparse_config(kv_lora_rank=16)
        with self.assertRaises(ValueError):
            _sparse_config(qk_rope_head_dim=32)

    def test_requires_an_indexer_and_mla(self):
        # It is a *sparse* kernel: it only ever reads the keys the indexer selected.
        with self.assertRaises(ValueError):
            _sparse_config(index_topk=None)
        with self.assertRaises(ValueError):
            _config(dsa_backend="tilelang")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            _sparse_config(dsa_backend="triton")

    def test_flags_reach_the_attention_module(self):
        attn = DFlyDraftModel(_sparse_config(dsa_fp8_indexer=True)).layers[0].self_attn
        self.assertEqual(attn.dsa_backend, "tilelang")
        self.assertTrue(attn.fp8_indexer)


class TestDflyHy4AutoBackend(unittest.TestCase):
    """``dsa_backend="auto"``: tilelang where it is usable, dense torch elsewhere.

    Resolved per module at construction, in the process that runs the kernels, so
    one config works on a training image that ships TileLang and on a box that
    does not (checkpoint conversion, CPU tests) without being edited.
    """

    def _resolved(self, available, **kw):
        with mock.patch.object(hy4_mla, "tilelang_dsa_available", return_value=available):
            cfg = _sparse_config(dsa_backend="auto", **kw)
            return cfg, DFlyDraftModel(cfg).layers[0].self_attn.dsa_backend

    def test_uses_tilelang_when_the_kernels_run(self):
        _, resolved = self._resolved(True)
        self.assertEqual(resolved, "tilelang")

    def test_falls_back_to_torch_without_the_kernels(self):
        _, resolved = self._resolved(False)
        self.assertEqual(resolved, "torch")

    def test_falls_back_when_the_config_does_not_fit_the_kernels(self):
        # Same constraints "tilelang" is rejected for, but auto downgrades instead
        # of failing: the model is still trainable, just densely.
        for kw in ({"index_topk": None}, {"kv_lora_rank": 16}, {"qk_rope_head_dim": 32}):
            _, resolved = self._resolved(True, **kw)
            self.assertEqual(resolved, "torch", kw)

    def test_the_config_itself_keeps_saying_auto(self):
        # Baking the resolved value in would follow the checkpoint to a machine
        # where the answer is different.
        cfg, _ = self._resolved(False)
        self.assertEqual(cfg.dsa_backend, "auto")

    def test_explicit_backends_are_left_alone(self):
        with mock.patch.object(hy4_mla, "tilelang_dsa_available", return_value=False):
            self.assertEqual(hy4_mla.resolve_dsa_backend(_sparse_config()), "tilelang")
        with mock.patch.object(hy4_mla, "tilelang_dsa_available", return_value=True):
            self.assertEqual(
                hy4_mla.resolve_dsa_backend(_sparse_config(dsa_backend="torch")), "torch"
            )


class TestDflyHy4SparseIndexLists(unittest.TestCase):
    """The kernels take masking only as indices, so those lists are the contract."""

    def test_ranges_from_an_interval_mask(self):
        keep = torch.tensor([[[False, True, True, False], [False] * 4]])
        starts, ends = keep_mask_to_ranges(keep)
        self.assertEqual(starts.tolist(), [[1, 0]])
        # A row that sees nothing gets an empty range, not a bogus one.
        self.assertEqual(ends.tolist(), [[3, 0]])

    def test_ranges_reject_a_mask_with_gaps(self):
        # The indexer kernel has one range per query; a gap cannot be expressed,
        # and widening it silently would let a block read a forbidden document.
        with self.assertRaises(ValueError):
            keep_mask_to_ranges(torch.tensor([[[True, False, True]]]))

    def test_indices_are_ascending_offset_and_padded(self):
        keep = torch.tensor([[[True, False, True, True], [False, True, False, False]]])
        indices = keep_mask_to_indices(keep, offset=10)
        self.assertEqual(indices.tolist(), [[[10, 12, 13], [11, -1, -1]]])
        self.assertEqual(indices.dtype, torch.int32)

    def test_index_list_matches_the_training_mask(self):
        # With the context shorter than index_topk the gate is a no-op, so the list
        # must be exactly what the BlockMask allows -- no more, no less.
        torch.manual_seed(0)
        cfg = _sparse_config()
        attn = Hy4DraftMLAAttention(cfg)
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg, ctx_len=16)
        ctx_len, draft_len = context.shape[1], draft.shape[1]
        kv_len = ctx_len + draft_len
        bm = _causal_block_mask(ctx_len, draft_len, draft_len, anchor=8)
        keep = materialize_keep_mask(bm, 1, draft_len, kv_len, draft.device)

        _, _, q_resid = attn._project_q(draft)
        indices, alive = attn._sparse_kernel_indices(
            draft, q_resid, None, None, None, keep, ctx_len, draft_len, run_dsa=False
        )
        self.assertEqual(indices.shape[-1] % 64, 0)
        self.assertTrue(alive.all())
        for row in range(draft_len):
            listed = {i for i in indices[0, row, 0].tolist() if i >= 0}
            self.assertEqual(listed, set(keep[0, 0, row].nonzero().flatten().tolist()))

    def test_dropped_block_row_is_marked_dead(self):
        # An all-(-1) row reads no key: the kernel returns zeros for it, and
        # ``alive`` marks it so the output is zeroed on every backend.
        cfg = _sparse_config()
        attn = Hy4DraftMLAAttention(cfg)
        draft, context, _, _ = _dual_source_inputs(cfg, ctx_len=16)
        keep = torch.zeros(
            1, 1, draft.shape[1], context.shape[1] + draft.shape[1], dtype=torch.bool
        )
        _, _, q_resid = attn._project_q(draft)
        indices, alive = attn._sparse_kernel_indices(
            draft, q_resid, None, None, None, keep, context.shape[1], draft.shape[1], run_dsa=False
        )
        self.assertFalse(alive.any())
        self.assertTrue((indices == -1).all())


class TestDflyHy4SparseAttention(unittest.TestCase):
    """The tilelang backend must be the same model as the torch backend."""

    def setUp(self):
        self._patches = _patch_sparse_kernels()
        for patch in self._patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in self._patches])

    def _paired_attns(self, **kw):
        """A tilelang block and a torch block sharing one set of weights."""
        torch.manual_seed(0)
        fast = Hy4DraftMLAAttention(_sparse_config(**kw))
        ref = Hy4DraftMLAAttention(_sparse_config(dsa_backend="torch", **kw))
        ref.load_state_dict(fast.state_dict())
        return fast, ref

    def test_matches_torch_backend_without_dsa(self):
        # Dense index list: the absorbed form and the expanded form are the same
        # math, so this isolates the kernel plumbing from the top-k selection.
        fast, ref = self._paired_attns(index_topk=64)
        cfg = _sparse_config()
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg, ctx_len=16)
        bm = _causal_block_mask(context.shape[1], draft.shape[1], draft.shape[1], anchor=8)
        torch.testing.assert_close(
            fast(draft, context, draft_pos, ctx_pos, block_mask=bm),
            ref(draft, context, draft_pos, ctx_pos, block_mask=bm),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_matches_torch_backend_with_dsa(self):
        # Now the top-k gate is live on both sides: the torch path masks with the
        # indexer's scores, the tilelang path turns them into ranges + an index
        # list. Agreement means the ranges really are the same mask.
        fast, ref = self._paired_attns(index_topk=4)
        cfg = _sparse_config()
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg, ctx_len=16)
        bm = _causal_block_mask(context.shape[1], draft.shape[1], draft.shape[1], anchor=16)
        torch.testing.assert_close(
            fast(draft, context, draft_pos, ctx_pos, block_mask=bm),
            ref(draft, context, draft_pos, ctx_pos, block_mask=bm),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_dropped_block_gets_zero_output(self):
        from torch.nn.attention.flex_attention import create_block_mask

        from angelspec.models.dflash import _create_dflash_mask_mod

        torch.manual_seed(0)
        cfg = _sparse_config()
        attn = Hy4DraftMLAAttention(cfg)
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg, ctx_len=16)
        ctx_len, draft_len = context.shape[1], draft.shape[1]
        mask_mod = _create_dflash_mask_mod(
            anchor_positions=torch.tensor([[8]]),
            block_keep_mask=torch.tensor([[False]]),
            ctx_len=ctx_len,
            block_size=draft_len,
            causal=True,
        )
        bm = create_block_mask(
            mask_mod, B=1, H=None, Q_LEN=draft_len, KV_LEN=ctx_len + draft_len, device="cpu"
        )
        out = attn(draft, context, draft_pos, ctx_pos, block_mask=bm)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_forward_backward_through_the_kernel(self):
        torch.manual_seed(0)
        cfg = _sparse_config(index_topk=4)
        attn = Hy4DraftMLAAttention(cfg)
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(cfg, ctx_len=16)
        draft.requires_grad_(True)
        bm = _causal_block_mask(context.shape[1], draft.shape[1], draft.shape[1], anchor=16)
        attn(draft, context, draft_pos, ctx_pos, block_mask=bm).sum().backward()

        named = dict(attn.named_parameters())
        for name in ("q_a_proj.weight", "kv_b_proj.weight", "o_proj.weight", "linear_gate.weight"):
            self.assertGreater(named[name].grad.abs().sum().item(), 0.0, f"no grad for {name}")
        self.assertIsNotNone(draft.grad)
        # Selecting keys is not differentiable; the indexer is loaded, not trained.
        self.assertIsNone(named["indexer.wk.weight"].grad)


def _flex_mla_config(**kw):
    """An MLA config ``flex_attention`` will actually take.

    qk_head_dim must be a real 128 (what every shipped hy4 config uses); at the 16
    the other MLA tests use, inductor finds no kernel and raises NoValidChoicesError.
    """
    kw.setdefault("qk_nope_head_dim", 64)
    kw.setdefault("qk_rope_head_dim", 64)
    kw.setdefault("v_head_dim", 64)
    kw.setdefault("max_position_embeddings", 4096)
    return _mla_config(**kw)


def _multi_block_mask(ctx_len, n_blocks, block_size, device, dropped=()):
    """A DFly training mask with ``n_blocks`` anchors spread across the context."""
    from angelspec.models.dflash import _create_dflash_mask_mod
    from angelspec.models.ops.flex_attention import compile_friendly_create_block_mask

    keep_blocks = torch.ones(1, n_blocks, dtype=torch.bool, device=device)
    for b in dropped:
        keep_blocks[0, b] = False
    mask_mod = _create_dflash_mask_mod(
        anchor_positions=torch.linspace(1, ctx_len - 1, n_blocks, device=device)
        .long()
        .unsqueeze(0),
        block_keep_mask=keep_blocks,
        ctx_len=ctx_len,
        block_size=block_size,
        causal=True,
    )
    draft_len = n_blocks * block_size
    return compile_friendly_create_block_mask(
        mask_mod, 1, None, draft_len, ctx_len + draft_len, device, _compile=True
    )


@unittest.skipUnless(torch.cuda.is_available(), "flex_attention has no CPU backward")
class TestDflyHy4MLAFlexParity(unittest.TestCase):
    """On CUDA the dense MLA path attends through flex_attention.

    It is the same model as ``_attend_torch`` — same weights, same mask, same
    softmax — so the two must agree on outputs and gradients. What makes that
    non-obvious is the sink: flex cannot apply it inside its own softmax, so it is
    folded in afterwards as ``sigmoid(lse - sink)``, and these tests are what pins
    that identity (and the ``lse = -inf`` row a dropped block produces).
    """

    CTX_LEN, N_BLOCKS, BLOCK = 512, 16, 8

    def _module_and_inputs(self, **cfg_kw):
        torch.manual_seed(0)
        cfg = _flex_mla_config(**cfg_kw)
        attn = Hy4DraftMLAAttention(cfg).to("cuda", torch.float32)
        with torch.no_grad():
            # Zero-init would make the rescale a no-op and prove nothing.
            attn.learnable_sink_param.normal_()
        draft_len = self.N_BLOCKS * self.BLOCK
        draft, context, draft_pos, ctx_pos = _dual_source_inputs(
            cfg, ctx_len=self.CTX_LEN, draft_len=draft_len
        )
        return attn, (
            draft.cuda().requires_grad_(True),
            context.cuda().requires_grad_(True),
            draft_pos.cuda(),
            ctx_pos.cuda(),
        )

    @staticmethod
    def _dense_reference(attn, inputs, block_mask):
        """``forward`` with the flex call swapped for the dense one it replaced."""

        def dense(self_, query_states, latent, bm):
            keep = materialize_keep_mask(
                bm, 1, query_states.shape[2], latent.shape[1], query_states.device
            )
            return self_._attend_torch(query_states, latent, keep)

        with mock.patch.object(Hy4DraftMLAAttention, "_attend_flex", dense):
            return attn(*inputs, block_mask=block_mask)

    def test_matches_the_dense_path(self):
        attn, inputs = self._module_and_inputs()
        bm = _multi_block_mask(self.CTX_LEN, self.N_BLOCKS, self.BLOCK, "cuda")
        torch.testing.assert_close(
            attn(*inputs, block_mask=bm),
            self._dense_reference(attn, inputs, bm),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_matches_the_dense_path_with_the_dsa_gate(self):
        # With the gate live the mask is data-dependent, so it reaches flex as a
        # BlockMask rebuilt from the keep tensor (_keep_to_block_mask) rather than
        # as the caller's block_mask. Agreement means that rebuild is faithful.
        attn, inputs = self._module_and_inputs(index_topk=64, index_head_dim=128, index_n_heads=2)
        bm = _multi_block_mask(self.CTX_LEN, self.N_BLOCKS, self.BLOCK, "cuda")
        torch.testing.assert_close(
            attn(*inputs, block_mask=bm),
            self._dense_reference(attn, inputs, bm),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_matches_the_dense_path_on_gradients(self):
        attn, inputs = self._module_and_inputs()
        bm = _multi_block_mask(self.CTX_LEN, self.N_BLOCKS, self.BLOCK, "cuda")
        seed = torch.randn(1, self.N_BLOCKS * self.BLOCK, attn.hidden_size, device="cuda")

        def grads(out):
            # No retain_graph: flex's compiled backward donates its buffers and
            # rejects a second traversal. Each call gets its own forward anyway.
            wrt = [inputs[0], inputs[1], attn.learnable_sink_param, attn.kv_b_proj.weight]
            return torch.autograd.grad(out, wrt, seed)

        for name, got, ref in zip(
            ("draft", "context", "sink", "kv_b_proj"),
            grads(attn(*inputs, block_mask=bm)),
            grads(self._dense_reference(attn, inputs, bm)),
        ):
            torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-4, msg=f"grad {name}")

    def test_dropped_block_reads_zero(self):
        # A dropped block masks a whole query row, so flex returns lse = -inf there
        # and sigmoid(-inf - sink) must zero it rather than produce nan.
        attn, inputs = self._module_and_inputs()
        bm = _multi_block_mask(self.CTX_LEN, self.N_BLOCKS, self.BLOCK, "cuda", dropped=(3,))
        out = attn(*inputs, block_mask=bm)
        self.assertTrue(torch.isfinite(out).all())
        dropped = out[:, 3 * self.BLOCK : 4 * self.BLOCK]
        torch.testing.assert_close(dropped, torch.zeros_like(dropped))
        self.assertGreater(out[:, : 3 * self.BLOCK].abs().max().item(), 0.0)


class TestDflyHy4MLAForward(unittest.TestCase):
    def test_forward_backward_with_dsa(self):
        torch.manual_seed(0)
        cfg = _mla_config(
            num_hidden_layers=2,
            index_topk=4,
            index_head_dim=16,
            index_n_heads=2,
            indexer_types=["full", "full"],
        )
        fly = AutoDraftModel.from_config(cfg).to(torch.float32)
        fly.freeze_embedding()
        m = DFlyModel(draft_model=fly, block_size=BS, num_anchors=6)

        B, S = 2, 24
        g = torch.Generator().manual_seed(1)
        loss = m(
            input_ids=torch.randint(0, V, (B, S), generator=g),
            hidden_states_list=[torch.randn(B, S, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(B, S),
            lm_head_weight=torch.randn(V, H, generator=g),
            last_hidden_states=torch.randn(B, S, H, generator=g),
        )[0]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        named = dict(fly.named_parameters())
        for name in (
            "layers.1.self_attn.q_a_proj.weight",
            "layers.1.self_attn.kv_b_proj.weight",
            "layers.1.self_attn.o_proj.weight",
            "layers.1.self_attn.linear_gate.weight",
            "layers.1.self_attn.learnable_sink_param",
        ):
            grad = named[name].grad
            self.assertIsNotNone(grad, f"no grad for {name}")
            self.assertGreater(grad.abs().sum().item(), 0.0, f"zero grad for {name}")
        # The indexer only selects keys (top-k is not differentiable) and runs
        # under no_grad, exactly as on the MTP path: it is loaded, not trained.
        self.assertIsNone(named["layers.0.self_attn.indexer.wk.weight"].grad)

    def test_forward_backward_with_shared_indexers(self):
        torch.manual_seed(0)
        fly = AutoDraftModel.from_config(_shared_indexer_config()).to(torch.float32)
        fly.freeze_embedding()
        m = DFlyModel(draft_model=fly, block_size=BS, num_anchors=6)

        B, S = 2, 24  # ctx_len 24 > index_topk 4, so the top-k gate really runs
        g = torch.Generator().manual_seed(1)
        loss = m(
            input_ids=torch.randint(0, V, (B, S), generator=g),
            hidden_states_list=[torch.randn(B, S, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(B, S),
            lm_head_weight=torch.randn(V, H, generator=g),
            last_hidden_states=torch.randn(B, S, H, generator=g),
        )[0]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        named = dict(fly.named_parameters())
        for name in ("layers.1.self_attn.o_proj.weight", "layers.3.self_attn.o_proj.weight"):
            grad = named[name].grad
            self.assertIsNotNone(grad, f"no grad for {name}")
            self.assertGreater(grad.abs().sum().item(), 0.0, f"zero grad for {name}")
        # A borrowed indexer selects keys under no_grad, same as an owned one.
        self.assertIsNone(named["indexers.0.wk.weight"].grad)


if __name__ == "__main__":
    unittest.main()
