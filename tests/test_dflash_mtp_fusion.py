"""MTP fusion (``DFlashConfig.mtp_fusion``): the opening of the target's MTP layer, added
to the draft block input.

Covers:
1. DFlashConfig: off by default; the fields round-trip; modes, inits, the hidden size and
   the drafters DFlashTrainer builds are enforced; the module is built last, so adding it
   leaves the rest of the drafter's init unchanged
2. MTPFusion: ``f = eh_proj(enorm(E) :: hnorm(h))``; slot gates for slot0 / broadcast
3. DFlashModel: a zero gate leaves the loss bit-identical; with a gate, slot0 changes only
   slot 0 and broadcast every slot, each by exactly ``gate * f(E(x_a), norm(h_{a-1}))``, and
   a block whose anchor has no predecessor in its own document, or that is invalid, gets
   nothing; the gate is the first to get a gradient and every parameter gets one; the
   logged gate / ratio; the final hidden states and norm are required; USP is rejected
4. Loading from the target's ``mtp_layers.0`` (nested prefix, decoys, missing tensors,
   shapes), the compatibility check, the trainer's init order against draft_init_path,
   and train_entry's validation
5. The real hy4 checkpoint (with ANGELSPEC_HY4_TARGET set)
"""

import json
import os
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDraftModel,
    DFlashRMSNorm,
    MTPFusion,
)
from angelspec.models.target.target_indexer import load_target_config
from angelspec.models.target.target_mtp_fusion import (
    check_target_mtp_fusion_compatible,
    init_mtp_fusion_from_target,
)

H, V, L, BS = 32, 64, 32, 4
MODES = ("slot0", "slot1", "broadcast")
# Block 0's anchor has no predecessor, block 2's opens document 1, block 4 is invalid.
ANCHORS = torch.tensor([[0, 5, 13, 20, 26]])
KEEP = torch.tensor([[True, True, True, True, False]])
FUSED_BLOCKS = (1, 3)
DOCS = torch.tensor([[0] * 13 + [1] * (L - 13)])
POSITIONS = torch.cat([torch.arange(13), torch.arange(L - 13)]).unsqueeze(0)
FUSION_KEYS = {"enorm.weight", "hnorm.weight", "eh_proj.weight"}


def _fusion_params(mode):
    return FUSION_KEYS | ({"gate", "mask_gate"} if mode == "broadcast" else {"gate"})


def _config(**kw):
    fields = dict(
        hidden_size=H,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=6,
        target_layer_ids=[3, 4],
        mask_token_id=V - 1,
    )
    fields.update(kw)
    return DFlashConfig(**fields)


def _target_norm():
    norm = DFlashRMSNorm(H, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.5 * torch.randn(H, generator=torch.Generator().manual_seed(3)))
    return norm


def _wrapper(mode):
    torch.manual_seed(0)
    draft = DFlashDraftModel(_config(mtp_fusion=mode)).float()
    draft.freeze_embedding()
    return DFlashModel(
        draft_model=draft, block_size=BS, num_anchors=ANCHORS.shape[1], causal_attn=True
    )


def _inputs(packed=True):
    g = torch.Generator().manual_seed(1)
    inputs = dict(
        input_ids=torch.randint(0, V - 1, (1, L), generator=g),
        hidden_states_list=[torch.randn(1, L, H, generator=g) for _ in range(2)],
        loss_mask=torch.ones(1, L),
        lm_head_weight=torch.randn(V, H, generator=g),
        last_hidden_states=3 * torch.randn(1, L, H, generator=g),
        target_norm=_target_norm(),
        injected_anchors=ANCHORS,
        injected_keep_mask=KEEP,
    )
    if packed:
        inputs.update(ctx_doc_ids=DOCS, base_position_ids=POSITIONS)
    return inputs


def _block_inputs(wrapper, inputs):
    """The block input the drafter receives, as ``[n_blocks, block_size, D]``."""
    seen = {}
    original = wrapper.draft_model.forward

    def spy(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    wrapper.draft_model.forward = spy
    try:
        with torch.no_grad():
            wrapper(**inputs)
    finally:
        del wrapper.draft_model.forward
    return seen["noise_embedding"].view(ANCHORS.shape[1], BS, H)


def _rms_norm(x, weight, eps=1e-6):
    return weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _expected_fusion(draft, inputs, anchor):
    """``f`` for one anchor, re-derived from the weights."""
    fusion = draft.mtp_fusion
    embed = draft.embed_tokens.weight[inputs["input_ids"][0, anchor]]
    hidden = inputs["target_norm"](inputs["last_hidden_states"][0, anchor - 1])
    joined = torch.cat(
        [_rms_norm(embed, fusion.enorm.weight), _rms_norm(hidden, fusion.hnorm.weight)]
    )
    return fusion.eh_proj.weight @ joined


def _without_fusion(wrapper, fn):
    fusion = wrapper.draft_model.mtp_fusion
    wrapper.draft_model.mtp_fusion = None
    try:
        return fn()
    finally:
        wrapper.draft_model.mtp_fusion = fusion


def _write_target(tmp, prefix="model.", skip=(), shapes=None, seed=11):
    """A sharded target checkpoint holding ``mtp_layers.0``'s fusion plus decoys (a second
    MTP layer, a main-stack layer). Returns the fusion tensors by MTPFusion name."""
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump({"num_nextn_predict_layers": 1, "hidden_size": H, "rms_norm_eps": 1e-6}, f)
    g = torch.Generator().manual_seed(seed)
    reference = {"enorm.weight": (H,), "hnorm.weight": (H,), "eh_proj.weight": (H, 2 * H)}
    tensors, expected = {}, {}
    for name, shape in reference.items():
        if name in skip:
            continue
        tensor = torch.randn((shapes or {}).get(name, shape), generator=g)
        tensors[f"{prefix}mtp_layers.0.{name}"] = tensor
        expected[name] = tensor
    tensors[f"{prefix}mtp_layers.1.eh_proj.weight"] = torch.zeros(H, 2 * H)
    tensors[f"{prefix}layers.0.eh_proj.weight"] = torch.zeros(H, 2 * H)
    keys = sorted(tensors)
    weight_map = {}
    for file_name, shard_keys in (("a.safetensors", keys[::2]), ("b.safetensors", keys[1::2])):
        save_file({k: tensors[k].contiguous() for k in shard_keys}, os.path.join(tmp, file_name))
        weight_map.update({k: file_name for k in shard_keys})
    with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return expected


class TestDFlashConfigMTPFusion(unittest.TestCase):
    def test_off_by_default_and_round_trips(self):
        config = _config()
        self.assertIsNone(config.mtp_fusion)
        self.assertEqual(config.mtp_fusion_init, "target")
        self.assertIsNone(DFlashDraftModel(config).mtp_fusion)
        config = _config(mtp_fusion="broadcast", mtp_fusion_init="random")
        restored = DFlashConfig(
            **{k: v for k, v in config.to_dict().items() if k != "transformers_version"}
        )
        self.assertEqual((restored.mtp_fusion, restored.mtp_fusion_init), ("broadcast", "random"))

    def test_bad_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "mtp_fusion must be one of"):
            _config(mtp_fusion="all")
        with self.assertRaisesRegex(ValueError, "mtp_fusion_init must be one of"):
            _config(mtp_fusion="slot0", mtp_fusion_init="mtp")
        with self.assertRaisesRegex(ValueError, "target_hidden_size"):
            _config(mtp_fusion="slot0", target_hidden_size=2 * H)
        from angelspec.models.draft.dfly import DFlyConfig
        from angelspec.models.draft.dspark import DSparkConfig

        fields = dict(hidden_size=H, target_hidden_size=H, num_hidden_layers=1)
        with self.assertRaisesRegex(ValueError, "DFlashTrainer"):
            DFlyConfig(mtp_fusion="slot0", **fields)
        DSparkConfig(mtp_fusion="slot0", **fields)

    def test_modules_and_the_rest_of_the_init_are_unchanged(self):
        torch.manual_seed(0)
        plain = DFlashDraftModel(_config()).state_dict()
        for mode in MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(0)
                draft = DFlashDraftModel(_config(mtp_fusion=mode))
                state = draft.state_dict()
                self.assertEqual(
                    set(state) - set(plain), {f"mtp_fusion.{k}" for k in _fusion_params(mode)}
                )
                for key, value in plain.items():
                    self.assertTrue(torch.equal(state[key], value), key)
                fusion = draft.mtp_fusion
                for name in _fusion_params(mode) - FUSION_KEYS:
                    self.assertTrue(torch.equal(getattr(fusion, name), torch.zeros(H)), name)
                self.assertEqual(tuple(fusion.eh_proj.weight.shape), (H, 2 * H))

    def test_the_gates_stay_on_adamw_under_muon(self):
        from angelspec.training.optimizer import _is_muon_matrix

        draft = DFlashDraftModel(_config(mtp_fusion="broadcast"))
        on_muon = {
            name: _is_muon_matrix(name, param)
            for name, param in draft.named_parameters()
            if name.startswith("mtp_fusion.")
        }
        self.assertEqual(
            on_muon,
            {
                "mtp_fusion.enorm.weight": False,
                "mtp_fusion.hnorm.weight": False,
                "mtp_fusion.eh_proj.weight": True,
                "mtp_fusion.gate": False,
                "mtp_fusion.mask_gate": False,
            },
        )


class TestMTPFusionModule(unittest.TestCase):
    def test_fusion_and_slot_gates(self):
        torch.manual_seed(0)
        fusion = MTPFusion(H, "broadcast")
        with torch.no_grad():
            fusion.enorm.weight.uniform_(0.5, 1.5)
            fusion.hnorm.weight.uniform_(0.5, 1.5)
            fusion.gate.normal_()
            fusion.mask_gate.normal_()
        embeds, hidden = torch.randn(3, H), torch.randn(3, H)
        expected = (
            torch.cat(
                [_rms_norm(embeds, fusion.enorm.weight), _rms_norm(hidden, fusion.hnorm.weight)],
                dim=-1,
            )
            @ fusion.eh_proj.weight.T
        )
        torch.testing.assert_close(fusion.fuse(embeds, hidden), expected)
        gates = fusion.slot_gates(BS)
        self.assertTrue(torch.equal(gates[0], fusion.gate))
        self.assertTrue(torch.equal(gates[1:], fusion.mask_gate.expand(BS - 1, -1)))
        self.assertTrue(
            torch.equal(
                fusion(embeds, hidden, BS), gates * fusion.fuse(embeds, hidden).unsqueeze(-2)
            )
        )

        for mode, slot in (("slot0", 0), ("slot1", 1)):
            with self.subTest(mode=mode):
                single = MTPFusion(H, mode)
                self.assertIsNone(single.mask_gate)
                with torch.no_grad():
                    single.gate.fill_(2.0)
                expected_gates = torch.zeros(BS, H)
                expected_gates[slot] = 2.0
                self.assertTrue(torch.equal(single.slot_gates(BS), expected_gates))
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            MTPFusion(H, "slot2")


class TestDFlashModelMTPFusion(unittest.TestCase):
    def test_a_zero_gate_leaves_the_drafter_unchanged(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                wrapper = _wrapper(mode)
                inputs = _inputs()
                with torch.no_grad():
                    fused = wrapper(**inputs)
                    plain = _without_fusion(wrapper, lambda w=wrapper, i=inputs: w(**i))
                for got, expected in zip(fused[:5], plain[:5]):
                    self.assertTrue(torch.equal(got, expected))

    def test_each_mode_adds_gate_times_fusion_to_its_slots(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                wrapper = _wrapper(mode)
                fusion = wrapper.draft_model.mtp_fusion
                g = torch.Generator().manual_seed(5)
                with torch.no_grad():
                    fusion.gate.normal_(generator=g)
                    if fusion.mask_gate is not None:
                        fusion.mask_gate.normal_(generator=g)
                inputs = _inputs()
                fused = _block_inputs(wrapper, inputs)
                plain = _without_fusion(wrapper, lambda w=wrapper, i=inputs: _block_inputs(w, i))
                delta = fused - plain
                for block, anchor in enumerate(ANCHORS[0].tolist()):
                    if block not in FUSED_BLOCKS:
                        self.assertTrue(torch.equal(fused[block], plain[block]), block)
                        continue
                    with torch.no_grad():
                        f = _expected_fusion(wrapper.draft_model, inputs, anchor)
                    if mode == "broadcast":
                        torch.testing.assert_close(delta[block, 0], fusion.gate * f)
                        rest = delta[block, 1:]
                        torch.testing.assert_close(rest, (fusion.mask_gate * f).expand_as(rest))
                        continue
                    slot = 0 if mode == "slot0" else 1
                    torch.testing.assert_close(delta[block, slot], fusion.gate * f)
                    for other in range(BS):
                        if other != slot:
                            self.assertTrue(
                                torch.equal(fused[block, other], plain[block, other]), other
                            )

    def test_unpacked_blocks_only_skip_anchor_zero_and_invalid_ones(self):
        wrapper = _wrapper("slot0")
        with torch.no_grad():
            wrapper.draft_model.mtp_fusion.gate.fill_(1.0)
        inputs = _inputs(packed=False)
        fused = _block_inputs(wrapper, inputs)
        plain = _without_fusion(wrapper, lambda: _block_inputs(wrapper, inputs))
        changed = [not torch.equal(fused[b], plain[b]) for b in range(ANCHORS.shape[1])]
        self.assertEqual(changed, [False, True, True, True, False])

    def test_the_gate_gets_a_gradient_first_and_every_parameter_one(self):
        wrapper = _wrapper("broadcast")
        fusion = wrapper.draft_model.mtp_fusion
        wrapper(**_inputs())[0].backward()
        for param in (fusion.gate, fusion.mask_gate):
            self.assertGreater(float(param.grad.abs().sum()), 0.0)
        for param in (fusion.eh_proj.weight, fusion.enorm.weight, fusion.hnorm.weight):
            self.assertTrue(torch.equal(param.grad, torch.zeros_like(param)))
        # REPLICATE wraps the drafter DDP-style, which rejects parameters left without one.
        self.assertEqual(
            [n for n, p in wrapper.named_parameters() if p.requires_grad and p.grad is None], []
        )
        wrapper.zero_grad()
        with torch.no_grad():
            fusion.gate.fill_(0.1)
            fusion.mask_gate.fill_(0.1)
        wrapper(**_inputs())[0].backward()
        for param in (fusion.eh_proj.weight, fusion.enorm.weight, fusion.hnorm.weight):
            self.assertGreater(float(param.grad.abs().sum()), 0.0)

    def test_logs_the_gate_and_the_injection_against_the_input_it_lands_on(self):
        for mode in ("slot0", "slot1"):
            with self.subTest(mode=mode):
                wrapper = _wrapper(mode)
                inputs = _inputs()
                with torch.no_grad():
                    components = wrapper(**inputs)[5]
                self.assertEqual(float(components["mtp_fusion_gate"]), 0.0)
                self.assertEqual(float(components["mtp_fusion_ratio"]), 0.0)
                with torch.no_grad():
                    wrapper.draft_model.mtp_fusion.gate.fill_(0.5)
                    components = wrapper(**inputs)[5]
                    draft = wrapper.draft_model
                    anchors = [ANCHORS[0, b].item() for b in FUSED_BLOCKS]
                    injected = sum(
                        (0.5 * _expected_fusion(draft, inputs, a)).pow(2).sum() for a in anchors
                    )
                    # slot0 lands on the anchor's embedding, slot1 on the mask token's.
                    ids = [
                        inputs["input_ids"][0, a] if mode == "slot0" else V - 1 for a in anchors
                    ]
                    base = sum(draft.embed_tokens.weight[i].pow(2).sum() for i in ids)
                self.assertAlmostEqual(float(components["mtp_fusion_gate"]), 0.5, places=6)
                torch.testing.assert_close(
                    components["mtp_fusion_ratio"], (injected / base).sqrt(), rtol=1e-5, atol=1e-6
                )
                plain = _without_fusion(wrapper, lambda w=wrapper, i=inputs: w(**i))[5]
                self.assertNotIn("mtp_fusion_gate", plain)

    def test_the_final_hidden_states_and_norm_are_required(self):
        wrapper = _wrapper("slot0")
        for missing, message in (
            ("last_hidden_states", "store_last_hidden_states"),
            ("target_norm", "target norm"),
        ):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, message):
                inputs = _inputs()
                inputs.pop(missing)
                with torch.no_grad():
                    wrapper(**inputs)

    def test_propose_blocks_reads_them_too(self):
        wrapper = _wrapper("broadcast")
        with torch.no_grad():
            wrapper.draft_model.mtp_fusion.gate.fill_(1.0)
        inputs = _inputs()
        proposals, anchors, keep = wrapper.propose_blocks(**inputs)
        self.assertEqual(tuple(proposals.shape), (1, ANCHORS.shape[1], BS))
        self.assertTrue(torch.equal(anchors, torch.where(KEEP, ANCHORS, 0)))
        self.assertTrue(torch.equal(keep, KEEP))
        inputs.pop("last_hidden_states")
        with self.assertRaisesRegex(ValueError, "store_last_hidden_states"):
            wrapper.propose_blocks(**inputs)

    def test_saved_once_with_the_drafter(self):
        wrapper = _wrapper("slot0")
        keys = [k for k in wrapper.state_dict() if "mtp_fusion" in k]
        self.assertEqual(
            sorted(keys), sorted(f"draft_model.mtp_fusion.{k}" for k in _fusion_params("slot0"))
        )

    def test_usp_is_rejected(self):
        draft = DFlashDraftModel(_config(mtp_fusion="slot0"))
        with self.assertRaisesRegex(NotImplementedError, "USP"):
            DFlashModel(draft_model=draft, block_size=BS, num_anchors=3, sp_size=2)

    def test_on_a_drafter_reading_the_targets_own_kv(self):
        from angelspec.models.target.target_kv_latent import TargetKVLatent

        rope = {"rope_theta": 10000.0, "rope_type": "default"}
        mla = dict(q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8)
        torch.manual_seed(0)
        draft = DFlashDraftModel(
            _config(
                attention_type="mla",
                v_head_dim=12,
                gated_mla=True,
                learnable_sink=True,
                rope_parameters=rope,
                target_kv_layers=[4, 5],
                mtp_fusion="broadcast",
                **mla,
            )
        ).float()
        draft.freeze_embedding()
        wrapper = DFlashModel(
            draft_model=draft, block_size=BS, num_anchors=ANCHORS.shape[1], causal_attn=True
        )
        target = dict(hidden_size=H, rms_norm_eps=1e-6, num_hidden_layers=6, rope_parameters=rope)
        wrapper.attach_target_kv_latents(
            [TargetKVLatent(dict(target, **mla), layer).float() for layer in (4, 5)]
        )
        inputs = _inputs()
        with torch.no_grad():
            fused = wrapper(**inputs)
            plain = _without_fusion(wrapper, lambda: wrapper(**inputs))
            self.assertTrue(torch.equal(fused[0], plain[0]))
            draft.mtp_fusion.gate.fill_(1.0)
            draft.mtp_fusion.mask_gate.fill_(1.0)
            self.assertFalse(torch.equal(wrapper(**inputs)[0], plain[0]))
        wrapper(**inputs)[0].backward()
        self.assertEqual(
            [n for n, p in wrapper.named_parameters() if p.requires_grad and p.grad is None], []
        )
        self.assertGreater(float(draft.mtp_fusion.eh_proj.weight.grad.abs().sum()), 0.0)

    def test_dspark_confidence_head_on_a_drafter_reading_the_targets_own_kv(self):
        from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
        from angelspec.models.dspark import DSparkModel
        from angelspec.models.target.target_kv_latent import TargetKVLatent

        rope = {"rope_theta": 10000.0, "rope_type": "default"}
        mla = dict(q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8)
        fields = _config().to_dict()
        for key in ("model_type", "architectures", "transformers_version"):
            fields.pop(key, None)
        fields.update(
            model_arch="dflash",
            attention_type="mla",
            v_head_dim=12,
            gated_mla=True,
            learnable_sink=True,
            rope_parameters=rope,
            target_kv_layers=[4, 5],
            mtp_fusion="slot0",
            markov_rank=0,
            enable_confidence_head=True,
            confidence_head_with_markov=False,
            **mla,
        )
        torch.manual_seed(0)
        draft = DSparkDraftModel(DSparkConfig(**fields)).float()
        draft.freeze_embedding()
        with torch.no_grad():
            draft.mtp_fusion.gate.fill_(1.0)
        wrapper = DSparkModel(
            draft_model=draft,
            block_size=BS,
            num_anchors=ANCHORS.shape[1],
            loss_objective="dpace",
            ce_loss_alpha=1.0,
            lk_loss_weight=1.0,
            confidence_head_alpha=1.0,
            causal_attn=True,
        )
        target = dict(hidden_size=H, rms_norm_eps=1e-6, num_hidden_layers=6, rope_parameters=rope)
        wrapper.attach_target_kv_latents(
            [TargetKVLatent(dict(target, **mla), layer).float() for layer in (4, 5)]
        )
        out = wrapper(**_inputs())
        self.assertGreater(float(out[-1]["confidence_loss"]), 0.0)
        out[0].backward()
        self.assertEqual(
            [n for n, p in wrapper.named_parameters() if p.requires_grad and p.grad is None], []
        )
        self.assertGreater(float(draft.confidence_head.proj.weight.grad.abs().sum()), 0.0)


class TestTargetMTPFusion(unittest.TestCase):
    def test_copies_mtp_layer_0_and_leaves_the_gate(self):
        for prefix in ("model.", "model.language_model."):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                expected = _write_target(tmp, prefix=prefix)
                fusion = MTPFusion(H, "broadcast")
                init_mtp_fusion_from_target(fusion, tmp)
                params = dict(fusion.named_parameters())
                for name, tensor in expected.items():
                    self.assertTrue(torch.equal(params[name], tensor), name)
                self.assertTrue(torch.equal(fusion.gate, torch.zeros(H)))
                self.assertTrue(torch.equal(fusion.mask_gate, torch.zeros(H)))

    def test_missing_tensors_and_shapes_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_target(tmp, skip=("hnorm.weight",))
            with self.assertRaisesRegex(KeyError, "hnorm"):
                init_mtp_fusion_from_target(MTPFusion(H, "slot0"), tmp)
        with tempfile.TemporaryDirectory() as tmp:
            _write_target(tmp, shapes={"eh_proj.weight": (H, H)})
            with self.assertRaisesRegex(ValueError, "shape"):
                init_mtp_fusion_from_target(MTPFusion(H, "slot0"), tmp)

    def test_the_target_needs_an_mtp_layer_of_the_drafters_shape(self):
        draft = _config(mtp_fusion="slot0")
        target = {"num_nextn_predict_layers": 1, "hidden_size": H, "rms_norm_eps": 1e-6}
        check_target_mtp_fusion_compatible(target, draft)
        check_target_mtp_fusion_compatible({"text_config": target}, draft)
        cases = {
            "num_nextn_predict_layers": dict(target, num_nextn_predict_layers=0),
            "hidden_size": dict(target, hidden_size=2 * H),
            "rms_norm_eps": dict(target, rms_norm_eps=1e-5),
        }
        for name, config in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                check_target_mtp_fusion_compatible(config, draft)

    def test_trainer_copies_it_before_draft_init_path(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        trainer = object.__new__(DFlashTrainer)
        with tempfile.TemporaryDirectory() as target, tempfile.TemporaryDirectory() as init:
            expected = _write_target(target)
            # An hf_final from a drafter trained without the fusion keeps the copy.
            torch.save(DFlashDraftModel(_config()).state_dict(), f"{init}/pytorch_model.bin")
            trainer.args = Namespace(draft_init_path=init)
            draft = DFlashDraftModel(_config(mtp_fusion="slot0"))
            DFlashTrainer._init_mtp_fusion_from_target(trainer, draft, target)
            DFlashTrainer._load_draft_init_weights(trainer, draft)
            params = dict(draft.mtp_fusion.named_parameters())
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(params[name], tensor), name)
            self.assertTrue(torch.equal(draft.mtp_fusion.gate, torch.zeros(H)))

            # One carrying fusion weights takes precedence.
            source = DFlashDraftModel(_config(mtp_fusion="slot0"))
            with torch.no_grad():
                source.mtp_fusion.gate.fill_(0.3)
            torch.save(source.state_dict(), f"{init}/pytorch_model.bin")
            draft = DFlashDraftModel(_config(mtp_fusion="slot0"))
            DFlashTrainer._init_mtp_fusion_from_target(trainer, draft, target)
            DFlashTrainer._load_draft_init_weights(trainer, draft)
            for name, value in source.mtp_fusion.state_dict().items():
                self.assertTrue(torch.equal(draft.mtp_fusion.state_dict()[name], value), name)

            # mtp_fusion_init="random" copies nothing.
            draft = DFlashDraftModel(_config(mtp_fusion="slot0", mtp_fusion_init="random"))
            before = draft.mtp_fusion.eh_proj.weight.clone()
            DFlashTrainer._init_mtp_fusion_from_target(trainer, draft, target)
            self.assertTrue(torch.equal(draft.mtp_fusion.eh_proj.weight, before))

        self.assertFalse(DFlashTrainer._mtp_fusion_on(trainer))
        trainer.dflash = SimpleNamespace(draft_model=DFlashDraftModel(_config()))
        self.assertFalse(DFlashTrainer._mtp_fusion_on(trainer))
        trainer.dflash = SimpleNamespace(draft_model=draft)
        self.assertTrue(DFlashTrainer._mtp_fusion_on(trainer))

    def test_train_entry_validation(self):
        from angelspec.train_entry import _validate_and_configure_dflash

        with tempfile.TemporaryDirectory() as tmp:
            _write_target(tmp)
            base = dict(
                inference_engine_type="vllm",
                dflash_block_size=BS,
                min_loss_tokens=2 * BS,
                aux_hidden_states_layers=[3, 4],
                dflash_num_target_layers=2,
                store_last_hidden_states=True,
                target_model_path=tmp,
            )
            _validate_and_configure_dflash(Namespace(**base), _config(mtp_fusion="slot0"))
            cases = {
                "store_last_hidden_states": (dict(store_last_hidden_states=False), ValueError),
                "vllm": (dict(inference_engine_type="sgl"), NotImplementedError),
            }
            for message, (override, error) in cases.items():
                with self.subTest(message=message), self.assertRaisesRegex(error, message):
                    _validate_and_configure_dflash(
                        Namespace(**dict(base, **override)), _config(mtp_fusion="slot0")
                    )
            with self.assertRaisesRegex(ValueError, "rms_norm_eps"):
                _validate_and_configure_dflash(
                    Namespace(**base), _config(mtp_fusion="slot0", rms_norm_eps=1e-5)
                )
            # A fusion started from scratch never reads the target's config.
            _validate_and_configure_dflash(
                Namespace(**dict(base, target_model_path=os.path.join(tmp, "missing"))),
                _config(mtp_fusion="broadcast", mtp_fusion_init="random"),
            )

    @unittest.skipUnless(
        os.environ.get("ANGELSPEC_HY4_TARGET"), "set ANGELSPEC_HY4_TARGET to a hy4 MTP target"
    )
    def test_real_hy4_checkpoint(self):
        path = os.environ["ANGELSPEC_HY4_TARGET"]
        draft = DFlashConfig(
            hidden_size=2816,
            target_hidden_size=2816,
            rms_norm_eps=1e-5,
            num_hidden_layers=1,
            target_num_hidden_layers=34,
            mtp_fusion="broadcast",
        )
        check_target_mtp_fusion_compatible(load_target_config(path), draft)
        fusion = MTPFusion(2816, "broadcast", rms_norm_eps=1e-5)
        init_mtp_fusion_from_target(fusion, path)
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        params = dict(fusion.named_parameters())
        for name in FUSION_KEYS:
            key = f"model.mtp_layers.0.{name}"
            with safe_open(os.path.join(path, weight_map[key]), framework="pt") as f:
                self.assertTrue(torch.equal(params[name], f.get_tensor(key).float()), name)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            fused = fusion.fuse(
                0.02 * torch.randn(4, 2816, generator=g), torch.randn(4, 2816, generator=g)
            )
        self.assertTrue(bool(torch.isfinite(fused).all()))
        self.assertGreater(float(fused.abs().mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
