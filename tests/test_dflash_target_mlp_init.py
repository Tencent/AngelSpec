"""Target-initialized, fully trainable MoE FFNs in the DFlash backbone."""

import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.draft.mtp import Hy3MoE
from angelspec.models.target.block_quant import FP4_E2M1_VALUES
from angelspec.models.target.target_indexer import load_target_config
from angelspec.models.target.target_mlp_init import (
    _checkpoint_layout,
    _layer_prefix,
    check_target_mlp_dense_init_compatible,
    check_target_mlp_init_compatible,
    init_dense_mlp_from_target,
    init_mlp_from_target,
)

H, I, E, TOP_K, LAYERS = 16, 6, 4, 2, [2, 3]


def _config(config_class=DFlashConfig, **overrides):
    fields = dict(
        hidden_size=H,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        target_hidden_size=H,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 2],
        num_target_layers=2,
        mask_token_id=31,
        draft_mlp_type="moe",
        target_mlp_init_layers=LAYERS,
        num_experts=E,
        num_shared_experts=1,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=I,
        moe_router_use_sigmoid=True,
        moe_router_enable_expert_bias=True,
        route_norm=True,
        router_scaling_factor=2.5,
        swiglu_limit=7.0,
        moe_skip_expert_init=True,
    )
    fields.update(overrides)
    return config_class(**fields)


def _target_config(**overrides):
    fields = dict(
        hidden_size=H,
        num_hidden_layers=4,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        n_routed_experts=E,
        n_shared_experts=1,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=I,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        swiglu_limit=7.0,
        rms_norm_eps=1e-6,
    )
    fields.update(overrides)
    return fields


class TestTargetMoEConfig(unittest.TestCase):
    def test_builds_complete_moe_layers(self):
        config = _config()
        model = DFlashDraftModel(config)
        self.assertEqual(config.target_mlp_init_layers, LAYERS)
        for layer in model.layers:
            self.assertIsInstance(layer.mlp, Hy3MoE)
            self.assertEqual(tuple(layer.mlp.experts_gate_proj.weight.shape), (E, H, I))
            self.assertEqual(tuple(layer.mlp.experts_down_proj.weight.shape), (E, I, H))
            for name, param in layer.mlp.named_parameters():
                self.assertEqual(param.requires_grad, name != "expert_bias", name)
        aux_free = DFlashDraftModel(_config(moe_aux_free_bias_update=True))
        self.assertTrue(all(layer.mlp.expert_bias.requires_grad for layer in aux_free.layers))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "draft_mlp_type"):
            _config(draft_mlp_type="dense")
        with self.assertRaisesRegex(ValueError, "lists 1 target layers"):
            _config(target_mlp_init_layers=[2])
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
            _config(num_experts_per_tok=E + 1)
        with self.assertRaisesRegex(ValueError, "complete target MoE"):
            check_target_mlp_init_compatible(_target_config(), _config(num_experts=E + 1), LAYERS)
        with self.assertRaisesRegex(ValueError, "are dense"):
            check_target_mlp_init_compatible(_target_config(), _config(), [0, 2])
        check_target_mlp_init_compatible(_target_config(), _config(), LAYERS)


class TestAuxFreeExpertBias(unittest.TestCase):
    def _module(self):
        model = DFlashDraftModel(
            _config(
                moe_skip_expert_init=False,
                moe_aux_free_bias_update=True,
                moe_aux_free_bias_beta=0.5,
                moe_aux_free_bias_gamma=1e-3,
            )
        ).float()
        moe = model.layers[0].mlp
        with torch.no_grad():
            moe.router.gate.weight.zero_()
            moe.expert_bias.copy_(torch.tensor([2.0, 1.0, 0.0, -1.0]))
        return model, moe

    def test_forward_accumulates_and_clears_hard_assignments(self):
        _, moe = self._module()
        with mock.patch.dict(os.environ, {"ANGELSPEC_MTP_MOE_REFERENCE": "1"}):
            moe(torch.randn(3, 5, H))
        counts = moe.take_expert_load_counts()
        torch.testing.assert_close(counts, torch.tensor([15.0, 15.0, 0.0, 0.0]))
        self.assertEqual(float(moe.take_expert_load_counts().sum()), 0.0)
        self.assertNotIn("_expert_load_counts", moe.state_dict())
        self.assertNotIn("expert_load_ema", moe.state_dict())

    def test_ema_and_trainer_controller_push_load_toward_uniform(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        model, moe = self._module()
        with mock.patch.dict(os.environ, {"ANGELSPEC_MTP_MOE_REFERENCE": "1"}):
            moe(torch.randn(2, 4, H))

        class _Optimizer:
            def __init__(self):
                self.updates = {}

            def add_to_parameter(self, name, delta):
                self.updates[name] = delta.clone()

        trainer = object.__new__(DFlashTrainer)
        trainer.draft_model = model
        trainer.optimizer = _Optimizer()
        with mock.patch("torch.distributed.all_reduce"):
            trainer._after_optimizer_step(0)

        delta = trainer.optimizer.updates["layers.0.mlp.expert_bias"]
        self.assertLess(float(delta[0]), 0.0)
        self.assertLess(float(delta[1]), 0.0)
        self.assertGreater(float(delta[2]), 0.0)
        self.assertGreater(float(delta[3]), 0.0)
        self.assertAlmostEqual(float(delta.sum()), 0.0, places=7)
        self.assertEqual(trainer._moe_aux_metrics["train/moe_layer_0/used_experts"], 2)
        self.assertGreater(trainer._moe_aux_metrics["train/moe_layer_0/load_cv"], 0.0)

    def test_bf16_optimizer_carries_the_controller_steps(self):
        from angelspec.training.optimizer import BF16Optimizer

        model, moe = self._module()
        model = model.to(torch.bfloat16)
        opt = BF16Optimizer(
            model,
            lr=1e-2,
            total_steps=10,
            warmup_ratio=0.0,
            optimizer_type="adamw",
            weight_decay=0.1,
        )
        name = "layers.0.mlp.expert_bias"
        master = opt.fp32_params[opt.param_name_to_index[name]]
        expected = master.detach().clone()
        delta = torch.tensor([1e-4, -1e-4, 1e-4, -1e-4])  # far below bf16 steps near 1-2
        for _ in range(100):
            opt.add_to_parameter(name, delta)
            expected += delta
        torch.testing.assert_close(master, expected)
        torch.testing.assert_close(moe.expert_bias, expected.to(torch.bfloat16))

        # An optimizer step with gradients everywhere else neither decays nor moves it.
        for param_name, param in model.named_parameters():
            if param.requires_grad and param_name != name:
                param.grad = torch.ones_like(param)
        opt.step()
        torch.testing.assert_close(master, expected)

    def test_train_entry_rejects_the_controller_under_replicate(self):
        from argparse import Namespace

        from angelspec.train_entry import _validate_and_configure_dflash

        config = _config(
            target_mlp_init_layers=None, moe_skip_expert_init=False, moe_aux_free_bias_update=True
        )
        args = dict(dflash_block_size=4, min_loss_tokens=8, fsdp_strategy="FULL_SHARD")
        _validate_and_configure_dflash(Namespace(**args), config)
        with self.assertRaisesRegex(ValueError, "REPLICATE"):
            _validate_and_configure_dflash(
                Namespace(**dict(args, fsdp_strategy="REPLICATE")), config
            )


class TestMoELRScales(unittest.TestCase):
    def test_scales_cover_exactly_the_target_copied_moe_params(self):
        from angelspec.training.dflash_trainer import moe_lr_scales

        model = DFlashDraftModel(_config())
        scales = moe_lr_scales(model, 0.2, router_scale=0.1)
        expected = set()
        for i, layer in enumerate(model.layers):
            expected |= {
                f"layers.{i}.mlp.{name}"
                for name, param in layer.mlp.named_parameters()
                if param.requires_grad
            }
            expected.add(f"layers.{i}.post_attention_layernorm.weight")
        self.assertEqual(set(scales), expected)
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertLessEqual(set(scales), trainable)
        self.assertEqual(scales["layers.0.mlp.router.gate.weight"], 0.1)
        self.assertEqual(scales["layers.1.mlp.experts_gate_proj.weight"], 0.2)
        self.assertEqual(scales["layers.1.mlp.shared_mlp.down_proj.weight"], 0.2)
        self.assertEqual(scales["layers.0.post_attention_layernorm.weight"], 0.2)
        self.assertFalse(any("self_attn" in name or "input_layernorm" in name for name in scales))

    def test_dense_draft_has_nothing_to_scale(self):
        from angelspec.training.dflash_trainer import moe_lr_scales

        dense = DFlashDraftModel(
            _config(
                draft_mlp_type="dense", target_mlp_init_layers=None, moe_skip_expert_init=False
            )
        )
        self.assertEqual(moe_lr_scales(dense, 0.2), {})


def _write_moe_target(root, prefix="model.layers."):
    generator = torch.Generator().manual_seed(4)
    tensors = {}
    expected = {}
    for layer in LAYERS:
        base = f"{prefix}{layer}."
        values = {
            "post_attention_layernorm.weight": torch.randn(H, generator=generator),
            "mlp.gate.weight": torch.randn(E, H, generator=generator),
            "mlp.gate.e_score_correction_bias": torch.randn(E, generator=generator),
            "mlp.shared_experts.gate_proj.weight": torch.randn(I, H, generator=generator),
            "mlp.shared_experts.up_proj.weight": torch.randn(I, H, generator=generator),
            "mlp.shared_experts.down_proj.weight": torch.randn(H, I, generator=generator),
            "mlp.experts.gate_up_proj": torch.randn(E, 2 * I, H, generator=generator),
            "mlp.experts.down_proj": torch.randn(E, H, I, generator=generator),
        }
        for suffix, tensor in values.items():
            tensors[base + suffix] = tensor
            expected[layer, suffix] = tensor
    # Decoys ensure layer-stack/prefix discovery does not pick MTP or another layer.
    tensors["model.mtp_layers.2.mlp.gate.weight"] = torch.zeros(E, H)
    tensors["model.layers.1.mlp.gate.weight"] = torch.zeros(E, H)
    _save_target(root, tensors, _target_config())
    return expected


def _save_target(root, tensors, config):
    keys = sorted(tensors)
    shards = {"a.safetensors": keys[::2], "b.safetensors": keys[1::2]}
    weight_map = {}
    for name, shard_keys in shards.items():
        save_file(
            {key: tensors[key].contiguous() for key in shard_keys},
            os.path.join(root, name),
        )
        weight_map.update({key: name for key in shard_keys})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config, f)


FP8_BLOCK = 4
PROJS = (("gate_proj", "w1", (I, H)), ("up_proj", "w3", (I, H)), ("down_proj", "w2", (H, I)))


def _write_glm_style_target(root):
    generator = torch.Generator().manual_seed(5)
    tensors, expected = {}, {}

    def fp8(key, rows, cols):
        scale = torch.rand(-(-rows // FP8_BLOCK), -(-cols // FP8_BLOCK), generator=generator) + 0.5
        tile = scale.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)[:rows, :cols]
        weight = (torch.randn(rows, cols, generator=generator) / tile).to(torch.float8_e4m3fn)
        tensors[key], tensors[key + "_scale_inv"] = weight, scale
        return weight.float() * tile

    for layer in LAYERS:
        base = f"model.language_model.layers.{layer}."
        plain = {
            "post_attention_layernorm.weight": torch.randn(H, generator=generator),
            "mlp.gate.weight": torch.randn(E, H, generator=generator).to(torch.bfloat16),
            "mlp.gate.e_score_correction_bias": torch.randn(E, generator=generator),
        }
        for suffix, tensor in plain.items():
            tensors[base + suffix] = tensor
            expected[layer, suffix] = tensor.float()
        for proj, _, (rows, cols) in PROJS:
            expected[layer, "shared", proj] = fp8(
                base + f"mlp.shared_experts.{proj}.weight", rows, cols
            )
            for e in range(E):
                expected[layer, e, proj] = fp8(base + f"mlp.experts.{e}.{proj}.weight", rows, cols)
    config = dict(
        _target_config(), quantization_config={"weight_block_size": [FP8_BLOCK, FP8_BLOCK]}
    )
    _save_target(root, tensors, config)
    return expected


def _write_deepseek_style_target(root):
    generator = torch.Generator().manual_seed(6)
    tensors, expected = {}, {}
    table = torch.tensor(FP4_E2M1_VALUES)

    def store_scale(key, shape):
        exponents = torch.randint(-3, 3, shape, generator=generator).float()
        tensors[key[: -len("weight")] + "scale"] = torch.exp2(exponents).to(torch.float8_e8m0fnu)
        return torch.exp2(exponents)

    def fp4(key, rows, cols):
        codes = torch.randint(0, 16, (rows, cols), generator=generator, dtype=torch.uint8)
        tensors[key] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
        scale = store_scale(key, (rows, -(-cols // 32)))
        return table[codes.long()] * scale.repeat_interleave(32, 1)[:, :cols]

    def fp8(key, rows, cols):
        weight = torch.randn(rows, cols, generator=generator).to(torch.float8_e4m3fn)
        tensors[key] = weight
        scale = store_scale(key, (-(-rows // 32), -(-cols // 32)))
        return (
            weight.float() * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)[:rows, :cols]
        )

    for layer in LAYERS:
        base = f"layers.{layer}."
        plain = {
            "ffn_norm.weight": torch.randn(H, generator=generator),
            "ffn.gate.weight": torch.randn(E, H, generator=generator).to(torch.bfloat16),
            "ffn.gate.bias": torch.randn(E, generator=generator),
            "ffn.gate.bias_vl": torch.randn(E, generator=generator),
        }
        for suffix, tensor in plain.items():
            tensors[base + suffix] = tensor
            expected[layer, suffix] = tensor.float()
        for proj, name, (rows, cols) in PROJS:
            expected[layer, "shared", proj] = fp8(
                base + f"ffn.shared_experts.{name}.weight", rows, cols
            )
            for e in range(E):
                expected[layer, e, proj] = fp4(base + f"ffn.experts.{e}.{name}.weight", rows, cols)
    tensors["mtp.0.ffn.gate.weight"] = torch.zeros(E, H)
    config = {
        "quantization_config": {"weight_block_size": [32, 32]},
        "text_config": _target_config(scoring_func="sqrtsoftplus"),
    }
    _save_target(root, tensors, config)
    return expected


K_DENSE = 2


def _dense_config(**overrides):
    fields = dict(
        draft_mlp_type="dense",
        target_mlp_init_layers=None,
        moe_skip_expert_init=False,
        intermediate_size=(1 + K_DENSE) * I,
        target_mlp_dense_init_layers=LAYERS,
    )
    fields.update(overrides)
    return _config(**fields)


def _swiglu(x, gate, up, down):
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


class TestTargetDenseFFNInit(unittest.TestCase):
    def test_dense_ffn_is_shared_plus_scaled_routed_experts(self):
        with tempfile.TemporaryDirectory() as root:
            expected = _write_moe_target(root, prefix="model.language_model.layers.")
            model = DFlashDraftModel(_dense_config()).float()
            chosen = init_dense_mlp_from_target(model, root, LAYERS)
        x = torch.randn(5, H)
        for draft_idx, layer in enumerate(LAYERS):

            def get(suffix, layer=layer):
                return expected[layer, suffix]

            bias = get("mlp.gate.e_score_correction_bias")
            ids = torch.argsort(bias, stable=True)[:K_DENSE].tolist()
            self.assertEqual(chosen[draft_idx], ids)
            want = _swiglu(
                x,
                get("mlp.shared_experts.gate_proj.weight"),
                get("mlp.shared_experts.up_proj.weight"),
                get("mlp.shared_experts.down_proj.weight"),
            )
            gate_up, down = get("mlp.experts.gate_up_proj"), get("mlp.experts.down_proj")
            for e in ids:
                # _target_config's routed_scaling_factor is 2.5.
                want = want + 2.5 / K_DENSE * _swiglu(x, gate_up[e, :I], gate_up[e, I:], down[e])
            draft_layer = model.layers[draft_idx]
            torch.testing.assert_close(draft_layer.mlp(x), want, rtol=1e-4, atol=1e-3)
            torch.testing.assert_close(
                draft_layer.post_attention_layernorm.weight, get("post_attention_layernorm.weight")
            )

    def test_explicit_experts_and_routed_scale(self):
        experts = [[3, 1], [0, 2]]
        with tempfile.TemporaryDirectory() as root:
            expected = _write_moe_target(root)
            config = _dense_config(
                target_mlp_dense_init_experts=experts, target_mlp_dense_init_routed_scale=0.5
            )
            model = DFlashDraftModel(config).float()
            chosen = init_dense_mlp_from_target(
                model,
                root,
                LAYERS,
                config.target_mlp_dense_init_experts,
                config.target_mlp_dense_init_routed_scale,
            )
        self.assertEqual(chosen, experts)
        mlp = model.layers[0].mlp
        gate_up = expected[LAYERS[0], "mlp.experts.gate_up_proj"]
        down = expected[LAYERS[0], "mlp.experts.down_proj"]
        shared_gate = expected[LAYERS[0], "mlp.shared_experts.gate_proj.weight"]
        shared_down = expected[LAYERS[0], "mlp.shared_experts.down_proj.weight"]
        torch.testing.assert_close(mlp.gate_proj.weight[:I], shared_gate)
        torch.testing.assert_close(mlp.gate_proj.weight[I : 2 * I], gate_up[3, :I])
        torch.testing.assert_close(mlp.up_proj.weight[2 * I :], gate_up[1, I:])
        torch.testing.assert_close(mlp.down_proj.weight[:, :I], shared_down)
        torch.testing.assert_close(mlp.down_proj.weight[:, I : 2 * I], 0.5 * down[3])

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "draft_mlp_type='dense'"):
            _dense_config(draft_mlp_type="moe")
        with self.assertRaisesRegex(ValueError, "'bias', 'random' or per-layer"):
            _dense_config(target_mlp_dense_init_experts="load")
        with self.assertRaisesRegex(ValueError, "distinct"):
            _dense_config(target_mlp_dense_init_experts=[[1, 1], [0, 2]])
        with self.assertRaisesRegex(ValueError, "distinct"):
            _dense_config(target_mlp_dense_init_experts=[[1, 2]])
        with self.assertRaisesRegex(ValueError, "routed_scale"):
            _dense_config(target_mlp_dense_init_routed_scale=0.0)
        self.assertEqual(
            check_target_mlp_dense_init_compatible(_target_config(), _dense_config(), LAYERS),
            K_DENSE,
        )
        with self.assertRaisesRegex(ValueError, "intermediate_size"):
            check_target_mlp_dense_init_compatible(
                _target_config(), _dense_config(intermediate_size=2 * I + 1), LAYERS
            )
        with self.assertRaisesRegex(ValueError, "needs 3 expert ids"):
            check_target_mlp_dense_init_compatible(
                _target_config(),
                _dense_config(
                    intermediate_size=4 * I, target_mlp_dense_init_experts=[[0, 1], [2, 3]]
                ),
                LAYERS,
            )
        with self.assertRaisesRegex(ValueError, "are dense"):
            check_target_mlp_dense_init_compatible(_target_config(), _dense_config(), [0, 2])


class TestTargetMoESubset(unittest.TestCase):
    def test_copies_listed_experts_with_their_router_rows(self):
        experts = [[3, 1], [0, 2]]
        with tempfile.TemporaryDirectory() as root:
            expected = _write_moe_target(root)
            model = DFlashDraftModel(
                _config(num_experts=2, target_mlp_init_experts=experts)
            ).float()
            chosen = init_mlp_from_target(model, root, LAYERS, experts)
        self.assertEqual(chosen, experts)
        for draft_idx, layer in enumerate(LAYERS):

            def get(suffix, layer=layer):
                return expected[layer, suffix]

            ids = experts[draft_idx]
            moe = model.layers[draft_idx].mlp
            torch.testing.assert_close(moe.router.gate.weight, get("mlp.gate.weight")[ids])
            torch.testing.assert_close(
                moe.expert_bias, get("mlp.gate.e_score_correction_bias")[ids]
            )
            gate_up = get("mlp.experts.gate_up_proj")[ids]
            torch.testing.assert_close(
                moe.experts_gate_proj.weight, gate_up[:, :I].transpose(1, 2)
            )
            torch.testing.assert_close(moe.experts_up_proj.weight, gate_up[:, I:].transpose(1, 2))
            torch.testing.assert_close(
                moe.experts_down_proj.weight, get("mlp.experts.down_proj")[ids].transpose(1, 2)
            )
            torch.testing.assert_close(
                moe.shared_mlp.gate_proj.weight, get("mlp.shared_experts.gate_proj.weight")
            )
        with mock.patch.dict(os.environ, {"ANGELSPEC_MTP_MOE_REFERENCE": "1"}):
            out = model.layers[0].mlp(torch.randn(2, 3, H))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_bias_picks_the_lowest_correction_biases(self):
        with tempfile.TemporaryDirectory() as root:
            expected = _write_moe_target(root)
            model = DFlashDraftModel(
                _config(num_experts=2, target_mlp_init_experts="bias")
            ).float()
            chosen = init_mlp_from_target(model, root, LAYERS, "bias")
        for draft_idx, layer in enumerate(LAYERS):
            bias = expected[layer, "mlp.gate.e_score_correction_bias"]
            self.assertEqual(chosen[draft_idx], torch.argsort(bias, stable=True)[:2].tolist())
            torch.testing.assert_close(
                model.layers[draft_idx].mlp.expert_bias, bias[chosen[draft_idx]]
            )

    def test_random_draw_is_seeded_by_draft_layer(self):
        with tempfile.TemporaryDirectory() as root:
            expected = _write_moe_target(root)
            model = DFlashDraftModel(
                _config(num_experts=2, target_mlp_init_experts="random")
            ).float()
            chosen = init_mlp_from_target(model, root, LAYERS, "random")
            again = init_mlp_from_target(model, root, LAYERS, "random")
        self.assertEqual(chosen, again)
        for draft_idx, layer in enumerate(LAYERS):
            generator = torch.Generator().manual_seed(draft_idx)
            self.assertEqual(
                chosen[draft_idx], sorted(torch.randperm(E, generator=generator)[:2].tolist())
            )
            torch.testing.assert_close(
                model.layers[draft_idx].mlp.router.gate.weight,
                expected[layer, "mlp.gate.weight"][chosen[draft_idx]],
            )

    def test_subset_may_route_fewer_experts_and_score_funcs_must_match(self):
        subset = _config(num_experts=2, num_experts_per_tok=1, target_mlp_init_experts="random")
        check_target_mlp_init_compatible(_target_config(), subset, LAYERS)
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
            check_target_mlp_init_compatible(
                _target_config(), _config(num_experts_per_tok=1), LAYERS
            )
        sqrtsoftplus = _target_config(scoring_func="sqrtsoftplus")
        with self.assertRaisesRegex(
            ValueError, "moe_router_score_func: target 'sqrtsoftplus' vs drafter 'sigmoid'"
        ):
            check_target_mlp_init_compatible(sqrtsoftplus, _config(), LAYERS)
        check_target_mlp_init_compatible(
            sqrtsoftplus, _config(moe_router_score_func="sqrtsoftplus"), LAYERS
        )

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "'bias', 'random' or per-layer"):
            _config(num_experts=2, target_mlp_init_experts="load")
        with self.assertRaisesRegex(ValueError, "2 distinct"):
            _config(num_experts=2, target_mlp_init_experts=[[0, 1, 2], [0, 1, 2]])
        with self.assertRaisesRegex(ValueError, "needs target_mlp_init_layers"):
            _config(
                target_mlp_init_layers=None,
                moe_skip_expert_init=False,
                target_mlp_init_experts="bias",
            )
        check_target_mlp_init_compatible(
            _target_config(), _config(num_experts=2, target_mlp_init_experts="bias"), LAYERS
        )
        with self.assertRaisesRegex(ValueError, "below 4"):
            check_target_mlp_init_compatible(
                _target_config(),
                _config(num_experts=2, target_mlp_init_experts=[[0, 4], [1, 2]]),
                LAYERS,
            )
        with self.assertRaisesRegex(ValueError, "of the target's 4 experts"):
            check_target_mlp_init_compatible(
                _target_config(), _config(num_experts=5, target_mlp_init_experts="bias"), LAYERS
            )


class TestTargetMoEWeights(unittest.TestCase):
    def _write_target(self, root, prefix="model.layers."):
        return _write_moe_target(root, prefix)

    def _assert_loaded(self, model, expected):
        for draft_idx, target_layer in enumerate(LAYERS):
            layer = model.layers[draft_idx]
            moe = layer.mlp

            def get(suffix, target_layer=target_layer):
                return expected[target_layer, suffix]

            torch.testing.assert_close(
                layer.post_attention_layernorm.weight,
                get("post_attention_layernorm.weight"),
            )
            torch.testing.assert_close(moe.router.gate.weight, get("mlp.gate.weight"))
            torch.testing.assert_close(moe.expert_bias, get("mlp.gate.e_score_correction_bias"))
            for proj in ("gate_proj", "up_proj", "down_proj"):
                torch.testing.assert_close(
                    getattr(moe.shared_mlp, proj).weight,
                    get(f"mlp.shared_experts.{proj}.weight"),
                )
            gate_up = get("mlp.experts.gate_up_proj")
            torch.testing.assert_close(
                moe.experts_gate_proj.weight, gate_up[:, :I].transpose(1, 2)
            )
            torch.testing.assert_close(moe.experts_up_proj.weight, gate_up[:, I:].transpose(1, 2))
            torch.testing.assert_close(
                moe.experts_down_proj.weight,
                get("mlp.experts.down_proj").transpose(1, 2),
            )
            for name, param in moe.named_parameters():
                self.assertEqual(param.requires_grad, name != "expert_bias", name)

    def test_loads_every_moe_tensor_with_nested_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            expected = self._write_target(root, prefix="model.language_model.layers.")
            model = DFlashDraftModel(_config()).float()
            before_attention = {
                name: tensor.clone()
                for name, tensor in model.state_dict().items()
                if ".self_attn." in name or ".input_layernorm." in name
            }
            init_mlp_from_target(model, root, LAYERS)
        self._assert_loaded(model, expected)
        for name, value in before_attention.items():
            torch.testing.assert_close(model.state_dict()[name], value, msg=name)

    def test_forward_and_backward_train_copied_weights(self):
        with tempfile.TemporaryDirectory() as root:
            expected = self._write_target(root)
            model = DFlashDraftModel(_config()).float()
            init_mlp_from_target(model, root, LAYERS)
        self._assert_loaded(model, expected)
        x = torch.randn(3, 5, H, requires_grad=True)
        with mock.patch.dict(os.environ, {"ANGELSPEC_MTP_MOE_REFERENCE": "1"}):
            out = model.layers[0].mlp(model.layers[0].post_attention_layernorm(x))
            out.square().mean().backward()
        moe = model.layers[0].mlp
        self.assertGreater(float(moe.router.gate.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(moe.shared_mlp.gate_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(moe.experts_gate_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_huggingface_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as target, tempfile.TemporaryDirectory() as output:
            self._write_target(target)
            model = DFlashDraftModel(_config()).float()
            init_mlp_from_target(model, target, LAYERS)
            model.save_pretrained(output)
            restored = DFlashDraftModel(_config()).float()
            restored.load_state_dict(load_file(os.path.join(output, "model.safetensors")))
        self.assertEqual(restored.config.draft_mlp_type, "moe")
        self.assertEqual(restored.config.target_mlp_init_layers, LAYERS)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor, msg=name)

    def test_shape_and_missing_key_are_errors(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_target(root)
            index = os.path.join(root, "model.safetensors.index.json")
            with open(index) as f:
                payload = json.load(f)
            payload["weight_map"].pop("model.layers.2.mlp.experts.down_proj")
            with open(index, "w") as f:
                json.dump(payload, f)
            with self.assertRaisesRegex(KeyError, "experts.down_proj"):
                init_mlp_from_target(DFlashDraftModel(_config()).float(), root, LAYERS)


class TestPerExpertTargets(unittest.TestCase):
    def _assert_moe(self, layer, expected, target_layer, ids, norm, router, bias):
        moe = layer.mlp
        torch.testing.assert_close(
            layer.post_attention_layernorm.weight, expected[target_layer, norm]
        )
        torch.testing.assert_close(moe.router.gate.weight, expected[target_layer, router][ids])
        torch.testing.assert_close(moe.expert_bias, expected[target_layer, bias][ids])
        for proj, _, _ in PROJS:
            param = getattr(moe, f"experts_{proj}").weight
            for slot, e in enumerate(ids):
                torch.testing.assert_close(param[slot], expected[target_layer, e, proj].t())
            torch.testing.assert_close(
                getattr(moe.shared_mlp, proj).weight, expected[target_layer, "shared", proj]
            )

    def test_glm_style_block_fp8_experts(self):
        config = _config(num_experts=2, num_experts_per_tok=1, target_mlp_init_experts="random")
        with tempfile.TemporaryDirectory() as root:
            expected = _write_glm_style_target(root)
            check_target_mlp_init_compatible(load_target_config(root), config, LAYERS)
            model = DFlashDraftModel(config).float()
            chosen = init_mlp_from_target(model, root, LAYERS, "random")
        for draft_idx, layer in enumerate(LAYERS):
            self._assert_moe(
                model.layers[draft_idx],
                expected,
                layer,
                chosen[draft_idx],
                "post_attention_layernorm.weight",
                "mlp.gate.weight",
                "mlp.gate.e_score_correction_bias",
            )

    def test_deepseek_v4_style_fp4_experts(self):
        config = _config(moe_router_score_func="sqrtsoftplus")
        with tempfile.TemporaryDirectory() as root:
            expected = _write_deepseek_style_target(root)
            check_target_mlp_init_compatible(load_target_config(root), config, LAYERS)
            model = DFlashDraftModel(config).float()
            self.assertIsNone(init_mlp_from_target(model, root, LAYERS))
        for draft_idx, layer in enumerate(LAYERS):
            self._assert_moe(
                model.layers[draft_idx],
                expected,
                layer,
                list(range(E)),
                "ffn_norm.weight",
                "ffn.gate.weight",
                "ffn.gate.bias",
            )
        moe = model.layers[0].mlp
        self.assertEqual(moe.score_func, "sqrtsoftplus")
        with mock.patch.dict(os.environ, {"ANGELSPEC_MTP_MOE_REFERENCE": "1"}):
            out = moe(torch.randn(2, 3, H))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_dspark_draft_takes_the_same_experts(self):
        config = _config(DSparkConfig, moe_router_score_func="sqrtsoftplus", markov_rank=4)
        with tempfile.TemporaryDirectory() as root:
            expected = _write_deepseek_style_target(root)
            check_target_mlp_init_compatible(load_target_config(root), config, LAYERS)
            model = DSparkDraftModel(config).float()
            heads = {
                name: tensor.clone()
                for name, tensor in model.state_dict().items()
                if name.startswith(("markov_head.", "confidence_head."))
            }
            self.assertIsNone(init_mlp_from_target(model, root, LAYERS))
        for draft_idx, layer in enumerate(LAYERS):
            self._assert_moe(
                model.layers[draft_idx],
                expected,
                layer,
                list(range(E)),
                "ffn_norm.weight",
                "ffn.gate.weight",
                "ffn.gate.bias",
            )
        self.assertTrue(heads)
        for name, tensor in heads.items():
            torch.testing.assert_close(model.state_dict()[name], tensor, msg=name)


@unittest.skipUnless(
    os.environ.get("ANGELSPEC_HY4_TARGET") and torch.cuda.is_available(),
    "set ANGELSPEC_HY4_TARGET and provide CUDA for a real full-MoE check",
)
class TestRealTargetMoE(unittest.TestCase):
    def test_real_checkpoint_grouped_mm_forward_backward(self):
        path = os.environ["ANGELSPEC_HY4_TARGET"]
        target = load_target_config(path)
        text = target.get("text_config") or target
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            config = DFlashConfig(
                hidden_size=text["hidden_size"],
                intermediate_size=text["intermediate_size"],
                num_hidden_layers=1,
                num_attention_heads=text["num_attention_heads"],
                num_key_value_heads=text["num_key_value_heads"],
                vocab_size=1,
                rms_norm_eps=text["rms_norm_eps"],
                target_hidden_size=text["hidden_size"],
                target_num_hidden_layers=text["num_hidden_layers"],
                target_layer_ids=[16],
                num_target_layers=1,
                mask_token_id=0,
                draft_mlp_type="moe",
                target_mlp_init_layers=[17],
                num_experts=text["n_routed_experts"],
                num_shared_experts=text["n_shared_experts"],
                num_experts_per_tok=text["num_experts_per_tok"],
                moe_intermediate_size=text["moe_intermediate_size"],
                moe_router_use_sigmoid=True,
                moe_router_enable_expert_bias=True,
                route_norm=text["norm_topk_prob"],
                router_scaling_factor=text["routed_scaling_factor"],
                swiglu_limit=text["swiglu_limit"],
                moe_skip_expert_init=True,
            )
            model = DFlashDraftModel(config)
        finally:
            torch.set_default_dtype(old_dtype)
        check_target_mlp_init_compatible(target, config, [17])
        init_mlp_from_target(model, path, [17])

        model = model.to(torch.bfloat16)
        moe = model.layers[0].mlp.cuda()
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in moe.parameters()))
        x = torch.randn(
            1,
            8,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        out = moe(x)
        out.float().square().mean().backward()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertGreater(float(moe.router.gate.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(moe.experts_gate_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(moe.shared_mlp.gate_proj.weight.grad.abs().sum()), 0.0)


@unittest.skipUnless(
    os.environ.get("ANGELSPEC_HY4_TARGET"), "set ANGELSPEC_HY4_TARGET for a real dense-FFN check"
)
class TestRealTargetDenseFFN(unittest.TestCase):
    def test_real_checkpoint_shared_plus_top_k(self):
        path = os.environ["ANGELSPEC_HY4_TARGET"]
        target = load_target_config(path)
        text = target.get("text_config") or target
        width, k = text["moe_intermediate_size"], text["num_experts_per_tok"]
        config = DFlashConfig(
            hidden_size=text["hidden_size"],
            intermediate_size=(text["n_shared_experts"] + k) * width,
            num_hidden_layers=1,
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            vocab_size=1,
            rms_norm_eps=text["rms_norm_eps"],
            target_hidden_size=text["hidden_size"],
            target_num_hidden_layers=text["num_hidden_layers"],
            target_layer_ids=[16],
            num_target_layers=1,
            mask_token_id=0,
            target_mlp_dense_init_layers=[17],
        )
        self.assertEqual(check_target_mlp_dense_init_compatible(target, config, [17]), k)
        model = DFlashDraftModel(config)
        chosen = init_dense_mlp_from_target(model, path, [17])

        weight_map = _checkpoint_layout(path)
        prefix = _layer_prefix(weight_map, 17)

        def read(suffix):
            with safe_open(os.path.join(path, weight_map[prefix + suffix]), framework="pt") as f:
                return f.get_tensor(prefix + suffix).float()

        bias = read("mlp.gate.e_score_correction_bias")
        self.assertEqual(chosen, [torch.argsort(bias, stable=True)[:k].tolist()])
        mlp = model.layers[0].mlp
        torch.testing.assert_close(
            mlp.gate_proj.weight[:width], read("mlp.shared_experts.gate_proj.weight")
        )
        torch.testing.assert_close(
            model.layers[0].post_attention_layernorm.weight,
            read("post_attention_layernorm.weight"),
        )
        out = mlp(torch.randn(4, config.hidden_size))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_real_checkpoint_moe_expert_subset(self):
        path = os.environ["ANGELSPEC_HY4_TARGET"]
        target = load_target_config(path)
        text = target.get("text_config") or target
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            config = DFlashConfig(
                hidden_size=text["hidden_size"],
                intermediate_size=text["intermediate_size"],
                num_hidden_layers=1,
                num_attention_heads=text["num_attention_heads"],
                num_key_value_heads=text["num_key_value_heads"],
                vocab_size=1,
                rms_norm_eps=text["rms_norm_eps"],
                target_hidden_size=text["hidden_size"],
                target_num_hidden_layers=text["num_hidden_layers"],
                target_layer_ids=[16],
                num_target_layers=1,
                mask_token_id=0,
                draft_mlp_type="moe",
                target_mlp_init_layers=[17],
                target_mlp_init_experts="bias",
                num_experts=64,
                num_shared_experts=text["n_shared_experts"],
                num_experts_per_tok=text["num_experts_per_tok"],
                moe_intermediate_size=text["moe_intermediate_size"],
                route_norm=text["norm_topk_prob"],
                router_scaling_factor=text["routed_scaling_factor"],
                swiglu_limit=text["swiglu_limit"],
                moe_skip_expert_init=True,
            )
            model = DFlashDraftModel(config)
        finally:
            torch.set_default_dtype(old_dtype)
        check_target_mlp_init_compatible(target, config, [17])
        (ids,) = init_mlp_from_target(model, path, [17], "bias")

        weight_map = _checkpoint_layout(path)
        prefix = _layer_prefix(weight_map, 17)

        def read(suffix):
            with safe_open(os.path.join(path, weight_map[prefix + suffix]), framework="pt") as f:
                return f.get_tensor(prefix + suffix)

        bias = read("mlp.gate.e_score_correction_bias")
        self.assertEqual(ids, torch.argsort(bias.float(), stable=True)[:64].tolist())
        moe = model.layers[0].mlp
        torch.testing.assert_close(moe.router.gate.weight, read("mlp.gate.weight")[ids].float())
        torch.testing.assert_close(moe.expert_bias, bias[ids].to(moe.expert_bias.dtype))
        with safe_open(
            os.path.join(path, weight_map[prefix + "mlp.experts.down_proj"]), framework="pt"
        ) as f:
            down = f.get_slice(prefix + "mlp.experts.down_proj")[ids[5] : ids[5] + 1][0]
        torch.testing.assert_close(moe.experts_down_proj.weight[5], down.transpose(0, 1))


if __name__ == "__main__":
    unittest.main()
