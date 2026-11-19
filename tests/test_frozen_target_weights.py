"""Target layers a drafter reads (DFly2's KV reuse, index picks and attention init) ride in
the frozen bundle, so offline training reads the same tensors online training does."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from angelspec.data.offline import atomic_json
from angelspec.data.offline_assets import export_target_assets, target_weight_prefixes
from angelspec.models.draft.dflash import DFlashConfig
from angelspec.models.target import target_indexer, target_kv_latent
from angelspec.models.target.target_indexer import TargetIndexer
from angelspec.models.target.target_kv_latent import TargetKVLatent
from angelspec.training.offline import check_frozen_target_weights
from tests.test_dflash_target_kv import RANK, ROPE, H, V, _target_config

LAYERS = [3, 5]


def _draft_config():
    return DFlashConfig(
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=6,
        target_layer_ids=[layer - 1 for layer in LAYERS],
        mask_token_id=V - 1,
        attention_type="mla",
        q_lora_rank=32,
        kv_lora_rank=RANK,
        qk_nope_head_dim=8,
        qk_rope_head_dim=ROPE,
        v_head_dim=12,
        target_kv_layers=LAYERS,
        target_index_layer=LAYERS,
        target_index_topk=8,
    )


class TestFrozenTargetWeights(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.target.mkdir()
        config = dict(_target_config(), vocab_size=V, model_type="hy_v4_flash")
        atomic_json(self.target / "config.json", config)
        generator = torch.Generator().manual_seed(0)
        layer_keys = {
            **target_kv_latent._CHECKPOINT_KEYS,
            **target_indexer._CHECKPOINT_KEYS,
        }
        shapes = {
            name: tuple(param.shape)
            for module in (TargetKVLatent(config, 3), TargetIndexer(config, 3, 8))
            for name, param in module.named_parameters()
        }
        tensors = {
            "model.embed_tokens.weight": torch.randn(V, H, generator=generator),
            "lm_head.weight": torch.randn(V, H, generator=generator),
            "model.norm.weight": torch.ones(H),
        }
        for layer in (3, 4, 5):
            for name, suffix in layer_keys.items():
                tensors[f"model.layers.{layer}.{suffix}"] = torch.randn(
                    shapes[name], generator=generator
                )
            fp8 = f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"
            tensors[fp8] = tensors[fp8].to(torch.float8_e4m3fn)
            tensors[fp8 + "_scale_inv"] = torch.rand(1, 1, generator=generator) + 0.5
            tensors[f"model.layers.{layer}.mlp.gate.weight"] = torch.randn(4, H)
        decoy = "model.mtp_layers.3.self_attn.kv_a_proj_with_mqa.weight"
        tensors[decoy] = torch.randn(RANK + ROPE, H, generator=generator)
        keys = sorted(tensors)
        shards = {"a.safetensors": keys[::2], "b.safetensors": keys[1::2]}
        for name, shard in shards.items():
            save_file({key: tensors[key] for key in shard}, str(self.target / name))
        atomic_json(
            self.target / "model.safetensors.index.json",
            {"weight_map": {key: name for name, shard in shards.items() for key in shard}},
        )
        self.tensors = tensors
        self.draft = _draft_config()

    def test_bundle_carries_exactly_the_layers_the_draft_reads(self):
        prefixes = target_weight_prefixes(self.target, self.draft)
        self.assertEqual(
            prefixes,
            [
                f"model.layers.{layer}.{part}."
                for layer in LAYERS
                for part in ("input_layernorm", "self_attn")
            ],
        )
        frozen = self.root / "frozen"
        export_target_assets(self.target, frozen, weight_prefixes=prefixes)
        bundled = load_file(str(frozen / "frozen.safetensors"))
        expected = {
            key
            for key in self.tensors
            if key.startswith(tuple(prefixes))
            or key in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")
        }
        self.assertEqual(set(bundled), expected)
        for key in expected:
            self.assertTrue(
                torch.equal(bundled[key].view(torch.uint8), self.tensors[key].view(torch.uint8))
            )
        for layer in LAYERS:
            for cls, args in ((TargetKVLatent, ()), (TargetIndexer, (8,))):
                full = cls.from_pretrained(str(self.target), layer, *args, device="cpu")
                offline = cls.from_pretrained(str(frozen), layer, *args, device="cpu")
                for (name, a), (_, b) in zip(full.named_parameters(), offline.named_parameters()):
                    self.assertTrue(torch.equal(a, b), f"{cls.__name__} layer {layer} {name}")
        check_frozen_target_weights(frozen, self.draft)

    def test_offline_refuses_a_bundle_without_those_layers(self):
        frozen = self.root / "plain"
        export_target_assets(self.target, frozen)
        with self.assertRaisesRegex(ValueError, "lacks target layers"):
            check_frozen_target_weights(frozen, self.draft)
        with self.assertRaisesRegex(ValueError, "other target layers"):
            export_target_assets(
                self.target,
                frozen,
                weight_prefixes=target_weight_prefixes(self.target, self.draft),
            )


if __name__ == "__main__":
    unittest.main()
