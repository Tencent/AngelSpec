import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from angelspec.config.train_config import config_to_flat_args, load_config
from angelspec.data.offline import HiddenDataset, HiddenWriter, OfflineBatchFetcher
from angelspec.data.utils import DFlashPackingCollator
from angelspec.inference.online_dump import build_online_capture_recipe
from angelspec.models.draft.auto import AutoDraftModel, AutoDraftModelConfig
from angelspec.models.draft.dflash import DFlashConfig
from angelspec.models.draft.dfly import DFlyConfig
from angelspec.models.draft.dspark import DSparkConfig
from angelspec.models.draft.offline_family import offline_family, offline_training_knobs
from angelspec.training.activation_checkpointing import (
    enable_dflash_activation_checkpointing,
)
from angelspec.training.dflash_trainer import DFlashTrainer
from angelspec.training.dfly_trainer import DFlyTrainer
from angelspec.training.dspark_trainer import DSparkTrainer
from angelspec.training.offline import attach_offline_data, configure_model
from tests.offline_native import native_recipe, to_native


def family_config(family):
    values = dict(
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        mask_token_id=31,
        num_target_layers=2,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        max_position_embeddings=512,
        head_dim=8,
    )
    if family.endswith("_mla"):
        family = family.removesuffix("_mla")
        values.update(
            attention_type="mla",
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            gated_mla=True,
            learnable_sink=True,
        )
    if family == "dfly":
        return DFlyConfig(**values)
    if family == "dspark":
        return DSparkConfig(**values, markov_rank=4, enable_hidden_correction=True)
    if family == "dflare":
        return DFlashConfig(**values, model_arch="dflare")
    if family == "gated":
        return DFlashConfig(**values, fusion_type="gated_sum")
    return DFlashConfig(**values)


def trainer_for(config, args):
    family = offline_family(config)
    cls = {"dfly": DFlyTrainer, "dspark": DSparkTrainer}.get(family, DFlashTrainer)
    with mock.patch("angelspec.training.trainer.Trainer.__init__", return_value=None):
        trainer = cls(args)
    trainer.args = args
    trainer.sp_group = None
    return trainer


class TestOfflineFamily(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = config_to_flat_args(
            load_config(
                cli_args=[
                    f"offline.manifest_path={self.root}",
                    "offline.frozen_target_path=/tmp/frozen",
                    "training.attention_backend=flex_attention",
                    "training.micro_batch_size=1",
                    "training.max_seq_length=32",
                    "training.dflash_num_target_layers=2",
                    "training.dspark_num_target_layers=2",
                    "training.dflash_num_anchors=2",
                    "training.dspark_num_anchors=2",
                    "training.dflash_block_size=2",
                    "dataset.min_loss_tokens=4",
                    "training.prefetch_depth=0",
                    "training.dflash_causal_attn=true",
                    "training.dflash_ce_loss_alpha=0.0",
                    "training.dflash_e2e_tv_loss_weight=1.0",
                ]
            )
        )
        self.recipe = native_recipe([1], 3, 16, 32)
        self.samples = []
        writer = HiddenWriter(self.root, self.recipe)
        for i in range(2):
            sample = {
                "input_ids": torch.randint(0, 31, (1, 8)),
                "loss_mask": torch.ones(1, 8, dtype=torch.uint8),
                "attention_mask": torch.ones(1, 8, dtype=torch.bool),
                "hidden_states": torch.randn(1, 8, 16),
                "last_hidden_states": torch.randn(1, 8, 16),
            }
            self.samples.append(sample)
            writer.write(str(i), to_native(sample, [1], 3))
        writer.publish(["0", "1"])
        self.dataset = HiddenDataset(self.root)
        self.target = {"hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 4}

    def test_feature_contracts_and_trainer_dispatch(self):
        for family in ("dflash", "dfly", "dfly_mla", "dflare", "dspark", "dspark_mla", "gated"):
            with self.subTest(family=family):
                config = family_config(family)
                args = copy.deepcopy(self.args)
                configure_model(args, self.dataset, config, self.target)
                trainer = trainer_for(config, args)
                draft = trainer._build_draft_model(config)
                wrapper = trainer._build_training_wrapper(draft)
                self.assertIs(wrapper.draft_model, draft)
                self.assertEqual(wrapper.num_anchors, 2)
                self.assertEqual(
                    args.offline_model_family,
                    "dflash" if family == "gated" else family.removesuffix("_mla"),
                )
                if family == "dfly":
                    self.assertIsInstance(trainer, DFlyTrainer)
                if family == "dspark":
                    self.assertIsInstance(trainer, DSparkTrainer)
                    self.assertEqual(
                        wrapper.confidence_head_alpha, args.dspark_confidence_head_alpha
                    )

    def test_disk_packing_recompute_and_gradients_all_families(self):
        direct = DFlashPackingCollator(32)(self.samples)
        fetcher = OfflineBatchFetcher(
            self.dataset,
            rank=0,
            world_size=1,
            seed=0,
            shuffle=False,
            max_length=32,
            hidden_size=16,
            aux_layer_ids=[1],
            final_layer_id=3,
            vocab_size=32,
            min_loss_tokens=4,
            packing=True,
        )
        disk = next(fetcher)
        for family in ("dflash", "dfly", "dfly_mla", "dflare", "dspark", "dspark_mla", "gated"):
            with self.subTest(family=family):
                config = family_config(family)
                trainer = trainer_for(config, self.args)
                draft = trainer._build_draft_model(config)
                draft.freeze_embedding()
                reference = trainer._build_training_wrapper(draft)
                actual = copy.deepcopy(reference)
                enable_dflash_activation_checkpointing(actual.draft_model)
                head = torch.randn(32, 16)
                norm = torch.nn.RMSNorm(16).requires_grad_(False)

                def forward(model, batch, norm=norm, head=head):
                    return model(
                        input_ids=batch["input_ids"],
                        loss_mask=batch["loss_mask"],
                        hidden_states_list=[batch["hidden_states"], batch["last_hidden_states"]],
                        last_hidden_states=batch["last_hidden_states"],
                        target_norm=norm,
                        lm_head_weight=head,
                        ctx_doc_ids=batch["ctx_doc_ids"],
                        base_position_ids=batch["base_position_ids"],
                        injected_anchors=torch.tensor([[1, 9]]),
                        injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
                    )

                output = forward(reference, direct)
                other = forward(actual, disk)
                torch.testing.assert_close(output[0], other[0])
                if family == "dspark":
                    self.assertGreater(float(other[-1]["confidence_loss"]), 0.0)
                output[0].backward()
                other[0].backward()
                actual_state = actual.state_dict(keep_vars=True)
                for name, parameter in reference.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, actual_state[name].grad)
                for model in (reference, actual):
                    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(value, actual.state_dict()[name])

    def test_native_capture_contract_is_reusable_between_families(self):
        recipes = []
        for family in ("dflash", "dfly", "dflare", "dspark", "gated"):
            path = self.root / f"{family}.json"
            config = family_config(family)
            architecture = {
                "dfly": "DFlyDraftModel",
                "dspark": "DSparkDraftModel",
                "dflare": "DFlareDraftModel",
                "gated": "DFlashGatedDraftModel",
            }.get(family, "DFlashDraftModel")
            config.architectures = [architecture]
            config.to_json_file(path)
            draft = AutoDraftModelConfig.from_file(path)
            recipe = build_online_capture_recipe(self.args, draft, self.target, {"test": "id"})
            # model_family only records which config captured; loading does not check it.
            recipes.append({key: value for key, value in recipe.items() if key != "model_family"})
        self.assertTrue(all(value == recipes[0] for value in recipes))

    def test_exported_configs_reconstruct_correct_model(self):
        for family in ("dflash", "dfly", "dflare", "dspark", "gated"):
            with self.subTest(family=family):
                config = family_config(family)
                original = AutoDraftModel.from_config(config)
                output = self.root / family
                original.save_pretrained(output, safe_serialization=False)
                restored_config = AutoDraftModelConfig.from_file(output / "config.json")
                restored = AutoDraftModel.from_config(restored_config)
                self.assertIs(type(restored), type(original))
                restored.load_state_dict(original.state_dict(), strict=True)

    def test_dspark_offline_and_online_use_shared_init(self):
        trainer = trainer_for(family_config("dspark"), self.args)
        with mock.patch.object(DFlashTrainer, "init_model", return_value=5) as initialize:
            config = family_config("dspark")
            self.assertEqual(trainer.init_model(config, "/tmp/frozen"), 5)
            initialize.assert_called_once_with(config, "/tmp/frozen", None)
        with mock.patch.object(DFlashTrainer, "init_model", return_value=7) as initialize:
            trainer.args.offline_manifest_path = None
            config = family_config("dspark")
            self.assertEqual(trainer.init_model(config, "/tmp/target"), 7)
            initialize.assert_called_once_with(config, "/tmp/target", None)

    def test_architecture_restrictions_and_dspark_knobs(self):
        config = family_config("dspark")
        args = copy.deepcopy(self.args)
        args.dflash_num_anchors = 99
        self.assertEqual(offline_training_knobs(args, config), ("dspark", 2, 2))
        args.attention_backend = "usp"
        self.assertEqual(offline_training_knobs(args, config), ("dspark", 2, 2))
        bad = family_config("dflare")
        bad.attention_type = "mla"
        with self.assertRaisesRegex(ValueError, "GQA"):
            offline_family(bad)
        bad = family_config("dfly")
        bad.fusion_type = "gated_sum"
        with self.assertRaisesRegex(ValueError, "gated_sum"):
            offline_family(bad)
        with self.assertRaisesRegex(ValueError, "not MTP"):
            offline_family(SimpleNamespace())
        for family in ("dfly", "dflare"):
            self.assertEqual(offline_training_knobs(args, family_config(family))[0], family)

    def test_unset_dspark_knobs_follow_dflash(self):
        config = family_config("dspark")
        args = copy.deepcopy(self.args)
        args.dspark_num_anchors = args.dspark_num_target_layers = None
        args.dflash_num_anchors = 99
        self.assertEqual(offline_training_knobs(args, config), ("dspark", 2, 99))
        args.dflash_num_target_layers = 3
        with self.assertRaisesRegex(ValueError, "training.dflash_num_target_layers differs"):
            offline_training_knobs(args, config)

    def test_hy_v4_flash_dfly_config_runs_offline(self):
        path = Path(__file__).resolve().parents[1] / "angelspec/config"
        dfly = AutoDraftModelConfig.from_file(path / "dfly_hy_v4_flash_draft_config.json")
        self.assertEqual(offline_family(dfly), "dfly")

    def test_hy_v4_flash_dfly2_config_runs_offline_as_dspark(self):
        path = Path(__file__).resolve().parents[1] / "angelspec/config"
        dfly2 = AutoDraftModelConfig.from_file(path / "dfly2_hy_v4_flash_draft_config.json")
        self.assertEqual(offline_family(dfly2), "dspark")

    def test_confidence_only_dspark_requires_real_teacher_norm(self):
        args = copy.deepcopy(self.args)
        args.offline_hidden_size, args.offline_vocab_size = 16, 32
        args.offline_aux_layer_ids, args.offline_final_layer_id = [1], 3
        trainer = SimpleNamespace(
            args=args,
            dp_rank=0,
            dp_size=1,
            _distill_enabled=False,
            l1_loss_alpha=0,
            dflash=SimpleNamespace(_extra_distill_needed=lambda: True),
            target_lm_head=SimpleNamespace(norm=None),
        )
        with self.assertRaisesRegex(ValueError, "real target final norm"):
            attach_offline_data(trainer, self.dataset.identity)

    def test_dfly_selector_configuration_is_forwarded(self):
        args = copy.deepcopy(self.args)
        args.dflash_path_selector_loss_alpha = 0.3
        args.dflash_path_selector_teacher_labels = False
        args.dflash_path_selector_objective_weighting = False
        config = family_config("dfly")
        trainer = trainer_for(config, args)
        wrapper = trainer._build_training_wrapper(trainer._build_draft_model(config))
        self.assertEqual(wrapper.path_selector_loss_alpha, 0.3)
        self.assertFalse(wrapper.path_selector_teacher_labels)
        self.assertFalse(wrapper.path_selector_objective_weighting)


if __name__ == "__main__":
    unittest.main()
