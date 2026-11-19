import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from angelspec.config.train_config import config_to_flat_args, load_config
from angelspec.data.offline import (
    HiddenDataset,
    HiddenWriter,
    OfflineBatchFetcher,
    atomic_json,
    read_json,
    validate_sample,
)
from angelspec.data.offline_assets import export_target_assets, mtp_init_prefix
from angelspec.data.utils import DataCollatorWithPadding, MTPPackingCollator
from angelspec.inference.online_dump import (
    OnlineDumpWriter,
    build_online_capture_recipe,
)
from angelspec.models.draft.auto import AutoDraftModelConfig
from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel
from angelspec.models.mtp import MTPModel
from angelspec.training.mtp_trainer import MTPTrainer, align_mtp_inputs
from angelspec.training.offline import (
    attach_offline_data,
    configure_model,
    evaluate_offline,
    run_offline_training,
    validate_offline_args,
)
from angelspec.training.offline_mtp import initialize_offline_mtp
from tests.offline_native import native_recipe, to_native


def tiny_config(**kwargs):
    return MTPConfig(
        hidden_size=16,
        target_hidden_size=16,
        vocab_size=32,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        mlp_type="dense",
        max_position_embeddings=512,
        **kwargs,
    )


class TestOfflineMTP(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = config_to_flat_args(
            load_config(
                cli_args=[
                    f"offline.manifest_path={self.root / 'hidden'}",
                    f"offline.frozen_target_path={self.root / 'frozen'}",
                    f"model.draft_model_config={self.root / 'draft.json'}",
                    f"output_dir={self.root / 'output'}",
                    "training.attention_backend=sdpa",
                    "training.fsdp_strategy=FULL_SHARD",
                    "training.training_num_nodes=1",
                    "training.training_num_gpus_per_node=2",
                    "training.micro_batch_size=1",
                    "training.ttt_length=3",
                    "training.max_seq_length=32",
                    "dataset.min_loss_tokens=5",
                    "training.prefetch_depth=0",
                ]
            )
        )
        self.target = {"hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 4}
        self.recipe = native_recipe([], 3, 16, 32, target_metadata_sha256={"test": "identity"})
        self.samples = [
            {
                "input_ids": torch.randint(0, 32, (1, 8)),
                "loss_mask": torch.ones(1, 8, dtype=torch.uint8),
                "attention_mask": torch.ones(1, 8, dtype=torch.bool),
                "last_hidden_states": torch.randn(1, 8, 16),
            }
            for _ in range(2)
        ]
        writer = HiddenWriter(self.args.offline_manifest_path, self.recipe)
        for index, sample in enumerate(self.samples):
            writer.write(str(index), to_native(sample, [], 3))
        writer.publish(["0", "1"])
        self.dataset = HiddenDataset(self.args.offline_manifest_path)

    def fetcher(self, packing=False, dataset=None):
        return OfflineBatchFetcher(
            dataset or self.dataset,
            rank=0,
            world_size=1,
            seed=0,
            shuffle=False,
            max_length=32,
            hidden_size=16,
            aux_layer_ids=[],
            final_layer_id=3,
            vocab_size=32,
            min_loss_tokens=5,
            packing=packing,
            model_family="mtp",
        )

    def test_last_hidden_only_roundtrip_and_native_verify(self):
        self.assertEqual(self.dataset.verify(), 2)
        self.assertEqual(self.dataset.layer_ids, [3])
        batch = next(self.fetcher())
        self.assertIsNone(batch["hidden_states"])
        torch.testing.assert_close(
            batch["last_hidden_states"][:, :8], self.samples[0]["last_hidden_states"]
        )
        broken = {**to_native(self.samples[0], [], 3), "hidden_layer_1": torch.zeros(8, 16)}
        with self.assertRaisesRegex(ValueError, "tensor keys"):
            validate_sample(broken, 8, self.recipe)

    def test_finite_scan_is_optional(self):
        tensors = to_native(self.samples[0], [], 3)
        tensors["hidden_layer_3"] = tensors["hidden_layer_3"].clone()
        tensors["hidden_layer_3"][2, 5] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite layer 3"):
            validate_sample(tensors, 8, self.recipe)
        validate_sample(tensors, 8, self.recipe, check_finite=False)

    def test_mtp_consumes_existing_dflash_features(self):
        root = self.root / "dflash"
        recipe = native_recipe([1], 3, 16, 32, target_metadata_sha256={"test": "identity"})
        writer = HiddenWriter(root, recipe)
        for i, sample in enumerate(self.samples):
            writer.write(
                str(i), to_native({**sample, "hidden_states": torch.randn(1, 8, 16)}, [1], 3)
            )
        writer.publish(["0", "1"])
        dataset = HiddenDataset(root)
        configure_model(self.args, dataset, tiny_config(), self.target)
        self.assertEqual(self.args.offline_aux_layer_ids, [])
        self.assertEqual(self.args.offline_final_layer_id, 3)
        loaded = dataset.load(dataset.records[0], [3])
        self.assertEqual(list(loaded["layers"]), [3])
        batch = next(self.fetcher(dataset=dataset))
        self.assertIsNone(batch["hidden_states"])
        torch.testing.assert_close(
            batch["last_hidden_states"][:, :8], self.samples[0]["last_hidden_states"]
        )

    def test_config_sp_packing_and_norm_contracts(self):
        validate_offline_args(self.args)
        configure_model(self.args, self.dataset, tiny_config(), self.target)
        self.assertEqual(self.args.offline_model_family, "mtp")
        self.assertTrue(self.args.last_hidden_states_prenorm)
        self.args.attention_backend = "usp"
        self.args.sp_ulysses_size = 2
        self.args.usp_local_shard = True
        self.args.mtp_packing = True
        self.args.dflash_num_anchors = 3
        validate_offline_args(self.args)
        configure_model(self.args, self.dataset, tiny_config(), self.target)
        self.assertEqual(self.args.dp_size, 1)
        self.args.dflash_packing = True
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_offline_args(self.args)

    def test_raw_alignment_shift_and_doc_seams(self):
        packed = next(self.fetcher(packing=True))
        expected = MTPPackingCollator(32)(self.samples)
        for key in (
            "input_ids",
            "loss_mask",
            "last_hidden_states",
            "ctx_doc_ids",
            "base_position_ids",
        ):
            torch.testing.assert_close(packed[key], expected[key])
        norm = torch.nn.RMSNorm(16).requires_grad_(False)
        ids, draft, teacher, mask = align_mtp_inputs(
            packed["input_ids"],
            packed["last_hidden_states"],
            packed["loss_mask"],
            verifier_norm=norm,
            ctx_doc_ids=packed["ctx_doc_ids"],
        )
        torch.testing.assert_close(ids[:, :-1], packed["input_ids"][:, 1:])
        torch.testing.assert_close(draft, packed["last_hidden_states"])
        torch.testing.assert_close(teacher[:, :-1], norm(packed["last_hidden_states"][:, 1:]))
        self.assertEqual(mask[0, 6:8].tolist(), [0, 0])
        self.assertEqual(mask[0, 14:].sum().item(), 0)
        _, postnorm, _, _ = align_mtp_inputs(
            packed["input_ids"],
            packed["last_hidden_states"],
            packed["loss_mask"],
            verifier_norm=norm,
            draft_input_postnorm=True,
            ctx_doc_ids=packed["ctx_doc_ids"],
        )
        torch.testing.assert_close(postnorm, norm(packed["last_hidden_states"]))

    def test_ttt_checkpoint_and_disk_gradient_parity(self):
        torch.manual_seed(13)
        head = torch.randn(32, 16)
        norm = torch.nn.RMSNorm(16).requires_grad_(False)
        for packing in (False, True):
            for on_policy in (False, True):
                with self.subTest(packing=packing, on_policy=on_policy):
                    disk = next(self.fetcher(packing=packing))
                    direct = (
                        MTPPackingCollator(32)(self.samples)
                        if packing
                        else DataCollatorWithPadding()(self.samples[:1])
                    )
                    draft = MTPDraftModel(tiny_config())
                    draft.freeze_embedding()
                    draft.set_lm_head_weight(head)
                    reference = MTPModel(
                        draft,
                        length=3,
                        on_policy=on_policy,
                        gradient_checkpointing=False,
                    )
                    actual = copy.deepcopy(reference)
                    actual.gradient_checkpointing = True

                    def forward(model, batch):
                        ids, hidden, teacher, mask = align_mtp_inputs(
                            batch["input_ids"],
                            batch["last_hidden_states"],
                            batch["loss_mask"],
                            verifier_norm=norm,
                            ctx_doc_ids=batch.get("ctx_doc_ids"),
                        )
                        result = model(
                            input_ids=ids,
                            hidden_states=hidden,
                            target_hidden_states=teacher,
                            target_lm_head_weight=head,
                            loss_mask=mask,
                            attention_mask=batch["attention_mask"],
                            ctx_doc_ids=batch.get("ctx_doc_ids"),
                            base_position_ids=batch.get("base_position_ids"),
                        )
                        return sum(result[0])

                    left, right = forward(reference, direct), forward(actual, disk)
                    torch.testing.assert_close(left, right)
                    left.backward()
                    right.backward()
                    for (name, p), (_, q) in zip(
                        reference.named_parameters(), actual.named_parameters()
                    ):
                        if p.requires_grad:
                            self.assertIsNotNone(p.grad, name)
                            torch.testing.assert_close(p.grad, q.grad)
                    for model in (reference, actual):
                        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
                    for name, value in reference.state_dict().items():
                        torch.testing.assert_close(value, actual.state_dict()[name])

    def test_mtp_capture_has_no_auxiliary_tensor(self):
        path = Path(self.args.draft_model_config)
        config = tiny_config()
        config.architectures = ["MTPDraftModel"]
        config.to_json_file(path)
        draft = AutoDraftModelConfig.from_file(path)
        recipe = build_online_capture_recipe(self.args, draft, self.target, {"test": "id"})
        self.assertEqual(recipe["capture_aux_layer_order"], [])
        self.assertEqual(recipe["layer_ids"], [3])
        self.assertEqual(recipe["layer_tensors"], {"3": "hidden_layer_3"})
        store = mock.Mock()
        input_ids = self.samples[0]["input_ids"].reshape(-1)
        store.get.return_value = SimpleNamespace(
            input_ids=input_ids,
            last_hidden_states=torch.randn(8, 16, dtype=torch.bfloat16),
        )
        store.get_into.side_effect = lambda key, tensors: tensors["input_ids"].copy_(input_ids)
        writer = OnlineDumpWriter(
            None,
            self.root / "captured",
            recipe,
            validation_fraction=0.5,
            seed=0,
            min_loss_tokens=1,
            max_pending_bytes=1 << 20,
            store=store,
        )
        result = {
            "data_id": "a",
            "mooncake_key": "key",
            "packed_loss_mask": "1,7",
            "tensor_shapes": {"input_ids": [8], "last_hidden_states": [8, 16]},
            "tensor_dtypes": {"input_ids": "int64", "last_hidden_states": "bfloat16"},
        }
        self.assertEqual(writer.stage_batch([result])["staged"], 1)
        writer.release([result])
        self.assertFalse(store.remove_tensors.call_args.kwargs["has_hidden_states"])
        writer.finalize()
        self.assertEqual(HiddenDataset(self.root / "captured").verify(), 1)

    def test_target_mtp_init_export_and_load(self):
        config = tiny_config(mtp_init_from_mtp_layer=0)
        draft = MTPDraftModel(config)
        source = self.root / "target"
        source.mkdir()
        atomic_json(source / "config.json", self.target)
        tensors = {
            "model.embed_tokens.weight": draft.embed_tokens.weight.detach().clone(),
            "lm_head.weight": torch.randn(32, 16),
            "model.norm.weight": torch.ones(16),
        }
        prefix = mtp_init_prefix(self.args, config)
        for key, value in draft.state_dict().items():
            if key == "embed_tokens.weight":
                continue
            key = key.removeprefix("midlayer.")
            tensors[prefix + key] = value.clone()
        save_file(tensors, str(source / "model.safetensors"))
        output = Path(self.args.offline_frozen_target_path)
        export_target_assets(source, output, init_prefix=prefix)
        self.assertEqual(read_json(output / "frozen_assets.json")["mtp_init_prefix"], prefix)
        restored = MTPDraftModel(config)
        trainer = object.__new__(MTPTrainer)
        trainer.args = self.args
        with mock.patch("torch.distributed.get_rank", return_value=0):
            initialize_offline_mtp(trainer, restored, config, str(output))
        for key, value in draft.state_dict().items():
            if key != "embed_tokens.weight":
                torch.testing.assert_close(value, restored.state_dict()[key])
        configure_model(self.args, self.dataset, config, self.target)
        with self.assertRaisesRegex(ValueError, "different MTP"):
            export_target_assets(source, output)

    def test_missing_mtp_init_and_missing_norm_fail(self):
        with self.assertRaises(FileNotFoundError):
            configure_model(
                self.args, self.dataset, tiny_config(mtp_init_from_mtp_layer=0), self.target
            )
        configure_model(self.args, self.dataset, tiny_config(), self.target)
        trainer = SimpleNamespace(args=self.args, target_lm_head=SimpleNamespace(norm=None))
        with self.assertRaisesRegex(ValueError, "real target final norm"):
            attach_offline_data(trainer, self.dataset.identity)

    def test_mtp_evaluation_uses_existing_trainer_alignment_and_metrics(self):
        args = copy.deepcopy(self.args)
        configure_model(args, self.dataset, tiny_config(), self.target)
        args.mtp_packing = True
        trainer = object.__new__(MTPTrainer)
        trainer.args = args
        trainer.dp_rank, trainer.dp_size = 0, 1
        head = torch.randn(32, 16)
        norm = torch.nn.RMSNorm(16).requires_grad_(False)
        trainer.target_lm_head = SimpleNamespace(norm=norm)
        trainer.target_lm_head_weight = head
        trainer.verifier_norm = norm
        draft = MTPDraftModel(tiny_config())
        draft.freeze_embedding()
        draft.set_lm_head_weight(head)
        trainer.model = MTPModel(draft, length=3)
        attach_offline_data(trainer, self.dataset.identity)
        trainer._offline_fetcher.dataset.records = [
            {**record, "split": "validation"} for record in self.dataset.records
        ]
        fork_rng = torch.random.fork_rng
        with (
            mock.patch.object(torch.Tensor, "cuda", lambda value, *a, **k: value),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.random.fork_rng", side_effect=lambda **k: fork_rng(devices=[])),
            mock.patch("torch.distributed.all_reduce"),
        ):
            metrics = evaluate_offline(trainer)
        self.assertTrue(trainer.model.training)
        self.assertIn("eval/ce_2", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval/avg_loss"])))

    def test_offline_mtp_driver_dispatches_without_dflash_knobs(self):
        args = copy.deepcopy(self.args)
        args.offline_eval_samples = 0
        args.mtp_packing = True
        args.num_train_steps = 1
        actors = [mock.Mock(), mock.Mock()]
        group = mock.Mock()
        group._actor_handlers = actors
        group.train_from_queue.return_value = [{"train/ce_0": 1.0}] * 2
        with (
            mock.patch(
                "angelspec.training.offline.verify_frozen_target", return_value=self.target
            ),
            mock.patch("angelspec.AutoDraftModelConfig.from_file", return_value=tiny_config()),
            mock.patch(
                "angelspec.ray.placement_group.create_placement_groups",
                return_value={"training": ("pg", [], [])},
            ),
            mock.patch("angelspec.ray.placement_group.allocate_train_group", return_value=group),
            mock.patch("ray.get", side_effect=lambda value: value),
            mock.patch("ray.kill"),
            mock.patch("ray.util.placement_group.remove_placement_group"),
        ):
            run_offline_training(args)
        effective = group.async_init.call_args.args[0]
        self.assertEqual(effective.offline_model_family, "mtp")
        self.assertTrue(effective.last_hidden_states_prenorm)
        self.assertTrue(effective.mtp_packing)
        group.train_from_queue.assert_called_once_with(0, 1)

    def test_native_verify_cli_accepts_zero_context_layers(self):
        from angelspec.offline_entry import main

        result = main(["verify", "--manifest", self.args.offline_manifest_path])
        self.assertEqual(result["verified"], 2)

    def test_full_warmstart_strict_loading_and_embedding_identity(self):
        config = tiny_config()
        draft = MTPDraftModel(config)
        checkpoint = self.root / "warm"
        checkpoint.mkdir()
        save_file(draft.state_dict(), str(checkpoint / "model.safetensors"))
        restored = copy.deepcopy(draft)
        restored.eh_proj.weight.data.zero_()
        trainer = object.__new__(MTPTrainer)
        trainer.args = copy.deepcopy(self.args)
        trainer.args.draft_init_path = str(checkpoint)
        with mock.patch("torch.distributed.get_rank", return_value=0):
            initialize_offline_mtp(trainer, restored, config, "/unused")
            torch.testing.assert_close(restored.eh_proj.weight, draft.eh_proj.weight)
            restored.embed_tokens.weight.data.add_(1)
            with self.assertRaisesRegex(ValueError, "embedding differs"):
                initialize_offline_mtp(trainer, restored, config, "/unused")


if __name__ == "__main__":
    unittest.main()
