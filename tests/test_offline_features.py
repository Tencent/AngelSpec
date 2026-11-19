import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from angelspec.data import offline
from angelspec.data.offline import (
    HiddenDataset,
    HiddenWriter,
    OfflineBatchFetcher,
    atomic_json,
    file_hash,
    object_hash,
    validate_tensors,
)
from angelspec.training.offline import (
    attach_offline_data,
    configure_model,
    run_offline_training,
    validate_offline_args,
    verify_frozen_target,
)
from tests.offline_native import native_recipe, to_native


def recipe():
    return native_recipe(
        [1],
        3,
        8,
        16,
        target_metadata_sha256={"config.json": "test-only"},
        tokenizer="test",
        mask_policy="provided",
    )


def legacy_recipe():
    return {
        "target_metadata_sha256": {"config.json": "test-only"},
        "last_hidden_states_pre_final_norm": True,
        "hidden_size": 8,
        "vocab_size": 16,
        "tokenizer": "test",
        "mask_policy": "provided",
    }


def native(tensors):
    return to_native(tensors, [1], 3)


def sample(token=1):
    return {
        "input_ids": torch.full((1, 8), token, dtype=torch.int64),
        "loss_mask": torch.ones(1, 8, dtype=torch.uint8),
        "attention_mask": torch.ones(1, 8, dtype=torch.bool),
        "hidden_states": torch.randn(1, 8, 8),
        "last_hidden_states": torch.randn(1, 8, 8),
    }


class TestOfflineFeatures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_dataset(self, count=3):
        writer = HiddenWriter(self.root, recipe())
        samples = [sample(i + 1) for i in range(count)]
        for i, tensors in enumerate(samples):
            writer.write(str(i), native(tensors))
        writer.publish([str(i) for i in range(count)])
        return HiddenDataset(self.root), samples

    def test_roundtrip_immutable_and_verify(self):
        dataset, samples = self.make_dataset()
        self.assertEqual(dataset.verify(), 3)
        self.assertEqual(dataset.layer_ids, [1, 3])
        record = dataset.records[0]
        self.assertEqual(record["layer_ids"], [1, 3])
        self.assertEqual(
            set(record["tensor_sha256"]),
            {"input_ids", "loss_mask", "hidden_layer_1", "hidden_layer_3"},
        )
        loaded = dataset.load(record)
        for key in ("input_ids", "loss_mask", "attention_mask"):
            torch.testing.assert_close(loaded[key], samples[0][key])
        torch.testing.assert_close(loaded["layers"][1], samples[0]["hidden_states"])
        torch.testing.assert_close(loaded["layers"][3], samples[0]["last_hidden_states"])
        self.assertEqual(list(dataset.load(record, [3])["layers"]), [3])
        with self.assertRaisesRegex(ValueError, "does not store layers"):
            dataset.load(record, [2])
        from safetensors import safe_open

        with safe_open(str(self.root / record["file"]), framework="pt") as reader:
            metadata = reader.metadata()
        self.assertEqual(metadata["layer_ids"], "1,3")
        self.assertEqual(metadata["layer_semantics"], "decoder_layer_output")
        with self.assertRaises(FileExistsError):
            HiddenWriter(self.root, recipe())

    def test_resume_validates_same_sample_and_recipe(self):
        writer = HiddenWriter(self.root, recipe())
        tensors = native(sample())
        first = writer.write("a", tensors)
        resumed = HiddenWriter(self.root, recipe())
        self.assertEqual(first, resumed.write("a", tensors))
        changed = copy.deepcopy(tensors)
        changed["hidden_layer_1"].add_(1)
        with self.assertRaisesRegex(ValueError, "different tensors"):
            resumed.write("a", changed)
        with self.assertRaisesRegex(ValueError, "different capture recipe"):
            HiddenWriter(self.root, {**recipe(), "tokenizer": "changed"})
        with self.assertRaises(FileNotFoundError):
            resumed.publish(["missing"])
        self.assertFalse((self.root / "manifest.json").exists())

    def test_corruption_is_rejected(self):
        dataset, _ = self.make_dataset()
        path = self.root / dataset.records[0]["file"]
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            dataset.load(dataset.records[0])

    def test_bad_tensors_and_traversal_rejected(self):
        writer = HiddenWriter(self.root, recipe())
        tensors = native(sample())
        tensors["hidden_layer_1"][0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            writer.write("a", tensors)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            writer.write("../a", native(sample()))
        unnamed = native(sample())
        unnamed["hidden_states"] = unnamed.pop("hidden_layer_1")
        with self.assertRaisesRegex(ValueError, "Expected tensor keys"):
            writer.write("b", unnamed)
        tensors = sample()
        tensors["loss_mask"][0, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_tensors(tensors, 8, 8, 1, 16)
        dataset, _ = self.make_dataset()
        manifest = dataset.manifest
        manifest["records"][0]["file"] = "../escape.safetensors"
        atomic_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "escapes"):
            HiddenDataset(self.root)

    def test_manifest_and_recipe_validation(self):
        dataset, _ = self.make_dataset()
        manifest = copy.deepcopy(dataset.manifest)
        manifest["complete"] = False
        atomic_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            HiddenDataset(self.root)
        manifest["complete"] = True
        manifest["recipe"]["hidden_size"] = 9
        atomic_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "recipe hash"):
            HiddenDataset(self.root)

    def fetcher(self, dataset, rank=0, world=1, max_length=512):
        return OfflineBatchFetcher(
            dataset,
            rank=rank,
            world_size=world,
            seed=42,
            max_length=max_length,
            hidden_size=8,
            aux_layer_ids=[1],
            final_layer_id=3,
            vocab_size=16,
            min_loss_tokens=4,
        )

    def test_schedule_rank_sharding_seek_and_padding(self):
        dataset, _ = self.make_dataset()
        serial = self.fetcher(dataset)
        expected = [int(next(serial)["input_ids"][0, 0]) for _ in range(12)]
        ranks = [self.fetcher(dataset, rank=r, world=2) for r in range(2)]
        actual = [int(next(ranks[i % 2])["input_ids"][0, 0]) for i in range(12)]
        self.assertEqual(actual, expected)
        ranks[0].seek(6)
        self.assertEqual(int(next(ranks[0])["input_ids"][0, 0]), expected[6])
        batch = next(serial)
        self.assertEqual(batch["input_ids"].shape, (1, 256))
        self.assertEqual(int(batch["attention_mask"].sum()), 8)
        with self.assertRaisesRegex(ValueError, "do not crop"):
            next(self.fetcher(dataset, max_length=8))

    def test_dfly_adapter_checks_pilot_and_receipts(self):
        dataset, samples = self.make_dataset()
        manifest = dataset.manifest
        manifest.pop("format")
        pilot_root = self.root / "pilot"
        pilot_recipe = {"source": "test"}
        pilot = {
            "complete": True,
            "recipe": pilot_recipe,
            "recipe_sha256": object_hash(pilot_recipe),
            "records": manifest["records"],
        }
        atomic_json(pilot_root / "manifest.json", pilot)
        capture = {
            **legacy_recipe(),
            "pilot_identity": "recipe-v1",
            "pilot_recipe_sha256": pilot["recipe_sha256"],
            "capture_post_layer_ids": [1, 3],
        }
        capture_hash = object_hash(capture)
        from safetensors.torch import save_file

        for record, tensors in zip(manifest["records"], samples):
            path = self.root / record["file"]
            save_file(
                tensors,
                str(path),
                metadata={
                    "offline": json.dumps(
                        {
                            "id": record["id"],
                            "recipe_sha256": capture_hash,
                            "seq_len": record["seq_len"],
                            "split": record["split"],
                            "target_layers": [1, 3],
                            "last_hidden_states_pre_final_norm": True,
                        }
                    )
                },
            )
            record.update(
                recipe_sha256=capture_hash, bytes=path.stat().st_size, sha256=file_hash(path)
            )
            atomic_json(self.root / "receipts" / f"{record['id']}.json", record)
        manifest.update(recipe=capture, recipe_sha256=capture_hash)
        atomic_json(self.root / "manifest.json", manifest)
        atomic_json(
            self.root / "validation.json",
            {
                "complete": True,
                "status": "passed",
                "recipe_sha256": capture_hash,
                "records": manifest["records"],
            },
        )
        imported = HiddenDataset(self.root, tokenized_path=pilot_root)
        self.assertEqual(imported.verify(8, 1, 16), 3)
        with self.assertRaisesRegex(ValueError, "Bind"):
            imported.load(imported.records[0])
        imported.bind_final_layer(3)
        self.assertEqual(imported.layer_ids, [1, 3])
        loaded = imported.load(imported.records[0])
        torch.testing.assert_close(loaded["layers"][1], samples[0]["hidden_states"])
        torch.testing.assert_close(loaded["layers"][3], samples[0]["last_hidden_states"])
        with self.assertRaisesRegex(ValueError, "requires"):
            HiddenDataset(self.root)
        pilot["records"][0]["split"] = "validation"
        atomic_json(pilot_root / "manifest.json", pilot)
        with self.assertRaisesRegex(ValueError, "split/length"):
            HiddenDataset(self.root, tokenized_path=pilot_root)

    def test_config_defaults_and_offline_guardrails(self):
        from angelspec.config.train_config import config_to_flat_args, load_config

        config = load_config(
            cli_args=[
                "offline.manifest_path=/tmp/features",
                "offline.frozen_target_path=/tmp/frozen",
                "training.attention_backend=flex_attention",
                "training.fsdp_strategy=FULL_SHARD",
                "training.micro_batch_size=1",
                "training.training_num_gpus_per_node=2",
                "model.draft_model_config=/tmp/draft.json",
                "output_dir=/tmp/offline-test",
            ]
        )
        args = config_to_flat_args(config)
        validate_offline_args(args)
        args.sp_ulysses_size = 8
        with self.assertRaisesRegex(ValueError, "SP1"):
            validate_offline_args(args)
        args.sp_ulysses_size = 1
        args.load_path = "/tmp/checkpoint"
        with self.assertRaisesRegex(ValueError, "resume"):
            validate_offline_args(args)

    def test_model_contract_and_frozen_identity(self):
        from angelspec.models.draft.dflash import DFlashConfig

        dataset, _ = self.make_dataset()
        args = SimpleNamespace(
            dflash_num_target_layers=2,
            dflash_block_size=2,
            dflash_num_anchors=2,
            min_loss_tokens=4,
            max_seq_length=512,
            offline_frozen_target_path="/tmp/frozen",
        )
        config = DFlashConfig(
            hidden_size=8,
            target_hidden_size=8,
            vocab_size=16,
            num_target_layers=2,
            target_layer_ids=[1, 3],
            target_num_hidden_layers=4,
        )
        target = {"hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 4}
        configure_model(args, dataset, config, target)
        self.assertTrue(args.dflash_final_layer_from_last_hidden)
        self.assertEqual(args.offline_context_layers, 1)
        self.assertEqual((args.offline_aux_layer_ids, args.offline_final_layer_id), ([1], 3))
        config.target_layer_ids = [0, 3]
        with self.assertRaisesRegex(ValueError, r"lacks target layers \[0\]"):
            configure_model(args, dataset, config, target)

    def test_offline_entry_does_not_enter_online_setup(self):
        from angelspec.train_entry import run_training

        args = SimpleNamespace(offline_manifest_path="/tmp/features")
        with mock.patch("angelspec.training.offline.run_offline_training", return_value=7) as run:
            self.assertEqual(run_training(args), 7)
            run.assert_called_once_with(args)

    def test_frozen_component_integrity(self):
        from safetensors.torch import save_file

        atomic_json(self.root / "config.json", {"hidden_size": 8})
        keys = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
        save_file({key: torch.randn(8) for key in keys}, str(self.root / "frozen.safetensors"))
        atomic_json(
            self.root / "model.safetensors.index.json",
            {
                "weight_map": {key: "frozen.safetensors" for key in keys},
            },
        )
        files = {p.name: file_hash(p) for p in self.root.iterdir()}
        expected = {"source": "test"}
        atomic_json(
            self.root / "frozen_assets.json",
            {
                "frozen_components_only": True,
                "source_target_metadata_sha256": expected,
                "bundle_files_sha256": files,
            },
        )
        self.assertEqual(verify_frozen_target(self.root, expected), {"hidden_size": 8})
        with self.assertRaisesRegex(ValueError, "identity"):
            verify_frozen_target(self.root, {"source": "other"})
        (self.root / "frozen.safetensors").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_frozen_target(self.root, expected)

    def test_disk_and_direct_batches_have_equal_tv_gradients(self):
        from angelspec.data.utils import DataCollatorWithPadding
        from angelspec.models.dflash import DFlashModel
        from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel

        torch.manual_seed(9)
        dataset, samples = self.make_dataset(count=1)
        direct = DataCollatorWithPadding(usp_enabled=False)(samples)
        disk = next(self.fetcher(dataset))
        config = DFlashConfig(
            hidden_size=8,
            target_hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=16,
            mask_token_id=15,
            num_target_layers=2,
            target_num_hidden_layers=4,
            target_layer_ids=[1, 3],
            max_position_embeddings=512,
            attention_type="mla",
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            enable_hidden_correction=True,
        )
        draft = DFlashDraftModel(config)
        draft.freeze_embedding()
        reference = DFlashModel(
            draft_model=draft,
            block_size=2,
            num_anchors=2,
            causal_attn=True,
            ce_loss_alpha=0.0,
            e2e_tv_loss_weight=1.0,
        )
        actual = copy.deepcopy(reference)
        head = torch.randn(16, 8)
        norm = torch.nn.RMSNorm(8)
        norm.requires_grad_(False)

        def forward(model, batch):
            return model(
                input_ids=batch["input_ids"],
                loss_mask=batch["loss_mask"],
                hidden_states_list=[batch["hidden_states"], batch["last_hidden_states"]],
                last_hidden_states=batch["last_hidden_states"],
                target_norm=norm,
                lm_head_weight=head,
                injected_anchors=torch.tensor([[1, 4]]),
                injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
            )[0]

        loss = forward(reference, direct)
        actual_loss = forward(actual, disk)
        torch.testing.assert_close(actual_loss, loss)
        loss.backward()
        actual_loss.backward()
        for (name, expected), (_, parameter) in zip(
            reference.named_parameters(), actual.named_parameters()
        ):
            if expected.requires_grad:
                self.assertIsNotNone(expected.grad, name)
                torch.testing.assert_close(parameter.grad, expected.grad)

    def test_partial_sample_never_publishes(self):
        writer = HiddenWriter(self.root, recipe())
        with mock.patch("angelspec.data.offline.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                writer.write("a", native(sample()))
        self.assertFalse((self.root / "dumps/a.safetensors").exists())
        self.assertFalse((self.root / "receipts/a.json").exists())
        self.assertEqual(list((self.root / "dumps").glob(".partial-*")), [])
        with self.assertRaises(FileNotFoundError):
            writer.publish(["a"])

    def test_finite_check_covers_every_storage_dtype_and_chunk(self):
        rows = offline._FINITE_CHECK_ROWS + 3
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            finfo = torch.finfo(dtype)
            tensor = torch.randn(rows, 4).to(dtype)
            tensor[0] = torch.tensor([finfo.max, -finfo.max, finfo.tiny / 2, -0.0])
            self.assertTrue(offline._all_finite(tensor))
            self.assertTrue(offline._all_finite(tensor.t()))
            for value in (float("inf"), float("-inf"), float("nan")):
                for row in (1, rows - 1):
                    bad = tensor.clone()
                    bad[row, 2] = value
                    self.assertFalse(offline._all_finite(bad), (dtype, value, row))
                    self.assertFalse(offline._all_finite(bad.t()), (dtype, value, row))
        nan64 = torch.tensor([[1.0, float("nan")]], dtype=torch.float64)
        self.assertFalse(offline._all_finite(nan64))
        self.assertTrue(offline._all_finite(torch.empty(0, 4, dtype=torch.bfloat16)))

    def test_receipts_are_checked_in_threads_in_sample_order(self):
        writer = HiddenWriter(self.root, recipe())
        ids = [f"s{index}" for index in range(5)]
        for index, sample_id in enumerate(ids):
            writer.write(sample_id, native(sample(index + 1)))
        with mock.patch.dict(offline._CHECK_THREADS, {"header": 3, "full": 2}):
            records = writer.committed_records(verify="header")
            self.assertEqual([record["id"] for record in records], ids)
            self.assertEqual(writer.committed_records(verify="full"), records)
            for record in (records[3], records[1]):
                (self.root / record["file"]).write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "size mismatch: s1"):
                writer.committed_records(verify="header")
            with self.assertRaisesRegex(ValueError, "size mismatch: s1"):
                writer.publish(ids, verify="header")
        self.assertFalse((self.root / "manifest.json").exists())

    def test_atomic_json_creates_its_directory(self):
        atomic_json(self.root / "nested" / "value.json", {"a": 1})
        self.assertEqual(json.loads((self.root / "nested" / "value.json").read_text()), {"a": 1})
        self.assertEqual(list((self.root / "nested").glob(".partial-*")), [])

    def test_prefetched_data_attaches_without_mooncake(self):
        dataset, _ = self.make_dataset()
        args = SimpleNamespace(
            offline_manifest_path=str(self.root),
            offline_format="native",
            offline_verify_hashes=True,
            offline_tokenized_path=None,
            seed=7,
            max_seq_length=512,
            offline_hidden_size=8,
            offline_aux_layer_ids=[1],
            offline_final_layer_id=3,
            offline_vocab_size=16,
            min_loss_tokens=4,
            shuffle_dataset=False,
            prefetch_depth=1,
        )
        trainer = SimpleNamespace(args=args, dp_rank=1, dp_size=2, _distill_enabled=False)
        self.assertEqual(attach_offline_data(trainer, dataset.identity), 3)
        try:
            batch = next(iter(trainer.data_fetcher))
            self.assertEqual(int(batch["input_ids"][0, 0]), 2)
        finally:
            trainer.data_fetcher.close()
            trainer.data_fetcher._thread.join(timeout=5)
        with self.assertRaisesRegex(ValueError, "changed"):
            attach_offline_data(trainer, "wrong-manifest")

    def test_training_orchestration_only_allocates_training_actors(self):
        from angelspec.config.train_config import config_to_flat_args, load_config
        from angelspec.models.draft.dflash import DFlashConfig

        dataset, _ = self.make_dataset()
        config = load_config(
            cli_args=[
                f"offline.manifest_path={self.root}",
                "offline.frozen_target_path=/tmp/frozen",
                "offline.eval_samples=0",
                "training.attention_backend=flex_attention",
                "training.fsdp_strategy=FULL_SHARD",
                "training.training_num_gpus_per_node=2",
                "training.micro_batch_size=1",
                "training.num_train_steps=1",
                "training.max_seq_length=512",
                "training.dflash_num_target_layers=2",
                "training.dflash_block_size=2",
                "dataset.min_loss_tokens=4",
                "model.draft_model_config=/tmp/draft.json",
                f"output_dir={self.root / 'output'}",
            ]
        )
        args = config_to_flat_args(config)
        draft = DFlashConfig(
            hidden_size=8,
            target_hidden_size=8,
            vocab_size=16,
            num_target_layers=2,
            target_layer_ids=[1, 3],
            target_num_hidden_layers=4,
        )
        actors = [mock.Mock(), mock.Mock()]
        group = mock.Mock()
        group._actor_handlers = actors
        group.train_from_queue.return_value = [{"train/loss": 1.0}, {"train/loss": 1.0}]
        target = {"hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 4}
        with (
            mock.patch("angelspec.training.offline.verify_frozen_target", return_value=target),
            mock.patch("angelspec.AutoDraftModelConfig.from_file", return_value=draft),
            mock.patch(
                "angelspec.ray.placement_group.create_placement_groups",
                return_value={"training": ("pg", [0, 1], [0, 1])},
            ) as placement,
            mock.patch("angelspec.ray.placement_group.allocate_train_group", return_value=group),
            mock.patch("ray.get", side_effect=lambda value: value),
            mock.patch("ray.kill") as kill,
            mock.patch("ray.util.placement_group.remove_placement_group") as remove,
        ):
            run_offline_training(args)
        effective = placement.call_args.args[0]
        self.assertTrue(effective.debug_train_only)
        self.assertEqual(effective.inference_num_gpus, 0)
        self.assertEqual(effective.global_batch_size, 2)
        self.assertEqual(effective.target_model_path, "/tmp/frozen")
        group.async_init.assert_called_once_with(effective, role="training", mooncake_config=None)
        group.train_from_queue.assert_called_once_with(0, 1)
        group.save_model.assert_called_once_with(0, force_sync=True)
        for actor in actors:
            actor.set_offline_data.remote.assert_called_once_with(dataset.identity)
            actor.close.remote.assert_called_once()
        self.assertEqual(kill.call_count, 2)
        remove.assert_called_once_with("pg")
        self.assertTrue((self.root / "output/offline_complete.json").exists())


if __name__ == "__main__":
    unittest.main()
