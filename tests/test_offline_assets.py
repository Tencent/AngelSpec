import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from angelspec.data.offline import atomic_json, file_hash, read_json
from angelspec.data.offline_assets import export_target_assets, read_target_components


class TestOfflineAssets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.target.mkdir()
        atomic_json(
            self.target / "config.json",
            {
                "model_type": "qwen3",
                "hidden_size": 8,
                "vocab_size": 16,
                "num_hidden_layers": 4,
            },
        )
        atomic_json(self.target / "tokenizer_config.json", {"test": True})
        atomic_json(self.target / "tokenizer.json", {"test": True})
        save_file(
            {
                "model.embed_tokens.weight": torch.randn(16, 8),
                "lm_head.weight": torch.randn(16, 8),
                "model.norm.weight": torch.ones(8),
                "model.layers.0.weight": torch.randn(8, 8),
            },
            str(self.target / "model.safetensors"),
        )
        self.frozen = self.root / "frozen"

    def test_export_assets_roundtrip_and_identity_guard(self):
        identity = export_target_assets(self.target, self.frozen)
        self.assertEqual(export_target_assets(self.target, self.frozen), identity)
        tensors = load_file(str(self.frozen / "frozen.safetensors"))
        self.assertEqual(
            set(tensors),
            {
                "model.embed_tokens.weight",
                "lm_head.weight",
                "model.norm.weight",
            },
        )
        (self.target / "tokenizer_config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "identity"):
            export_target_assets(self.target, self.frozen)
        with self.assertRaisesRegex(ValueError, "full target"):
            read_target_components(self.frozen)

    def test_engram_vocab_map_ships_without_changing_identity(self):
        identity = export_target_assets(self.target, self.frozen)
        (self.target / "engram_vocab_map.npz").write_bytes(b"map")
        self.assertEqual(export_target_assets(self.target, self.frozen), identity)
        self.assertEqual((self.frozen / "engram_vocab_map.npz").read_bytes(), b"map")
        fresh = self.root / "fresh"
        self.assertEqual(export_target_assets(self.target, fresh), identity)
        files = read_json(fresh / "frozen_assets.json")["bundle_files_sha256"]
        self.assertEqual(files["engram_vocab_map.npz"], file_hash(fresh / "engram_vocab_map.npz"))

    def test_fp8_mtp_init_weights_ship_as_stored_and_nan_is_rejected(self):
        def write(weight):
            save_file(
                {
                    "model.embed_tokens.weight": torch.randn(16, 8),
                    "lm_head.weight": torch.randn(16, 8),
                    "model.norm.weight": torch.ones(8),
                    "model.layers.0.weight": weight,
                    "model.layers.0.weight_scale_inv": torch.ones(1, 1),
                },
                str(self.target / "model.safetensors"),
            )

        weight = torch.randn(8, 8).to(torch.float8_e4m3fn)
        write(weight)
        export_target_assets(self.target, self.frozen, init_prefix="model.layers.0.")
        tensors = load_file(str(self.frozen / "frozen.safetensors"))
        self.assertEqual(tensors["model.layers.0.weight"].dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(tensors["model.layers.0.weight"].float(), weight.float()))
        nan = weight.clone()
        nan.view(torch.uint8)[0, 0] = 0x7F
        write(nan)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            export_target_assets(self.target, self.root / "nan", init_prefix="model.layers.0.")

    def test_missing_target_component_fails_without_publication(self):
        save_file(
            {"model.embed_tokens.weight": torch.ones(16, 8)},
            str(self.target / "model.safetensors"),
        )
        with self.assertRaises(SafetensorError):
            export_target_assets(self.target, self.frozen)
        self.assertFalse(self.frozen.exists())

    def test_export_atomic_failure_leaves_no_bundle(self):
        with mock.patch(
            "angelspec.data.offline_assets.save_file", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                export_target_assets(self.target, self.frozen)
        self.assertFalse(self.frozen.exists())
        self.assertEqual(list(self.root.glob(".frozen-partial-*")), [])

    def test_cli_exports_assets_and_rejects_removed_capture_commands(self):
        from angelspec.offline_entry import main

        path = self.root / "config.yaml"
        path.write_text(f"model:\n  target_model_path: {self.target}\n")
        result = main(
            ["export-assets", "--config", str(path), f"offline.frozen_target_path={self.frozen}"]
        )
        self.assertEqual(result["identity"], export_target_assets(self.target, self.frozen))
        with self.assertRaises(SystemExit):
            main(["export-assets", "--config", str(path), "--unknown"])
        for removed in ("prepare", "dump"):
            with self.assertRaises(SystemExit):
                main([removed, "--config", str(path)])


if __name__ == "__main__":
    unittest.main()
