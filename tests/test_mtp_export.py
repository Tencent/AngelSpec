import unittest

import torch

from tools.assemble_hy4_mtp_served import cast_like_target, remap_draft_to_mtp_layer
from tools.convert_mtp_to_hf import _remap_to_hf


class TestMTPExport(unittest.TestCase):
    def setUp(self):
        self.gate = torch.randn(4, 8, 16)
        self.up = torch.randn(4, 8, 16)
        self.down = torch.randn(4, 16, 8)

    def test_convert_accepts_wrapped_expert_names(self):
        draft = {
            "midlayer.mlp.experts_gate_proj.weight": self.gate,
            "midlayer.mlp.experts_up_proj.weight": self.up,
            "midlayer.mlp.experts_down_proj.weight": self.down,
        }
        result = _remap_to_hf(draft, spec_layer=3, dtype=torch.float32)
        self.assertEqual(len(result), 12)
        self.assertTrue(
            torch.equal(
                result["model.layers.3.mlp.experts.0.gate_proj.weight"],
                self.gate[0].t(),
            )
        )

    def test_assemble_accepts_wrapped_expert_names(self):
        draft = {
            "midlayer.mlp.experts_gate_proj.weight": self.gate,
            "midlayer.mlp.experts_up_proj.weight": self.up,
            "midlayer.mlp.experts_down_proj.weight": self.down,
        }
        result = remap_draft_to_mtp_layer(draft)
        expected_gate_up = torch.cat([self.gate.transpose(1, 2), self.up.transpose(1, 2)], dim=1)
        self.assertTrue(
            torch.equal(result["model.mtp_layers.0.mlp.experts.gate_up_proj"], expected_gate_up)
        )
        self.assertTrue(
            torch.equal(
                result["model.mtp_layers.0.mlp.experts.down_proj"],
                self.down.transpose(1, 2),
            )
        )

    def test_assemble_stores_each_tensor_in_the_target_dtype(self):
        import json
        import os
        import tempfile

        from safetensors.torch import save_file

        bias = "model.mtp_layers.0.mlp.e_score_correction_bias"
        weight = "model.mtp_layers.0.eh_proj.weight"
        with tempfile.TemporaryDirectory() as target:
            save_file(
                {bias: torch.zeros(4), weight: torch.zeros(2, 2, dtype=torch.bfloat16)},
                os.path.join(target, "model-1.safetensors"),
            )
            with open(os.path.join(target, "model.safetensors.index.json"), "w") as f:
                json.dump(
                    {"weight_map": {bias: "model-1.safetensors", weight: "model-1.safetensors"}}, f
                )
            head = {bias: torch.randn(4), weight: torch.randn(2, 2), "extra": torch.randn(3)}
            out = cast_like_target(head, target)
        self.assertEqual(out[bias].dtype, torch.float32)
        self.assertTrue(torch.equal(out[bias], head[bias]))
        self.assertEqual(out[weight].dtype, torch.bfloat16)
        self.assertEqual(out["extra"].dtype, torch.float32)
