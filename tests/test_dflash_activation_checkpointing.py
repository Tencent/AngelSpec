import copy
import unittest

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.training.activation_checkpointing import (
    enable_dflash_activation_checkpointing,
)


class CountingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.calls = 0

    def forward(self, hidden, *, context):
        self.calls += 1
        return self.linear(hidden + context).sin()


class TestDFlashActivationCheckpointing(unittest.TestCase):
    def test_recomputes_with_frozen_inputs_and_preserves_state_keys(self):
        torch.manual_seed(7)
        reference = nn.Module()
        reference.layers = nn.ModuleList([CountingLayer(), CountingLayer()])
        checkpointed = copy.deepcopy(reference)
        keys = set(reference.state_dict())
        enable_dflash_activation_checkpointing(checkpointed)
        self.assertEqual(keys, set(checkpointed.state_dict()))
        checkpointed.load_state_dict(reference.state_dict(), strict=True)
        hidden, context = torch.randn(2, 8), torch.randn(2, 8)

        def run(model):
            result = hidden
            for layer in model.layers:
                result = layer(result, context=context)
            result.square().sum().backward()
            return result

        torch.testing.assert_close(run(reference), run(checkpointed))
        for plain, wrapped in zip(reference.layers, checkpointed.layers):
            self.assertEqual(plain.calls, 1)
            self.assertEqual(wrapped.calls, 2)
        for name, parameter in reference.named_parameters():
            actual = checkpointed.state_dict(keep_vars=True)[name]
            torch.testing.assert_close(parameter.grad, actual.grad)

    def test_idempotent_and_no_recompute_under_no_grad(self):
        model = nn.Module()
        model.layers = nn.ModuleList([CountingLayer()])
        enable_dflash_activation_checkpointing(model)
        first = model.layers[0]
        enable_dflash_activation_checkpointing(model)
        self.assertIs(first, model.layers[0])
        self.assertIsInstance(first, CheckpointWrapper)
        model.eval()
        with torch.no_grad():
            model.layers[0](torch.zeros(1, 8), context=torch.ones(1, 8))
        self.assertEqual(first.calls, 1)

    def test_mla_hc_tv_loss_and_gradients(self):
        torch.manual_seed(11)
        config = DFlashConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            target_hidden_size=32,
            target_num_hidden_layers=6,
            num_target_layers=2,
            mask_token_id=63,
            attention_type="mla",
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            gated_mla=True,
            learnable_sink=True,
            enable_hidden_correction=True,
            max_position_embeddings=64,
        )
        draft = DFlashDraftModel(config)
        draft.freeze_embedding()
        reference = DFlashModel(
            draft_model=draft,
            block_size=4,
            num_anchors=2,
            causal_attn=True,
            ce_loss_alpha=0.0,
            e2e_tv_loss_weight=1.0,
            e2e_tv_vocab_chunk=16,
        )
        checkpointed = copy.deepcopy(reference)
        enable_dflash_activation_checkpointing(checkpointed.draft_model)
        inputs = dict(
            input_ids=torch.randint(0, 63, (1, 16)),
            hidden_states_list=[torch.randn(1, 16, 32) for _ in range(2)],
            loss_mask=torch.ones(1, 16),
            last_hidden_states=torch.randn(1, 16, 32),
            lm_head_weight=torch.randn(64, 32),
            target_norm=nn.RMSNorm(32),
            injected_anchors=torch.tensor([[2, 8]]),
            injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        loss_ref = reference(**inputs)[0]
        loss_actual = checkpointed(**inputs)[0]
        torch.testing.assert_close(loss_ref, loss_actual)
        loss_ref.backward()
        loss_actual.backward()
        actual_state = checkpointed.state_dict(keep_vars=True)
        for name, parameter in reference.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(actual_state[name].grad)
                continue
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, actual_state[name].grad)
        for model in (reference, checkpointed):
            torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, checkpointed.state_dict()[name])
        restored = copy.deepcopy(reference)
        restored.load_state_dict(checkpointed.state_dict(), strict=True)
        reference.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(reference(**inputs)[0], restored(**inputs)[0])


if __name__ == "__main__":
    unittest.main()
