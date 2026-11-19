"""Cross-layer context KV sharing (``DFlashConfig.context_kv_source_layers``).

Covers:
1. DFlashConfig: off by default; accepted maps round-trip; length, range, chains, MLA, no
   target_kv_layers, no DSA indexer of the drafter's own and the DFlash backbone are enforced
2. A lending layer returns exactly the context latent its own attention reads
3. DFlashDraftModel.forward hands the lender's latent to the layers that read it; an
   identity map leaves the drafter unchanged
4. DFlashModel: every trainable parameter still gets a gradient; USP and drafters that
   override forward are rejected
5. CUDA + TileLang kernels (skipped without them): a sharing drafter on tilelang matches
   the same drafter on flex
"""

import unittest

import torch

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from tests import test_dflash_target_index as target_index

H, V, L = 64, 128, 40


def _config(**kw):
    fields = dict(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=6,
        target_layer_ids=[3, 4],
        mask_token_id=V - 1,
        attention_type="mla",
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=12,
        gated_mla=True,
        learnable_sink=True,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    fields.update(kw)
    return DFlashConfig(**fields)


def _wrapper(draft_cls=DFlashDraftModel, sp_size=1, **kw):
    torch.manual_seed(0)
    draft = draft_cls(_config(**kw)).to(torch.float32)
    draft.freeze_embedding()
    return DFlashModel(draft_model=draft, block_size=4, num_anchors=3, sp_size=sp_size)


def _run(wrapper, backward=False):
    """Loss of one fixed batch, plus the kwargs and output of the drafter's forward."""
    seen = {}
    forward = wrapper.draft_model.forward

    def spy(**kwargs):
        seen["kwargs"] = kwargs
        seen["output"] = forward(**kwargs)
        return seen["output"]

    wrapper.draft_model.forward = spy
    g = torch.Generator().manual_seed(1)
    with torch.set_grad_enabled(backward):
        loss = wrapper(
            input_ids=torch.randint(0, 100, (1, L), generator=g),
            hidden_states_list=[torch.randn(1, L, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(1, L),
            lm_head_weight=torch.randn(V, H, generator=g),
            injected_anchors=torch.tensor([[6, 20, 31]]),
        )[0]
    if backward:
        loss.backward()
    return loss.detach(), seen


def _layer_kwargs(kwargs, index):
    block_mask = kwargs.get("block_mask")
    return dict(
        context_hidden=kwargs["context_feature"],
        draft_position_ids=kwargs["draft_position_ids"],
        context_position_ids=kwargs["context_position_ids"],
        block_mask=block_mask[index] if isinstance(block_mask, list) else block_mask,
        full_ctx_len=kwargs.get("full_ctx_len"),
        sparse_kv_indices=kwargs.get("sparse_kv_indices"),
    )


def _block_input(draft, kwargs):
    """The hidden the first draft layer reads, as DFlashDraftModel.forward builds it."""
    hidden = kwargs.get("noise_embedding")
    if hidden is None:
        hidden = draft.embed_tokens(kwargs["draft_input_ids"])
    return hidden.to(kwargs["context_feature"].dtype)


class TestContextKvSourceLayersConfig(unittest.TestCase):
    def test_off_by_default_and_round_trips(self):
        self.assertIsNone(_config().context_kv_source_layers)
        config = _config(context_kv_source_layers=(0, 0, 2))
        self.assertEqual(config.context_kv_source_layers, [0, 0, 2])
        again = DFlashConfig.from_dict(config.to_dict())
        self.assertEqual(again.context_kv_source_layers, [0, 0, 2])

    def test_rejected_maps(self):
        rejected = {
            "one draft layer per draft layer": dict(context_kv_source_layers=[0, 0]),
            "must list one": dict(context_kv_source_layers=0),
            r"\[2\] must be a draft layer in \[0, 2\], got 3": dict(
                context_kv_source_layers=[0, 1, 3]
            ),
            r"\[1\] must be a draft layer in \[0, 1\], got -1": dict(
                context_kv_source_layers=[0, -1, 0]
            ),
            r"got True": dict(context_kv_source_layers=[0, True, 0]),
            "does not project its own": dict(context_kv_source_layers=[0, 0, 1]),
            "attention_type='mla'": dict(context_kv_source_layers=[0, 0, 0], attention_type="gqa"),
            "drop context_kv_source_layers": dict(
                context_kv_source_layers=[0, 0, 0], target_kv_layers=4
            ),
            "drop index_topk": dict(context_kv_source_layers=[0, 0, 0], index_topk=4),
        }
        for message, kw in rejected.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _config(**kw)


class TestSharedContextLatents(unittest.TestCase):
    def test_a_lender_returns_the_latent_its_attention_reads(self):
        wrapper = _wrapper()
        _, seen = _run(wrapper)
        kwargs = seen["kwargs"]
        layer = wrapper.draft_model.layers[0]
        draft_hidden = _block_input(wrapper.draft_model, kwargs)
        layer_kwargs = dict(draft_hidden=draft_hidden, **_layer_kwargs(kwargs, 0))
        with torch.no_grad():
            plain = layer(**layer_kwargs)
            lent, latent = layer(**layer_kwargs, return_context_latent=True)
            given = layer(**layer_kwargs, context_latent=latent)
        self.assertEqual(latent.shape, (1, L, 16 + 8))
        self.assertTrue(torch.equal(lent, plain))
        self.assertTrue(torch.equal(given, plain))

    def test_later_layers_read_the_lenders_latent(self):
        wrapper = _wrapper(context_kv_source_layers=[0, 0, 0])
        _, seen = _run(wrapper)
        kwargs, draft = seen["kwargs"], wrapper.draft_model
        hidden = _block_input(draft, kwargs)
        with torch.no_grad():
            hidden, latent = draft.layers[0](
                draft_hidden=hidden, return_context_latent=True, **_layer_kwargs(kwargs, 0)
            )
            for index in (1, 2):
                hidden = draft.layers[index](
                    draft_hidden=hidden, context_latent=latent, **_layer_kwargs(kwargs, index)
                )
            expected = draft.final_norm(hidden)
        self.assertTrue(torch.equal(seen["output"], expected))

    def test_an_identity_map_changes_nothing_and_sharing_does(self):
        baseline, _ = _run(_wrapper())
        identity, _ = _run(_wrapper(context_kv_source_layers=[0, 1, 2]))
        shared, _ = _run(_wrapper(context_kv_source_layers=[0, 0, 0]))
        self.assertTrue(torch.equal(identity, baseline))
        self.assertFalse(torch.allclose(shared, baseline))

    def test_every_trainable_parameter_still_gets_a_gradient(self):
        def with_grad(wrapper):
            _run(wrapper, backward=True)
            named = list(wrapper.named_parameters())
            got = {name for name, p in named if p.grad is not None}
            return got, {name for name, p in named if p.requires_grad}

        shared_grads, trainable = with_grad(_wrapper(context_kv_source_layers=[0, 0, 0]))
        baseline_grads, _ = with_grad(_wrapper())
        self.assertEqual(shared_grads, baseline_grads)
        self.assertEqual(shared_grads, trainable)

    def test_usp_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "context_kv_source_layers"):
            _wrapper(context_kv_source_layers=[0, 0, 0], sp_size=2)

    def test_drafters_overriding_forward_are_rejected(self):
        class Overriding(DFlashDraftModel):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)

        with self.assertRaisesRegex(ValueError, "context_kv_source_layers"):
            _wrapper(draft_cls=Overriding, context_kv_source_layers=[0, 0, 0])


class TestSharedContextKvOnCuda(target_index.TestSparseKvAttentionOnCuda):
    """The tilelang-vs-flex comparison, for a three-layer drafter whose later layers read
    layer 0's context latent."""

    CONFIG = dict(num_hidden_layers=3, context_kv_source_layers=[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
