"""Target KV reuse (``DFlashConfig.target_kv_layers``) and per-layer target candidates.

Covers:
1. DFlashConfig: off by default; per-layer lists round-trip; list lengths, captured
   inputs, MLA, no DSA indexer of the drafter's own and the plain DFlash drafter are
   enforced; a reusing drafter builds no context projection
2. TargetKVLatent vs an independent fp64 re-derivation of vLLM's hy4 MLA latent
   (``kv_a_layernorm(c)``, and ``k_pe`` rotated by vLLM's own GPT-J-style
   ``ApplyRotaryEmb`` then permuted from its interleaved cache layout to the split one),
   1D, packed and M-RoPE; equal to the drafter's own ``_project_kv_latent`` under the
   same weights; fp32 rotary tables under bf16 weights
3. Checkpoint loading by name, shape checks, non-MLA layers rejected; the
   target/drafter compatibility check
4. DFlashModel: a reusing drafter whose target latents equal the latents a KV-injection
   drafter projects reproduces that drafter's loss (dense, per-layer candidates,
   M-RoPE, packed); each draft layer gets its own target layer's latent and block mask;
   every trainable parameter gets a gradient; the modules must be attached and fit, and
   are never registered or saved; under USP only the readers of the whole target context
   (attention diagnostic / oracle) are rejected
5. tilelang backbone on CPU (kernel stubbed out): one index list per distinct target
   layer, built from that layer's candidates, and the latents passed per layer; the
   eval-time attention oracle / diagnostic hooks only eval forwards, in the chosen layers
6. CUDA + TileLang kernels (skipped without them): a per-layer reusing drafter on tilelang
   matches the same drafter on flex
7. Target attention init (``target_attn_init_layers``): each draft layer gets exactly its
   target layer's input_layernorm and attention tensors and nothing else, shapes and
   missing tensors are errors, and the copied layer then projects a hidden into the very
   latent its target layer caches for it
"""

import json
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.nn.attention.flex_attention import create_mask

from angelspec.data.positions import PositionRecipe, build_position_ids
from angelspec.models.dflash import (
    DFlashModel,
    _create_dflash_mask_mod,
    _create_dflash_sparse_kv_indices,
)
from angelspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDraftModel,
    target_layers_per_draft_layer,
)
from angelspec.models.ops.tilelang_dsa import (
    DSA_INDEX_ALIGNMENT,
    tilelang_dsa_available,
)
from angelspec.models.target.target_attention_init import (
    check_target_attn_init_compatible,
    init_attention_from_target,
    target_attention_keys,
)
from angelspec.models.target.target_indexer import (
    check_target_index_layer,
    load_target_config,
)
from angelspec.models.target.target_kv_latent import (
    TargetKVLatent,
    check_target_kv_compatible,
    check_target_kv_layer,
)

IMG_START, IMG_END, IMG_NL, IMG_PAD = 100, 101, 102, 103
RECIPE = PositionRecipe(
    image_start_token_id=IMG_START,
    image_end_token_id=IMG_END,
    image_newline_token_id=IMG_NL,
    image_token_id=IMG_PAD,
)


def hy_v4_flash_mrope_position_ids(input_ids, doc_ids=None):
    return build_position_ids(input_ids, RECIPE, doc_ids=doc_ids)


TXT = 5
H, V = 64, 128
RANK, ROPE = 16, 8


def _image(rows, cols):
    return [IMG_START] + ([IMG_PAD] * cols + [IMG_NL]) * rows + [IMG_END]


def _last_full_indexer_layers(text, count=5):
    """The last ``count`` target layers that run a DSA indexer of their own."""
    types = text.get("indexer_types") or ["full"] * text["num_hidden_layers"]
    return [layer for layer, kind in enumerate(types) if kind == "full"][-count:]


def _target_config(mrope=False, theta=10000.0, **overrides):
    config = {
        "hidden_size": H,
        "rms_norm_eps": 1e-6,
        "q_lora_rank": 32,
        "kv_lora_rank": RANK,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": ROPE,
        "index_n_heads": 4,
        "index_head_dim": 16,
        "index_topk": 8,
        "num_hidden_layers": 6,
        "layer_types": ["deepseek_sparse_attention"] * 6,
        "indexer_types": ["full", "full", "shared", "full", "shared", "full"],
        "rope_parameters": {"rope_theta": theta, "rope_type": "default"},
    }
    if mrope:
        config["rope_parameters"] = dict(
            config["rope_parameters"], mrope_section=[2, 1, 1], mrope_interleaved=True
        )
        config.update(
            image_start_token_id=IMG_START,
            image_end_token_id=IMG_END,
            image_newline_token_id=IMG_NL,
            image_token_id=IMG_PAD,
        )
    config.update(overrides)
    return config


def _draft_config(mrope=False, theta=10000.0, **kw):
    fields = dict(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=2,
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
        kv_lora_rank=RANK,
        qk_nope_head_dim=8,
        qk_rope_head_dim=ROPE,
        v_head_dim=12,
        gated_mla=True,
        learnable_sink=True,
        rope_parameters={"rope_theta": theta, "rope_type": "default"},
    )
    if mrope:
        fields["rope_parameters"] = dict(
            fields["rope_parameters"], mrope_section=[2, 1, 1], mrope_interleaved=True
        )
        fields.update(
            image_start_token_id=IMG_START,
            image_end_token_id=IMG_END,
            image_newline_token_id=IMG_NL,
        )
    fields.update(kw)
    return DFlashConfig(**fields)


def _random_latent(config, layer, seed=0):
    module = TargetKVLatent(config, layer)
    g = torch.Generator().manual_seed(seed)
    for param in module.parameters():
        param.data.copy_(torch.randn(param.shape, generator=g) * 0.5)
    return module


def _vllm_rope_available():
    try:
        from vllm.model_executor.layers.rotary_embedding.common import (  # noqa: F401
            ApplyRotaryEmb,
        )
        from vllm.model_executor.layers.rotary_embedding.mrope import (  # noqa: F401
            apply_interleaved_rope,
        )
    except Exception:
        return False
    return True


def _vllm_latent(module, config, hidden, positions):
    """``[L, kv_lora_rank + rope_dim]``, re-derived from vllm/models/hy_v4/nvidia/attention.py,
    then moved to the drafter's split RoPE layout.

    fp64: x = input_layernorm(h); c, k_pe = split(kv_a_proj_with_mqa(x));
    kv_c_normed = kv_a_layernorm(c); k_pe goes through vLLM's ApplyRotaryEmb with
    is_neox_style=False (pairs (2i, 2i+1) at frequency i), which leaves it interleaved
    as vLLM caches it; its [even | odd] columns are the split layout.
    ``positions`` is ``[L]`` or ``[3, L]`` (M-RoPE, rows interleaved as vLLM does).
    """
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    p = {name: t.detach().double() for name, t in module.named_parameters()}
    eps = config["rms_norm_eps"]
    rank, rope_dim = config["kv_lora_rank"], config["qk_rope_head_dim"]

    def rms(x, weight):
        return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    x = rms(hidden[0].double(), p["input_layernorm.weight"])
    c, k_pe = (x @ p["kv_a_proj_with_mqa.weight"].T).split([rank, rope_dim], dim=-1)

    rope = config["rope_parameters"]
    inv_freq = 1.0 / (
        rope["rope_theta"] ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim)
    )
    freqs = torch.outer(torch.arange(int(positions.max()) + 1, dtype=torch.float64), inv_freq)
    cos, sin = freqs.cos()[positions], freqs.sin()[positions]
    if positions.dim() == 2:
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            apply_interleaved_rope,
        )

        cos = apply_interleaved_rope(cos, rope["mrope_section"])
        sin = apply_interleaved_rope(sin, rope["mrope_section"])
    cached = ApplyRotaryEmb.forward_static(k_pe.unsqueeze(1), cos, sin, is_neox_style=False)
    cached = cached.squeeze(1)
    return torch.cat([rms(c, p["kv_a_layernorm.weight"]), cached[:, ::2], cached[:, 1::2]], -1)


class _FixedCandidates:
    """Stands in for TargetIndexer: one precomputed candidate table, as indices."""

    def __init__(self, table, layer, topk):
        self.table, self.layer, self.topk = table, layer, topk

    def candidate_indices(
        self,
        hidden,
        input_ids,
        anchor_positions,
        ctx_doc_ids=None,
        position_ids=None,
        sp_group=None,
    ):
        table = self.table.to(hidden.device)
        width = int(table.sum(-1).max())
        order = torch.argsort((~table).int(), dim=-1, stable=True)[..., :width]
        return torch.where(torch.gather(table, -1, order), order, torch.full_like(order, -1))


def _random_table(anchors, seq_len, topk, seed):
    g = torch.Generator().manual_seed(seed)
    table = torch.zeros(1, anchors.shape[1], seq_len, dtype=torch.bool)
    for j, anchor in enumerate(anchors[0].tolist()):
        table[0, j, torch.randperm(anchor, generator=g)[:topk]] = True
    return table


class TestDFlashConfigTargetKV(unittest.TestCase):
    def _config(self, **kw):
        fields = dict(
            hidden_size=64,
            num_hidden_layers=2,
            num_target_layers=2,
            target_hidden_size=64,
            target_num_hidden_layers=6,
            target_layer_ids=[3, 4],
            attention_type="mla",
            q_lora_rank=32,
        )
        fields.update(kw)
        return DFlashConfig(**fields)

    def test_off_by_default_and_round_trips(self):
        self.assertIsNone(self._config().target_kv_layers)
        config = self._config(
            target_index_layer=[4, 5], target_index_topk=8, target_kv_layers=[5, 4]
        )
        restored = DFlashConfig(
            **{k: v for k, v in config.to_dict().items() if k != "transformers_version"}
        )
        self.assertEqual(
            (restored.target_index_layer, restored.target_index_topk, restored.target_kv_layers),
            ([4, 5], 8, [5, 4]),
        )

    def test_an_int_means_every_draft_layer(self):
        self.assertEqual(self._config(target_kv_layers=5).target_kv_layers, 5)
        self.assertEqual(target_layers_per_draft_layer(5, 2), [5, 5])
        self.assertEqual(target_layers_per_draft_layer([4, 5], 2), [4, 5])
        self.assertIsNone(target_layers_per_draft_layer(None, 2))

    def test_lists_hold_one_captured_layer_per_draft_layer(self):
        for name, extra in (
            ("target_kv_layers", {}),
            ("target_index_layer", {"target_index_topk": 8}),
        ):
            with self.subTest(name=name):
                self._config(**{name: [4, 5]}, **extra)
                with self.assertRaisesRegex(
                    ValueError, "lists 1 target layers for 2 draft layers"
                ):
                    self._config(**{name: [5]}, **extra)
                with self.assertRaisesRegex(ValueError, "target_layer_id 5"):
                    self._config(**{name: [5, 6]}, **extra)
                with self.assertRaisesRegex(ValueError, "target_layer_id 2"):
                    self._config(**{name: [3, 5]}, **extra)
                for bad in (0, True, 5.0, "5"):
                    with self.assertRaises(ValueError, msg=repr(bad)):
                        self._config(**{name: [5, bad]}, **extra)

    def test_reuse_needs_mla_no_own_indexer_and_the_plain_drafter(self):
        with self.assertRaisesRegex(ValueError, "attention_type='mla'"):
            self._config(attention_type="gqa", target_kv_layers=[4, 5])
        with self.assertRaisesRegex(ValueError, "index_topk"):
            self._config(index_topk=8, target_kv_layers=[4, 5])
        with self.assertRaisesRegex(ValueError, "DFlash backbone"):
            self._config(model_arch="dflare", target_kv_layers=[4, 5])
        with self.assertRaisesRegex(ValueError, "DFlash backbone"):
            self._config(fusion_type="gated_sum", target_kv_layers=[4, 5])
        # Subclassed configs whose drafters build their own context path.
        from angelspec.models.draft.dfly import DFlyConfig
        from angelspec.models.draft.dspark import DSparkConfig

        fields = dict(
            hidden_size=64,
            num_hidden_layers=2,
            num_target_layers=2,
            target_hidden_size=64,
            target_num_hidden_layers=6,
            target_layer_ids=[3, 4],
            attention_type="mla",
            q_lora_rank=32,
        )
        for config_class in (DFlyConfig, DSparkConfig):
            with self.subTest(config=config_class.__name__):
                config_class(**fields)
                with self.assertRaisesRegex(ValueError, "DFlash backbone"):
                    config_class(target_kv_layers=[4, 5], **fields)
        # DSpark heads on the DFlash backbone take the target's latents like DFlash.
        DSparkConfig(model_arch="dflash", target_kv_layers=[4, 5], **fields)

    def test_per_layer_targets_need_the_plain_drafter_forward(self):
        class OwnForward(DFlashDraftModel):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)

        for kw in (
            dict(target_kv_layers=[4, 5]),
            dict(target_index_layer=[4, 5], target_index_topk=8),
        ):
            with self.subTest(**kw), self.assertRaisesRegex(ValueError, "overrides"):
                DFlashModel(draft_model=OwnForward(_draft_config(**kw)), block_size=4)
        # One target layer for every draft layer is the single-layer path it always had.
        DFlashModel(
            draft_model=OwnForward(_draft_config(target_index_layer=[5, 5], target_index_topk=8)),
            block_size=4,
        )

    def test_a_reusing_drafter_builds_no_context_projection(self):
        plain = DFlashDraftModel(_draft_config())
        reuse = DFlashDraftModel(_draft_config(target_kv_layers=[4, 5]))
        self.assertFalse(hasattr(reuse, "context_proj") or hasattr(reuse, "context_norm"))
        self.assertEqual(
            set(plain.state_dict()) - set(reuse.state_dict()),
            {"context_proj.weight", "context_norm.weight"},
        )


@unittest.skipUnless(_vllm_rope_available(), "vLLM's rotary helpers are unavailable")
class TestTargetKVLatentMatchesVllm(unittest.TestCase):
    def _check(self, config, tokens, positions, doc_ids=None):
        module = _random_latent(config, 5)
        hidden = torch.randn(1, len(tokens), H, generator=torch.Generator().manual_seed(3))
        got = module.latents(hidden, torch.tensor([tokens]), doc_ids)
        self.assertEqual(tuple(got.shape), (1, len(tokens), RANK + ROPE))
        expected = _vllm_latent(module, config, hidden, positions)
        torch.testing.assert_close(got[0].double(), expected, rtol=1e-5, atol=1e-4)

    def test_1d(self):
        self._check(_target_config(theta=10.0), list(range(1, 61)), torch.arange(60))

    def test_packed_positions_restart_per_document(self):
        doc_ids = torch.tensor([[0] * 25 + [1] * 35])
        positions = torch.cat([torch.arange(25), torch.arange(35)])
        self._check(_target_config(theta=10.0), list(range(1, 61)), positions, doc_ids)

    def test_mrope_with_an_image(self):
        # A low theta and a 4 x 6 image make the t/h/w column assignment matter.
        tokens = [TXT] * 20 + _image(4, 6) + [TXT] * 30
        positions = hy_v4_flash_mrope_position_ids(torch.tensor([tokens]))
        self._check(_target_config(mrope=True, theta=10.0), tokens, positions[0])


class TestTargetKVLatentMatchesTheDrafter(unittest.TestCase):
    def test_the_drafters_own_projection_under_the_same_weights(self):
        tokens = [TXT] * 20 + _image(4, 6) + [TXT] * 30
        input_ids = torch.tensor([tokens])
        hidden = torch.randn(1, len(tokens), H, generator=torch.Generator().manual_seed(4))
        for mrope in (False, True):
            with self.subTest(mrope=mrope):
                module = _random_latent(_target_config(mrope=mrope, theta=10.0), 5)
                attn = DFlashDraftModel(_draft_config(mrope=mrope, theta=10.0)).layers[0].self_attn
                with torch.no_grad():
                    attn.kv_a_proj_with_mqa.weight.copy_(module.kv_a_proj_with_mqa.weight)
                    attn.kv_a_layernorm.weight.copy_(module.kv_a_layernorm.weight)
                positions = (
                    hy_v4_flash_mrope_position_ids(input_ids)
                    if mrope
                    else torch.arange(len(tokens)).unsqueeze(0)
                )
                cos, sin = attn._cos_sin(hidden, positions, len(tokens))
                with torch.no_grad():
                    expected = attn._project_kv_latent(module.input_layernorm(hidden), cos, sin)
                torch.testing.assert_close(
                    module.latents(hidden, input_ids), expected, rtol=0, atol=1e-6
                )

    def test_rotary_tables_stay_fp32_under_bf16_weights(self):
        module = TargetKVLatent(_target_config(), 5)
        module._cast_weights(torch.bfloat16)
        self.assertEqual(module.kv_a_proj_with_mqa.weight.dtype, torch.bfloat16)
        self.assertEqual(module.rotary_emb.cos_cached.dtype, torch.float32)
        hidden = torch.randn(1, 12, H, dtype=torch.bfloat16)
        self.assertEqual(
            module.latents(hidden, torch.ones(1, 12, dtype=torch.long)).dtype, torch.bfloat16
        )


class TestNestedTargetRope(unittest.TestCase):
    """Hy4 flash nests rope_parameters per layer type, with different thetas: the latent and
    the checks must read the reused layer's own entry."""

    def _nested(self):
        config = _target_config(mrope=True, theta=10.0)
        config["layer_types"] = ["deepseek_sparse_attention"] * 6
        config["layer_types"][4] = "sliding_attention"
        sparse = config["rope_parameters"]
        config["rope_parameters"] = {
            "deepseek_sparse_attention": sparse,
            "sliding_attention": dict(sparse, rope_theta=1e4),
        }
        return config

    def test_the_latent_rotates_with_its_layers_entry(self):
        tokens = [TXT] * 20 + _image(4, 6) + [TXT] * 30
        input_ids = torch.tensor([tokens])
        hidden = torch.randn(1, len(tokens), H, generator=torch.Generator().manual_seed(4))
        flat = _random_latent(_target_config(mrope=True, theta=10.0), 5)
        nested = TargetKVLatent(self._nested(), 5)
        nested.load_state_dict(flat.state_dict())
        self.assertTrue(
            torch.equal(nested.latents(hidden, input_ids), flat.latents(hidden, input_ids))
        )

    def test_the_checks_read_the_reused_layers_entry(self):
        nested = self._nested()
        draft = _draft_config(mrope=True, theta=10.0)
        check_target_kv_layer(nested, 5)
        check_target_kv_compatible(nested, draft, layer=5)
        with self.assertRaisesRegex(ValueError, "nested per layer type"):
            check_target_kv_compatible(nested, draft)
        with self.assertRaisesRegex(ValueError, "rope_theta"):
            check_target_kv_compatible(nested, _draft_config(mrope=True, theta=1e4), layer=5)


class TestTargetKVLatentLoading(unittest.TestCase):
    SUFFIXES = {
        "input_layernorm.weight": "input_layernorm.weight",
        "kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
    }

    def _write(self, tmp, config, prefix="model.layers.5.", shapes=None, extra=None):
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f)
        ref = TargetKVLatent(config, 5)
        g = torch.Generator().manual_seed(7)
        tensors = {}
        for name, param in ref.named_parameters():
            shape = (shapes or {}).get(name, param.shape)
            tensors[prefix + self.SUFFIXES[name]] = torch.randn(shape, generator=g)
        tensors.update(extra or {})
        keys = sorted(tensors)
        shards = {"a.safetensors": keys[::2], "b.safetensors": keys[1::2]}
        weight_map = {}
        for file_name, shard_keys in shards.items():
            save_file(
                {k: tensors[k].contiguous() for k in shard_keys}, os.path.join(tmp, file_name)
            )
            weight_map.update({k: file_name for k in shard_keys})
        with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)
        return {name: tensors[prefix + suffix] for name, suffix in self.SUFFIXES.items()}

    def test_loads_its_three_tensors_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            decoy = {
                "model.mtp_layers.5.self_attn.kv_a_proj_with_mqa.weight": torch.zeros(24, 64),
                "model.layers.3.self_attn.kv_a_proj_with_mqa.weight": torch.zeros(24, 64),
            }
            expected = self._write(tmp, _target_config(), extra=decoy)
            module = TargetKVLatent.from_pretrained(tmp, 5, device="cpu", dtype=torch.float32)
            params = dict(module.named_parameters())
            self.assertEqual(set(params), set(expected))
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(params[name], tensor), name)
            self.assertFalse(any(p.requires_grad for p in module.parameters()))

    def test_nested_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = self._write(tmp, _target_config(), prefix="model.language_model.layers.5.")
            module = TargetKVLatent.from_pretrained(tmp, 5, device="cpu", dtype=torch.float32)
            self.assertTrue(
                torch.equal(
                    module.kv_a_proj_with_mqa.weight, expected["kv_a_proj_with_mqa.weight"]
                )
            )

    def test_shape_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, _target_config(), shapes={"kv_a_proj_with_mqa.weight": (20, 64)})
            with self.assertRaisesRegex(ValueError, "shape"):
                TargetKVLatent.from_pretrained(tmp, 5, device="cpu")

    def test_non_mla_layers_and_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "kv_lora_rank"):
            check_target_kv_layer(_target_config(kv_lora_rank=None), 5)
        with self.assertRaisesRegex(ValueError, "outside"):
            check_target_kv_layer(_target_config(), 6)
        linear = _target_config(
            layer_types=["deepseek_sparse_attention"] * 5 + ["linear_attention"]
        )
        with self.assertRaisesRegex(ValueError, "no MLA latent"):
            check_target_kv_layer(linear, 5)
        yarn = _target_config(
            rope_parameters={"rope_theta": 1e4, "rope_type": "yarn", "factor": 4}
        )
        with self.assertRaisesRegex(ValueError, "plain RoPE"):
            check_target_kv_layer(yarn, 5)
        with self.assertRaisesRegex(ValueError, "plain RoPE"):
            check_target_kv_layer(_target_config(rope_scaling={"type": "linear", "factor": 2}), 5)
        check_target_kv_layer(_target_config(mrope=True), 5)
        check_target_kv_layer(_target_config(rope_scaling={"type": "mrope"}), 5)

    def test_the_drafter_must_share_the_targets_latent_and_rope(self):
        check_target_kv_compatible(_target_config(), _draft_config())
        check_target_kv_compatible(_target_config(mrope=True), _draft_config(mrope=True))
        cases = {
            "kv_lora_rank": (_target_config(kv_lora_rank=32), _draft_config()),
            "rope_theta": (_target_config(theta=1e6), _draft_config()),
            "mrope_section": (_target_config(mrope=True), _draft_config()),
            "image_end_token_id": (
                _target_config(mrope=True, image_end_token_id=99),
                _draft_config(mrope=True),
            ),
        }
        for name, (target, draft) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                check_target_kv_compatible(target, draft)

    @unittest.skipUnless(
        os.environ.get("ANGELSPEC_HY4_TARGET"), "set ANGELSPEC_HY4_TARGET to a hy4 DSA target"
    )
    def test_real_hy4_checkpoint(self):
        path = os.environ["ANGELSPEC_HY4_TARGET"]
        config = load_target_config(path)
        text = config.get("text_config") or config
        layers = _last_full_indexer_layers(text)
        draft = DFlashConfig(
            hidden_size=text["hidden_size"],
            num_hidden_layers=5,
            num_target_layers=5,
            target_hidden_size=text["hidden_size"],
            target_num_hidden_layers=text["num_hidden_layers"],
            target_layer_ids=[layer - 1 for layer in layers],
            attention_type="mla",
            q_lora_rank=text["q_lora_rank"],
            kv_lora_rank=text["kv_lora_rank"],
            qk_nope_head_dim=text["qk_nope_head_dim"],
            qk_rope_head_dim=text["qk_rope_head_dim"],
            rope_parameters={
                "rope_theta": 10000000,
                "rope_type": "default",
                "mrope_section": [12, 10, 10],
                "mrope_interleaved": True,
            },
            image_start_token_id=120118,
            image_end_token_id=120119,
            image_newline_token_id=120121,
            target_index_layer=layers,
            target_index_topk=2048,
            target_kv_layers=layers,
        )
        check_target_kv_compatible(config, draft)
        for layer in layers:
            check_target_index_layer(config, layer)
            check_target_kv_layer(config, layer)
            module = TargetKVLatent.from_pretrained(path, layer, device="cpu")
            self.assertEqual(
                tuple(module.kv_a_proj_with_mqa.weight.shape),
                (text["kv_lora_rank"] + text["qk_rope_head_dim"], text["hidden_size"]),
            )
            self.assertEqual(tuple(module.kv_a_layernorm.weight.shape), (text["kv_lora_rank"],))
            recipe = module._position_recipe
            self.assertEqual(
                (
                    recipe.image_start_token_id,
                    recipe.image_end_token_id,
                    recipe.image_newline_token_id,
                ),
                (120118, 120119, 120121),
            )
            self.assertGreater(float(module.kv_a_layernorm.weight.float().abs().mean()), 0.0)


class TestDFlashModelTargetKV(unittest.TestCase):
    L = 40
    ANCHORS = torch.tensor([[6, 20, 31]])
    TOPK = 4
    KV_LAYERS = (4, 5)

    def _pair(self, index_layers=None, window=2, mrope=False):
        """A KV-injection wrapper and a reusing one with the same weights, whose target
        latents equal the injection drafter's own context latents when both captured
        hiddens are the same tensor ``h``: its context feature is then RMSNorm(h) (an
        identity projection of the first captured layer, unit norm weight), which is
        the target's input_layernorm under a unit weight."""
        kw = dict(context_local_window=window)
        if index_layers is not None:
            kw.update(target_index_layer=index_layers, target_index_topk=self.TOPK)
        torch.manual_seed(0)
        inject_draft = DFlashDraftModel(_draft_config(mrope=mrope, **kw)).float()
        inject_draft.freeze_embedding()
        with torch.no_grad():
            inject_draft.context_proj.weight.zero_()
            inject_draft.context_proj.weight[:, :H].copy_(torch.eye(H))
            inject_draft.context_norm.weight.fill_(1.0)
        reuse_draft = DFlashDraftModel(
            _draft_config(mrope=mrope, target_kv_layers=list(self.KV_LAYERS), **kw)
        ).float()
        reuse_draft.freeze_embedding()
        reuse_draft.load_state_dict(
            {
                k: v
                for k, v in inject_draft.state_dict().items()
                if not k.startswith(("context_proj.", "context_norm."))
            }
        )
        inject, reuse = (
            DFlashModel(draft_model=draft, block_size=4, num_anchors=3, causal_attn=True)
            for draft in (inject_draft, reuse_draft)
        )
        modules = {}
        for layer, draft_layer in zip(self.KV_LAYERS, inject_draft.layers):
            module = TargetKVLatent(_target_config(mrope=mrope), layer)
            with torch.no_grad():
                module.input_layernorm.weight.fill_(1.0)
                module.kv_a_proj_with_mqa.weight.copy_(
                    draft_layer.self_attn.kv_a_proj_with_mqa.weight
                )
                module.kv_a_layernorm.weight.copy_(draft_layer.self_attn.kv_a_layernorm.weight)
            modules[layer] = module
        reuse.attach_target_kv_latents(list(modules.values()))
        return inject, reuse, modules

    def _tables(self):
        return {
            layer: _random_table(self.ANCHORS, self.L, self.TOPK, seed=layer)
            for layer in self.KV_LAYERS
        }

    def _attach_candidates(self, wrapper, tables):
        layers = sorted(set(wrapper._target_index_layers))
        wrapper.attach_target_indexers(
            [_FixedCandidates(tables[layer], layer, self.TOPK) for layer in layers]
        )

    def _inputs(self, mrope=False, packed=False, same_hidden=True):
        g = torch.Generator().manual_seed(1)
        if mrope:
            # Spans 10..19, clear of every ANCHORS block: M-RoPE blocks must be text.
            image = _image(2, 3)
            input_ids = torch.tensor([[TXT] * 10 + image + [TXT] * (self.L - 10 - len(image))])
        else:
            input_ids = torch.randint(0, 100, (1, self.L), generator=g)
        hidden = [torch.randn(1, self.L, H, generator=g) for _ in range(2)]
        inputs = dict(
            input_ids=input_ids,
            hidden_states_list=[hidden[0], hidden[0]] if same_hidden else hidden,
            loss_mask=torch.ones(1, self.L),
            lm_head_weight=torch.randn(V, H, generator=g),
            injected_anchors=self.ANCHORS,
        )
        if packed:
            inputs["ctx_doc_ids"] = torch.tensor([[0] * 18 + [1] * (self.L - 18)])
            inputs["base_position_ids"] = torch.cat(
                [torch.arange(18), torch.arange(self.L - 18)]
            ).unsqueeze(0)
        if mrope:
            inputs["position_ids"] = hy_v4_flash_mrope_position_ids(
                input_ids, doc_ids=inputs.get("ctx_doc_ids")
            )
        return inputs

    def _loss(self, wrapper, **kw):
        with torch.no_grad():
            return wrapper(**self._inputs(**kw))[0]

    def test_reuse_reproduces_kv_injection_with_the_same_latents(self):
        cases = {
            "dense": dict(),
            "per-layer candidates": dict(index_layers=[4, 5]),
            "mrope": dict(mrope=True),
            "packed + per-layer candidates": dict(index_layers=[4, 5], packed=True),
        }
        for name, kw in cases.items():
            with self.subTest(case=name):
                index_layers = kw.get("index_layers")
                inject, reuse, _ = self._pair(
                    index_layers=index_layers, mrope=kw.get("mrope", False)
                )
                if index_layers is not None:
                    tables = self._tables()
                    self._attach_candidates(inject, tables)
                    self._attach_candidates(reuse, tables)
                inputs = dict(mrope=kw.get("mrope", False), packed=kw.get("packed", False))
                torch.testing.assert_close(
                    self._loss(reuse, **inputs), self._loss(inject, **inputs), rtol=1e-5, atol=1e-6
                )

    def test_per_layer_candidates_differ_from_shared_ones(self):
        tables = self._tables()
        losses = {}
        for index_layers in ([4, 5], [5, 5], [4, 4]):
            _, reuse, _ = self._pair(index_layers=index_layers)
            self._attach_candidates(reuse, tables)
            losses[tuple(index_layers)] = self._loss(reuse)
        self.assertFalse(torch.allclose(losses[(4, 5)], losses[(5, 5)]))
        self.assertFalse(torch.allclose(losses[(4, 5)], losses[(4, 4)]))

    def test_each_layer_gets_its_own_latent_and_block_mask(self):
        tables = self._tables()
        _, reuse, modules = self._pair(index_layers=[4, 5])
        self._attach_candidates(reuse, tables)
        seen = {}
        original = reuse.draft_model.forward

        def spy(**kwargs):
            seen.update(kwargs)
            return original(**kwargs)

        reuse.draft_model.forward = spy
        inputs = self._inputs(same_hidden=False)
        with torch.no_grad():
            reuse(**inputs)

        self.assertIsNone(seen["context_feature"])
        hidden = inputs["hidden_states_list"]
        latents = seen["context_latents"]
        # Target layer L reads captured id L - 1: target_layer_ids [3, 4] -> list index L - 4.
        for latent, layer in zip(latents, self.KV_LAYERS):
            expected = modules[layer].latents(hidden[layer - 4], inputs["input_ids"])
            self.assertTrue(torch.equal(latent, expected), layer)

        masks = seen["block_mask"]
        self.assertIsInstance(masks, list)
        draft_len = self.ANCHORS.shape[1] * 4
        grids = []
        for mask, layer in zip(masks, self.KV_LAYERS):
            expected_mod = _create_dflash_mask_mod(
                anchor_positions=self.ANCHORS,
                block_keep_mask=torch.ones_like(self.ANCHORS, dtype=torch.bool),
                ctx_len=self.L,
                block_size=4,
                causal=True,
                context_local_window=2,
                ctx_candidates=tables[layer],
            )
            got = create_mask(mask.mask_mod, 1, 1, draft_len, self.L + draft_len, device="cpu")
            expected = create_mask(expected_mod, 1, 1, draft_len, self.L + draft_len, device="cpu")
            self.assertTrue(torch.equal(got, expected), layer)
            grids.append(got)
        self.assertFalse(torch.equal(grids[0], grids[1]))

    def test_layers_reading_one_target_layer_share_one_mask(self):
        _, reuse, _ = self._pair(index_layers=[5, 5])
        self._attach_candidates(reuse, self._tables())
        seen = {}
        original = reuse.draft_model.forward

        def spy(**kwargs):
            seen.update(kwargs)
            return original(**kwargs)

        reuse.draft_model.forward = spy
        self._loss(reuse)
        self.assertNotIsInstance(seen["block_mask"], list)

    def test_every_trainable_parameter_gets_a_gradient(self):
        # REPLICATE wraps the drafter DDP-style, which rejects parameters left without one.
        _, reuse, _ = self._pair(index_layers=[4, 5])
        self._attach_candidates(reuse, self._tables())
        reuse(**self._inputs())[0].backward()
        trainable = [name for name, p in reuse.named_parameters() if p.requires_grad]
        self.assertGreater(len(trainable), 0)
        self.assertEqual(
            [name for name, p in reuse.named_parameters() if p.requires_grad and p.grad is None],
            [],
        )

    def test_the_latents_must_be_attached(self):
        draft = DFlashDraftModel(_draft_config(target_kv_layers=[4, 5])).float()
        wrapper = DFlashModel(draft_model=draft, block_size=4, num_anchors=3)
        with self.assertRaisesRegex(RuntimeError, "no TargetKVLatent is attached"):
            self._loss(wrapper)

    def test_mismatched_latents_are_rejected(self):
        draft = DFlashDraftModel(_draft_config(target_kv_layers=[4, 5])).float()
        wrapper = DFlashModel(draft_model=draft, block_size=4, num_anchors=3)
        with self.assertRaisesRegex(ValueError, "do not match"):
            wrapper.attach_target_kv_latents([TargetKVLatent(_target_config(), 5)])
        with self.assertRaisesRegex(ValueError, "do not match"):
            wrapper.attach_target_kv_latents(
                [TargetKVLatent(_target_config(), layer) for layer in (3, 4, 5)]
            )
        cases = {
            "kv_lora_rank": _target_config(kv_lora_rank=32),
            "rope_theta": _target_config(theta=1e6),
            "M-RoPE": _target_config(mrope=True),
        }
        for name, config in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                wrapper.attach_target_kv_latents(
                    [TargetKVLatent(config, layer) for layer in self.KV_LAYERS]
                )

    def test_a_bf16_drafter_still_fits_fp32_target_rope(self):
        draft = DFlashDraftModel(_draft_config(target_kv_layers=[4, 5])).to(torch.bfloat16)
        wrapper = DFlashModel(draft_model=draft, block_size=4, num_anchors=3)
        wrapper.attach_target_kv_latents(
            [TargetKVLatent(_target_config(), layer) for layer in self.KV_LAYERS]
        )

    def test_the_latents_are_neither_registered_nor_saved(self):
        _, reuse, modules = self._pair()
        module_params = {id(p) for m in modules.values() for p in m.parameters()}
        self.assertFalse(any(id(p) in module_params for p in reuse.parameters()))
        bare = DFlashModel(
            draft_model=DFlashDraftModel(_draft_config(target_kv_layers=[4, 5])),
            block_size=4,
            num_anchors=3,
        )
        self.assertEqual(set(reuse.state_dict()), set(bare.state_dict()))

    def test_usp_rejects_only_the_whole_context_readers(self):
        draft = DFlashDraftModel(_draft_config(target_kv_layers=[4, 5]))
        DFlashModel(draft_model=draft, block_size=4, num_anchors=3, sp_size=2)
        for knobs in (dict(attn_diag_blocks=2), dict(attn_oracle="target")):
            with self.assertRaisesRegex(NotImplementedError, "USP"):
                DFlashModel(draft_model=draft, block_size=4, num_anchors=3, sp_size=2, **knobs)


#: Target config fields the drafter's attention must share (see _draft_config).
_ATTN_FIELDS = dict(
    num_attention_heads=4,
    v_head_dim=12,
    gated_mla=True,
    gating_type="elementwise",
    learnable_sink=True,
)


class TestTargetAttentionInit(unittest.TestCase):
    """``target_attn_init_layers``: draft layer i's input_layernorm and attention start as
    target layer L_i's."""

    LAYERS = [4, 5]

    def _write_target(self, tmp, prefix="model.layers.", skip=(), shapes=None, draft_config=None):
        """A sharded target checkpoint: every attention tensor of target layers 4 and 5 (as
        ``draft_config``'s attention has them), plus decoys that must not be picked up.
        Returns ``{(layer, name): tensor}``."""
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(_target_config(**_ATTN_FIELDS), f)
        layer0 = DFlashDraftModel(draft_config or _draft_config()).layers[0]
        reference = dict(layer0.named_parameters())
        names = target_attention_keys(layer0)
        g = torch.Generator().manual_seed(11)
        tensors, expected = {}, {}
        for layer in self.LAYERS:
            for name in names:
                if name in skip:
                    continue
                shape = (shapes or {}).get(name, reference[name].shape)
                tensor = torch.randn(shape, generator=g) * 0.5 + (1.0 if len(shape) == 1 else 0.0)
                tensors[f"{prefix}{layer}.{name}"] = tensor
                expected[layer, name] = tensor
        tensors["model.mtp_layers.5.self_attn.kv_b_proj.weight"] = torch.zeros_like(
            reference["self_attn.kv_b_proj.weight"]
        )
        tensors[f"{prefix}5.mlp.down_proj.weight"] = torch.zeros(64, 256)
        keys = sorted(tensors)
        weight_map = {}
        for file_name, shard_keys in (("a.safetensors", keys[::2]), ("b.safetensors", keys[1::2])):
            save_file(
                {k: tensors[k].contiguous() for k in shard_keys}, os.path.join(tmp, file_name)
            )
            weight_map.update({k: file_name for k in shard_keys})
        with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)
        return expected

    def test_config(self):
        base = dict(target_kv_layers=[4, 5])
        self.assertIsNone(_draft_config().target_attn_init_layers)
        config = _draft_config(target_attn_init_layers=[5, 0], **base)
        restored = DFlashConfig(
            **{k: v for k, v in config.to_dict().items() if k != "transformers_version"}
        )
        self.assertEqual(restored.target_attn_init_layers, [5, 0])
        self.assertEqual(_draft_config(target_attn_init_layers=5).target_attn_init_layers, 5)
        with self.assertRaisesRegex(ValueError, "lists 1 target layers for 2 draft layers"):
            _draft_config(target_attn_init_layers=[5])
        for bad in (-1, 6, True, 5.0):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"\[0, 6\)"):
                _draft_config(target_attn_init_layers=[5, bad])
        with self.assertRaisesRegex(ValueError, "attention_type='mla'"):
            DFlashConfig(
                hidden_size=64,
                num_hidden_layers=2,
                target_num_hidden_layers=6,
                target_layer_ids=[3, 4],
                num_target_layers=2,
                target_attn_init_layers=[4, 5],
            )
        from angelspec.models.draft.dfly import DFlyConfig
        from angelspec.models.draft.dspark import DSparkConfig

        fields = dict(
            hidden_size=64,
            num_hidden_layers=2,
            num_target_layers=2,
            target_num_hidden_layers=6,
            target_layer_ids=[3, 4],
            attention_type="mla",
            q_lora_rank=32,
        )
        with self.assertRaisesRegex(ValueError, "DFlash backbone"):
            DFlyConfig(target_attn_init_layers=[4, 5], **fields)
        DSparkConfig(target_attn_init_layers=[4, 5], **fields)

    def test_each_draft_layer_gets_its_target_layers_attention_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = self._write_target(tmp)
            draft = DFlashDraftModel(_draft_config()).float()
            before = {k: v.clone() for k, v in draft.state_dict().items()}
            init_attention_from_target(draft, tmp, self.LAYERS)
        copied = set()
        for i, layer in enumerate(self.LAYERS):
            params = dict(draft.layers[i].named_parameters())
            names = {name for (src, name) in expected if src == layer}
            self.assertEqual(len(names), 10)  # 9 attention tensors + input_layernorm
            for name in names:
                self.assertTrue(torch.equal(params[name], expected[layer, name]), (i, name))
                copied.add(f"layers.{i}.{name}")
        for key, value in draft.state_dict().items():
            if key not in copied:
                self.assertTrue(torch.equal(value, before[key]), key)

    def test_nested_prefix_shape_and_missing_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = self._write_target(tmp, prefix="model.language_model.layers.")
            draft = DFlashDraftModel(_draft_config()).float()
            init_attention_from_target(draft, tmp, self.LAYERS)
            self.assertTrue(
                torch.equal(
                    draft.layers[1].self_attn.kv_b_proj.weight,
                    expected[5, "self_attn.kv_b_proj.weight"],
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            self._write_target(tmp, shapes={"self_attn.kv_b_proj.weight": (20, RANK)})
            with self.assertRaisesRegex(ValueError, "shape"):
                init_attention_from_target(DFlashDraftModel(_draft_config()), tmp, self.LAYERS)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_target(tmp, skip=("self_attn.linear_gate.weight",))
            with self.assertRaisesRegex(KeyError, "linear_gate"):
                init_attention_from_target(DFlashDraftModel(_draft_config()), tmp, self.LAYERS)
        with self.assertRaisesRegex(ValueError, "1 target layers for 2 draft layers"):
            init_attention_from_target(DFlashDraftModel(_draft_config()), "unused", [5])

    def test_block_latents_start_in_the_targets_latent_space(self):
        # After the copy, a draft layer projects a hidden into exactly the latent its target
        # layer caches for it, so context and block keys share one space.
        tokens = [TXT] * 20 + _image(4, 6) + [TXT] * 30
        input_ids = torch.tensor([tokens])
        hidden = torch.randn(1, len(tokens), H, generator=torch.Generator().manual_seed(4))
        for mrope in (False, True):
            with self.subTest(mrope=mrope), tempfile.TemporaryDirectory() as tmp:
                self._write_target(tmp)
                with open(os.path.join(tmp, "config.json"), "w") as f:
                    json.dump(_target_config(mrope=mrope, **_ATTN_FIELDS), f)
                draft = DFlashDraftModel(
                    _draft_config(
                        mrope=mrope,
                        target_kv_layers=self.LAYERS,
                        target_attn_init_layers=self.LAYERS,
                    )
                ).float()
                init_attention_from_target(draft, tmp, self.LAYERS)
                positions = (
                    hy_v4_flash_mrope_position_ids(input_ids)
                    if mrope
                    else torch.arange(len(tokens)).unsqueeze(0)
                )
                for draft_layer, layer in zip(draft.layers, self.LAYERS):
                    attn = draft_layer.self_attn
                    cos, sin = attn._cos_sin(hidden, positions, len(tokens))
                    with torch.no_grad():
                        own = attn._project_kv_latent(
                            draft_layer.input_layernorm(hidden), cos, sin
                        )
                    target = TargetKVLatent.from_pretrained(
                        tmp, layer, device="cpu", dtype=torch.float32
                    ).latents(hidden, input_ids)
                    torch.testing.assert_close(own, target, rtol=0, atol=1e-5)

    def test_a_post_proj_norm_is_copied_and_follows_o_proj(self):
        config = _draft_config(attn_post_proj_norm=True)
        with tempfile.TemporaryDirectory() as tmp:
            expected = self._write_target(tmp, draft_config=config)
            draft = DFlashDraftModel(config).float()
            init_attention_from_target(draft, tmp, self.LAYERS)
        for i, layer in enumerate(self.LAYERS):
            self.assertTrue(
                torch.equal(
                    draft.layers[i].self_attn.post_proj_norm.weight,
                    expected[layer, "self_attn.post_proj_norm.weight"],
                )
            )
        attn = draft.layers[0].self_attn
        plain = DFlashDraftModel(_draft_config()).float().layers[0].self_attn
        plain.load_state_dict(
            {k: v for k, v in attn.state_dict().items() if not k.startswith("post_proj_norm.")}
        )
        g = torch.Generator().manual_seed(3)
        out = torch.randn(1, 5, attn.num_heads, attn.v_head_dim, generator=g)
        hidden = torch.randn(1, 5, H, generator=g)
        with torch.no_grad():
            torch.testing.assert_close(
                attn._output_tail(out, hidden),
                attn.post_proj_norm(plain._output_tail(out, hidden)),
            )
        with self.assertRaisesRegex(ValueError, "attention_type='mla'"):
            _draft_config(attention_type="gqa", attn_post_proj_norm=True)

    def test_the_post_proj_norm_must_match_the_targets(self):
        # Hy4 flash ends each attention in post_proj_norm (sandwich_norm), which
        # dsa_skip_attn_post_norm drops from its DSA (deepseek_sparse_attention) layers.
        target = _target_config(sandwich_norm=True, **_ATTN_FIELDS)
        with_norm = _draft_config(attn_post_proj_norm=True)
        check_target_attn_init_compatible(target, with_norm, self.LAYERS)
        cases = {
            "target norm, drafter none": (target, _draft_config()),
            "drafter norm, target none": (_target_config(**_ATTN_FIELDS), with_norm),
            "DSA layers skip it": (dict(target, dsa_skip_attn_post_norm=True), with_norm),
        }
        for name, (config, draft_config) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "attn_post_proj_norm"):
                check_target_attn_init_compatible(config, draft_config, self.LAYERS)
        check_target_attn_init_compatible(
            dict(target, dsa_skip_attn_post_norm=True), _draft_config(), self.LAYERS
        )

    def test_the_drafter_must_have_the_targets_attention_shapes(self):
        check_target_attn_init_compatible(_target_config(**_ATTN_FIELDS), _draft_config())
        cases = {
            "num_attention_heads": dict(_ATTN_FIELDS, num_attention_heads=8),
            "learnable_sink": dict(_ATTN_FIELDS, learnable_sink=False),
            "gating_type": dict(_ATTN_FIELDS, gating_type="headwise"),
            "rms_norm_eps": dict(_ATTN_FIELDS, rms_norm_eps=1e-5),
        }
        for name, fields in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                check_target_attn_init_compatible(_target_config(**fields), _draft_config())

    @unittest.skipUnless(
        os.environ.get("ANGELSPEC_HY4_TARGET"), "set ANGELSPEC_HY4_TARGET to a hy4 DSA target"
    )
    def test_real_hy4_checkpoint(self):
        path = os.environ["ANGELSPEC_HY4_TARGET"]
        config = load_target_config(path)
        text = config.get("text_config") or config
        layers = _last_full_indexer_layers(text)
        draft_config = DFlashConfig(
            hidden_size=text["hidden_size"],
            intermediate_size=6912,
            num_hidden_layers=5,
            num_attention_heads=text["num_attention_heads"],
            vocab_size=1024,
            rms_norm_eps=text["rms_norm_eps"],
            max_position_embeddings=4096,
            num_target_layers=5,
            target_hidden_size=text["hidden_size"],
            target_num_hidden_layers=text["num_hidden_layers"],
            target_layer_ids=[layer - 1 for layer in layers],
            attention_type="mla",
            q_lora_rank=text["q_lora_rank"],
            kv_lora_rank=text["kv_lora_rank"],
            qk_nope_head_dim=text["qk_nope_head_dim"],
            qk_rope_head_dim=text["qk_rope_head_dim"],
            v_head_dim=text["v_head_dim"],
            gated_mla=True,
            learnable_sink=True,
            rope_parameters={
                "rope_theta": 10000000,
                "rope_type": "default",
                "mrope_section": [12, 10, 10],
                "mrope_interleaved": True,
            },
            image_start_token_id=120118,
            image_end_token_id=120119,
            image_newline_token_id=120121,
            target_kv_layers=layers,
            target_attn_init_layers=layers,
        )
        check_target_attn_init_compatible(config, draft_config)
        draft = DFlashDraftModel(draft_config).float()
        init_attention_from_target(draft, path, layers)
        hidden = (
            torch.randn(1, 64, text["hidden_size"], generator=torch.Generator().manual_seed(0)) * 2
        )
        input_ids = torch.full((1, 64), 1000)
        positions = build_position_ids(input_ids, PositionRecipe())
        for draft_layer, layer in zip(draft.layers, layers):
            attn = draft_layer.self_attn
            cos, sin = attn._cos_sin(hidden, positions, 64)
            with torch.no_grad():
                own = attn._project_kv_latent(draft_layer.input_layernorm(hidden), cos, sin)
            target = TargetKVLatent.from_pretrained(
                path, layer, device="cpu", dtype=torch.float32
            ).latents(hidden, input_ids)
            torch.testing.assert_close(own, target, rtol=0, atol=1e-4, msg=f"layer {layer}")


class TestSparseBackbonePerLayer(unittest.TestCase):
    """With every layer on tilelang, the candidates reach the kernel as one index list per
    distinct target layer; the drafter itself is stubbed out (no kernel on CPU)."""

    L, TOPK = 40, 4
    ANCHORS = torch.tensor([[6, 20, 31]])

    def test_one_index_list_per_distinct_target_layer(self):
        layers = [4, 5, 5]
        config = _draft_config(
            num_hidden_layers=3,
            kv_lora_rank=512,
            qk_nope_head_dim=16,
            qk_rope_head_dim=64,
            v_head_dim=16,
            context_local_window=2,
            target_index_layer=layers,
            target_index_topk=self.TOPK,
            target_kv_layers=layers,
            dsa_backend="tilelang",
        )
        draft = DFlashDraftModel(config).float()
        wrapper = DFlashModel(draft_model=draft, block_size=4, num_anchors=3, causal_attn=True)
        self.assertTrue(wrapper._sparse_kv_attention)
        indexers = {
            layer: _FixedCandidates(
                _random_table(self.ANCHORS, self.L, self.TOPK, seed=layer), layer, self.TOPK
            )
            for layer in (4, 5)
        }
        wrapper.attach_target_indexers(list(indexers.values()))
        target = _target_config(kv_lora_rank=512, qk_rope_head_dim=64)
        latents = {layer: _random_latent(target, layer, seed=layer) for layer in (4, 5)}
        wrapper.attach_target_kv_latents(list(latents.values()))

        seen = {}

        def fake_forward(**kwargs):
            seen.update(kwargs)
            return torch.zeros(1, kwargs["noise_embedding"].shape[1], H)

        draft.forward = fake_forward
        g = torch.Generator().manual_seed(1)
        input_ids = torch.randint(0, 100, (1, self.L), generator=g)
        hidden = [torch.randn(1, self.L, H, generator=g) for _ in range(2)]
        wrapper._draft_backbone(
            input_ids, hidden, torch.ones(1, self.L), injected_anchors=self.ANCHORS
        )

        self.assertIsNone(seen["block_mask"])
        self.assertIsNone(seen["context_feature"])
        rows, ctx = seen["sparse_kv_indices"], seen["context_latents"]
        self.assertEqual((len(rows), len(ctx)), (3, 3))
        self.assertIs(rows[1], rows[2])
        self.assertIs(ctx[1], ctx[2])
        width = -(-(self.TOPK + 2 + 4) // DSA_INDEX_ALIGNMENT) * DSA_INDEX_ALIGNMENT
        for got_rows, got_latent, layer in zip(rows, ctx, layers):
            expected = _create_dflash_sparse_kv_indices(
                anchor_positions=self.ANCHORS,
                block_keep_mask=torch.ones_like(self.ANCHORS, dtype=torch.bool),
                ctx_len=self.L,
                block_size=4,
                ctx_candidates=indexers[layer].candidate_indices(
                    hidden[0], input_ids, self.ANCHORS
                ),
                width=width,
                causal=True,
                context_local_window=2,
            )
            self.assertTrue(torch.equal(got_rows, expected), layer)
            self.assertTrue(
                torch.equal(got_latent, latents[layer].latents(hidden[layer - 4], input_ids)),
                layer,
            )
        self.assertFalse(torch.equal(rows[0], rows[1]))


class TestAttnProbeWiring(unittest.TestCase):
    """The eval-time attention oracle and diagnostic hook the drafter's attention modules in
    eval forwards only (the oracle in its chosen layers) and unhook them afterwards."""

    L, TOPK = 40, 4
    ANCHORS = torch.tensor([[6, 20, 31]])

    def _hooked_layers(self, training, **probe):
        from angelspec.models.target.target_attention_init import TargetAttentionTeacher

        config = _draft_config(
            kv_lora_rank=512,
            qk_nope_head_dim=16,
            qk_rope_head_dim=64,
            v_head_dim=16,
            context_local_window=2,
            target_index_layer=[4, 5],
            target_index_topk=self.TOPK,
            target_kv_layers=[4, 5],
            dsa_backend="tilelang",
        )
        draft = DFlashDraftModel(config).float()
        wrapper = DFlashModel(
            draft_model=draft, block_size=4, num_anchors=3, causal_attn=True, **probe
        )
        wrapper.attach_target_indexers(
            [
                _FixedCandidates(
                    _random_table(self.ANCHORS, self.L, self.TOPK, seed=layer), layer, self.TOPK
                )
                for layer in (4, 5)
            ]
        )
        target = _target_config(kv_lora_rank=512, qk_rope_head_dim=64)
        wrapper.attach_target_kv_latents(
            [_random_latent(target, layer, seed=layer) for layer in (4, 5)]
        )
        wrapper.attach_attn_diag_teachers(
            [TargetAttentionTeacher(config, layer) for layer in (4, 5)]
        )
        wrapper.train(training)

        hooked = []

        def fake_forward(**kwargs):
            hooked.extend(
                i for i, layer in enumerate(draft.layers) if layer.self_attn._forward_hooks
            )
            return torch.zeros(1, kwargs["noise_embedding"].shape[1], H)

        draft.forward = fake_forward
        g = torch.Generator().manual_seed(1)
        input_ids = torch.randint(0, 100, (1, self.L), generator=g)
        hidden = [torch.randn(1, self.L, H, generator=g) for _ in range(2)]
        wrapper._draft_backbone(
            input_ids, hidden, torch.ones(1, self.L), injected_anchors=self.ANCHORS
        )
        self.assertFalse(any(layer.self_attn._forward_hooks for layer in draft.layers))
        return hooked

    def test_probes_hook_eval_forwards_only(self):
        oracle_last = dict(attn_oracle="image", attn_oracle_layers=[1])
        self.assertEqual(self._hooked_layers(False, **oracle_last), [1])
        self.assertEqual(self._hooked_layers(False, attn_oracle="context"), [0, 1])
        self.assertEqual(self._hooked_layers(False, attn_diag_blocks=4), [0, 1])
        self.assertEqual(self._hooked_layers(True, attn_oracle="image"), [])
        self.assertEqual(self._hooked_layers(False), [])


@unittest.skipUnless(
    torch.cuda.is_available() and tilelang_dsa_available(), "needs CUDA and the TileLang kernels"
)
class TestTargetKVOnCuda(unittest.TestCase):
    """A per-layer reusing drafter on the TileLang kernel (index lists + target latents)
    matches the same drafter on flex (block masks), to the kernel's bf16 precision."""

    H, V, L, TOPK = 128, 128, 384, 16
    ANCHORS = torch.linspace(4, 376, 32).long().unsqueeze(0)
    SPLIT = 200

    def _latents(self):
        config = _target_config(hidden_size=self.H, kv_lora_rank=512, qk_rope_head_dim=64)
        modules = []
        for layer in (4, 5):
            module = TargetKVLatent(config, layer)
            g = torch.Generator().manual_seed(layer)
            with torch.no_grad():
                module.input_layernorm.weight.copy_(1 + 0.1 * torch.randn(self.H, generator=g))
                module.kv_a_proj_with_mqa.weight.copy_(
                    torch.randn(module.kv_a_proj_with_mqa.weight.shape, generator=g) * self.H**-0.5
                )
                module.kv_a_layernorm.weight.copy_(1 + 0.1 * torch.randn(512, generator=g))
            modules.append(module.cuda())
        return modules

    def _wrapper(self, backend, latents):
        torch.manual_seed(0)
        config = DFlashConfig(
            hidden_size=self.H,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=self.V,
            rms_norm_eps=1e-6,
            max_position_embeddings=1024,
            num_target_layers=2,
            target_hidden_size=self.H,
            target_num_hidden_layers=6,
            target_layer_ids=[3, 4],
            mask_token_id=self.V - 1,
            attention_type="mla",
            q_lora_rank=64,
            kv_lora_rank=512,
            qk_nope_head_dim=64,
            qk_rope_head_dim=64,
            v_head_dim=64,
            gated_mla=True,
            learnable_sink=True,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
            context_local_window=2,
            target_index_layer=[4, 5],
            target_index_topk=self.TOPK,
            target_kv_layers=[4, 5],
            dsa_backend=backend,
        )
        draft = DFlashDraftModel(config).to(torch.float32)
        draft.freeze_embedding()
        wrapper = DFlashModel(
            draft_model=draft, block_size=4, num_anchors=self.ANCHORS.shape[1], causal_attn=True
        ).cuda()
        wrapper.attach_target_indexers(
            [
                _FixedCandidates(
                    _random_table(self.ANCHORS, self.L, self.TOPK, seed=layer), layer, self.TOPK
                )
                for layer in (4, 5)
            ]
        )
        wrapper.attach_target_kv_latents(latents)
        return wrapper

    def _run(self, wrapper, packed):
        g = torch.Generator().manual_seed(1)
        inputs = dict(
            input_ids=torch.randint(0, 100, (1, self.L), generator=g),
            loss_mask=torch.ones(1, self.L),
            lm_head_weight=torch.randn(self.V, self.H, generator=g),
            injected_anchors=self.ANCHORS,
        )
        hidden = [torch.randn(1, self.L, self.H, generator=g).cuda() for _ in range(2)]
        if packed:
            rest = self.L - self.SPLIT
            inputs["ctx_doc_ids"] = torch.tensor([[0] * self.SPLIT + [1] * rest])
            inputs["base_position_ids"] = torch.cat(
                [torch.arange(self.SPLIT), torch.arange(rest)]
            ).unsqueeze(0)
        inputs = {name: t.cuda() for name, t in inputs.items()}
        loss = wrapper(hidden_states_list=hidden, **inputs)[0]
        loss.backward()
        grads = {
            n: p.grad.float().cpu() for n, p in wrapper.named_parameters() if p.grad is not None
        }
        return loss.detach().float().cpu(), grads

    def test_tilelang_matches_flex(self):
        latents = self._latents()
        for packed in (False, True):
            with self.subTest(packed=packed):
                flex = self._wrapper("torch", latents)
                with torch.no_grad():
                    for layer in flex.draft_model.layers:
                        layer.self_attn.learnable_sink_param.normal_()
                fast = self._wrapper("tilelang", latents)
                fast.load_state_dict(flex.state_dict())
                self.assertFalse(flex._sparse_kv_attention)
                self.assertTrue(fast._sparse_kv_attention)

                loss_flex, grads_flex = self._run(flex, packed)
                loss_fast, grads_fast = self._run(fast, packed)
                torch.testing.assert_close(loss_fast, loss_flex, rtol=1e-2, atol=1e-3)
                self.assertEqual(set(grads_fast), set(grads_flex))
                for name, ref in grads_flex.items():
                    a, b = grads_fast[name].flatten(), ref.flatten()
                    if b.norm() < 1e-10 and a.norm() < 1e-10:
                        continue
                    cos = F.cosine_similarity(a, b, dim=0).item()
                    rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
                    self.assertGreater(cos, 0.99, f"{name}: cos={cos:.5f}")
                    self.assertLess(rel, 5e-2, f"{name}: rel={rel:.4f}")


if __name__ == "__main__":
    unittest.main()
