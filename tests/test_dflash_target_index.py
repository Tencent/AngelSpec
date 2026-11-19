"""Context candidates from the target's DSA indexer (``DFlashConfig.target_index_layer``).

Covers:
1. DFlashConfig: off by default, set together, positive, and the layer's input must be a
   captured target layer below the last one
2. TargetIndexer vs an independent fp64 re-derivation of vLLM's hy4 ``Indexer.forward``
   (1D and M-RoPE positions): the same top-k for every anchor, with and without vLLM's
   fp8 q/k rounding (itself checked against an explicit e4m3 rounding); prepared M-RoPE
   positions are read as given
3. candidate_table: only [doc_start, anchor), everything when the budget covers it,
   nothing without context; fp32 rotary tables under bf16 weights
4. Checkpoint loading by name, shape checks, shared layers and non-DSA targets rejected
5. Mask: window and candidates are unioned, never at or past the anchor, draft keys and
   packed documents untouched, the window alone unchanged
6. DFlashModel: a budget covering the context reproduces the dense loss, a small one does
   not; an M-RoPE drafter's positions reach the indexer; the indexer must be attached, is
   never registered or saved
7. CUDA (skipped without a GPU): flex attention with the candidate table matches the CPU
   materialized mask in loss and gradients, packed or not
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
    windows_per_draft_layer,
)
from angelspec.models.draft.mtp import _fp8_round_trip
from angelspec.models.ops.mrope import MRotaryEmbedding
from angelspec.models.ops.tilelang_dsa import tilelang_dsa_available
from angelspec.models.target.target_indexer import (
    TargetIndexer,
    candidates_to_table,
    check_target_index_layer,
    document_starts,
    load_target_config,
)

IMG_START, IMG_END, IMG_NL, IMG_PAD = 100, 101, 102, 103
TXT = 5
INDEX_LAYER = 5
RECIPE = PositionRecipe(
    image_start_token_id=IMG_START,
    image_end_token_id=IMG_END,
    image_newline_token_id=IMG_NL,
    image_token_id=IMG_PAD,
)


def _image(rows, cols):
    return [IMG_START] + ([IMG_PAD] * cols + [IMG_NL]) * rows + [IMG_END]


def _target_config(mrope=True, theta=10000.0, **overrides):
    config = {
        "hidden_size": 64,
        "rms_norm_eps": 1e-5,
        "q_lora_rank": 32,
        "index_n_heads": 4,
        "index_head_dim": 16,
        "qk_rope_head_dim": 8,
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


def _random_indexer(config, topk=8, seed=0):
    indexer = TargetIndexer(config, INDEX_LAYER, topk)
    g = torch.Generator().manual_seed(seed)
    for param in indexer.parameters():
        param.data.copy_(torch.randn(param.shape, generator=g) * 0.5)
    return indexer


def _fp8_reference(x):
    """e4m3fn rounding written out (3 mantissa bits, subnormal below 2**-6, ties to even)
    after one power-of-two scale per last-dim vector (vLLM's ue8m0 q/k quantization)."""
    x = x.double()
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-10)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    y = x / scale
    step = torch.exp2(torch.floor(torch.log2(y.abs().clamp(min=2.0**-6))) - 3)
    return torch.round(y / step) * step * scale


def _vllm_indexer_logits(indexer, config, hidden, positions, fp8=False):
    """``[L, L]`` index scores, re-derived from vllm/models/hy_v4/nvidia/attention.py.

    fp64: x = input_layernorm(h); q_c = q_a_layernorm(q_a_proj(x)); q = wq_b(q_c) and
    k = k_norm(wk(x)), whose last qk_rope_head_dim dims are rotated GPT-J style
    (is_neox_style=False); ``fp8`` then rounds q and k like vLLM's quantization;
    weights = weights_proj(x) * head_dim**-0.5 * n_heads**-0.5;
    score = sum_h weights_h * relu(q_h . k).
    ``positions`` is ``[L]`` or ``[3, L]`` (M-RoPE, rows interleaved as vLLM does).
    """
    p = {name: t.detach().double() for name, t in indexer.named_parameters()}
    eps = config["rms_norm_eps"]
    heads, head_dim = config["index_n_heads"], config["index_head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    nope = head_dim - rope_dim

    def rms(x, weight):
        return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    x = rms(hidden[0].double(), p["input_layernorm.weight"])
    q_c = rms(x @ p["q_a_proj.weight"].T, p["q_a_layernorm.weight"])
    q = (q_c @ p["indexer.wq_b.weight"].T).view(-1, heads, head_dim)
    k = F.layer_norm(
        x @ p["indexer.wk.weight"].T,
        (head_dim,),
        p["indexer.k_norm.weight"],
        p["indexer.k_norm.bias"],
        eps,
    )

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

    def rotate(t, c, s):
        t1, t2 = t[..., ::2], t[..., 1::2]
        return torch.stack((t1 * c - t2 * s, t2 * c + t1 * s), dim=-1).flatten(-2)

    q = torch.cat([q[..., :nope], rotate(q[..., nope:], cos[:, None], sin[:, None])], dim=-1)
    k = torch.cat([k[..., :nope], rotate(k[..., nope:], cos, sin)], dim=-1)
    if fp8:
        q, k = _fp8_reference(q), _fp8_reference(k)
    weights = (x @ p["indexer.weights_proj.weight"].T) * head_dim**-0.5 * heads**-0.5
    return (torch.einsum("shd,td->sht", q, k).relu() * weights[..., None]).sum(1)


class TestDFlashConfigTargetIndex(unittest.TestCase):
    def _config(self, **kw):
        fields = dict(
            hidden_size=64,
            num_target_layers=2,
            target_hidden_size=64,
            target_num_hidden_layers=6,
            target_layer_ids=[3, 4],
        )
        fields.update(kw)
        return DFlashConfig(**fields)

    def test_off_by_default_and_round_trips(self):
        self.assertIsNone(self._config().target_index_layer)
        config = self._config(target_index_layer=5, target_index_topk=2048)
        restored = DFlashConfig(
            **{k: v for k, v in config.to_dict().items() if k != "transformers_version"}
        )
        self.assertEqual((restored.target_index_layer, restored.target_index_topk), (5, 2048))

    def test_set_together_and_positive(self):
        with self.assertRaisesRegex(ValueError, "together"):
            self._config(target_index_layer=5)
        with self.assertRaisesRegex(ValueError, "together"):
            self._config(target_index_topk=8)
        for bad in (0, -1, True, 8.0, "8"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self._config(target_index_layer=5, target_index_topk=bad)

    def test_layer_input_must_be_captured_and_below_the_last_layer(self):
        self._config(target_index_layer=4, target_index_topk=8)  # input id 3
        with self.assertRaisesRegex(ValueError, "target_layer_id 5"):
            self._config(target_index_layer=6, target_index_topk=8)
        with self.assertRaisesRegex(ValueError, "target_layer_id 2"):
            self._config(target_index_layer=3, target_index_topk=8)
        # id 5 is the target's last layer: it arrives after the head, not as an input.
        with self.assertRaisesRegex(ValueError, "last layer"):
            self._config(target_layer_ids=[3, 5], target_index_layer=6, target_index_topk=8)


class TestFp8RoundTrip(unittest.TestCase):
    def test_matches_an_explicit_e4m3_rounding(self):
        g = torch.Generator().manual_seed(5)
        x = torch.randn(512, 128, generator=g) * torch.logspace(-4, 4, 512).unsqueeze(1)
        self.assertTrue(torch.equal(_fp8_round_trip(x).double(), _fp8_reference(x)))

    def test_ties_round_to_even(self):
        # amax 448 gives scale 1. 17 and 19 sit halfway between e4m3 neighbours 16|18 and
        # 18|20, 0.53125 between 0.5|0.5625, and 3 * 2**-10 between the subnormals
        # 2**-9|2**-8: each goes to the even mantissa.
        x = torch.tensor([[448.0, 17.0, 19.0, 0.53125, 3 * 2.0**-10]])
        expected = [[448.0, 16.0, 20.0, 0.5, 2.0**-8]]
        self.assertEqual(_fp8_round_trip(x).tolist(), expected)
        self.assertEqual(_fp8_reference(x).tolist(), expected)


class TestTargetIndexerMatchesVllm(unittest.TestCase):
    def _check(self, config, tokens, anchors, topk=8, use_fp8=True):
        indexer = _random_indexer(config, topk=topk)
        indexer.use_fp8 = use_fp8
        g = torch.Generator().manual_seed(3)
        hidden = torch.randn(1, len(tokens), config["hidden_size"], generator=g)
        input_ids = torch.tensor([tokens])
        table = indexer.candidate_table(hidden, input_ids, torch.tensor([anchors]))

        if "mrope_section" in config["rope_parameters"]:
            positions = build_position_ids(input_ids, RECIPE)[0]
        else:
            positions = torch.arange(len(tokens))
        logits = _vllm_indexer_logits(indexer, config, hidden, positions, fp8=use_fp8)

        compared = 0
        for j, anchor in enumerate(anchors):
            row = logits[anchor - 1, :anchor]
            keep = min(topk, anchor)
            vals, order = row.sort(descending=True)
            if anchor > topk and (vals[keep - 1] - vals[keep]).abs() < 1e-4 * vals.abs().max():
                continue  # a near-tie at the cut: either side is a correct top-k
            self.assertEqual(
                set(table[0, j].nonzero().flatten().tolist()),
                set(order[:keep].tolist()),
                f"anchor {anchor}",
            )
            compared += 1
        # With 4 heads, a key whose q.k is negative for every head scores exactly 0, so
        # ties at the cut turn up once there are many anchors.
        self.assertGreaterEqual(compared, len(anchors) - max(1, len(anchors) // 10))
        return table

    def test_mrope_with_an_image(self):
        try:
            import vllm.model_executor.layers.rotary_embedding.mrope  # noqa: F401
        except Exception as exc:
            self.skipTest(f"vLLM unavailable: {exc}")
        # A low theta and a 4 x 6 image make the t/h/w column assignment matter: several
        # queries (anchor - 1 = 26, 34, 44, 49) sit inside or at the end of the image.
        tokens = [TXT] * 20 + _image(4, 6) + [TXT] * 30
        anchors = [9, 20, 27, 35, 45, 50, 62, 79]
        for use_fp8 in (True, False):
            with self.subTest(use_fp8=use_fp8):
                config = _target_config(mrope=True, theta=10.0)
                self._check(config, tokens, anchors, use_fp8=use_fp8)

    def test_prepared_positions_are_read_as_given(self):
        # The wrapper hands over the batch's M-RoPE positions; without them the indexer
        # rebuilds the same ones from the token ids.
        indexer = _random_indexer(_target_config(mrope=True, theta=10.0))
        tokens = [TXT] * 20 + _image(4, 6) + [TXT] * 30
        input_ids = torch.tensor([tokens])
        hidden = torch.randn(1, len(tokens), 64, generator=torch.Generator().manual_seed(3))
        anchors = torch.tensor([[9, 27, 35, 50, 62, 79]])
        rebuilt = indexer.candidate_table(hidden, input_ids, anchors)
        prepared = build_position_ids(input_ids, RECIPE)
        self.assertTrue(torch.equal(indexer._positions(input_ids, None), prepared))
        self.assertTrue(
            torch.equal(
                indexer.candidate_table(hidden, input_ids, anchors, position_ids=prepared),
                rebuilt,
            )
        )
        flat = torch.arange(len(tokens)).expand(1, 3, -1)
        self.assertTrue(torch.equal(indexer._positions(input_ids, None, flat), flat))
        self.assertFalse(
            torch.equal(
                indexer.candidate_table(hidden, input_ids, anchors, position_ids=flat), rebuilt
            )
        )

    def test_1d(self):
        tokens = list(range(1, 61))
        for use_fp8 in (True, False):
            with self.subTest(use_fp8=use_fp8):
                self._check(
                    _target_config(mrope=False), tokens, [3, 9, 20, 33, 47, 59], use_fp8=use_fp8
                )

    def test_fp8_moves_the_cut(self):
        # Every anchor of a longer 1D sequence: the rounding changes some top-k sets, and
        # the fp8 reference still matches all of them.
        tokens = list(range(1, 81))
        anchors = list(range(10, 80))
        fp8 = self._check(_target_config(mrope=False), tokens, anchors, use_fp8=True)
        exact = self._check(_target_config(mrope=False), tokens, anchors, use_fp8=False)
        self.assertGreater(int((fp8 != exact).any(-1).sum()), 0)


class TestCandidateTable(unittest.TestCase):
    L = 60

    def _table(self, anchors, topk=8, doc_ids=None):
        config = _target_config(mrope=False)
        indexer = _random_indexer(config, topk=topk)
        hidden = torch.randn(1, self.L, 64, generator=torch.Generator().manual_seed(4))
        return indexer.candidate_table(
            hidden, torch.arange(1, self.L + 1).unsqueeze(0), torch.tensor([anchors]), doc_ids
        )

    def test_candidates_stay_in_the_document_and_before_the_anchor(self):
        doc_ids = torch.tensor([[0] * 25 + [1] * 35])
        table = self._table([10, 30, 45], doc_ids=doc_ids)
        for j, (anchor, start) in enumerate(((10, 0), (30, 25), (45, 25))):
            picked = table[0, j].nonzero().flatten()
            self.assertTrue(bool(((picked >= start) & (picked < anchor)).all()), anchor)
            self.assertEqual(picked.numel(), min(8, anchor - start), anchor)
        # Fewer visible keys than the budget: all of them, exactly.
        self.assertEqual(table[0, 1].nonzero().flatten().tolist(), list(range(25, 30)))

    def test_a_budget_covering_the_context_keeps_everything(self):
        table = self._table([5, 33, 59], topk=64)
        for j, anchor in enumerate((5, 33, 59)):
            self.assertEqual(table[0, j].nonzero().flatten().tolist(), list(range(anchor)))

    def test_an_anchor_without_context_gets_nothing(self):
        self.assertFalse(bool(self._table([0])[0, 0].any()))

    def test_document_starts(self):
        docs = torch.tensor([[0, 0, 1, 1, 1, -1, -1]])
        starts = document_starts(docs, 1, 7, torch.device("cpu"))
        self.assertEqual(starts.tolist(), [[0, 0, 2, 2, 2, 5, 5]])

    def test_rotary_tables_stay_fp32_under_bf16_weights(self):
        for mrope in (True, False):
            with self.subTest(mrope=mrope):
                indexer = TargetIndexer(_target_config(mrope=mrope), INDEX_LAYER, 8)
                indexer._cast_weights(torch.bfloat16)
                self.assertEqual(indexer.q_a_proj.weight.dtype, torch.bfloat16)
                self.assertEqual(indexer.indexer.wk.weight.dtype, torch.bfloat16)
                input_ids = torch.full((1, 6), TXT)
                cos, sin = indexer._cos_sin(indexer._positions(input_ids, None), 6)
                self.assertEqual((cos.dtype, sin.dtype), (torch.float32, torch.float32))


class TestTargetIndexerLoading(unittest.TestCase):
    def _write(self, tmp, config, prefix="model.layers.5.", shapes=None, extra=None):
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f)
        ref = TargetIndexer(config, INDEX_LAYER, 8)
        g = torch.Generator().manual_seed(7)
        tensors = {}
        suffixes = {
            "input_layernorm.weight": "input_layernorm.weight",
            "q_a_proj.weight": "self_attn.q_a_proj.weight",
            "q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
            "indexer.wq_b.weight": "self_attn.indexer.wq_b.weight",
            "indexer.wk.weight": "self_attn.indexer.wk.weight",
            "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
            "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
            "indexer.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
        }
        for name, param in ref.named_parameters():
            shape = (shapes or {}).get(name, param.shape)
            tensors[prefix + suffixes[name]] = torch.randn(shape, generator=g)
        tensors.update(extra or {})
        keys = sorted(tensors)
        shards = {"a.safetensors": keys[::2], "b.safetensors": keys[1::2]}
        weight_map = {}
        for file_name, shard_keys in shards.items():
            shard = {k: tensors[k].contiguous() for k in shard_keys}
            save_file(shard, os.path.join(tmp, file_name))
            weight_map.update({k: file_name for k in shard_keys})
        with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)
        return {name: tensors[prefix + suffix] for name, suffix in suffixes.items()}

    def test_loads_its_eight_tensors_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            decoy = {
                "model.mtp_layers.5.self_attn.indexer.wq_b.weight": torch.zeros(64, 32),
                "model.layers.3.self_attn.indexer.wq_b.weight": torch.zeros(64, 32),
            }
            expected = self._write(tmp, _target_config(), extra=decoy)
            indexer = TargetIndexer.from_pretrained(
                tmp, INDEX_LAYER, 8, device="cpu", dtype=torch.float32
            )
            params = dict(indexer.named_parameters())
            self.assertEqual(set(params), set(expected))
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(params[name], tensor), name)

    def test_nested_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = self._write(tmp, _target_config(), prefix="model.language_model.layers.5.")
            indexer = TargetIndexer.from_pretrained(
                tmp, INDEX_LAYER, 8, device="cpu", dtype=torch.float32
            )
            self.assertTrue(torch.equal(indexer.indexer.wk.weight, expected["indexer.wk.weight"]))

    def test_shape_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, _target_config(), shapes={"indexer.wq_b.weight": (64, 16)})
            with self.assertRaisesRegex(ValueError, "shape"):
                TargetIndexer.from_pretrained(tmp, INDEX_LAYER, 8, device="cpu")

    def test_shared_layers_and_non_dsa_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "borrows"):
            TargetIndexer(_target_config(), 2, 8)
        with self.assertRaisesRegex(ValueError, "index_topk"):
            check_target_index_layer(_target_config(index_topk=None), INDEX_LAYER)
        with self.assertRaisesRegex(ValueError, "outside"):
            check_target_index_layer(_target_config(), 6)

    @unittest.skipUnless(
        os.environ.get("ANGELSPEC_HY4_TARGET"), "set ANGELSPEC_HY4_TARGET to a hy4 DSA target"
    )
    def test_real_hy4_checkpoint(self):
        path = os.environ["ANGELSPEC_HY4_TARGET"]
        check_target_index_layer(load_target_config(path), 33)
        with self.assertRaisesRegex(ValueError, "borrows"):
            check_target_index_layer(load_target_config(path), 32)
        indexer = TargetIndexer.from_pretrained(path, 33, 2048, device="cpu")
        self.assertEqual(tuple(indexer.indexer.wq_b.weight.shape), (2048, 1536))
        recipe = indexer._position_recipe
        self.assertEqual(
            (
                recipe.image_start_token_id,
                recipe.image_end_token_id,
                recipe.image_newline_token_id,
            ),
            (120118, 120119, 120121),
        )
        self.assertIsInstance(indexer.rotary_emb, MRotaryEmbedding)
        self.assertTrue(indexer.rotary_emb.interleaved)


class TestCandidateMask(unittest.TestCase):
    CTX = 32

    def _grid(self, **kw):
        base = dict(
            anchor_positions=torch.tensor([[6, 20]]),
            block_keep_mask=torch.tensor([[True, True]]),
            ctx_len=self.CTX,
            block_size=4,
            causal=True,
        )
        base.update(kw)
        mask_mod = _create_dflash_mask_mod(**base)
        return create_mask(mask_mod, 1, 1, 8, self.CTX + 8, device="cpu")[0, 0]

    def _candidates(self, rows):
        table = torch.zeros(1, 2, self.CTX, dtype=torch.bool)
        for block, positions in rows.items():
            table[0, block, positions] = True
        return table

    def _context(self, grid, query):
        return grid[query, : self.CTX].nonzero().flatten().tolist()

    def test_window_and_candidates_are_unioned(self):
        # Block 1 (anchor 20): 25 lies past the anchor and must never show up.
        cand = self._candidates({0: [1], 1: [2, 5, 25]})
        grid = self._grid(context_local_window=3, ctx_candidates=cand)
        self.assertEqual(self._context(grid, 4), [2, 5, 17, 18, 19])
        self.assertEqual(self._context(grid, 0), [1, 3, 4, 5])
        # Shared by every query of the block.
        for q in range(4, 8):
            self.assertEqual(self._context(grid, q), [2, 5, 17, 18, 19])

    def test_candidates_alone(self):
        grid = self._grid(ctx_candidates=self._candidates({1: [2, 5, 25]}))
        self.assertEqual(self._context(grid, 4), [2, 5])
        self.assertEqual(self._context(grid, 0), [])

    def test_window_alone_is_the_sliding_window(self):
        grid = self._grid(context_local_window=3)
        self.assertEqual(self._context(grid, 4), [17, 18, 19])

    def test_draft_keys_are_untouched(self):
        plain = self._grid()
        grid = self._grid(context_local_window=3, ctx_candidates=self._candidates({1: [2]}))
        self.assertTrue(torch.equal(grid[:, self.CTX :], plain[:, self.CTX :]))

    def test_packed_documents_still_apply(self):
        doc_ids = torch.tensor([[0] * 10 + [1] * 22])
        grid = self._grid(
            ctx_doc_ids=doc_ids,
            context_local_window=3,
            ctx_candidates=self._candidates({1: [2, 12]}),
        )
        self.assertEqual(self._context(grid, 4), [12, 17, 18, 19])


class TestDFlashModelTargetIndex(unittest.TestCase):
    H, V, L = 64, 128, 40

    def _draft_config(self, **kw):
        fields = dict(
            hidden_size=self.H,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=self.V,
            rms_norm_eps=1e-6,
            max_position_embeddings=512,
            num_target_layers=2,
            target_hidden_size=self.H,
            target_num_hidden_layers=6,
            target_layer_ids=[3, 4],
            mask_token_id=self.V - 1,
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

    def _wrapper(self, topk=None, window=None):
        kw = {}
        if window is not None:
            kw["context_local_window"] = window
        if topk is not None:
            kw.update(target_index_layer=INDEX_LAYER, target_index_topk=topk)
        torch.manual_seed(0)
        draft = DFlashDraftModel(self._draft_config(**kw)).to(torch.float32)
        draft.freeze_embedding()
        return DFlashModel(draft_model=draft, block_size=4, num_anchors=3)

    def _loss(self, wrapper):
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            return wrapper(
                input_ids=torch.randint(0, 100, (1, self.L), generator=g),
                hidden_states_list=[torch.randn(1, self.L, self.H, generator=g) for _ in range(2)],
                loss_mask=torch.ones(1, self.L),
                lm_head_weight=torch.randn(self.V, self.H, generator=g),
                injected_anchors=torch.tensor([[6, 20, 31]]),
            )[0]

    def _attached(self, topk, window=2):
        wrapper = self._wrapper(topk=topk, window=window)
        wrapper.attach_target_indexer(_random_indexer(_target_config(mrope=False), topk=topk))
        return wrapper

    def test_a_budget_covering_the_context_reproduces_the_dense_loss(self):
        dense = self._loss(self._wrapper())
        self.assertTrue(torch.equal(self._loss(self._attached(topk=64)), dense))

    def test_a_small_budget_changes_the_loss(self):
        dense = self._loss(self._wrapper())
        self.assertFalse(torch.allclose(self._loss(self._attached(topk=4)), dense))

    def test_candidates_come_from_the_layer_input_hidden(self):
        # Block anchors 20 and 31 have more context than the budget, so their
        # candidates depend on which hidden the indexer reads: target_layer_ids
        # [3, 4] with target_index_layer 5 must read hidden_states_list[1].
        wrapper = self._attached(topk=4)
        seen = {}
        indexer = wrapper._target_indexer_ref[0]
        original = indexer.candidate_indices

        def spy(hidden, *args, **kwargs):
            seen["hidden"] = hidden
            return original(hidden, *args, **kwargs)

        indexer.candidate_indices = spy
        g = torch.Generator().manual_seed(1)
        hidden_list = [torch.randn(1, self.L, self.H, generator=g) for _ in range(2)]
        with torch.no_grad():
            wrapper(
                input_ids=torch.randint(0, 100, (1, self.L), generator=g),
                hidden_states_list=hidden_list,
                loss_mask=torch.ones(1, self.L),
                lm_head_weight=torch.randn(self.V, self.H, generator=g),
                injected_anchors=torch.tensor([[6, 20, 31]]),
            )
        self.assertIs(seen["hidden"], hidden_list[1])

    def test_an_mrope_drafter_hands_its_positions_to_the_indexer(self):
        rope = {
            "rope_theta": 10000.0,
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        }
        torch.manual_seed(0)
        draft = DFlashDraftModel(
            self._draft_config(
                rope_parameters=rope,
                image_start_token_id=IMG_START,
                image_end_token_id=IMG_END,
                image_newline_token_id=IMG_NL,
                image_token_id=IMG_PAD,
                context_local_window=2,
                target_index_layer=INDEX_LAYER,
                target_index_topk=4,
            )
        ).to(torch.float32)
        draft.freeze_embedding()
        wrapper = DFlashModel(draft_model=draft, block_size=4, num_anchors=3)
        indexer = _random_indexer(_target_config(mrope=True), topk=4)
        wrapper.attach_target_indexer(indexer)
        seen = {}
        original = indexer.candidate_indices

        def spy(*args, **kwargs):
            seen["position_ids"] = kwargs.get("position_ids")
            return original(*args, **kwargs)

        indexer.candidate_indices = spy
        tokens = [TXT] * 12 + _image(2, 3) + [TXT] * (self.L - 22)
        input_ids = torch.tensor([tokens])
        position_ids = build_position_ids(input_ids, RECIPE)
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            loss = wrapper(
                input_ids=input_ids,
                hidden_states_list=[torch.randn(1, self.L, self.H, generator=g) for _ in range(2)],
                loss_mask=torch.ones(1, self.L),
                lm_head_weight=torch.randn(self.V, self.H, generator=g),
                position_ids=position_ids,
                injected_anchors=torch.tensor([[6, 25, 31]]),
            )[0]
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(seen["position_ids"])
        self.assertTrue(torch.equal(seen["position_ids"], position_ids))

    def test_the_indexer_must_be_attached(self):
        with self.assertRaisesRegex(RuntimeError, "no TargetIndexer is attached"):
            self._loss(self._wrapper(topk=4, window=2))

    def test_a_mismatched_indexer_is_rejected(self):
        wrapper = self._wrapper(topk=4)
        with self.assertRaisesRegex(ValueError, "does not match"):
            wrapper.attach_target_indexer(_random_indexer(_target_config(mrope=False), topk=8))

    def test_the_indexer_is_neither_registered_nor_saved(self):
        wrapper = self._attached(topk=4)
        indexer_params = {id(p) for p in wrapper._target_indexer_ref[0].parameters()}
        self.assertFalse(any(id(p) in indexer_params for p in wrapper.parameters()))
        self.assertEqual(set(wrapper.state_dict()), set(self._wrapper(topk=4).state_dict()))

    def test_layers_may_mix_candidates_and_windows(self):
        config = self._draft_config(
            target_index_layer=[INDEX_LAYER, None],
            target_index_topk=4,
            context_local_window=[2, 3],
        )
        self.assertEqual(config.target_index_layer, [INDEX_LAYER, None])
        self.assertEqual(config.context_local_window, [2, 3])
        self.assertEqual(
            target_layers_per_draft_layer([INDEX_LAYER, None], 2), [INDEX_LAYER, None]
        )
        self.assertEqual(windows_per_draft_layer(3, 2), [3, 3])
        self.assertEqual(windows_per_draft_layer([2, None], 2), [2, None])
        rejected = {
            "names no target layer": dict(
                target_index_layer=[None, None], target_index_topk=4, context_local_window=2
            ),
            "neither a target_index_layer nor": dict(
                target_index_layer=[INDEX_LAYER, None], target_index_topk=4
            ),
            "lists 1 windows for 2": dict(context_local_window=[2]),
            "positive int": dict(context_local_window=[2, 0]),
        }
        for message, kw in rejected.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self._draft_config(**kw)

    def test_a_layer_without_a_target_layer_reads_its_window_alone(self):
        torch.manual_seed(0)
        draft = DFlashDraftModel(
            self._draft_config(
                target_index_layer=[INDEX_LAYER, None],
                target_index_topk=4,
                context_local_window=[2, 3],
            )
        ).to(torch.float32)
        draft.freeze_embedding()
        wrapper = DFlashModel(draft_model=draft, block_size=4, num_anchors=3)
        wrapper.attach_target_indexer(_random_indexer(_target_config(mrope=False), topk=4))
        seen = {}
        original = wrapper.draft_model.forward

        def spy(**kwargs):
            seen.update(kwargs)
            return original(**kwargs)

        wrapper.draft_model.forward = spy
        self.assertTrue(torch.isfinite(self._loss(wrapper)))

        masks = seen["block_mask"]
        self.assertIsInstance(masks, list)
        self.assertEqual(len(masks), 2)
        anchors = torch.tensor([[6, 20, 31]])
        draft_len = anchors.shape[1] * 4
        window_only = _create_dflash_mask_mod(
            anchor_positions=anchors,
            block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
            ctx_len=self.L,
            block_size=4,
            causal=wrapper.causal_attn,
            context_local_window=3,
        )
        grids = [
            create_mask(mask.mask_mod, 1, 1, draft_len, self.L + draft_len, device="cpu")
            for mask in masks
        ]
        expected = create_mask(window_only, 1, 1, draft_len, self.L + draft_len, device="cpu")
        self.assertTrue(torch.equal(grids[1], expected))
        self.assertFalse(torch.equal(grids[0], grids[1]))


class TestSparseKvIndices(unittest.TestCase):
    """The tilelang path's per-query index lists hold exactly the keys the mask allows.

    ``_create_dflash_sparse_kv_indices`` re-applies the mask's rules to the candidate
    indices instead of reading the mask, so it is checked against the mask itself:
    the lists are scattered back into a keep-mask and compared bit for bit with
    ``create_mask`` over ``_create_dflash_mask_mod`` fed the same candidates as a
    table. The candidates include junk the indexer never returns (at or past the
    anchor, other documents, padding) so the rules are exercised, not just copied.
    """

    PAD = 6  # packed cases end in this many padding tokens (doc id -1)

    def _case(self, seed, bsz, L, n_blocks, bs, topk, window, causal, packed, dead, cand_width):
        g = torch.Generator().manual_seed(seed)
        doc_ids = None
        doc_start = torch.zeros(bsz, L, dtype=torch.long)
        if packed:
            cuts = sorted((torch.randperm(L - 4 * bs, generator=g)[:2] + 2 * bs).tolist())
            row = torch.full((L,), -1, dtype=torch.long)
            row[: cuts[0]], row[cuts[0] : cuts[1]], row[cuts[1] : L - self.PAD] = 0, 1, 2
            doc_ids = row.expand(bsz, -1).clone()
            doc_start = document_starts(doc_ids, bsz, L, torch.device("cpu"))

        anchors = torch.stack(
            [torch.randperm(L - bs + 1, generator=g)[:n_blocks].sort().values for _ in range(bsz)]
        )
        if packed:
            # A block at each document start (no context) and one in the padding tail
            # (doc id -1: context closed, own block still visible).
            anchors[:, 0], anchors[:, 1], anchors[:, -1] = cuts[0], cuts[1], L - bs
            anchors = anchors.sort(dim=1).values
        keep = torch.ones(bsz, n_blocks, dtype=torch.bool)
        if dead:
            keep[:, 2::3] = False
            anchors = torch.where(keep, anchors, 0)  # as _sample_anchor_positions does

        cands = None  # cand_width 0: a layer that reads its window alone
        if cand_width:
            cands = torch.full((bsz, n_blocks, cand_width), -1, dtype=torch.long)
            for b in range(bsz):
                for j in range(n_blocks):
                    a = int(anchors[b, j])
                    lo = int(doc_start[b, a]) if packed else 0
                    inside = torch.arange(lo, a)[torch.randperm(a - lo, generator=g)]
                    outside = torch.cat([torch.arange(0, lo), torch.arange(a, L)])
                    outside = outside[torch.randperm(outside.numel(), generator=g)][:2]
                    picked = torch.cat([inside[: max(0, cand_width - outside.numel())], outside])
                    cands[b, j, : picked.numel()] = picked

        mask_mod = _create_dflash_mask_mod(
            anchor_positions=anchors,
            block_keep_mask=keep,
            ctx_len=L,
            block_size=bs,
            ctx_doc_ids=doc_ids,
            causal=causal,
            context_local_window=window,
            ctx_candidates=None if cands is None else candidates_to_table(cands, L),
        )
        draft_len, kv_len = n_blocks * bs, L + n_blocks * bs
        expected = create_mask(mask_mod, bsz, 1, draft_len, kv_len, device="cpu")[:, 0]

        width = -(-((topk if cand_width else 0) + (window or 0) + bs) // 64) * 64
        indices = _create_dflash_sparse_kv_indices(
            anchor_positions=anchors,
            block_keep_mask=keep,
            ctx_len=L,
            block_size=bs,
            ctx_candidates=cands,
            width=width,
            ctx_doc_ids=doc_ids,
            causal=causal,
            context_local_window=window,
        )
        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(tuple(indices.shape), (bsz, draft_len, width))
        got = torch.zeros(bsz, draft_len, kv_len + 1, dtype=torch.bool)
        got.scatter_(2, torch.where(indices >= 0, indices, kv_len).long(), True)
        got = got[..., :kv_len]
        self.assertTrue(torch.equal(got, expected), "index lists disagree with the mask")
        # Every listed key is distinct: the kernel would weigh a repeated one twice.
        self.assertTrue(torch.equal((indices >= 0).sum(-1), got.sum(-1)))
        self.assertGreater(int(got.sum()), 0)

    def test_matches_the_mask(self):
        cases = {
            "plain": dict(window=None, causal=True, packed=False, dead=False),
            "window+dead": dict(window=2, causal=True, packed=False, dead=True),
            "packed+window+dead": dict(window=3, causal=True, packed=True, dead=True),
            "packed bidirectional": dict(window=None, causal=False, packed=True, dead=False),
            "window wider than context": dict(window=12, causal=True, packed=True, dead=True),
        }
        for i, (name, kw) in enumerate(cases.items()):
            for cand_width in (16, 9):  # the full budget, and a short sequence's min(topk, L)
                with self.subTest(case=name, cand_width=cand_width):
                    self._case(
                        seed=i,
                        bsz=2,
                        L=160,
                        n_blocks=12,
                        bs=4,
                        topk=16,
                        cand_width=cand_width,
                        **kw,
                    )

    def test_a_window_without_candidates_matches_the_mask(self):
        cases = {
            "window": dict(window=5, causal=True, packed=False, dead=False),
            "packed+window+dead": dict(window=7, causal=True, packed=True, dead=True),
        }
        for i, (name, kw) in enumerate(cases.items()):
            with self.subTest(case=name):
                self._case(
                    seed=10 + i, bsz=2, L=160, n_blocks=12, bs=4, topk=16, cand_width=0, **kw
                )

    def test_rows_need_candidates_or_a_window(self):
        with self.assertRaisesRegex(ValueError, "need ctx_candidates"):
            _create_dflash_sparse_kv_indices(
                anchor_positions=torch.tensor([[10]]),
                block_keep_mask=torch.tensor([[True]]),
                ctx_len=32,
                block_size=4,
                ctx_candidates=None,
                width=64,
            )

    def test_width_too_small_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "slots"):
            _create_dflash_sparse_kv_indices(
                anchor_positions=torch.tensor([[10]]),
                block_keep_mask=torch.tensor([[True]]),
                ctx_len=32,
                block_size=4,
                ctx_candidates=torch.full((1, 1, 70), -1),
                width=64,
            )


class _FixedCandidates:
    """Stands in for TargetIndexer: one precomputed candidate table, on any device."""

    def __init__(self, table, layer, topk):
        self.table, self.layer, self.topk = table, layer, topk

    def candidate_table(
        self,
        hidden,
        input_ids,
        anchor_positions,
        ctx_doc_ids=None,
        position_ids=None,
        sp_group=None,
    ):
        return self.table.to(hidden.device)

    def candidate_indices(
        self,
        hidden,
        input_ids,
        anchor_positions,
        ctx_doc_ids=None,
        position_ids=None,
        sp_group=None,
    ):
        """The same rows as indices, ``-1`` padded — what the real indexer returns."""
        table = self.table.to(hidden.device)
        width = int(table.sum(-1).max())
        order = torch.argsort((~table).int(), dim=-1, stable=True)[..., :width]
        return torch.where(torch.gather(table, -1, order), order, torch.full_like(order, -1))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestCandidatesOnCuda(unittest.TestCase):
    """On CUDA the block mask is built by torch.compile and flex attention reads the
    candidate table inside its kernel; on CPU the mask is materialized. Same inputs and
    candidates must give the same loss and gradients."""

    H, V, L, TOPK = 64, 128, 384, 16
    ANCHORS = torch.linspace(4, 376, 32).long().unsqueeze(0)
    SPLIT = 200  # packed variant: documents [0, 200) and [200, 384)

    def _table(self):
        g = torch.Generator().manual_seed(2)
        table = torch.zeros(1, self.ANCHORS.shape[1], self.L, dtype=torch.bool)
        for j, anchor in enumerate(self.ANCHORS[0].tolist()):
            table[0, j, torch.randperm(anchor, generator=g)[: self.TOPK]] = True
        return table

    def _run(self, device, packed):
        torch.manual_seed(0)
        config = DFlashConfig(
            hidden_size=self.H,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=self.V,
            rms_norm_eps=1e-6,
            max_position_embeddings=1024,
            num_target_layers=2,
            target_hidden_size=self.H,
            target_num_hidden_layers=6,
            target_layer_ids=[3, 4],
            mask_token_id=self.V - 1,
            attention_type="mla",
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            gated_mla=True,
            learnable_sink=True,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
            context_local_window=2,
            target_index_layer=INDEX_LAYER,
            target_index_topk=self.TOPK,
        )
        draft = DFlashDraftModel(config).to(torch.float32)
        draft.freeze_embedding()
        wrapper = DFlashModel(
            draft_model=draft, block_size=4, num_anchors=self.ANCHORS.shape[1], causal_attn=True
        ).to(device)
        wrapper.attach_target_indexer(_FixedCandidates(self._table(), INDEX_LAYER, self.TOPK))

        g = torch.Generator().manual_seed(1)
        inputs = dict(
            input_ids=torch.randint(0, 100, (1, self.L), generator=g),
            loss_mask=torch.ones(1, self.L),
            lm_head_weight=torch.randn(self.V, self.H, generator=g),
            injected_anchors=self.ANCHORS,
        )
        hidden = [torch.randn(1, self.L, self.H, generator=g).to(device) for _ in range(2)]
        if packed:
            rest = self.L - self.SPLIT
            inputs["ctx_doc_ids"] = torch.tensor([[0] * self.SPLIT + [1] * rest])
            inputs["base_position_ids"] = torch.cat(
                [torch.arange(self.SPLIT), torch.arange(rest)]
            ).unsqueeze(0)
        inputs = {name: t.to(device) for name, t in inputs.items()}

        loss = wrapper(hidden_states_list=hidden, **inputs)[0]
        loss.backward()
        grads = {n: p.grad.cpu() for n, p in wrapper.named_parameters() if p.grad is not None}
        return loss.detach().cpu(), grads

    def test_flex_attention_matches_the_materialized_mask(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                loss_cpu, grads_cpu = self._run("cpu", packed)
                loss_cuda, grads_cuda = self._run("cuda", packed)
                torch.testing.assert_close(loss_cuda, loss_cpu, rtol=1e-4, atol=1e-5)
                self.assertEqual(set(grads_cuda), set(grads_cpu))
                # flex attention's backward matches the dense reference only to ~1e-3 of
                # each gradient's scale, with or without a mask.
                for name, grad in grads_cpu.items():
                    scale = float(grad.abs().max()) + 1e-12
                    torch.testing.assert_close(
                        grads_cuda[name], grad, rtol=0, atol=3e-3 * scale, msg=name
                    )


@unittest.skipUnless(
    torch.cuda.is_available() and tilelang_dsa_available(), "needs CUDA and the TileLang kernels"
)
class TestSparseKvAttentionOnCuda(unittest.TestCase):
    """``dsa_backend='tilelang'`` hands the candidates to the sparse kernel as index lists;
    ``'torch'`` reads them through the block mask in flex. Same weights and inputs must
    give the same loss and gradients, to the kernel's bf16 precision."""

    H, V, L, TOPK = 128, 128, 384, 16
    ANCHORS = torch.linspace(4, 376, 32).long().unsqueeze(0)
    SPLIT = 200
    # DFlashConfig fields a subclass compares under another layout.
    CONFIG = {}

    def _table(self):
        g = torch.Generator().manual_seed(2)
        table = torch.zeros(1, self.ANCHORS.shape[1], self.L, dtype=torch.bool)
        for j, anchor in enumerate(self.ANCHORS[0].tolist()):
            table[0, j, torch.randperm(anchor, generator=g)[: self.TOPK]] = True
        return table

    def _wrapper(self, backend):
        torch.manual_seed(0)
        fields = dict(
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
            kv_lora_rank=512,  # the kernel's fixed latent width
            qk_nope_head_dim=64,
            qk_rope_head_dim=64,
            v_head_dim=64,
            gated_mla=True,
            learnable_sink=True,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
            context_local_window=2,
            target_index_layer=INDEX_LAYER,
            target_index_topk=self.TOPK,
            dsa_backend=backend,
        )
        fields.update(self.CONFIG)
        config = DFlashConfig(**fields)
        draft = DFlashDraftModel(config).to(torch.float32)
        draft.freeze_embedding()
        wrapper = DFlashModel(
            draft_model=draft, block_size=4, num_anchors=self.ANCHORS.shape[1], causal_attn=True
        ).cuda()
        wrapper.attach_target_indexer(_FixedCandidates(self._table(), INDEX_LAYER, self.TOPK))
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
        for packed in (False, True):
            with self.subTest(packed=packed):
                flex = self._wrapper("torch")
                with torch.no_grad():  # a non-trivial sink, so its fusion is exercised
                    for layer in flex.draft_model.layers:
                        layer.self_attn.learnable_sink_param.normal_()
                fast = self._wrapper("tilelang")
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


class TestMixedLayersOnCuda(TestSparseKvAttentionOnCuda):
    """The same comparison for draft layers that mix candidates and windows, whose index
    lists then differ in width: candidates + window 2, window 3 alone, candidates alone."""

    CONFIG = dict(
        num_hidden_layers=3,
        target_index_layer=[INDEX_LAYER, None, INDEX_LAYER],
        context_local_window=[2, 3, None],
    )


if __name__ == "__main__":
    unittest.main()
