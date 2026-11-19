"""Attentive hidden-states correction (``DFlashConfig.attentive_correction``).

Covers:
1. Config: needs the hidden correction, the target's latents and q_lora_rank; the source
   layer must be a draft layer; off by default
2. The module on CPU (dense reference path): copied from a draft attention it reads the
   context exactly like that attention over the same keys; o_proj at zero changes nothing;
   slots without a key get 0; any row layout, one slot at a time included (how serving walks
   a block), gives the same rows; o_proj takes the first gradient, then the reading (both
   halves of q_in) learns
3. DFlashModel with the drafter stubbed out (no TileLang kernel on CPU): the context is the
   source layer's target latent and its keys minus the block's own; corrections chunk by
   block like the whole; the chunked losses equal the unchunked one; the attention trains;
   acorr_out_ratio is logged; dflash_acorr_eval_off leaves just the attention out at eval.
   All of it without block keys and with them shared or gated (also with their own kv_b or
   with keys from the hiddens alone; the gates are logged); dflash_hc_topk_diag refuses the
   block keys
4. Block keys (attentive_correction_block_keys): kv_in starts as the source layer's
   kv_a_proj_with_mqa on the hidden half; slot k reads its block's slots 0..k and nothing
   past them; a block cut short at slot k (how serving drafts) gives slot k what the full
   block does; with no context key a slot still reads its block; the slots must come as
   whole blocks; kv_in learns. A warm start lacking only the block keys gets just those
5. Gated block keys (attentive_correction_block_mode "gated"): with the gate closed (as it
   starts) the module reads exactly as without block keys; the gate acts per slot and the
   output is linear in it; causality and a block cut short as above; the gate learns first,
   then the block read; per-slot gate stats; the sink starts as the source layer's. A warm
   start from shared block keys keeps them and gets just the closed gate
6. The gated block read's own kv_b (attentive_correction_block_kv_b "separate"): only with
   the gated read; started as kv_b_proj it reads exactly as the shared one; apart from it,
   it moves only the block read (a closed gate leaves the context read as it was); it
   learns once the gate opens; its drift is logged; a warm start starts only what it lacks
   (the kv_b as the loaded kv_b_proj, the gate only if missing, the keys only if missing)
7. Block keys from the hiddens alone (attentive_correction_block_kv_in "hidden"), shared or
   gated: kv_in reads D channels; it reads exactly as [h; e] keys with a zero previous-token
   half; a drafted token reaches only its own slot's read (its query and gate), a slot's
   hidden still the key of the slots after it; a fresh start and a warm start from a
   drafter without block keys start kv_in as the source layer's whole kv_a_proj_with_mqa
8. On CUDA: the TileLang kernel, which training runs, matches the dense reference forward
   and backward, to its bf16 precision, without block keys and with them shared, gated,
   gated with their own kv_b or gated with keys from the hiddens alone
"""

import copy
import math
import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.attentive_correction import (
    AttentiveCorrection,
    init_block_kv_b,
    init_from_attention,
)
from angelspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDraftModel,
    HiddenStatesCorrection,
)
from angelspec.models.draft.hy4_mla import Hy4DraftMLAAttention
from angelspec.models.draft.mtp import _apply_rope_split, _interleaved_to_split_rope
from angelspec.models.ops.tilelang_dsa import tilelang_dsa_available
from tests.test_dflash_target_kv import (
    RANK,
    ROPE,
    H,
    V,
    _draft_config,
    _FixedCandidates,
    _random_latent,
    _random_table,
    _target_config,
)


def _config(**kw):
    fields = dict(
        mrope=True,
        gated_mla=False,
        enable_hidden_correction=True,
        attentive_correction=True,
        target_kv_layers=[4, 5],
    )
    fields.update(kw)
    return _draft_config(**fields)


def _randomize(module, seed, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=g) * scale)


class TestConfig(unittest.TestCase):
    def test_requirements_and_default(self):
        with self.assertRaisesRegex(ValueError, "enable_hidden_correction"):
            _config(enable_hidden_correction=False)
        with self.assertRaisesRegex(ValueError, "target_kv_layers"):
            _config(target_kv_layers=None)
        for layer in (2, -3, True):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "source_layer"):
                _config(attentive_correction_source_layer=layer)
        with self.assertRaisesRegex(ValueError, "block_keys"):
            _draft_config(attentive_correction_block_keys=True)
        with self.assertRaisesRegex(ValueError, "block_mode"):
            _config(attentive_correction_block_keys=True, attentive_correction_block_mode="soft")
        with self.assertRaisesRegex(ValueError, "set attentive_correction_block_keys"):
            _config(attentive_correction_block_mode="gated")
        config = _config()
        self.assertTrue(config.attentive_correction)
        self.assertEqual(config.attentive_correction_source_layer, -1)
        self.assertFalse(config.attentive_correction_block_keys)
        self.assertEqual(config.attentive_correction_block_mode, "shared")
        self.assertFalse(_draft_config().attentive_correction)
        self.assertTrue(
            _config(attentive_correction_block_keys=True).attentive_correction_block_keys
        )


class TestModule(unittest.TestCase):
    B, S, C, K = 2, 6, 20, 8

    def setUp(self):
        self.config = _config()
        self.attn = Hy4DraftMLAAttention(self.config).float()
        _randomize(self.attn, 1)
        g = torch.Generator().manual_seed(2)
        self.h = torch.randn(self.B, self.S, H, generator=g)
        self.e = torch.randn(self.B, self.S, H, generator=g)
        self.latent = torch.randn(self.B, self.C, RANK + ROPE, generator=g)
        self.pos = torch.randint(self.C, self.C + 40, (self.B, 3, self.S), generator=g)
        idx = torch.full((self.B, self.S, self.K), -1, dtype=torch.long)
        for b in range(self.B):
            for s in range(self.S):
                n = int(torch.randint(1, self.K + 1, (1,), generator=g))
                idx[b, s, :n] = torch.randperm(self.C, generator=g)[:n]
        idx[1, 3] = -1  # a slot with nothing to read
        self.idx = idx

    def _copy(self):
        corr = AttentiveCorrection(self.config).float()
        _randomize(corr, 3)
        init_from_attention(corr, self.attn)
        return corr

    def _read(self, corr, h=None, e=None, pos=None, idx=None):
        return corr(
            self.h if h is None else h,
            self.e if e is None else e,
            self.pos if pos is None else pos,
            self.latent,
            self.idx if idx is None else idx,
        )

    @torch.no_grad()
    def test_copies_the_source_attention(self):
        corr = self._copy()
        torch.testing.assert_close(corr.q_in.weight[:, :H], self.attn.q_a_proj.weight)
        self.assertEqual(float(corr.q_in.weight[:, H:].abs().sum()), 0.0)
        for name in ("q_a_layernorm", "q_b_proj", "kv_b_proj"):
            torch.testing.assert_close(
                getattr(corr, name).weight, getattr(self.attn, name).weight, msg=name
            )
        torch.testing.assert_close(corr.learnable_sink_param, self.attn.learnable_sink_param)
        self.assertEqual(float(corr.o_proj.weight.abs().sum()), 0.0)
        other = Hy4DraftMLAAttention(_config(q_lora_rank=16))
        with self.assertRaisesRegex(ValueError, "q_a_proj"):
            init_from_attention(corr, other)

    def test_reads_like_the_source_attention(self):
        corr = self._copy()
        with torch.no_grad():
            corr.o_proj.weight.copy_(self.attn.o_proj.weight)
            ours, alive = self._read(corr)
            attn = self.attn
            q_nope, q_pe, _ = attn._project_q(self.h)
            cos, sin = attn._cos_sin(q_nope, self.pos, self.C)
            q_pe = _apply_rope_split(_interleaved_to_split_rope(q_pe), cos, sin)
            keep = torch.zeros(self.B, self.S, self.C + 1, dtype=torch.bool)
            keep.scatter_(-1, torch.where(self.idx >= 0, self.idx, self.C), True)
            keep = keep[..., : self.C]
            reference = attn._output_tail(
                attn._attend_torch(torch.cat([q_nope, q_pe], -1), self.latent, keep.unsqueeze(1)),
                self.h,
            )
        torch.testing.assert_close(ours, reference, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(alive, keep.any(-1))

    def test_zero_o_proj_changes_nothing(self):
        plain = HiddenStatesCorrection(H, H, 32).float()
        _randomize(plain, 4)
        attentive = HiddenStatesCorrection(H, H, 32, attention=self._copy()).float()
        attentive.load_state_dict(plain.state_dict(), strict=False)
        context = dict(position_ids=self.pos, latent=self.latent, indices=self.idx)
        stats = {}
        with torch.no_grad():
            out = attentive(self.h, self.e, context=context, attention_stats=stats)
            torch.testing.assert_close(out, plain(self.h, self.e))
            torch.testing.assert_close(attentive(self.h, self.e), out)
        self.assertEqual(float(stats["attention"]), 0.0)
        self.assertGreater(float(stats["hidden"]), 0.0)

    def test_dead_slots_and_row_layouts(self):
        corr = self._copy()
        _randomize(corr.o_proj, 5, scale=0.05)
        with torch.no_grad():
            flat, alive = self._read(corr)
            self.assertFalse(bool(alive[1, 3]))
            self.assertEqual(float(flat[1, 3].abs().sum()), 0.0)
            shaped, _ = self._read(
                corr,
                self.h.view(self.B, 2, 3, H),
                self.e.view(self.B, 2, 3, H),
                self.pos.view(self.B, 3, 2, 3),
                self.idx.view(self.B, 2, 3, self.K),
            )
            torch.testing.assert_close(shaped.view(self.B, self.S, H), flat)
            for s in range(self.S):
                one, _ = self._read(
                    corr,
                    self.h[:, s : s + 1],
                    self.e[:, s : s + 1],
                    self.pos[..., s : s + 1],
                    self.idx[:, s : s + 1],
                )
                torch.testing.assert_close(one[:, 0], flat[:, s], msg=f"slot {s}")

    def test_o_proj_learns_first(self):
        corr = self._copy()
        target = torch.randn(self.B, self.S, H, generator=torch.Generator().manual_seed(6))
        out, _ = self._read(corr)
        (out * target).sum().backward()
        self.assertGreater(float(corr.o_proj.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(corr.q_in.weight.grad.abs().sum()), 0.0)
        corr.zero_grad()
        _randomize(corr.o_proj, 7, scale=0.05)
        out, _ = self._read(corr)
        (out * target).sum().backward()
        for name, grad in (
            ("hidden half of q_in", corr.q_in.weight.grad[:, :H]),
            ("previous-token half of q_in", corr.q_in.weight.grad[:, H:]),
            ("kv_b_proj", corr.kv_b_proj.weight.grad),
        ):
            self.assertGreater(float(grad.abs().sum()), 0.0, name)


class _BlockFixture:
    B, S, C, K, BS = 2, 8, 20, 6, 4

    def setUp(self):
        self.attn = Hy4DraftMLAAttention(_config(attentive_correction_block_keys=True)).float()
        _randomize(self.attn, 1)
        g = torch.Generator().manual_seed(2)
        self.h = torch.randn(self.B, self.S, H, generator=g)
        self.e = torch.randn(self.B, self.S, H, generator=g)
        self.latent = torch.randn(self.B, self.C, RANK + ROPE, generator=g)
        self.pos = torch.randint(self.C, self.C + 40, (self.B, 3, self.S), generator=g)
        idx = torch.full((self.B, self.S, self.K), -1, dtype=torch.long)
        for b in range(self.B):
            for s in range(self.S):
                n = int(torch.randint(1, self.K + 1, (1,), generator=g))
                idx[b, s, :n] = torch.randperm(self.C, generator=g)[:n]
        self.idx = idx

    def _read(self, corr, rows=None, idx=None, block_size=None, stats=None):
        rows = slice(None) if rows is None else rows
        return corr(
            self.h[:, rows],
            self.e[:, rows],
            self.pos[..., rows],
            self.latent,
            (self.idx if idx is None else idx)[:, rows],
            block_size=self.BS if block_size is None else block_size,
            stats=stats,
        )


class TestBlockKeys(_BlockFixture, unittest.TestCase):
    def _module(self, block_keys=True):
        corr = AttentiveCorrection(_config(attentive_correction_block_keys=block_keys)).float()
        _randomize(corr, 3)
        init_from_attention(corr, self.attn)
        _randomize(corr.o_proj, 5, scale=0.05)
        if block_keys:
            g = torch.Generator().manual_seed(6)
            with torch.no_grad():
                corr.kv_in.weight[:, H:].copy_(
                    torch.randn(corr.kv_in.weight[:, H:].shape, generator=g)
                )
        return corr

    @torch.no_grad()
    def test_copies_the_source_kv_a(self):
        corr = AttentiveCorrection(_config(attentive_correction_block_keys=True)).float()
        _randomize(corr, 3)
        init_from_attention(corr, self.attn)
        torch.testing.assert_close(corr.kv_in.weight[:, :H], self.attn.kv_a_proj_with_mqa.weight)
        self.assertEqual(float(corr.kv_in.weight[:, H:].abs().sum()), 0.0)
        torch.testing.assert_close(corr.kv_a_layernorm.weight, self.attn.kv_a_layernorm.weight)

    @torch.no_grad()
    def test_reads_its_block_up_to_itself(self):
        corr = self._module()
        out, _ = self._read(corr)
        self.e[:, 1] += 1.0
        moved, _ = self._read(corr)
        diff = (moved - out).abs().sum(dim=(0, 2))
        self.assertEqual(float(diff[0]), 0.0)
        self.assertGreater(float(diff[1]), 0.0)
        self.assertGreater(float(diff[2]), 0.0, "slot 2 reads slot 1 as a block key")
        self.assertGreater(float(diff[3]), 0.0)
        self.assertEqual(float(diff[4:].sum()), 0.0, "the next block does not")

    @torch.no_grad()
    def test_a_block_cut_short_matches_the_full_block(self):
        corr = self._module()
        full, _ = self._read(corr)
        for start in (0, self.BS):
            for k in range(self.BS):
                part, _ = self._read(corr, slice(start, start + k + 1), block_size=k + 1)
                torch.testing.assert_close(part[:, k], full[:, start + k], msg=f"slot {start + k}")

    @torch.no_grad()
    def test_reads_the_block_without_context(self):
        dead = torch.full_like(self.idx, -1)
        out, alive = self._read(self._module(), idx=dead)
        self.assertTrue(bool(alive.all()))
        self.assertTrue(bool((out.abs().sum(-1) > 0).all()))
        out, alive = self._read(self._module(block_keys=False), idx=dead)
        self.assertFalse(bool(alive.any()))
        self.assertEqual(float(out.abs().sum()), 0.0)

    @torch.no_grad()
    def test_needs_whole_blocks(self):
        corr = self._module()
        for block_size in (None, 3, 0):
            with (
                self.subTest(block_size=block_size),
                self.assertRaisesRegex(ValueError, "whole blocks"),
            ):
                corr(self.h, self.e, self.pos, self.latent, self.idx, block_size=block_size)

    def test_kv_in_learns(self):
        corr = self._module()
        out, _ = self._read(corr)
        out.sum().backward()
        for name, grad in (
            ("hidden half", corr.kv_in.weight.grad[:, :H]),
            ("previous-token half", corr.kv_in.weight.grad[:, H:]),
            ("kv_a_layernorm", corr.kv_a_layernorm.weight.grad),
        ):
            self.assertGreater(float(grad.abs().sum()), 0.0, name)

    def test_warm_start_lacking_only_the_block_keys(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        draft = DFlashDraftModel(_config(attentive_correction_block_keys=True)).float()
        attention = draft.hidden_correction.attention
        _randomize(attention, 7)
        source = draft.layers[-1].self_attn
        trainer = object.__new__(DFlashTrainer)
        prefix = "hidden_correction.attention."
        before = {n: p.detach().clone() for n, p in attention.named_parameters()}

        DFlashTrainer._init_attentive_correction(trainer, draft, [])
        for name, param in attention.named_parameters():
            torch.testing.assert_close(param, before[name], msg=name)

        missing = [prefix + "kv_in.weight", prefix + "kv_a_layernorm.weight"]
        DFlashTrainer._init_attentive_correction(trainer, draft, missing)
        torch.testing.assert_close(attention.kv_in.weight[:, :H], source.kv_a_proj_with_mqa.weight)
        self.assertEqual(float(attention.kv_in.weight[:, H:].abs().sum()), 0.0)
        torch.testing.assert_close(attention.kv_a_layernorm.weight, source.kv_a_layernorm.weight)
        for name, param in attention.named_parameters():
            if not name.startswith(("kv_in", "kv_a_layernorm")):
                torch.testing.assert_close(param, before[name], msg=name)

        DFlashTrainer._init_attentive_correction(trainer, draft, [prefix + "q_in.weight"])
        self.assertEqual(float(attention.o_proj.weight.abs().sum()), 0.0)
        torch.testing.assert_close(attention.q_in.weight[:, :H], source.q_a_proj.weight)


def _gated_config():
    return _config(attentive_correction_block_keys=True, attentive_correction_block_mode="gated")


class TestGatedBlockKeys(_BlockFixture, unittest.TestCase):
    def _pair(self):
        """A module without block keys and a gated one sharing every common parameter, the
        gated one's block read made non-trivial (gate still closed)."""
        plain = AttentiveCorrection(_config()).float()
        _randomize(plain, 3)
        init_from_attention(plain, self.attn)
        _randomize(plain.o_proj, 5, scale=0.05)
        gated = AttentiveCorrection(_gated_config()).float()
        _randomize(gated, 4)
        init_from_attention(gated, self.attn)
        _, unexpected = gated.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        g = torch.Generator().manual_seed(6)
        with torch.no_grad():
            half = gated.kv_in.weight[:, H:]
            half.copy_(torch.randn(half.shape, generator=g))
            gated.block_rel_bias.copy_(torch.randn(gated.block_rel_bias.shape, generator=g))
        return plain, gated

    @staticmethod
    @torch.no_grad()
    def _open(gated, value=1.0, slot=None):
        if slot is None:
            gated.block_gate_bias.fill_(value)
        else:
            gated.block_gate_bias[:, slot] = value

    @torch.no_grad()
    def test_a_closed_gate_reads_as_without_block_keys(self):
        plain, gated = self._pair()
        for idx in (self.idx, torch.full_like(self.idx, -1)):
            ref, _ = self._read(plain, idx=idx)
            out, alive = self._read(gated, idx=idx)
            torch.testing.assert_close(out, ref)
            self.assertTrue(bool(alive.all()))

    @torch.no_grad()
    def test_the_gate_acts_per_slot(self):
        plain, gated = self._pair()
        self._open(gated, 1.0, slot=2)
        ref, _ = self._read(plain)
        out, _ = self._read(gated)
        diff = (out - ref).abs().sum(dim=(0, 2))
        for row in range(self.S):
            if row % self.BS == 2:
                self.assertGreater(float(diff[row]), 0.0, f"row {row}")
            else:
                self.assertEqual(float(diff[row]), 0.0, f"row {row}")

    @torch.no_grad()
    def test_the_output_is_linear_in_the_gate(self):
        plain, gated = self._pair()
        ref, _ = self._read(plain)
        delta = {}
        for value in (0.3, 1.2):
            self._open(gated, value)
            delta[value] = self._read(gated)[0] - ref
        torch.testing.assert_close(
            delta[0.3] * math.tanh(1.2), delta[1.2] * math.tanh(0.3), rtol=1e-4, atol=1e-6
        )

    @torch.no_grad()
    def test_reads_its_block_up_to_itself(self):
        _, gated = self._pair()
        self._open(gated)
        out, _ = self._read(gated)
        self.e[:, 1] += 1.0
        moved, _ = self._read(gated)
        diff = (moved - out).abs().sum(dim=(0, 2))
        self.assertEqual(float(diff[0]), 0.0)
        for row in (1, 2, 3):
            self.assertGreater(float(diff[row]), 0.0, f"row {row}")
        self.assertEqual(float(diff[4:].sum()), 0.0, "the next block does not")

    @torch.no_grad()
    def test_a_block_cut_short_matches_the_full_block(self):
        _, gated = self._pair()
        self._open(gated)
        gated.block_gate.weight.copy_(
            torch.randn(gated.block_gate.weight.shape, generator=torch.Generator().manual_seed(8))
            * 0.05
        )
        full, _ = self._read(gated)
        for start in (0, self.BS):
            for k in range(self.BS):
                part, _ = self._read(gated, slice(start, start + k + 1), block_size=k + 1)
                torch.testing.assert_close(part[:, k], full[:, start + k], msg=f"slot {start + k}")

    def test_the_gate_learns_first(self):
        _, gated = self._pair()
        out, _ = self._read(gated)
        out.sum().backward()
        self.assertGreater(float(gated.block_gate_bias.grad.abs().sum()), 0.0)
        self.assertGreater(float(gated.block_gate.weight.grad.abs().sum()), 0.0)
        for name, param in (
            ("kv_in", gated.kv_in.weight),
            ("block_sink", gated.block_sink),
            ("block_rel_bias", gated.block_rel_bias),
        ):
            self.assertEqual(float(param.grad.abs().sum()), 0.0, name)
        gated.zero_grad()
        self._open(gated)
        out, _ = self._read(gated)
        out.sum().backward()
        for name, grad in (
            ("hidden half of kv_in", gated.kv_in.weight.grad[:, :H]),
            ("previous-token half of kv_in", gated.kv_in.weight.grad[:, H:]),
            ("block_sink", gated.block_sink.grad),
            ("block_rel_bias", gated.block_rel_bias.grad[:, : self.BS]),
        ):
            self.assertGreater(float(grad.abs().sum()), 0.0, name)

    @torch.no_grad()
    def test_gate_stats(self):
        _, gated = self._pair()
        stats = {}
        self._read(gated, stats=stats)
        self.assertEqual(float(stats["block_gate"].abs().sum()), 0.0)
        torch.testing.assert_close(
            stats["block_gate_count"], torch.full((self.BS,), float(self.B * self.S // self.BS))
        )
        self._open(gated, 1.0, slot=2)
        stats = {}
        self._read(gated, stats=stats)
        expected = torch.zeros(self.BS)
        expected[2] = math.tanh(1.0)
        torch.testing.assert_close(stats["block_gate"] / stats["block_gate_count"], expected)

    @torch.no_grad()
    def test_starts_closed_with_the_source_sink(self):
        corr = AttentiveCorrection(_gated_config()).float()
        _randomize(corr, 3)
        init_from_attention(corr, self.attn)
        torch.testing.assert_close(corr.block_sink, self.attn.learnable_sink_param)
        for name in ("block_rel_bias", "block_gate_bias"):
            self.assertEqual(float(getattr(corr, name).abs().sum()), 0.0, name)
        self.assertEqual(float(corr.block_gate.weight.abs().sum()), 0.0)
        with (
            mock.patch("angelspec.models.draft.attentive_correction.MAX_BLOCK_SLOTS", 2),
            self.assertRaisesRegex(ValueError, "up to 2 slots"),
        ):
            self._read(corr)

    def test_warm_start_lacking_only_the_gated_block(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        draft = DFlashDraftModel(_gated_config()).float()
        attention = draft.hidden_correction.attention
        _randomize(attention, 7)
        source = draft.layers[-1].self_attn
        before = {n: p.detach().clone() for n, p in attention.named_parameters()}
        prefix = "hidden_correction.attention."
        block = [
            n
            for n, _ in attention.named_parameters()
            if n.startswith(("kv_in.", "kv_a_layernorm.", "block_"))
        ]
        self.assertEqual(len(block), 6)
        trainer = object.__new__(DFlashTrainer)
        DFlashTrainer._init_attentive_correction(trainer, draft, [prefix + n for n in block])
        torch.testing.assert_close(attention.block_sink, source.learnable_sink_param)
        self.assertEqual(float(attention.block_gate_bias.abs().sum()), 0.0)
        self.assertEqual(float(attention.block_gate.weight.abs().sum()), 0.0)
        torch.testing.assert_close(attention.kv_in.weight[:, :H], source.kv_a_proj_with_mqa.weight)
        for name, param in attention.named_parameters():
            if name not in block:
                torch.testing.assert_close(param, before[name], msg=name)

    def test_warm_start_from_shared_block_keys_keeps_them(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        draft = DFlashDraftModel(_gated_config()).float()
        attention = draft.hidden_correction.attention
        _randomize(attention, 7)
        source = draft.layers[-1].self_attn
        before = {n: p.detach().clone() for n, p in attention.named_parameters()}
        prefix = "hidden_correction.attention."
        gate = [n for n, _ in attention.named_parameters() if n.startswith("block_")]
        self.assertEqual(len(gate), 4)
        trainer = object.__new__(DFlashTrainer)
        DFlashTrainer._init_attentive_correction(trainer, draft, [prefix + n for n in gate])
        torch.testing.assert_close(attention.block_sink, source.learnable_sink_param)
        for name in ("block_rel_bias", "block_gate_bias", "block_gate.weight"):
            self.assertEqual(float(attention.get_parameter(name).abs().sum()), 0.0, name)
        for name, param in attention.named_parameters():
            if name not in gate:
                torch.testing.assert_close(param, before[name], msg=name)


def _sep_config():
    return _config(
        attentive_correction_block_keys=True,
        attentive_correction_block_mode="gated",
        attentive_correction_block_kv_b="separate",
    )


_GATED_MODES = ("gated", "gated-sep", "gated-hidden")


def _block_fields(block_mode):
    """Config fields for a test block mode: None, "shared", "gated", "gated-sep" (gated
    with the block read's own kv_b) or "gated-hidden" (gated with keys from the hiddens
    alone)."""
    return dict(
        attentive_correction_block_keys=block_mode is not None,
        attentive_correction_block_mode="gated" if block_mode in _GATED_MODES else "shared",
        attentive_correction_block_kv_b="separate" if block_mode == "gated-sep" else "shared",
        attentive_correction_block_kv_in=(
            "hidden" if block_mode == "gated-hidden" else "hidden_embed"
        ),
    )


class TestSeparateBlockKvB(_BlockFixture, unittest.TestCase):
    """``attentive_correction_block_kv_b='separate'``: the gated block read's own kv_b."""

    def _pair(self):
        """A gated module sharing kv_b and one with its own, every common parameter equal,
        the block read made non-trivial and the gate open."""
        shared = AttentiveCorrection(_gated_config()).float()
        _randomize(shared, 4)
        init_from_attention(shared, self.attn)
        _randomize(shared.o_proj, 5, scale=0.05)
        g = torch.Generator().manual_seed(6)
        with torch.no_grad():
            half = shared.kv_in.weight[:, H:]
            half.copy_(torch.randn(half.shape, generator=g))
            shared.block_rel_bias.copy_(torch.randn(shared.block_rel_bias.shape, generator=g))
            shared.block_gate_bias.fill_(0.8)
        separate = AttentiveCorrection(_sep_config()).float()
        missing, unexpected = separate.load_state_dict(shared.state_dict(), strict=False)
        self.assertEqual((missing, unexpected), (["block_kv_b_proj.weight"], []))
        init_block_kv_b(separate)
        return shared, separate

    def test_validated(self):
        self.assertEqual(_config().attentive_correction_block_kv_b, "shared")
        with self.assertRaisesRegex(ValueError, "'shared' or 'separate'"):
            _config(attentive_correction_block_kv_b="own")
        for kw in (
            {},
            {"attentive_correction_block_keys": True},
        ):
            with self.subTest(**kw), self.assertRaisesRegex(ValueError, "gated block read"):
                _config(attentive_correction_block_kv_b="separate", **kw)
        self.assertFalse(hasattr(AttentiveCorrection(_gated_config()), "block_kv_b_proj"))

    @torch.no_grad()
    def test_starts_as_the_shared_kv_b(self):
        shared, separate = self._pair()
        stats = {}
        out_separate, _ = self._read(separate, stats=stats)
        torch.testing.assert_close(out_separate, self._read(shared)[0])
        self.assertEqual(float(stats["block_kv_b_drift"]), 0.0)

    @torch.no_grad()
    def test_its_kv_b_moves_only_the_block_read(self):
        shared, separate = self._pair()
        _randomize(separate.block_kv_b_proj, 7)
        stats = {}
        out_separate, _ = self._read(separate, stats=stats)
        self.assertFalse(torch.allclose(out_separate, self._read(shared)[0]))
        self.assertGreater(float(stats["block_kv_b_drift"]), 0.0)
        # A closed gate leaves the context read as it was, whatever the block's kv_b.
        for module in (shared, separate):
            module.block_gate_bias.zero_()
        torch.testing.assert_close(self._read(separate)[0], self._read(shared)[0])

    def test_it_learns_once_the_gate_opens(self):
        _, separate = self._pair()
        self._read(separate)[0].pow(2).sum().backward()
        self.assertGreater(float(separate.block_kv_b_proj.weight.grad.abs().sum()), 0.0)
        separate.zero_grad()
        with torch.no_grad():
            separate.block_gate_bias.zero_()
        self._read(separate)[0].pow(2).sum().backward()
        self.assertEqual(float(separate.block_kv_b_proj.weight.grad.abs().sum()), 0.0)

    @torch.no_grad()
    def test_starts_as_the_source_kv_b(self):
        corr = AttentiveCorrection(_sep_config()).float()
        _randomize(corr, 3)
        init_from_attention(corr, self.attn)
        torch.testing.assert_close(corr.block_kv_b_proj.weight, self.attn.kv_b_proj.weight)

    def test_a_warm_start_starts_only_what_it_lacks(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        prefix = "hidden_correction.attention."
        trainer = object.__new__(DFlashTrainer)
        draft = DFlashDraftModel(_sep_config()).float()
        attention = draft.hidden_correction.attention
        source = draft.layers[-1].self_attn
        kv_b = ["block_kv_b_proj.weight"]
        gate = ["block_sink", "block_rel_bias", "block_gate.weight", "block_gate_bias"]
        keys = ["kv_in.weight", "kv_a_layernorm.weight"]
        for missing in (kv_b, gate + kv_b, keys + gate + kv_b):
            with self.subTest(missing=missing):
                _randomize(attention, 7)
                before = {n: p.detach().clone() for n, p in attention.named_parameters()}
                DFlashTrainer._init_attentive_correction(
                    trainer, draft, [prefix + n for n in missing]
                )
                torch.testing.assert_close(
                    attention.block_kv_b_proj.weight, attention.kv_b_proj.weight
                )
                for name, param in attention.named_parameters():
                    if name not in missing:
                        torch.testing.assert_close(param, before[name], msg=name)
                if "block_gate.weight" in missing:
                    self.assertEqual(float(attention.block_gate.weight.abs().sum()), 0.0)
                    torch.testing.assert_close(attention.block_sink, source.learnable_sink_param)
                if "kv_in.weight" in missing:
                    torch.testing.assert_close(
                        attention.kv_in.weight[:, :H], source.kv_a_proj_with_mqa.weight
                    )


def _hidden_config(block_mode="gated"):
    return _config(
        attentive_correction_block_keys=True,
        attentive_correction_block_mode=block_mode,
        attentive_correction_block_kv_in="hidden",
    )


class TestHiddenBlockKeys(_BlockFixture, unittest.TestCase):
    """``attentive_correction_block_kv_in='hidden'``: block keys from the hiddens alone."""

    BLOCK_MODES = ("shared", "gated")

    def _pair(self, block_mode):
        """A module projecting its block keys from [h; e] and one projecting them from h
        alone, every parameter equal and the former's previous-token half of kv_in at zero
        (as it starts); the previous token enters the query (and the gate), the block read
        is non-trivial and the gate open."""
        both = AttentiveCorrection(
            _config(
                attentive_correction_block_keys=True, attentive_correction_block_mode=block_mode
            )
        ).float()
        _randomize(both, 4)
        init_from_attention(both, self.attn)
        _randomize(both.o_proj, 5, scale=0.05)
        g = torch.Generator().manual_seed(6)
        with torch.no_grad():
            half = both.q_in.weight[:, H:]
            half.copy_(torch.randn(half.shape, generator=g) * 0.3)
            if block_mode == "gated":
                both.block_rel_bias.copy_(torch.randn(both.block_rel_bias.shape, generator=g))
                both.block_gate_bias.fill_(0.8)
                gate = both.block_gate.weight
                gate.copy_(torch.randn(gate.shape, generator=g) * 0.05)
        hidden = AttentiveCorrection(_hidden_config(block_mode)).float()
        state = both.state_dict()
        state["kv_in.weight"] = state["kv_in.weight"][:, :H]
        hidden.load_state_dict(state)
        return both, hidden

    def test_validated(self):
        self.assertEqual(_config().attentive_correction_block_kv_in, "hidden_embed")
        with self.assertRaisesRegex(ValueError, "'hidden_embed' or 'hidden'"):
            _config(attentive_correction_block_keys=True, attentive_correction_block_kv_in="h")
        with self.assertRaisesRegex(ValueError, "block_kv_in is what the block keys"):
            _config(attentive_correction_block_kv_in="hidden")
        for block_mode in self.BLOCK_MODES:
            with self.subTest(block_mode=block_mode):
                corr = AttentiveCorrection(_hidden_config(block_mode))
                self.assertEqual(corr.kv_in.in_features, H)
        self.assertEqual(AttentiveCorrection(_gated_config()).kv_in.in_features, 2 * H)

    @torch.no_grad()
    def test_reads_as_a_zero_previous_token_half(self):
        for block_mode in self.BLOCK_MODES:
            with self.subTest(block_mode=block_mode):
                both, hidden = self._pair(block_mode)
                torch.testing.assert_close(self._read(hidden)[0], self._read(both)[0])

    @torch.no_grad()
    def test_a_drafted_token_reaches_only_its_own_slot(self):
        """The token before slot 1 enters slot 1's query and gate and no other slot's read,
        while slot 1's hidden is a key for the slots after it in its block."""
        for block_mode in self.BLOCK_MODES:
            with self.subTest(block_mode=block_mode):
                _, hidden = self._pair(block_mode)
                out, _ = self._read(hidden)
                for inputs, rows in ((self.e, (1,)), (self.h, (1, 2, 3))):
                    saved = inputs.clone()
                    inputs[:, 1] += 1.0
                    moved, _ = self._read(hidden)
                    inputs.copy_(saved)
                    diff = (moved - out).abs().sum(dim=(0, 2))
                    for row in range(self.S):
                        if row in rows:
                            self.assertGreater(float(diff[row]), 0.0, f"row {row}")
                        else:
                            self.assertEqual(float(diff[row]), 0.0, f"row {row}")

    def test_a_fresh_start_or_one_from_acorr_starts_kv_in_as_the_source_kv_a(self):
        from angelspec.training.dflash_trainer import DFlashTrainer

        prefix = "hidden_correction.attention."
        trainer = object.__new__(DFlashTrainer)
        draft = DFlashDraftModel(_hidden_config()).float()
        attention = draft.hidden_correction.attention
        source = draft.layers[-1].self_attn
        block = [
            prefix + n
            for n, _ in attention.named_parameters()
            if n.startswith(("kv_in.", "kv_a_layernorm.", "block_"))
        ]
        for missing in (None, block):
            with self.subTest(fresh=missing is None):
                _randomize(attention, 7)
                DFlashTrainer._init_attentive_correction(trainer, draft, missing)
                torch.testing.assert_close(
                    attention.kv_in.weight, source.kv_a_proj_with_mqa.weight
                )
                torch.testing.assert_close(
                    attention.kv_a_layernorm.weight, source.kv_a_layernorm.weight
                )
                self.assertEqual(float(attention.block_gate.weight.abs().sum()), 0.0)


class TestDFlashModel(unittest.TestCase):
    """The wrapper with the drafter stubbed out: its attention needs the TileLang kernel,
    the correction's dense path does not."""

    L, TOPK = 40, 4
    ANCHORS = torch.tensor([[6, 20, 31]])

    # See _block_fields for the gated variants.
    BLOCK_MODES = (None, "shared", *_GATED_MODES)

    def _wrapper(self, block_mode=None, **kw):
        torch.manual_seed(0)
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
            enable_hidden_correction=True,
            attentive_correction=True,
            **_block_fields(block_mode),
        )
        draft = DFlashDraftModel(config).float()
        wrapper = DFlashModel(
            draft_model=draft, block_size=4, num_anchors=3, causal_attn=True, **kw
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
        self.latents = {layer: _random_latent(target, layer, seed=layer) for layer in (4, 5)}
        wrapper.attach_target_kv_latents([self.latents[4], self.latents[5]])
        correction = draft.hidden_correction
        init_from_attention(correction.attention, draft.layers[-1].self_attn)
        _randomize(correction.attention.o_proj, 8, scale=0.05)
        _randomize(correction.down_proj, 9, scale=0.05)
        if block_mode in _GATED_MODES:
            with torch.no_grad():
                correction.attention.block_gate_bias.fill_(0.5)
        if block_mode == "gated-sep":
            _randomize(correction.attention.block_kv_b_proj, 11, scale=0.05)
        g = torch.Generator().manual_seed(10)
        self.stub_hidden = torch.randn(1, 3 * 4, H, generator=g, requires_grad=True)
        draft.forward = lambda **kwargs: self.stub_hidden
        return wrapper

    def _inputs(self):
        g = torch.Generator().manual_seed(1)
        input_ids = torch.randint(0, 100, (1, self.L), generator=g)
        self.hidden = [torch.randn(1, self.L, H, generator=g) for _ in range(2)]
        return dict(
            input_ids=input_ids,
            hidden_states_list=self.hidden,
            loss_mask=torch.ones(1, self.L),
            lm_head_weight=torch.randn(V, H, generator=g),
            injected_anchors=self.ANCHORS,
        )

    def test_hc_topk_diag_refuses_block_keys(self):
        for block_mode in ("shared", "gated"):
            with (
                self.subTest(block_mode=block_mode),
                self.assertRaisesRegex(ValueError, "block keys"),
            ):
                self._wrapper(block_mode, hc_topk_diag=[4])
        self._wrapper(None, hc_topk_diag=[4])

    def test_context_and_chunks(self):
        for block_mode in self.BLOCK_MODES:
            with self.subTest(block_mode=block_mode):
                self._check_context_and_chunks(block_mode)

    def _check_context_and_chunks(self, block_mode):
        wrapper = self._wrapper(block_mode)
        inputs = self._inputs()
        with torch.no_grad():
            draft_hidden, _, _, n_blocks = wrapper._draft_backbone(
                inputs["input_ids"],
                inputs["hidden_states_list"],
                inputs["loss_mask"],
                injected_anchors=self.ANCHORS,
            )
            context = wrapper._acorr_context
            self.assertTrue(
                torch.equal(
                    context["latent"],
                    self.latents[5].latents(self.hidden[1], inputs["input_ids"]),
                )
            )
            keys = context["indices"]
            self.assertTrue(bool(((keys == -1) | ((keys >= 0) & (keys < self.L))).all()))
            anchors = self.ANCHORS.repeat_interleave(4, dim=1).unsqueeze(-1)
            self.assertTrue(bool(((keys < anchors) | (keys == -1)).all()))
            self.assertTrue(bool((keys >= 0).any(-1).all()))

            prev = torch.randint(0, 100, (1, n_blocks, 4), generator=torch.Generator())
            whole = wrapper._apply_hidden_correction(draft_hidden, prev)
            parts = torch.cat(
                [
                    wrapper._apply_hidden_correction(draft_hidden[:, :4], prev[:, :1], 0),
                    wrapper._apply_hidden_correction(draft_hidden[:, 4:], prev[:, 1:], 1),
                ],
                dim=1,
            )
            torch.testing.assert_close(parts, whole)
            components = wrapper._acorr_components()
            self.assertGreater(float(components["acorr_out_ratio"]), 0.0)
            gates = sorted(k for k in components if k.startswith("acorr_gate_s"))
            if block_mode in _GATED_MODES:
                self.assertEqual(gates, [f"acorr_gate_s{s}" for s in (1, 2, 3)])
                torch.testing.assert_close(
                    components["acorr_gate_s2"], torch.tensor(math.tanh(0.5))
                )
            else:
                self.assertEqual(gates, [])
            self.assertEqual(
                float(components.get("acorr_block_kv_b_drift", 0.0)) > 0.0,
                block_mode == "gated-sep",
            )

            wrapper.eval()
            wrapper.acorr_eval_off = True
            embeds = wrapper.draft_model.embed_tokens(prev).view(1, -1, H)
            torch.testing.assert_close(
                wrapper._apply_hidden_correction(draft_hidden, prev),
                wrapper.draft_model.hidden_correction(draft_hidden, embeds),
            )

    def test_chunked_losses_and_training(self):
        for block_mode in self.BLOCK_MODES:
            with self.subTest(block_mode=block_mode):
                self._check_chunked_losses(block_mode)

    def _check_chunked_losses(self, block_mode):
        losses = {}
        for name, env, checkpoint_chunks in (
            ("whole", "0", False),
            ("chunked", "4", True),
        ):
            with mock.patch.dict(os.environ, {"ANGELSPEC_DFLASH_LOSS_CHUNK": env}):
                wrapper = self._wrapper(
                    block_mode, loss_objective="dpace", loss_checkpoint_chunks=checkpoint_chunks
                )
                out = wrapper(**self._inputs())
            losses[name] = out[0]
            self.assertIn("acorr_out_ratio", out[5], name)
            self.assertEqual("acorr_gate_s1" in out[5], block_mode in _GATED_MODES, name)
            out[0].backward()
            attention = wrapper.draft_model.hidden_correction.attention
            self.assertGreater(float(attention.q_in.weight.grad.abs().sum()), 0.0, name)
        torch.testing.assert_close(losses["chunked"], losses["whole"], rtol=1e-5, atol=1e-6)


@unittest.skipUnless(
    torch.cuda.is_available() and tilelang_dsa_available(), "needs CUDA and the TileLang kernels"
)
class TestOnCuda(unittest.TestCase):
    H, B, S, C, K = 128, 2, 24, 300, 40

    def _module(self, block_mode):
        config = DFlashConfig(
            hidden_size=self.H,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=128,
            rms_norm_eps=1e-6,
            max_position_embeddings=1024,
            num_target_layers=2,
            target_hidden_size=self.H,
            target_num_hidden_layers=6,
            target_layer_ids=[3, 4],
            mask_token_id=127,
            attention_type="mla",
            q_lora_rank=64,
            kv_lora_rank=512,
            qk_nope_head_dim=64,
            qk_rope_head_dim=64,
            v_head_dim=64,
            learnable_sink=True,
            rope_parameters={
                "rope_theta": 10000000,
                "rope_type": "default",
                "mrope_section": [12, 10, 10],
                "mrope_interleaved": True,
            },
            image_start_token_id=124,
            image_end_token_id=125,
            image_newline_token_id=126,
            target_index_layer=[4, 5],
            target_index_topk=16,
            target_kv_layers=[4, 5],
            dsa_backend="tilelang",
            enable_hidden_correction=True,
            attentive_correction=True,
            **_block_fields(block_mode),
        )
        module = AttentiveCorrection(config)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, param in module.named_parameters():
                if param.ndim == 2:
                    noise = torch.randn(param.shape, generator=g) * param.shape[1] ** -0.5
                elif "sink" in name:
                    noise = torch.randn(param.shape, generator=g)
                else:
                    noise = 1 + 0.1 * torch.randn(param.shape, generator=g)
                param.copy_(noise)
        return module.cuda()

    def test_tilelang_matches_the_reference(self):
        for block_mode in (None, "shared", *_GATED_MODES):
            with self.subTest(block_mode=block_mode):
                self._check(block_mode)

    def _check(self, block_mode):
        fast = self._module(block_mode)
        self.assertEqual(fast.backend, "tilelang")
        reference = copy.deepcopy(fast)
        reference.backend = "torch"

        g = torch.Generator().manual_seed(1)
        h = torch.randn(self.B, self.S, self.H, generator=g)
        e = torch.randn(self.B, self.S, self.H, generator=g)
        latent = torch.randn(self.B, self.C, 512 + 64, generator=g)
        pos = torch.randint(self.C, self.C + 40, (self.B, 3, self.S), generator=g)
        idx = torch.full((self.B, self.S, self.K), -1, dtype=torch.long)
        for b in range(self.B):
            for s in range(self.S):
                n = int(torch.randint(1, self.K + 1, (1,), generator=g))
                idx[b, s, :n] = torch.randperm(self.C, generator=g)[:n]
        idx[1, 5] = -1
        weight = torch.randn(self.B, self.S, self.H, generator=g).cuda()

        def run(module):
            hidden = h.cuda().requires_grad_(True)
            embed = e.cuda().requires_grad_(True)
            out, alive = module(hidden, embed, pos.cuda(), latent.cuda(), idx.cuda(), block_size=8)
            (out.float() * weight).sum().backward()
            grads = {n: p.grad.float().cpu() for n, p in module.named_parameters()}
            grads.update(hidden=hidden.grad.float().cpu(), embed=embed.grad.float().cpu())
            return out.detach().float().cpu(), alive.cpu(), grads

        out_fast, alive_fast, grads_fast = run(fast)
        out_ref, alive_ref, grads_ref = run(reference)
        self.assertTrue(torch.equal(alive_fast, alive_ref))
        # The slot with no context key still reads its block with block keys.
        self.assertEqual(bool(alive_fast[1, 5]), block_mode is not None)
        if block_mode is None:
            self.assertEqual(float(out_fast[1, 5].abs().sum()), 0.0)
        rel = ((out_fast - out_ref).norm() / out_ref.norm()).item()
        self.assertLess(rel, 2e-2, f"output: rel={rel:.4f}")
        for name, ref in grads_ref.items():
            a, b = grads_fast[name].flatten(), ref.flatten()
            cos = F.cosine_similarity(a, b, dim=0).item()
            rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
            self.assertGreater(cos, 0.99, f"{name}: cos={cos:.5f}")
            self.assertLess(rel, 5e-2, f"{name}: rel={rel:.4f}")


if __name__ == "__main__":
    unittest.main()
