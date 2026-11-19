"""Tests for BF16Optimizer's two weight-decay groups.

Under ``optimizer_type="muon"`` the >=2-D weight matrices are decayed by
``muon_weight_decay`` and everything else (norms, gates, embeddings, LM head) by
``weight_decay``. Leaving ``muon_weight_decay`` unset must reproduce the earlier
single-value behaviour exactly, so existing configs and in-flight runs are
unaffected.
"""

import copy
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn

from angelspec.training.muon_utils import (
    adjust_lr_for_muon,
    zeropower_via_newtonschulz5,
)
from angelspec.training.optimizer import BF16Optimizer

_TMPDIR = None


def setUpModule():
    # The Muon branch logs through print_on_rank0, which calls dist.get_rank()
    # unconditionally; a single-rank gloo group is enough to construct it.
    global _TMPDIR
    if not dist.is_initialized():
        _TMPDIR = tempfile.TemporaryDirectory()
        dist.init_process_group(
            backend="gloo", init_method=f"file://{_TMPDIR.name}/pg", world_size=1, rank=0
        )


def tearDownModule():
    if _TMPDIR is not None:
        dist.destroy_process_group()
        _TMPDIR.cleanup()


class _Tiny(nn.Module):
    """One Muon-eligible matrix (``proj``) plus one AdamW-only 1-D param."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)
        self.norm = nn.Parameter(torch.ones(8))


def _build(**kw):
    torch.manual_seed(0)
    model = _Tiny()
    opt = BF16Optimizer(
        model, lr=1e-2, total_steps=100, warmup_ratio=0.0, optimizer_type="muon", **kw
    )
    return model, opt


def _step(model, opt, seed=1):
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g)
    opt.step()


class TestMuonWeightDecay(unittest.TestCase):
    def test_external_controller_updates_model_and_fp32_master(self):
        model, opt = _build(weight_decay=0.0)
        before = model.norm.detach().clone()
        delta = torch.linspace(-0.2, 0.2, model.norm.numel())
        opt.add_to_parameter("norm", delta)
        torch.testing.assert_close(model.norm, before + delta)
        index = opt.param_name_to_index["norm"]
        torch.testing.assert_close(opt.fp32_params[index], (before + delta).float())

    def test_unset_inherits_weight_decay(self):
        _, opt = _build(weight_decay=0.05)
        self.assertEqual(opt.muon_weight_decay, 0.05)
        self.assertEqual(opt.weight_decay, 0.05)

    def test_unset_reproduces_the_single_value_behaviour(self):
        # The regression guard: passing the same value explicitly must land on the
        # same weights as not passing it at all.
        m_default, o_default = _build(weight_decay=0.05)
        m_explicit, o_explicit = _build(weight_decay=0.05, muon_weight_decay=0.05)
        m_explicit.load_state_dict(copy.deepcopy(m_default.state_dict()))
        o_explicit.sync_fp32_params_from_model()

        _step(m_default, o_default)
        _step(m_explicit, o_explicit)
        torch.testing.assert_close(m_default.proj.weight, m_explicit.proj.weight)

    def test_muon_group_uses_its_own_decay(self):
        _, opt = _build(weight_decay=0.0, muon_weight_decay=0.1)
        self.assertEqual(opt.muon_weight_decay, 0.1)
        self.assertEqual(opt.weight_decay, 0.0)
        # The inner AdamW keeps the AdamW-group value, not the Muon one.
        for group in opt.optimizer.param_groups:
            self.assertEqual(group["weight_decay"], 0.0)

    def test_decay_is_applied_to_the_matrix_only(self):
        # Zero LR isolates the decay term: the Muon update scales the matrix by
        # (1 - lr*wd) before the Newton-Schulz step, so with wd=0 it is untouched
        # and with wd>0 it shrinks. The 1-D param is on AdamW either way.
        decayed, opt_decayed = _build(weight_decay=0.0, muon_weight_decay=0.1)
        plain, opt_plain = _build(weight_decay=0.0)
        plain.load_state_dict(copy.deepcopy(decayed.state_dict()))
        opt_plain.sync_fp32_params_from_model()
        before = decayed.proj.weight.detach().clone()

        _step(decayed, opt_decayed)
        _step(plain, opt_plain)

        self.assertFalse(torch.allclose(decayed.proj.weight, plain.proj.weight))
        # Shrunk towards zero relative to the undecayed run.
        self.assertLess(decayed.proj.weight.norm().item(), plain.proj.weight.norm().item())
        self.assertFalse(torch.allclose(decayed.proj.weight, before))

    def test_adamw_optimizer_type_is_unaffected(self):
        torch.manual_seed(0)
        model = _Tiny()
        opt = BF16Optimizer(
            model,
            lr=1e-2,
            total_steps=100,
            warmup_ratio=0.0,
            weight_decay=0.0,
            muon_weight_decay=0.1,
            optimizer_type="adamw",
        )
        # No Muon group exists, so the setting cannot reach any parameter.
        self.assertEqual(opt.muon_indices, [])
        for group in opt.optimizer.param_groups:
            self.assertEqual(group["weight_decay"], 0.0)


class _Pair(nn.Module):
    """Two same-shape matrices, so one can be LR-scaled against the other."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8, bias=False)
        self.b = nn.Linear(8, 8, bias=False)
        self.norm = nn.Parameter(torch.ones(8))


def _build_pair(optimizer_type="adamw", warmup_ratio=0.1, **kw):
    torch.manual_seed(0)
    model = _Pair()
    opt = BF16Optimizer(
        model,
        lr=1e-2,
        min_lr=1e-3,
        total_steps=100,
        warmup_ratio=warmup_ratio,
        optimizer_type=optimizer_type,
        **kw,
    )
    return model, opt


class TestLRScales(unittest.TestCase):
    def test_no_scales_keep_one_plain_group(self):
        for optimizer_type in ("adamw", "muon"):
            _, opt = _build_pair(optimizer_type)
            self.assertEqual(len(opt.optimizer.param_groups), 1)
            self.assertNotIn("lr_scale", opt.optimizer.param_groups[0])

    def test_scaled_group_follows_the_whole_curve(self):
        model, opt = _build_pair(lr_scales={"b.weight": 0.2})
        base, scaled = opt.optimizer.param_groups
        self.assertEqual((base["lr_scale"], scaled["lr_scale"]), (1.0, 0.2))
        b_master = opt.fp32_params[opt.param_name_to_index["b.weight"]]
        self.assertEqual([id(p) for p in scaled["params"]], [id(b_master)])
        base_lrs, scaled_lrs = [], []
        for _ in range(120):  # warmup, cosine decay, then past total_steps
            base_lrs.append(base["lr"])
            scaled_lrs.append(scaled["lr"])
            _step(model, opt)
        for b, s in zip(base_lrs, scaled_lrs):
            self.assertAlmostEqual(s, 0.2 * b, places=12)
        self.assertAlmostEqual(max(base_lrs), 1e-2)
        self.assertAlmostEqual(base_lrs[-1], 1e-3)
        self.assertAlmostEqual(opt.get_learning_rate(), base["lr"])
        self.assertAlmostEqual(opt.get_learning_rate(0.2), scaled["lr"])

    def _scaled_update_ratio(self, optimizer_type):
        model, opt = _build_pair(
            optimizer_type,
            warmup_ratio=0.0,
            weight_decay=0.0,
            lr_scales={"b.weight": 0.2},
        )
        with torch.no_grad():
            model.b.weight.copy_(model.a.weight)
        opt.sync_fp32_params_from_model()
        before = model.a.weight.detach().clone()
        grad = torch.randn(8, 8, generator=torch.Generator().manual_seed(3))
        model.a.weight.grad = grad.clone()
        model.b.weight.grad = grad.clone()
        model.norm.grad = torch.zeros(8)
        opt.step()
        moved_a = model.a.weight.detach() - before
        moved_b = model.b.weight.detach() - before
        self.assertGreater(moved_a.abs().max().item(), 0.0)
        torch.testing.assert_close(moved_b, 0.2 * moved_a, rtol=1e-3, atol=1e-7)

    def test_adamw_step_is_scaled(self):
        self._scaled_update_ratio("adamw")

    def test_muon_step_is_scaled(self):
        _, opt = _build_pair("muon", lr_scales={"b.weight": 0.2})
        self.assertEqual(len(opt.optimizer.param_groups), 1)  # both matrices are on Muon
        self._scaled_update_ratio("muon")

    def test_restore_keeps_the_configured_scale(self):
        for optimizer_type in ("adamw", "muon"):
            model, opt = _build_pair(optimizer_type, lr_scales={"norm": 0.2})
            _step(model, opt)
            state = copy.deepcopy(opt.state_dict())
            _, resumed = _build_pair(optimizer_type, lr_scales={"norm": 0.1})
            resumed.load_state_dict(state)
            scales = [g["lr_scale"] for g in resumed.optimizer.param_groups]
            self.assertEqual(scales, [1.0, 0.1])

    def test_rejects_unknown_names_and_negative_scales(self):
        with self.assertRaisesRegex(ValueError, "no trainable parameter"):
            _build_pair(lr_scales={"missing.weight": 0.5})
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            _build_pair(lr_scales={"b.weight": -0.5})

    def test_a_zero_scale_holds_the_parameter_still(self):
        """Zero is a value here, not a mistake: it holds a param by learning rate rather than
        ``requires_grad``. Weight decay rides on the same scale, so the param really does
        not move."""
        model, opt = _build_pair(lr_scales={"b.weight": 0.0})
        opt.sync_fp32_params_from_model()
        before = model.b.weight.detach().clone()
        grad = torch.randn(8, 8, generator=torch.Generator().manual_seed(5))
        model.a.weight.grad = grad.clone()
        model.b.weight.grad = grad.clone()
        model.norm.grad = torch.zeros(8)
        opt.step()
        torch.testing.assert_close(model.b.weight.detach(), before, rtol=0, atol=0)
        self.assertGreater((model.a.weight.detach() - before).abs().max().item(), 0.0)


E_, IN_, OUT_ = 3, 4, 6
LR_, WD_ = 1e-2, 0.1


class _Stack(nn.Module):
    """A fused-expert-like [E, in, out] stack plus a 1-D param for the AdamW group."""

    def __init__(self):
        super().__init__()
        self.experts = nn.Parameter(torch.randn(E_, IN_, OUT_))
        self.norm = nn.Parameter(torch.ones(IN_))


def _build_stack(per_expert=None, model=None):
    torch.manual_seed(0)
    model = model or _Stack()
    kwargs = {} if per_expert is None else {"muon_per_expert": per_expert}
    opt = BF16Optimizer(
        model,
        lr=LR_,
        # No clipping: Newton-Schulz then sees exactly the grads the expected values use.
        max_grad_norm=1e9,
        total_steps=100,
        warmup_ratio=0.0,
        weight_decay=0.0,
        muon_weight_decay=WD_,
        optimizer_type="muon",
        **kwargs,
    )
    return model, opt


def _stack_grad():
    return torch.randn(E_, IN_, OUT_, generator=torch.Generator().manual_seed(5))


class TestMuonPerExpert(unittest.TestCase):
    def _step(self, model, opt, grad):
        model.experts.grad = grad.clone()
        model.norm.grad = torch.zeros(IN_)
        opt.step()

    def test_each_expert_is_orthogonalized_on_its_own_by_default(self):
        model, opt = _build_stack()
        before = model.experts.detach().clone()
        grad = _stack_grad()
        self._step(model, opt, grad)
        # First step: the momentum buffer is the grad, and Nesterov adds momentum * buffer
        # (same op as the optimizer, so the bf16 Newton-Schulz input matches bit for bit).
        update = grad.add(grad, alpha=0.95)
        adjusted = adjust_lr_for_muon(LR_, (IN_, OUT_), 0.2)
        expected = torch.stack(
            [
                before[e] * (1 - LR_ * WD_)
                - adjusted * zeropower_via_newtonschulz5(update[e], steps=5).float()
                for e in range(E_)
            ]
        )
        torch.testing.assert_close(model.experts.detach(), expected, rtol=1e-3, atol=1e-4)

    def test_flattening_stays_available(self):
        model, opt = _build_stack(per_expert=False)
        before = model.experts.detach().clone()
        grad = _stack_grad()
        self._step(model, opt, grad)
        flat = grad.add(grad, alpha=0.95).view(E_, -1)
        u = zeropower_via_newtonschulz5(flat, steps=5).float().view(E_, IN_, OUT_)
        expected = before * (1 - LR_ * WD_) - adjust_lr_for_muon(LR_, flat.shape, 0.2) * u
        torch.testing.assert_close(model.experts.detach(), expected, rtol=1e-3, atol=1e-4)

    def test_training_config_turns_it_on(self):
        import dataclasses

        from angelspec.config.train_config import TrainingConfig

        defaults = {f.name: f.default for f in dataclasses.fields(TrainingConfig)}
        self.assertIs(defaults["muon_per_expert"], True)

    def test_sharded_stack_updates_its_local_experts_in_place(self):
        # A one-rank mesh puts the whole stack in the local shard, which runs the
        # DTensor path (to_local + in-place update) against the plain-tensor one.
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        mesh = init_device_mesh("cpu", (1,))
        plain, plain_opt = _build_stack(per_expert=True)
        torch.manual_seed(0)
        sharded = _Stack()
        for name in ("experts", "norm"):
            value = getattr(plain, name).detach().clone()
            setattr(sharded, name, nn.Parameter(distribute_tensor(value, mesh, [Shard(0)])))
        _, sharded_opt = _build_stack(per_expert=True, model=sharded)

        grad = _stack_grad()
        self._step(plain, plain_opt, grad)
        sharded.experts.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
        sharded.norm.grad = distribute_tensor(torch.zeros(IN_), mesh, [Shard(0)])
        sharded_opt.step()
        torch.testing.assert_close(sharded.experts.full_tensor(), plain.experts.detach())


HEADS_, HEAD_ROWS_, COLS_ = 3, 4, 6


class _HeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(COLS_, HEADS_ * HEAD_ROWS_, bias=False)

    def muon_head_rows(self):
        return {"q.weight": HEAD_ROWS_}


class _HeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _HeadAttention()
        self.norm = nn.Parameter(torch.ones(COLS_))


def _build_heads(model=None):
    from angelspec.training.optimizer import muon_head_rows

    torch.manual_seed(0)
    model = model or _HeadModel()
    opt = BF16Optimizer(
        model,
        lr=LR_,
        max_grad_norm=1e9,
        total_steps=100,
        warmup_ratio=0.0,
        weight_decay=0.0,
        muon_weight_decay=WD_,
        optimizer_type="muon",
        muon_head_rows=muon_head_rows(model),
    )
    return model, opt


class TestMuonPerHead(unittest.TestCase):
    def _grad(self):
        return torch.randn(HEADS_ * HEAD_ROWS_, COLS_, generator=torch.Generator().manual_seed(7))

    def test_each_head_is_orthogonalized_on_its_own(self):
        from angelspec.training.optimizer import muon_head_rows

        self.assertEqual(muon_head_rows(_HeadModel()), {"attn.q.weight": HEAD_ROWS_})
        model, opt = _build_heads()
        before = model.attn.q.weight.detach().clone()
        grad = self._grad()
        model.attn.q.weight.grad = grad.clone()
        model.norm.grad = torch.zeros(COLS_)
        opt.step()
        update = grad.add(grad, alpha=0.95).view(HEADS_, HEAD_ROWS_, COLS_)
        adjusted = adjust_lr_for_muon(LR_, (HEAD_ROWS_, COLS_), 0.2)
        expected = before * (1 - LR_ * WD_) - adjusted * torch.cat(
            [zeropower_via_newtonschulz5(update[h], steps=5).float() for h in range(HEADS_)]
        )
        torch.testing.assert_close(model.attn.q.weight.detach(), expected, rtol=1e-3, atol=1e-4)

    def test_sharded_matrix_matches_the_plain_one(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        mesh = init_device_mesh("cpu", (1,))
        plain, plain_opt = _build_heads()
        torch.manual_seed(0)
        sharded = _HeadModel()
        sharded.attn.q.weight = nn.Parameter(
            distribute_tensor(plain.attn.q.weight.detach().clone(), mesh, [Shard(0)])
        )
        sharded.norm = nn.Parameter(
            distribute_tensor(plain.norm.detach().clone(), mesh, [Shard(0)])
        )
        _, sharded_opt = _build_heads(model=sharded)

        grad = self._grad()
        plain.attn.q.weight.grad = grad.clone()
        plain.norm.grad = torch.zeros(COLS_)
        plain_opt.step()
        sharded.attn.q.weight.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
        sharded.norm.grad = distribute_tensor(torch.zeros(COLS_), mesh, [Shard(0)])
        sharded_opt.step()
        torch.testing.assert_close(
            sharded.attn.q.weight.full_tensor(), plain.attn.q.weight.detach()
        )

    def test_rejects_rows_that_do_not_split(self):
        with self.assertRaisesRegex(ValueError, "does not split into heads"):
            BF16Optimizer(
                _HeadModel(),
                lr=LR_,
                total_steps=10,
                optimizer_type="muon",
                muon_head_rows={"attn.q.weight": 5},
            )


class TestMuonExclusions(unittest.TestCase):
    def test_dspark_markov_head_stays_on_adamw(self):
        from angelspec.training.optimizer import _is_muon_matrix

        for name in ("markov_head.markov_w1.weight", "markov_head.markov_w2.weight"):
            self.assertFalse(_is_muon_matrix(name, torch.empty(16, 4)))
        self.assertTrue(_is_muon_matrix("layers.0.attn.wq_b.weight", torch.empty(16, 4)))


class TestFp32GradLifetime(unittest.TestCase):
    def test_fp32_grads_are_released_after_step(self):
        for optimizer_type in ("muon", "adamw"):
            torch.manual_seed(0)
            model = _Tiny()
            opt = BF16Optimizer(
                model, lr=1e-2, total_steps=100, warmup_ratio=0.0, optimizer_type=optimizer_type
            )
            before = [p.detach().clone() for p in model.parameters()]
            _step(model, opt)
            self.assertTrue(all(mp.grad is None for mp in opt.fp32_params))
            for old, new in zip(before, model.parameters()):
                self.assertFalse(torch.equal(old, new))


class TestMTPMoELRScales(unittest.TestCase):
    def test_router_scale_reaches_the_muon_router(self):
        from angelspec.models.draft.mtp import MTPDraftModel
        from angelspec.training.optimizer import moe_lr_scales
        from tests.test_mtp import _tiny_config

        draft = MTPDraftModel(_tiny_config(), attention_backend="sdpa")
        scales = moe_lr_scales(draft, 1.0, router_scale=0.1)
        router = "midlayer.mlp.router.gate.weight"
        self.assertEqual(scales[router], 0.1)
        self.assertIn("midlayer.mlp.experts_gate_proj.weight", scales)
        self.assertIn("midlayer.post_attention_layernorm.weight", scales)
        self.assertTrue(all(scale == 1.0 for name, scale in scales.items() if name != router))
        self.assertFalse(any("self_attn" in name or "eh_proj" in name for name in scales))

        opt = BF16Optimizer(
            draft,
            lr=1e-3,
            total_steps=10,
            warmup_ratio=0.0,
            optimizer_type="muon",
            lr_scales=scales,
        )
        index = opt.param_name_to_index[router]
        self.assertIn(index, opt.muon_indices)
        self.assertEqual(opt.lr_scales[index], 0.1)


if __name__ == "__main__":
    unittest.main()
