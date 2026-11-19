"""Resume round trips through checkpoint.save / checkpoint.load (CPU, one gloo rank)."""

import json
import math
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from angelspec.training import checkpoint
from angelspec.training.optimizer import BF16Optimizer


@pytest.fixture(scope="module", autouse=True)
def _process_group():
    if dist.is_initialized():
        yield
        return
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group("gloo", init_method=f"file://{tmp}/pg", rank=0, world_size=1)
        yield
        dist.destroy_process_group()


class _MoE(nn.Module):
    """``expert_bias`` never receives a gradient, like the MTP router bias."""

    def __init__(self, freeze_bias: bool = False, extra: bool = False):
        super().__init__()
        if extra:
            self.extra = nn.Parameter(torch.ones(3))
        self.expert_bias = nn.Parameter(torch.zeros(4), requires_grad=not freeze_bias)
        self.norm = nn.Parameter(torch.ones(8))
        self.gate = nn.Linear(8, 4, bias=False)
        self.up = nn.Linear(8, 16, bias=False)
        self.down = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        out = self.down(torch.relu(self.up(x))) * self.norm + self.gate(x).sum(-1, keepdim=True)
        if hasattr(self, "extra"):
            out = out + self.extra.sum()
        return out


def _build(
    seed,
    *,
    optimizer_type="adamw",
    betas=(0.9, 0.999),
    weight_decay=0.0,
    lr_scales=None,
    **model_kw,
):
    torch.manual_seed(seed)
    model = _MoE(**model_kw).to(torch.bfloat16)
    optimizer = BF16Optimizer(
        model,
        lr=1e-2,
        total_steps=100,
        warmup_ratio=0.1,
        betas=betas,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        lr_scales=lr_scales,
    )
    return model, optimizer


def _train(model, optimizer, steps, seed):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(4, 8, generator=gen).to(torch.bfloat16)
        model(x).float().pow(2).mean().backward()
        optimizer.step()


def _actor(model, optimizer, root, *, load_path=None, continual=False, global_step=0):
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        lr_scheduler=optimizer.lr_scheduler,
        global_step=global_step,
        args=SimpleNamespace(
            checkpoint_dir=str(root), load_path=load_path, continual_training=continual
        ),
    )


def _save(model, optimizer, root, steps=3):
    checkpoint.save(_actor(model, optimizer, root, global_step=steps), step=steps)
    return root / f"iter_{steps + 1:07d}"


def _resume(model, optimizer, root, **kw):
    actor = _actor(model, optimizer, root, load_path=str(root), **kw)
    payload = checkpoint.load(actor)
    checkpoint.finalize_load(actor, payload)
    return actor


def _drop_recorded_names(iteration_dir):
    meta_path = iteration_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("optimizer_param_names")
    meta_path.write_text(json.dumps(meta))


def _by_name(optimizer):
    """Per-parameter-name fp32 master, AdamW state and Muon buffer."""
    inner = optimizer.optimizer.state_dict()["state"]
    adamw_position = {i: pos for pos, i in enumerate(optimizer.adamw_indices)}
    out = {}
    for i, name in enumerate(optimizer.param_names):
        entry = {"fp32": optimizer.fp32_params[i].detach().clone()}
        pos = adamw_position.get(i)
        if pos is not None and pos in inner:
            entry.update({k: v.detach().clone() for k, v in inner[pos].items()})
        if i in optimizer.muon_bufs:
            entry["muon"] = optimizer.muon_bufs[i].detach().clone()
        out[name] = entry
    return out


def _assert_same_state(expected, actual, names):
    for name in names:
        assert expected[name].keys() == actual[name].keys(), name
        for key, value in expected[name].items():
            assert torch.equal(value, actual[name][key]), f"{name}.{key}"


@pytest.mark.parametrize("optimizer_type", ["adamw", "muon"])
def test_resume_restores_optimizer_and_continues_identically(tmp_path, optimizer_type):
    model_a, opt_a = _build(0, optimizer_type=optimizer_type)
    _train(model_a, opt_a, 3, seed=1)
    _save(model_a, opt_a, tmp_path)

    model_b, opt_b = _build(1, optimizer_type=optimizer_type)
    actor = _resume(model_b, opt_b, tmp_path)

    assert actor.global_step == 3
    _assert_same_state(_by_name(opt_a), _by_name(opt_b), opt_a.param_names)
    assert opt_b.get_learning_rate() == opt_a.get_learning_rate()

    _train(model_a, opt_a, 1, seed=2)
    _train(model_b, opt_b, 1, seed=2)
    for (name, a), b in zip(model_a.named_parameters(), model_b.parameters()):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("optimizer_type", ["adamw", "muon"])
def test_resume_with_lr_scale_groups(tmp_path, optimizer_type):
    """LR-scale groups put AdamW's parameters out of parameter order."""
    scales = {"expert_bias": 0.5, "gate.weight": 0.25}
    model_a, opt_a = _build(0, optimizer_type=optimizer_type, lr_scales=scales)
    adamw_names = [opt_a.param_names[i] for i in opt_a.adamw_indices]
    assert adamw_names != [n for n in opt_a.param_names if n in adamw_names]
    _train(model_a, opt_a, 3, seed=1)
    iteration = _save(model_a, opt_a, tmp_path)
    meta = json.loads((iteration / "meta.json").read_text())
    assert meta["optimizer_adamw_param_names"] == adamw_names

    model_b, opt_b = _build(1, optimizer_type=optimizer_type, lr_scales=scales)
    _resume(model_b, opt_b, tmp_path)
    _assert_same_state(_by_name(opt_a), _by_name(opt_b), opt_a.param_names)
    for scale in (1.0, 0.5, 0.25):
        assert opt_b.get_learning_rate(scale) == opt_a.get_learning_rate(scale)

    _train(model_a, opt_a, 1, seed=2)
    _train(model_b, opt_b, 1, seed=2)
    for (name, a), b in zip(model_a.named_parameters(), model_b.parameters()):
        assert torch.equal(a, b), name


def test_resume_without_optimizer_dir_keeps_loaded_weights(tmp_path):
    model_a, opt_a = _build(0)
    _train(model_a, opt_a, 3, seed=1)
    iteration = _save(model_a, opt_a, tmp_path)
    shutil.rmtree(iteration / "optimizer")

    model_b, opt_b = _build(1)
    _resume(model_b, opt_b, tmp_path)
    for p in model_b.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    opt_b.step()
    for (name, a), b in zip(model_a.named_parameters(), model_b.parameters()):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("optimizer_type", ["adamw", "muon"])
@pytest.mark.parametrize("recorded_names", [True, False])
def test_resume_after_freezing_a_parameter(tmp_path, optimizer_type, recorded_names):
    """Checkpoints written while expert_bias was trainable (before 3f0e6ca)."""
    model_a, opt_a = _build(0, optimizer_type=optimizer_type)
    _train(model_a, opt_a, 3, seed=1)
    iteration = _save(model_a, opt_a, tmp_path)
    if not recorded_names:
        _drop_recorded_names(iteration)

    model_b, opt_b = _build(1, optimizer_type=optimizer_type, freeze_bias=True)
    _resume(model_b, opt_b, tmp_path)

    assert "expert_bias" not in opt_b.param_names
    _assert_same_state(_by_name(opt_a), _by_name(opt_b), opt_b.param_names)

    _train(model_a, opt_a, 1, seed=2)
    _train(model_b, opt_b, 1, seed=2)
    for (name, a), b in zip(model_a.named_parameters(), model_b.parameters()):
        assert torch.equal(a, b), name


def test_resume_follows_configured_betas_and_weight_decay(tmp_path):
    model_a, opt_a = _build(0, betas=(0.9, 0.999), weight_decay=0.0)
    _train(model_a, opt_a, 3, seed=1)
    _save(model_a, opt_a, tmp_path)

    model_b, opt_b = _build(1, betas=(0.9, 0.95), weight_decay=0.1)
    _resume(model_b, opt_b, tmp_path)

    group = opt_b.param_groups[0]
    assert tuple(group["betas"]) == (0.9, 0.95)
    assert group["weight_decay"] == 0.1
    _assert_same_state(_by_name(opt_a), _by_name(opt_b), opt_a.param_names)


def test_unmatched_optimizer_checkpoint_falls_back_to_model_weights(tmp_path):
    model_a, opt_a = _build(0, extra=True)
    _train(model_a, opt_a, 3, seed=1)
    iteration = _save(model_a, opt_a, tmp_path)
    _drop_recorded_names(iteration)
    # The model checkpoint must match; only the optimizer layout is unknown.
    shutil.rmtree(iteration / "model")
    model_ref, _ = _build(0)
    model_ref.load_state_dict({k: v for k, v in model_a.state_dict().items() if k != "extra"})
    checkpoint.dcp.save(
        {"model_state": checkpoint.ModelState(model_ref)}, checkpoint_id=str(iteration / "model")
    )

    model_b, opt_b = _build(1)
    _resume(model_b, opt_b, tmp_path)

    assert not opt_b.optimizer.state
    for p, master in zip(opt_b.model_params, opt_b.fp32_params):
        assert torch.equal(master, p.float())


def test_resume_exposes_saved_data_position(tmp_path):
    model_a, opt_a = _build(0)
    iteration = _save(model_a, opt_a, tmp_path)
    model_b, opt_b = _build(1)
    assert _resume(model_b, opt_b, tmp_path).resume_data_position is None

    meta = json.loads((iteration / "meta.json").read_text())
    meta["data_position"] = {"dataset_epoch": 1, "epoch_consumed_samples": 7}
    (iteration / "meta.json").write_text(json.dumps(meta))
    actor = _resume(model_b, opt_b, tmp_path)
    assert actor.resume_data_position == meta["data_position"]


def test_lr_scheduler_recomputed_when_missing(tmp_path):
    model_a, opt_a = _build(0)
    _train(model_a, opt_a, 3, seed=1)
    iteration = _save(model_a, opt_a, tmp_path)
    shutil.rmtree(iteration / "lr_scheduler")

    model_b, opt_b = _build(1)
    _resume(model_b, opt_b, tmp_path)
    assert opt_b.lr_scheduler.last_epoch == 3
    assert opt_b.get_learning_rate() == opt_a.get_learning_rate()


def test_data_switch_replans_lr_decay_from_the_restored_lr(tmp_path):
    def resume(seed):
        model, opt = _build(seed)
        actor = _actor(model, opt, tmp_path, load_path=str(tmp_path))
        actor.args.data_switch_step = 20
        actor.args.lr_total_steps = 60
        checkpoint.finalize_load(actor, checkpoint.load(actor))
        return model, opt

    model_a, opt_a = _build(0)
    _train(model_a, opt_a, 20, seed=1)
    _save(model_a, opt_a, tmp_path, steps=20)
    lr_at_switch = opt_a.get_learning_rate()

    model_b, opt_b = resume(1)
    assert opt_b.get_learning_rate() == pytest.approx(lr_at_switch)
    _train(model_b, opt_b, 10, seed=2)
    lr_at_30 = opt_b.get_learning_rate()
    assert lr_at_30 == pytest.approx(0.5 * lr_at_switch * (1 + math.cos(math.pi * 10 / 40)))

    # A later resume keeps the re-planned schedule instead of re-planning again.
    _save(model_b, opt_b, tmp_path, steps=30)
    model_c, opt_c = resume(2)
    assert opt_c.get_learning_rate() == pytest.approx(lr_at_30)
    _train(model_c, opt_c, 30, seed=3)
    assert opt_c.get_learning_rate() == pytest.approx(0.0, abs=1e-12)


def test_load_path_errors(tmp_path):
    model, opt = _build(0)
    missing = _actor(model, opt, tmp_path, load_path=str(tmp_path / "missing"))
    assert checkpoint.load(missing) is None

    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    with pytest.raises(FileNotFoundError, match="iter_0000007"):
        checkpoint.load(_actor(model, opt, tmp_path, load_path=str(tmp_path)))

    (tmp_path / "latest_checkpointed_iteration.txt").write_text("")
    with pytest.raises(ValueError, match="not a step"):
        checkpoint.load(_actor(model, opt, tmp_path, load_path=str(tmp_path)))


def test_remap_metadata_keys_handles_shifted_indices():
    keys = {f"k.{i}": f"v{i}" for i in range(4)}
    metadata = SimpleNamespace(state_dict_metadata=dict(keys), storage_data={})
    # Shift every index up by one; the last entry has no destination.
    checkpoint._remap_metadata_keys(
        metadata, lambda k: f"k.{int(k[2:]) + 1}" if int(k[2:]) < 3 else "dropped.3"
    )
    assert metadata.state_dict_metadata == {
        "k.1": "v0",
        "k.2": "v1",
        "k.3": "v2",
        "dropped.3": "v3",
    }


def test_infer_saved_param_names_requires_a_unique_alignment():
    class _Twins(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.zeros(4), requires_grad=False)
            self.b = nn.Parameter(torch.zeros(4), requires_grad=False)
            self.c = nn.Parameter(torch.zeros(2))

    # Either frozen twin could have been the saved one.
    assert checkpoint._infer_saved_param_names(_Twins(), [(4,), (2,)]) is None
    assert checkpoint._infer_saved_param_names(_Twins(), [(4,), (4,), (2,)]) == ["a", "b", "c"]
    assert checkpoint._infer_saved_param_names(_Twins(), [(3,), (2,)]) is None
