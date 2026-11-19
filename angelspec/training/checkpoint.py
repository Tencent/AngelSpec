# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from angelspec.training.optimizer import _is_muon_matrix
from angelspec.utils.logging import logger

_WRAPPER_FQN_TOKENS = ("_checkpoint_wrapped_module.", "_orig_mod.", "_fsdp_wrapped_module.")
# DCP raises planning and read failures as CheckpointException, a BaseException.
_LOAD_ERRORS = (Exception, CheckpointException)

_FP32_KEY = re.compile(r"optim_state\.fp32_params\.(\d+)")
_MUON_KEY = re.compile(r"optim_state\.optim\.muon_bufs\.(\d+)")
_ADAMW_STATE_KEY = re.compile(r"optim_state\.(optim(?:\.adamw)?)\.state\.(\d+)\.(\w+)")


def _normalize_fqn(name: str) -> str:
    for token in _WRAPPER_FQN_TOKENS:
        name = name.replace(token, "")
    return name


def frozen_target_embedding_keys(model) -> frozenset[str]:
    """FQNs of frozen ``embed_tokens`` weights, i.e. copies of the current target's.

    They are reloaded from the target on every start, so checkpoints neither save
    nor restore them; otherwise resuming against a different target would bring
    back the old target's embedding.
    """
    return frozenset(
        _normalize_fqn(name)
        for name, param in model.named_parameters()
        if name.endswith("embed_tokens.weight") and not param.requires_grad
    )


def _remap_metadata_keys(metadata: Any, remap_key: Callable[[str], str]) -> None:
    """Rename flat FQNs in a DCP ``Metadata`` in place via ``remap_key``.

    ``dcp.load`` resolves keys against ``metadata.state_dict_metadata`` while
    building the load plan, and against ``metadata.storage_data`` while locating
    the serialized bytes. Both are indexed by the checkpoint's saved FQNs, so
    both must be updated for remapped keys to be found. All keys are renamed at
    once, so a remap that shifts indices (``3 -> 2`` while ``2 -> 1``) is safe.
    """
    if metadata is None:
        return

    state_dict_metadata = getattr(metadata, "state_dict_metadata", None)
    if isinstance(state_dict_metadata, dict):
        renamed = {}
        for old_key, value in state_dict_metadata.items():
            new_key = remap_key(old_key)
            if new_key in renamed:
                raise ValueError(f"Two checkpoint entries remap to {new_key!r}")
            renamed[new_key] = value
        state_dict_metadata.clear()
        state_dict_metadata.update(renamed)

    storage_data = getattr(metadata, "storage_data", None)
    if isinstance(storage_data, dict):
        renamed = {}
        for old_index, value in storage_data.items():
            new_fqn = remap_key(old_index.fqn)
            if new_fqn != old_index.fqn:
                old_index = MetadataIndex(new_fqn, old_index.offset, old_index.index)
            renamed[old_index] = value
        storage_data.clear()
        storage_data.update(renamed)


class KeyRemapLoadPlanner(DefaultLoadPlanner):
    """Load planner that transparently renames checkpoint FQNs before matching.

    A ``load_state_dict`` pre-hook registered on a module only fires for
    ``nn.Module.load_state_dict``, which the DCP ``set_state_dict`` path never
    calls. Instead, DCP matches keys earlier, when building the load plan. This
    planner applies ``remap_key`` to the checkpoint metadata keys before that
    matching happens, so legacy FQNs (e.g. bare fused-expert parameters) load
    into their new module names (e.g. ``...experts_gate_proj.weight``).

    ``remap_key`` is a caller-supplied, structure-specific function; this class
    stays model-agnostic so unrelated models are never forced through a remap.
    """

    def __init__(self, remap_key: Callable[[str], str]):
        super().__init__()
        self._remap_key = remap_key

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        _remap_metadata_keys(metadata, self._remap_key)
        super().set_up_planner(state_dict, metadata, is_coordinator)


class ModelState(Stateful):
    """Wrapper for model state only.

    ``skip_keys`` are excluded from the saved state and are not restored on load.
    """

    def __init__(self, model, skip_keys: frozenset[str] = frozenset()):
        self.model = model
        self.skip_keys = skip_keys

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        if self.skip_keys:
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if k not in self.skip_keys
            }
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            optimizers=[],
            model_state_dict=state_dict["model"],
            optim_state_dict=None,
            options=StateDictOptions(strict=not self.skip_keys),
        )


class OptimizerState(Stateful):
    """Wrapper for optimizer + fp32 master params.

    Saving writes the complete index-keyed layout. Loading a ``BF16Optimizer``
    goes through a ``_BF16LoadPlan``, which names exactly the entries the
    checkpoint holds for the current parameters.
    """

    def __init__(self, model, optimizer, plan: Optional[_BF16LoadPlan] = None):
        self.model = model
        self.optimizer = optimizer
        self._is_bf16_optimizer = hasattr(optimizer, "sync_fp32_params_from_model")
        self._plan = plan

    def state_dict(self):
        if self._plan is not None:
            return self._plan.skeleton
        if self._is_bf16_optimizer:
            return {
                "optim": self.optimizer.state_dict(),
                "fp32_params": {str(i): p.data for i, p in enumerate(self.optimizer.fp32_params)},
            }
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=[self.optimizer])
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        if self._plan is not None:
            self._plan.install(self.optimizer, state_dict)
            return
        if self._is_bf16_optimizer:
            self.optimizer.load_state_dict(state_dict["optim"])
            with torch.no_grad():
                for i, mp in enumerate(self.optimizer.fp32_params):
                    mp.data.copy_(state_dict["fp32_params"][str(i)])
            return
        set_state_dict(
            self.model,
            optimizers=[self.optimizer],
            model_state_dict=None,
            optim_state_dict=state_dict["optim"],
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def _unused_key(key: str) -> str:
    return "optim_state.__unused__." + key[len("optim_state.") :]


@dataclass
class _BF16LoadPlan:
    """Checkpoint entries a ``BF16Optimizer`` load restores, keyed by current index.

    ``fp32`` and ``muon`` map a current parameter index to its saved index.
    ``adamw`` maps a current AdamW-group position to its saved position and the
    saved state fields (name -> (shape, dtype)). Saved entries outside the plan
    are renamed out of the way, so a shifted index never lands on the wrong
    parameter.
    """

    state_path: str
    fp32: dict[int, int] = field(default_factory=dict)
    muon: dict[int, int] = field(default_factory=dict)
    adamw: dict[int, tuple[int, dict[str, tuple[tuple[int, ...], torch.dtype]]]] = field(
        default_factory=dict
    )
    skeleton: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._fp32_source = {saved: current for current, saved in self.fp32.items()}
        self._muon_source = {saved: current for current, saved in self.muon.items()}
        self._adamw_source = {saved: current for current, (saved, _) in self.adamw.items()}

    @property
    def empty(self) -> bool:
        return not (self.fp32 or self.muon or self.adamw)

    def remap_key(self, key: str) -> str:
        for pattern, sources, template in (
            (_FP32_KEY, self._fp32_source, "optim_state.fp32_params.{}"),
            (_MUON_KEY, self._muon_source, "optim_state.optim.muon_bufs.{}"),
        ):
            match = pattern.fullmatch(key)
            if match:
                current = sources.get(int(match.group(1)))
                return template.format(current) if current is not None else _unused_key(key)
        match = _ADAMW_STATE_KEY.fullmatch(key)
        if match and match.group(1) == self.state_path:
            current = self._adamw_source.get(int(match.group(2)))
            if current is None:
                return _unused_key(key)
            return f"optim_state.{self.state_path}.state.{current}.{match.group(3)}"
        return key

    def build_skeleton(self, optimizer) -> None:
        skeleton: dict[str, Any] = {}
        if self.fp32:
            skeleton["fp32_params"] = {str(i): optimizer.fp32_params[i].data for i in self.fp32}
        state = {}
        for pos, (_, fields) in self.adamw.items():
            master = optimizer.fp32_params[optimizer.adamw_indices[pos]]
            # A scalar "step" stays on the CPU, as AdamW keeps it when not capturable.
            state[pos] = {
                name: (
                    torch.zeros((), dtype=dtype)
                    if size == ()
                    else torch.zeros_like(master, dtype=dtype)
                )
                for name, (size, dtype) in fields.items()
            }
        optim: dict[str, Any] = {}
        if self.state_path == "optim.adamw":
            if state:
                optim["adamw"] = {"state": state}
            if self.muon:
                optim["muon_bufs"] = {str(i): optimizer.muon_bufs[i] for i in self.muon}
        elif state:
            optim["state"] = state
        if optim:
            skeleton["optim"] = optim
        self.skeleton = skeleton

    def install(self, optimizer, state_dict: dict) -> None:
        # fp32 masters and Muon buffers were loaded in place; only AdamW state is new.
        optim = state_dict.get("optim", {})
        if self.state_path == "optim.adamw":
            optim = optim.get("adamw", {})
        state = optim.get("state", {})
        if not state:
            return
        inner = optimizer.optimizer
        # Keep the configured hyperparameters: the scheduler sets the LR, and the
        # betas and weight decay follow the current config.
        inner.load_state_dict(
            {
                "state": {int(pos): value for pos, value in state.items()},
                "param_groups": inner.state_dict()["param_groups"],
            }
        )


def _infer_saved_param_names(model, saved_shapes: list[tuple[int, ...]]) -> list[str] | None:
    """Recover which parameters a checkpoint without parameter names saved.

    Old checkpoints key optimizer entries by position among the parameters that
    were trainable when they were written. Every currently trainable parameter
    must appear in that list, and a currently frozen one may or may not (e.g.
    the router ``expert_bias`` became frozen after such checkpoints were
    written). Returns the names only when exactly one assignment matches every
    saved shape in order.
    """
    params = [
        (_normalize_fqn(name), param.requires_grad, tuple(param.shape))
        for name, param in model.named_parameters()
    ]
    num_params, num_saved = len(params), len(saved_shapes)
    # ways[m][j]: number of ways (capped at 2) to align params[m:] with saved_shapes[j:].
    ways = [[0] * (num_saved + 2) for _ in range(num_params + 1)]
    ways[num_params][num_saved] = 1
    for m in range(num_params - 1, -1, -1):
        _, trainable, shape = params[m]
        row, below = ways[m], ways[m + 1]
        for j in range(num_saved, -1, -1):
            total = below[j + 1] if j < num_saved and saved_shapes[j] == shape else 0
            if not trainable:
                total += below[j]
            row[j] = min(total, 2)
    if ways[0][0] != 1:
        return None
    names, j = [], 0
    for m, (name, _, shape) in enumerate(params):
        if j < num_saved and saved_shapes[j] == shape and ways[m + 1][j + 1] == 1:
            names.append(name)
            j += 1
    return names


def _plan_bf16_optimizer_load(
    optimizer,
    metadata: Any,
    saved_names: list[str] | None,
    *,
    include_state: bool,
    saved_adamw_names: list[str] | None = None,
) -> tuple[_BF16LoadPlan | None, str]:
    """Match checkpoint entries to the current parameters.

    Entries are matched by the parameter names recorded at save time; checkpoints
    written before names were recorded match by position when the parameter
    count is unchanged, and otherwise by aligning the saved shapes. AdamW state
    follows ``saved_adamw_names``, the saved AdamW order, when recorded; earlier
    checkpoints had a single AdamW group in parameter order. An entry whose shape
    differs from the current parameter is left out. Returns ``(plan,
    description)``; ``plan`` is None when nothing can be matched.
    """
    saved_fp32: dict[int, tuple[int, ...]] = {}
    saved_muon: dict[int, tuple[int, ...]] = {}
    saved_state: dict[str, dict[int, dict[str, tuple[tuple[int, ...], torch.dtype]]]] = {}
    for key, value in (getattr(metadata, "state_dict_metadata", None) or {}).items():
        if not isinstance(value, TensorStorageMetadata):
            continue
        size = tuple(int(x) for x in value.size)
        if match := _FP32_KEY.fullmatch(key):
            saved_fp32[int(match.group(1))] = size
        elif match := _MUON_KEY.fullmatch(key):
            saved_muon[int(match.group(1))] = size
        elif match := _ADAMW_STATE_KEY.fullmatch(key):
            fields = saved_state.setdefault(match.group(1), {}).setdefault(int(match.group(2)), {})
            fields[match.group(3)] = (size, value.properties.dtype)

    current_names = [_normalize_fqn(name) for name in optimizer.param_names]
    current_shapes = [tuple(p.shape) for p in optimizer.fp32_params]
    num_saved = len(saved_fp32)
    contiguous = sorted(saved_fp32) == list(range(num_saved))
    how = "parameter names"
    if saved_names is None and contiguous:
        if num_saved == len(current_names):
            how, saved_names = "position", current_names
        else:
            how = "shape alignment"
            saved_names = _infer_saved_param_names(
                optimizer.model, [saved_fp32[j] for j in range(num_saved)]
            )
    if saved_names is None:
        return None, (
            f"its {num_saved} fp32 masters cannot be matched one-to-one to the "
            f"{len(current_names)} trainable parameters"
        )

    saved_index = {name: j for j, name in enumerate(saved_names)}
    mapping = {i: saved_index[name] for i, name in enumerate(current_names) if name in saved_index}
    fp32 = {i: j for i, j in mapping.items() if saved_fp32.get(j) == current_shapes[i]}

    current_is_muon = optimizer.optimizer_type == "muon"
    saved_is_muon = "optim.adamw" in saved_state or bool(saved_muon)
    state_path = "optim.adamw" if current_is_muon else "optim"
    muon: dict[int, int] = {}
    adamw: dict[int, tuple[int, dict]] = {}
    if include_state and current_is_muon == saved_is_muon:
        if current_is_muon:
            for i in optimizer.muon_indices:
                j = mapping.get(i)
                if j is not None and saved_muon.get(j) == tuple(optimizer.muon_bufs[i].shape):
                    muon[i] = j
            saved_adamw = [
                j
                for j, name in enumerate(saved_names)
                if not _is_muon_matrix(name, torch.empty(saved_fp32.get(j, ()), device="meta"))
            ]
        else:
            saved_adamw = list(range(len(saved_names)))
        if saved_adamw_names is not None:
            saved_position = {
                saved_index[name]: pos
                for pos, name in enumerate(saved_adamw_names)
                if name in saved_index
            }
        else:
            saved_position = {j: pos for pos, j in enumerate(saved_adamw)}
        saved_fields = saved_state.get(state_path, {})
        for pos, i in enumerate(optimizer.adamw_indices):
            j = mapping.get(i)
            if j is None or j not in saved_position:
                continue
            fields = saved_fields.get(saved_position[j])
            if fields and all(size in ((), current_shapes[i]) for size, _ in fields.values()):
                adamw[pos] = (saved_position[j], fields)

    plan = _BF16LoadPlan(state_path=state_path, fp32=fp32, muon=muon, adamw=adamw)
    plan.build_skeleton(optimizer)
    parts = [f"fp32 masters {len(fp32)}/{len(current_names)}"]
    if include_state:
        parts.append(f"AdamW state {len(adamw)}/{len(optimizer.adamw_indices)}")
        if current_is_muon:
            parts.append(f"Muon momentum {len(muon)}/{len(optimizer.muon_indices)}")
        if current_is_muon != saved_is_muon:
            parts.append("optimizer type changed, so its state starts fresh")
    return plan, f"{', '.join(parts)} (matched by {how})"


def _load_bf16_optimizer(
    actor: Any,
    optimizer_dir: Path,
    saved_names: list[str] | None,
    *,
    include_state: bool,
    saved_adamw_names: list[str] | None = None,
) -> None:
    """Restore what ``optimizer_dir`` holds for the current parameters.

    The caller has already synced the fp32 masters from the loaded model, so an
    entry this cannot restore keeps the model's weights rather than the masters
    cloned before the model was loaded.
    """
    optimizer = actor.optimizer
    strict_opd = getattr(actor.args, "mtp_draft_opd", False)
    fallback = "fp32 master weights come from the loaded model and optimizer state starts fresh"
    try:
        metadata = dcp.FileSystemReader(str(optimizer_dir)).read_metadata()
        plan, description = _plan_bf16_optimizer_load(
            optimizer,
            metadata,
            saved_names,
            include_state=include_state,
            saved_adamw_names=saved_adamw_names,
        )
    except _LOAD_ERRORS as e:
        if strict_opd:
            raise RuntimeError(f"Cannot read OPD optimizer checkpoint {optimizer_dir}") from e
        logger.warning(f"Cannot read optimizer checkpoint {optimizer_dir} ({e}); {fallback}.")
        return
    if strict_opd and (
        plan is None
        or len(plan.fp32) != len(optimizer.fp32_params)
        or len(plan.adamw) != len(optimizer.adamw_indices)
        or any(
            not {"step", "exp_avg", "exp_avg_sq"} <= fields.keys()
            for _, fields in plan.adamw.values()
        )
    ):
        raise RuntimeError(f"Incomplete OPD optimizer checkpoint {optimizer_dir}: {description}")
    if plan is None or plan.empty:
        logger.warning(
            f"Optimizer checkpoint {optimizer_dir} not restored: {description}; {fallback}."
        )
        return
    try:
        dcp.load(
            state_dict={"optim_state": OptimizerState(actor.model, optimizer, plan=plan)},
            checkpoint_id=str(optimizer_dir),
            planner=KeyRemapLoadPlanner(plan.remap_key),
        )
    except _LOAD_ERRORS as e:
        if strict_opd:
            raise RuntimeError(f"Failed to load OPD optimizer checkpoint {optimizer_dir}") from e
        optimizer.optimizer.state.clear()
        for buf in optimizer.muon_bufs.values():
            buf.zero_()
        optimizer.sync_fp32_params_from_model()
        logger.warning(f"Failed to load optimizer from {optimizer_dir} ({e}); {fallback}.")
        return
    logger.info(f"Loaded optimizer from {optimizer_dir}: {description}")


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(metadata, indent=2, sort_keys=True))


def _complete_iterations(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("iter_*") if (p / "model").is_dir())


def _restore_lr_scheduler(actor: Any, lr_scheduler_dir: Path, metadata: dict[str, Any]) -> None:
    scheduler = actor.lr_scheduler
    reason = None
    if lr_scheduler_dir.exists():
        try:
            dcp.load(
                state_dict={"lr_scheduler_state": LRSchedulerState(scheduler)},
                checkpoint_id=str(lr_scheduler_dir),
            )
            logger.info(f"Loaded LR scheduler from {lr_scheduler_dir}")
        except _LOAD_ERRORS as e:
            reason = f"failed to load {lr_scheduler_dir}: {e}"
    else:
        reason = f"{lr_scheduler_dir} not found"

    if reason is not None:
        if getattr(actor.args, "mtp_draft_opd", False):
            raise RuntimeError(f"OPD LR scheduler must be restored: {reason}")
        step = metadata.get("global_step")
        if step is None or not hasattr(scheduler, "_get_lr_for_group"):
            logger.warning(f"LR scheduler not restored ({reason}); the schedule restarts.")
            return
        # LRSchedulerWithWarmup is a pure function of the step count.
        scheduler.last_epoch = int(step)
        scheduler._last_lr = scheduler.get_lr()
        logger.warning(f"LR scheduler not restored ({reason}); recomputed it at step {step}.")

    switch_step = getattr(actor.args, "data_switch_step", None)
    if switch_step is not None and metadata.get("global_step") == switch_step:
        _replan_lr_decay(scheduler, int(actor.args.lr_total_steps))

    # The first resumed step must use the restored LR, not the one set at construction.
    for group, lr in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr()):
        group["lr"] = lr


def _replan_lr_decay(scheduler: Any, total_steps: int) -> None:
    """Decay from the restored LR to min_lr by ``total_steps``, keeping the decay style.

    The restored scheduler state carries the old total_steps, so a shortened run would
    otherwise end far above min_lr. Treating the steps so far as warmup makes the current
    LR the peak of the remaining decay; the new fields persist in later checkpoints.
    """
    step = scheduler.last_epoch
    lrs = scheduler.get_lr()
    if total_steps <= step or len(set(lrs)) != 1:
        raise ValueError(
            f"Cannot re-plan the LR decay at step {step} to end at {total_steps} (lrs={lrs})"
        )
    scheduler.max_lr = lrs[0]
    scheduler.warmup_steps = step
    scheduler.total_steps = total_steps
    scheduler._last_lr = scheduler.get_lr()
    logger.info(
        f"LR decay re-planned at step {step}: {lrs[0]:.3e} -> {scheduler.min_lr:.1e} "
        f"({scheduler.decay_style}) by step {total_steps}"
    )


def load(actor: Any, key_remap_fn: Optional[Callable[[str], str]] = None) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.

    ``key_remap_fn`` is an optional, structure-specific FQN renamer applied to the
    checkpoint's saved keys before DCP builds its load plan. It is only needed to
    accept legacy checkpoints whose flat keys no longer match the current module
    layout (e.g. the MTP fused-expert rename). Leave it ``None`` for the default
    (no remap) behavior.

    A ``load_path`` that does not exist yet, or holds no tracker file, starts
    training from scratch. One that names a checkpoint that cannot be loaded
    raises instead of silently starting over.
    """
    load_root = getattr(actor.args, "load_path", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.warning(
            f"Checkpoint directory {root_path} not found; training starts from scratch."
        )
        return None

    # Accept load_path pointing either at a checkpoints ROOT (containing
    # latest_checkpointed_iteration.txt + iter_XXX/) or directly at a single
    # iter_XXX dir (containing model/). The latter is the natural thing to pass
    # and previously fell through to a silent "no tracker file" skip.
    if (root_path / "model").exists():
        checkpoint_dir = root_path
        # Recover the step from the iter_XXXXXXX dir name for continual-training
        # global_step restore; fall back to ckpt_step / None.
        target_step = getattr(actor.args, "ckpt_step", None)
        if target_step is None and root_path.name.startswith("iter_"):
            try:
                target_step = int(root_path.name.split("_")[-1])
            except ValueError:
                target_step = None
    else:
        target_step = getattr(actor.args, "ckpt_step", None)
        if target_step is None:
            tracker_file = root_path / "latest_checkpointed_iteration.txt"
            if not tracker_file.exists():
                logger.warning(f"No tracker file at {tracker_file}; training starts from scratch.")
                return None
            tracker_text = tracker_file.read_text().strip()
            try:
                target_step = int(tracker_text)
            except ValueError:
                raise ValueError(
                    f"Checkpoint tracker {tracker_file} holds {tracker_text!r}, not a step"
                ) from None
        checkpoint_dir = root_path / f"iter_{target_step:07d}"

    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        available = _complete_iterations(checkpoint_dir.parent)
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_dir} has no model/ directory; refusing to start from "
            f"scratch with load_path={root_path}. Complete checkpoints there: "
            f"{available[-5:] or 'none'} (iter_N holds N-1 completed steps)."
        )

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    skip_keys = frozen_target_embedding_keys(actor.model)
    if skip_keys:
        logger.info(
            f"Keeping current target embedding; not restoring {sorted(skip_keys)} from checkpoint"
        )
    model_state = ModelState(actor.model, skip_keys=skip_keys)
    try:
        planner = KeyRemapLoadPlanner(key_remap_fn) if key_remap_fn is not None else None
        dcp.load(
            state_dict={"model_state": model_state},
            checkpoint_id=str(model_dir),
            planner=planner,
        )
    except _LOAD_ERRORS as e:
        raise RuntimeError(f"Failed to load model from {model_dir}: {e}") from e
    logger.info(f"Loaded model from {model_dir}")

    continual_training = getattr(actor.args, "continual_training", False)
    optimizer = getattr(actor, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "sync_fp32_params_from_model"):
        # The fp32 masters were cloned from the weights the model had before this
        # load, and every optimizer step copies them back into the model.
        optimizer.sync_fp32_params_from_model()
        if optimizer_dir.exists():
            # Continual training starts a fresh optimizer but keeps the exact masters.
            _load_bf16_optimizer(
                actor,
                optimizer_dir,
                metadata.get("optimizer_param_names"),
                include_state=not continual_training,
                saved_adamw_names=metadata.get("optimizer_adamw_param_names"),
            )
        else:
            logger.warning(
                f"Optimizer checkpoint not found at {optimizer_dir}; fp32 master weights "
                "come from the loaded model and optimizer state starts fresh."
            )
    elif optimizer is not None and not continual_training and optimizer_dir.exists():
        try:
            dcp.load(
                state_dict={"optim_state": OptimizerState(actor.model, optimizer)},
                checkpoint_id=str(optimizer_dir),
            )
            logger.info(f"Loaded optimizer from {optimizer_dir}")
        except _LOAD_ERRORS as e:
            logger.warning(f"Failed to load optimizer from {optimizer_dir}: {e}")

    # Continual training keeps the freshly configured LR schedule.
    if getattr(actor, "lr_scheduler", None) is not None and not continual_training:
        _restore_lr_scheduler(actor, lr_scheduler_dir, metadata)

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
        "optimizer_dir": optimizer_dir,
    }


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    continual_training = getattr(actor.args, "continual_training", False)

    if checkpoint_payload.get("rng") is not None and not continual_training:
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if not continual_training:
        actor.resume_data_position = metadata.get("data_position")
    if metadata and not continual_training:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        next_step = metadata.get("next_step") or metadata.get("next_inference_id")
        if next_step is not None:
            actor.args.start_step = next_step
    elif iteration is not None and not continual_training:
        if getattr(actor.args, "start_step", None) is None:
            actor.args.start_step = iteration

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, step: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    base_dir = Path(actor.args.checkpoint_dir).expanduser()
    step_id = step + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model_state = ModelState(actor.model, skip_keys=frozen_target_embedding_keys(actor.model))
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    optimizer = getattr(actor, "optimizer", None)
    if optimizer is not None:
        optimizer_state = OptimizerState(actor.model, optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    if hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "step": step,
            "inference_id": step,  # compat: old checkpoints use this key
            "next_step": step + 1,
            "next_inference_id": step + 1,  # compat: old checkpoints use this key
            "global_step": actor.global_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        if optimizer is not None and hasattr(optimizer, "param_names"):
            # Optimizer entries are saved by position; the names let a later load
            # match them even when the set of trainable parameters changes.
            metadata["optimizer_param_names"] = [
                _normalize_fqn(name) for name in optimizer.param_names
            ]
            if hasattr(optimizer, "adamw_indices"):
                # AdamW keys its state by position in its own parameter order, which
                # LR-scale groups make differ from the order above.
                metadata["optimizer_adamw_param_names"] = [
                    _normalize_fqn(optimizer.param_names[i]) for i in optimizer.adamw_indices
                ]
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        _write_text_atomic(tracker_file, str(step_id))
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
