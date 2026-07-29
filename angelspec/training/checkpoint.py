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

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from angelspec.utils.logging import logger


class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None
        )


class OptimizerState(Stateful):
    """Wrapper for optimizer + fp32 master params."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self._is_bf16_optimizer = hasattr(optimizer, "sync_fp32_params_from_model")

    def state_dict(self):
        if self._is_bf16_optimizer:
            return {
                "optim": self.optimizer.state_dict(),
                "fp32_params": {str(i): p.data for i, p in enumerate(self.optimizer.fp32_params)},
            }
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=[self.optimizer])
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        if self._is_bf16_optimizer:
            self.optimizer.load_state_dict(state_dict["optim"])
            if "fp32_params" in state_dict:
                with torch.no_grad():
                    for i, mp in enumerate(self.optimizer.fp32_params):
                        mp.data.copy_(state_dict["fp32_params"][str(i)])
                logger.info("Restored fp32 master params from checkpoint")
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


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    load_root = getattr(actor.args, "load_path", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"No tracker file at {tracker_file}; skipping load.")
            return None
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        logger.info(f"Model checkpoint {model_dir} not found; skipping load.")
        return None

    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"Loaded model from {model_dir}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {e}")
        return None

    continual_training = getattr(actor.args, "continual_training", False)

    # Keep optimizer/LR state out of continual training so it starts fresh.
    load_optimizer = not continual_training and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            logger.warning(f"Failed to load optimizer from {optimizer_dir}: {e}")
    elif load_optimizer:
        logger.info(f"Optimizer checkpoint not found at {optimizer_dir}, skipping optimizer load.")

    # Load LR scheduler state (optional)
    load_lr_scheduler = (
        not continual_training and hasattr(actor, "lr_scheduler") and lr_scheduler_dir.exists()
    )
    if load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"Loaded LR scheduler from {lr_scheduler_dir}")
        except Exception as e:
            logger.warning(f"Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif hasattr(actor, "lr_scheduler"):
        logger.info(
            f"LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping LR scheduler load."
        )

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
        "optimizer_dir": optimizer_dir,
    }


def _restore_fp32_master_params(actor: Any, optim_dir: Path) -> None:
    """Sync BF16Optimizer's fp32 master params after model-only checkpoint load.

    When optimizer state is skipped for continual training, the fp32 master copies
    still hold pre-checkpoint (random init) values.  The first optimizer step
    would copy these back to the model, overwriting the loaded weights.

    Strategy: temporarily load the optimizer checkpoint to recover the fp32 master
    params, then restore the freshly configured param_groups and clear Adam state
    so continual training still uses the new optimizer hyperparameters. Falls
    back to copying from the bf16 model weights if the optimizer checkpoint is
    unavailable.
    """
    opt = actor.optimizer
    if not hasattr(opt, "fp32_params"):
        return

    if optim_dir.exists() and (optim_dir / ".metadata").exists():
        try:
            fresh_param_groups = [
                {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
                for group in opt.optimizer.param_groups
            ]
            optim_state = OptimizerState(actor.model, opt)
            optim_sd = {"optim_state": optim_state}
            dcp.load(state_dict=optim_sd, checkpoint_id=str(optim_dir))
            for group, fresh_group in zip(opt.optimizer.param_groups, fresh_param_groups):
                params = group["params"]
                group.clear()
                group.update(copy.deepcopy(fresh_group))
                group["params"] = params
            opt.optimizer.state.clear()
            logger.info(f"Loaded fp32 master params from {optim_dir}")
            return
        except Exception as e:
            logger.warning(f"Failed to load fp32 params from optimizer checkpoint: {e}")

    if hasattr(opt, "sync_fp32_params_from_model"):
        opt.sync_fp32_params_from_model()
        logger.info("Synced optimizer fp32 master params from bf16 model weights (lossy)")


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
    if metadata and not continual_training:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        next_step = metadata.get("next_step") or metadata.get("next_inference_id")
        if next_step is not None:
            actor.args.start_step = next_step
    elif iteration is not None and not continual_training:
        if getattr(actor.args, "start_step", None) is None:
            actor.args.start_step = iteration

    if continual_training and hasattr(actor, "optimizer"):
        _restore_fp32_master_params(actor, checkpoint_payload["optimizer_dir"])

    torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, step: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
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

    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    if hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    if hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
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
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
