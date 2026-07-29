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

"""Unified learning rate schedulers for angelspec training."""

import math
from typing import Literal, Optional

import torch
from torch.optim.lr_scheduler import LRScheduler

from angelspec.utils.logging import logger

DecayStyle = Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"]
WSDDecayStyle = Literal["linear", "cosine", "exponential", "minus_sqrt"]


class LRSchedulerWithWarmup(LRScheduler):
    """Learning rate scheduler with warmup and multiple decay styles.

    Supports: constant, linear, cosine, inverse-square-root, and WSD (Warmup-Stable-Decay).

    Args:
        optimizer: The optimizer to schedule.
        max_lr: Maximum (peak) learning rate after warmup.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps (linear warmup from init_lr to max_lr).
        min_lr: Minimum learning rate after decay. Defaults to 0.0.
        init_lr: Initial learning rate at step 0. Defaults to 0.0.
        decay_style: Decay style after warmup. One of: constant, linear, cosine,
            inverse-square-root, WSD. Defaults to "cosine".
        wsd_decay_steps: For WSD style, number of steps for final decay phase.
        wsd_decay_style: For WSD style, decay curve in final phase.
        last_epoch: The index of last epoch. Defaults to -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        init_lr: float = 0.0,
        decay_style: DecayStyle = "cosine",
        wsd_decay_steps: Optional[int] = None,
        wsd_decay_style: Optional[WSDDecayStyle] = None,
        last_epoch: int = -1,
    ) -> None:
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.init_lr = float(init_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.decay_style = decay_style
        self.wsd_decay_steps = wsd_decay_steps
        self.wsd_decay_style = wsd_decay_style

        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr
        assert self.total_steps > 0
        assert self.warmup_steps < self.total_steps

        if self.decay_style == "WSD":
            assert self.wsd_decay_steps is not None, "WSD decay requires wsd_decay_steps"

        super().__init__(optimizer, last_epoch)

        logger.debug(
            f"LR scheduler: decay_style={decay_style}, warmup={warmup_steps}, total={total_steps}"
        )

    def _get_lr_for_group(self, param_group: dict) -> float:
        """Compute learning rate for a specific parameter group."""
        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)
        step = self.last_epoch

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.init_lr + (max_lr - self.init_lr) * step / self.warmup_steps

        if self.decay_style == "constant":
            return max_lr

        if step > self.total_steps:
            return min_lr

        if self.decay_style == "inverse-square-root":
            warmup_steps = max(self.warmup_steps, 1)
            num_steps = max(step, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = step - self.warmup_steps
        decay_steps_ = self.total_steps - self.warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        decay_ratio = max(0.0, min(1.0, decay_ratio))

        delta_lr = max_lr - min_lr

        if self.decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.decay_style == "WSD":
            coeff = self._compute_wsd_coeff(step, decay_ratio)
        else:
            raise ValueError(f"Unknown decay style: {self.decay_style}")

        return min_lr + coeff * delta_lr

    def _compute_wsd_coeff(self, step: int, decay_ratio: float) -> float:
        """Compute coefficient for WSD (Warmup-Stable-Decay) schedule."""
        wsd_anneal_start = self.total_steps - self.wsd_decay_steps
        if step <= wsd_anneal_start:
            return 1.0

        wsd_steps = step - wsd_anneal_start
        wsd_ratio = float(wsd_steps) / float(self.wsd_decay_steps)

        if self.wsd_decay_style == "linear":
            return 1.0 - wsd_ratio
        elif self.wsd_decay_style == "cosine":
            return 0.5 * (math.cos(math.pi * wsd_ratio) + 1.0)
        elif self.wsd_decay_style == "exponential":
            return (2.0 * math.pow(0.5, wsd_ratio)) - 1.0
        elif self.wsd_decay_style == "minus_sqrt":
            return 1.0 - math.sqrt(wsd_ratio)
        else:
            raise ValueError(f"Unknown WSD decay style: {self.wsd_decay_style}")

    def get_lr(self) -> list[float]:
        return [self._get_lr_for_group(group) for group in self.optimizer.param_groups]
