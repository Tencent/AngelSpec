"""Block-quantized checkpoint weights: FP8 e4m3 with one scale per tile (DeepSeek-V3's
``weight_scale_inv``)."""

from typing import Tuple

import torch


def dequant_fp8_blocks(
    weight: torch.Tensor, scale: torch.Tensor, block: Tuple[int, int] = (32, 32)
) -> torch.Tensor:
    """e4m3 weight with one scale per ``block`` (rows, cols) tile."""
    rows, cols = weight.shape
    expanded = scale.float().repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    return weight.float() * expanded[:rows, :cols]
