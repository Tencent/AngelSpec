"""Block-quantized checkpoint weights: FP8 e4m3 with one scale per tile (DeepSeek-V3's
``weight_scale_inv``, DeepSeek-V4's ``scale``) and DeepSeek-V4's packed FP4 e2m1 experts."""

from typing import Tuple

import torch

FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def dequant_fp8_blocks(
    weight: torch.Tensor, scale: torch.Tensor, block: Tuple[int, int] = (32, 32)
) -> torch.Tensor:
    """e4m3 weight with one scale per ``block`` (rows, cols) tile."""
    rows, cols = weight.shape
    expanded = scale.float().repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    return weight.float() * expanded[:rows, :cols]


def dequant_fp4_packed(weight: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Two e2m1 values per byte (low nibble first) with one scale per ``block`` along K."""
    table = torch.tensor(FP4_E2M1_VALUES, dtype=torch.float32, device=weight.device)
    packed = weight.view(torch.uint8)
    values = torch.stack([table[(packed & 0x0F).long()], table[(packed >> 4).long()]], dim=-1)
    values = values.flatten(-2)
    return values * scale.float().repeat_interleave(block, 1)[:, : values.shape[1]]
