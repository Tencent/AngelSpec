"""Native Hy4 flash Hadamard/FP8 coarse scoring for gradient-free draft selection."""

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _native_ops():
    from vllm.models.hy_v4_flash.nvidia.indexpool_ops import fwht_quantize
    from vllm.utils.deep_gemm import (
        fp8_fp4_mqa_logits,
        mqa_logits_mx_inputs,
        mqa_logits_use_mx_scales,
    )

    return fwht_quantize, fp8_fp4_mqa_logits, mqa_logits_mx_inputs, mqa_logits_use_mx_scales


@torch.no_grad()
def quantize_keys(keys):
    quant, _, _, mx_scales = _native_ops()
    values, scale = quant(keys.bfloat16())
    if mx_scales():
        scale = (scale.view(torch.int32) >> 23) * 0x01010101
    return values, scale


@torch.no_grad()
def fp8_coarse_logits(q, keys, weights, ends=None):
    """``[N, len(keys)]`` FP32 scores, as vLLM's IndexPool scores a recall (heads padded
    to the kernel's 32 or 64); columns at or past a row's end are -inf. Query chunks
    bound temporary allocations."""
    quant, logits_fn, mx_inputs, _ = _native_ops()
    k, ks = keys
    n, h, d = q.shape
    if not n or not len(k):
        return torch.empty((n, len(k)), device=q.device, dtype=torch.float32)
    if ends is None:
        ends = torch.full((n,), len(k), device=q.device, dtype=torch.int32)
    heads = 32 if h <= 32 else 64
    outputs = []
    chunk = max(1, min(n, (16 << 20) // max(4 * len(k), 1)))
    for begin in range(0, n, chunk):
        end = min(begin + chunk, n)
        qb, qs = quant(q[begin:end].reshape(-1, d).bfloat16())
        qf = F.pad(qb.view(torch.uint8).view(end - begin, h, d), (0, 0, 0, heads - h))
        qf = qf.view(torch.float8_e4m3fn)
        w = F.pad(weights[begin:end].float() * qs.reshape(end - begin, h), (0, heads - h))
        q_scale, w = mx_inputs(qf, None, w, torch.float32)
        starts = torch.zeros(end - begin, device=q.device, dtype=torch.int32)
        limits = ends[begin:end].int().clamp(1, len(k)).contiguous()
        width = int(limits.max())
        logits = logits_fn((qf, q_scale), (k, ks), w, starts, limits, width)[:, :width]
        columns = torch.arange(width, device=q.device)
        logits = logits.masked_fill(columns[None] >= limits[:, None], -torch.inf)
        outputs.append(F.pad(logits, (0, len(k) - width), value=-torch.inf))
    return torch.cat(outputs)
