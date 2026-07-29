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

import math
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig

from angelspec.models.draft.base import Eagle3DraftModel
from angelspec.models.ops.flex_attention import (
    compile_friendly_flex_attention,
    eagle3_block_mask,
    generate_eagle3_mask,
)
from angelspec.utils.distributed import get_sp_ring_group, get_sp_ulysses_group
from angelspec.utils.logging import logger, print_with_rank

_flash_attn_import_error: ImportError | None = None
try:
    # cutlass-dsl >= 4.4.2 removed cutlass.utils.ampere_helpers, but FA4
    # still imports it for SMEM_CAPACITY. Inject a shim before any FA4 import.
    import importlib  # noqa: E401
    import importlib.util

    if importlib.util.find_spec("cutlass.utils") and not importlib.util.find_spec(
        "cutlass.utils.ampere_helpers"
    ):
        import types  # noqa: E401

        import cutlass.utils

        _shim = types.ModuleType("cutlass.utils.ampere_helpers")
        _shim.SMEM_CAPACITY = {
            "sm80": (164 - 1) * 1024,
            "sm86": (100 - 1) * 1024,
            "sm89": (100 - 1) * 1024,
        }
        import sys

        sys.modules["cutlass.utils.ampere_helpers"] = _shim

    from flash_attn.cute import flash_attn_func as _flash_attn_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
except ImportError as _e:
    _flash_attn_import_error = _e
    _flash_attn_func = None
    _flash_attn_fwd = None
    _flash_attn_bwd = None


def _import_standard_flash_attn():
    try:
        import flash_attn as mod
        from flash_attn import flash_attn_varlen_func as varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import (
            _flash_attn_backward as backward,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_forward as forward,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_backward as varlen_backward,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_forward as varlen_forward,
        )
    except ImportError as exc:
        return None, None, None, None, None, None, None, None, exc

    return (
        mod,
        varlen_func,
        pad_input,
        unpad_input,
        forward,
        backward,
        varlen_forward,
        varlen_backward,
        None,
    )


(
    _std_flash_attn_mod,
    _std_flash_attn_varlen_func,
    _std_flash_pad_input,
    _std_flash_unpad_input,
    _std_flash_attn_forward,
    _std_flash_attn_backward,
    _std_flash_attn_varlen_forward,
    _std_flash_attn_varlen_backward,
    _std_flash_attn_import_error,
) = _import_standard_flash_attn()


def _raise_standard_flash_attn_unavailable() -> None:
    raise RuntimeError(
        "LlamaFlashAttention requires the standard flash-attn interface "
        f"(import error: {_std_flash_attn_import_error!r})"
    )


try:
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils as fa_cute_utils
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    _has_cute_dsl = True

    def _patch_cutlass_compilation() -> None:
        """Two patches to reduce flash_attn cute DSL compilation overhead.

        Patch 1 — Disk cache (BaseDSL.compile_and_cache):
            flash_attn hardcodes no_cache=True, bypassing CUTE_DSL_CACHE_DIR.
            Overriding to False lets compiled cubins be written on the first run
            and read back on every subsequent process start (seconds, not minutes).
            Set CUTE_DSL_CACHE_DIR to a persistent path to survive reboots:
                export CUTE_DSL_CACHE_DIR=/scratch/$USER/cutlass_cache

        Patch 2 — Compilation opt-level (Compiler._compile):
            flash_attn passes options="--enable-tvm-ffi" with no explicit
            --opt-level, so the parser default (3) is used. Lower levels reduce
            LLVM-IR and ptxas work at the cost of slightly slower kernels.

            Controlled by ANGELSPEC_FLASH_ATTN_OPT_LEVEL (default: 3):
              3  – current default; ~12 min; fastest kernel
              2  – ~6–8 min; <5% slower kernel  (recommended for training)
              1  – ~3–5 min; ~15% slower kernel
              0  – ~1–2 min; ~50% slower kernel  (debugging only)

            ANGELSPEC_FLASH_ATTN_PTXAS_OPT controls ptxas separately
            (default: 3). Use 1 for additional compile-time savings when
            combined with opt-level 1:
                export ANGELSPEC_FLASH_ATTN_OPT_LEVEL=1
                export ANGELSPEC_FLASH_ATTN_PTXAS_OPT=1
        """
        import os

        # ── Patch 1: disk cache ───────────────────────────────────────────────
        try:
            from cutlass.base_dsl.dsl import BaseDSL
        except ImportError:
            return
        if not getattr(BaseDSL, "_disk_cache_enabled", False):
            _orig_compile_and_cache = BaseDSL.compile_and_cache

            def _compile_and_cache_with_disk(
                self,
                module,
                module_hash,
                function_name,
                pipeline,
                args_spec,
                no_cache,
                *args,
                **kwargs,
            ):
                return _orig_compile_and_cache(
                    self,
                    module,
                    module_hash,
                    function_name,
                    pipeline,
                    args_spec,
                    False,  # override no_cache=True from Compiler._compile
                    *args,
                    **kwargs,
                )

            BaseDSL.compile_and_cache = _compile_and_cache_with_disk
            BaseDSL._disk_cache_enabled = True

            from cutlass.base_dsl.cache_helpers import default_generated_ir_path

            cache_dir = os.environ.get("CUTE_DSL_CACHE_DIR", default_generated_ir_path)
            logger.debug(
                "CUTLASS DSL disk cache enabled → %s  (set CUTE_DSL_CACHE_DIR for a persistent location)",
                cache_dir,
            )

        # ── Patch 2: opt-level injection ──────────────────────────────────────
        # cute.compile is a CompileCallable instance; _compile is the instance method.
        try:
            from cutlass.base_dsl.compiler import CompileCallable
        except ImportError:
            return
        if getattr(CompileCallable, "_opt_level_patched", False):
            return

        _opt_level = int(os.environ.get("ANGELSPEC_FLASH_ATTN_OPT_LEVEL", "3"))
        _ptxas_opt = int(os.environ.get("ANGELSPEC_FLASH_ATTN_PTXAS_OPT", "3"))

        _orig_compile_callable = CompileCallable._compile

        def _compile_with_opt_level(self, func, *args, **kwargs):
            options: str = kwargs.get("options", "") or ""
            if "--opt-level" not in options:
                options = options.strip() + f" --opt-level {_opt_level}"
            if "--ptxas-options" not in options and _ptxas_opt != 3:
                options = options + f" --ptxas-options=-O{_ptxas_opt}"
            kwargs["options"] = options.strip()
            return _orig_compile_callable(self, func, *args, **kwargs)

        CompileCallable._compile = _compile_with_opt_level
        CompileCallable._opt_level_patched = True

        if _opt_level != 3 or _ptxas_opt != 3:
            logger.debug(
                f"flash_attn compilation: opt-level={_opt_level}, "
                f"ptxas-opt={_ptxas_opt}  "
                f"(ANGELSPEC_FLASH_ATTN_OPT_LEVEL / ANGELSPEC_FLASH_ATTN_PTXAS_OPT)"
            )

    _patch_cutlass_compilation()

except ImportError:
    _has_cute_dsl = False

# Detect SM major version once at import time.
# SM90 (H100/H200) requires block_sparse_tensors in _flash_attn_bwd with mask_mod.
_cuda_sm_major: int = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(dynamic=True)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        orig_max_position=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        # Llama3 style rotary embedding frequency scaling
        if all(
            v is not None
            for v in [
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                orig_max_position,
            ]
        ):
            logger.info(
                f"Using Llama3 style rotary embedding with scaling_factor={scaling_factor}, low_freq_factor={low_freq_factor}, high_freq_factor={high_freq_factor}, orig_max_position={orig_max_position}"
            )
            self.scaling_factor = scaling_factor
            self.low_freq_factor = low_freq_factor
            self.high_freq_factor = high_freq_factor
            self.orig_max_position = orig_max_position

            low_freq_wavelen = orig_max_position / low_freq_factor
            high_freq_wavelen = orig_max_position / high_freq_factor
            wave_len = 2 * math.pi / inv_freq

            if low_freq_factor != high_freq_factor:
                smooth = (orig_max_position / wave_len - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
            else:
                smooth = 0

            new_freqs = torch.where(
                wave_len < high_freq_wavelen,
                inv_freq,
                torch.where(
                    wave_len > low_freq_wavelen,
                    inv_freq / self.scaling_factor,
                    (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
                ),
            )
            inv_freq = new_freqs

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings + 20,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    @torch.compile(dynamic=True)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaMutiRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__(dim, max_position_embeddings, base, device)
        self.scaling_factor = scaling_factor

    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.scaling_factor
            sin = emb.sin() * self.scaling_factor

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001  # Prevent singularity
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class LlamaYarnRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            (emb.cos() * _mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (emb.sin() * _mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )


_SNAP_Q = 128  # Alignment granularity for Q_LEN bucketing.


def _snap_to_multiple(n: int, multiple: int) -> int:
    """Round *n* up to the next multiple of *multiple*."""
    return ((n + multiple - 1) // multiple) * multiple


def _snap_to_power_of_2(n: int, min_val: int = 128) -> int:
    """Round *n* up to the next power of 2, at least *min_val*."""
    n = max(n, min_val)
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# EAGLE3 mask_mod mode selection
# ---------------------------------------------------------------------------
# Available modes (SM100+ only; SM90 always uses "closure"):
#   "seqlen_po2" - (default) read Q_LEN from seqlen_info, bitmask modulo.
#                  Requires power-of-2 Q_LEN.  Single compiled kernel.
#   "seqlen"     - read Q_LEN from seqlen_info, runtime integer modulo.
#                  Single compiled kernel but ~1.5-3x slower per-call.
#   "dynamic"    - read Q_LEN from aux_tensors.  Single compiled kernel
#                  but vec_size penalty (fwd: 2->1, bwd: 4->1).
#   "simple"     - closure-captured Q_LEN without valid_seq_len checks.
#                  One kernel per Q_LEN bucket (same as "closure" but fewer ops).
#   "closure"    - original closure-captured Q_LEN + valid_seq_len.
#                  One kernel per Q_LEN bucket.  Used on SM90.
_VALID_MASK_MODES = {"seqlen_po2", "seqlen", "dynamic", "simple", "closure"}
_EAGLE3_MASK_MODE: str = "seqlen_po2"


def set_eagle3_mask_mode(mode: str) -> None:
    """Set the EAGLE3 mask_mod strategy for SM100+.

    Has no effect on SM90 (always uses "closure").
    Valid modes: "seqlen_po2", "seqlen", "dynamic", "simple", "closure".
    """
    global _EAGLE3_MASK_MODE
    if mode not in _VALID_MASK_MODES:
        raise ValueError(f"Unknown eagle3 mask mode {mode!r}, must be one of {_VALID_MASK_MODES}")
    _EAGLE3_MASK_MODE = mode


def _effective_mask_mode() -> str:
    """Return the effective mask mode, accounting for SM90 override."""
    if _cuda_sm_major == 9:
        return "closure"
    return _EAGLE3_MASK_MODE


def _snap_q_len(q_len: int, mode: str) -> int:
    """Snap *q_len* to the bucket size required by *mode*.

    "seqlen_po2" requires power-of-2 Q_LEN; all other modes use _SNAP_Q alignment.
    """
    if mode == "seqlen_po2":
        return _snap_to_power_of_2(q_len)
    return _snap_to_multiple(q_len, _SNAP_Q)


# Caches for cute.jit mask_mod functions.
# _flash_mask_mod_cache: (Q_LEN, valid_seq_len) -> compiled mask_mod (SM90).
# _flash_mask_mod_simple_cache: Q_LEN -> compiled mask_mod (SM100+, no valid_seq_len).
# Keyed by integer constants so the same function object is returned for repeated calls,
# allowing _flash_attn_fwd/_flash_attn_bwd to reuse their compiled CUDA kernels.
_flash_mask_mod_cache: dict = {}
_flash_mask_mod_simple_cache: dict = {}


def _make_eagle3_flash_mask_mod(Q_LEN: int, valid_seq_len: int):
    """Build a cute.jit mask_mod for the EAGLE3 attention pattern.

    KV layout: [k0 (Q_LEN tokens), k1, k2, ..., kN], total KV = Q_LEN * lck.

    Causal part  (kv_idx < Q_LEN):
        q_idx >= kv_idx AND kv_idx < valid_seq_len AND q_idx < valid_seq_len
    Suffix part  (kv_idx >= Q_LEN):
        (kv_idx % Q_LEN) < valid_seq_len AND (kv_idx - q_idx) % Q_LEN == 0

    Both Q_LEN and valid_seq_len are compile-time constants captured via closure,
    so no aux_tensors are needed and the SM90 backward kernel works correctly.

    Results are cached by (Q_LEN, valid_seq_len) to prevent recompilation.
    """
    cache_key = (Q_LEN, valid_seq_len)
    if cache_key in _flash_mask_mod_cache:
        return _flash_mask_mod_cache[cache_key]

    if not _has_cute_dsl:
        return None

    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        q_len_ssa = fa_cute_utils.scalar_to_ssa(Q_LEN, cutlass.Int32)
        seq_len_ssa = fa_cute_utils.scalar_to_ssa(valid_seq_len, cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)

        # Causal part: kv token lives in k0
        in_k0 = kv_idx < q_len_ssa
        causal = q_idx >= kv_idx
        kv_valid = kv_idx < seq_len_ssa
        q_valid = q_idx < seq_len_ssa
        causal_part = in_k0 & causal & kv_valid & q_valid

        # Suffix part: kv token lives in k1..kN (diagonal EAGLE3 pattern)
        in_suffix = kv_idx >= q_len_ssa
        kv_mod = kv_idx % q_len_ssa
        suffix_padding = kv_mod < seq_len_ssa
        diff = kv_idx - q_idx
        diagonal = (diff % q_len_ssa) == zero_ssa
        suffix_part = in_suffix & suffix_padding & diagonal

        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = f"eagle3_flash_mask_Q{Q_LEN}_S{valid_seq_len}"
    _flash_mask_mod_cache[cache_key] = _eagle3_mask_mod
    return _eagle3_mask_mod


def _make_eagle3_flash_mask_mod_simple(Q_LEN: int):
    """Simplified mask_mod for SM100+ where valid_seq_len == Q_LEN.

    Mathematically equivalent to _make_eagle3_flash_mask_mod(Q_LEN, Q_LEN),
    but with the three always-True checks (kv_valid, q_valid, suffix_padding)
    removed at source rather than relying on the compiler to eliminate them.
    Fewer closure variables -> simpler hash -> fewer SSA ops in kernel.
    """
    if Q_LEN in _flash_mask_mod_simple_cache:
        return _flash_mask_mod_simple_cache[Q_LEN]

    if not _has_cute_dsl:
        return None

    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        q_len_ssa = fa_cute_utils.scalar_to_ssa(Q_LEN, cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)
        in_k0 = kv_idx < q_len_ssa
        causal_part = in_k0 & (q_idx >= kv_idx)
        in_suffix = kv_idx >= q_len_ssa
        diff = kv_idx - q_idx
        suffix_part = in_suffix & ((diff % q_len_ssa) == zero_ssa)
        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = f"eagle3_flash_mask_simple_Q{Q_LEN}"
    _flash_mask_mod_simple_cache[Q_LEN] = _eagle3_mask_mod
    return _eagle3_mask_mod


_eagle3_flash_mask_mod_dynamic = None


def _make_eagle3_flash_mask_mod_dynamic():
    """Build a single cute.jit mask_mod that reads Q_LEN from aux_tensors at runtime.

    Unlike _make_eagle3_flash_mask_mod_simple which captures Q_LEN as a closure
    variable (compile-time constant, one kernel per Q_LEN bucket), this version
    reads Q_LEN from aux_tensors[0] at runtime, enabling a single compiled kernel
    for all Q_LEN values.

    Trade-off: avoids per-bucket recompilation, but has_aux_tensors=True reduces
    vec_size (fwd: 2->1, bwd: 4->1) and Q_LEN becomes a runtime value (global
    memory read + loss of constant folding / branch elimination).
    """
    global _eagle3_flash_mask_mod_dynamic
    if _eagle3_flash_mask_mod_dynamic is not None:
        return _eagle3_flash_mask_mod_dynamic

    if not _has_cute_dsl:
        return None

    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        # Read Q_LEN from aux_tensors[0] (a 1-element Int32 tensor)
        q_len_ssa = fa_cute_utils.scalar_to_ssa(aux_tensors[0][0], cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)
        in_k0 = kv_idx < q_len_ssa
        causal_part = in_k0 & (q_idx >= kv_idx)
        in_suffix = kv_idx >= q_len_ssa
        diff = kv_idx - q_idx
        suffix_part = in_suffix & ((diff % q_len_ssa) == zero_ssa)
        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = "eagle3_flash_mask_dynamic"
    _eagle3_flash_mask_mod_dynamic = _eagle3_mask_mod
    return _eagle3_mask_mod


_eagle3_flash_mask_mod_seqlen = None


def _make_eagle3_flash_mask_mod_seqlen():
    """Build a single cute.jit mask_mod that reads Q_LEN from seqlen_info.seqlen_q.

    The flash_attn compile cache key includes mask_mod_hash (from closure variables)
    but NOT tensor shapes (seqlen dimensions are dynamic). This means:
    - Closure approach: different Q_LEN -> different closure -> different hash -> recompile.
    - This approach: single function, no closure -> fixed hash -> single compiled kernel.

    seqlen_info.seqlen_q == Q_LEN because we call _flash_attn_fwd with q of shape
    [bsz, q_len, num_heads, head_dim] and no cu_seqlens_q/seqused_q.

    No aux_tensors needed -> no vec_size penalty (fwd: 2, bwd: 4 unchanged).
    """
    global _eagle3_flash_mask_mod_seqlen
    if _eagle3_flash_mask_mod_seqlen is not None:
        return _eagle3_flash_mask_mod_seqlen

    if not _has_cute_dsl:
        return None

    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        q_len_ssa = fa_cute_utils.scalar_to_ssa(seqlen_info.seqlen_q, cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)
        in_k0 = kv_idx < q_len_ssa
        causal_part = in_k0 & (q_idx >= kv_idx)
        in_suffix = kv_idx >= q_len_ssa
        diff = kv_idx - q_idx
        suffix_part = in_suffix & ((diff % q_len_ssa) == zero_ssa)
        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = "eagle3_flash_mask_seqlen"
    _eagle3_flash_mask_mod_seqlen = _eagle3_mask_mod
    return _eagle3_mask_mod


_eagle3_flash_mask_mod_seqlen_po2 = None


def _make_eagle3_flash_mask_mod_seqlen_po2():
    """Mask_mod reading Q_LEN from seqlen_info.seqlen_q, using bitmask for modulo.

    Requires Q_LEN to be a power of 2 (use _snap_to_power_of_2 for bucketing).
    Replaces the expensive runtime `diff % Q_LEN == 0` with `diff & (Q_LEN-1) == 0`,
    which is a single AND instruction.

    Combined with reading Q_LEN from seqlen_info (no aux_tensors -> no vec_size penalty),
    this should match or approach closure-captured compile-time constant performance.
    """
    global _eagle3_flash_mask_mod_seqlen_po2
    if _eagle3_flash_mask_mod_seqlen_po2 is not None:
        return _eagle3_flash_mask_mod_seqlen_po2

    if not _has_cute_dsl:
        return None

    @cute.jit
    def _eagle3_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        q_len_ssa = fa_cute_utils.scalar_to_ssa(seqlen_info.seqlen_q, cutlass.Int32)
        one_ssa = fa_cute_utils.scalar_to_ssa(1, cutlass.Int32)
        zero_ssa = fa_cute_utils.scalar_to_ssa(0, cutlass.Int32)
        mask_ssa = q_len_ssa - one_ssa  # Q_LEN - 1 = bitmask for power-of-2
        in_k0 = kv_idx < q_len_ssa
        causal_part = in_k0 & (q_idx >= kv_idx)
        in_suffix = kv_idx >= q_len_ssa
        diff = kv_idx - q_idx
        suffix_part = in_suffix & ((diff & mask_ssa) == zero_ssa)
        return causal_part | suffix_part

    _eagle3_mask_mod.__name__ = "eagle3_flash_mask_seqlen_po2"
    _eagle3_flash_mask_mod_seqlen_po2 = _eagle3_mask_mod
    return _eagle3_mask_mod


def _build_eagle3_mask_pair(
    q_len: int,
    kv_len: int,
    bsz: int,
    lck: int,
    device: torch.device,
) -> tuple:
    """Return (mask_mod_cute, mask_mod_flex, aux_tensors) for the current SM architecture.

    SM90: always uses "closure" mode (block_sparse backward required).
    SM100+: uses the mode selected by set_eagle3_mask_mode().
    """
    mode = _effective_mask_mode()
    aux_tensors = None

    if mode == "closure":
        mask_mod_cute = _make_eagle3_flash_mask_mod(q_len, q_len)
    elif mode == "simple":
        mask_mod_cute = _make_eagle3_flash_mask_mod_simple(q_len)
    elif mode == "dynamic":
        mask_mod_cute = _make_eagle3_flash_mask_mod_dynamic()
        aux_tensors = [torch.tensor([q_len], dtype=torch.int32, device=device)]
    elif mode == "seqlen":
        mask_mod_cute = _make_eagle3_flash_mask_mod_seqlen()
    elif mode == "seqlen_po2":
        mask_mod_cute = _make_eagle3_flash_mask_mod_seqlen_po2()
    else:
        raise ValueError(f"Unknown eagle3 mask mode {mode!r}")

    # Always build a flex-compatible mask_mod for block-sparse iteration.
    mask_mod_flex = generate_eagle3_mask(
        Q_LEN=q_len,
        KV_LEN=kv_len,
        lck=lck,
    )

    return mask_mod_cute, mask_mod_flex, aux_tensors


# Block-sparse tensor caches for FA4 forward and backward.
#
# BlockMask.as_tuple(flatten=True) layout:
#   [0] seq_lengths_q, [1] seq_lengths_kv,
#   [2] kv_num_blocks, [3] kv_indices, [4] full_kv_num_blocks, [5] full_kv_indices,
#   [6] q_num_blocks,  [7] q_indices,  [8] full_q_num_blocks,  [9] full_q_indices,
#   [10] KV_BLOCK_SIZE, [11] Q_BLOCK_SIZE, [12] mask_mod
#
# Forward needs KV-direction (indices 2-5): for each Q block, which KV blocks to visit.
# Backward needs Q-direction (indices 6-9): for each KV block, which Q blocks affect it.

_BM_KV_NUM_BLOCKS = 2
_BM_KV_INDICES = 3
_BM_FULL_KV_NUM_BLOCKS = 4
_BM_FULL_KV_INDICES = 5
_BM_Q_NUM_BLOCKS = 6
_BM_Q_INDICES = 7
_BM_FULL_Q_NUM_BLOCKS = 8
_BM_FULL_Q_INDICES = 9

# Level 1: B=1, H=1 raw tensors keyed by (q_len, kv_len, block_size, device_index).
_block_mask_raw_cache: dict = {}
# Level 2: expanded BlockSparseTensorsTorch keyed by (q_len, kv_len, bsz, num_heads, direction, block_size, device_index).
_block_sparse_cache: dict = {}


def _fa4_block_size() -> tuple:
    """Return (Q_BLOCK_SIZE, KV_BLOCK_SIZE) for FA4 block-sparse iteration.

    SM100+ (Blackwell) requires Q_BLOCK_SIZE=256 for both forward and backward
    due to subtile_factor=2 (q_stage=2, two M-tiles per CTA).
    SM90 (Hopper) uses 128 x 128.
    """
    if _cuda_sm_major >= 10:
        return (256, 128)
    return (128, 128)


def _get_block_sparse(
    mask_mod_flex,
    q_len,
    kv_len,
    bsz,
    num_heads,
    device,
    direction="kv",
):
    """Return BlockSparseTensorsTorch for FA4 forward or backward.

    direction="kv": KV-direction tensors for forward (skip empty KV blocks per Q block).
    direction="q":  Q-direction tensors for backward (skip empty Q blocks per KV block).

    The EAGLE3 mask is batch/head-independent, so we compute with B=1, H=1
    and expand to (bsz, num_heads).
    """
    block_size = _fa4_block_size()
    device_idx = device.index or 0

    raw_key = (q_len, kv_len, block_size, device_idx)
    if raw_key not in _block_mask_raw_cache:
        bm = create_block_mask(
            mask_mod_flex,
            B=1,
            H=1,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
            BLOCK_SIZE=block_size,
        )
        t = bm.as_tuple()
        _block_mask_raw_cache[raw_key] = {
            "kv": (
                t[_BM_KV_NUM_BLOCKS],
                t[_BM_KV_INDICES],
                t[_BM_FULL_KV_NUM_BLOCKS],
                t[_BM_FULL_KV_INDICES],
            ),
            "q": (
                t[_BM_Q_NUM_BLOCKS],
                t[_BM_Q_INDICES],
                t[_BM_FULL_Q_NUM_BLOCKS],
                t[_BM_FULL_Q_INDICES],
            ),
        }

    cache_key = (q_len, kv_len, bsz, num_heads, direction, block_size, device_idx)
    if cache_key not in _block_sparse_cache:
        cnt, idx, f_cnt, f_idx = _block_mask_raw_cache[raw_key][direction]
        _block_sparse_cache[cache_key] = BlockSparseTensorsTorch(
            mask_block_cnt=cnt.expand(bsz, num_heads, -1).contiguous(),
            mask_block_idx=idx.expand(bsz, num_heads, -1, -1).contiguous(),
            full_block_cnt=f_cnt.expand(bsz, num_heads, -1).contiguous()
            if f_cnt is not None
            else None,
            full_block_idx=f_idx.expand(bsz, num_heads, -1, -1).contiguous()
            if f_idx is not None
            else None,
            block_size=block_size,
        )
    return _block_sparse_cache[cache_key]


class _EagleMaskedFlashAttnFunc(torch.autograd.Function):
    """Autograd wrapper for flash_attn fwd/bwd with mask_mod + block sparsity.

    The public flash_attn_func does not pass mask_mod to its backward, making
    gradients incorrect for non-causal custom masks. This wrapper correctly
    forwards mask_mod to both _flash_attn_fwd and _flash_attn_bwd.

    Block-sparse iteration lets the kernel skip tiles that are entirely masked
    out, which is critical for the EAGLE3 pattern at long sequences where the
    mask is very sparse (causal + diagonal suffix).
    """

    @staticmethod
    def forward(
        ctx, q, k, v, mask_mod_cute, mask_mod_flex, softmax_scale, max_seq_len, aux_tensors=None
    ):
        bsz, q_len, num_heads, _ = q.shape
        kv_len = k.shape[1]

        fwd_block_sparse = _get_block_sparse(
            mask_mod_flex,
            q_len,
            kv_len,
            bsz,
            num_heads,
            q.device,
            direction="kv",
        )

        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            mask_mod=mask_mod_cute,
            block_sparse_tensors=fwd_block_sparse,
            return_lse=True,
            aux_tensors=aux_tensors,
        )

        bwd_block_sparse = _get_block_sparse(
            mask_mod_flex,
            q_len,
            kv_len,
            bsz,
            num_heads,
            q.device,
            direction="q",
        )

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.mask_mod_cute = mask_mod_cute
        ctx.bwd_block_sparse = bwd_block_sparse
        ctx.aux_tensors = aux_tensors
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            lse,
            softmax_scale=ctx.softmax_scale,
            causal=False,
            mask_mod=ctx.mask_mod_cute,
            block_sparse_tensors=ctx.bwd_block_sparse,
            aux_tensors=ctx.aux_tensors,
        )
        return dq, dk, dv, None, None, None, None, None


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        rope_scaling = self.config.rope_scaling

        def rope_get(key, default=None):
            if rope_scaling is None:
                return default
            if isinstance(rope_scaling, dict):
                return rope_scaling.get(key, default)
            return getattr(rope_scaling, key, default)

        scaling_type = rope_get("rope_type", rope_get("type"))

        if rope_scaling is None or scaling_type == "default":
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(self.config, "rope_theta", 10000),
            )
        else:
            scaling_factor = rope_get("factor")

            if scaling_type == "linear":
                if scaling_factor is None:
                    raise ValueError(
                        "Linear RoPE scaling requires 'factor' in rope_scaling config."
                    )
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                if scaling_factor is None:
                    raise ValueError(
                        "Dynamic RoPE scaling requires 'factor' in rope_scaling config."
                    )
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "llama3":
                # for nv type
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=getattr(self.config, "rope_theta", 10000),
                    scaling_factor=(scaling_factor if scaling_factor is not None else 1.0),
                    low_freq_factor=rope_get("low_freq_factor"),
                    high_freq_factor=rope_get("high_freq_factor"),
                    orig_max_position=rope_get("original_max_position_embeddings"),
                )
            elif scaling_type == "mrope":
                self.rotary_emb = LlamaMutiRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings
                )
            elif scaling_type == "yarn":
                self.rotary_emb = LlamaYarnRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=getattr(self.config, "rope_theta", 10000),
                    original_max_position_embeddings=rope_get("original_max_position_embeddings"),
                    scaling_factor=scaling_factor,
                    beta_fast=rope_get("beta_fast"),
                    beta_slow=rope_get("beta_slow"),
                    mscale=rope_get("mscale"),
                    mscale_all_dim=rope_get("mscale_all_dim"),
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if not use_cache:
            # Standard path — no caching
            if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
                cos, sin = self.rotary_emb(query_states, position_ids)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    self.config.rope_scaling["mrope_section"],
                )
            else:
                cos, sin = self.rotary_emb(query_states, seq_len=q_len)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=0.0,
            )

        else:
            # Cached path — cache_keys shape: [bsz, num_heads, num_cached, seq_len, head_dim]
            lck = 0 if cache_keys is None else cache_keys.shape[2]

            if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
                cos, sin = self.rotary_emb(query_states, position_ids + lck)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    self.config.rope_scaling["mrope_section"],
                )
            else:
                cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids + lck
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Append to tensor cache: [bsz, num_heads, num_cached, seq_len, head_dim]
            if cache_keys is None:
                cache_keys = key_states.unsqueeze(2)
                cache_values = value_states.unsqueeze(2)
            else:
                cache_keys = torch.cat([cache_keys, key_states.unsqueeze(2)], dim=2)
                cache_values = torch.cat([cache_values, value_states.unsqueeze(2)], dim=2)

            lck = cache_keys.shape[2]
            k0 = cache_keys[:, :, 0]
            v0 = cache_values[:, :, 0]

            # causal
            attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )

            attn_weights = attn_weights + attention_mask

            for i in range(1, lck):
                ki = cache_keys[:, :, i]
                attn_weightsi = (query_states * ki).sum(-1) / math.sqrt(self.head_dim)
                attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_weights0 = attn_weights[..., :q_len]

            attn_output = torch.matmul(attn_weights0, v0)

            for i in range(1, lck):
                vi = cache_values[:, :, i]
                attn_weightsi = attn_weights[..., q_len + i - 1]
                attn_outputi = attn_weightsi[..., None] * vi
                attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)

        attn_output = self.o_proj(attn_output)

        return attn_output, cache_keys, cache_values


class LlamaFlexAttention(LlamaAttention):
    """
    Attention layer implemented with flex attention. We keep the parameters consistent with LlamaAttention.
    The used parameters are:
        - hidden_states: input hidden states
        - attention_mask: attention mask not expanded, straight from data loader.
        - position_ids: position ids
        - cache_keys/cache_values: tensor caches for storing past key and value states.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # cache_keys shape: [bsz, num_kv_heads, cache_seq_len, head_dim] (concatenated along seq dim)
        past_seen_tokens = cache_keys.shape[2] if cache_keys is not None else 0

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        lck = past_seen_tokens // q_len
        if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            # Keep positions ids aligned when padding so the KV cache is unaffected.
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck
            )

        # Concatenate along sequence dimension: [bsz, num_kv_heads, total_seq_len, head_dim]
        if cache_keys is not None:
            key_cache = torch.cat([cache_keys, key_states], dim=2)
            value_cache = torch.cat([cache_values, value_states], dim=2)
        else:
            key_cache = key_states
            value_cache = value_states

        flex_attention_func = flex_attention if q_len <= 128 else compile_friendly_flex_attention

        block_mask = eagle3_block_mask(
            Q_LEN=q_len,
            KV_LEN=key_cache.shape[-2],
            B=bsz,
            H=1,  # Rely on broadcast
            device=query_states.device,
            lck=lck,
        )
        attn_output = flex_attention_func(
            query=query_states,
            key=key_cache.contiguous(),
            value=value_cache.contiguous(),
            block_mask=block_mask,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output, key_cache, value_cache


class LlamaFlashAttention(LlamaAttention):
    """
    Attention layer implemented with flash attention. We keep the parameters consistent with LlamaAttention.
    The used parameters are:
        - hidden_states: input hidden states
        - position_ids: position ids
        - cache_keys/cache_values: tensor caches for storing past key and value states
    """

    def __init__(self, config):
        super().__init__(config)
        if (
            _std_flash_attn_forward is None
            or _std_flash_attn_backward is None
            or _std_flash_unpad_input is None
            or _std_flash_pad_input is None
            or _std_flash_attn_varlen_backward is None
        ):
            _raise_standard_flash_attn_unavailable()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        lck = 0 if cache_keys is None else cache_keys.shape[1]
        if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
                unsqueeze_dim=2,
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck, unsqueeze_dim=2
            )

        if cache_keys is not None:
            cache_keys = torch.cat([cache_keys, key_states.unsqueeze(1)], dim=1)
            cache_values = torch.cat([cache_values, value_states.unsqueeze(1)], dim=1)
        else:
            cache_keys = key_states.unsqueeze(1)
            cache_values = value_states.unsqueeze(1)

        assert attention_mask is not None, (
            "LlamaFlashAttention cached path requires attention_mask"
        )
        valid_lengths = attention_mask.sum(dim=-1, dtype=torch.long) - lck
        valid_lengths = valid_lengths.clamp_(0, q_len)

        attn_output = _FlashCachedMergeFunc.apply(
            query_states,
            cache_keys,
            cache_values,
            valid_lengths,
            1.0 / math.sqrt(self.head_dim),
        )

        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output, cache_keys, cache_values


class LlamaFlashAttentionMasked(LlamaAttention):
    """[EXPERIMENTAL] Attention layer using flash-attention cute DSL mask_mod for EAGLE3.

    Status: experimental.  Known limitations:
      - Requires flash-attention with cute DSL (flash_attn.cute) and cutlass-dsl.
      - SM90 (H100/H200): validated.
      - SM80 (A100): validated.
      - SM100+ (Blackwell): uses simplified mask_mod (no backward block-sparse).
      - Variable-length batches: correctness is maintained (loss masking protects
        gradients), but attention quality at padding positions is not guaranteed.

    This implementation concatenates all KV blocks into a single tensor and passes
    the full EAGLE3 causal+suffix attention pattern to a single flash_attn kernel via
    cute.jit mask_mod, eliminating the nested logsumexp gradient errors of
    LlamaFlashAttention.

    The used parameters are:
        - hidden_states: input hidden states
        - position_ids: position ids
        - cache_keys/cache_values: tensor caches [bsz, num_cached, seq_len, num_kv_heads, head_dim]
        - attention_mask: bool mask [bsz, seq_len] for padding
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, orig_q_len, _ = hidden_states.size()

        if not _has_cute_dsl or _flash_attn_fwd is None:
            raise RuntimeError(
                "LlamaFlashAttentionMasked requires flash-attention cute interface "
                "and the cutlass DSL (pip install cutlass-dsl)"
            )

        # Snap q_len so every batch in the same bucket shares the same compiled
        # flash_attn kernel.  "seqlen_po2" requires power-of-2 Q_LEN;
        # all other modes only need _SNAP_Q alignment.
        q_len = _snap_q_len(orig_q_len, _effective_mask_mode())
        pad_sz = q_len - orig_q_len
        if pad_sz > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_sz))
            if position_ids is not None:
                extra_ids = (
                    torch.arange(orig_q_len, q_len, dtype=torch.long, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
                position_ids = torch.cat([position_ids, extra_ids], dim=1)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # FA layout: [bsz, seq_len, heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        lck = 0 if cache_keys is None else cache_keys.shape[1]
        if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
                unsqueeze_dim=2,
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck, unsqueeze_dim=2
            )

        # Append to tensor cache: [bsz, num_cached, seq_len, num_kv_heads, head_dim]
        if cache_keys is not None:
            cache_keys = torch.cat([cache_keys, key_states.unsqueeze(1)], dim=1)
            cache_values = torch.cat([cache_values, value_states.unsqueeze(1)], dim=1)
        else:
            cache_keys = key_states.unsqueeze(1)
            cache_values = value_states.unsqueeze(1)

        # Flatten all KV cache blocks: [bsz, num_cached * seq_len, num_kv_heads, head_dim]
        k_all = cache_keys.reshape(bsz, -1, self.num_key_value_heads, self.head_dim)
        v_all = cache_values.reshape(bsz, -1, self.num_key_value_heads, self.head_dim)

        max_seq_len = q_len
        mask_mod_cute, mask_mod_flex, aux_tensors = _build_eagle3_mask_pair(
            q_len,
            k_all.shape[1],
            bsz,
            lck,
            hidden_states.device,
        )

        attn_output = _EagleMaskedFlashAttnFunc.apply(
            query_states,
            k_all,
            v_all,
            mask_mod_cute,
            mask_mod_flex,
            1.0 / math.sqrt(self.head_dim),
            max_seq_len,
            aux_tensors,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        if pad_sz > 0:
            attn_output = attn_output[:, :orig_q_len, :]
        return attn_output, cache_keys, cache_values


def _standard_flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = _std_flash_attn_forward(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    return out, lse


def _standard_flash_attn_backward_call(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> None:
    _std_flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
        None,
    )


def _standard_flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = _std_flash_unpad_input(q, attention_mask)
    k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = _std_flash_unpad_input(k, attention_mask)
    v_unpad, _, _, _, _ = _std_flash_unpad_input(v, attention_mask)
    out_unpad, lse_unpad, _ = _std_flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
    out = _std_flash_pad_input(out_unpad, indices_q, q.shape[0], q.shape[1])
    lse_padded = _std_flash_pad_input(
        lse_unpad.transpose(0, 1), indices_q, q.shape[0], q.shape[1]
    ).transpose(1, 2)
    return out, lse_padded, indices_q, cu_seqlens_q


def _standard_flash_attn_varlen_backward_call(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    attention_mask: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> None:
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = _std_flash_unpad_input(q, attention_mask)
    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = _std_flash_unpad_input(k, attention_mask)
    v_unpad, _, _, _, _ = _std_flash_unpad_input(v, attention_mask)
    dout_unpad, _, _, _, _ = _std_flash_unpad_input(dout, attention_mask)
    out_unpad, _, _, _, _ = _std_flash_unpad_input(out, attention_mask)
    lse_unpad, _, _, _, _ = _std_flash_unpad_input(softmax_lse.transpose(1, 2), attention_mask)
    lse_unpad = lse_unpad.transpose(0, 1).contiguous()

    dq_unpad = torch.empty_like(q_unpad)
    dk_unpad = torch.empty_like(k_unpad)
    dv_unpad = torch.empty_like(v_unpad)
    _std_flash_attn_varlen_backward(
        dout_unpad,
        q_unpad,
        k_unpad,
        v_unpad,
        out_unpad,
        lse_unpad,
        dq_unpad,
        dk_unpad,
        dv_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
        None,
    )
    dq.copy_(_std_flash_pad_input(dq_unpad, indices_q, q.shape[0], q.shape[1]))
    dk.copy_(_std_flash_pad_input(dk_unpad, indices_k, k.shape[0], k.shape[1]))
    dv.copy_(_std_flash_pad_input(dv_unpad, indices_k, v.shape[0], v.shape[1]))


class _FlashCachedMergeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, cache_k, cache_v, valid_lengths, softmax_scale: float):
        bsz, q_len, num_heads, head_dim = q.shape
        num_blocks = cache_k.shape[1]
        num_kv_heads = cache_k.shape[3]
        num_groups = num_heads // num_kv_heads
        q_expanded = q.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        valid_lengths = valid_lengths.to(device=q.device, dtype=torch.long).clamp_(0, q_len)
        valid_mask = (
            torch.arange(q_len, device=q.device).unsqueeze(0) < valid_lengths.unsqueeze(1)
        ).view(bsz, q_len, 1, 1)
        attention_mask = valid_mask.view(bsz, q_len)

        k0 = cache_k[:, 0].contiguous()
        v0 = cache_v[:, 0].contiguous()
        out0, lse0_kernel, _, _ = _standard_flash_attn_varlen_forward(
            q.contiguous(),
            k0,
            v0,
            attention_mask,
            softmax_scale=softmax_scale,
            causal=True,
        )
        out0_expanded = out0.view(bsz, q_len, num_kv_heads, num_groups, head_dim).float()
        neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)
        lse0 = lse0_kernel.transpose(1, 2).view(bsz, q_len, num_kv_heads, num_groups).float()
        lse0 = torch.where(valid_mask, lse0, neg_inf)
        lse_terms = [lse0]
        attn_terms = [out0_expanded]
        for i in range(1, num_blocks):
            ki = cache_k[:, i].unsqueeze(-2).float()
            vi = cache_v[:, i].unsqueeze(-2).float()
            lse_i = (q_expanded.float() * ki).sum(-1) * softmax_scale
            lse_terms.append(torch.where(valid_mask, lse_i, neg_inf))
            attn_terms.append(vi.expand_as(out0_expanded))

        merged_lse = torch.logsumexp(torch.stack(lse_terms, dim=-1), dim=-1)
        out = sum(
            term * torch.exp(lse - merged_lse).unsqueeze(-1)
            for term, lse in zip(attn_terms, lse_terms)
        )
        out = torch.where(valid_mask.unsqueeze(-1), out, 0.0)
        merged_lse = torch.where(valid_mask, merged_lse, 0.0)
        ctx.save_for_backward(q, cache_k, cache_v, out, merged_lse, valid_lengths)
        ctx.softmax_scale = softmax_scale
        return out.to(q.dtype).reshape_as(q)

    @staticmethod
    def backward(ctx, grad_out):
        q, cache_k, cache_v, out, merged_lse, valid_lengths = ctx.saved_tensors
        bsz, q_len, num_heads, head_dim = q.shape
        num_blocks = cache_k.shape[1]
        num_kv_heads = cache_k.shape[3]
        num_groups = num_heads // num_kv_heads
        scale = ctx.softmax_scale
        if grad_out.ndim == 3:
            grad_out = grad_out.view(bsz, q_len, num_heads, head_dim)
        valid_lengths = valid_lengths.to(device=q.device, dtype=torch.long)
        valid_mask = (
            torch.arange(q_len, device=q.device).unsqueeze(0) < valid_lengths.unsqueeze(1)
        ).view(bsz, q_len, 1, 1)
        attention_mask = valid_mask.view(bsz, q_len)
        grad_out = torch.where(valid_mask, grad_out, 0.0)

        grad_out_f = grad_out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        q_f = q.float()
        q_expanded = q_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_f = out.float()
        out_expanded = out_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_q = out.to(q.dtype).reshape(bsz, q_len, num_heads, head_dim)

        dq = torch.zeros_like(q_f)
        dcache_k = torch.zeros_like(cache_k.float())
        dcache_v = torch.zeros_like(cache_v.float())

        dq0 = torch.zeros_like(q)
        dk0 = torch.zeros_like(cache_k[:, 0])
        dv0 = torch.zeros_like(cache_v[:, 0])
        merged_lse_kernel = merged_lse.reshape(bsz, q_len, num_heads).transpose(1, 2).contiguous()
        _standard_flash_attn_varlen_backward_call(
            grad_out.contiguous(),
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            out_q.contiguous(),
            merged_lse_kernel,
            attention_mask,
            dq0,
            dk0,
            dv0,
            softmax_scale=scale,
            causal=True,
        )
        dq += dq0.float()
        dcache_k[:, 0] += dk0.float()
        dcache_v[:, 0] += dv0.float()

        for i in range(1, num_blocks):
            ki = cache_k[:, i].float().unsqueeze(-2)
            vi = cache_v[:, i].float().unsqueeze(-2)
            lse_i = (q_expanded * ki).sum(-1) * scale
            wi = torch.where(valid_mask, torch.exp(lse_i - merged_lse), 0.0)
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            d_lse_i = wi * (grad_out_f * (vi.expand_as(out_expanded) - out_expanded)).sum(-1)
            dq += (d_lse_i.unsqueeze(-1) * scale * ki).reshape_as(q)
            dcache_k[:, i] += (d_lse_i.unsqueeze(-1) * scale * q_expanded).sum(dim=3)
            dcache_v[:, i] += d_out_i.sum(dim=3)

        return dq.to(q.dtype), dcache_k.to(cache_k.dtype), dcache_v.to(cache_v.dtype), None, None


class _USPFlashCachedMergeFunc(torch.autograd.Function):
    @staticmethod
    def _merge_dims(q: torch.Tensor, cache_k: torch.Tensor):
        bsz, q_len, num_heads, head_dim = q.shape
        num_blocks = cache_k.shape[1]
        num_kv_heads = cache_k.shape[3]
        num_groups = num_heads // num_kv_heads
        return bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups

    @staticmethod
    def _kernel_lse(lse: torch.Tensor, bsz: int, q_len: int, num_heads: int) -> torch.Tensor:
        return lse.reshape(bsz, q_len, num_heads).transpose(1, 2).contiguous()

    @staticmethod
    def forward(ctx, q, cache_k, cache_v, softmax_scale: float):
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _USPFlashCachedMergeFunc._merge_dims(q, cache_k)
        )
        q_expanded = q.view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        k0 = cache_k[:, 0].contiguous()
        v0 = cache_v[:, 0].contiguous()
        assert _std_flash_attn_forward is not None and _std_flash_attn_backward is not None, (
            "USP cached merge requires standard flash-attn forward/backward "
            f"(standard import error: {_std_flash_attn_import_error!r})"
        )
        out0, lse0_kernel = _standard_flash_attn_forward(
            q.contiguous(),
            k0,
            v0,
            softmax_scale=softmax_scale,
            causal=True,
        )
        out0_expanded = out0.view(bsz, q_len, num_kv_heads, num_groups, head_dim).float()
        lse0 = lse0_kernel.transpose(1, 2).reshape(bsz, q_len, num_kv_heads, num_groups).float()
        lse_terms = [lse0]
        attn_terms = [out0_expanded]
        for i in range(1, num_blocks):
            ki = cache_k[:, i].unsqueeze(-2).float()
            vi = cache_v[:, i].unsqueeze(-2).float()
            lse_terms.append((q_expanded.float() * ki).sum(-1) * softmax_scale)
            attn_terms.append(vi.expand_as(out0_expanded))

        merged_lse = torch.logsumexp(torch.stack(lse_terms, dim=-1), dim=-1)
        out = sum(
            term * torch.exp(lse - merged_lse).unsqueeze(-1)
            for term, lse in zip(attn_terms, lse_terms)
        )
        ctx.save_for_backward(q, cache_k, cache_v, out, merged_lse)
        ctx.softmax_scale = softmax_scale
        return out.to(q.dtype).reshape_as(q)

    @staticmethod
    def backward(ctx, grad_out):
        q, cache_k, cache_v, out, merged_lse = ctx.saved_tensors
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _USPFlashCachedMergeFunc._merge_dims(q, cache_k)
        )
        scale = ctx.softmax_scale

        grad_out_f = grad_out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        q_f = q.float()
        q_expanded = q_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_f = out.float()
        out_expanded = out_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        dq = torch.zeros_like(q_f)
        dcache_k = torch.zeros_like(cache_k.float())
        dcache_v = torch.zeros_like(cache_v.float())

        merged_lse_kernel = _USPFlashCachedMergeFunc._kernel_lse(merged_lse, bsz, q_len, num_heads)
        dq0 = torch.empty_like(q)
        dk0 = torch.empty_like(cache_k[:, 0])
        dv0 = torch.empty_like(cache_v[:, 0])
        _standard_flash_attn_backward_call(
            grad_out.contiguous(),
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            out.to(q.dtype).reshape_as(q).contiguous(),
            merged_lse_kernel,
            dq0,
            dk0,
            dv0,
            softmax_scale=scale,
            causal=True,
        )
        dq += dq0.float()
        dcache_k[:, 0] += dk0.float()
        dcache_v[:, 0] += dv0.float()

        for i in range(1, num_blocks):
            ki = cache_k[:, i].float().unsqueeze(-2)
            vi = cache_v[:, i].float().unsqueeze(-2)
            lse_i = (q_expanded * ki).sum(-1) * scale
            wi = torch.exp(lse_i - merged_lse)
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            d_lse_i = wi * (grad_out_f * (vi.expand_as(out_expanded) - out_expanded)).sum(-1)
            dq += (d_lse_i.unsqueeze(-1) * scale * ki).reshape_as(q)
            dcache_k[:, i] += (d_lse_i.unsqueeze(-1) * scale * q_expanded).sum(dim=3)
            dcache_v[:, i] += d_out_i.sum(dim=3)

        return dq.to(q.dtype), dcache_k.to(cache_k.dtype), dcache_v.to(cache_v.dtype), None


def _update_ring_out_and_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.float()
    block_lse = block_lse.float()
    if out is None or lse is None:
        return block_out, block_lse
    new_lse = torch.logaddexp(lse, block_lse)
    out = out * torch.exp(lse - new_lse).unsqueeze(-1) + block_out * torch.exp(
        block_lse - new_lse
    ).unsqueeze(-1)
    return out, new_lse


class _USPRingFlashCachedMergeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, cache_k, cache_v, ring_group: dist.ProcessGroup, softmax_scale: float):
        from yunchang.ring.ring_flash_attn import ring_flash_attn_forward

        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _USPFlashCachedMergeFunc._merge_dims(q, cache_k)
        )
        q_expanded = q.view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        out_ring, lse_ring = ring_flash_attn_forward(
            ring_group,
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            softmax_scale=softmax_scale,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        if lse_ring.dim() == 3 and lse_ring.shape[1] == num_heads:
            lse_ring = lse_ring.transpose(1, 2)
        acc_out = out_ring.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        acc_lse = lse_ring.reshape(bsz, q_len, num_kv_heads, num_groups)

        for i in range(1, num_blocks):
            ki = cache_k[:, i].unsqueeze(-2).float()
            vi = cache_v[:, i].unsqueeze(-2).float()
            lse_i = (q_expanded.float() * ki).sum(-1) * softmax_scale
            out_i = vi.expand(bsz, q_len, num_kv_heads, num_groups, head_dim)
            acc_out, acc_lse = _update_ring_out_and_lse(acc_out, acc_lse, out_i, lse_i)

        ctx.save_for_backward(q, cache_k, cache_v, acc_out, acc_lse)
        ctx.ring_group = ring_group
        ctx.softmax_scale = softmax_scale
        return acc_out.to(q.dtype).reshape_as(q)

    @staticmethod
    def backward(ctx, grad_out):
        from yunchang.ring.ring_flash_attn import ring_flash_attn_backward

        q, cache_k, cache_v, out, merged_lse = ctx.saved_tensors
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _USPFlashCachedMergeFunc._merge_dims(q, cache_k)
        )
        scale = ctx.softmax_scale

        if grad_out.ndim == 3:
            grad_out = grad_out.view(bsz, q_len, num_heads, head_dim)

        grad_out_f = grad_out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        q_f = q.float()
        q_expanded = q_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_expanded = out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        dcache_k = torch.zeros_like(cache_k.float())
        dcache_v = torch.zeros_like(cache_v.float())
        out_q = out.to(q.dtype).reshape_as(q)
        merged_lse_kernel = _USPFlashCachedMergeFunc._kernel_lse(merged_lse, bsz, q_len, num_heads)

        dq, dk0, dv0 = ring_flash_attn_backward(
            ctx.ring_group,
            grad_out.contiguous(),
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            out_q.contiguous(),
            merged_lse_kernel,
            softmax_scale=scale,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        dq = dq.float()
        dcache_k[:, 0] = dk0.float()
        dcache_v[:, 0] = dv0.float()

        for i in range(1, num_blocks):
            ki = cache_k[:, i].float().unsqueeze(-2)
            vi = cache_v[:, i].float().unsqueeze(-2)
            lse_i = (q_expanded * ki).sum(-1) * scale
            wi = torch.exp(lse_i - merged_lse)
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            d_lse_i = wi * (grad_out_f * (vi.expand_as(out_expanded) - out_expanded)).sum(-1)
            dq += (d_lse_i.unsqueeze(-1) * scale * ki).reshape_as(q)
            dcache_k[:, i] += (d_lse_i.unsqueeze(-1) * scale * q_expanded).sum(dim=3)
            dcache_v[:, i] += d_out_i.sum(dim=3)

        return dq.to(q.dtype), dcache_k.to(cache_k.dtype), dcache_v.to(cache_v.dtype), None, None


class LlamaUSPFlashAttention(LlamaAttention):
    """USP attention with Ulysses all-to-all and local flash attention."""

    def __init__(self, config):
        super().__init__(config)
        if not dist.is_initialized():
            raise RuntimeError(
                "LlamaUSPFlashAttention requires torch.distributed to be initialized first."
            )
        try:
            from yunchang.comm import SeqAllToAll4D
        except ImportError as exc:
            raise RuntimeError(
                "USP requires yunchang; please install it to use USP attention."
            ) from exc

        self._SeqAllToAll4D = SeqAllToAll4D
        self.ring_pg = get_sp_ring_group()
        self.ulysses_pg = get_sp_ulysses_group()
        if self.ring_pg is None or self.ulysses_pg is None:
            raise RuntimeError("USP requires sp ring/ulysses groups to be initialized.")
        self.sp_ring_degree = dist.get_world_size(self.ring_pg)
        self.sp_ulysses_degree = dist.get_world_size(self.ulysses_pg)
        self.scatter_idx = 2
        self.gather_idx = 1
        self.use_sync = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        del attention_mask, use_cache
        bsz, q_len, _ = hidden_states.size()
        local_q_len = q_len

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        query_states = self._SeqAllToAll4D.apply(
            self.ulysses_pg,
            query_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        key_states = self._SeqAllToAll4D.apply(
            self.ulysses_pg,
            key_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self._SeqAllToAll4D.apply(
            self.ulysses_pg,
            value_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        q_len = query_states.shape[1]
        global_q_len = local_q_len * self.sp_ring_degree * self.sp_ulysses_degree

        lck = 0 if cache_keys is None else cache_keys.shape[1]
        cos, sin = self.rotary_emb(query_states, seq_len=global_q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids + lck, unsqueeze_dim=2
        )

        if cache_keys is not None:
            cache_keys = torch.cat([cache_keys, key_states.unsqueeze(1)], dim=1)
            cache_values = torch.cat([cache_values, value_states.unsqueeze(1)], dim=1)
        else:
            cache_keys = key_states.unsqueeze(1)
            cache_values = value_states.unsqueeze(1)

        softmax_scale = 1.0 / math.sqrt(self.head_dim)
        if self.sp_ring_degree > 1:
            attn_output = _USPRingFlashCachedMergeFunc.apply(
                query_states,
                cache_keys,
                cache_values,
                self.ring_pg,
                softmax_scale,
            )
        else:
            attn_output = _USPFlashCachedMergeFunc.apply(
                query_states,
                cache_keys,
                cache_values,
                softmax_scale,
            )

        attn_output = self._SeqAllToAll4D.apply(
            self.ulysses_pg,
            attn_output,
            self.gather_idx,
            self.scatter_idx,
            self.use_sync,
        )
        attn_output = attn_output.reshape(bsz, local_q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output, cache_keys, cache_values


def warmup_flash_attention_masked(
    q_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device=None,
    warmup_buckets: Optional[list] = None,
) -> None:
    """Pre-compile flash_attn cute DSL kernels for LlamaFlashAttentionMasked.

    Runs a fwd+bwd warmup to populate the in-process compile caches
    (_flash_attn_fwd.compile_cache / _flash_attn_bwd.compile_cache),
    so training step 0 avoids compilation.

    SM90:  Each Q_LEN bucket produces a distinct compiled kernel (closure-captured
           Q_LEN). All lck values within a bucket share the same compile_key, so
           one fwd+bwd call per bucket suffices.
    SM100+: A single kernel handles all Q_LEN values (seqlen_po2 reads Q_LEN from
            seqlen_info at runtime). Only one fwd+bwd call is needed regardless
            of the number of buckets.

    Args:
        warmup_buckets: Q_LEN values to pre-compile. Default (None) warms up
            only the max bucket (SM90) or a single representative (SM100+).
            For SM90 variable-length training, pass explicit bucket sizes.
    """
    if not _has_cute_dsl or _flash_attn_fwd is None:
        return
    if device is None:
        device = torch.cuda.current_device()

    mode = _effective_mask_mode()
    # Modes with closure-captured Q_LEN need one warmup per bucket;
    # runtime-Q_LEN modes ("seqlen_po2", "seqlen", "dynamic") need only one.
    needs_per_bucket = mode in ("closure", "simple")
    if needs_per_bucket:
        max_bucket = _snap_to_multiple(q_len, _SNAP_Q)
        if warmup_buckets is None:
            buckets = sorted({_SNAP_Q, max_bucket})
        else:
            buckets = sorted(
                {_SNAP_Q, max_bucket} | {_snap_to_multiple(b, _SNAP_Q) for b in warmup_buckets}
            )
    else:
        # Single compiled kernel for all Q_LEN values.
        buckets = [_SNAP_Q]

    softmax_scale = 1.0 / math.sqrt(head_dim)

    print_with_rank(
        f"Warming up flash_attn cute DSL kernels "
        f"({len(buckets)} Q_LEN bucket(s): {buckets}, mode={mode}, sm_major={_cuda_sm_major})..."
    )

    for warmup_q_len in buckets:
        mask_mod_cute, mask_mod_flex, aux_tensors = _build_eagle3_mask_pair(
            warmup_q_len,
            warmup_q_len,
            1,
            0,
            device,
        )

        q_t = torch.randn(
            1, warmup_q_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        k_t = torch.randn(1, warmup_q_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_t = torch.randn(1, warmup_q_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        out = _EagleMaskedFlashAttnFunc.apply(
            q_t, k_t, v_t, mask_mod_cute, mask_mod_flex, softmax_scale, warmup_q_len, aux_tensors
        )
        out.sum().backward()
        del q_t, k_t, v_t, out

    torch.cuda.synchronize()
    print_with_rank(f"flash_attn cute DSL warmup complete ({len(buckets)} bucket(s)).")


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @torch.compile(dynamic=True)
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size

        if attention_backend == "sdpa":
            self.self_attn = LlamaAttention(config=config)
        elif attention_backend == "flex_attention":
            print_with_rank("Using flex attention on draft model training!")
            self.self_attn = LlamaFlexAttention(config=config)
        elif attention_backend == "fa4":
            self.self_attn = LlamaFlashAttentionMasked(config=config)
        elif attention_backend == "fa":
            self.self_attn = LlamaFlashAttention(config=config)
        elif attention_backend == "usp":
            self.self_attn = LlamaUSPFlashAttention(config=config)
        else:
            raise ValueError(f"Unknown attention backend {attention_backend}")

        self.attention_backend = attention_backend
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        hidden_states, cache_keys, cache_values = self.self_attn(
            hidden_states=hidden_states,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache_keys, cache_values


class LlamaForCausalLMEagle3(Eagle3DraftModel):
    config_class = LlamaConfig

    def __init__(self, config, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config

        self.target_vocab_size = config.vocab_size
        self.vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = LlamaDecoderLayer(config, attention_backend=attention_backend)

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)

        self.fc = torch.nn.Linear(
            target_hidden_size * self.num_aux_hidden_states,
            config.hidden_size,
            bias=False,
        )
        use_fc_norm = getattr(config, "fc_norm", None)
        if use_fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    LlamaRMSNorm(target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(self.num_aux_hidden_states)
                ]
            )
        else:
            self.fc_norm = None

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_output = getattr(config, "norm_output", False)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        if self.vocab_size != self.target_vocab_size:
            self.register_buffer("t2d", torch.ones(self.target_vocab_size, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(self.vocab_size, dtype=torch.int64))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # eagle 3 requires hidden states from 3 layers
        expected_size = self.fc.in_features
        if hidden_states.size(-1) != expected_size:
            raise ValueError(
                f"Target hidden states size mismatch: {hidden_states.size(-1)} != expected: {expected_size}"
            )
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )
        if os.environ.get("ANGELSPEC_EAGLE3_PROJ_FP32", "1") in {"0", "false", "False"}:
            return self.fc(hidden_states.to(self.fc.weight.dtype))
        proj = F.linear(
            hidden_states.to(torch.float32),
            self.fc.weight.to(torch.float32),
            None if self.fc.bias is None else self.fc.bias.to(torch.float32),
        )
        return proj.to(self.fc.weight.dtype)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_keys=cache_keys,
            cache_values=cache_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
