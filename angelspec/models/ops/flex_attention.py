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

import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
    or_masks,
)
from transformers.utils import is_torchdynamo_compiling

# flex_attention recompiles when a tensor shape changes (e.g. KV_LEN as the
# sequence length varies step to step); dynamo promotes that dim to dynamic on
# the first such recompile, so steady state is ~2 compiled graphs no matter how
# many distinct lengths appear (verified torch 2.11, tools/repro_flex_recompile.py).
# Varying anchor positions do NOT recompile — they enter mask_mod as tensor
# inputs. This limit is a cheap safety cap against pathological multi-shape
# specialization across the shared draft paths (DFlash/DSpark/MTP); the default
# (8) is usually enough, so this rarely bites.
try:
    dynamo.config.recompile_limit = 128
except AttributeError:
    dynamo.config.cache_size_limit = 128

# Without ATEN fallback, inductor's GEMM autotuner can fail with
# NoValidChoicesError during FlexAttention backward (Issue 10).
if "ATEN" not in getattr(inductor_config, "max_autotune_gemm_backends", ""):
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"


# Reference Implementation https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py
class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled:
            self._compiled_flex_attention = torch.compile(
                flex_attention,
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    flex_attention_compiled = (
        WrappedFlexAttention()() if not is_torchdynamo_compiling() else flex_attention
    )
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


def compile_friendly_create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
    BLOCK_SIZE: "int | tuple[int, int]" = 128,
    _compile: bool = False,
):
    """Create block mask directly (no compilation wrapper).

    create_block_mask is fast enough without
    torch.compile, and compiling it adds overhead with torch 2.9.1.

    ``_compile=True`` routes through flex's compiled block-mask builder, which is
    O(num_blocks) instead of materialising the full (Q_LEN, KV_LEN) boolean grid.
    Required for long sequences: at 128k the un-compiled path tries to allocate a
    128 GiB grid and OOMs. Callers with small Q_LEN keep the default.
    """
    return create_block_mask(
        mask_mod,
        B,
        H,
        Q_LEN,
        KV_LEN,
        device,
        BLOCK_SIZE=BLOCK_SIZE,
        _compile=_compile,
    )


def generate_eagle3_mask(Q_LEN: int, KV_LEN: int, lck: int = 0):
    """Eagle3 causal+suffix mask_mod.

    Note: to support packed sequences (multiple variable-length samples
    concatenated into one row), seq_lengths must be passed in here so the
    mask can clamp causal and suffix clauses to per-sample boundaries; for
    the current single-sample-per-row case the legacy seq_lengths clauses
    are tautological on every valid q row and are omitted.
    """

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def suffix_mask(b, h, q_idx, kv_idx):
        return (kv_idx >= Q_LEN) & ((kv_idx - q_idx) % Q_LEN == 0)

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod


def _build_eagle3_block_mask_tensors(
    Q_LEN: int,
    KV_LEN: int,
    B: int,
    H: int,
    Q_BS: int,
    KV_BS: int,
    device: torch.device,
):
    """Return Eagle3 BlockMask tensors as 8 tensors (kv-dir + q-dir, partial + full).

    Returns (in BlockMask layout order):
        kv_num, kv_idx, full_kv_num, full_kv_idx,
        q_num,  q_idx,  full_q_num,  full_q_idx

    Split out from BlockMask wrapping so it can be torch.compile'd: BlockMask
    carries Python-side closures and cannot be a compiled-graph return value.

    Supports rectangular Q×KV block sizes (FA4 Blackwell uses Q_BS=256, KV_BS=128).
    Requires KV_BS to divide Q_BS so each Q-block aligns to an integer number of
    KV-blocks (the only configuration FA4 emits today).

    Block classification:
      Causal region (kv < Q_LEN):
        * Strictly below diagonal (kj < qi*r) -> FULL (all True, skip mask_mod)
        * Diagonal slab (kj in [qi*r, (qi+1)*r)) -> PARTIAL (lower-triangular)
        * Above diagonal -> empty (omitted)
      Suffix region (kv >= Q_LEN), one round per Q_LEN cols:
        * Per round, Q-block qi has r PARTIAL blocks (one per BK-slot within Q-block).
        * Suffix blocks are never FULL (mask is at most a thin diagonal).
    """
    r = Q_BS // KV_BS  # KV-blocks per Q-block (1 on Hopper, 2 on Blackwell SM100+)
    n_q = Q_LEN // Q_BS
    n_kv = KV_LEN // KV_BS
    n_kv_causal = Q_LEN // KV_BS  # KV-blocks covering the first (causal) round
    n_rounds = KV_LEN // Q_LEN

    # ---- KV direction (forward iteration: per Q-block, list KV-blocks) -----
    # Both partial and full kv_indices must be width n_kv -- this is what FA4
    # BlockSparseTensorsTorch + flex_attention's create_block_mask emit; FA4's
    # infer_block_sparse_expected_shapes asserts on it explicitly.  Only the
    # first kv_num/full_kv_num entries per row are read; the rest are padding.
    qi = torch.arange(n_q, device=device, dtype=torch.int32)
    qi_b = qi.unsqueeze(1)
    col_kv = torch.arange(n_kv, device=device, dtype=torch.int32).unsqueeze(0)

    # FULL: kj in [0, qi*r) (causal-below-diagonal).
    full_kv_num_1d = (qi * r).to(torch.int32)
    full_kv_idx_2d = torch.where(
        col_kv < full_kv_num_1d.unsqueeze(1),
        col_kv.expand(n_q, n_kv),
        torch.zeros_like(col_kv).expand(n_q, n_kv),
    )

    # PARTIAL: r diagonal + r * (n_rounds - 1) suffix per Q-block.
    # Round s in [0, n_rounds): kj in [qi*r + s*n_kv_causal, (qi+1)*r + s*n_kv_causal).
    partial_count = r * n_rounds
    s_b = col_kv // r  # which round
    sub_b = col_kv % r  # offset within round
    kv_idx_2d = torch.where(
        col_kv < partial_count,
        qi_b * r + sub_b + s_b * n_kv_causal,
        torch.zeros_like(col_kv).expand(n_q, n_kv),
    )
    kv_num_1d = torch.full((n_q,), partial_count, dtype=torch.int32, device=device)

    # ---- Q direction (backward iteration: per KV-block, list Q-blocks) -----
    # Same n_q-wide padding for FA4 backward block-sparsity.
    kj = torch.arange(n_kv, device=device, dtype=torch.int32)
    kj_b = kj.unsqueeze(1)
    col_q = torch.arange(n_q, device=device, dtype=torch.int32).unsqueeze(0)
    is_causal_kj = kj < n_kv_causal

    # PARTIAL: every kj has exactly 1 partial Q-block (the diagonal Q-block).
    #   Causal kj: qi = kj // r.
    #   Suffix kj: qi = (kj % n_kv_causal) // r.
    q_idx_diag = torch.where(is_causal_kj, kj // r, (kj % n_kv_causal) // r)
    q_idx_2d = torch.where(
        col_q == 0,
        q_idx_diag.unsqueeze(1).expand(n_kv, n_q),
        torch.zeros_like(col_q).expand(n_kv, n_q),
    )
    q_num_1d = torch.ones(n_kv, dtype=torch.int32, device=device)

    # FULL: causal kj has (n_q - 1 - kj//r) full Q-blocks (qi > kj//r); suffix has 0.
    diag_qi = kj_b // r
    full_q_num_per_causal = (n_q - 1) - (kj // r)
    full_q_num_1d = torch.where(is_causal_kj, full_q_num_per_causal, torch.zeros_like(kj)).to(
        torch.int32
    )
    full_q_idx_2d = torch.where(
        col_q < full_q_num_1d.unsqueeze(1),
        diag_qi + 1 + col_q,
        torch.zeros_like(col_q).expand(n_kv, n_q),
    )

    # flex_attention iterates these directly; force contiguous storage.
    def _expand_1d(t):
        return t.unsqueeze(0).unsqueeze(0).expand(B, H, -1).contiguous()

    def _expand_2d(t):
        return t.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).contiguous()

    return (
        _expand_1d(kv_num_1d),
        _expand_2d(kv_idx_2d),
        _expand_1d(full_kv_num_1d),
        _expand_2d(full_kv_idx_2d),
        _expand_1d(q_num_1d),
        _expand_2d(q_idx_2d),
        _expand_1d(full_q_num_1d),
        _expand_2d(full_q_idx_2d),
    )


# dynamic=True so KV_LEN growing per TTT step doesn't recompile; inductor's
# persistent cache amortises the one-off compile across runs.
_compiled_build_tensors = None


@torch.compiler.disable(recursive=False)
def _get_compiled_build_tensors():
    global _compiled_build_tensors
    if _compiled_build_tensors is None:
        _compiled_build_tensors = torch.compile(
            _build_eagle3_block_mask_tensors,
            dynamic=True,
            fullgraph=True,
        )
    return _compiled_build_tensors


def _normalize_block_size(BLOCK_SIZE) -> tuple[int, int]:
    """Accept BLOCK_SIZE as int or (Q_BS, KV_BS) tuple; return (Q_BS, KV_BS)."""
    if isinstance(BLOCK_SIZE, int):
        return BLOCK_SIZE, BLOCK_SIZE
    Q_BS, KV_BS = BLOCK_SIZE
    return int(Q_BS), int(KV_BS)


def build_eagle3_block_mask(
    Q_LEN: int,
    KV_LEN: int,
    B: int = 1,
    H: int = 1,
    device: torch.device = "cuda",
    BLOCK_SIZE: "int | tuple[int, int]" = 128,
) -> "BlockMask":
    """Build Eagle3 BlockMask analytically -- O(num_blocks) memory and time.

    create_block_mask materialises the full (Q_LEN, KV_LEN) boolean grid
    (~112 GB at Q=49K, KV=245K). This builds the sparse kv/q indices
    directly from the known Eagle3 structure (causal first round + diagonal
    suffix rounds), so peak memory drops to a few MB.

    Requires Q_LEN multiple of Q_BS, KV_LEN multiple of KV_BS, KV_BS dividing
    Q_BS, and KV_LEN a multiple of Q_LEN.  ``BLOCK_SIZE`` accepts either an
    int (square Q_BS=KV_BS) or a ``(Q_BS, KV_BS)`` tuple (FA4 Blackwell uses
    256x128).  Use ``eagle3_block_mask`` for the dispatching wrapper that falls
    back to create_block_mask otherwise.

    Populates both PARTIAL (kv_*/q_*) and FULL (full_kv_*/full_q_*) block
    tensors so flex_attention and FA4 BlockSparseTensorsTorch can both fast-path
    fully-causal-below-diagonal blocks (skip mask_mod evaluation).
    """
    Q_BS, KV_BS = _normalize_block_size(BLOCK_SIZE)
    assert Q_LEN % Q_BS == 0 and KV_LEN % KV_BS == 0
    assert Q_BS % KV_BS == 0, f"Q_BS ({Q_BS}) must be a multiple of KV_BS ({KV_BS})"
    assert (
        KV_LEN % Q_LEN == 0
    ), f"build_eagle3_block_mask requires KV_LEN to be a multiple of Q_LEN; got Q_LEN={Q_LEN}, KV_LEN={KV_LEN}"

    # Skip the compiled path when nested inside another torch.compile graph.
    builder = (
        _build_eagle3_block_mask_tensors
        if is_torchdynamo_compiling()
        else _get_compiled_build_tensors()
    )
    kv_num, kv_idx, full_kv_num, full_kv_idx, q_num, q_idx, full_q_num, full_q_idx = builder(
        Q_LEN, KV_LEN, B, H, Q_BS, KV_BS, device
    )

    def mask_mod(b, h, q, kv):
        causal = (kv < Q_LEN) & (q >= kv)
        suffix = (kv >= Q_LEN) & ((kv - q) % Q_LEN == 0)
        return causal | suffix

    return BlockMask(
        seq_lengths=(Q_LEN, KV_LEN),
        kv_num_blocks=kv_num,
        kv_indices=kv_idx,
        full_kv_num_blocks=full_kv_num,
        full_kv_indices=full_kv_idx,
        q_num_blocks=q_num,
        q_indices=q_idx,
        full_q_num_blocks=full_q_num,
        full_q_indices=full_q_idx,
        BLOCK_SIZE=(Q_BS, KV_BS),
        mask_mod=mask_mod,
    )


def eagle3_block_mask(
    Q_LEN: int,
    KV_LEN: int,
    *,
    B: int = 1,
    H: int = 1,
    device: torch.device = "cuda",
    BLOCK_SIZE: "int | tuple[int, int]" = 128,
    lck: int = 0,
) -> "BlockMask":
    """Eagle3 block-mask dispatcher -- analytical when possible, fallback otherwise.

    Eagle3 training appends one full Q_LEN-sized round per step, so in normal
    training the analytical builder's preconditions
    ``(Q_LEN % Q_BS == 0 and KV_LEN % KV_BS == 0 and KV_LEN % Q_LEN == 0)``
    always hold.  The create_block_mask fallback only triggers for tests/edge
    cases (tiny sequence lengths, non-aligned shapes), where its O(Q*KV)
    memory cost is irrelevant.

    Args:
        Q_LEN: query length (current round).
        KV_LEN: total KV length (cached + current).
        B: batch size for the BlockMask (broadcast-friendly when 1).
        H: head count for the BlockMask (broadcast-friendly when 1).
        device: target device.
        BLOCK_SIZE: flex_attention block size; ``int`` (square) or
            ``(Q_BS, KV_BS)`` tuple.  FA4 Blackwell SM100+ uses ``(256, 128)``.
            Defaults to 128 (square).
        lck: number of completed rounds; only used to name the fallback
            mask_mod for debug clarity.

    Returns:
        A flex_attention BlockMask implementing the Eagle3 causal+suffix
        pattern.
    """
    Q_BS, KV_BS = _normalize_block_size(BLOCK_SIZE)
    use_analytical = (
        Q_LEN % Q_BS == 0 and KV_LEN % KV_BS == 0 and Q_BS % KV_BS == 0 and KV_LEN % Q_LEN == 0
    )
    if use_analytical:
        return build_eagle3_block_mask(
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            B=B,
            H=H,
            device=device,
            BLOCK_SIZE=(Q_BS, KV_BS),
        )

    # Fallback for non-aligned shapes (typically only seen in tests).
    # TODO: Remove the usage of uncompiled create_block_mask after
    # https://github.com/pytorch/pytorch/issues/160018
    creator = create_block_mask if Q_LEN <= 128 else compile_friendly_create_block_mask
    return creator(
        mask_mod=generate_eagle3_mask(Q_LEN=Q_LEN, KV_LEN=KV_LEN, lck=lck),
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
        BLOCK_SIZE=(Q_BS, KV_BS),
    )
