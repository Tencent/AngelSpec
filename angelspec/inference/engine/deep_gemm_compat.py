"""Expose DeepGEMM's unified FP8/FP4 MQA-logits entry points on older DeepGEMM builds.

vLLM's Hy4 flash index-pool selector (its compressed indexer) calls
``deep_gemm.fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits``, whose FP8 path
(``q = (values, None)``) is the older ``fp8_mqa_logits`` / ``fp8_paged_mqa_logits``
with the query wrapped in a tuple. Builds that only ship the older names fail at
the first prefill. This module has no AngelSpec imports so it can also be loaded
from a ``sitecustomize`` when the vLLM plugin is disabled.
"""

import sys


def _fp8_query(q):
    values, scale = q
    if scale is not None:
        raise NotImplementedError("this DeepGEMM build has no FP4 MQA-logits kernels")
    return values


def apply_deep_gemm_fp8_fp4_compat() -> bool:
    try:
        import deep_gemm
    except ImportError:
        return False

    if not hasattr(deep_gemm, "fp8_fp4_mqa_logits") and hasattr(deep_gemm, "fp8_mqa_logits"):

        def fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
            return deep_gemm.fp8_mqa_logits(
                _fp8_query(q), kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
            )

        deep_gemm.fp8_fp4_mqa_logits = fp8_fp4_mqa_logits

    if not hasattr(deep_gemm, "fp8_fp4_paged_mqa_logits") and hasattr(
        deep_gemm, "fp8_paged_mqa_logits"
    ):

        def fp8_fp4_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=False,
        ):
            return deep_gemm.fp8_paged_mqa_logits(
                _fp8_query(q),
                kv_cache,
                weights,
                context_lens,
                block_tables,
                schedule_metadata,
                max_model_len,
                clean_logits,
            )

        deep_gemm.fp8_fp4_paged_mqa_logits = fp8_fp4_paged_mqa_logits

    # vLLM resolves the symbols once (its _lazy_init returns early as soon as any
    # is set), so only refresh them when that already happened.
    vllm_deep_gemm = sys.modules.get("vllm.utils.deep_gemm")
    resolved = vllm_deep_gemm is not None and any(
        getattr(vllm_deep_gemm, attr, None) is not None
        for attr in ("_fp8_gemm_nt_impl", "_get_paged_mqa_logits_metadata_impl")
    )
    if resolved:
        for name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"):
            attr = f"_{name}_impl"
            if getattr(vllm_deep_gemm, attr, None) is None and hasattr(deep_gemm, name):
                setattr(vllm_deep_gemm, attr, getattr(deep_gemm, name))
    return True
