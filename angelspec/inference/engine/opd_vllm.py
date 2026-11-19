"""Deterministic Hy4 flash teacher adapters over the installed vLLM kernels."""

import torch

TEACHER_VERSION = "hy4-flash-opd-v1"


class PrefillTeacherWorkerExt:
    def opd_configure_teacher(self):
        return configure_teacher(self.model_runner)


def stable_coarse_topk(logits, ends, k):
    """Exact FP32 ordering, breaking ties by smaller causal group id.

    Pack the monotonic float bit pattern and index into an int64 key. This
    keeps unequal scores unchanged, unlike adding an epsilon to break ties.
    Native CUDA radix top-k may choose different equal-score groups on replay.
    """
    width = logits.shape[1]
    ids = torch.arange(width, device=logits.device)
    scores = logits.float().contiguous()
    scores = scores.masked_fill(ids[None] >= ends[:, None], -torch.inf)
    scores = torch.where(scores == 0, 0.0, scores)  # canonicalize signed zero
    bits = scores.view(torch.int32).long()
    ordered = torch.where(bits < 0, ~(bits & 0x7FFFFFFF), bits)
    keys = (ordered << 32) | (0xFFFFFFFF - ids[None])
    chosen = keys.topk(min(k, width), dim=-1).indices
    chosen = chosen.masked_fill(chosen >= ends[:, None], -1)
    if width < k:
        chosen = torch.nn.functional.pad(chosen, (0, k - width), value=-1)
    return chosen.to(torch.int32)


def stable_select_groups(logits, upper, budget):
    """vLLM's ``select_groups`` (recalled groups ascending, -1 padded) with equal scores
    broken by the smaller group id, so a replayed prefill recalls the same groups."""
    chosen = stable_coarse_topk(logits, upper, budget)
    sentinel = torch.iinfo(torch.int32).max
    chosen = torch.where(chosen >= 0, chosen, sentinel).sort(dim=-1).values
    return chosen.masked_fill(chosen == sentinel, -1)


def configure_teacher(mr):
    """Configure after KV allocation, before the first scored forward.

    Under ``VLLM_BATCH_INVARIANT`` every scored row is computed on its own; what is
    left to fix is the IndexPool recall, whose top-k picks any of equally scored
    groups. Window and sparse attention take one kernel path for every query length
    (bf16 caches, and fp8 ones through the mixed-batch metadata), so trunk prefill and
    short branches already share their math.
    """
    import vllm.envs as envs
    from vllm.models.hy_v4_flash.nvidia import compressed_indexer
    from vllm.models.hy_v4_flash.nvidia.swa import FlashWindowIndexer

    if not envs.VLLM_BATCH_INVARIANT:
        raise ValueError("OPD teacher requires VLLM_BATCH_INVARIANT=1")
    modules = list(mr.get_model().modules())
    indexers = sum(isinstance(m, compressed_indexer.FlashCompressedIndexer) for m in modules)
    windows = sum(isinstance(m, FlashWindowIndexer) for m in modules)
    if not indexers or not windows:
        raise ValueError("OPD teacher requires Hy4 flash compressed indexer and SWA")
    compressed_indexer.select_groups = stable_select_groups
    return {"version": TEACHER_VERSION, "indexers": indexers, "window_indexers": windows}
