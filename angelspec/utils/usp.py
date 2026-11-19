from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def usp_chunk_size(global_len: int, sp_size: int) -> int:
    """Return the padded per-rank sequence length for USP."""
    if global_len < 0:
        raise ValueError(f"global_len must be non-negative, got {global_len}")
    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")
    return (global_len + sp_size - 1) // sp_size


def usp_dp_average_factor(group_world_size: int, sp_size: int) -> int:
    """DP divisor after summing sequence-sharded gradients over a flat group."""
    if group_world_size <= 0:
        raise ValueError(f"group_world_size must be positive, got {group_world_size}")
    if sp_size <= 0 or group_world_size % sp_size != 0:
        raise ValueError(
            f"group_world_size ({group_world_size}) must be divisible by sp_size ({sp_size})"
        )
    return group_world_size // sp_size


def slice_usp_sequence_shard(
    tensor: torch.Tensor,
    *,
    sp_rank: int,
    sp_size: int,
    padded_seq_len: int,
    pad_value: float | int = 0,
) -> torch.Tensor:
    if tensor.dim() < 2:
        raise ValueError(f"Expected sequence tensor with dim >= 2, got {tensor.shape}")
    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")
    if not 0 <= sp_rank < sp_size:
        raise ValueError(f"sp_rank ({sp_rank}) must be in [0, {sp_size})")
    if padded_seq_len < tensor.shape[1]:
        raise ValueError(
            f"padded_seq_len ({padded_seq_len}) is shorter than tensor seq len "
            f"({tensor.shape[1]})"
        )
    if padded_seq_len % sp_size != 0:
        raise ValueError(
            f"padded_seq_len ({padded_seq_len}) must be divisible by sp_size ({sp_size})"
        )
    shard_len = padded_seq_len // sp_size
    start = sp_rank * shard_len
    end = min(start + shard_len, tensor.shape[1])
    local = tensor[:, start:end]
    missing = shard_len - local.shape[1]
    if missing > 0:
        pad = [0, 0] * (tensor.dim() - 2) + [0, missing]
        local = F.pad(local, pad, value=pad_value)
    return local.contiguous()


def validate_mtp_usp_layout(
    *,
    attention_backend: Optional[str],
    usp_local_shard: bool,
    sp_ring_size: int,
) -> None:
    """Fail fast for USP layouts that MTP cannot execute correctly."""
    if attention_backend != "usp":
        return
    if not usp_local_shard:
        raise ValueError(
            "MTP with attention_backend=usp requires training.usp_local_shard=true. "
            "MTPModel shards the full sequence inside forward; USP pre-sharded input "
            "would be sharded a second time and silently train on incorrect positions."
        )
    if sp_ring_size != 1:
        raise NotImplementedError(
            "MTP USP currently supports pure Ulysses only: training.sp_ring_size "
            f"must be 1, got {sp_ring_size}. Ring attention is not implemented "
            "by the MTP attention backend."
        )


def validate_dflash_usp_layout(
    *,
    attention_backend: Optional[str],
    usp_local_shard: bool = False,
    num_anchors: int = 512,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    opd_enabled: bool = False,
    num_attention_heads: Optional[int] = None,
) -> None:
    """Validate DFlash-family USP constraints.

    DFlash USP combines:
      * **Context sequence sharding** (each rank holds S/sp of target hidden
        states; attention all-gathers context K/V along seq), and
      * **Anchor striding** across SP (each rank trains num_anchors/sp blocks).

    Under ``usp_local_shard``, the controller fans each sample to the SP group and
    the DataLoader may collate ``micro_batch_size >= 1`` full sequences before
    in-forward sharding.

    ``num_attention_heads`` is accepted for API compatibility but not required
    to divide ``sp_ulysses_size`` (draft Q stays full-head; only context is
    seq-sharded).
    """
    del num_attention_heads  # unused; kept for call-site compatibility
    if attention_backend != "usp":
        return
    if not usp_local_shard:
        raise ValueError(
            "DFlash with attention_backend=usp requires training.usp_local_shard=true. "
            "Each SP rank must receive the full sample and slice its own context "
            "shard in-forward; usp_local_shard=false would pre-shard the sequence "
            "and corrupt global anchor coordinates on non-zero ranks."
        )
    if sp_ring_size != 1:
        raise NotImplementedError(
            "DFlash USP currently supports pure Ulysses only: "
            f"sp_ring_size must be 1, got {sp_ring_size}."
        )
    if num_anchors % sp_ulysses_size != 0:
        raise ValueError(
            f"dflash_num_anchors ({num_anchors}) must be divisible by "
            f"sp_ulysses_size ({sp_ulysses_size}) for anchor-strided USP."
        )
    if opd_enabled:
        raise NotImplementedError(
            "OPD (on-policy distillation) is not yet supported with DFlash USP. "
            "Each SP rank holds only 1/sp_size of proposals; downstream scoring "
            "expects the full set. Disable dflash_opd_enabled for USP training."
        )


def split_usp_batch(
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    ttt_length: int,
    sp_rank: int,
    sp_size: int,
    ring_rank: int,
    sp_ring_size: int,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)

    batch_size, input_len = input_ids.shape
    global_len = min(max_len, input_len) if max_len is not None else input_len
    chunk_size = usp_chunk_size(global_len, sp_size)
    sp_ulysses_size = max(1, sp_size // sp_ring_size)
    start = sp_rank * chunk_size
    local_len = chunk_size + ttt_length
    end = min(start + local_len, global_len)

    loss_mask = loss_mask[:, :global_len].clone()

    def _slice_and_pad(tensor: torch.Tensor, axis: int, pad_value: int = 0):
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if axis == 0:
            tensor = tensor[:global_len, :]
            sliced = tensor[start : min(end, tensor.shape[0]), :]
            valid_len = sliced.shape[0]
            if valid_len < local_len:
                sliced = F.pad(sliced, (0, 0, 0, local_len - valid_len), value=pad_value)
        else:
            tensor = tensor[:, :global_len]
            sliced = tensor[:, start : min(end, tensor.shape[1])]
            valid_len = sliced.shape[1]
            if valid_len < local_len:
                pad_len = local_len - valid_len
                if tensor.dim() == 2:
                    sliced = F.pad(sliced, (0, pad_len), value=pad_value)
                else:
                    sliced = F.pad(sliced, (0, 0, 0, pad_len), value=pad_value)
        return sliced.contiguous(), valid_len

    input_ids, valid_len = _slice_and_pad(input_ids, axis=1, pad_value=0)
    loss_mask, _ = _slice_and_pad(loss_mask, axis=1, pad_value=0)
    if hidden_states.dim() == 2:
        hidden_states, _ = _slice_and_pad(hidden_states, axis=0, pad_value=0)
        hidden_states = hidden_states.unsqueeze(0)
    else:
        hidden_states, _ = _slice_and_pad(hidden_states, axis=1, pad_value=0)
    if target_hidden_states.dim() == 2:
        target_hidden_states, _ = _slice_and_pad(target_hidden_states, axis=0, pad_value=0)
        target_hidden_states = target_hidden_states.unsqueeze(0)
    else:
        target_hidden_states, _ = _slice_and_pad(target_hidden_states, axis=1, pad_value=0)

    attention_mask = torch.zeros(
        (batch_size, local_len), dtype=torch.long, device=input_ids.device
    )
    attention_mask[:, :valid_len] = 1

    local_chunk_size = max(local_len - ttt_length, 0)
    ring_chunk = local_chunk_size * sp_ulysses_size
    ring_start = ring_rank * ring_chunk
    position_ids = torch.arange(
        ring_start, ring_start + ring_chunk, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0)
    if batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)

    return (
        input_ids,
        attention_mask,
        loss_mask,
        hidden_states,
        target_hidden_states,
        position_ids,
    )
