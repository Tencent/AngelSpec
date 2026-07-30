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


def validate_dflash_usp_layout(*, attention_backend: Optional[str]) -> None:
    """DFlash-family attention has no sequence-parallel implementation yet."""
    if attention_backend == "usp":
        raise NotImplementedError(
            "DFlash/DSpark/DFly do not currently support attention_backend=usp. "
            "Use DFlash packing/windowing for efficiency, or MTP for USP training."
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

    chunk_len = max(local_len - ttt_length, 0)
    ring_chunk = chunk_len * sp_ulysses_size
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
