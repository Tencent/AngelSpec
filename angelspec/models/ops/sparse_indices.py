import torch


def build_contiguous_causal_ranges(
    seq_len: int,
    *,
    query_offset: int = 0,
    query_len: int | None = None,
    doc_ids: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_len is None:
        query_len = seq_len - query_offset
    if query_offset < 0 or query_len < 0 or query_offset + query_len > seq_len:
        raise ValueError(
            f"invalid query range offset={query_offset}, length={query_len}, seq_len={seq_len}"
        )

    ref = doc_ids if doc_ids is not None else valid_mask
    if ref is not None:
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        if ref.shape[-1] != seq_len:
            raise ValueError(f"range metadata length {ref.shape[-1]} != seq_len {seq_len}")
        batch = ref.shape[0]
        device = ref.device
    else:
        batch = 1
        device = device or torch.device("cpu")

    positions = torch.arange(seq_len, device=device, dtype=torch.int64).expand(batch, -1)
    if doc_ids is None:
        starts_full = torch.zeros_like(positions)
    else:
        docs = doc_ids.unsqueeze(0) if doc_ids.dim() == 1 else doc_ids
        boundary = torch.ones_like(docs, dtype=torch.bool)
        boundary[:, 1:] = docs[:, 1:] != docs[:, :-1]
        starts_full = (
            torch.where(boundary, positions, torch.zeros_like(positions)).cummax(-1).values
        )

    q_positions = torch.arange(
        query_offset,
        query_offset + query_len,
        device=device,
        dtype=torch.int64,
    ).expand(batch, -1)
    starts = starts_full.gather(-1, q_positions)
    ends = q_positions + 1

    if valid_mask is not None:
        valid = valid_mask.unsqueeze(0) if valid_mask.dim() == 1 else valid_mask
        if tuple(valid.shape) != (batch, seq_len):
            raise ValueError("validity mask must match [batch, sequence]")
        if valid.dtype != torch.bool and not bool(((valid == 0) | (valid == 1)).all()):
            raise ValueError("validity mask must contain only 0 and 1")
        valid = valid.to(torch.bool)
        if doc_ids is not None:
            valid = valid & (docs >= 0)
        last_invalid = torch.where(~valid, positions, -1).cummax(-1).values
        valid_starts = torch.maximum(starts_full, last_invalid + 1)
        counts = torch.nn.functional.pad(valid.long().cumsum(-1), (1, 0))
        holes = valid & (counts.gather(-1, valid_starts) > counts.gather(-1, starts_full))
        if bool(holes.any()):
            raise ValueError(
                "the DSA indexer requires a contiguous valid key span within each document; "
                "mask contains holes"
            )
        q_valid = valid.gather(-1, q_positions)
        starts = torch.where(q_valid, valid_starts.gather(-1, q_positions), starts)
        ends = torch.where(q_valid, ends, starts)
    elif doc_ids is not None:
        ends = torch.where(docs.gather(-1, q_positions) >= 0, ends, starts)

    return starts.to(torch.int32), ends.to(torch.int32)


def compute_compressed_cu_seqlens(cu_seqlens: torch.Tensor, ratio: int) -> torch.Tensor:
    """Cumulative counts of each document's complete ``ratio``-token groups."""
    counts = torch.div(cu_seqlens[1:] - cu_seqlens[:-1], ratio, rounding_mode="floor")
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(cu_seqlens.dtype)


def compressed_group_token_offsets(cu_seqlens: torch.Tensor, ratio: int) -> torch.Tensor:
    """First token of every complete group, documents in order."""
    compressed = compute_compressed_cu_seqlens(cu_seqlens, ratio).long()
    counts = compressed[1:] - compressed[:-1]
    doc_starts = torch.repeat_interleave(cu_seqlens[:-1].long(), counts)
    first_group = torch.repeat_interleave(compressed[:-1], counts)
    group = torch.arange(int(compressed[-1]), device=cu_seqlens.device)
    return doc_starts + (group - first_group) * ratio


def compressed_middle_group_ranges(
    query_starts: torch.Tensor,
    query_ends: torch.Tensor,
    compressed_doc_starts: torch.Tensor,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled groups each query scores: its document's complete groups before its own."""
    complete = torch.div(
        (query_ends - query_starts - 1).clamp(min=0), ratio, rounding_mode="floor"
    )
    return compressed_doc_starts, compressed_doc_starts + complete


def build_forced_tail_candidates(
    query_starts: torch.Tensor, query_ends: torch.Tensor, ratio: int
) -> torch.Tensor:
    """Tokens of each query's own, possibly partial, group up to the query; ``-1`` pads."""
    offset = torch.div((query_ends - query_starts - 1).clamp(min=0), ratio, rounding_mode="floor")
    tokens = (query_starts + offset * ratio).unsqueeze(-1) + torch.arange(
        ratio, device=query_starts.device
    )
    return torch.where(tokens < query_ends.unsqueeze(-1), tokens, -1)


def expand_compressed_groups_to_tokens(
    groups: torch.Tensor, cu_seqlens: torch.Tensor, ratio: int
) -> torch.Tensor:
    """Token indices of compressed groups, ``ratio`` per group; ``-1`` groups stay ``-1``."""
    starts = compressed_group_token_offsets(cu_seqlens, ratio)
    if starts.numel() == 0:
        return groups.new_full((*groups.shape[:-1], groups.shape[-1] * ratio), -1)
    tokens = starts[groups.clamp(min=0)].unsqueeze(-1) + torch.arange(ratio, device=groups.device)
    return torch.where((groups >= 0).unsqueeze(-1), tokens, -1).flatten(-2)


def append_diagonal_kv_indices(
    base_indices: torch.Tensor,
    *,
    seq_len: int,
    num_cached_steps: int,
    alignment: int = 64,
) -> torch.Tensor:
    if base_indices.dim() not in (2, 3, 4):
        raise ValueError(f"base_indices must be 2D/3D/4D, got {base_indices.shape}")
    if num_cached_steps < 1:
        raise ValueError(f"num_cached_steps must be >= 1, got {num_cached_steps}")
    if alignment < 1:
        raise ValueError(f"alignment must be >= 1, got {alignment}")

    indices = base_indices.to(torch.int32)
    if indices.dim() == 2:
        indices = indices.unsqueeze(0).unsqueeze(2)
    elif indices.dim() == 3:
        indices = indices.unsqueeze(2)
    if indices.shape[2] != 1:
        raise ValueError(f"sparse MLA requires kv_group=1, got {indices.shape[2]}")
    if indices.shape[1] != seq_len:
        raise ValueError(f"query length {indices.shape[1]} != seq_len {seq_len}")

    base_width = indices.shape[-1]
    width = base_width + num_cached_steps - 1
    padded = ((width + alignment - 1) // alignment) * alignment
    if num_cached_steps == 1 and padded == base_width:
        return indices.contiguous()
    merged = indices.new_empty((*indices.shape[:-1], padded))
    merged[..., :base_width].copy_(indices)
    if num_cached_steps > 1:
        positions = torch.arange(seq_len, device=indices.device, dtype=torch.int32)
        positions = positions.view(1, seq_len, 1)
        for step in range(1, num_cached_steps):
            merged[..., base_width + step - 1].copy_(positions + step * seq_len)
    if padded > width:
        merged[..., width:].fill_(-1)
    return merged
