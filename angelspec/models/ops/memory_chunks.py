import torch
from torch.utils.checkpoint import checkpoint


def chunked_context_call(function, tensors, chunk_size=0, recompute=False):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError("chunk_size must be a nonnegative integer")
    length = tensors[0].shape[1]
    if any(tensor.shape[1] != length for tensor in tensors):
        raise ValueError("Context sequence lengths differ")

    def apply(parts):
        if recompute and torch.is_grad_enabled():
            return checkpoint(function, *parts, use_reentrant=False)
        return function(*parts)

    if not chunk_size or length <= chunk_size:
        return apply(tensors)
    return torch.cat(
        [
            apply(tuple(tensor[:, start : start + chunk_size] for tensor in tensors))
            for start in range(0, length, chunk_size)
        ],
        dim=1,
    )


@torch.no_grad()
def normalize_shifted_hidden(hidden, norm, chunk_size, *, shift):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    length = hidden.shape[1]
    if length == 0:
        return norm(hidden)
    output = None
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        value = hidden[:, start + int(shift) : min(end + int(shift), length)]
        if shift and end == length:
            value = torch.cat(
                [value, hidden.new_zeros(hidden.shape[0], 1, hidden.shape[2])], dim=1
            )
        normalized = norm(value)
        if output is None:
            output = normalized.new_empty(hidden.shape)
        output[:, start:end].copy_(normalized)
    return output
