import torch


@torch.no_grad()
def chunked_argmax(hidden_states, compute_logits, chunk_size=0):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError("chunk_size must be a nonnegative integer")
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have a token and hidden dimension")
    if chunk_size == 0:
        return compute_logits(hidden_states).argmax(-1)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    output = torch.empty(flat.shape[0], device=hidden_states.device, dtype=torch.long)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        output[start:end] = compute_logits(flat[start:end]).argmax(-1)
    return output.reshape(hidden_states.shape[:-1])
