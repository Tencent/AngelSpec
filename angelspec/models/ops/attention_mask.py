import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

MTP_ATTENTION_BACKENDS = ("sdpa", "fa", "flex_attention", "usp")


def validate_mtp_backend(backend):
    if backend not in MTP_ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported MTP attention_backend={backend!r}; expected {MTP_ATTENTION_BACKENDS}"
        )
    return backend


def mtp_validity_mask(mask, batch_size, seq_len, device):
    if mask is None:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    if mask.ndim != 2 or tuple(mask.shape) != (batch_size, seq_len):
        raise ValueError(
            "MTP validity mask must have shape [batch, sequence]; additive masks cannot be flattened"
        )
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("MTP validity mask must contain only 0 and 1")
    return mask.to(device=device, dtype=torch.bool)


def mtp_causal_mask(mask, doc_ids, batch_size, seq_len, device, dtype):
    positions = torch.arange(seq_len, device=device)
    keep = (positions[:, None] >= positions[None, :])[None, None]
    if mask is not None and mask.ndim == 4:
        if mask.shape[0] not in (1, batch_size) or tuple(mask.shape[1:]) != (1, seq_len, seq_len):
            raise ValueError("MTP additive mask must have shape [batch, 1, sequence, sequence]")
        if not mask.is_floating_point():
            raise ValueError("MTP four-dimensional masks must be floating-point additive masks")
        additive = mask.to(device=device, dtype=dtype)
    else:
        valid = mtp_validity_mask(mask, batch_size, seq_len, device)
        keep = keep & valid[:, None, None, :]
        additive = torch.zeros((), device=device, dtype=dtype)
    if doc_ids is not None:
        if tuple(doc_ids.shape) != (batch_size, seq_len):
            raise ValueError("MTP document IDs must match the token grid")
        same = (doc_ids[:, :, None] == doc_ids[:, None, :]) & (doc_ids[:, :, None] >= 0)
        keep = keep & same[:, None]
    return additive.masked_fill(~keep, torch.finfo(dtype).min)


def materialize_attention_mask(block_mask, batch_size, q_len, kv_len, device):
    if block_mask is None:
        return None
    if isinstance(block_mask, torch.Tensor):
        if block_mask.dtype != torch.bool or block_mask.ndim != 4:
            raise ValueError("Dense block masks must be four-dimensional boolean tensors")
        return block_mask
    return create_mask(block_mask.mask_mod, batch_size, 1, q_len, kv_len, device=device)


def block_mask_sdpa(query, key, value, block_mask, **kwargs):
    keep = materialize_attention_mask(
        block_mask, query.shape[0], query.shape[-2], key.shape[-2], query.device
    )
    return F.scaled_dot_product_attention(query, key, value, attn_mask=keep, **kwargs)
