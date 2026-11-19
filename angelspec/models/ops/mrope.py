import torch
from torch import nn


def validate_mrope_positions(position_ids, batch_size, seq_length):
    if position_ids is None or tuple(position_ids.shape) != (batch_size, 3, seq_length):
        shape = None if position_ids is None else tuple(position_ids.shape)
        raise ValueError(
            f"MRoPE requires prepared position_ids [{batch_size}, 3, {seq_length}], got {shape}; "
            "prepare positions from final tokens before packing or input shifting"
        )
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("MRoPE position_ids must be integer tensors")
    return position_ids.long()


def advance_mtp_positions(position_ids, cached_steps):
    if position_ids.ndim == 3 and cached_steps:
        return position_ids[:, :1, :].expand(-1, 3, -1) + cached_steps
    return position_ids + cached_steps


def validate_mrope_parameters(dim, rope_parameters):
    section = tuple(rope_parameters["mrope_section"])
    if (
        dim <= 0
        or dim % 2
        or len(section) != 3
        or any(type(s) is not int or s < 0 for s in section)
        or sum(section) * 2 != dim
    ):
        raise ValueError("mrope_section must contain three nonnegative integers summing to dim/2")
    if rope_parameters.get("rope_type", rope_parameters.get("type", "default")) != "default":
        raise ValueError("Hy4 MLA MRoPE currently supports only rope_type=default")
    if rope_parameters.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("Hy4 MLA MRoPE requires full qk_rope_head_dim rotation")
    if not 0 < float(rope_parameters["rope_theta"]) < float("inf"):
        raise ValueError("rope_theta must be positive and finite")


def draft_block_positions(anchor_indices, context_positions, block_size):
    offsets = torch.arange(block_size, device=anchor_indices.device)
    if context_positions.ndim == 3:
        indices = anchor_indices[:, None].expand(-1, 3, -1)
        bases = torch.gather(context_positions, -1, indices)
        return (bases[..., None] + offsets).flatten(2)
    bases = torch.gather(context_positions, -1, anchor_indices)
    return (bases[..., None] + offsets).flatten(1)


def mrope_text_block_mask(position_ids, block_size, *, input_ids=None, visual_token_ids=()):
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if position_ids is None or position_ids.ndim != 3:
        raise ValueError("MRoPE text blocks require position_ids [B, 3, L]")
    bsz, _, seq_len = position_ids.shape
    position_ids = validate_mrope_positions(position_ids, bsz, seq_len)
    text = (position_ids == position_ids[:, :1]).all(dim=1)
    if visual_token_ids:
        if input_ids is None or input_ids.shape != text.shape:
            raise ValueError("Visual token filtering requires input_ids [B, L]")
        for token_id in visual_token_ids:
            text &= input_ids != token_id
    if seq_len < block_size:
        return torch.zeros_like(text)
    prefix = torch.nn.functional.pad((~text).long().cumsum(-1), (1, 0))
    valid = prefix[:, block_size:] == prefix[:, :-block_size]
    if block_size > 1:
        broken = position_ids[:, 0, 1:] != position_ids[:, 0, :-1] + 1
        steps = torch.nn.functional.pad(broken.long().cumsum(-1), (1, 0))
        valid &= steps[:, block_size - 1 :] == steps[:, : seq_len - block_size + 1]
    return torch.nn.functional.pad(valid, (0, block_size - 1), value=False)


class MRotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_parameters):
        super().__init__()
        self.dim = int(dim)
        self.base = float(rope_parameters["rope_theta"])
        self.section = tuple(rope_parameters["mrope_section"])
        self.interleaved = bool(rope_parameters.get("mrope_interleaved", False))
        validate_mrope_parameters(self.dim, rope_parameters)
        self.register_buffer("inv_freq", self._frequencies(torch.device("cpu")), persistent=False)

    def _frequencies(self, device):
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self._frequencies(self.inv_freq.device)
        return self

    @torch.no_grad()
    def forward(self, ref, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[:, None, :].expand(-1, 3, -1)
        position_ids = validate_mrope_positions(
            position_ids, position_ids.shape[0], position_ids.shape[-1]
        ).to(ref.device)
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, 1, -1)
        if self.interleaved:
            idx = torch.arange(self.dim // 2, device=ref.device)
            h_mask = (idx % 3 == 1) & (idx < 3 * self.section[1])
            w_mask = (idx % 3 == 2) & (idx < 3 * self.section[2])
            freqs = torch.where(w_mask, freqs[:, 2], torch.where(h_mask, freqs[:, 1], freqs[:, 0]))
        else:
            freqs = torch.cat(
                [part[:, axis] for axis, part in enumerate(freqs.split(self.section, dim=-1))],
                dim=-1,
            )
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
