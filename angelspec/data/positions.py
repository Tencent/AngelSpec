import hashlib
import json
from dataclasses import asdict, dataclass

import torch

from angelspec.models.ops.mrope import validate_mrope_positions

POSITION_RECIPE_KEY = "_position_recipe"
# Sample metadata: token indices of image-row newlines that the data loader replaced
# with the image (in video frames, the video) placeholder: the serving layout, see
# megatron_bin.
IMAGE_NEWLINE_SLOTS_KEY = "image_newline_slots"


@dataclass(frozen=True)
class PositionRecipe:
    provider: str = "hy_v4_flash"
    version: int = 1
    image_start_token_id: int = 120118
    image_end_token_id: int = 120119
    image_newline_token_id: int = 120121
    image_token_id: int = 120120
    video_token_id: int = 120124
    video_start_token_id: int = 120122
    video_end_token_id: int = 120123

    def __post_init__(self):
        if (
            self.provider not in ("hy_v4_flash", "provided")
            or type(self.version) is not int
            or self.version != 1
        ):
            raise ValueError(
                f"Unsupported position provider/version: {self.provider}/{self.version}"
            )
        ids = [v for k, v in asdict(self).items() if k.endswith("token_id")]
        if any(type(v) is not int or v < 0 for v in ids) or len(set(ids)) != len(ids):
            raise ValueError("Position token IDs must be distinct nonnegative integers")

    @property
    def identity(self):
        value = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


def position_recipe_from_config(config):
    if config is None or getattr(config, "attention_type", None) != "mla":
        return None
    rope = getattr(config, "rope_parameters", None) or {}
    if not rope.get("mrope_section"):
        return None
    provider = getattr(config, "position_provider", None)
    if provider is None:
        raise ValueError("MRoPE requires an explicit position_provider or target configuration")
    fields = asdict(PositionRecipe())
    fields["provider"] = provider
    fields["version"] = getattr(config, "position_recipe_version", 1)
    for key in fields:
        if key.endswith("token_id"):
            value = getattr(config, key, None)
            if value is not None:
                fields[key] = value
    return PositionRecipe(**fields)


def _spans(ids, start_id, end_id, label):
    starts = (ids == start_id).nonzero(as_tuple=True)[0].tolist()
    ends = (ids == end_id).nonzero(as_tuple=True)[0].tolist()
    if len(starts) != len(ends):
        raise ValueError(f"Unclosed or unmatched {label} boundary")
    if any(s >= e for s, e in zip(starts, ends)) or any(e >= s for e, s in zip(ends, starts[1:])):
        raise ValueError(f"Nested or malformed {label} boundaries")
    return list(zip(starts, ends))


def _hy4_document(ids, recipe):
    images = _spans(ids, recipe.image_start_token_id, recipe.image_end_token_id, "image")
    videos = _spans(ids, recipe.video_start_token_id, recipe.video_end_token_id, "video")
    pos = torch.empty((3, ids.numel()), dtype=torch.long)
    cursor = next_pos = 0
    visual_tokens = (recipe.image_token_id, recipe.video_token_id, recipe.image_newline_token_id)
    video_index = 0
    frames = [0] * len(videos)
    video_grids = {}
    for start, end in images:
        while video_index < len(videos) and videos[video_index][1] < start:
            video_index += 1
        if video_index < len(videos):
            vs, ve = videos[video_index]
            if start < vs < end or start < ve < end:
                raise ValueError("Image span crosses a video boundary")
            if vs < start < end < ve:
                frames[video_index] += 1
        inside_video = (
            video_index < len(videos)
            and videos[video_index][0] < start < end < videos[video_index][1]
        )
        text = ids[cursor:start]
        if any(bool((text == token).any()) for token in visual_tokens):
            raise ValueError("Visual token outside an image/frame span")
        lo = start + 1
        pos[:, cursor:lo] = torch.arange(next_pos, next_pos + lo - cursor)
        next_pos += lo - cursor
        body = ids[lo:end]
        newlines = (body == recipe.image_newline_token_id).nonzero(as_tuple=True)[0]
        if newlines.numel() == 0 or int(newlines[-1]) != body.numel() - 1:
            raise ValueError(
                "Image rows must end with image_newline tokens "
                f"(or name the rewritten ones in {IMAGE_NEWLINE_SLOTS_KEY!r} metadata)"
            )
        width = int(newlines[0]) + 1
        rows = newlines.numel()
        if (
            width < 2
            or body.numel() != rows * width
            or not torch.equal(newlines, torch.arange(width - 1, body.numel(), width))
        ):
            raise ValueError("Image span has irregular or empty rows")
        if inside_video:
            grid = (rows, width)
            if video_grids.setdefault(video_index, grid) != grid:
                raise ValueError("Hy4 video frames must share the same spatial grid")
        pixels = body.reshape(rows, width)[:, :-1]
        if not bool(((pixels == recipe.image_token_id) | (pixels == recipe.video_token_id)).all()):
            raise ValueError("Unsupported tokens inside image/frame grid")
        pos[0, lo:end] = next_pos
        pos[1, lo:end] = torch.arange(rows).repeat_interleave(width) + next_pos
        pos[2, lo:end] = torch.arange(width).repeat(rows) + next_pos
        next_pos += max(rows, width)
        cursor = end
    if videos and any(count == 0 for count in frames):
        raise ValueError("Video contains no complete frames")
    if any(bool((ids[cursor:] == token).any()) for token in visual_tokens):
        raise ValueError("Visual token outside an image/frame span")
    pos[:, cursor:] = torch.arange(next_pos, next_pos + ids.numel() - cursor)
    return pos


def _restore_image_newlines(ids, slots, recipe):
    if ids.shape[0] != 1:
        raise ValueError(f"{IMAGE_NEWLINE_SLOTS_KEY} describe a single sample")
    slots = torch.as_tensor(slots, dtype=torch.long)
    if slots.numel() == 0:
        return ids
    held = ids[0, slots] if int(slots.min()) >= 0 and int(slots.max()) < ids.shape[1] else None
    if held is None or not bool(
        ((held == recipe.image_token_id) | (held == recipe.video_token_id)).all()
    ):
        raise ValueError(f"{IMAGE_NEWLINE_SLOTS_KEY} do not match the sample's image placeholders")
    ids = ids.clone()
    ids[0, slots] = recipe.image_newline_token_id
    return ids


def build_position_ids(
    input_ids, recipe, *, multimodal_metadata=None, position_ids=None, doc_ids=None
):
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Position generation requires integer input_ids [B, L]")
    bsz, seq_len = input_ids.shape
    if recipe.provider == "provided":
        return validate_mrope_positions(position_ids, bsz, seq_len).to(input_ids.device)
    if multimodal_metadata and multimodal_metadata.get("multimodal_pruning"):
        raise ValueError("Hy4 token reconstruction does not support multimodal pruning")
    ids = input_ids.detach().cpu().long()
    slots = (multimodal_metadata or {}).get(IMAGE_NEWLINE_SLOTS_KEY)
    if slots is not None:
        ids = _restore_image_newlines(ids, slots, recipe)
    docs = doc_ids.detach().cpu() if doc_ids is not None else None
    if docs is not None and docs.shape != ids.shape:
        raise ValueError("doc_ids must match input_ids")
    out = torch.zeros((bsz, 3, seq_len), dtype=torch.long)
    for b in range(bsz):
        bounds = [0, seq_len]
        if docs is not None:
            changes = (docs[b, 1:] != docs[b, :-1]).nonzero(as_tuple=True)[0] + 1
            bounds = [0, *changes.tolist(), seq_len]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi > lo and (docs is None or int(docs[b, lo]) >= 0):
                out[b, :, lo:hi] = _hy4_document(ids[b, lo:hi], recipe)
    return out.to(input_ids.device)


def prepare_sample_positions(sample, recipe, *, multimodal_metadata=None):
    if recipe is None:
        return sample
    ids = sample.get("input_ids_cpu")
    if ids is None:
        ids = sample["input_ids"]
    if ids.numel() != sample["input_ids"].numel():
        raise ValueError("CPU and target token sequence lengths differ")
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    existing = sample.get("position_ids")
    if existing is not None and sample.get(POSITION_RECIPE_KEY) == recipe.identity:
        validate_mrope_positions(existing, ids.shape[0], ids.shape[-1])
        return sample
    sample["position_ids"] = build_position_ids(
        ids,
        recipe,
        multimodal_metadata=multimodal_metadata,
        position_ids=existing,
        doc_ids=sample.get("ctx_doc_ids"),
    )
    sample[POSITION_RECIPE_KEY] = recipe.identity
    return sample
