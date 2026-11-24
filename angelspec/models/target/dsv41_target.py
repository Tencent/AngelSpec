"""DeepSeek-V4.1 target reuse for DFly drafters: a target layer's KV latents and indexer picks,
recomputed on the trainer from the attention inputs vLLM captures. The arithmetic follows vLLM's
serving kernels op for op, fp8 records included, so the latents and picks are the ones served."""

import functools
import inspect
import json
import os
import weakref
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open

from angelspec.models.draft.dsv41_mla import apply_rope, rope_cos_sin, rope_inv_freq
from angelspec.models.target.block_quant import dequant_fp4_packed, dequant_fp8_blocks
from angelspec.models.target.target_indexer import (
    _text_config,
    document_starts,
    load_target_config,
)
from angelspec.utils.logging import logger

DSV41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")

#: KV record channels kept in e4m3 (the rest, the RoPE channels, stay bf16) and their group.
_RECORD_NOPE = 448
_RECORD_GROUP = 64
_FP8_MAX = 448.0

#: fp32 bytes of query-by-key scores one indexer chunk may hold.
_SCORE_BUDGET = 1 << 30


class _TargetReader:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        self._handles: Dict[str, object] = {}

    def tensor(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        if filename not in self._handles:
            self._handles[filename] = safe_open(
                os.path.join(self.model_path, filename), framework="pt", device=self.device
            )
        return self._handles[filename].get_tensor(key)

    def dense(self, prefix: str) -> torch.Tensor:
        weight = self.tensor(prefix + ".weight")
        if prefix + ".scale" not in self.weight_map:
            return weight
        scale = self.tensor(prefix + ".scale")
        if weight.dtype == torch.float8_e4m3fn:
            return dequant_fp8_blocks(weight, scale)
        if weight.dtype == torch.int8:
            return dequant_fp4_packed(weight, scale)
        raise TypeError(f"{prefix}: unexpected quantized dtype {weight.dtype}")


def is_dsv41_target(config: dict) -> bool:
    return any(
        str(c.get("model_type")) in DSV41_MODEL_TYPES for c in (config, _text_config(config))
    )


def kv_source_layer(text: dict, layer: int) -> int:
    sources = [int(s) for s in text["kv_source_layer_ids"] if int(s) <= layer]
    if not sources:
        raise ValueError(f"target layer {layer} precedes every KV source")
    return max(sources)


def index_source_layer(text: dict, layer: int) -> int:
    sources = [int(s) for s in text["index_source_layer_ids"] if int(s) <= layer]
    if not sources:
        raise ValueError(f"target layer {layer} precedes every index source")
    return max(sources)


def candidate_source_layer(text: dict, layer: int) -> Optional[int]:
    """The layer whose candidate blocks restrict index source ``layer``'s pick, or None."""
    source = int(text.get("candidate_source_layer_id", -1))
    return source if 0 <= source < layer else None


def check_dsv41_reuse_layer(config: dict, layer: int, field: str) -> None:
    """Raise unless target ``layer`` reads a 1:1 global KV, the only kind the reuse rebuilds."""
    text = _text_config(config)
    num_layers = int(text["num_hidden_layers"])
    if not 0 <= layer < num_layers:
        raise ValueError(f"{field} entry {layer} is outside the target's {num_layers} layers")
    ratio = int(text["compress_ratios"][layer])
    if ratio != 1:
        raise ValueError(
            f"{field} entry {layer}: target layer {layer} has compress ratio {ratio}; the reuse "
            "rebuilds the 1:1 global KV of the decoder layers only "
            f"({[i for i, r in enumerate(text['compress_ratios'][:num_layers]) if int(r) == 1]})"
        )
    if int(text["compress_ratios"][kv_source_layer(text, layer)]) != 1:
        raise ValueError(f"{field} entry {layer}: its KV source does not compress 1:1")


def dsv41_reuse_input_layers(
    config: dict, kv_layers: Sequence[int], index_layers: Sequence[int]
) -> List[int]:
    text = _text_config(config)
    layers = set()
    for layer in kv_layers:
        layers |= {int(layer), kv_source_layer(text, int(layer))}
    for layer in index_layers:
        source = index_source_layer(text, int(layer))
        layers |= {source, kv_source_layer(text, source)}
        candidate = candidate_source_layer(text, source)
        if candidate is not None:
            layers |= {candidate, kv_source_layer(text, candidate)}
    return sorted(layers)


def _target_rope(text: dict) -> Tuple[float, dict]:
    scaling = dict(text.get("rope_scaling") or text.get("rope_parameters") or {})
    rope_type = str(scaling.get("rope_type", scaling.get("type", "default")))
    if rope_type == "default":
        return float(text["compress_rope_theta"]), {}
    if scaling.get("attention_factor") == 1.0:
        raise NotImplementedError(
            "the target's compressed layers run deepseek_llama_scaling RoPE "
            "(attention_factor 1.0), which dsv41_mla does not implement"
        )
    return float(text["compress_rope_theta"]), dict(
        original_seq_len=int(scaling["original_max_position_embeddings"]),
        factor=float(scaling["factor"]),
        beta_fast=float(scaling.get("beta_fast", 32)),
        beta_slow=float(scaling.get("beta_slow", 1)),
    )


def check_dsv41_rope_compatible(config: dict, draft_config, field: str) -> None:
    text = _text_config(config)
    theta, yarn = _target_rope(text)
    pairs = {
        "attention_type": ("dsv41_mla", getattr(draft_config, "attention_type", None)),
        "head_dim": (int(text["head_dim"]), getattr(draft_config, "head_dim", None)),
        "qk_rope_head_dim": (int(text["qk_rope_head_dim"]), draft_config.qk_rope_head_dim),
        "rope_theta": (theta, float(draft_config.rope_theta)),
        "original_seq_len": (
            yarn.get("original_seq_len", 0),
            int(getattr(draft_config, "original_seq_len", 0) or 0),
        ),
    }
    if yarn:
        pairs["rope_factor"] = (yarn["factor"], float(getattr(draft_config, "rope_factor", 1.0)))
        pairs["beta_fast"] = (yarn["beta_fast"], float(getattr(draft_config, "beta_fast", 32)))
        pairs["beta_slow"] = (yarn["beta_slow"], float(getattr(draft_config, "beta_slow", 1)))
    mismatched = [
        f"{name}: target {target!r} vs drafter {draft!r}"
        for name, (target, draft) in pairs.items()
        if target != draft
    ]
    if mismatched:
        raise ValueError(
            f"{field} needs the drafter's latent widths and RoPE to be the target's: "
            + "; ".join(mismatched)
        )


def check_dsv41_reuse_compatible(config: dict, draft_config, field: str) -> None:
    check_dsv41_rope_compatible(config, draft_config, field)
    window = int(_text_config(config)["sliding_window"])
    windows = getattr(draft_config, "context_local_window", None)
    for w in windows if isinstance(windows, (list, tuple)) else [windows]:
        if w is not None and int(w) > window:
            raise ValueError(
                f"context_local_window {w} reaches past the {window} tokens the target's "
                "sliding-window cache keeps"
            )


def check_dsv41_attn_init_compatible(config: dict, draft_config) -> None:
    text = _text_config(config)
    names = [
        "hidden_size",
        "num_attention_heads",
        "head_dim",
        "qk_rope_head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "rms_norm_eps",
    ]
    mismatched = [
        f"{name}: target {text.get(name)!r} vs drafter {getattr(draft_config, name, None)!r}"
        for name in names
        if text.get(name) != getattr(draft_config, name, None)
    ]
    if getattr(draft_config, "attention_type", None) != "dsv41_mla":
        mismatched.insert(0, f"attention_type: drafter {draft_config.attention_type!r}")
    if mismatched:
        raise ValueError(
            "target_attn_init_layers copies the target's attention into the drafter, so these "
            "must match: " + "; ".join(mismatched)
        )


#: Each layer's mHC parameters, as the checkpoint names them under layers.<L>.
HC_PARAMS = (
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
)


def check_dsv41_hc_init_compatible(config: dict, draft_config, layers: List[int]) -> None:
    text = _text_config(config)
    mismatched = [
        f"{name}: target {text.get(name)!r} vs drafter {getattr(draft_config, name, None)!r}"
        for name in ("hidden_size", "hc_mult", "hc_sinkhorn_iters", "hc_eps")
        if text.get(name) != getattr(draft_config, name, None)
    ]
    if mismatched:
        raise ValueError(
            "target_hc_init_layers copies the target's mHC mixes into the drafter, so these "
            "must match: " + "; ".join(mismatched)
        )
    bad = [layer for layer in layers if not 0 <= layer < int(text["num_hidden_layers"])]
    if bad:
        raise ValueError(f"target_hc_init_layers {bad} are not target layers")


@torch.no_grad()
def init_hc_from_dsv41_target(draft_model: nn.Module, model_path: str, layers: List[int]) -> None:
    reader = _TargetReader(model_path)
    for draft_layer, source in zip(draft_model.layers, layers):
        for name in HC_PARAMS:
            theirs = reader.tensor(f"layers.{source}.{name}")
            mine = getattr(draft_layer, name)
            if mine.shape != theirs.shape:
                raise ValueError(
                    f"layers.{source}.{name}: target {tuple(theirs.shape)} vs drafter "
                    f"{tuple(mine.shape)}"
                )
            mine.copy_(theirs.to(mine.dtype))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16 RMSNorm as vLLM's fused kernels run it: fp32 throughout, one rounding at the end."""
    h = x.float()
    return (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * weight.float()).to(
        torch.bfloat16
    )


def kv_record(latent: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The bf16 latent vLLM's ``fp8_ds_mla`` KV record hands attention back: channels 0-447
    through e4m3 with a power-of-two scale per 64, channels 448-511 rotated."""
    nope = latent[..., :_RECORD_NOPE].float().unflatten(-1, (-1, _RECORD_GROUP))
    amax = nope.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(amax * (1.0 / _FP8_MAX)))
    fp8 = (nope * torch.exp2(-exponent)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    nope = (fp8.float() * torch.exp2(exponent)).flatten(-2).to(torch.bfloat16)
    return torch.cat([nope, apply_rope(latent[..., _RECORD_NOPE:], cos, sin)], dim=-1)


def index_key_fp8(keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """e4m3 index keys and fp32 per-key scales, as vLLM's indexer K cache holds them."""
    x = keys.float()
    absmax = x.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(absmax * (1.0 / _FP8_MAX)))
    fp8 = (x * torch.exp2(-exponent)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return fp8, torch.exp2(exponent).squeeze(-1)


def index_query_fp8(query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """e4m3 index queries and fp32 per-head scales, as ``fused_indexer_q_rope_quant`` makes."""
    amax = query.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / _FP8_MAX)))
    return (query / scale).to(torch.float8_e4m3fn), scale.squeeze(-1)


@functools.lru_cache(maxsize=1)
def _deep_gemm_mqa_logits():
    """vLLM's DeepGEMM ``fp8_fp4_mqa_logits``, or None. FP8 MMA accumulates below fp32,
    so only this kernel reproduces vLLM's scores, and so its picks, bit for bit."""
    try:
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, has_deep_gemm
    except Exception:
        kernel = None
    else:
        # Older builds take ``clean_logits`` there and return full-width rows.
        packed = "max_seqlen_k" in inspect.signature(fp8_fp4_mqa_logits).parameters
        kernel = fp8_fp4_mqa_logits if packed and has_deep_gemm() else None
    logger.info(
        "target reuse: index scores from "
        + ("vLLM's DeepGEMM kernel" if kernel is not None else "the torch fallback")
    )
    return kernel


def index_logits(
    query: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    key_scale: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """fp32 ``sum_h w_h * relu(scale * q_h . k)``, ``-inf`` outside ``[starts, ends)``."""
    kernel = _deep_gemm_mqa_logits() if query.is_cuda else None
    if kernel is not None and query.dtype == torch.float8_e4m3fn:
        spans = (ends - starts).to(torch.int32)
        logits = query.new_full(
            (query.shape[0], keys.shape[0]), float("-inf"), dtype=torch.float32
        )
        width = int(spans.max()) if spans.numel() else 0
        if width <= 0:
            return logits
        # Row i's column j is key starts[i] + j; columns past its span are not cleaned.
        packed = kernel(
            (query.contiguous(), None),
            (keys.contiguous(), key_scale.float().contiguous()),
            weights.float().contiguous(),
            starts.to(torch.int32).contiguous(),
            ends.to(torch.int32).contiguous(),
            width,
        )
        cols = torch.arange(width, device=query.device)
        inside = cols < spans.unsqueeze(-1)
        rows = torch.arange(query.shape[0], device=query.device).unsqueeze(-1).expand_as(inside)
        logits[rows[inside], (starts.unsqueeze(-1) + cols)[inside]] = packed[:, :width][inside]
        return logits
    rows, heads, dim = query.shape
    scores = _mm_fp32(query.to(torch.bfloat16).reshape(rows * heads, dim), keys.to(torch.bfloat16))
    scores = scores.view(rows, heads, -1)
    logits = ((scores * key_scale.float()).relu() * weights.float().unsqueeze(-1)).sum(1)
    cols = torch.arange(logits.shape[-1], device=logits.device)
    inside = (cols >= starts.unsqueeze(-1)) & (cols < ends.unsqueeze(-1))
    return logits.masked_fill(~inside, float("-inf"))


def select_candidate_blocks(
    logits: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    topk_blocks: int,
) -> torch.Tensor:
    """As vLLM's ``select_candidate_blocks``: blocks counted from each row's start, scored by
    their best key, the newest always kept."""
    rows, width = logits.shape
    n_blocks = -(-width // block_size)
    cols = torch.arange(width, device=logits.device)
    rel = cols - starts.unsqueeze(-1)
    valid = (rel >= 0) & (cols < ends.unsqueeze(-1))
    block = torch.where(valid, rel // block_size, n_blocks)
    scores = torch.full((rows, n_blocks + 1), float("-inf"), device=logits.device)
    scores.scatter_reduce_(
        1, block, logits.float().masked_fill(~valid, float("-inf")), reduce="amax"
    )
    scores = scores[:, :n_blocks]
    newest = ((ends - starts - 1).clamp(min=0) // block_size).unsqueeze(-1)
    pinned = torch.where((ends > starts).unsqueeze(-1), float("inf"), scores.gather(1, newest))
    scores.scatter_(1, newest, pinned)
    top = scores.topk(min(topk_blocks, n_blocks), dim=-1)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(
        1, top.indices, top.values > float("-inf")
    )


def _mm_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` accumulated and returned in fp32, as vLLM's ``torch.mm(..., out_dtype)``;
    ``out_dtype`` has no derivative, so a pass with a graph attached takes the fp32 matmul."""
    flat = x.reshape(-1, x.shape[-1])
    if flat.is_cuda and not (flat.requires_grad or weight.requires_grad):
        out = torch.mm(flat, weight.t(), out_dtype=torch.float32)
    else:
        out = flat.float() @ weight.float().t()
    return out.view(*x.shape[:-1], weight.shape[0])


class _LayerParts(nn.Module):
    def __init__(self, text: dict, layer: int, window: bool, query: bool, source: bool):
        super().__init__()
        hidden = int(text["hidden_size"])
        head_dim = int(text["head_dim"])
        q_lora = int(text["q_lora_rank"])
        index_dim = int(text["index_head_dim"])
        index_heads = int(text["index_n_heads"])
        self.layer = int(layer)
        # Parameter -> its name under "layers.<layer>.": a linear's prefix, or a norm's key.
        self.attn_norm = nn.Parameter(torch.ones(hidden))
        self.keys: Dict[str, str] = {"attn_norm": "attn_norm.weight"}
        if window:
            self.wkv = nn.Linear(hidden, head_dim, bias=False)
            self.kv_norm = nn.Parameter(torch.ones(head_dim))
            self.keys.update({"wkv.weight": "attn.wkv", "kv_norm": "attn.kv_norm.weight"})
        if query:
            self.wq_a = nn.Linear(hidden, q_lora, bias=False)
            self.q_norm = nn.Parameter(torch.ones(q_lora))
            self.index_wq_b = nn.Linear(q_lora, index_heads * index_dim, bias=False)
            self.weights_proj = nn.Linear(hidden, index_heads, bias=False)
            self.keys.update(
                {
                    "wq_a.weight": "attn.wq_a",
                    "q_norm": "attn.q_norm.weight",
                    "index_wq_b.weight": "attn.indexer.wq_b",
                    "weights_proj.weight": "attn.indexer.weights_proj",
                }
            )
        if source:
            self.compressor_wkv = nn.Linear(hidden, head_dim, bias=False)
            self.compressor_norm = nn.Parameter(torch.ones(head_dim))
            self.index_wk = nn.Linear(head_dim, index_dim, bias=False)
            self.k_norm = nn.Parameter(torch.ones(index_dim))
            self.keys.update(
                {
                    "compressor_wkv.weight": "attn.compressor.wkv",
                    "compressor_norm": "attn.compressor.norm.weight",
                    "index_wk.weight": "attn.indexer.wk",
                    "k_norm": "attn.indexer.k_norm.weight",
                }
            )

    def load(self, reader) -> None:
        params = dict(self.named_parameters())
        for name, key in self.keys.items():
            full = f"layers.{self.layer}.{key}"
            tensor = reader.tensor(full) if key.endswith(".weight") else reader.dense(full)
            if tuple(tensor.shape) != tuple(params[name].shape):
                raise ValueError(
                    f"'{full}' has shape {tuple(tensor.shape)}, expected {tuple(params[name].shape)}"
                )
            params[name].data.copy_(tensor)


class DSV41TargetReuse(nn.Module):
    """Frozen target tensors behind ``DSV41TargetKVLatent`` and ``DSV41TargetIndexer``."""

    def __init__(
        self,
        config: dict,
        kv_layers: Sequence[int],
        index_layers: Sequence[int],
        topk: Optional[int] = None,
        quantize: bool = True,
    ):
        super().__init__()
        if not is_dsv41_target(config):
            raise ValueError("DSV41TargetReuse needs a DeepSeek-V4.1 target config")
        text = _text_config(config)
        self.kv_layers = sorted({int(layer) for layer in kv_layers})
        self.index_layers = sorted({int(layer) for layer in index_layers})
        for layer in self.kv_layers:
            check_dsv41_reuse_layer(config, layer, "target_kv_layers")
        for layer in self.index_layers:
            check_dsv41_reuse_layer(config, layer, "target_index_layer")
        if self.index_layers and (isinstance(topk, bool) or not isinstance(topk, int) or topk < 1):
            raise ValueError(f"topk must be a positive int, got {topk!r}")
        self.topk = topk
        # vLLM's caches hold the latents and index keys in fp8; False keeps them bf16.
        self.quantize = bool(quantize)
        self.head_dim = int(text["head_dim"])
        self.rope_dim = int(text["qk_rope_head_dim"])
        self.index_heads = int(text["index_n_heads"])
        self.index_dim = int(text["index_head_dim"])
        self.eps = float(text["rms_norm_eps"])
        self.window = int(text["sliding_window"])
        self.candidate_blocks = int(text["candidate_topk_blocks"])
        self.candidate_block_size = int(text["candidate_block_size"])
        self.rope_theta, self.rope_yarn = _target_rope(text)

        self.kv_source = {layer: kv_source_layer(text, layer) for layer in self.kv_layers}
        self.index_source = {layer: index_source_layer(text, layer) for layer in self.index_layers}
        self.candidate_source = {
            source: candidate_source_layer(text, source) for source in self.index_source.values()
        }
        needs: Dict[int, Dict[str, bool]] = {}

        def need(layer: int, part: str) -> None:
            needs.setdefault(layer, {"window": False, "query": False, "source": False})[part] = (
                True
            )

        for layer, source in self.kv_source.items():
            need(layer, "window")
            need(source, "source")
        for source, candidate in self.candidate_source.items():
            need(source, "query")
            need(kv_source_layer(text, source), "source")
            if candidate is not None:
                need(candidate, "query")
                need(kv_source_layer(text, candidate), "source")
        self.parts = nn.ModuleDict(
            {
                str(layer): _LayerParts(text, layer, **flags)
                for layer, flags in sorted(needs.items())
            }
        )
        self._text_kv_source = {layer: kv_source_layer(text, layer) for layer in needs}
        self.requires_grad_(False)
        self.eval()
        self._memo: Dict[str, tuple] = {}
        self._checked_inputs: set = set()

    @classmethod
    def from_config(
        cls, model_path: str, kv_layers, index_layers, topk=None, device="cuda"
    ) -> "DSV41TargetReuse":
        """Uninitialized weights, e.g. on ranks that receive them by broadcast."""
        module = cls(load_target_config(model_path), kv_layers, index_layers, topk)
        return module.to(device=device, dtype=torch.bfloat16)

    @classmethod
    def from_pretrained(
        cls, model_path: str, kv_layers, index_layers, topk=None, device="cuda"
    ) -> "DSV41TargetReuse":
        module = cls(load_target_config(model_path), kv_layers, index_layers, topk)
        module.load_weights(model_path)
        return module.to(device=device, dtype=torch.bfloat16)

    def load_weights(self, model_path: str) -> None:
        reader = _TargetReader(model_path)
        for part in self.parts.values():
            part.load(reader)

    def kv_modules(self) -> List["DSV41TargetKVLatent"]:
        return [DSV41TargetKVLatent(self, layer) for layer in self.kv_layers]

    def indexers(self) -> List["DSV41TargetIndexer"]:
        return [DSV41TargetIndexer(self, layer) for layer in self.index_layers]

    def kv_inputs(self, layer: int) -> Tuple[int, ...]:
        return tuple(dict.fromkeys((layer, self.kv_source[layer])))

    def index_inputs(self, layer: int) -> Tuple[int, ...]:
        source = self.index_source[layer]
        candidate = self.candidate_source[source]
        layers = [source, self._text_kv_source[source]]
        if candidate is not None:
            layers += [candidate, self._text_kv_source[candidate]]
        return tuple(dict.fromkeys(layers))

    def draft_mismatch(self, attn: nn.Module) -> Optional[str]:
        """Why ``attn`` cannot read these latents as its own context, else None."""
        widths = (getattr(attn, "head_dim", None), getattr(attn, "rope_dim", None))
        if widths != (self.head_dim, self.rope_dim):
            return f"(head_dim, qk_rope_head_dim) {(self.head_dim, self.rope_dim)} vs the drafter's {widths}"
        mine = rope_inv_freq(self.rope_dim, self.rope_theta, **self.rope_yarn)
        theirs = rope_inv_freq(attn.rope_dim, attn.rope_theta, **attn.rope_yarn)
        if not torch.equal(mine, theirs):
            return "the RoPE frequencies (rope_theta / YaRN) differ"
        return None

    def _check_inputs(self, hidden: Dict[int, torch.Tensor]) -> None:
        """Raise unless each captured hidden is its layer's ``attn_norm`` output. vLLM's plain
        capture, the mean of the mHC streams, would give silently wrong latents and picks."""
        for layer, x in hidden.items():
            if layer in self._checked_inputs:
                continue
            weight = self.parts[str(layer)].attn_norm.float()
            tokens = x.reshape(-1, x.shape[-1])
            picked = torch.linspace(
                0, tokens.shape[0] - 1, min(256, tokens.shape[0]), device=x.device
            )
            sample = tokens[picked.long()].float()
            sample = sample[sample.abs().amax(-1) > 0]
            nonzero = weight != 0
            rms = (sample[:, nonzero] / weight[nonzero]).square().mean(-1).sqrt()
            if rms.numel() and float(((rms - 1).abs() < 0.05).float().mean()) < 0.9:
                raise RuntimeError(
                    f"the captured hidden for target layer {layer} is not its attention input "
                    f"(median RMS of hidden / attn_norm.weight {rms.median().item():.3f}, not 1): "
                    "vLLM has to run AngelSpec's apply_dsv41_attention_input_capture for "
                    f"target_layer_id {layer - 1}"
                )
            self._checked_inputs.add(layer)

    def _cached(self, tag: str, tensors: Sequence[Optional[torch.Tensor]], compute):
        """``compute()``, reused while ``tensors`` are the same objects at the same versions; held
        weakly, so a batch's tensors are never mistaken for a later one's."""
        key = tuple((None, None) if t is None else (weakref.ref(t), t._version) for t in tensors)
        entry = self._memo.get(tag)
        if entry is not None and len(entry[0]) == len(key):
            hit = all(
                (old[0] is None and t is None)
                or (
                    old[0] is not None and t is not None and old[0]() is t and old[1] == t._version
                )
                for old, t in zip(entry[0], tensors)
            )
            if hit:
                return entry[1]
        value = compute()
        self._memo[tag] = (key, value)
        return value

    def _positions(
        self, bsz: int, seq_len: int, device: torch.device, ctx_doc_ids: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        starts = document_starts(ctx_doc_ids, bsz, seq_len, device)
        return torch.arange(seq_len, device=device).expand(bsz, -1) - starts, starts

    def _rope(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return rope_cos_sin(positions, self.rope_dim, self.rope_theta, **self.rope_yarn)

    def _record(self, latent: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return kv_record(latent, cos, sin) if self.quantize else apply_rope(latent, cos, sin)

    def window_latents(
        self, layer: int, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        part = self.parts[str(layer)]
        return self._record(rms_norm(part.wkv(hidden), part.kv_norm, self.eps), cos, sin)

    def _source(self, layer: int, hidden: torch.Tensor, ctx_doc_ids: Optional[torch.Tensor]):
        def compute():
            part = self.parts[str(layer)]
            bsz, seq_len, _ = hidden.shape
            positions, _ = self._positions(bsz, seq_len, hidden.device, ctx_doc_ids)
            cos, sin = self._rope(positions)
            latent = rms_norm(
                _mm_fp32(hidden, part.compressor_wkv.weight), part.compressor_norm, self.eps
            )
            keys = apply_rope(rms_norm(part.index_wk(latent), part.k_norm, self.eps), cos, sin)
            if self.quantize:
                keys, scale = index_key_fp8(keys)
            else:
                scale = torch.ones(keys.shape[:-1], dtype=torch.float32, device=keys.device)
            return self._record(latent, cos, sin), keys, scale

        return self._cached(f"source{layer}", (hidden, ctx_doc_ids), compute)

    @torch.no_grad()
    def context_latent(
        self, layer: int, hidden: Dict[int, torch.Tensor], ctx_doc_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """``[B, 2 * L, 512]``: layer ``layer``'s window latents, then its KV source's global ones."""
        self._check_inputs(hidden)
        source = self.kv_source[layer]
        global_latents, _, _ = self._source(source, hidden[source], ctx_doc_ids)
        x = hidden[layer]
        positions, _ = self._positions(x.shape[0], x.shape[1], x.device, ctx_doc_ids)
        cos, sin = self._rope(positions)
        window = self.window_latents(layer, x, cos, sin)
        return torch.cat([window, global_latents.to(window.dtype)], dim=1)

    def _queries(
        self, layer: int, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``[N, H, D]`` e4m3 index queries and ``[N, H]`` head weights, the query scale folded in."""
        part = self.parts[str(layer)]
        q_resid = rms_norm(part.wq_a(hidden), part.q_norm, self.eps)
        query = part.index_wq_b(q_resid).unflatten(-1, (self.index_heads, self.index_dim))
        query = apply_rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2)).float()
        if self.quantize:
            query, scale = index_query_fp8(query)
        else:
            scale = torch.ones(query.shape[:-1], dtype=torch.float32, device=query.device)
        # vLLM's kernel multiplies by the two constant scales folded into one.
        weights = (
            part.weights_proj(hidden).float()
            * scale
            * (self.index_dim**-0.5 * self.index_heads**-0.5)
        )
        return query, weights

    def _row_chunk(self, seq_len: int) -> int:
        return max(1, _SCORE_BUDGET // (self.index_heads * max(seq_len, 1) * 4 * 2))

    def _candidate_blocks(
        self,
        layer: int,
        hidden: Dict[int, torch.Tensor],
        anchors: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Blocks candidate source ``layer`` keeps for ``anchor - 1``, from its document start."""

        def compute():
            x = hidden[layer]
            bsz, seq_len, _ = x.shape
            device = x.device
            source = self._text_kv_source[layer]
            _, keys, key_scale = self._source(source, hidden[source], ctx_doc_ids)
            positions, starts = self._positions(bsz, seq_len, device, ctx_doc_ids)
            cos, sin = self._rope(positions)
            size = self.candidate_block_size
            keep = torch.ones(
                bsz, anchors.shape[1], -(-seq_len // size), dtype=torch.bool, device=device
            )
            chunk = self._row_chunk(seq_len)
            for b in range(bsz):
                rows = (anchors[b] - 1).clamp(0, seq_len - 1)
                doc_start = starts[b].gather(0, anchors[b])
                for lo in range(0, rows.shape[0], chunk):
                    sl = slice(lo, lo + chunk)
                    if not bool(
                        (anchors[b, sl] - doc_start[sl] > self.candidate_blocks * size).any()
                    ):
                        continue
                    query, weights = self._queries(
                        layer, x[b, rows[sl]], cos[b, rows[sl]], sin[b, rows[sl]]
                    )
                    logits = index_logits(
                        query, weights, keys[b], key_scale[b], doc_start[sl], anchors[b, sl]
                    )
                    keep[b, sl] = select_candidate_blocks(
                        logits, doc_start[sl], anchors[b, sl], size, self.candidate_blocks
                    )
            return keep

        return self._cached(f"candidates{layer}", (hidden[layer], anchors, ctx_doc_ids), compute)

    @torch.no_grad()
    def candidates(
        self,
        layer: int,
        hidden: Dict[int, torch.Tensor],
        anchor_positions: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Indexer picks for ``anchor - 1`` within ``[doc_start, anchor)``, ``-1`` padded."""
        self._check_inputs(hidden)
        source = self.index_source[layer]
        x = hidden[source]
        bsz, seq_len, _ = x.shape
        device = x.device
        anchors = anchor_positions.to(device=device, dtype=torch.long)
        _, keys, key_scale = self._source(
            self._text_kv_source[source], hidden[self._text_kv_source[source]], ctx_doc_ids
        )
        candidate = self.candidate_source[source]
        keep = (
            None
            if candidate is None
            else self._candidate_blocks(candidate, hidden, anchors, ctx_doc_ids)
        )
        positions, starts = self._positions(bsz, seq_len, device, ctx_doc_ids)
        cos, sin = self._rope(positions)
        width = min(self.topk, seq_len)
        out = torch.full((bsz, anchors.shape[1], self.topk), -1, dtype=torch.long, device=device)
        arange = torch.arange(seq_len, device=device)
        chunk = self._row_chunk(seq_len)
        size = self.candidate_block_size
        for b in range(bsz):
            rows = (anchors[b] - 1).clamp(0, seq_len - 1)
            doc_start = starts[b].gather(0, anchors[b])
            for lo in range(0, rows.shape[0], chunk):
                sl = slice(lo, lo + chunk)
                query, weights = self._queries(
                    source, x[b, rows[sl]], cos[b, rows[sl]], sin[b, rows[sl]]
                )
                logits = index_logits(
                    query, weights, keys[b], key_scale[b], doc_start[sl], anchors[b, sl]
                )
                if keep is not None:
                    rel = (arange - doc_start[sl, None]).clamp(min=0) // size
                    in_block = keep[b, sl].gather(1, rel.clamp(max=keep.shape[-1] - 1))
                    logits = logits.masked_fill(~in_block, float("-inf"))
                top = logits.topk(width, dim=-1)
                out[b, sl, :width] = torch.where(top.values > float("-inf"), top.indices, -1)
        return out


class DSV41TargetKVLatent:
    """``target_kv_layers`` context for dsv41_mla: window latents, then global ones."""

    context_layout = "window_global"

    def __init__(self, reuse: DSV41TargetReuse, layer: int):
        self.reuse = reuse
        self.layer = int(layer)
        self.input_layers = reuse.kv_inputs(self.layer)
        self.head_dim = reuse.head_dim
        self.qk_rope_head_dim = reuse.rope_dim
        # Where the latent's RoPE channels start, as TargetKVLatent's kv_lora_rank marks it.
        self.kv_lora_rank = reuse.head_dim - reuse.rope_dim

    def latents(
        self,
        hidden: Sequence[torch.Tensor],
        input_ids: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        sp_group=None,
    ) -> torch.Tensor:
        """``hidden``: the captured attention inputs of ``input_layers``, in that order."""
        if sp_group is not None:
            raise NotImplementedError("DeepSeek-V4.1 target KV reuse does not support USP")
        return self.reuse.context_latent(
            self.layer, dict(zip(self.input_layers, hidden)), ctx_doc_ids
        )

    def mismatch(self, attn: nn.Module) -> Optional[str]:
        return self.reuse.draft_mismatch(attn)


class DSV41TargetIndexer:
    """``target_index_layer``'s pick per draft block, in TargetIndexer's place."""

    def __init__(self, reuse: DSV41TargetReuse, layer: int):
        self.reuse = reuse
        self.layer = int(layer)
        self.topk = reuse.topk
        self.input_layers = reuse.index_inputs(self.layer)

    def candidate_indices(
        self,
        hidden: Sequence[torch.Tensor],
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``hidden``: the captured attention inputs of ``input_layers``, in that order."""
        return self.reuse.candidates(
            self.layer, dict(zip(self.input_layers, hidden)), anchor_positions, ctx_doc_ids
        )


@torch.no_grad()
def init_attention_from_dsv41_target(
    draft_model: nn.Module, model_path: str, layers: List[int]
) -> None:
    """Copy target layer ``layers[i]``'s ``attn_norm`` and attention into draft layer ``i``."""
    if len(layers) != len(draft_model.layers):
        raise ValueError(f"{len(layers)} target layers for {len(draft_model.layers)} draft layers")
    reader = _TargetReader(model_path)
    for draft_layer, layer in zip(draft_model.layers, layers):
        attn = draft_layer.self_attn
        prefix = f"layers.{layer}."
        copies = [
            (draft_layer.input_layernorm.weight, reader.tensor(prefix + "attn_norm.weight")),
            (attn.attn_sink, reader.tensor(prefix + "attn.attn_sink")),
            (attn.q_norm.weight, reader.tensor(prefix + "attn.q_norm.weight")),
            (attn.kv_norm.weight, reader.tensor(prefix + "attn.kv_norm.weight")),
        ] + [
            (getattr(attn, name).weight, reader.dense(prefix + f"attn.{name}"))
            for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
        ]
        for param, tensor in copies:
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(
                    f"target layer {layer}: a tensor of shape {tuple(tensor.shape)} for a draft "
                    f"parameter of shape {tuple(param.shape)}"
                )
            param.copy_(tensor)
