"""GLM-5.3-Flash's DSA layers recomputed for DFly drafters: the MLA latent, its FP8 weights
dequantized to bf16 as vLLM does, and the kpool indexer's picks, on vLLM's kernels under CUDA.
Pools restart at each packed document, each being its own vLLM request."""

import functools
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.target.dsv41_target import index_logits, rms_norm
from angelspec.models.target.target_indexer import (
    _text_config,
    document_starts,
    load_target_config,
)

logger = logging.getLogger(__name__)

GLM53_TEXT_TYPE = "glm5_next_text"
REUSE_LAYER_TYPE = "deepseek_sparse_attention"
_FP8_MAX = 448.0


def is_glm53_target(config: dict) -> bool:
    return _text_config(config).get("model_type") == GLM53_TEXT_TYPE


def _reuse_layers(text: dict) -> List[int]:
    return [i for i, kind in enumerate(text["layer_types"]) if kind == REUSE_LAYER_TYPE]


def check_glm53_reuse_layer(config: dict, layer: int, field: str) -> None:
    text = _text_config(config)
    num_layers = int(text["num_hidden_layers"])
    if not 0 <= layer < num_layers:
        raise ValueError(f"{field} entry {layer} is outside the target's {num_layers} layers")
    kind = text["layer_types"][layer]
    if kind != REUSE_LAYER_TYPE:
        raise ValueError(
            f"{field}: target layer {layer} is {kind!r}; GLM-5.3 holds an MLA latent and an "
            f"indexer on its {REUSE_LAYER_TYPE} layers {_reuse_layers(text)} only"
        )
    if int(text.get("index_kpool") or 1) < 2:
        raise ValueError(f"{field}: this GLM-5.3 config has no kpool indexer (index_kpool)")


def check_glm53_index_topk(config: dict, topk: int, field: str) -> None:
    """Raise unless ``topk`` is whole pools; the indexer reruns for any multiple of ``index_kpool``."""
    kpool = int(_text_config(config)["index_kpool"])
    if int(topk) % kpool:
        raise ValueError(
            f"{field}: GLM-5.3's indexer keeps whole pools of {kpool} tokens, so "
            f"target_index_topk must be a multiple of {kpool}; got {topk}"
        )


def check_glm53_reuse_compatible(config: dict, draft_config, field: str) -> None:
    text = _text_config(config)
    pairs = {
        "attention_type": ("glm53_mla", getattr(draft_config, "attention_type", None)),
        "kv_lora_rank": (int(text["kv_lora_rank"]), draft_config.kv_lora_rank),
        "qk_rope_head_dim": (int(text["qk_rope_head_dim"]), draft_config.qk_rope_head_dim),
    }
    if field == "target_index_layer":
        check_glm53_index_topk(config, draft_config.target_index_topk, field)
    mismatched = [
        f"{name}: target {target!r} vs drafter {draft!r}"
        for name, (target, draft) in pairs.items()
        if target != draft
    ]
    if mismatched:
        raise ValueError(
            f"{field} reads GLM-5.3's latents and indexer picks as they are, so these must "
            "match: " + "; ".join(mismatched)
        )


def _ue8m0(scale: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(scale)))


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """fp32 Hadamard-128 rotation over the last dim, butterflies in vLLM's kernel order."""
    flat = x.float().reshape(-1, 128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = flat.view(flat.shape[0], -1, 2, stride)
        a, b = pairs[:, :, 0], pairs[:, :, 1]
        flat = torch.stack([a + b, a - b], dim=2).reshape(flat.shape[0], 128)
    return (flat * 0.08838834764831845).view_as(x)


def _absmax_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax e4m3 with a power-of-two scale, as the kpool kernels quantize."""
    absmax = x.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = _ue8m0(absmax * (1.0 / _FP8_MAX))
    return (x / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn), scale


@functools.lru_cache(maxsize=1)
def _kpool_kernels():
    try:
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
        from vllm.models.glm5next.nvidia.ops import kpool_compress

        kernels = (
            kpool_compress.fwht128_quant_fp8,
            kpool_compress.kpool_compress_and_write_cache,
            kpool_compress.expand_pools_and_append_tail,
            torch.ops._C.top_k_per_row_prefill,
        )
    except Exception:
        kernels = None
    logger.info(
        "GLM-5.3 target reuse: indexer pools, query quantization, top-k and expansion on "
        + ("vLLM's kernels" if kernels is not None else "the torch equivalents")
    )
    return kernels


def quantize_queries(query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """As vLLM's ``fwht128_quant_fp8``: Hadamard-rotated e4m3 rows and ``[N, 1]`` scales."""
    kernels = _kpool_kernels() if query.is_cuda else None
    if kernels is not None:
        return kernels[0](query.to(torch.bfloat16).contiguous())
    rotated = hadamard128(query.to(torch.bfloat16)).to(torch.bfloat16).float()
    return _absmax_fp8(rotated)


def pool_keys(
    keys: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """As vLLM's ``kpool_compress_and_write_cache``: e4m3 pooled keys and fp32 scales."""
    kernels = _kpool_kernels() if keys.is_cuda else None
    if kernels is not None:
        cache = torch.empty(1, 64, 132, dtype=torch.uint8, device=keys.device)
        return kernels[1](
            cache,
            keys.to(torch.bfloat16).contiguous(),
            gate.to(torch.bfloat16).contiguous(),
            ape.float().contiguous(),
            torch.zeros(keys.shape[0], dtype=torch.int64, device=keys.device),
            pool_size=keys.shape[1],
            head_dim=keys.shape[2],
            round_scale=True,
            return_compressed=True,
            write_cache=False,
        )
    weights = torch.softmax(gate.float() + ape.float(), dim=1)
    pooled = (weights * keys.float()).sum(1).to(torch.bfloat16).float()
    rotated = hadamard128(pooled).to(torch.bfloat16).float()
    quantized, scale = _absmax_fp8(rotated)
    return quantized, scale.squeeze(-1)


def top_pools(
    logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, k: int
) -> torch.Tensor:
    """As vLLM's ``top_k_per_row_prefill``: best pools, offsets from ``starts``, ``-1`` padded."""
    rows = logits.shape[0]
    kernels = _kpool_kernels() if logits.is_cuda else None
    if kernels is not None:
        out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
        if rows:
            kernels[3](
                logits.contiguous(),
                starts.to(torch.int32).contiguous(),
                ends.to(torch.int32).contiguous(),
                out,
                rows,
                logits.contiguous().stride(0),
                logits.contiguous().stride(1),
                k,
            )
        return out.long()
    cols = torch.arange(logits.shape[1], device=logits.device)
    inside = (cols >= starts.unsqueeze(-1)) & (cols < ends.unsqueeze(-1))
    masked = logits.masked_fill(~inside, float("-inf"))
    width = min(k, logits.shape[1])
    values, picks = masked.topk(width, dim=-1)
    picks = torch.where(torch.isfinite(values), picks - starts.unsqueeze(-1), -1)
    return F.pad(picks, (0, k - width), value=-1)


def expand_pools(pool_ids: torch.Tensor, visible: torch.Tensor, kpool: int) -> torch.Tensor:
    """As vLLM's ``expand_pools_and_append_tail``: the pools' tokens, then the incomplete pool."""
    kernels = _kpool_kernels() if pool_ids.is_cuda else None
    if kernels is not None:
        return kernels[2](
            pool_ids.contiguous(), visible.to(torch.int32).contiguous(), kpool
        ).long()
    rows, pools = pool_ids.shape
    cols = torch.arange(pools * kpool + kpool - 1, device=pool_ids.device)
    tail_start = (visible // kpool * kpool).unsqueeze(-1)
    history_len = tail_start.clamp(max=pools * kpool)
    pool = pool_ids.gather(1, (cols // kpool).clamp(max=pools - 1).expand(rows, -1))
    history = torch.where(pool >= 0, pool * kpool + cols % kpool, -1)
    tail = tail_start + cols - history_len
    tail = torch.where((cols >= history_len) & (tail < visible.unsqueeze(-1)), tail, -1)
    return torch.where(cols < history_len, history, tail)


def _document_spans(doc_ids: Optional[torch.Tensor], seq_len: int) -> List[Tuple[int, int]]:
    if doc_ids is None:
        return [(0, seq_len)]
    ids = doc_ids.tolist()
    spans, start = [], 0
    for pos in range(1, seq_len + 1):
        if pos == seq_len or ids[pos] != ids[start]:
            if ids[start] >= 0:
                spans.append((start, pos))
            start = pos
    return spans


class _LayerParts(nn.Module):
    """Target layer ``layer``'s tensors the reuse reads, in the form vLLM computes with."""

    def __init__(self, text: dict, layer: int, kv: bool, index: bool):
        super().__init__()
        self.layer = int(layer)
        hidden, q_lora = int(text["hidden_size"]), int(text["q_lora_rank"])
        kv_lora = int(text["kv_lora_rank"])

        def param(*shape, dtype=torch.bfloat16):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.input_norm = param(hidden)
        self.has_kv, self.has_index = kv, index
        if kv:
            self.kv_a = param(kv_lora, hidden)
            self.kv_norm = param(kv_lora)
        if index:
            heads, dim = int(text["index_n_heads"]), int(text["index_head_dim"])
            self.q_a = param(q_lora, hidden)
            self.q_norm = param(q_lora)
            self.wq_b = param(heads * dim, q_lora)
            # vLLM's fused [wk; weights_proj] GEMM, the keys its first ``dim`` columns.
            self.wk_weights_proj = param(dim + heads, hidden)
            self.k_norm_weight = param(dim)
            self.k_norm_bias = param(dim)
            self.gate = param(dim, hidden)
            self.ape = param(int(text["index_kpool"]), dim, dtype=torch.float32)

    def load(self, checkpoint, prefix: str) -> None:
        from angelspec.models.target.target_mlp_init import _copy

        def put(param: torch.Tensor, key: str, transform=None) -> None:
            tensor = checkpoint.weight(prefix + key)
            _copy(param, transform(tensor) if transform else tensor.to(param.dtype), prefix + key)

        put(self.input_norm, "input_layernorm.weight")
        attn = "self_attn."
        if self.has_kv:
            put(self.kv_a, attn + "kv_a_proj_with_mqa.weight")
            put(self.kv_norm, attn + "kv_a_layernorm.weight")
        if self.has_index:
            put(self.q_a, attn + "q_a_proj.weight")
            put(self.q_norm, attn + "q_a_layernorm.weight")
            put(self.wq_b, attn + "indexer.wq_b.weight")
            wk = checkpoint.weight(prefix + attn + "indexer.wk.weight")
            weights = checkpoint.weight(prefix + attn + "indexer.weights_proj.weight")
            _copy(
                self.wk_weights_proj,
                torch.cat([wk, weights]).to(torch.bfloat16),
                prefix + attn + "indexer.{wk,weights_proj}.weight",
            )
            put(self.k_norm_weight, attn + "indexer.k_norm.weight")
            put(self.k_norm_bias, attn + "indexer.k_norm.bias")
            put(self.gate, attn + "indexer.index_kpool_compress_gate")
            put(self.ape, attn + "indexer.index_kpool_compress_ape")


class Glm53TargetReuse(nn.Module):
    """GLM-5.3-Flash's DSA layers ``kv_layers`` (latents) and ``index_layers`` (picks)."""

    def __init__(
        self, config: dict, kv_layers: List[int], index_layers: List[int], topk: Optional[int]
    ):
        super().__init__()
        if not is_glm53_target(config):
            raise ValueError("Glm53TargetReuse needs a GLM-5.3-Flash target (glm5_next_text)")
        text = _text_config(config)
        self.kv_layers = list(dict.fromkeys(int(layer) for layer in kv_layers))
        self.index_layers = list(dict.fromkeys(int(layer) for layer in index_layers))
        for layer in self.kv_layers:
            check_glm53_reuse_layer(config, layer, "target_kv_layers")
        for layer in self.index_layers:
            check_glm53_reuse_layer(config, layer, "target_index_layer")
        if self.index_layers:
            check_glm53_index_topk(config, topk, "target_index_topk")
        self.topk = None if topk is None else int(topk)
        self.kv_lora_rank = int(text["kv_lora_rank"])
        self.eps = float(text["rms_norm_eps"])
        self.kpool = int(text["index_kpool"])
        self.index_heads = int(text.get("index_n_heads") or 0)
        self.index_dim = int(text.get("index_head_dim") or 0)
        self.parts = nn.ModuleDict(
            {
                str(layer): _LayerParts(
                    text, layer, kv=layer in self.kv_layers, index=layer in self.index_layers
                )
                for layer in dict.fromkeys(self.kv_layers + self.index_layers)
            }
        )
        self._checked: set = set()
        self.requires_grad_(False)

    @classmethod
    def from_config(cls, model_path: str, kv_layers, index_layers, topk, device="cuda"):
        """Uninitialized weights, e.g. on ranks that receive them by broadcast."""
        return cls(load_target_config(model_path), kv_layers, index_layers, topk).to(device)

    @classmethod
    def from_pretrained(cls, model_path: str, kv_layers, index_layers, topk, device="cuda"):
        module = cls(load_target_config(model_path), kv_layers, index_layers, topk)
        module.load_weights(model_path)
        return module.to(device)

    @torch.no_grad()
    def load_weights(self, model_path: str) -> None:
        from angelspec.models.target.target_mlp_init import _Checkpoint

        checkpoint = _Checkpoint(model_path)
        for layer, part in self.parts.items():
            anchor = f"layers.{layer}.self_attn.kv_b_proj.weight"
            keys = [key for key in checkpoint.weight_map if key.endswith(anchor)]
            if len(keys) != 1:
                raise KeyError(f"expected exactly one '*{anchor}' in the checkpoint, found {keys}")
            part.load(checkpoint, keys[0][: -len("self_attn.kv_b_proj.weight")])

    def kv_modules(self) -> List["Glm53TargetKVLatent"]:
        return [Glm53TargetKVLatent(self, layer) for layer in self.kv_layers]

    def indexers(self) -> List["Glm53TargetIndexer"]:
        return [Glm53TargetIndexer(self, layer) for layer in self.index_layers]

    def _check_input(self, layer: int, hidden: torch.Tensor) -> None:
        if layer in self._checked:
            return
        norm = self.parts[str(layer)].input_norm.float()
        rms = (hidden.float() / norm).square().mean(-1).sqrt()
        rms = rms[rms > 0]
        median = float(rms.median()) if rms.numel() else 1.0
        if not 0.5 < median < 2.0:
            raise RuntimeError(
                f"the captured hidden for target layer {layer} is not its attention input "
                f"(median RMS of hidden / input_layernorm.weight {median:.3f}, not 1): vLLM "
                "has to run AngelSpec's apply_glm53_attention_input_capture for "
                f"target_layer_id {layer - 1}"
            )
        self._checked.add(layer)

    @torch.no_grad()
    def latent(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, L, kv_lora_rank]`` bf16 latents of target layer ``layer``, as its MLA caches them."""
        self._check_input(layer, hidden)
        part = self.parts[str(layer)]
        return rms_norm(F.linear(hidden.to(torch.bfloat16), part.kv_a), part.kv_norm, self.eps)

    def _queries(self, part: _LayerParts, x: torch.Tensor):
        heads, dim = self.index_heads, self.index_dim
        q_lora = rms_norm(F.linear(x, part.q_a), part.q_norm, self.eps)
        query = F.linear(q_lora, part.wq_b).view(-1, dim)
        q8, q_scale = quantize_queries(query)
        weights = torch.mm(x.float(), part.wk_weights_proj[dim:].float().t())
        weights = (
            weights.unsqueeze(-1) * q_scale.view(-1, heads, 1) * (dim**-0.5 * heads**-0.5)
        ).squeeze(-1)
        return q8.view(-1, heads, dim), weights

    def _pooled_keys(self, part: _LayerParts, x: torch.Tensor, spans: List[Tuple[int, int]]):
        """Each document's complete pools in one row: e4m3 keys, scales, each document's first pool."""
        dim, kpool = self.index_dim, self.kpool
        keys = F.linear(x, part.wk_weights_proj)[:, :dim]
        keys = F.layer_norm(
            keys.float(), (dim,), part.k_norm_weight.float(), part.k_norm_bias.float(), 1e-6
        ).to(torch.bfloat16)
        gate = F.linear(x, part.gate)
        starts, first_pool = [], {}
        for start, end in spans:
            first_pool[start] = sum(s.numel() for s in starts)
            starts.append(
                torch.arange(start, start + (end - start) // kpool * kpool, kpool, device=x.device)
            )
        pool_starts = (
            torch.cat(starts) if starts else torch.zeros(0, dtype=torch.long, device=x.device)
        )
        slots = pool_starts.unsqueeze(-1) + torch.arange(kpool, device=x.device)
        k8, k_scale = pool_keys(keys[slots], gate[slots], part.ape)
        return k8, k_scale, first_pool

    @torch.no_grad()
    def candidates(
        self,
        layer: int,
        hidden: torch.Tensor,
        anchor_positions: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Indexer picks for ``anchor - 1`` per draft block, ``-1`` unused."""
        self._check_input(layer, hidden)
        part = self.parts[str(layer)]
        bsz, seq_len, _ = hidden.shape
        width = self.topk + self.kpool - 1
        out = torch.full(
            (bsz, anchor_positions.shape[1], width), -1, dtype=torch.long, device=hidden.device
        )
        doc_start = document_starts(ctx_doc_ids, bsz, seq_len, hidden.device)
        for b in range(bsz):
            x = hidden[b].to(torch.bfloat16)
            rows = anchor_positions[b].to(hidden.device).long() - 1
            alive = rows >= 0
            if ctx_doc_ids is not None:
                docs = ctx_doc_ids[b].to(hidden.device)
                alive &= docs.gather(0, rows.clamp(min=0)) >= 0
            rows = rows.clamp(min=0)
            start = doc_start[b].gather(0, rows)
            visible = rows - start + 1
            short = alive & (visible <= self.topk)
            if bool(short.any()):
                every = start[short].unsqueeze(-1) + torch.arange(width, device=hidden.device)
                count = visible[short].unsqueeze(-1)
                out[b, short] = torch.where(
                    torch.arange(width, device=hidden.device) < count, every, -1
                )
            scored = alive & ~short
            if not bool(scored.any()):
                continue
            spans = _document_spans(None if ctx_doc_ids is None else ctx_doc_ids[b], seq_len)
            k8, k_scale, first_pool = self._pooled_keys(part, x, spans)
            base = torch.tensor(
                [first_pool[int(s)] for s in start[scored].tolist()],
                dtype=torch.long,
                device=hidden.device,
            )
            q8, weights = self._queries(part, x[rows[scored]])
            ends = base + visible[scored] // self.kpool
            logits = index_logits(q8, weights, k8, k_scale, base, ends)
            pools = top_pools(logits, base, ends, self.topk // self.kpool)
            tokens = expand_pools(pools, visible[scored], self.kpool)
            out[b, scored] = torch.where(tokens >= 0, tokens + start[scored].unsqueeze(-1), -1)
        return out


class Glm53TargetKVLatent:
    """Target layer ``layer``'s latents in place of a draft layer's own (``target_kv_layers``)."""

    context_layout = "shared"
    qk_rope_head_dim = 0

    def __init__(self, reuse: Glm53TargetReuse, layer: int):
        self.reuse, self.layer = reuse, int(layer)
        self.kv_lora_rank = reuse.kv_lora_rank

    def latents(
        self, hidden, input_ids, ctx_doc_ids=None, position_ids=None, sp_group=None
    ) -> torch.Tensor:
        if sp_group is not None:
            raise NotImplementedError("GLM-5.3 target KV reuse does not support USP")
        return self.reuse.latent(self.layer, hidden)


class Glm53TargetIndexer:
    """Target layer ``layer``'s indexer picks as a draft layer's context (``target_index_layer``)."""

    def __init__(self, reuse: Glm53TargetReuse, layer: int):
        self.reuse, self.layer, self.topk = reuse, int(layer), reuse.topk

    def candidate_indices(
        self, hidden, input_ids, anchor_positions, ctx_doc_ids=None, position_ids=None
    ) -> torch.Tensor:
        return self.reuse.candidates(self.layer, hidden, anchor_positions, ctx_doc_ids)


def glm53_reuse_layers(
    config: dict, kv_layers: Optional[List[int]], index_layers: Optional[List[int]]
) -> Dict[str, List[int]]:
    out = {}
    for field, layers in (("target_kv_layers", kv_layers), ("target_index_layer", index_layers)):
        for layer in layers or []:
            check_glm53_reuse_layer(config, int(layer), field)
        out[field] = list(layers or [])
    return out
