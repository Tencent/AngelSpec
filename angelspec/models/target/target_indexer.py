"""Frozen copy of one target layer's DSA indexer, picking context candidates for DFlash.

A drafter with ``DFlashConfig.target_index_layer = L`` lets each draft block read the
``target_index_topk`` context positions that target layer L's lightning indexer scores
highest for the last accepted token (the anchor's predecessor), plus the
``context_local_window`` tokens right before the anchor. This module reproduces that
selection on the training side.

hy4's layer-L indexer scores from that layer's attention input: ``input_layernorm`` of
the residual stream entering layer L, and the LoRA-q residual
``q_a_layernorm(q_a_proj(.))`` of the same. With vLLM, the drafter's captured hidden for
``target_layer_id`` L-1 is exactly the tensor ``input_layernorm`` receives: the engine
shifts post-layer ids by one, and under iHC reduces the streams with layer L's own
attention pre-gate (``vllm_runtime_patch._hyv4_flash_inner_forward``). Re-running the indexer
on it, with q and k rounded to fp8 as vLLM's indexer quantizes them, therefore
reproduces the target's top-k.
Positions follow the target, not the drafter: M-RoPE whenever the target config has an
``mrope_section``, read from the batch's prepared ``position_ids`` when the drafter has
them and rebuilt from the token ids (:func:`build_position_ids`) otherwise.

Hy4 flash's DSA layers carry the two-level compressed indexer (``self_attn.compressed_indexer``)
instead: keys pooled per document in groups of ``compressed_indexer_compress_ratio``, the
query recalling the top groups before its own plus its own group. It runs through the MTP
head's :class:`~angelspec.models.draft.mtp.MTPCompressedIndexer`, whose fp8 scoring rotates
q and k by Hadamard-128 before quantizing, as vLLM's does.

Only the tensors that path reads are loaded. The trainer holds the module outside
the FSDP-wrapped drafter, so it is neither sharded nor saved with the drafter.
"""

import glob
import json
import os
from dataclasses import asdict
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open

from angelspec.data.positions import PositionRecipe, build_position_ids
from angelspec.models.draft.llama3_eagle import LlamaRMSNorm, LlamaRotaryEmbedding
from angelspec.models.draft.mtp import MTPCompressedIndexer, MTPIndexer
from angelspec.models.ops.mrope import MRotaryEmbedding, validate_mrope_positions
from angelspec.models.ops.tilelang_dsa import tilelang_dsa_available
from angelspec.models.ops.ulysses import all_gather_seq, gather_seq_rows
from angelspec.models.target.block_quant import dequant_fp8_blocks

#: Initial RoPE table length; the table grows on demand. The target's own
#: max_position_embeddings (1M) would cost ~0.5 GB of fp32 tables up front.
_ROPE_TABLE_LEN = 65536

#: ``layer_types`` entry of a target layer that runs DSA, with an indexer.
DSA_LAYER_TYPE = "deepseek_sparse_attention"

#: Module parameter -> checkpoint key below ``<prefix>layers.<L>.``.
_CHECKPOINT_KEYS = {
    "input_layernorm.weight": "input_layernorm.weight",
    "q_a_proj.weight": "self_attn.q_a_proj.weight",
    "q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
    "indexer.wq_b.weight": "self_attn.indexer.wq_b.weight",
    "indexer.wk.weight": "self_attn.indexer.wk.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "indexer.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
}

#: The same for a target with Hy4 flash's compressed indexer.
_COMPRESSED_CHECKPOINT_KEYS = {
    "input_layernorm.weight": "input_layernorm.weight",
    "q_a_proj.weight": "self_attn.q_a_proj.weight",
    "q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
    **{
        f"indexer.{name}": f"self_attn.compressed_indexer.{name}"
        for name in (
            "query_projection.weight",
            "head_weight_projection.weight",
            "token_key_projection.weight",
            "token_key_norm.weight",
            "key_value_projection.weight",
            "key_gate_projection.weight",
            "key_norm.weight",
            "intra_group_ape",
        )
    },
}


def load_target_config(model_path: str) -> dict:
    """The target's raw ``config.json``."""
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


def _text_config(config: dict) -> dict:
    return config.get("text_config") or config


def check_target_index_layer(config: dict, layer: int) -> None:
    """Raise unless target ``layer`` runs DSA with an indexer of its own."""
    text = _text_config(config)
    if text.get("index_topk") is None:
        raise ValueError("target_index_layer needs a DSA target, but its config has no index_topk")
    num_layers = int(text["num_hidden_layers"])
    if not 0 <= layer < num_layers:
        raise ValueError(f"target_index_layer={layer} is outside the target's {num_layers} layers")
    layer_types = text.get("layer_types")
    if layer_types is not None and str(layer_types[layer]) != DSA_LAYER_TYPE:
        raise ValueError(
            f"target layer {layer} is {layer_types[layer]!r}, not {DSA_LAYER_TYPE}: "
            "it has no indexer"
        )
    indexer_types = text.get("indexer_types")
    if indexer_types is not None and str(indexer_types[layer]) != "full":
        raise ValueError(
            f"target layer {layer} borrows an earlier layer's indexer "
            f"(indexer_types[{layer}]={indexer_types[layer]!r}); point target_index_layer "
            "at a 'full' layer"
        )


def candidates_to_table(indices: torch.Tensor, seq_len: int) -> torch.Tensor:
    """``[B, n_blocks, K]`` candidate indices -> ``[B, n_blocks, seq_len]`` bool.

    ``-1`` marks an unused slot; those land in a scratch column past the end that
    is then dropped. The only place the two candidate forms are converted, so the
    mask-based and index-based attention paths cannot disagree about the set.
    """
    table = torch.zeros(
        indices.shape[0], indices.shape[1], seq_len + 1, dtype=torch.bool, device=indices.device
    )
    table.scatter_(2, torch.where(indices >= 0, indices, seq_len), True)
    return table[..., :seq_len]


def document_starts(
    doc_ids: Optional[torch.Tensor], bsz: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    """``[B, L]`` long: index of the first token of each position's document (0 unpacked)."""
    if doc_ids is None:
        return torch.zeros(bsz, seq_len, dtype=torch.long, device=device)
    docs = doc_ids.to(device)
    positions = torch.arange(seq_len, device=device).expand(bsz, -1)
    boundary = torch.ones_like(docs, dtype=torch.bool)
    boundary[:, 1:] = docs[:, 1:] != docs[:, :-1]
    return torch.where(boundary, positions, torch.zeros_like(positions)).cummax(-1).values


def target_rope_parameters(config: dict, layer: Optional[int] = None) -> dict:
    """The target's ``rope_parameters`` (``rope_scaling`` without them), as target layer
    ``layer`` applies them.

    Hy4 flash nests them per layer type (``{"deepseek_sparse_attention": {...},
    "sliding_attention": {...}}``, with different thetas); ``layer`` picks the entry of its
    own type, so such a target needs it.
    """
    text = _text_config(config)
    rope = dict(text.get("rope_parameters") or text.get("rope_scaling") or {})
    nested = {key: value for key, value in rope.items() if isinstance(value, dict)}
    if nested:
        layer_types = text.get("layer_types")
        layer_type = (
            str(layer_types[layer]) if layer is not None and layer_types is not None else None
        )
        if layer_type not in nested:
            raise ValueError(
                f"the target's rope_parameters are nested per layer type {sorted(nested)}, "
                f"but target layer {layer} has type {layer_type!r}"
            )
        rope = {key: value for key, value in rope.items() if not isinstance(value, dict)}
        rope.update(nested[layer_type])
    return rope


def target_rope(
    config: dict, rope_dim: int, layer: Optional[int] = None
) -> Tuple[nn.Module, Optional[PositionRecipe]]:
    """The target's RoPE over a ``rope_dim`` slice, as target layer ``layer`` applies it.

    Returns ``(rotary_emb, position_recipe)``: an ``MRotaryEmbedding`` and the recipe
    that rebuilds the target's positions from token ids when the target config has an
    ``mrope_section``, else a 1D ``LlamaRotaryEmbedding`` and ``None``. See
    :func:`target_rope_parameters` for ``layer``.
    """
    text = _text_config(config)
    rope = target_rope_parameters(config, layer)
    theta = float(rope.get("rope_theta", text.get("rope_theta", 10000.0)))
    if rope.get("mrope_section") is None:
        return (
            LlamaRotaryEmbedding(rope_dim, max_position_embeddings=_ROPE_TABLE_LEN, base=theta),
            None,
        )
    rope.setdefault("rope_theta", theta)
    rope.setdefault("rope_type", rope.get("type", "default"))
    recipe = PositionRecipe(
        **{
            key: int(config.get(key, text.get(key, default)))
            for key, default in asdict(PositionRecipe()).items()
            if key.endswith("token_id")
        }
    )
    return MRotaryEmbedding(rope_dim, rope), recipe


def target_positions(
    input_ids: torch.Tensor,
    position_recipe: Optional[PositionRecipe],
    ctx_doc_ids: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The target's positions: ``[B, 3, L]`` with M-RoPE, else ``[B, L]``, per document.

    Under M-RoPE the batch's prepared ``position_ids`` are used when given, else the
    positions are rebuilt from ``input_ids``; a 1D-RoPE target ignores them.
    """
    bsz, seq_len = input_ids.shape
    if position_recipe is not None:
        if position_ids is not None:
            return validate_mrope_positions(position_ids, bsz, seq_len).to(input_ids.device)
        return build_position_ids(input_ids, position_recipe, doc_ids=ctx_doc_ids)
    positions = torch.arange(seq_len, device=input_ids.device).expand(bsz, -1)
    return positions - document_starts(ctx_doc_ids, bsz, seq_len, input_ids.device)


def target_cos_sin(
    rotary_emb: nn.Module,
    position_recipe: Optional[PositionRecipe],
    positions: torch.Tensor,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)``, ``[B, L, rope_dim]``, at the target's ``positions``."""
    ref = torch.zeros(1, dtype=torch.float32, device=positions.device)
    if position_recipe is not None:
        return rotary_emb(ref, positions)
    cache_len = max(int(seq_len), int(positions.max().item()) + 1)
    cos, sin = rotary_emb(ref, seq_len=cache_len)
    return cos.squeeze(1).squeeze(0)[positions], sin.squeeze(1).squeeze(0)[positions]


def load_layer_tensors(
    module: nn.Module,
    model_path: str,
    layer: int,
    checkpoint_keys: Dict[str, str],
    anchor: str,
    stack: str = "layers",
) -> None:
    """Copy ``module``'s parameters from target layer ``layer`` of a safetensors checkpoint.

    ``checkpoint_keys`` maps each parameter to its key below ``<prefix><stack>.<layer>.``
    (``stack="mtp_layers"`` for the MTP layers); the prefix comes from the one checkpoint
    key that ends in parameter ``anchor``'s.
    """
    index_files = glob.glob(os.path.join(model_path, "*.index.json"))
    if index_files:
        with open(index_files[0]) as f:
            weight_map: Dict[str, str] = json.load(f)["weight_map"]
    else:
        files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected an index or one safetensors file in {model_path}")
        with safe_open(files[0], framework="pt") as f:
            weight_map = {key: os.path.basename(files[0]) for key in f.keys()}

    anchor_suffix = f"{stack}.{layer}.{checkpoint_keys[anchor]}"
    # "layers." must start a dotted component, or "mtp_layers.0." would match layer 0.
    matches = [
        key for key in weight_map if key == anchor_suffix or key.endswith("." + anchor_suffix)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"expected exactly one '*{anchor_suffix}' in the checkpoint, found {matches}"
        )
    prefix = matches[0][: -len(checkpoint_keys[anchor])]

    params = dict(module.named_parameters())
    by_file: Dict[str, list] = {}
    for name, suffix in checkpoint_keys.items():
        key = prefix + suffix
        if key not in weight_map:
            raise KeyError(f"'{key}' is not in the checkpoint")
        by_file.setdefault(weight_map[key], []).append((name, key))
    config_path = os.path.join(model_path, "config.json")
    config = load_target_config(model_path) if os.path.exists(config_path) else {}
    quant = (
        config.get("quantization_config") or _text_config(config).get("quantization_config") or {}
    )
    block = tuple(quant.get("weight_block_size") or (128, 128))
    for file_name, entries in by_file.items():
        with safe_open(os.path.join(model_path, file_name), framework="pt") as f:
            for name, key in entries:
                tensor = f.get_tensor(key)
                # FP8 projections (GLM-5.3's) come with one scale per weight block.
                scale_key = key + "_scale_inv"
                if scale_key in weight_map:
                    with safe_open(
                        os.path.join(model_path, weight_map[scale_key]), framework="pt"
                    ) as scales:
                        tensor = dequant_fp8_blocks(tensor, scales.get_tensor(scale_key), block)
                param = params[name]
                if tuple(tensor.shape) != tuple(param.shape):
                    raise ValueError(
                        f"'{key}' has shape {tuple(tensor.shape)}, expected {tuple(param.shape)}"
                    )
                param.data.copy_(tensor)


class TargetIndexer(nn.Module):
    """Target layer ``layer``'s DSA indexer, returning ``topk`` context candidates per block.

    With a compressed indexer the candidates come in whole groups: the ``topk / ratio - 1``
    best-scored complete groups before the query's group, plus that group up to the query.
    """

    def __init__(self, config: dict, layer: int, topk: int, use_fp8: bool = True):
        super().__init__()
        check_target_index_layer(config, layer)
        if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
            raise ValueError(f"topk must be a positive int, got {topk!r}")
        text = _text_config(config)
        hidden_size = int(text["hidden_size"])
        q_lora_rank = int(text["q_lora_rank"])
        eps = float(text["rms_norm_eps"])
        rope_dim = int(text["qk_rope_head_dim"])
        self.layer = int(layer)
        self.topk = topk
        # vLLM's indexer scores from fp8 q and k; without that, ~3% of the top-2048
        # at 12k context differ from what the target selects at inference.
        self.use_fp8 = use_fp8

        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=eps)
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = LlamaRMSNorm(q_lora_rank, eps=eps)
        self.compressed = bool(text.get("use_compressed_indexer", False))
        if self.compressed:
            ratio = int(text["compressed_indexer_compress_ratio"])
            if (
                not text.get("compressed_indexer_skip_fine_select", False)
                or text.get("compressed_indexer_key_pooling", "gated") != "gated"
                or text.get("compressed_indexer_reuse_fine_indexer", False)
            ):
                raise ValueError(
                    "the compressed indexer is reproduced only with skip_fine_select, gated "
                    "key pooling and its own coarse projections"
                )
            if topk % ratio:
                raise ValueError(
                    f"topk={topk} must be a multiple of the compressed indexer's group size {ratio}"
                )
            self.indexer = MTPCompressedIndexer(
                SimpleNamespace(
                    compressed_indexer_compress_ratio=ratio,
                    compressed_indexer_num_attention_heads=int(
                        text["compressed_indexer_num_attention_heads"]
                    ),
                    compressed_indexer_head_dim=int(text["compressed_indexer_head_dim"]),
                    index_head_dim=int(text["index_head_dim"]),
                    qk_rope_head_dim=rope_dim,
                    # topk / ratio groups: the target's own candidate_topk recalls its own
                    # group_topk, and a wider topk recalls proportionally more.
                    compressed_indexer_group_topk=topk // ratio,
                    compressed_indexer_candidate_topk=topk,
                    q_lora_rank=q_lora_rank,
                    hidden_size=hidden_size,
                    rms_norm_eps=eps,
                )
            )
            self._checkpoint_keys = _COMPRESSED_CHECKPOINT_KEYS
        else:
            self.indexer = MTPIndexer(
                SimpleNamespace(
                    index_n_heads=int(text["index_n_heads"]),
                    index_head_dim=int(text["index_head_dim"]),
                    qk_rope_head_dim=rope_dim,
                    index_topk=topk,
                    q_lora_rank=q_lora_rank,
                    hidden_size=hidden_size,
                    rms_norm_eps=eps,
                )
            )
            self._checkpoint_keys = _CHECKPOINT_KEYS

        # Kept fp32 whatever dtype the weights take (see _cast_weights), as vLLM
        # rotates the indexer's q/k in fp32.
        self.rotary_emb, self._position_recipe = target_rope(config, rope_dim, self.layer)

        if set(dict(self.named_parameters())) != set(self._checkpoint_keys):
            raise RuntimeError("TargetIndexer parameters drifted from the checkpoint key map")
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_config(
        cls,
        model_path: str,
        layer: int,
        topk: int,
        device="cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TargetIndexer":
        """Uninitialized weights, e.g. on ranks that receive them by broadcast."""
        module = cls(load_target_config(model_path), layer, topk)
        module._cast_weights(dtype)
        return module.to(device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        layer: int,
        topk: int,
        device="cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TargetIndexer":
        module = cls(load_target_config(model_path), layer, topk)
        module.load_weights(model_path)
        module._cast_weights(dtype)
        return module.to(device)

    def _cast_weights(self, dtype: torch.dtype) -> None:
        # Module-wise, so the rotary tables (non-persistent buffers) stay fp32, and so do
        # the compressed indexer's pooling weights, which the target keeps in fp32 too.
        modules = [self.input_layernorm, self.q_a_proj, self.q_a_layernorm]
        if self.compressed:
            modules += [
                self.indexer.query_projection,
                self.indexer.head_weight_projection,
                self.indexer.token_key_projection,
                self.indexer.token_key_norm,
            ]
        else:
            modules.append(self.indexer)
        for module in modules:
            module.to(dtype)

    def load_weights(self, model_path: str) -> None:
        """Copy the tensors this module needs from a safetensors checkpoint."""
        anchor = "indexer.query_projection.weight" if self.compressed else "indexer.wq_b.weight"
        load_layer_tensors(self, model_path, self.layer, self._checkpoint_keys, anchor)

    def _positions(
        self,
        input_ids: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The target's positions: ``[B, 3, L]`` with M-RoPE, else ``[B, L]``, per document."""
        return target_positions(input_ids, self._position_recipe, ctx_doc_ids, position_ids)

    def _cos_sin(self, positions: torch.Tensor, seq_len: int):
        return target_cos_sin(self.rotary_emb, self._position_recipe, positions, seq_len)

    @torch.no_grad()
    def candidate_indices(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        sp_group=None,
    ) -> torch.Tensor:
        """``[B, n_blocks, topk]`` long: the same candidates, as indices.

        What the indexer natively returns, ``-1`` in unused slots.
        :meth:`candidate_table` scatters this into a bool table for the mask-based
        attention path; a sparse-attention kernel takes the indices as they are, so
        it reads them here instead of recovering them from the table. Both go
        through :meth:`_score`, so the two forms cannot drift.
        """
        return self._score(
            hidden, input_ids, anchor_positions, ctx_doc_ids, position_ids, sp_group
        )

    @torch.no_grad()
    def candidate_table(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        sp_group=None,
    ) -> torch.Tensor:
        """``[B, n_blocks, L]`` bool: the context candidates of each draft block.

        Args:
            hidden: ``[B, L, D]`` captured input of target layer ``layer``'s
                ``input_layernorm`` (the drafter's ``target_layer_id`` ``layer - 1``);
                with ``sp_group``, this rank's contiguous ``[B, L / sp, D]`` shard.
            input_ids: ``[B, L]`` token ids, for the target's (M-RoPE) positions.
            anchor_positions: ``[B, n_blocks]`` block anchors.
            ctx_doc_ids: ``[B, L]`` packed-document ids, or ``None``.
            position_ids: ``[B, 3, L]`` prepared M-RoPE positions, or ``None`` to
                build them from ``input_ids``. Ignored by a 1D-RoPE target.
            sp_group: the Ulysses group ``hidden`` is sharded over (USP), else ``None``.
                Every rank of it must call with its own anchors.

        Block ``j`` gets the ``topk`` keys the indexer scores highest for query position
        ``anchor - 1`` among ``[doc_start(anchor), anchor)``; fewer keys than that are all
        kept, and an anchor with no context gets none.
        """
        indices = self._score(
            hidden, input_ids, anchor_positions, ctx_doc_ids, position_ids, sp_group
        )
        return candidates_to_table(indices, input_ids.shape[1])

    @torch.no_grad()
    def _score(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        ctx_doc_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        sp_group=None,
    ) -> torch.Tensor:
        """``[B, n_blocks, topk]`` long candidate indices, ``-1`` where unfilled."""
        bsz, seq_len = input_ids.shape
        device = hidden.device
        dtype = self.q_a_proj.weight.dtype
        anchors = anchor_positions.to(device=device, dtype=torch.long)

        cos, sin = self._cos_sin(self._positions(input_ids, ctx_doc_ids, position_ids), seq_len)
        normed = self.input_layernorm(hidden.to(dtype))
        rows = (anchors - 1).clamp(min=0).unsqueeze(-1)
        if sp_group is None:
            keys = self.indexer.project_keys(normed, cos, sin)
            normed_q = normed.gather(1, rows.expand(-1, -1, normed.shape[-1]))
        else:
            world, rank = dist.get_world_size(sp_group), dist.get_rank(sp_group)
            shard = normed.shape[1]
            if shard * world != seq_len:
                raise ValueError(
                    f"hidden is a {shard}-position shard, but {seq_len} positions over "
                    f"{world} ranks need {seq_len / world:g}"
                )
            local = slice(rank * shard, (rank + 1) * shard)
            keys = all_gather_seq(
                self.indexer.project_keys(normed, cos[:, local], sin[:, local]).unsqueeze(1),
                sp_group,
            ).squeeze(1)
            normed_q = gather_seq_rows(normed, rows.squeeze(-1), sp_group)
        cos_q = cos.gather(1, rows.expand(-1, -1, cos.shape[-1]))
        sin_q = sin.gather(1, rows.expand(-1, -1, sin.shape[-1]))
        q_resid = self.q_a_layernorm(self.q_a_proj(normed_q))

        doc_start = document_starts(ctx_doc_ids, bsz, seq_len, device).gather(1, anchors)
        if self.compressed:
            indices, _ = self.indexer.sparse_indices(
                normed_q,
                q_resid,
                cos_q,
                sin_q,
                doc_ids=ctx_doc_ids,
                precomputed_keys=keys,
                query_positions=rows.squeeze(-1),
                use_fp8=self.use_fp8,
                kernel=normed_q.is_cuda and tilelang_dsa_available(),
            )
            # Row anchor - 1 of an anchor without context lies outside its document.
            return torch.where((anchors > doc_start).unsqueeze(-1), indices.long(), -1)

        kv = torch.arange(seq_len, device=device)
        allowed = (kv < anchors.unsqueeze(-1)) & (kv >= doc_start.unsqueeze(-1))
        add_mask = torch.where(
            allowed,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
        ).unsqueeze(1)

        indices, _ = self.indexer(
            hidden_states=normed_q,
            q_resid=q_resid,
            cos=cos_q,
            sin=sin_q,
            attention_mask=add_mask,
            return_indices=True,
            precomputed_keys=keys,
            use_fp8=self.use_fp8,
        )
        return indices.long()
