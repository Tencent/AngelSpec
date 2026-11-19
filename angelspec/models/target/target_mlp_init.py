"""DFlash backbone MoE initialized from corresponding target decoder layers.

The HYV4 target stores each layer's routed experts as two fused tensors:

``mlp.experts.gate_up_proj``
    ``[experts, 2 * intermediate, hidden]`` in checkpoint Linear layout.
``mlp.experts.down_proj``
    ``[experts, hidden, intermediate]`` in checkpoint Linear layout.

The training-friendly :class:`~angelspec.models.draft.mtp.Hy3MoE` stores three
``[experts, in, out]`` tensors for ``torch._grouped_mm``.  This module streams
the target tensors one gate/up half at a time and transposes them into that
layout, while copying the router, correction bias, shared expert and
post-attention norm directly.

Targets that store one tensor per expert (DeepSeek-V3 style
``mlp.experts.{e}.{gate,up,down}_proj``) are read an expert at a time, block-FP8
weights dequantized on the way.

A dense drafter FFN can start from the same experts instead
(``target_mlp_dense_init_layers``): its intermediate dim is the shared expert(s)
followed by k routed experts, which only reads those k experts' slices.
"""

import glob
import json
import math
import os
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors import safe_open

from angelspec.models.draft.mtp import Hy3MoE
from angelspec.models.target.block_quant import dequant_fp8_blocks
from angelspec.models.target.target_indexer import _text_config, load_target_config


def _target_value(config: dict, *names: str, default=None):
    for name in names:
        if name in config:
            return config[name]
    return default


def _check_moe_layers(text: dict, layers: List[int], field: str) -> None:
    num_layers = int(text["num_hidden_layers"])
    for layer in layers:
        if not 0 <= layer < num_layers:
            raise ValueError(f"{field} layer {layer} is outside target [0, {num_layers})")

    layer_types = text.get("mlp_layer_types")
    if layer_types is not None:
        dense = [layer for layer in layers if layer_types[layer] not in ("moe", "sparse")]
    else:
        first_dense = int(text.get("first_k_dense_replace", 0) or 0)
        dense = [layer for layer in layers if layer < first_dense]
    if dense:
        raise ValueError(f"{field} copies target MoE experts, but target layers {dense} are dense")


def check_target_mlp_init_compatible(config: dict, draft_config, layers: List[int]) -> None:
    """Raise unless target ``layers`` can initialize the draft's MoE FFNs: the
    complete target MoE, or the experts ``target_mlp_init_experts`` picks (which
    may route fewer experts per token than the target)."""
    text = _text_config(config)
    _check_moe_layers(text, layers, "target_mlp_init_layers")

    target_experts = int(_target_value(text, "num_experts", "n_routed_experts"))
    subset = getattr(draft_config, "target_mlp_init_experts", None)
    if subset is not None:
        if not 1 <= int(draft_config.num_experts) <= target_experts:
            raise ValueError(
                f"target_mlp_init_experts picks num_experts ({draft_config.num_experts}) of "
                f"the target's {target_experts} experts"
            )
        if not isinstance(subset, str):
            for ids in subset:
                if max(ids) >= target_experts:
                    raise ValueError(
                        f"target_mlp_init_experts ids must be below {target_experts}, got {ids}"
                    )
    expected = {
        "hidden_size": int(text["hidden_size"]),
        "num_experts": target_experts if subset is None else int(draft_config.num_experts),
        "num_shared_experts": int(
            _target_value(text, "num_shared_experts", "n_shared_experts", default=0)
        ),
        "moe_intermediate_size": int(
            _target_value(text, "expert_hidden_dim", "moe_intermediate_size")
        ),
        "moe_router_score_func": text.get("scoring_func")
        or ("sigmoid" if text.get("moe_router_use_sigmoid", True) else "softmax"),
        "moe_router_enable_expert_bias": bool(text.get("moe_router_enable_expert_bias", True)),
        "route_norm": bool(_target_value(text, "route_norm", "norm_topk_prob", default=True)),
        "router_scaling_factor": float(
            _target_value(text, "router_scaling_factor", "routed_scaling_factor", default=1.0)
        ),
        "swiglu_limit": float(text.get("swiglu_limit", 0.0) or 0.0),
        "rms_norm_eps": float(text["rms_norm_eps"]),
    }
    if subset is None:
        expected["num_experts_per_tok"] = int(text["num_experts_per_tok"])
    drafted = {name: getattr(draft_config, name, None) for name in expected}
    drafted["moe_router_score_func"] = getattr(draft_config, "moe_router_score_func", None) or (
        "sigmoid" if draft_config.moe_router_use_sigmoid else "softmax"
    )
    mismatched = []
    for name, target_value in expected.items():
        draft_value = drafted[name]
        equal = (
            math.isclose(float(target_value), float(draft_value), rel_tol=0.0, abs_tol=1e-9)
            if isinstance(target_value, float)
            else target_value == draft_value
        )
        if not equal:
            mismatched.append(f"{name}: target {target_value!r} vs drafter {draft_value!r}")
    if mismatched:
        copied = "the complete target MoE" if subset is None else "target experts"
        raise ValueError(
            f"target_mlp_init_layers copies {copied}, so these must match: "
            + "; ".join(mismatched)
        )


def _checkpoint_layout(model_path: str) -> Dict[str, str]:
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
    return weight_map


class _MoELayout(NamedTuple):
    router: str
    bias: str
    norm: str
    shared: str
    expert: str
    projs: Tuple[str, str, str]


_HF_LAYOUT = _MoELayout(
    router="mlp.gate.weight",
    bias="mlp.gate.e_score_correction_bias",
    norm="post_attention_layernorm.weight",
    shared="mlp.shared_experts.{}.weight",
    expert="mlp.experts.{}.{}.weight",
    projs=("gate_proj", "up_proj", "down_proj"),
)


class _Checkpoint:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.weight_map = _checkpoint_layout(model_path)
        config_path = os.path.join(model_path, "config.json")
        config = load_target_config(model_path) if os.path.exists(config_path) else {}
        quant = (
            config.get("quantization_config")
            or _text_config(config).get("quantization_config")
            or {}
        )
        self.block = tuple(quant.get("weight_block_size") or (128, 128))
        self._files: Dict[str, object] = {}

    def _file(self, key: str):
        if key not in self.weight_map:
            raise KeyError(f"'{key}' is not in the checkpoint")
        name = self.weight_map[key]
        if name not in self._files:
            self._files[name] = safe_open(
                os.path.join(self.model_path, name), framework="pt", device=self.device
            )
        return self._files[name]

    def tensor(self, key: str) -> torch.Tensor:
        return self._file(key).get_tensor(key)

    def shape(self, key: str) -> Tuple[int, ...]:
        return tuple(self._file(key).get_slice(key).get_shape())

    def weight(self, key: str) -> torch.Tensor:
        tensor = self.tensor(key)
        scale_key = key + "_scale_inv"
        if scale_key in self.weight_map:
            return dequant_fp8_blocks(tensor, self.tensor(scale_key), self.block)
        return tensor


def _layer_layout(weight_map: Dict[str, str], layer: int) -> Tuple[str, _MoELayout]:
    layout = _HF_LAYOUT
    anchor = f"layers.{layer}.{layout.router}"
    found = [
        (key[: -len(layout.router)], layout)
        for key in weight_map
        if key == anchor or key.endswith("." + anchor)
    ]
    if len(found) != 1:
        raise KeyError(
            f"expected exactly one 'layers.{layer}' MoE router in the checkpoint, found "
            f"{[prefix for prefix, _ in found]}"
        )
    return found[0]


def _layer_prefix(weight_map: Dict[str, str], layer: int) -> str:
    return _layer_layout(weight_map, layer)[0]


def _copy(target: torch.Tensor, tensor: torch.Tensor, key: str) -> None:
    if tuple(tensor.shape) != tuple(target.shape):
        raise ValueError(
            f"'{key}' has shape {tuple(tensor.shape)}, expected {tuple(target.shape)}"
        )
    target.copy_(tensor)


def _copy_router_norm_shared(
    layer: nn.Module,
    ckpt: _Checkpoint,
    prefix: str,
    layout: _MoELayout,
    ids: Optional[List[int]] = None,
) -> None:
    moe = layer.mlp
    entries = [
        (layer.post_attention_layernorm.weight, layout.norm, False),
        (moe.router.gate.weight, layout.router, True),
    ]
    if moe.expert_bias is not None:
        entries.append((moe.expert_bias, layout.bias, True))
    for param, suffix, per_expert in entries:
        tensor = ckpt.tensor(prefix + suffix)
        if ids is not None and per_expert:
            tensor = tensor[ids]
        _copy(param.data, tensor, prefix + suffix)
    if moe.shared_mlp is not None:
        for proj, name in zip(("gate_proj", "up_proj", "down_proj"), layout.projs):
            key = prefix + layout.shared.format(name)
            _copy(getattr(moe.shared_mlp, proj).weight.data, ckpt.weight(key), key)


def _copy_split_experts(
    moe: Hy3MoE,
    ckpt: _Checkpoint,
    prefix: str,
    layout: _MoELayout,
    ids: Optional[List[int]] = None,
) -> None:
    params = (
        moe.experts_gate_proj.weight,
        moe.experts_up_proj.weight,
        moe.experts_down_proj.weight,
    )
    for slot, expert in enumerate(range(moe.num_experts) if ids is None else ids):
        for param, name in zip(params, layout.projs):
            key = prefix + layout.expert.format(expert, name)
            _copy(param.data[slot], ckpt.weight(key).t(), key)


def _copy_fused_experts(
    moe: Hy3MoE,
    model_path: str,
    weight_map: Dict[str, str],
    prefix: str,
    ids: Optional[List[int]] = None,
) -> None:
    experts = moe.num_experts
    hidden = moe.router.gate.in_features
    intermediate = moe.experts_gate_proj.weight.shape[-1]

    gate_up_key = prefix + "mlp.experts.gate_up_proj"
    down_key = prefix + "mlp.experts.down_proj"
    for key in (gate_up_key, down_key):
        if key not in weight_map:
            raise KeyError(f"'{key}' is not in the checkpoint")

    if ids is not None:
        gate_up = torch.stack(_read_experts(model_path, weight_map, gate_up_key, ids))
        down = torch.stack(_read_experts(model_path, weight_map, down_key, ids))
        for key, tensor, expected in (
            (gate_up_key, gate_up, (experts, 2 * intermediate, hidden)),
            (down_key, down, (experts, hidden, intermediate)),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"'{key}' experts {ids} have shape {tuple(tensor.shape)}, expected {expected}"
                )
        moe.experts_gate_proj.weight.data.copy_(gate_up[:, :intermediate].transpose(1, 2))
        moe.experts_up_proj.weight.data.copy_(gate_up[:, intermediate:].transpose(1, 2))
        moe.experts_down_proj.weight.data.copy_(down.transpose(1, 2))
        return

    with safe_open(os.path.join(model_path, weight_map[gate_up_key]), framework="pt") as f:
        source = f.get_slice(gate_up_key)
        shape = tuple(source.get_shape())
        expected = (experts, 2 * intermediate, hidden)
        if shape != expected:
            raise ValueError(f"'{gate_up_key}' has shape {shape}, expected {expected}")
        gate = source[:, :intermediate, :]
        moe.experts_gate_proj.weight.data.copy_(gate.transpose(1, 2))
        del gate
        up = source[:, intermediate:, :]
        moe.experts_up_proj.weight.data.copy_(up.transpose(1, 2))
        del up

    with safe_open(os.path.join(model_path, weight_map[down_key]), framework="pt") as f:
        source = f.get_slice(down_key)
        shape = tuple(source.get_shape())
        expected = (experts, hidden, intermediate)
        if shape != expected:
            raise ValueError(f"'{down_key}' has shape {shape}, expected {expected}")
        down = source[:, :, :]
        moe.experts_down_proj.weight.data.copy_(down.transpose(1, 2))


@torch.no_grad()
def init_mlp_from_target(
    draft_model: nn.Module,
    model_path: str,
    layers: List[int],
    experts: Union[None, str, List[List[int]]] = None,
    device: str = "cpu",
) -> Optional[List[List[int]]]:
    """Copy target layer ``layers[i]``'s norm and MoE into draft layer ``i``.

    ``experts`` None copies every routed expert; "bias" (the lowest correction
    biases), "random" (a draw seeded by the draft layer index) or one id list per
    draft layer copies only the drafter's num_experts of them, with their router
    rows and biases. Quantized target weights are dequantized on ``device``.
    Returns the experts used per layer (None when all were copied).
    """
    if len(layers) != len(draft_model.layers):
        raise ValueError(f"{len(layers)} target layers for {len(draft_model.layers)} draft layers")
    ckpt = _Checkpoint(model_path, device)
    chosen = []
    for index, (draft_layer, layer) in enumerate(zip(draft_model.layers, layers)):
        if not isinstance(draft_layer.mlp, Hy3MoE):
            raise TypeError(
                f"draft layer MLP must be Hy3MoE, got {type(draft_layer.mlp).__name__}"
            )
        prefix, layout = _layer_layout(ckpt.weight_map, layer)
        ids = (
            None
            if experts is None
            else _choose_experts(ckpt, prefix, layout, experts, index, draft_layer.mlp.num_experts)
        )
        _copy_router_norm_shared(draft_layer, ckpt, prefix, layout, ids)
        if prefix + "mlp.experts.gate_up_proj" in ckpt.weight_map:
            _copy_fused_experts(draft_layer.mlp, model_path, ckpt.weight_map, prefix, ids)
        else:
            _copy_split_experts(draft_layer.mlp, ckpt, prefix, layout, ids)
        chosen.append(ids)
    return None if experts is None else chosen


def _expert_dims(text: dict) -> tuple:
    width = int(_target_value(text, "expert_hidden_dim", "moe_intermediate_size"))
    shared = int(_target_value(text, "num_shared_experts", "n_shared_experts", default=0))
    routed = int(_target_value(text, "num_experts", "n_routed_experts"))
    return width, shared, routed


def check_target_mlp_dense_init_compatible(config: dict, draft_config, layers: List[int]) -> int:
    """Raise unless target ``layers`` can seed the draft's dense FFNs from their
    experts; return k, the number of routed experts each dense FFN takes."""
    text = _text_config(config)
    _check_moe_layers(text, layers, "target_mlp_dense_init_layers")
    mismatched = []
    if int(text["hidden_size"]) != int(draft_config.hidden_size):
        mismatched.append(
            f"hidden_size: target {text['hidden_size']} vs drafter {draft_config.hidden_size}"
        )
    if not math.isclose(
        float(text["rms_norm_eps"]), float(draft_config.rms_norm_eps), rel_tol=0.0, abs_tol=1e-12
    ):
        mismatched.append(
            f"rms_norm_eps: target {text['rms_norm_eps']} vs drafter {draft_config.rms_norm_eps}"
        )
    if mismatched:
        raise ValueError(
            "target_mlp_dense_init_layers copies target experts and post-attention norms, so "
            "these must match: " + "; ".join(mismatched)
        )

    width, shared, routed = _expert_dims(text)
    intermediate = int(draft_config.intermediate_size)
    k = intermediate // width - shared
    if intermediate % width or not 1 <= k <= routed:
        raise ValueError(
            f"target_mlp_dense_init_layers lays {shared} shared and k routed {width}-wide "
            f"experts along the intermediate dim, so intermediate_size must be "
            f"({shared} + k) * {width} with 1 <= k <= {routed}; got {intermediate}"
        )
    experts = draft_config.target_mlp_dense_init_experts
    if not isinstance(experts, str):
        for ids in experts:
            if len(ids) != k or max(ids) >= routed:
                raise ValueError(
                    f"target_mlp_dense_init_experts needs {k} expert ids below {routed} per "
                    f"layer, got {ids}"
                )
    return k


def _read_tensor(model_path: str, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    if key not in weight_map:
        raise KeyError(f"'{key}' is not in the checkpoint")
    with safe_open(os.path.join(model_path, weight_map[key]), framework="pt") as f:
        return f.get_tensor(key)


def _read_experts(
    model_path: str, weight_map: Dict[str, str], key: str, ids: List[int]
) -> List[torch.Tensor]:
    if key not in weight_map:
        raise KeyError(f"'{key}' is not in the checkpoint")
    with safe_open(os.path.join(model_path, weight_map[key]), framework="pt") as f:
        source = f.get_slice(key)
        num_experts = source.get_shape()[0]
        if max(ids) >= num_experts:
            raise ValueError(f"'{key}' has {num_experts} experts, asked for {ids}")
        return [source[e : e + 1][0] for e in ids]


def _choose_experts(
    ckpt: _Checkpoint,
    prefix: str,
    layout: _MoELayout,
    experts: Union[str, List[List[int]]],
    index: int,
    count: int,
) -> List[int]:
    """Draft layer ``index``'s ``count`` target experts: the lowest correction biases
    ("bias"), a draw seeded by ``index`` ("random") or its entry in the per-layer id
    lists."""
    if experts == "bias":
        bias = ckpt.tensor(prefix + layout.bias).float().cpu()
        return torch.argsort(bias, stable=True)[:count].tolist()
    if experts == "random":
        total = ckpt.shape(prefix + layout.router)[0]
        generator = torch.Generator().manual_seed(index)
        return sorted(torch.randperm(total, generator=generator)[:count].tolist())
    if isinstance(experts, str):
        raise ValueError(f"unknown expert choice {experts!r}")
    ids = list(experts[index])
    if len(ids) != count:
        raise ValueError(f"draft layer {index} needs {count} target experts, got {ids}")
    return ids


@torch.no_grad()
def init_dense_mlp_from_target(
    draft_model: nn.Module,
    model_path: str,
    layers: List[int],
    experts: Union[str, List[List[int]]] = "bias",
    routed_scale: Optional[float] = None,
) -> List[List[int]]:
    """Seed draft layer ``i``'s dense FFN from target layer ``layers[i]``'s experts.

    The FFN's intermediate dim is the shared expert(s) followed by k routed experts,
    so it computes ``shared(x) + routed_scale * sum_e expert_e(x)`` over a fixed set
    of experts for every token; ``routed_scale`` defaults to the target's routed
    scaling factor / k. The routed experts' ``swiglu_limit`` clamp is not kept.
    ``experts`` is "bias" (the k lowest expert-correction biases per layer),
    "random" or one id list per draft layer. The target's post-attention norm is
    copied as well. Returns the routed experts used per layer.
    """
    from angelspec.models.draft.dflash import DFlashMLP

    if len(layers) != len(draft_model.layers):
        raise ValueError(f"{len(layers)} target layers for {len(draft_model.layers)} draft layers")
    text = _text_config(load_target_config(model_path))
    width, shared, _ = _expert_dims(text)
    scaling = float(
        _target_value(text, "router_scaling_factor", "routed_scaling_factor", default=1.0)
    )
    ckpt = _Checkpoint(model_path)
    weight_map = ckpt.weight_map
    chosen = []
    for index, (draft_layer, layer) in enumerate(zip(draft_model.layers, layers)):
        mlp = draft_layer.mlp
        if not isinstance(mlp, DFlashMLP):
            raise TypeError(f"draft layer MLP must be DFlashMLP, got {type(mlp).__name__}")
        k = mlp.intermediate_size // width - shared
        prefix = _layer_prefix(weight_map, layer)
        ids = _choose_experts(ckpt, prefix, _HF_LAYOUT, experts, index, k)
        scale = scaling / k if routed_scale is None else float(routed_scale)

        gate = [
            _read_tensor(model_path, weight_map, prefix + "mlp.shared_experts.gate_proj.weight")
        ]
        up = [_read_tensor(model_path, weight_map, prefix + "mlp.shared_experts.up_proj.weight")]
        down = [
            _read_tensor(model_path, weight_map, prefix + "mlp.shared_experts.down_proj.weight")
        ]
        for gate_up in _read_experts(
            model_path, weight_map, prefix + "mlp.experts.gate_up_proj", ids
        ):
            gate.append(gate_up[:width])
            up.append(gate_up[width:])
        for expert_down in _read_experts(
            model_path, weight_map, prefix + "mlp.experts.down_proj", ids
        ):
            down.append(expert_down.float() * scale)

        for proj, tensor in (
            (mlp.gate_proj, torch.cat(gate, dim=0)),
            (mlp.up_proj, torch.cat(up, dim=0)),
            (mlp.down_proj, torch.cat([d.float() for d in down], dim=1)),
        ):
            if tuple(tensor.shape) != tuple(proj.weight.shape):
                raise ValueError(
                    f"target layer {layer}'s shared + {k} routed experts give shape "
                    f"{tuple(tensor.shape)}, expected {tuple(proj.weight.shape)}"
                )
            proj.weight.data.copy_(tensor)
        norm = _read_tensor(model_path, weight_map, prefix + "post_attention_layernorm.weight")
        draft_layer.post_attention_layernorm.weight.data.copy_(norm)
        chosen.append(ids)
    return chosen


__all__ = [
    "check_target_mlp_dense_init_compatible",
    "check_target_mlp_init_compatible",
    "init_dense_mlp_from_target",
    "init_mlp_from_target",
]
