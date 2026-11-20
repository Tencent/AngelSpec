#!/usr/bin/env python3
"""Pack an AngelSpec DFly2 drafter into a Hy4 or Hy4 flash target checkpoint as vLLM
serves it: ``model.dfly_layers.*`` tensors, ``dfly_*`` config fields and, for a block-FP8
target, the drafter's layer projections in the same block-FP8 format.

The output directory links every target file except the index, the config and any shard
holding a drafter the target already serves (dropped, or rewritten without it when it
shares the shard with target weights), adds one shard holding the drafter and rewrites the
index and config.

    python tools/export_dfly_to_target.py --target <Hy4 or Hy4 flash checkpoint> \
        --draft <drafter from tools/convert_to_hf.py> --out <served checkpoint>
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PREFIX = "model.dfly_layers."
SHARD = "model-dfly.safetensors"
INDEX = "model.safetensors.index.json"
FP8_MAX = 448.0
FP8_PROJECTIONS = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_b_proj",
    "self_attn.o_proj",
    "self_attn.linear_gate",
)
SHARED_WITH_TARGET = ("embed_tokens.weight", "lm_head.weight")
DFLY2_MODEL_TYPES = ("dfly2", "dspark")
_DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "U8": 1}


def load_draft(path: str):
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    if config.get("model_type") not in DFLY2_MODEL_TYPES or not config.get("target_kv_layers"):
        raise ValueError(
            f"{path} is not a DFly2 drafter (model_type dfly2 or dspark with target_kv_layers): "
            f"got model_type={config.get('model_type')!r}, "
            f"target_kv_layers={config.get('target_kv_layers')!r}"
        )
    weights = os.path.join(path, "model.safetensors")
    if os.path.exists(weights):
        return config, load_file(weights)
    state = torch.load(
        os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    return config, state


def dfly_fields(config: dict, block_size=None) -> dict:
    fields = {
        "dfly_mtp_fusion": config.get("mtp_fusion"),
        "dfly_target_index_topk": config.get("target_index_topk"),
        "dfly_context_local_window": config.get("context_local_window"),
        "dfly_enable_hidden_correction": bool(config.get("enable_hidden_correction", False)),
        "dfly_num_hidden_layers": config["num_hidden_layers"],
        "dfly_mask_token_id": config["mask_token_id"],
        "dfly_target_index_layer": config.get("target_index_layer"),
        "dfly_target_kv_layers": config.get("target_kv_layers"),
        "dfly_enable_ihc": False,
        "dfly_mlp_type": "dense",
        "dfly_attn_causal": True,
    }
    if block_size is not None:
        fields["dfly_block_size"] = block_size
    return fields


def serving_key(key: str) -> str:
    return PREFIX + (key[len("layers.") :] if key.startswith("layers.") else key)


def is_fp8_projection(key: str) -> bool:
    parts = key.split(".")
    return (
        parts[0] == "layers" and parts[-1] == "weight" and ".".join(parts[2:-1]) in FP8_PROJECTIONS
    )


def quantize_fp8_blocks(weight: torch.Tensor, block, scale_dtype: torch.dtype):
    """e4m3 ``weight`` with one ``amax / 448`` scale per ``block`` tile."""
    rows, cols = weight.shape
    padded = F.pad(weight.float(), (0, -cols % block[1], 0, -rows % block[0]))
    tiles = padded.view(
        padded.shape[0] // block[0], block[0], padded.shape[1] // block[1], block[1]
    )
    scale = (tiles.abs().amax(dim=(1, 3)) / FP8_MAX).clamp(min=1e-12).to(scale_dtype)
    expanded = scale.float().repeat_interleave(block[0], 0)[:rows]
    expanded = expanded.repeat_interleave(block[1], 1)[:, :cols]
    quantized = (weight.float() / expanded).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale


def _quantization_config(config: dict) -> dict:
    text = config.get("text_config") or {}
    return config.get("quantization_config") or text.get("quantization_config") or {}


def _scale_dtype(target: str, weight_map: dict) -> torch.dtype:
    key = next((key for key in weight_map if key.endswith(".weight_scale_inv")), None)
    if key is None:
        return torch.float32
    with safe_open(os.path.join(target, weight_map[key]), framework="pt") as f:
        return torch.bfloat16 if f.get_slice(key).get_dtype() == "BF16" else torch.float32


def _tensor_bytes(target: str, weight_map: dict, keys) -> int:
    by_file = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    total = 0
    for file_name, names in by_file.items():
        with safe_open(os.path.join(target, file_name), framework="pt") as f:
            for name in names:
                tensor = f.get_slice(name)
                numel = 1
                for size in tensor.get_shape():
                    numel *= size
                total += numel * _DTYPE_BYTES[tensor.get_dtype()]
    return total


def export(target: str, draft: str, out: str, fp8=None, block_size=None) -> None:
    """Write ``out``: ``target`` serving ``draft`` as its DFly drafter. ``fp8=None``
    quantizes the projections exactly when the target is block-FP8."""
    with open(os.path.join(target, "config.json")) as f:
        target_config = json.load(f)
    with open(os.path.join(target, INDEX)) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    quant = _quantization_config(target_config)
    if fp8 is None:
        fp8 = quant.get("quant_method") == "fp8"
    block = tuple(quant.get("weight_block_size") or (128, 128))
    scale_dtype = _scale_dtype(target, weight_map)

    draft_config, state = load_draft(draft)
    shard, not_converted = {}, []
    for key, value in state.items():
        if key in SHARED_WITH_TARGET:
            continue
        name = serving_key(key)
        if fp8 and is_fp8_projection(key):
            shard[name], shard[name + "_scale_inv"] = quantize_fp8_blocks(
                value, block, scale_dtype
            )
        else:
            shard[name] = value.to(torch.bfloat16).contiguous()
            if fp8 and value.dim() == 2 and key.endswith(".weight"):
                not_converted.append(name[: -len(".weight")])

    os.makedirs(out, exist_ok=True)
    save_file(shard, os.path.join(out, SHARD), metadata={"format": "pt"})

    stale = [key for key in weight_map if key.startswith(PREFIX)]
    stale_files = {weight_map[key] for key in stale}
    removed = _tensor_bytes(target, weight_map, stale)
    for key in stale:
        del weight_map[key]
    # vLLM loads every tensor of a shard it reads, so no output shard may keep the old drafter.
    for file_name in sorted(stale_files & set(weight_map.values())):
        with safe_open(os.path.join(target, file_name), framework="pt") as f:
            kept = {key: f.get_tensor(key) for key in f.keys() if not key.startswith(PREFIX)}
            shard_metadata = f.metadata()
        save_file(kept, os.path.join(out, file_name), metadata=shard_metadata)
    weight_map.update({key: SHARD for key in shard})
    metadata = index.get("metadata") or {}
    if "total_size" in metadata:
        added = sum(tensor.numel() * tensor.element_size() for tensor in shard.values())
        metadata["total_size"] = metadata["total_size"] - removed + added
    with open(os.path.join(out, INDEX), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    for file_name in os.listdir(target):
        if file_name in (INDEX, "config.json", SHARD) or file_name.startswith("."):
            continue
        if file_name in stale_files:
            continue
        link = os.path.join(out, file_name)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(os.path.join(target, file_name)), link)

    served = {key: value for key, value in target_config.items() if not key.startswith("dfly_")}
    served.update(dfly_fields(draft_config, block_size))
    if fp8 and "modules_to_not_convert" in quant:
        kept = [name for name in quant["modules_to_not_convert"] if not name.startswith(PREFIX)]
        quant["modules_to_not_convert"] = kept + sorted(not_converted)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(served, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True, help="Hy4 or Hy4 flash checkpoint directory")
    parser.add_argument("--draft", required=True, help="drafter directory (config + weights)")
    parser.add_argument("--out", required=True, help="output checkpoint directory")
    parser.add_argument(
        "--bf16", action="store_true", help="keep the drafter in bf16 even for an FP8 target"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="the training dflash_block_size, anchor included, written as dfly_block_size; "
        "vLLM then drafts block_size - 1 tokens unless num_speculative_tokens is given",
    )
    args = parser.parse_args()
    export(
        args.target,
        args.draft,
        args.out,
        fp8=False if args.bf16 else None,
        block_size=args.block_size,
    )
    print(f"served checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
