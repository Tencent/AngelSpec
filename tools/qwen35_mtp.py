"""Assemble native, unquantized Qwen3.5 MTP weights without changing the base."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

INDEX = "model.safetensors.index.json"
FAMILIES = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4}


def read_json(path):
    return json.loads(Path(path).read_text())


def text_config(config):
    text = config.get("text_config", config)
    if text.get("model_type") not in FAMILIES:
        raise ValueError("Expected a Qwen3.5 Dense or MoE config")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("Quantized checkpoints are not supported")
    if text.get("mtp_use_dedicated_embeddings", False):
        raise ValueError("Dedicated MTP embeddings are not supported")
    return text


def read_header(path):
    with Path(path).open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 100_000_000:
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(f.read(size))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def checkpoint_index(directory):
    directory = Path(directory)
    if (directory / INDEX).is_file():
        index = read_json(directory / INDEX)
        weight_map = index["weight_map"]
    elif (directory / "model.safetensors").is_file():
        weight_map = {k: "model.safetensors" for k in read_header(directory / "model.safetensors")}
        index = {"metadata": {}, "weight_map": weight_map}
    else:
        raise ValueError(f"No HF safetensors checkpoint in {directory}")
    if not weight_map:
        raise ValueError("Empty checkpoint index")
    headers = {}
    for shard in set(weight_map.values()):
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"Unsafe shard name: {shard}")
        headers[shard] = read_header(directory / shard)
    for key, shard in weight_map.items():
        if key not in headers[shard]:
            raise ValueError(f"Index references missing tensor: {key} in {shard}")
    return index, headers


def expected_shapes(config, layers, fused=False):
    h = config["hidden_size"]
    d = config["head_dim"]
    q = config["num_attention_heads"] * d
    kv = config["num_key_value_heads"] * d
    shapes = {
        "mtp.fc.weight": (h, 2 * h),
        "mtp.norm.weight": (h,),
        "mtp.pre_fc_norm_embedding.weight": (h,),
        "mtp.pre_fc_norm_hidden.weight": (h,),
    }
    for layer in range(layers):
        p = f"mtp.layers.{layer}."
        block = {
            "input_layernorm.weight": (h,),
            "post_attention_layernorm.weight": (h,),
            "self_attn.q_proj.weight": (q * (2 if config.get("attn_output_gate", True) else 1), h),
            "self_attn.k_proj.weight": (kv, h),
            "self_attn.v_proj.weight": (kv, h),
            "self_attn.o_proj.weight": (h, q),
            "self_attn.q_norm.weight": (d,),
            "self_attn.k_norm.weight": (d,),
        }
        if config.get("attention_bias", False):
            raise ValueError("attention_bias=True is not supported")
        if "moe" in config["model_type"]:
            e = config["num_experts"]
            i = config["moe_intermediate_size"]
            s = config["shared_expert_intermediate_size"]
            block.update(
                {
                    "mlp.gate.weight": (e, h),
                    "mlp.shared_expert.gate_proj.weight": (s, h),
                    "mlp.shared_expert.up_proj.weight": (s, h),
                    "mlp.shared_expert.down_proj.weight": (h, s),
                    "mlp.shared_expert_gate.weight": (1, h),
                }
            )
            if fused:
                block["mlp.experts.gate_up_proj"] = (e, 2 * i, h)
                block["mlp.experts.down_proj"] = (e, h, i)
            else:
                for expert in range(e):
                    for proj, shape in (("gate", (i, h)), ("up", (i, h)), ("down", (h, i))):
                        block[f"mlp.experts.{expert}.{proj}_proj.weight"] = shape
        else:
            i = config["intermediate_size"]
            block.update(
                {
                    "mlp.gate_proj.weight": (i, h),
                    "mlp.up_proj.weight": (i, h),
                    "mlp.down_proj.weight": (h, i),
                }
            )
        shapes.update({p + key: value for key, value in block.items()})
    return shapes


def validate_head(head, config):
    if not head:
        raise ValueError(
            "No native mtp.* tensors found; AngelSpec HY/Qwen3 draft weights cannot "
            "be renamed into Qwen3.5 (attention gate and RMSNorm semantics differ)"
        )
    layer_ids = {int(m.group(1)) for key in head if (m := re.match(r"^mtp\.layers\.(\d+)\.", key))}
    if not layer_ids or layer_ids != set(range(max(layer_ids) + 1)):
        raise ValueError("MTP layer indices must be contiguous starting at zero")
    layers = len(layer_ids)
    fused = any(".experts.gate_up_proj" in key for key in head)
    shapes = expected_shapes(config, layers, fused)
    missing, extra = set(shapes) - set(head), set(head) - set(shapes)
    if missing or extra:
        raise ValueError(f"MTP key mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for key, shape in shapes.items():
        if tuple(head[key]["shape"]) != shape:
            raise ValueError(f"Shape mismatch for {key}: {head[key]['shape']} != {shape}")
        if head[key]["dtype"] not in DTYPE_BYTES:
            raise ValueError(f"Unsupported MTP dtype for {key}: {head[key]['dtype']}")
    return layers


def load_head(source):
    import torch
    from safetensors import safe_open

    source = Path(source)
    config = None
    if source.is_dir() and (source / "config.json").is_file():
        config = read_json(source / "config.json")
    if source.is_dir() and (
        (source / INDEX).is_file() or (source / "model.safetensors").is_file()
    ):
        index, _ = checkpoint_index(source)
        selected = {k: v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
        head = {}
        for shard in set(selected.values()):
            with safe_open(source / shard, framework="pt", device="cpu") as f:
                for key, filename in selected.items():
                    if filename == shard:
                        head[key] = f.get_tensor(key)
        return head, config
    if source.is_dir() and (source / "pytorch_model.bin").is_file():
        source = source / "pytorch_model.bin"
    if source.suffix == ".safetensors" and source.is_file():
        with safe_open(source, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys() if k.startswith("mtp.")}, config
    if source.suffix == ".bin" and source.is_file():
        raw = torch.load(source, map_location="cpu", weights_only=True)
        raw = raw.get("state_dict", raw)
        if not isinstance(raw, dict):
            raise ValueError("Expected a tensor state_dict")
        return {
            k: v for k, v in raw.items() if k.startswith("mtp.") and isinstance(v, torch.Tensor)
        }, config
    raise ValueError("MTP input must be native HF safetensors, .safetensors, or pytorch_model.bin")


def tensor_headers(tensors):
    names = {"torch.float16": "F16", "torch.bfloat16": "BF16", "torch.float32": "F32"}
    return {
        k: {"shape": list(t.shape), "dtype": names.get(str(t.dtype), str(t.dtype))}
        for k, t in tensors.items()
    }


def validate_source_config(base, source):
    if source is None:
        return
    source = text_config(source)
    fields = (
        "hidden_size",
        "head_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "vocab_size",
        "rms_norm_eps",
        "hidden_act",
        "rope_parameters",
        "rope_scaling",
        "rope_theta",
        "partial_rotary_factor",
    )
    for field in fields:
        if base.get(field) != source.get(field):
            raise ValueError(f"Source/base config mismatch: {field}")
    if base.get("attn_output_gate", True) != source.get("attn_output_gate", True):
        raise ValueError("Source/base config mismatch: attn_output_gate")


def tensor_nbytes(header):
    offsets = header.get("data_offsets")
    if offsets is not None:
        return offsets[1] - offsets[0]
    size = DTYPE_BYTES[header["dtype"]]
    for dimension in header["shape"]:
        size *= dimension
    return size


def check_paths(base, source, output):
    if not base.is_dir():
        raise ValueError(f"Base model directory not found: {base}")
    if not source.exists():
        raise ValueError(f"MTP input not found: {source}")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output must not exist: {output}")
    for path in (base, source):
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("Output and input paths must not overlap")
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        raise ValueError("Model output must be outside the code repository")


def assemble(base_model, source, output, mode, dry_run=False):
    from safetensors import safe_open
    from safetensors.torch import save_file

    if mode not in {"replace", "add"}:
        raise ValueError(f"Unknown mode: {mode}")
    raw_output = Path(output).expanduser()
    if raw_output.is_symlink():
        raise ValueError("Output must not be a symlink")
    base = Path(base_model).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    output = raw_output.resolve()
    check_paths(base, source, output)
    config = read_json(base / "config.json")
    text = text_config(config)
    index, headers = checkpoint_index(base)
    weight_map = index["weight_map"]
    old_head = {k: headers[s][k] for k, s in weight_map.items() if k.startswith("mtp.")}
    if mode == "replace" and not old_head:
        raise ValueError("Base has no MTP weights; use add_qwen35_mtp.py")
    if mode == "add" and old_head:
        raise ValueError("Base already has MTP weights; use replace_qwen35_mtp.py")
    if old_head:
        old_layers = validate_head(old_head, text)
        if text.get("mtp_num_hidden_layers", old_layers) != old_layers:
            raise ValueError("Base MTP layer count disagrees with config")
    head, donor_config = load_head(source)
    validate_source_config(text, donor_config)
    head_headers = tensor_headers(head)
    layers = validate_head(head_headers, text)
    if donor_config is not None:
        donor_layers = text_config(donor_config).get("mtp_num_hidden_layers", layers)
        if donor_layers != layers:
            raise ValueError("Source MTP layer count disagrees with config")
    report = {"mode": mode, "mtp_tensors": len(head), "mtp_layers": layers, "output": str(output)}
    if dry_run:
        return report

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        new_map = {k: s for k, s in weight_map.items() if not k.startswith("mtp.")}
        total_size = sum(tensor_nbytes(headers[s][k]) for k, s in new_map.items())
        for shard in set(new_map.values()):
            if any(k.startswith("mtp.") for k in headers[shard]):
                with safe_open(base / shard, framework="pt", device="cpu") as f:
                    payload = {k: f.get_tensor(k) for k, s in new_map.items() if s == shard}
                    save_file(payload, stage / shard, metadata=f.metadata())
                del payload
            else:
                shutil.copy2(base / shard, stage / shard)
        override = "angelspec_mtp.safetensors"
        suffix = 0
        while override in headers or (base / override).exists():
            suffix += 1
            override = f"angelspec_mtp_{suffix}.safetensors"
        save_file(
            {k: t.contiguous().clone() for k, t in head.items()},
            stage / override,
            metadata={"format": "pt"},
        )
        new_map.update({k: override for k in head})
        total_size += sum(tensor_nbytes(h) for h in head_headers.values())
        for path in base.iterdir():
            if (
                path.name.startswith(".")
                or path.name in {INDEX, "config.json"}
                or path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
                or path.name.endswith(".index.json")
            ):
                continue
            if path.is_dir():
                shutil.copytree(path, stage / path.name)
            else:
                shutil.copy2(path, stage / path.name)
        new_config = copy.deepcopy(config)
        new_config.get("text_config", new_config)["mtp_num_hidden_layers"] = layers
        new_config["angelspec_mtp"] = {
            "mode": mode,
            "source": str(source),
            "base_model": str(base),
            "format": "qwen3.5-native",
            "base_weights": "copy",
        }
        new_index = copy.deepcopy(index)
        new_index["weight_map"] = new_map
        new_index.setdefault("metadata", {})["total_size"] = total_size
        for name, value in (("config.json", new_config), (INDEX, new_index)):
            (stage / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output appeared during assembly: {output}")
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return report


def main(mode):
    parser = argparse.ArgumentParser(description=f"{mode.capitalize()} native Qwen3.5 MTP weights")
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--input", required=True, help="Native Qwen3.5 MTP checkpoint or full donor model"
    )
    parser.add_argument(
        "--output", required=True, help="New model directory outside the code repo"
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without creating output")
    args = parser.parse_args()
    try:
        result = assemble(args.base_model, args.input, args.output, mode, dry_run=args.dry_run)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False))
