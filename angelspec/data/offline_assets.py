import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from angelspec.data.offline import (
    atomic_json,
    child_path,
    file_hash,
    object_hash,
    read_json,
    tensor_digest,
)

FROZEN_KEYS = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")
METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
)
# Required next to config.json at load time; kept out of METADATA_FILES so existing
# target identities, and the dumps pinned to them, stay valid.
AUXILIARY_FILES = ("engram_vocab_map.npz",)


tensor_hash = tensor_digest


def _all_finite(value: torch.Tensor) -> bool:
    if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        value = value.float()
    return not value.is_floating_point() or bool(torch.isfinite(value).all())


def _weight_map(root):
    if (root / "model.safetensors.index.json").is_file():
        return read_json(root / "model.safetensors.index.json")["weight_map"]
    if (root / "model.safetensors").is_file():
        with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as f:
            return {key: "model.safetensors" for key in f.keys()}
    raise ValueError("Target must have local safetensors weights")


def _stack_prefix(weight_map, anchor):
    """``<prefix><stack>.<L>.`` of the one key ending in ``anchor`` (``<stack>.<L>.<rest>``)."""
    matches = [key for key in weight_map if key == anchor or key.endswith("." + anchor)]
    if len(matches) > 1:
        raise ValueError(f"Expected one '*{anchor}' in the target, found {matches}")
    if not matches:
        return None
    return matches[0][: -len(anchor.split(".", 2)[2])]


def target_weight_prefixes(target, draft_config):
    """Key prefixes of the target weights a DFlash-family drafter reads beyond the frozen
    components: the input norm and whole attention (DSA indexer included) of each layer named
    by ``target_kv_layers``, ``target_index_layer``, ``target_index_union_layers`` or
    ``target_attn_init_layers``, and with ``mtp_fusion_init`` "target" the opening of the
    target's first MTP layer."""
    from angelspec.models.draft.dflash import DFlashConfig

    if not isinstance(draft_config, DFlashConfig):
        return []
    layers = set(int(x) for x in getattr(draft_config, "target_index_union_layers", None) or ())
    for name in ("target_kv_layers", "target_index_layer", "target_attn_init_layers"):
        value = getattr(draft_config, name, None)
        for layer in value if isinstance(value, (list, tuple)) else [value]:
            if layer is not None:
                layers.add(int(layer))
    mtp_fusion = (
        getattr(draft_config, "mtp_fusion", None) is not None
        and getattr(draft_config, "mtp_fusion_init", "target") == "target"
    )
    if not layers and not mtp_fusion:
        return []
    weight_map = _weight_map(Path(target).resolve())
    prefixes = []
    for layer in sorted(layers):
        anchor = f"layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"
        stack = _stack_prefix(weight_map, anchor)
        if stack is None:
            raise ValueError(f"{target} has no target layer {layer} attention ('*{anchor}')")
        prefixes += [stack + "input_layernorm.", stack + "self_attn."]
    if mtp_fusion:
        anchor = "mtp_layers.0.eh_proj.weight"
        stack = _stack_prefix(weight_map, anchor)
        if stack is None:
            raise ValueError(f"{target} has no MTP layer ('*{anchor}') for mtp_fusion_init")
        prefixes += [stack + "enorm.", stack + "hnorm.", stack + "eh_proj."]
    return prefixes


def read_target_components(target):
    root = Path(target).resolve()
    if (root / "frozen_assets.json").exists():
        raise ValueError("Dump needs a full target, not a frozen component bundle")
    config = read_json(root / "config.json")
    text = config.get("text_config", config)
    if (root / "model.safetensors.index.json").is_file():
        weight_map = read_json(root / "model.safetensors.index.json")["weight_map"]
    elif (root / "model.safetensors").is_file():
        weight_map = {key: "model.safetensors" for key in FROZEN_KEYS}
    else:
        raise ValueError("Target must have local safetensors weights")
    for relative in set(weight_map.values()):
        if not child_path(root, relative).is_file():
            raise FileNotFoundError(f"Full target shard missing: {relative}")
    tensors = {}
    expected = {
        FROZEN_KEYS[0]: (text["vocab_size"], text["hidden_size"]),
        FROZEN_KEYS[1]: (text["vocab_size"], text["hidden_size"]),
        FROZEN_KEYS[2]: (text["hidden_size"],),
    }
    for key in FROZEN_KEYS:
        if key not in weight_map:
            raise ValueError(f"Target lacks explicit frozen component {key}")
        with safe_open(str(child_path(root, weight_map[key])), framework="pt", device="cpu") as f:
            value = f.get_tensor(key).contiguous()
        if tuple(value.shape) != expected[key] or not value.is_floating_point():
            raise ValueError(f"Invalid frozen component {key}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite frozen component {key}")
        tensors[key] = value
    identity = {name: file_hash(root / name) for name in METADATA_FILES if (root / name).is_file()}
    identity["frozen_components"] = object_hash({k: tensor_hash(v) for k, v in tensors.items()})
    identity["weight_inventory"] = object_hash(
        {
            name: {
                "bytes": child_path(root, name).stat().st_size,
                "mtime_ns": child_path(root, name).stat().st_mtime_ns,
            }
            for name in sorted(set(weight_map.values()))
        }
    )
    return config, identity, tensors


def mtp_init_prefix(args, config):
    from angelspec.models.draft.mtp import MTPConfig

    if not isinstance(config, MTPConfig):
        return None
    mtp_layer = getattr(config, "mtp_init_from_mtp_layer", None)
    if mtp_layer is None:
        mtp_layer = getattr(args, "mtp_init_from_mtp_layer", None)
    target_layer = getattr(config, "mtp_init_from_target_layer", None)
    if target_layer is None:
        target_layer = getattr(args, "mtp_init_from_target_layer", None)
    if mtp_layer is not None and target_layer is not None:
        raise ValueError("Select only one MTP initialization source")
    layer = mtp_layer if mtp_layer is not None else target_layer
    if layer is None:
        return None
    if int(layer) != layer or layer < 0:
        raise ValueError("MTP initialization layer must be a nonnegative integer")
    return f"model.{'mtp_layers' if mtp_layer is not None else 'layers'}.{int(layer)}."


def _read_prefixed(root, prefixes):
    weight_map = _weight_map(root)
    by_file = {}
    for key, name in weight_map.items():
        if key.startswith(tuple(prefixes)):
            by_file.setdefault(name, []).append(key)
    tensors = {}
    for name, keys in by_file.items():
        with safe_open(str(child_path(root, name)), framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key).contiguous()
    return tensors


def export_target_assets(target, output, init_prefix=None, weight_prefixes=()):
    root, output = Path(target).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Frozen output must not be inside the target directory")
    weight_prefixes = sorted(weight_prefixes)
    _, identity, tensors = read_target_components(root)
    if output.exists():
        from angelspec.training.offline import verify_frozen_target

        verify_frozen_target(output, identity)
        metadata = read_json(output / "frozen_assets.json")
        if init_prefix != metadata.get("mtp_init_prefix"):
            raise ValueError("Existing frozen bundle has a different MTP initialization source")
        if weight_prefixes != metadata.get("target_weight_prefixes", []):
            raise ValueError(
                "Existing frozen bundle holds other target layers than this draft reads; "
                "export to a new frozen_target_path"
            )
        for name in AUXILIARY_FILES:
            if (root / name).is_file() and not (output / name).is_file():
                partial = output / f".{name}.partial"
                shutil.copy2(root / name, partial)
                with partial.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(partial, output / name)
        return identity
    if init_prefix is not None:
        from angelspec.training.mtp_trainer import MTPTrainer

        init_state = MTPTrainer._read_target_layer_weights(str(root), init_prefix)
        if not init_state:
            raise ValueError(f"Target has no initialization weights for {init_prefix}")
        if any(not _all_finite(value) for value in init_state.values()):
            raise ValueError("Non-finite MTP initialization weights")
        tensors.update(init_state)
    if weight_prefixes:
        layer_state = _read_prefixed(root, weight_prefixes)
        if any(not _all_finite(value) for value in layer_state.values()):
            raise ValueError("Non-finite target layer weights")
        tensors.update(layer_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".frozen-partial-", dir=output.parent))
    try:
        save_file(tensors, str(staging / "frozen.safetensors"))
        for name in METADATA_FILES + AUXILIARY_FILES:
            if name != "model.safetensors.index.json" and (root / name).is_file():
                shutil.copy2(root / name, staging / name)
        atomic_json(
            staging / "model.safetensors.index.json",
            {
                "metadata": {"frozen_components_only": True},
                "weight_map": {key: "frozen.safetensors" for key in tensors},
            },
        )
        atomic_json(
            staging / "frozen_assets.json",
            {
                "schema_version": 1,
                "frozen_components_only": True,
                "source_target_metadata_sha256": identity,
                "bundle_files_sha256": {p.name: file_hash(p) for p in staging.iterdir()},
                "tensor_sha256": {k: tensor_hash(v) for k, v in tensors.items()},
                "mtp_init_prefix": init_prefix,
                "target_weight_prefixes": weight_prefixes,
                "identity_scope": "metadata, frozen component content, full-shard sizes/mtimes; not full weight hashes",
            },
        )
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return identity
