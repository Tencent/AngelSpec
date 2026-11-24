import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


def is_qwen35_mtp(config):
    return getattr(config, "rms_norm_type", "llama") == "qwen3_5"


def read_weights(directory, *, prefix=None):
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        files = sorted({v for k, v in mapping.items() if prefix is None or k.startswith(prefix)})
    elif (directory / "model.safetensors").is_file():
        files = ["model.safetensors"]
    elif (directory / "pytorch_model.bin").is_file():
        raw = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
        return {k: v for k, v in raw.items() if prefix is None or k.startswith(prefix)}
    else:
        raise FileNotFoundError(f"No supported weights in {directory}")
    weights = {}
    for name in files:
        if Path(name).name != name:
            raise ValueError("Checkpoint shard must be a local filename")
        with safe_open(directory / name, framework="pt", device="cpu") as reader:
            keys = mapping if index.is_file() else reader.keys()
            for key in keys:
                if index.is_file() and mapping[key] != name:
                    continue
                if prefix is None or key.startswith(prefix):
                    weights[key] = reader.get_tensor(key)
    return weights


_NATIVE_TO_DRAFT = {
    "mtp.fc.weight": "eh_proj.weight",
    "mtp.pre_fc_norm_embedding.weight": "enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "hnorm.weight",
    "mtp.norm.weight": "final_layernorm.weight",
}
_NATIVE_MOE_TO_DRAFT = {
    "mlp.gate.weight": "mlp.router.gate.weight",
    "mlp.shared_expert.gate_proj.weight": "mlp.shared_mlp.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight": "mlp.shared_mlp.up_proj.weight",
    "mlp.shared_expert.down_proj.weight": "mlp.shared_mlp.down_proj.weight",
}
_EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")
_DRAFT_EXPERT = re.compile(r"mlp\.experts_(gate|up|down)_proj\.weight")


def native_to_draft(state):
    result, experts = {}, {}
    for name, value in state.items():
        if name in _NATIVE_TO_DRAFT:
            result[_NATIVE_TO_DRAFT[name]] = value
            continue
        if not name.startswith("mtp.layers.0."):
            raise ValueError(f"Unsupported native MTP tensor: {name}")
        rest = name[len("mtp.layers.0.") :]
        expert = _EXPERT.fullmatch(rest)
        if expert:
            experts.setdefault(expert.group(2), {})[int(expert.group(1))] = value
        elif rest == "mlp.experts.gate_up_proj":
            gate, up = value.chunk(2, dim=1)
            result["midlayer.mlp.experts_gate_proj.weight"] = gate.transpose(1, 2).contiguous()
            result["midlayer.mlp.experts_up_proj.weight"] = up.transpose(1, 2).contiguous()
        elif rest == "mlp.experts.down_proj":
            result["midlayer.mlp.experts_down_proj.weight"] = value.transpose(1, 2).contiguous()
        else:
            result["midlayer." + _NATIVE_MOE_TO_DRAFT.get(rest, rest)] = value
    for kind, by_expert in experts.items():
        if sorted(by_expert) != list(range(len(by_expert))):
            raise ValueError(f"Native MTP {kind}_proj experts are not 0..{len(by_expert) - 1}")
        stacked = torch.stack([by_expert[e].t() for e in range(len(by_expert))])
        result[f"midlayer.mlp.experts_{kind}_proj.weight"] = stacked
    return result


def draft_to_native(state):
    draft_to_native_names = {v: k for k, v in _NATIVE_TO_DRAFT.items()}
    moe_names = {v: k for k, v in _NATIVE_MOE_TO_DRAFT.items()}
    result = {}
    for name, value in state.items():
        if name == "embed_tokens.weight":
            continue
        value = value.detach().cpu()
        if name in draft_to_native_names:
            result[draft_to_native_names[name]] = value.contiguous()
            continue
        if not name.startswith("midlayer."):
            raise ValueError(f"Unsupported draft tensor: {name}")
        rest = name[len("midlayer.") :]
        expert = _DRAFT_EXPERT.fullmatch(rest)
        if expert:
            for e in range(value.shape[0]):
                key = f"mtp.layers.0.mlp.experts.{e}.{expert.group(1)}_proj.weight"
                result[key] = value[e].t().contiguous()
        else:
            result["mtp.layers.0." + moe_names.get(rest, rest)] = value.contiguous()
    return result


def validate_qwen35_target(config, target):
    if target.get("quantization_config"):
        raise ValueError("Quantized targets are not supported")
    target = target.get("text_config", target)
    moe = target.get("model_type") in ("qwen3_5_moe", "qwen3_5_moe_text")
    if not moe and target.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError("This MTP adapter requires a Qwen3.5 target")
    shared = ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    moe_fields = ("num_experts", "num_experts_per_tok", "moe_intermediate_size")
    for name in (*shared, *(moe_fields if moe else ("intermediate_size",))):
        if getattr(config, name) != target[name]:
            raise ValueError(f"Qwen3.5 target/draft mismatch: {name}")
    for name in ("vocab_size", "rms_norm_eps"):
        if getattr(config, name) != target[name]:
            raise ValueError(f"Qwen3.5 target/draft mismatch: {name}")
    if not config.tie_lm_head or config.attention_type != "gqa":
        raise ValueError("Qwen3.5 MTP requires GQA and frozen target LM head")
    if config.mlp_type != ("moe" if moe else "dense"):
        raise ValueError(f"Qwen3.5 MTP mlp_type must be {'moe' if moe else 'dense'}")
    if moe and (
        config.num_shared_experts * config.moe_intermediate_size
        != target["shared_expert_intermediate_size"]
        or not config.shared_expert_gate
        or (getattr(config, "moe_router_score_func", None) or "softmax") != "softmax"
        or config.moe_router_use_sigmoid
        or config.moe_router_enable_expert_bias
        or not config.route_norm
        or config.router_scaling_factor != 1.0
    ):
        raise ValueError(
            "Qwen3.5-MoE MTP routes by softmax over top-k, renormalized and unscaled, with no "
            "expert bias, and one gated shared expert of shared_expert_intermediate_size"
        )
    if config.rms_norm_type != "qwen3_5" or not config.qk_norm:
        raise ValueError("Qwen3.5 MTP requires native RMSNorm and Q/K normalization")
    if bool(config.attn_output_gate) != bool(target.get("attn_output_gate", True)):
        raise ValueError("Qwen3.5 attention gate mismatch")
    rope = target.get("rope_parameters", {})
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Only default Qwen3.5 RoPE is currently supported")
    if config.rope_theta != rope.get("rope_theta", target.get("rope_theta", 10000000.0)):
        raise ValueError("Qwen3.5 rope_theta mismatch")
    if config.partial_rotary_factor != rope.get(
        "partial_rotary_factor", target.get("partial_rotary_factor", 1.0)
    ):
        raise ValueError("Qwen3.5 partial RoPE mismatch")
    draft_rope = config.rope_parameters or {}
    for key in ("mrope_section", "mrope_interleaved"):
        if key in draft_rope and key in rope and draft_rope[key] != rope[key]:
            raise ValueError(f"Qwen3.5 RoPE metadata mismatch: {key}")
    if target.get("mtp_use_dedicated_embeddings", False) or target.get("attention_bias", False):
        raise ValueError("Dedicated MTP embeddings and attention bias are not supported")
    if target.get("quantization_config"):
        raise ValueError("Quantized MTP checkpoints are not supported")


def configure_qwen35_target(args, config, target_path):
    target_path = Path(target_path)
    metadata = json.loads((target_path / "config.json").read_text())
    if metadata.get("quantization_config"):
        raise ValueError("Quantized Qwen3.5 targets are not supported")
    validate_qwen35_target(config, metadata)
    if getattr(args, "vllm_enable_multimodal", False):
        raise ValueError(
            "Qwen3.5 MTP currently supports text positions only; disable multimodal input"
        )
    if getattr(args, "offline_manifest_path", None):
        raise ValueError("Offline Qwen3.5 MTP initialization is not validated yet")
    if not getattr(args, "mtp_draft_input_postnorm", False):
        raise ValueError("Qwen3.5 MTP requires mtp_draft_input_postnorm=true")
    nested = "text_config" in metadata
    args.embedding_key = (
        "model.language_model.embed_tokens.weight" if nested else "model.embed_tokens.weight"
    )
    args.norm_key = "model.language_model.norm.weight" if nested else "model.norm.weight"
    index = target_path / "model.safetensors.index.json"
    if index.is_file():
        keys = json.loads(index.read_text())["weight_map"]
    elif (target_path / "model.safetensors").is_file():
        with safe_open(target_path / "model.safetensors", framework="pt") as reader:
            keys = list(reader.keys())
    else:
        raise ValueError("Target must use HF safetensors weights")
    if "lm_head.weight" in keys:
        args.lm_head_key = "lm_head.weight"
    elif metadata.get("text_config", metadata).get("tie_word_embeddings", False):
        args.lm_head_key = args.embedding_key
    else:
        raise ValueError("Missing untied target LM head")
    for key in (args.embedding_key, args.norm_key, args.lm_head_key):
        if key not in keys:
            raise ValueError(f"Missing target component: {key}")


def initialize_qwen35_mtp(draft, args, config, target_path):
    resume = getattr(args, "load_path", None)
    external = getattr(args, "draft_init_path", None)
    native_layer = getattr(config, "mtp_init_from_mtp_layer", None)
    if native_layer is None:
        native_layer = getattr(args, "mtp_init_from_mtp_layer", None)
    decoder_layer = getattr(config, "mtp_init_from_target_layer", None)
    if decoder_layer is None:
        decoder_layer = getattr(args, "mtp_init_from_target_layer", None)
    if decoder_layer is not None:
        raise ValueError(
            "Use random initialization or native Qwen3.5 MTP weights; target-layer warm start is not wired"
        )
    if sum(bool(x) for x in (resume, external, native_layer is not None)) > 1:
        raise ValueError("Choose one of load_path, draft_init_path, mtp_init_from_mtp_layer")
    if resume:
        return "resume"
    if external:
        donor = Path(external)
        target_tokenizer = Path(target_path) / "tokenizer.json"
        donor_tokenizer = donor / "tokenizer.json"
        if target_tokenizer.is_file() and donor_tokenizer.is_file():
            target_tokens = json.loads(target_tokenizer.read_text())
            donor_tokens = json.loads(donor_tokenizer.read_text())

            def vocab(data):
                values = dict(data["model"]["vocab"])
                values.update(
                    {item["content"]: item["id"] for item in data.get("added_tokens", [])}
                )
                return values

            current, previous = vocab(target_tokens), vocab(donor_tokens)
            if any(
                current[token] != previous[token] for token in current.keys() & previous.keys()
            ):
                raise ValueError("Donor and target tokenizer IDs conflict")

        if not (donor / "config.json").is_file():
            raise ValueError("draft_init_path requires config.json to validate MTP semantics")
        donor_config = json.loads((donor / "config.json").read_text())
        generic = donor_config.get("model_type") == "mtp"
        if generic:
            from angelspec.models.draft.mtp import MTPConfig

            check = MTPConfig(**donor_config)
            for key in (
                "rms_norm_type",
                "attn_output_gate",
                "partial_rotary_factor",
                "rope_theta",
                "mlp_type",
                "attention_type",
                "qk_norm",
                "rms_norm_eps",
                "hidden_size",
                "target_hidden_size",
                "vocab_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "intermediate_size",
            ):
                if getattr(check, key) != getattr(config, key):
                    raise ValueError(f"Incompatible exported draft: {key}")
            state = read_weights(donor)
            state.pop("embed_tokens.weight", None)
        else:
            validate_qwen35_target(config, donor_config)
            state = native_to_draft(read_weights(donor, prefix="mtp."))
    elif native_layer is not None:
        if native_layer != 0:
            raise ValueError("Only one native MTP layer (index 0) is supported")
        state = native_to_draft(read_weights(target_path, prefix="mtp."))
    else:
        return "random"
    expected = {k: v for k, v in draft.state_dict().items() if k != "embed_tokens.weight"}
    if set(state) != set(expected):
        raise ValueError(
            f"Incomplete MTP weights: missing={sorted(set(expected) - set(state))}, unexpected={sorted(set(state) - set(expected))}"
        )
    for name, value in state.items():
        if tuple(value.shape) != tuple(expected[name].shape) or not value.is_floating_point():
            raise ValueError(f"MTP tensor mismatch: {name}")
    draft.load_state_dict({**state, "embed_tokens.weight": draft.embed_tokens.weight}, strict=True)
    return "warm_start" if external else "target_mtp"
