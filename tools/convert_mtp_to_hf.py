"""Convert a trained single-head MTP draft (DCP checkpoint) to HuggingFace
safetensors aligned with the Hy3 ``model.layers.<N>.*`` MTP layer layout,
so the result loads into vLLM ``hy_v3_mtp`` for speculative decoding.

The training checkpoint is a PyTorch Distributed Checkpoint (DCP) whose tensor
keys follow ``MTPDraftModel``'s state_dict:

    draft_model.enorm/hnorm/eh_proj/final_layernorm.weight
    draft_model.midlayer.{input_layernorm,post_attention_layernorm}.weight
    draft_model.midlayer.self_attn.{q,k,v,o}_proj.weight, {q,k}_norm.weight
    draft_model.midlayer.mlp.router.gate.weight, mlp.expert_bias
    draft_model.midlayer.mlp.shared_mlp.{gate,up,down}_proj.weight
    draft_model.midlayer.mlp.experts_{gate,up,down}_proj   # FUSED [E, in, out]
    draft_model.embed_tokens.weight                        # frozen, == target
    draft_model.lm_head_weight                             # tied, == target head

vLLM (``hy_v3_mtp.load_weights`` + ``get_spec_layer_idx_from_weight_name``)
expects the trained MTP block under the HF checkpoint prefix
``model.layers.<num_hidden_layers>.*`` with PER-EXPERT weights:

    model.layers.<N>.{enorm,hnorm,eh_proj,final_layernorm}.weight
    model.layers.<N>.{input_layernorm,post_attention_layernorm}.weight
    model.layers.<N>.self_attn.{q,k,v,o}_proj.weight, {q,k}_norm.weight
    model.layers.<N>.mlp.router.gate.weight, mlp.expert_bias
    model.layers.<N>.mlp.shared_mlp.{gate,up,down}_proj.weight
    model.layers.<N>.mlp.experts.<e>.{gate,up,down}_proj.weight  # PER-EXPERT [out, in]

This is the exact inverse of the trainer's CPT loader (``_load_mtp_from_target``):
the fused ``[E, in, out]`` params are sliced per expert and transposed back to the
``nn.Linear`` ``[out, in]`` layout.

embed_tokens and lm_head are NOT emitted: in V3 MTP, vLLM takes embed_tokens and
lm_head from the MAIN model, and our draft's copies are frozen (== target) anyway.

Usage:
    python tools/convert_mtp_to_hf.py \
        --input-dir outputs/qwen3-8b-mtp/checkpoints/iter_0000021 \
        --target-config /path/to/target-model/config.json \
        --output-dir outputs/qwen3-8b-mtp/iter_0000021_hf
"""

import argparse
import json
import logging
import os
import re
import time
from typing import Optional

import torch
import torch.distributed.checkpoint as dist_cp
from safetensors.torch import save_file
from typing_extensions import override

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "version.txt")

# MTP fusion modules that stay at the spec-layer top level (vLLM keeps them on
# HYV3MultiTokenPredictorLayer, not inside .mtp_block).
_TOP_LEVEL = ("enorm", "hnorm", "eh_proj", "final_layernorm")

# midlayer.* (our decoder block) maps to the bare spec-layer sub-name; vLLM's
# _rewrite_spec_layer_name re-inserts the .mtp_block. itself at load time.
_MIDLAYER_PREFIX = "midlayer."


def _get_version() -> str:
    try:
        with open(_VERSION_FILE) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


# ── DCP loading (no torch.distributed; single-process full read) ──────────────


class _EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    """Materialise every (non-optimizer) tensor from the DCP metadata so a full
    state dict can be read without knowing the keys up front."""

    @override
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optim" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                state_dict[k] = torch.empty(v.size, dtype=v.properties.dtype)
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _load_dcp_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=dist_cp.FileSystemReader(model_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _strip_to_draft(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only draft tensors, returning ``<sub> -> tensor``.

    Handles two layouts:
      - full training checkpoint keys ``...draft_model.<sub>`` (DCP path), and
      - a draft-only state_dict already rooted at the draft model (bin path,
        e.g. from ``Trainer.save_draft_model_for_serving``), whose keys are the
        bare ``<sub>`` (``enorm``/``hnorm``/``eh_proj``/``midlayer.*``/…).
    """
    tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    prefixed = {k.split("draft_model.")[-1]: v for k, v in tensors.items() if "draft_model." in k}
    if prefixed:
        return prefixed
    # No wrapper prefix — assume the dict is already draft-rooted. Recognise it by
    # the MTP fusion modules so we don't silently accept an unrelated state_dict.
    markers = ("enorm", "hnorm", "eh_proj", "midlayer.", "final_layernorm")
    if any(any(k.startswith(m) or f".{m}" in k for m in markers) for k in tensors):
        return tensors
    return {}


# ── fused → per-expert + key remap ────────────────────────────────────────────


def _remap_to_hf(
    draft_state: dict[str, torch.Tensor], spec_layer: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{spec_layer}."
    fused_pat = re.compile(r"^midlayer\.mlp\.experts_(gate|up|down)_proj$")
    out: dict[str, torch.Tensor] = {}

    for k, v in draft_state.items():
        # embed_tokens / lm_head come from the main model; drop our frozen copies.
        if k.startswith("embed_tokens") or k == "lm_head_weight" or k.startswith("lm_head"):
            continue

        m = fused_pat.match(k)
        if m is not None:
            # FUSED [E, in, out] → per-expert [out, in] (inverse of CPT loader).
            proj = m.group(1) + "_proj"  # gate_proj / up_proj / down_proj
            fused = v
            n_e = fused.shape[0]
            for e in range(n_e):
                out[f"{prefix}mlp.experts.{e}.{proj}.weight"] = fused[e].t().contiguous().to(dtype)
            continue

        top = k.split(".")[0]
        if top in _TOP_LEVEL:
            out[f"{prefix}{k}"] = v.to(dtype)
        elif k.startswith(_MIDLAYER_PREFIX):
            sub = k[len(_MIDLAYER_PREFIX) :]
            out[f"{prefix}{sub}"] = v.to(dtype)
        else:
            logger.warning("Unrecognised draft key, emitting as-is under layer prefix: %s", k)
            out[f"{prefix}{k}"] = v.to(dtype)

    return out


def _detect_model_dir(input_dir: str) -> str:
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _resolve_spec_layer(
    target_config: Optional[str], spec_layer_arg: Optional[int]
) -> tuple[int, dict]:
    """Determine the MTP spec-layer index (== num_hidden_layers) and carry the
    target config forward so the exported config.json matches what vLLM expects."""
    cfg = {}
    if target_config:
        with open(target_config) as f:
            cfg = json.load(f)
    if spec_layer_arg is not None:
        return spec_layer_arg, cfg
    if "num_hidden_layers" in cfg:
        return int(cfg["num_hidden_layers"]), cfg
    raise ValueError(
        "Cannot determine spec layer index. Pass --spec-layer N or "
        "--target-config <target config.json with num_hidden_layers>."
    )


def convert(
    input_dir: str,
    output_dir: str,
    target_config: Optional[str],
    spec_layer_arg: Optional[int],
    dtype: torch.dtype,
    force: bool,
) -> None:
    if os.path.exists(output_dir) and not force:
        raise ValueError(f"Output directory {output_dir} already exists. Use -f to overwrite.")

    spec_layer, target_cfg = _resolve_spec_layer(target_config, spec_layer_arg)
    logger.info("MTP spec layer index = %d", spec_layer)

    model_dir = _detect_model_dir(input_dir)
    # Accept either a DCP checkpoint dir (training checkpoint) or a plain
    # draft-only state_dict saved as pytorch_model.bin (save_draft_model_for_serving).
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.isdir(model_dir) and os.path.isfile(input_dir):
        bin_path = input_dir
    t = time.time()
    if os.path.isfile(bin_path):
        logger.info("Loading draft state_dict from %s", bin_path)
        raw = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        logger.info("Loading DCP checkpoint from %s", model_dir)
        raw = _load_dcp_state_dict(model_dir)
    draft_state = _strip_to_draft(raw)
    logger.info(
        "Loaded %d draft tensors in %.1fs (%d total checkpoint keys)",
        len(draft_state),
        time.time() - t,
        len(raw),
    )
    if not draft_state:
        raise ValueError(
            "No draft_model.* tensors found. Pass an MTP checkpoint dir (iter_xxxxxxx or iter_xxxxxxx/model)."
        )

    hf_tensors = _remap_to_hf(draft_state, spec_layer, dtype)
    n_experts = sum(1 for k in hf_tensors if re.search(r"mlp\.experts\.\d+\.gate_proj", k))
    logger.info(
        "Remapped to %d HF tensors (%d experts, per-expert [out,in] layout, dtype=%s)",
        len(hf_tensors),
        n_experts,
        dtype,
    )

    os.makedirs(output_dir, exist_ok=True)
    version = _get_version()
    save_file(
        hf_tensors,
        os.path.join(output_dir, "model.safetensors"),
        metadata={"angelspec_version": version, "format": "pt"},
    )

    # Emit a config.json that carries the target's MTP-relevant fields so vLLM's
    # hy_v3_mtp can resolve num_hidden_layers / num_nextn_predict_layers / expert
    # count. Prefer the target config verbatim (it already describes the MoE).
    export_cfg = dict(target_cfg) if target_cfg else {}
    export_cfg["_angelspec_version"] = version
    export_cfg["_mtp_spec_layer"] = spec_layer
    export_cfg["torch_dtype"] = str(dtype).replace("torch.", "")
    # vLLM enables MTP only when num_nextn_predict_layers > 0; the trained head
    # is 1 layer. Force to 1 whenever the carried-over target value is missing or
    # <= 0 — some targets ship num_nextn_predict_layers=0 (no native MTP head),
    # so a plain "if not present" check would leave it 0 and silently disable
    # MTP at serve time.
    if int(export_cfg.get("num_nextn_predict_layers", 0)) <= 0:
        export_cfg["num_nextn_predict_layers"] = 1
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(export_cfg, f, indent=2)

    logger.info("Saved MTP HF checkpoint to %s", output_dir)
    logger.info(
        "  keys under model.layers.%d.* — load into vLLM hy_v3_mtp alongside the "
        "main target model (embed_tokens + lm_head come from the target).",
        spec_layer,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a trained MTP DCP checkpoint to HF safetensors "
        "(HY-v3 model.layers.<N>.* per-expert layout for vLLM hy_v3_mtp)."
    )
    p.add_argument(
        "--input-dir", required=True, help="MTP checkpoint dir (iter_xxx or iter_xxx/model)"
    )
    p.add_argument("--output-dir", default=None, help="Output dir (default: {input_dir}_hf)")
    p.add_argument(
        "--target-config",
        default=None,
        help="Target model config.json (provides num_hidden_layers → spec layer, "
        "and is carried into the exported config.json).",
    )
    p.add_argument(
        "--spec-layer",
        type=int,
        default=None,
        help="MTP spec layer index (== target num_hidden_layers, e.g. 80 for A20B). "
        "Overrides the value derived from --target-config.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Output dtype (default: bfloat16, matching the HY checkpoint).",
    )
    p.add_argument("-f", "--force", action="store_true", help="Overwrite output dir if it exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_dir = args.input_dir.rstrip("/")
    output_dir = args.output_dir or f"{input_dir}_hf"
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    convert(
        input_dir=input_dir,
        output_dir=output_dir,
        target_config=args.target_config,
        spec_layer_arg=args.spec_layer,
        dtype=dtype_map[args.dtype],
        force=args.force,
    )
