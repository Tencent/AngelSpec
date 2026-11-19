#!/usr/bin/env python3
"""Assemble a vLLM-servable target dir whose MTP head (model.mtp_layers.0.*) is
replaced by an AngelSpec-trained MLA+DSA MTP draft.

The draft (hf_final, MTPDraftModel internal naming) is remapped back to the
target's native model.mtp_layers.0.* layout (inverse of MTPTrainer's
_remap_mtp_layer_state): fused experts.gate_up_proj [E, 2I, H] + down_proj
[E, H, I], DeepSeek MoE names, self_attn.* passthrough.

Output dir symlinks every target file, adds mtp_override.safetensors holding the
new head, and rewrites model.safetensors.index.json so the 27 mtp_layers.0 keys
point at the override (the stale copies in the original shards become unreferenced).
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file


def remap_draft_to_mtp_layer(draft, layer=0):
    p = f"model.mtp_layers.{layer}."
    out = {}
    top = {"enorm", "hnorm", "eh_proj", "final_layernorm"}
    for k, v in draft.items():
        if k in ("embed_tokens.weight", "lm_head_weight"):
            continue  # shared with target, not part of the head
        if k in {f"{t}.weight" for t in top}:
            out[p + k] = v
        elif k in (
            "midlayer.mlp.experts_gate_proj",
            "midlayer.mlp.experts_gate_proj.weight",
        ):  # [E, H, I] -> gate half of [E, 2I, H]
            gate = v.transpose(1, 2).contiguous()  # [E, I, H]
            out["_gate"] = gate
        elif k in (
            "midlayer.mlp.experts_up_proj",
            "midlayer.mlp.experts_up_proj.weight",
        ):
            up = v.transpose(1, 2).contiguous()  # [E, I, H]
            out["_up"] = up
        elif k in (
            "midlayer.mlp.experts_down_proj",
            "midlayer.mlp.experts_down_proj.weight",
        ):  # [E, I, H] -> [E, H, I]
            out[p + "mlp.experts.down_proj"] = v.transpose(1, 2).contiguous()
        elif k == "midlayer.mlp.router.gate.weight":
            out[p + "mlp.gate.weight"] = v
        elif k == "midlayer.mlp.expert_bias":
            out[p + "mlp.gate.e_score_correction_bias"] = v
        elif k.startswith("midlayer.mlp.shared_mlp."):
            out[p + "mlp.shared_experts." + k[len("midlayer.mlp.shared_mlp.") :]] = v
        elif k.startswith("midlayer."):
            out[p + k[len("midlayer.") :]] = v
        else:
            raise KeyError(f"unmapped draft key {k}")
    # fuse gate+up -> [E, 2I, H] (gate rows first, matching target)
    gate, up = out.pop("_gate"), out.pop("_up")
    out[p + "mlp.experts.gate_up_proj"] = torch.cat([gate, up], dim=1).contiguous()
    return out


_SAFETENSORS_DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}


def cast_like_target(head: dict, target: str) -> dict:
    """Store each tensor in the dtype the target checkpoint uses for that key
    (F32 for e.g. ``e_score_correction_bias``), else as produced."""
    from safetensors import safe_open

    wmap = json.load(open(os.path.join(target, "model.safetensors.index.json")))["weight_map"]
    dtypes = {}
    by_file = {}
    for k in head:
        if k in wmap:
            by_file.setdefault(wmap[k], []).append(k)
    for fn, keys in by_file.items():
        with safe_open(os.path.join(target, fn), framework="pt") as f:
            for k in keys:
                dtypes[k] = _SAFETENSORS_DTYPES[f.get_slice(k).get_dtype()]
    return {k: v.to(dtypes.get(k, v.dtype)).contiguous() for k, v in head.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True, help="hf_final dir with pytorch_model.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    draft = torch.load(
        os.path.join(args.draft, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    head = cast_like_target(remap_draft_to_mtp_layer(draft, args.layer), args.target)
    print(f"remapped {len(head)} mtp_layers.{args.layer} tensors")

    override = "mtp_override.safetensors"
    save_file(head, os.path.join(args.out, override), metadata={"format": "pt"})

    idx = json.load(open(os.path.join(args.target, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    prefix = f"model.mtp_layers.{args.layer}."
    # repoint every mtp_layers.<layer>.* key to the override file
    for k in list(wmap):
        if k.startswith(prefix):
            wmap[k] = override
    for k in head:
        wmap[k] = override
    # symlink all original files (shards, tokenizer, chat template) except index/config
    for fn in os.listdir(args.target):
        if fn in ("model.safetensors.index.json", "config.json") or fn.startswith("."):
            continue
        src = os.path.join(args.target, fn)
        dst = os.path.join(args.out, fn)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    json.dump(idx, open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(args.target, "config.json")))
    cfg["num_nextn_predict_layers"] = max(1, cfg.get("num_nextn_predict_layers", 0))
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print(f"served dir ready: {args.out}")


if __name__ == "__main__":
    main()
