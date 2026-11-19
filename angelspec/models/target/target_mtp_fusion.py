"""Draft MTP fusion initialized from the target's MTP layer (``DFlashConfig.mtp_fusion_init``).

hy4's MTP layer opens with ``eh_proj(enorm(E(x_{t+1})) :: hnorm(h_t))``, embedding first,
ahead of its decoder block (vLLM ``HYV4MultiTokenPredictorLayer``). The drafter's
:class:`~angelspec.models.draft.dflash.MTPFusion` is that opening name for name and shape
for shape, so copying the target's first MTP layer makes the drafter's fusion start out as
the target's own; the gate is the drafter's and stays at zero.

The copy runs once, on the rank holding the real weights before FSDP broadcasts them; a
``draft_init_path`` carrying fusion weights, or a resumed checkpoint, loaded afterwards
overwrites it.
"""

from typing import Dict

import torch.nn as nn

from angelspec.models.target.target_indexer import _text_config, load_layer_tensors

#: MTPFusion parameter -> checkpoint key below ``<prefix>mtp_layers.0.``.
_CHECKPOINT_KEYS: Dict[str, str] = {
    "enorm.weight": "enorm.weight",
    "hnorm.weight": "hnorm.weight",
    "eh_proj.weight": "eh_proj.weight",
}
_ANCHOR = "eh_proj.weight"


def check_target_mtp_fusion_compatible(config: dict, draft_config) -> None:
    """Raise unless the target has an MTP layer whose fusion has the drafter's shapes."""
    text = _text_config(config)
    if int(text.get("num_nextn_predict_layers") or 0) < 1:
        raise ValueError(
            "mtp_fusion_init='target' copies the target's MTP layer, but its config has no "
            "num_nextn_predict_layers; use mtp_fusion_init='random'"
        )
    mismatched = [
        f"{name}: target {text.get(name)!r} vs drafter {getattr(draft_config, name, None)!r}"
        for name in ("hidden_size", "rms_norm_eps")
        if text.get(name) != getattr(draft_config, name, None)
    ]
    if mismatched:
        raise ValueError(
            "mtp_fusion_init='target' copies the target's MTP fusion, so these must match: "
            + "; ".join(mismatched)
        )


def init_mtp_fusion_from_target(fusion: nn.Module, model_path: str) -> None:
    """Copy the target's first MTP layer's ``enorm`` / ``hnorm`` / ``eh_proj`` into ``fusion``."""
    load_layer_tensors(fusion, model_path, 0, _CHECKPOINT_KEYS, _ANCHOR, stack="mtp_layers")
