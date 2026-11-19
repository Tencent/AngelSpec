"""DFlash-family draft attention blocks, chosen by ``config.attention_type``.

Each target family's attention lives in its own module and is registered here by name
as a ``"module:builder"`` path, resolved on first use. Adding a family touches neither
the drafter nor another family's module, and a config only imports the stack it names.
A builder is called with ``(config, layer_idx)`` and returns a module taking the
DFlash decoder layer's attention arguments.
"""

import importlib
from typing import Dict, Tuple

_BUILDERS: Dict[str, str] = {
    "gqa": "angelspec.models.draft.dflash:build_gqa_attention",
    "mla": "angelspec.models.draft.hy4_mla:build_hy4_self_attn",
}


def register_draft_attention(name: str, builder: str) -> None:
    """Register ``builder`` (``"module:function"``) as attention type ``name``."""
    if _BUILDERS.get(name, builder) != builder:
        raise ValueError(f"attention_type {name!r} is already registered as {_BUILDERS[name]}")
    _BUILDERS[name] = builder


def draft_attention_types() -> Tuple[str, ...]:
    return tuple(_BUILDERS)


def _family(config) -> Tuple[str, str]:
    name = str(getattr(config, "attention_type", "gqa"))
    if name not in _BUILDERS:
        raise ValueError(f"attention_type must be one of {draft_attention_types()}, got {name!r}")
    module_name, _, builder = _BUILDERS[name].partition(":")
    return module_name, builder


def build_draft_attention(config, layer_idx: int):
    module_name, builder = _family(config)
    return getattr(importlib.import_module(module_name), builder)(config, layer_idx)


def resolve_draft_dsa_backend(config) -> str:
    """``config.dsa_backend`` as ``config``'s attention family runs it in this process: its
    module's ``resolve_dsa_backend``, or "torch" for a family without one."""
    resolve = getattr(importlib.import_module(_family(config)[0]), "resolve_dsa_backend", None)
    return "torch" if resolve is None else resolve(config)
