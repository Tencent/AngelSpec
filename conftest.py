# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Root conftest: stubs heavy dependencies when they are not installed.

When torch is not available (e.g. Mac dev machine), installs:
1. A meta-path finder that mocks torch/mooncake/transformers/etc.
2. A lightweight ``angelspec`` package stub so submodule imports don't
   trigger ``angelspec/__init__.py``'s eager model imports.
"""

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _is_available(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


class _MockFinder(importlib.abc.MetaPathFinder):
    """Returns mock modules for missing heavy deps."""

    PREFIXES = (
        "torch",
        "mooncake",
        "transformers",
        "flash_attn",
        "triton",
        "vllm",
        "sglang",
        "numba",
        "ray",
        "omegaconf",
        "openai",
        "huggingface_hub",
        "safetensors",
        "accelerate",
        "peft",
        "wandb",
        "datasets",
        "tokenizers",
        "sentencepiece",
        "pyzmq",
        "zmq",
    )

    def find_spec(self, fullname, path, target=None):
        for prefix in self.PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        class _Proxy(ModuleType):
            def __getattr__(self, name):
                return MagicMock()

        proxy = _Proxy(spec.name)
        proxy.__path__ = []
        proxy.__package__ = spec.name
        return proxy

    def exec_module(self, module):
        pass


if not _is_available("torch"):
    # 1. Install the mock finder for heavy deps
    sys.meta_path.insert(0, _MockFinder())

    # 2. Pre-seed angelspec as a namespace package so that
    #    ``from angelspec.config.mooncake_config import ...`` does NOT
    #    trigger angelspec/__init__.py (which eagerly imports models → torch).
    _root = os.path.dirname(os.path.abspath(__file__))
    _pkg = ModuleType("angelspec")
    _pkg.__path__ = [os.path.join(_root, "angelspec")]
    _pkg.__package__ = "angelspec"
    _pkg.__file__ = os.path.join(_root, "angelspec", "__init__.py")
    sys.modules["angelspec"] = _pkg
