import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from angelspec.models.ops.tilelang_dsa import (
    DSA_INDEX_ALIGNMENT,
    DSA_KERNEL_DTYPE,
    tilelang_dsa_available,
    tilelang_lightning_indexer,
    tilelang_sparse_mla,
)


def _no_options(config: Any) -> Dict[str, Any]:
    return {}


def _always_safe(options: Dict[str, Any]) -> bool:
    return True


@dataclasses.dataclass(frozen=True)
class DSAKernel:
    """A sparse-MLA kernel and its lightning indexer, selected by ``dsa_backend``. Both take
    the arguments of :func:`tilelang_sparse_mla` and :func:`tilelang_lightning_indexer` plus
    the keyword arguments ``options`` reads from a draft config; ``recompute_safe`` says
    whether those options reproduce the forward under activation checkpointing."""

    name: str
    available: Callable[[], bool]
    sparse_mla: Callable[..., Any]
    lightning_indexer: Callable[..., Any]
    index_alignment: int = DSA_INDEX_ALIGNMENT
    dtype: torch.dtype = DSA_KERNEL_DTYPE
    options: Callable[[Any], Dict[str, Any]] = _no_options
    recompute_safe: Callable[[Dict[str, Any]], bool] = _always_safe


@dataclasses.dataclass(frozen=True)
class BoundDSAKernel:
    kernel: DSAKernel
    options: Dict[str, Any]

    @property
    def name(self) -> str:
        return self.kernel.name

    @property
    def dtype(self) -> torch.dtype:
        return self.kernel.dtype

    @property
    def index_alignment(self) -> int:
        return self.kernel.index_alignment

    @property
    def recompute_safe(self) -> bool:
        return self.kernel.recompute_safe(self.options)

    def sparse_mla(self, *args, **kwargs):
        return self.kernel.sparse_mla(*args, **kwargs, **self.options)

    def lightning_indexer(self, *args, **kwargs):
        return self.kernel.lightning_indexer(*args, **kwargs, **self.options)


_KERNELS: Dict[str, DSAKernel] = {}


def register_dsa_kernel(kernel: DSAKernel) -> None:
    _KERNELS[kernel.name] = kernel


def dsa_kernel_names() -> Tuple[str, ...]:
    from angelspec.plugins import load_plugins

    load_plugins()
    return tuple(_KERNELS)


def is_dsa_kernel(name: Optional[str]) -> bool:
    return name is not None and name in dsa_kernel_names()


def bind_dsa_kernel(name: Optional[str], config: Any) -> Optional[BoundDSAKernel]:
    """``name``'s kernel with ``config``'s options, or None when ``name`` is no DSA kernel."""
    if not is_dsa_kernel(name):
        return None
    kernel = _KERNELS[name]
    return BoundDSAKernel(kernel, kernel.options(config))


register_dsa_kernel(
    DSAKernel("tilelang", tilelang_dsa_available, tilelang_sparse_mla, tilelang_lightning_indexer)
)
