"""Every module in this package is imported before the first DSA kernel or HF config lookup,
so a deployment registers its own kernels and configs by adding a module here."""

import importlib
import pkgutil

_loaded = False


def load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{module.name}")
