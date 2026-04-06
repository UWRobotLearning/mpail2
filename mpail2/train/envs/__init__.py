"""Auto-discover and import env modules so Hydra ConfigStore nodes register on import."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def discover_env_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for mod in pkgutil.iter_modules(__path__):
        name = mod.name
        if name == "__init__" or name.startswith("_"):
            continue
        modules.append(importlib.import_module(f"{__name__}.{name}"))
    return modules


# Import once at package import time for Hydra ConfigStore side effects.
_DISCOVERED_ENV_MODULES = discover_env_modules()


__all__ = ["discover_env_modules"]
