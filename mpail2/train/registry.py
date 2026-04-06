"""Dynamic train-env registry built from metadata in ``mpail2.train.envs.*`` modules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any, Mapping

import mpail2.train.envs as envs


@dataclass(frozen=True)
class TrainEnvSpec:
    config_name: str
    demo_env_var: str
    default_demo_rel: str
    default_num_iterations: int
    suite: str = "isaac_image"


def _coerce_spec(module: ModuleType, raw: Any) -> TrainEnvSpec:
    if isinstance(raw, TrainEnvSpec):
        return raw
    if isinstance(raw, Mapping):
        return TrainEnvSpec(**raw)
    raise TypeError(
        f"{module.__name__}: ENV_SPEC must be TrainEnvSpec or mapping, got {type(raw)}"
    )


@lru_cache(maxsize=1)
def _build_registry() -> tuple[dict[str, TrainEnvSpec], dict[str, str]]:
    registry: dict[str, TrainEnvSpec] = {}
    aliases: dict[str, str] = {}

    for module in envs.discover_env_modules():
        env_name = getattr(module, "ENV_NAME", None)
        spec_raw = getattr(module, "ENV_SPEC", None)
        if env_name is None or spec_raw is None:
            # Backward-compatible: modules without metadata are ignored.
            continue

        spec = _coerce_spec(module, spec_raw)
        extra_aliases = tuple(getattr(module, "ENV_ALIASES", ()))
        names = (env_name, *extra_aliases)

        if env_name in registry:
            raise ValueError(f"Duplicate ENV_NAME {env_name!r} found in {module.__name__}")
        registry[env_name] = spec

        for token in names:
            existing = aliases.get(token)
            if existing is not None and existing != env_name:
                raise ValueError(
                    f"Alias collision for {token!r}: {existing!r} vs {env_name!r} (module {module.__name__})"
                )
            aliases[token] = env_name

    if not registry:
        raise RuntimeError(
            "No env modules with ENV_NAME/ENV_SPEC were discovered under mpail2.train.envs"
        )
    return registry, aliases


def resolve_train_env(name: str) -> TrainEnvSpec:
    registry, aliases = _build_registry()
    key = aliases.get(name)
    if key is None:
        supported = ", ".join(sorted(aliases.keys()))
        raise ValueError(f"Unknown --env {name!r}. Supported values: {supported}")
    return registry[key]
