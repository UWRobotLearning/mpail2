"""Shared helpers for applying Hydra-style nested dict overrides."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any


def prune_placeholder_overrides(node: Any) -> Any:
    """Drop placeholder values (``None`` / empty containers) from override trees."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            pruned = prune_placeholder_overrides(value)
            if pruned is not None:
                out[key] = pruned
        return out or None
    if isinstance(node, list):
        return node if node else None
    return None if node is None else node


def apply_dataclass_overrides(node: Any, overrides: dict[str, Any], path: str = "cfg") -> None:
    """Recursively apply ``dict`` overrides to dataclass-like config objects."""
    for key, value in overrides.items():
        if not hasattr(node, key):
            raise KeyError(f"Unknown override key: {path}.{key}")
        current = getattr(node, key)
        if isinstance(value, dict):
            if is_dataclass(current):
                apply_dataclass_overrides(current, value, path=f"{path}.{key}")
            elif isinstance(current, dict):
                current.update(value)
            else:
                raise TypeError(
                    f"Cannot apply nested override at {path}.{key}: target is {type(current).__name__}"
                )
        else:
            setattr(node, key, value)


def to_override_template(node: Any) -> Any:
    """Convert dataclass-like tree to Hydra-friendly override schema."""
    if is_dataclass(node):
        out: dict[str, Any] = {}
        for f in fields(node):
            child = getattr(node, f.name)
            out[f.name] = to_override_template(child)
        return out
    if isinstance(node, dict):
        return {}
    if isinstance(node, list):
        return []
    # Keep leaves as None so users can override without '+' syntax.
    return None

