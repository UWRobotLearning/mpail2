"""Shared demonstration path resolution helpers for environment stacks."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def find_demo_file(
    env_id: str,
    demo_dir: Optional[str] = None,
    env_stacks: Optional[Sequence[str]] = None,
    extra_patterns: Optional[Sequence[str]] = None,
) -> str:
    """Auto-detect a demonstration file for an environment id.

    Search order:
    1) user-provided demo_dir
    2) stack-local demos under mpail2/envs/<stack>
    3) optional caller-provided extra patterns
    """
    root = _project_root()
    stacks = tuple(env_stacks) if env_stacks else ("gym_mujoco", "isaac", "real")

    search_patterns: list[str] = []
    if demo_dir:
        search_patterns.append(os.path.join(demo_dir, env_id, "**", "*.pt"))
        search_patterns.append(os.path.join(demo_dir, env_id, "*.pt"))

    for stack in stacks:
        stack_root = root / "mpail2" / "envs" / stack
        search_patterns.extend(
            [
                str(stack_root / "demos" / env_id / "*" / "expert*.pt"),
                str(stack_root / "demos" / env_id / "*.pt"),
                str(stack_root / "**" / "demos" / env_id / "*" / "expert*.pt"),
                str(stack_root / "**" / "demos" / env_id / "*.pt"),
            ]
        )

    if extra_patterns:
        search_patterns.extend(extra_patterns)

    search_patterns = _dedupe(search_patterns)

    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            demo_path = sorted(matches)[-1]
            print(f"[INFO] Auto-detected demo file: {demo_path}")
            return demo_path

    raise FileNotFoundError(
        f"No demonstration file found for environment '{env_id}'.\n"
        f"Searched patterns:\n"
        + "\n".join(f"  - {p}" for p in search_patterns)
        + "\n\nPlease provide a demo file with --demo-path or place demos under:\n"
        + f"  {root / 'mpail2' / 'envs' / '<stack>' / 'demos' / env_id}"
    )


def resolve_existing_demo_path(
    path: str,
    search_dirs: Optional[Sequence[str]] = None,
) -> str:
    """Resolve a demo file path against common run locations and search roots."""
    raw = Path(path)
    if raw.is_absolute() and raw.exists():
        return str(raw)

    root = _project_root()

    candidates: list[str] = [str(raw)]
    if not raw.is_absolute():
        candidates.append(str(root / raw))

    for directory in search_dirs or ():
        base = Path(directory)
        candidates.append(str(base / raw))
        candidates.append(str(base / raw.name))

    candidates = _dedupe(candidates)

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(f"Demonstration file not found: {path}\nTried:\n{tried}")


def resolve_demo_path(
    cfg_path: Optional[str],
    env_var_key: Optional[str],
    default_path: str,
    search_dirs: Optional[Sequence[str]] = None,
) -> str:
    """Prefer explicit config, then env var, then default; return an existing path."""
    candidate = cfg_path
    if not candidate and env_var_key:
        candidate = os.environ.get(env_var_key)
    if not candidate:
        candidate = default_path

    return resolve_existing_demo_path(candidate, search_dirs=search_dirs)
