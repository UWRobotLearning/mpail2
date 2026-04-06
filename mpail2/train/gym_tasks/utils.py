"""Utility functions for gym_tasks training scripts."""

import os
from typing import Optional

from mpail2.envs import find_demo_file as _find_demo_file


def find_demo_file(env_id: str, demo_dir: Optional[str] = None) -> str:
    """
    Auto-detect demonstration file for a given environment.

    Searches in this order:
    1) user-provided --demo-dir
    2) envs/gym_mujoco/demos
    3) local script-relative demos fallback
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extra_patterns = [
        os.path.join(script_dir, "gym", "demos", env_id, "*", "expert*.pt"),
        os.path.join(script_dir, "gym", "demos", env_id, "*.pt"),
    ]
    return _find_demo_file(
        env_id=env_id,
        demo_dir=demo_dir,
        env_stacks=("gym_mujoco",),
        extra_patterns=extra_patterns,
    )
