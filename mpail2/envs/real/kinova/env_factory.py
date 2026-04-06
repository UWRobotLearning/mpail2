"""Factory for the Kinova real stack (mirrors ``franka.env_factory``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .real_manipulation import KinovaManipulationEnv
from .robot_limits import ACTION_DIM, MAX_EPISODE_STEPS, STATE_DIM
from .wrappers import KinovaRealWrapper


@dataclass
class KinovaRealEnvArgs:
    device: str = "cuda"
    control_hz: float = 10.0
    max_episode_length: int = MAX_EPISODE_STEPS
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM


def make_kinova_env(args: KinovaRealEnvArgs | Any):
    base = KinovaManipulationEnv(
        control_frequency=args.control_hz,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        device=args.device,
        max_episode_steps=args.max_episode_length,
    )
    return KinovaRealWrapper(base, device=args.device)


def create_real_manipulation_env(
    control_frequency: float = 10.0,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    device: str = "cuda",
    max_episode_length: int = MAX_EPISODE_STEPS,
) -> KinovaRealWrapper:
    """Backward-compatible alias; prefer :func:`make_kinova_env`."""
    return make_kinova_env(
        KinovaRealEnvArgs(
            device=device,
            control_hz=control_frequency,
            max_episode_length=max_episode_length,
            state_dim=state_dim,
            action_dim=action_dim,
        )
    )
