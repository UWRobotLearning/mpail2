"""Factory for the SO-101 real-robot stack (mirrors kinova.env_factory)."""

from __future__ import annotations

from dataclasses import dataclass

from .robot_limits import (
    ACTION_DIM,
    CONTROL_FREQUENCY_HZ,
    DEFAULT_ROBOT_HOST,
    DEFAULT_ROBOT_PORT,
    MAX_EPISODE_STEPS,
    STATE_DIM,
)
from .so101_env import SO101RobotEnv
from .wrappers import SO101RealWrapper


@dataclass
class SO101RealEnvArgs:
    device: str = "cuda"
    host: str = DEFAULT_ROBOT_HOST
    port: int = DEFAULT_ROBOT_PORT
    control_hz: float = CONTROL_FREQUENCY_HZ
    max_episode_length: int = MAX_EPISODE_STEPS
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    mock: bool = False
    speed_scale: float = 1.0
    lpf_alpha: float = 1.0
    reset_pause_seconds: float = 3.0
    gripper_hold_steps: int = 1


def make_so101_env(args: SO101RealEnvArgs | None = None) -> SO101RealWrapper:
    """Build a fully wrapped SO-101 env ready for MPAIL2Runner."""
    if args is None:
        args = SO101RealEnvArgs()
    base = SO101RobotEnv(
        host=args.host,
        port=args.port,
        control_frequency=args.control_hz,
        max_episode_steps=args.max_episode_length,
        mock=args.mock,
        speed_scale=args.speed_scale,
        lpf_alpha=args.lpf_alpha,
        reset_pause_seconds=args.reset_pause_seconds,
        gripper_hold_steps=args.gripper_hold_steps,
    )
    return SO101RealWrapper(base, device=args.device)
