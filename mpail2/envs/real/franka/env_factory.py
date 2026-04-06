"""FrankaClient + RealSense + MPAIL-shaped gym wrapper stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .network import FrankaClient

from .robot_limits import (
    ACTION_DIM,
    LOWER_LIMITS,
    RESET_QPOS,
    STATE_DIM,
    TABLE_CAM_SERIAL,
    UPPER_LIMITS,
    WRIST_CAM_SERIAL,
)
from .wrappers import FrankaRealWrapper, RealSenseWrapper


def rename_camera_keys(serial_number: str) -> str:
    if serial_number == TABLE_CAM_SERIAL:
        return "table_cam"
    if serial_number == WRIST_CAM_SERIAL:
        return "wrist_cam"
    raise ValueError(f"Unknown camera serial: {serial_number}")


@dataclass
class FrankaRealEnvArgs:
    server_address: str
    device: str = "cuda"
    dynamics_factor: float = 1.0
    control_hz: float = 15.0
    max_episode_length: int = 120


def make_franka_env(args: FrankaRealEnvArgs | Any):
    """``args`` must provide server_address, device, dynamics_factor, control_hz, max_episode_length."""
    env = FrankaClient(
        server_address=args.server_address,
        dynamics_factor=args.dynamics_factor,
        control_hz=args.control_hz,
        reset_qpos=RESET_QPOS,
    )
    env = RealSenseWrapper(
        env,
        center_crop=True,
        resize=(64, 64),
        normalize=False,
        rename_camera_keys=rename_camera_keys,
        serial_numbers=[TABLE_CAM_SERIAL, WRIST_CAM_SERIAL],
    )
    env = FrankaRealWrapper(
        env,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_episode_length=args.max_episode_length,
        lower_limits=LOWER_LIMITS,
        upper_limits=UPPER_LIMITS,
        use_image=True,
        device=args.device,
    )
    return env
