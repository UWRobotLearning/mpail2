from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def push_success(
    env: ManagerBasedRLEnv,
    y_threshold: float = -0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """True when cube passes y_threshold in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object.data.root_pos_w[:, :3]
    )
    return pos_b[:, 1] <= y_threshold
