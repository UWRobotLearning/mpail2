from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ee_to_object_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return torch.norm(object.data.root_pos_w[:, :3] - ee_frame.data.target_pos_w[:, 0, :], dim=1)


def _object_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object.data.root_pos_w[:, :3]
    )
    return pos_b


def push_reward(
    env: ManagerBasedRLEnv,
    y_threshold: float = -0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense: -(EE-to-block dist) - (block-to-y_threshold dist)."""
    ee_to_block = _ee_to_object_distance(env, object_cfg, ee_frame_cfg)
    obj_b = _object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    return -ee_to_block - torch.abs(obj_b[:, 1] - y_threshold)


def push_sparse_reward(
    env: ManagerBasedRLEnv,
    y_threshold: float = -0.1,
    reward_amount: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse +reward_amount when cube crosses y_threshold in robot frame."""
    obj_b = _object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    return torch.where(
        obj_b[:, 1] < y_threshold,
        torch.full_like(obj_b[:, 1], reward_amount),
        torch.zeros_like(obj_b[:, 1]),
    )


def pick_place_dense_reward(
    env: ManagerBasedRLEnv,
    y_target: float = -0.1,
    min_height: float = 0.02,
    height_gain: float = 0.75,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward: approach cube, encourage lift, then reduce lateral error to goal strip."""
    ee_to_block = _ee_to_object_distance(env, object_cfg, ee_frame_cfg)
    obj_b = _object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    lift = height_gain * torch.clamp(obj_b[:, 2] - min_height, min=0.0)
    return -ee_to_block - torch.abs(obj_b[:, 1] - y_target) + lift


def pick_place_sparse_reward(
    env: ManagerBasedRLEnv,
    y_target: float = -0.1,
    min_height: float = 0.04,
    reward_amount: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse success when cube is past y_target and lifted above ``min_height``."""
    obj_b = _object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    ok = (obj_b[:, 1] < y_target) & (obj_b[:, 2] > min_height)
    return torch.where(ok, torch.full_like(obj_b[:, 1], reward_amount), torch.zeros_like(obj_b[:, 1]))
