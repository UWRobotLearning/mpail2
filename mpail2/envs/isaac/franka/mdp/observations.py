"""Observation functions for Franka push tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, FrameTransformer
from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Camera observations
##

def camera_image_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "rgb",
) -> torch.Tensor:
    """RGB image from a camera sensor.

    Returns:
        Tensor of shape ``(num_envs, H, W, 3)``.
    """
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    rgb = sensor.data.output[data_type]
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    return rgb[..., :3]


##
# Proprioception observations
##

def ee_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector XYZ position relative to robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def ee_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector linear velocity in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_body_idx = robot.find_bodies("panda_hand")[0][0]
    ee_vel_w = robot.data.body_lin_vel_w[:, ee_body_idx, :]
    return quat_rotate_inverse(robot.data.root_quat_w, ee_vel_w)


def object_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube position in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def object_quat_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube orientation in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_quat_w = object.data.root_quat_w
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w, object_quat_w
    )
    return object_quat_b


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scalar gripper opening (sum of finger joint positions)."""
    robot: Articulation = env.scene[robot_cfg.name]
    j1 = robot.data.joint_pos[:, -1]
    j2 = robot.data.joint_pos[:, -2]
    return (j1 + j2).unsqueeze(1)


def gripper_trigger(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Internal gripper open/close accumulator from the gripper action term (0=closed, 1=open)."""
    gripper_action = env.action_manager._terms["gripper_action"]
    return gripper_action.gripper_trigger
