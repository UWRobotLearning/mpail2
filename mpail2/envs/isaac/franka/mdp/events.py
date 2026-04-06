"""Franka-specific event functions for push tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the default joint pose for all envs at startup."""
    asset: Articulation = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Add Gaussian noise to joint positions at reset, clamped to joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Preserve gripper fingers unchanged
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_gripper_trigger(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Re-open gripper internal trigger on reset (matches finger default pose)."""
    gripper_action = env.action_manager._terms.get("gripper_action")
    if gripper_action is not None and hasattr(gripper_action, "gripper_trigger"):
        gripper_action.gripper_trigger[env_ids] = 1.0
