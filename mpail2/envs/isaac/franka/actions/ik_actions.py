"""IK action wrappers for Franka push tasks."""

from __future__ import annotations

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


class Broadcast3To6IKAction(DifferentialInverseKinematicsAction):
    """3-DoF IK action that broadcasts XYZ commands to full 6-DoF IK (zero rotation delta).

    Includes optional workspace clamping and contact-torque-based Z-lock.
    """

    def __init__(self, cfg: "Broadcast3To6IKActionCfg", env):
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)

        self._workspace_lower = torch.tensor(cfg.workspace_lower, device=self.device, dtype=torch.float32)
        self._workspace_upper = torch.tensor(cfg.workspace_upper, device=self.device, dtype=torch.float32)
        self._workspace_epsilon = cfg.workspace_epsilon

        self._contact_torque_threshold = cfg.contact_torque_threshold
        self._contact_enabled = cfg.contact_detection_enabled
        self._contact_unlock_threshold = cfg.contact_unlock_threshold

        self._contact_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def num_actions(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        clipped = actions.clone()

        # Workspace clamping per axis
        for axis in range(3):
            a = clipped[:, axis]
            mask = (
                (ee_pos_curr[:, axis] <= self._workspace_lower[axis] + self._workspace_epsilon) & (a < 0)
            ) | (
                (ee_pos_curr[:, axis] >= self._workspace_upper[axis] - self._workspace_epsilon) & (a > 0)
            )
            clipped[:, axis] = torch.where(mask, torch.zeros_like(a), a)

        # Contact-torque Z-lock
        if self._contact_enabled:
            joint_torques = self._asset.data.applied_torque[:, 4:7]
            contact_detected = torch.abs(joint_torques[:, 2]) > self._contact_torque_threshold
            self._contact_locked = self._contact_locked | contact_detected
            moving_up = clipped[:, 2] > self._contact_unlock_threshold
            self._contact_locked = self._contact_locked & ~moving_up
            block_down = self._contact_locked & (clipped[:, 2] < 0)
            clipped[:, 2] = torch.where(block_down, torch.zeros_like(clipped[:, 2]), clipped[:, 2])

        self._processed_actions[:, 0:3] = clipped * self._scale[:, 0:3]
        self._processed_actions[:, 3:6] = 0.0

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        super().apply_actions()

    def reset(self, env_ids: torch.Tensor) -> None:
        super().reset(env_ids)
        self._contact_locked[env_ids] = False

    def get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._compute_frame_pose()


@configclass
class Broadcast3To6IKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Config for :class:`Broadcast3To6IKAction`."""

    class_type: type = Broadcast3To6IKAction

    workspace_lower: tuple[float, float, float] = (0.30, -0.15, 0.0)
    workspace_upper: tuple[float, float, float] = (0.60, 0.15, 0.26)
    workspace_epsilon: float = 1e-6
    contact_detection_enabled: bool = True
    contact_torque_threshold: float = 6.0
    contact_unlock_threshold: float = 0.1
