"""1-D gripper command broadcast to both panda finger joints (position targets)."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass


class BroadcastGripperAction(joint_actions.JointPositionAction):
    """Scalar action integrates into ``gripper_trigger`` [0,1]; drives open/close finger targets."""

    def __init__(self, cfg: "BroadcastGripperActionCfg", env):
        super().__init__(cfg, env)
        self._n_joints = len(self._joint_ids)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._n_joints, device=self.device)
        self.gripper_trigger = torch.ones(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self.gripper_trigger.add_(actions * self.cfg.trigger_scale)
        self.gripper_trigger.clamp_(min=0.0, max=1.0)

        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        close_m = (self.gripper_trigger <= 0.0).squeeze(-1)
        open_m = (self.gripper_trigger >= 1.0).squeeze(-1)

        cmd = torch.zeros_like(current_pos)
        cmd[close_m] = -self.cfg.finger_step
        cmd[open_m] = self.cfg.finger_step

        target = current_pos + cmd
        self._processed_actions[:] = target.clamp(
            min=self.cfg.finger_pos_min, max=self.cfg.finger_pos_max
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class BroadcastGripperActionCfg(JointPositionActionCfg):
    class_type: type = BroadcastGripperAction

    asset_name: str = "robot"
    joint_names: list = ["panda_finger_joint1", "panda_finger_joint2"]
    scale: float = 1.0
    trigger_scale: float = 0.3
    finger_step: float = 0.003
    finger_pos_min: float = 0.01
    finger_pos_max: float = 0.04
