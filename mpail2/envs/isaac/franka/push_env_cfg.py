"""Franka push task: 3-DoF IK only (no gripper command)."""

from dataclasses import MISSING

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .actions.ik_actions import Broadcast3To6IKActionCfg
from .tabletop_env_cfg import (
    BaseObservationsCfg,
    FrankaTableEventCfg,
    FrankaTableSceneCfg,
    TimeOutTerminationsCfg,
    apply_franka_tabletop_physics_viewer,
    apply_franka_tabletop_scene,
)


@configclass
class PushActionsCfg:
    arm_action: Broadcast3To6IKActionCfg = MISSING


@configclass
class PushRewardsCfg:
    push_dense = RewTerm(
        func=mdp.push_reward,
        weight=1.0,
        params={"y_threshold": -0.08},
    )
    push_sparse = RewTerm(
        func=mdp.push_sparse_reward,
        weight=1.0,
        params={"y_threshold": -0.08, "reward_amount": 1.0},
    )


@configclass
class FrankaPushEnvCfg(ManagerBasedRLEnvCfg):
    """Franka tabletop push — EE proprio (6-D), 3-DoF Cartesian IK."""

    scene: FrankaTableSceneCfg = FrankaTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BaseObservationsCfg = BaseObservationsCfg()
    actions: PushActionsCfg = PushActionsCfg()
    events: FrankaTableEventCfg = FrankaTableEventCfg()
    terminations: TimeOutTerminationsCfg = TimeOutTerminationsCfg()
    rewards: PushRewardsCfg = PushRewardsCfg()

    cube_init_pos: tuple = (0.5, 0.13, 0.0203)
    cube_reset_pose_range: dict | None = None
    goal_marker_enabled: bool = True
    goal_marker_y: float = -0.1

    def __post_init__(self):
        apply_franka_tabletop_physics_viewer(self)
        apply_franka_tabletop_scene(self)

        self.actions.arm_action = Broadcast3To6IKActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=(0.03, 0.03, 0.03),
            body_offset=Broadcast3To6IKActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        super().__post_init__()
