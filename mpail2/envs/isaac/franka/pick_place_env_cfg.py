"""Franka pick-and-place: 3-DoF IK + 1-D gripper"""

from dataclasses import MISSING

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .actions.gripper_action import BroadcastGripperActionCfg
from .actions.ik_actions import Broadcast3To6IKActionCfg
from .tabletop_env_cfg import (
    FrankaPickPlaceEventCfg,
    FrankaTableSceneCfg,
    PickPlaceObservationsCfg,
    TimeOutTerminationsCfg,
    apply_franka_tabletop_physics_viewer,
    apply_franka_tabletop_scene,
)


@configclass
class PickPlaceActionsCfg:
    arm_action: Broadcast3To6IKActionCfg = MISSING
    gripper_action: BroadcastGripperActionCfg = MISSING


@configclass
class PickPlaceRewardsCfg:
    dense = RewTerm(
        func=mdp.pick_place_dense_reward,
        weight=1.0,
        params={"y_target": -0.1, "min_height": 0.02, "height_gain": 0.75},
    )
    sparse = RewTerm(
        func=mdp.pick_place_sparse_reward,
        weight=5.0,
        params={"y_target": -0.1, "min_height": 0.04, "reward_amount": 1.0},
    )


@configclass
class FrankaPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Same tabletop as push; 4-D actions (XYZ + gripper) and 15-D proprio (EE + cube + gripper)."""

    scene: FrankaTableSceneCfg = FrankaTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: PickPlaceObservationsCfg = PickPlaceObservationsCfg()
    actions: PickPlaceActionsCfg = PickPlaceActionsCfg()
    events: FrankaPickPlaceEventCfg = FrankaPickPlaceEventCfg()
    terminations: TimeOutTerminationsCfg = TimeOutTerminationsCfg()
    rewards: PickPlaceRewardsCfg = PickPlaceRewardsCfg()

    cube_init_pos: tuple = (0.5, 0.0, 0.0203)
    cube_reset_pose_range: dict | None = {"x": (-0.05, 0.05), "y": (0.0, 0.06), "z": (0.0, 0.0)}
    goal_marker_enabled: bool = True
    goal_marker_y: float = -0.1

    def __post_init__(self):
        apply_franka_tabletop_physics_viewer(self)
        self.episode_length_s = 20.0
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
        self.actions.gripper_action = BroadcastGripperActionCfg()
        super().__post_init__()
