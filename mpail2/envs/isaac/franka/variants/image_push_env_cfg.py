"""Image-based Franka push environment.

Gym id: ``Isaac-FrankaPush-Image-v0``
Obs:    proprioception (6-D), table_cam (64x64x3), wrist_cam (64x64x3)
Action: 3-DoF IK (XYZ delta, no gripper)
Reward: sparse +1 when cube crosses y = -0.1 in robot frame
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ..push_env_cfg import FrankaPushEnvCfg
from ..tabletop_env_cfg import ProprioceptionCfg
from .. import mdp


##
# Camera factory functions
##

def _table_cam_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=64,
        width=64,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.85, 0.0, 0.33), rot=(0.37, -0.65, -0.65, 0.37), convention="ros",
        ),
    )


def _wrist_cam_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=64,
        width=64,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros",
        ),
    )


##
# Observation groups
##

@configclass
class TableCamObsCfg(ObsGroup):
    table_cam = ObsTerm(
        func=mdp.camera_image_rgb,
        params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb"},
    )


@configclass
class WristCamObsCfg(ObsGroup):
    wrist_cam = ObsTerm(
        func=mdp.camera_image_rgb,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb"},
    )


@configclass
class ImageObservationsCfg:
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    table_cam: TableCamObsCfg = TableCamObsCfg()
    wrist_cam: WristCamObsCfg = WristCamObsCfg()


##
# Rewards
##

@configclass
class PushImageRewardsCfg:
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


##
# Full image env
##

@configclass
class FrankaPushImageEnvCfg(FrankaPushEnvCfg):
    observations: ImageObservationsCfg = ImageObservationsCfg()
    rewards: PushImageRewardsCfg = PushImageRewardsCfg()

    cube_init_pos: tuple = (0.5, 0.13, 0.0203)
    cube_reset_pose_range: dict = {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (0.0, 0.0)}
    goal_marker_enabled: bool = True
    goal_marker_y: float = -0.1

    image_obs_list: list = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_cam = _table_cam_cfg()
        self.scene.wrist_cam = _wrist_cam_cfg()
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["table_cam", "wrist_cam"]
