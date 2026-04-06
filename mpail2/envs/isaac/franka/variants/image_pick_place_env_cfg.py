"""Image-based Franka pick-and-place (4-D action, proprio includes gripper + cube)."""

from isaaclab.utils import configclass

from ..pick_place_env_cfg import FrankaPickPlaceEnvCfg, PickPlaceRewardsCfg
from ..tabletop_env_cfg import PickPlaceProprioceptionCfg
from .image_push_env_cfg import TableCamObsCfg, WristCamObsCfg, _table_cam_cfg, _wrist_cam_cfg


@configclass
class ImagePickPlaceObservationsCfg:
    proprioception: PickPlaceProprioceptionCfg = PickPlaceProprioceptionCfg()
    table_cam: TableCamObsCfg = TableCamObsCfg()
    wrist_cam: WristCamObsCfg = WristCamObsCfg()


@configclass
class FrankaPickPlaceImageEnvCfg(FrankaPickPlaceEnvCfg):
    observations: ImagePickPlaceObservationsCfg = ImagePickPlaceObservationsCfg()
    rewards: PickPlaceRewardsCfg = PickPlaceRewardsCfg()

    cube_init_pos: tuple = (0.5, 0.0, 0.0203)
    cube_reset_pose_range: dict = {"x": (-0.05, 0.05), "y": (0.0, 0.06), "z": (0.0, 0.0)}
    goal_marker_enabled: bool = True
    goal_marker_y: float = -0.1
    image_obs_list: list | None = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_cam = _table_cam_cfg()
        self.scene.wrist_cam = _wrist_cam_cfg()
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["table_cam", "wrist_cam"]
