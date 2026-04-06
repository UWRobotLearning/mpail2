"""Shared Franka tabletop scene, EE frame, cube, and default physics (push + pick-and-place)."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

from . import mdp


@configclass
class FrankaTableSceneCfg(InteractiveSceneCfg):
    """Table + plane + lights + Franka mount volume (robot / ee / object filled in ``__post_init__``)."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    table_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableSurface",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.0], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0), roughness=0.7, metallic=0.0, opacity=1.0,
            ),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    goal_marker: AssetBaseCfg | None = None


# Back-compat alias
PushSceneCfg = FrankaTableSceneCfg


@configclass
class ProprioceptionCfg(ObsGroup):
    """End-effector velocity + position in robot base frame (6-D)."""

    ee_vel = ObsTerm(func=mdp.ee_vel_in_robot_frame)
    ee_pos = ObsTerm(func=mdp.ee_pos_in_robot_frame)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class BaseObservationsCfg:
    proprioception: ProprioceptionCfg = ProprioceptionCfg()


@configclass
class PickPlaceProprioceptionCfg(ObsGroup):
    """Push-style EE proprio plus cube pose and gripper state (15-D when concatenated)."""

    ee_vel = ObsTerm(func=mdp.ee_vel_in_robot_frame)
    ee_pos = ObsTerm(func=mdp.ee_pos_in_robot_frame)
    gripper_pos = ObsTerm(func=mdp.gripper_pos)
    gripper_trigger = ObsTerm(func=mdp.gripper_trigger)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PickPlaceObservationsCfg:
    proprioception: PickPlaceProprioceptionCfg = PickPlaceProprioceptionCfg()


@configclass
class FrankaTableEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    init_franka_arm_pose = EventTerm(
        func=mdp.events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [
                0.0444,
                -0.1894,
                -0.1107,
                -2.5148,
                0.0044,
                2.3775,
                0.6952,
                0.0400,
                0.0400,
            ],
        },
    )

    randomize_franka_joints = EventTerm(
        func=mdp.events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.0, "asset_cfg": SceneEntityCfg("robot")},
    )


PushEventCfg = FrankaTableEventCfg


@configclass
class FrankaPickPlaceEventCfg(FrankaTableEventCfg):
    reset_gripper_trigger = EventTerm(func=mdp.reset_gripper_trigger, mode="reset")


@configclass
class TimeOutTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


PushTerminationsCfg = TimeOutTerminationsCfg


def apply_franka_tabletop_scene(cfg: ManagerBasedRLEnvCfg) -> None:
    """Populate ``cfg.scene`` robot / cube / EE frame / optional goal strip. Call from task ``__post_init__``."""
    cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=cfg.cube_init_pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.9, 0.0), roughness=0.2, metallic=0.3,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    cfg.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    if getattr(cfg, "cube_reset_pose_range", None) is not None:
        cfg.events.reset_object_position.params["pose_range"] = cfg.cube_reset_pose_range

    if cfg.goal_marker_enabled:
        cfg.scene.goal_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalMarker",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.5, cfg.goal_marker_y, 0.025), rot=(1, 0, 0, 0)
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.002, 0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.4, 0.0, 0.0), roughness=1.0, metallic=0.0, opacity=1.0,
                ),
            ),
        )
    else:
        cfg.scene.goal_marker = None


def apply_franka_tabletop_physics_viewer(cfg: ManagerBasedRLEnvCfg) -> None:
    """Shared sim / viewer / decimation."""
    cfg.decimation = 5
    cfg.episode_length_s = 10.0
    cfg.sim.dt = 0.02
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.bounce_threshold_velocity = 0.01
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
    cfg.sim.physx.friction_correlation_distance = 0.00625

    cfg.viewer.eye = (1.0, 1.0, 0.8)
    cfg.viewer.lookat = (0.5, 0.0, 0.2)
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
