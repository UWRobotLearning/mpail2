"""Minimal real-robot envs: Kinova (ROS twist + TF) and Franka (Frankz client + RealSense + wrapper)."""

from .kinova import (
    ACTION_DIM as KINOVA_ACTION_DIM,
    KinovaRealWrapper,
    KinovaRealEnvArgs,
    ManipulationAction,
    ManipulationObservation,
    MAX_EPISODE_STEPS as KINOVA_MAX_EPISODE_STEPS,
    KinovaManipulationEnv,
    STATE_DIM as KINOVA_STATE_DIM,
    create_real_manipulation_env,
    make_kinova_env,
)

__all__ = [
    "KinovaRealWrapper",
    "KinovaRealEnvArgs",
    "ManipulationAction",
    "ManipulationObservation",
    "KinovaManipulationEnv",
    "KINOVA_STATE_DIM",
    "KINOVA_ACTION_DIM",
    "KINOVA_MAX_EPISODE_STEPS",
    "create_real_manipulation_env",
    "make_kinova_env",
]

try:
    from .franka import (
        ACTION_DIM as FRANKA_ACTION_DIM,
        FrankaRealWrapper,
        FrankaRealEnvArgs,
        RealSenseWrapper,
        STATE_DIM as FRANKA_STATE_DIM,
        TABLE_CAM_SERIAL,
        TABLE_CAM_SERIAL_2,
        WRIST_CAM_SERIAL,
        LOWER_LIMITS,
        MAX_Z_FORCE,
        RESET_QPOS,
        UPPER_LIMITS,
        make_franka_env,
        rename_camera_keys,
    )

    __all__.extend(
        [
            "FrankaRealWrapper",
            "FrankaRealEnvArgs",
            "RealSenseWrapper",
            "rename_camera_keys",
            "RESET_QPOS",
            "MAX_Z_FORCE",
            "LOWER_LIMITS",
            "UPPER_LIMITS",
            "TABLE_CAM_SERIAL",
            "TABLE_CAM_SERIAL_2",
            "WRIST_CAM_SERIAL",
            "FRANKA_STATE_DIM",
            "FRANKA_ACTION_DIM",
            "make_franka_env",
        ]
    )
except ImportError:
    pass

try:
    from .so101 import (
        ACTION_DIM as SO101_ACTION_DIM,
        SO101RealWrapper,
        SO101RealEnvArgs,
        STATE_DIM as SO101_STATE_DIM,
        EE_PROPRIO_DIM as SO101_EE_PROPRIO_DIM,
        HOME_POSITION_DEG as SO101_HOME_POSITION_DEG,
        EE_LOWER_M as SO101_EE_LOWER_M,
        EE_UPPER_M as SO101_EE_UPPER_M,
        JOINT_LOWER_DEG as SO101_JOINT_LOWER_DEG,
        JOINT_UPPER_DEG as SO101_JOINT_UPPER_DEG,
        MAX_EPISODE_STEPS as SO101_MAX_EPISODE_STEPS,
        make_so101_env,
    )

    __all__.extend(
        [
            "SO101RealWrapper",
            "SO101RealEnvArgs",
            "SO101_STATE_DIM",
            "SO101_ACTION_DIM",
            "SO101_EE_PROPRIO_DIM",
            "SO101_HOME_POSITION_DEG",
            "SO101_EE_LOWER_M",
            "SO101_EE_UPPER_M",
            "SO101_JOINT_LOWER_DEG",
            "SO101_JOINT_UPPER_DEG",
            "SO101_MAX_EPISODE_STEPS",
            "make_so101_env",
        ]
    )
except ImportError:
    pass
