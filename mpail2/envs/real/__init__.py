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
