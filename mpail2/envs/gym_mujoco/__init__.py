"""Gymnasium task family and utilities."""

from .utils import (
    CustomRecordVideo,
    convert_to_mpail_format,
    create_transition_pairs,
    get_dataset_stats,
    get_demo_stats,
    get_env_dimensions,
    is_mpail_format,
    is_universal_format,
    load_demonstrations,
    load_expert_dataset,
    make_video_env,
    merge_demonstrations,
    package_mpail_demos,
    package_mpail_demos_multimodal,
    save_demonstrations,
    save_expert_dataset,
)
from .env_factory import (
    GymEnvArgs,
    GymMujocoEnvArgs,
    make_env,
    make_gym_env,
    make_mujoco_env,
    setup_environment,
)
from .wrappers import (
    GymnasiumWrapper,
    VectorizedGymnasiumWrapper,
    VideoRecordingWrapper,
)

__all__ = [
    # factory
    "GymEnvArgs",
    "GymMujocoEnvArgs",
    "make_env",
    "make_gym_env",
    "make_mujoco_env",
    "setup_environment",
    "get_env_dimensions",
    # wrappers
    "GymnasiumWrapper",
    "VectorizedGymnasiumWrapper",
    "VideoRecordingWrapper",
    # video
    "CustomRecordVideo",
    "make_video_env",
    # data
    "create_transition_pairs",
    "package_mpail_demos",
    "package_mpail_demos_multimodal",
    "save_demonstrations",
    "load_demonstrations",
    "merge_demonstrations",
    "get_demo_stats",
    "save_expert_dataset",
    "load_expert_dataset",
    "convert_to_mpail_format",
    "is_universal_format",
    "is_mpail_format",
    "get_dataset_stats",
]
