"""Shared Hydra dataclasses for Isaac image training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class LogConfig:
    run_log_dir: str = "./logs"
    run_name: str = "mpail2_isaac"
    no_wandb: bool = False
    video: bool = True
    video_interval: int = 100
    video_length: Optional[int] = None
    video_resolution: tuple = (640, 360)
    video_crf: int = 30
    checkpoint_every: int = 10
    model_save_dirname: str = "models"
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    log_level: str = "error"


@dataclass
class IsaacRunnerCfg:
    path_to_demonstrations: Optional[str] = None
    num_learning_iterations: Optional[int] = None
    seed: Optional[int] = 0
    logger: Optional[str] = None
    vis_rollouts: bool = False


@dataclass
class IsaacHydraTrainConfig:
    """Base fields shared by Isaac image envs; subclasses set task and runner defaults."""

    task: str = ""
    num_envs: int = 1
    device: str = "cuda"
    log: LogConfig = LogConfig()
    runner: IsaacRunnerCfg = IsaacRunnerCfg()
    # Optional CLI-tunable overrides applied onto task-specific learner presets.
    learner: Optional[dict[str, Any]] = None
    headless: bool = True
    enable_cameras: bool = True
    # If True, patch planner action_dim and encoder shapes from the live env after gym.make.
    sync_learner_dims_from_env: bool = True
