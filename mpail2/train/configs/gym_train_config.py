"""Hydra dataclass for Gymnasium / MuJoCo MPAIL training (ConfigStore registration)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

@dataclass
class GymRunnerHydraCfg:
    num_learning_iterations: int = 200
    seed: int = 42


@dataclass
class GymLogHydraCfg:
    run_log_dir: str = "./logs/mpail"
    run_name: Optional[str] = None
    no_wandb: bool = True
    checkpoint_every: int = 50
    video: bool = False
    video_interval: int = 5000
    video_length: Optional[int] = None
    video_fps: int = 30
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None


@dataclass
class GymHydraTrainConfig:
    env_id: str = "Ant-v5"
    num_envs: int = 1
    max_episode_length: int = 1000
    device: str = "cuda"

    demo_path: Optional[str] = None
    demo_dir: Optional[str] = None
    num_demos: Optional[int] = None

    runner: GymRunnerHydraCfg = GymRunnerHydraCfg()
    log: GymLogHydraCfg = GymLogHydraCfg()
    no_termination: bool = False

    # Optional CLI-tunable overrides applied onto gym learner defaults.
    learner: Optional[dict[str, Any]] = None


ConfigStore.instance().store(name="gym", node=GymHydraTrainConfig)

__all__ = [
    "GymHydraTrainConfig",
    "GymLogHydraCfg",
    "GymRunnerHydraCfg",
]
