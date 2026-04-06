"""Hydra config for Franka push (image) — registers ConfigStore node."""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from mpail2.train.configs.isaac_base_config import IsaacHydraTrainConfig, IsaacRunnerCfg, LogConfig

ENV_NAME = "push"
ENV_ALIASES = (
    "Isaac-FrankaPush-Image-Dense-v0",
)
ENV_SPEC = {
    "suite": "isaac_image",
    "config_name": "push_image",
    "demo_env_var": "MPAIL_PUSH_DEMO",
    "default_demo_rel": "../demos/push_image.pt",
    "default_num_iterations": 500,
}


@dataclass
class FrankaPushTrainConfig(IsaacHydraTrainConfig):
    task: str = "Isaac-FrankaPush-Image-Dense-v0"
    log: LogConfig = field(
        default_factory=lambda: LogConfig(
            run_name="mpail2_push",
        )
    )
    runner: IsaacRunnerCfg = field(
        default_factory=lambda: IsaacRunnerCfg(num_learning_iterations=500),
    )


ConfigStore.instance().store(name="push_image", node=FrankaPushTrainConfig)
