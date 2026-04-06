"""Dataclass learner presets for Franka image tasks."""

from __future__ import annotations

from dataclasses import dataclass

from mpail2.configs.cfgs import ObsNormalizerCfg
import mpail2.configs.defs as defs
from mpail2.train.utils.config_overrides import to_override_template


@dataclass
class FrankaPushPlannerConfig(defs.PlannerConfig):
    action_dim: int = 3
    seed: int = 0
    encoder_cfg: defs.MultiCoderConfig = defs.MultiCoderConfig(
        coder_list=[
            defs.MultiCoderConfig.ProprioCoderConfig(
                obs_key="proprioception",
                input_dim=6,
            ),
            defs.CNNCoderConfig(
                obs_key="table_cam",
                H=64,
                W=64,
                C=3,
            ),
            defs.CNNCoderConfig(
                obs_key="wrist_cam",
                H=64,
                W=64,
                C=3,
            ),
        ],
    )


@dataclass
class FrankaPickPlacePlannerConfig(defs.PlannerConfig):
    action_dim: int = 4
    seed: int = 0
    encoder_cfg: defs.MultiCoderConfig = defs.MultiCoderConfig(
        coder_list=[
            defs.MultiCoderConfig.ProprioCoderConfig(
                obs_key="proprioception",
                input_dim=8,
            ),
            defs.CNNCoderConfig(
                obs_key="table_cam",
                H=64,
                W=64,
                C=3,
            ),
            defs.CNNCoderConfig(
                obs_key="wrist_cam",
                H=64,
                W=64,
                C=3,
            ),
        ],
    )


@dataclass
class FrankaImageLearnerConfig(defs.LearnerConfig):
    """Declarative learner preset with nested planner/encoder dataclass config."""

    replay_size: int = 50_000
    use_terminations: bool = False
    obs_normalizer_cfg: ObsNormalizerCfg = ObsNormalizerCfg(normalization_type="cam_only")
    planner_cfg: defs.PlannerConfig | None = None


@dataclass
class FrankaPushImageLearnerConfig(FrankaImageLearnerConfig):
    planner_cfg: defs.PlannerConfig = FrankaPushPlannerConfig()


@dataclass
class FrankaPickPlaceImageLearnerConfig(FrankaImageLearnerConfig):
    planner_cfg: defs.PlannerConfig = FrankaPickPlacePlannerConfig()


DEFAULT_FRANKA_LEARNER_OVERRIDE_TEMPLATE: dict[str, object] = to_override_template(
    FrankaPushImageLearnerConfig()
)
