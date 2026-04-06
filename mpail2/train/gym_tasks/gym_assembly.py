"""Gym runner/learner assembly helpers (non-config behavior)."""

from __future__ import annotations

from typing import Any

import mpail2.configs.defs as defs
from mpail2.configs.cfgs import MPAIL2RunnerCfg, ObsNormalizerCfg

from mpail2.train.utils.config_overrides import (
    apply_dataclass_overrides,
    prune_placeholder_overrides,
    to_override_template,
)


def default_gym_learner_cfg(*, state_dim: int, action_dim: int, use_terminations: bool) -> defs.LearnerConfig:
    """Build gym learner defaults before applying Hydra overrides."""
    return defs.LearnerConfig(
        planner_cfg=defs.PlannerConfig(
            obs_dim=state_dim,
            action_dim=action_dim,
            encoder_cfg=defs.MLPCoderConfig(obs_key="obs"),
        ),
        obs_normalizer_cfg=ObsNormalizerCfg(normalization_type="fixed"),
        use_terminations=use_terminations,
    )


def default_gym_learner_override_template() -> dict[str, Any]:
    """Hydra schema placeholder for ``learner.*`` overrides."""
    template_cfg = default_gym_learner_cfg(state_dim=1, action_dim=1, use_terminations=True)
    return to_override_template(template_cfg)


def apply_gym_learner_overrides(learner_cfg: defs.LearnerConfig, overrides: Any) -> None:
    """Apply nested Hydra overrides from ``cfg.learner`` to a learner dataclass tree."""
    if overrides is None:
        return
    if not isinstance(overrides, dict):
        raise TypeError(f"Unsupported gym learner override type: {type(overrides)}")
    effective = prune_placeholder_overrides(overrides) or {}
    if effective:
        apply_dataclass_overrides(learner_cfg, effective, path="learner")


def build_gym_runner_cfg(
    *,
    learner_cfg: defs.LearnerConfig,
    num_learning_iterations: int,
    path_to_demonstrations: str,
    seed: int,
    logger: str | None,
    log_cfg: MPAIL2RunnerCfg.LogCfg,
) -> MPAIL2RunnerCfg:
    """Assemble ``MPAIL2RunnerCfg`` for gym from already-resolved components."""
    return MPAIL2RunnerCfg(
        learner_cfg=learner_cfg,
        num_learning_iterations=num_learning_iterations,
        path_to_demonstrations=path_to_demonstrations,
        seed=seed,
        logger=logger,
        log_cfg=log_cfg,
        vis_rollouts=False,
    )


__all__ = [
    "apply_gym_learner_overrides",
    "build_gym_runner_cfg",
    "default_gym_learner_cfg",
    "default_gym_learner_override_template",
]
