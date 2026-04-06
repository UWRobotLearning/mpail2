"""Runtime assembly helpers for Isaac image training."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import mpail2.configs.defs as defs
from mpail2.configs.cfgs import MPAIL2RunnerCfg
from mpail2.train.utils.demos import resolve_demo_path
from mpail2.train.utils.runtime import make_runner_log_cfg, maybe_init_wandb


def assemble_isaac_runner(
    cfg: Any,
    learner_cfg: defs.LearnerConfig,
    *,
    demo_env_var: str,
    default_demo_rel: str,
    default_num_iterations: int,
    default_seed: int = 0,
) -> MPAIL2RunnerCfg:
    r = cfg.runner
    log = cfg.log

    demo = resolve_demo_path(
        r.path_to_demonstrations,
        demo_env_var,
        default_demo_rel,
    )

    logger = r.logger if r.logger is not None else ("wandb" if not log.no_wandb else None)
    num_iter = r.num_learning_iterations if r.num_learning_iterations is not None else default_num_iterations
    seed = r.seed if r.seed is not None else default_seed

    runner_log = make_runner_log_cfg(log, logger=logger)
    vis_rollouts = r.vis_rollouts if r.vis_rollouts is not None else log.video

    runner_cfg = MPAIL2RunnerCfg(
        learner_cfg=learner_cfg,
        num_learning_iterations=num_iter,
        path_to_demonstrations=demo,
        seed=seed,
        logger=logger,
        log_cfg=runner_log,
        vis_rollouts=vis_rollouts,
    )

    model_save_path = f"{log.run_log_dir}/{log.model_save_dirname}"
    os.makedirs(log.run_log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    maybe_init_wandb(
        log_cfg=log,
        run_name=log.run_name,
        run_dir=log.run_log_dir,
        config_obj=asdict(runner_cfg),
    )

    return runner_cfg
