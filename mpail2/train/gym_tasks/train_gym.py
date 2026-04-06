"""
Train MPAIL on Gymnasium environments (Hydra-configurable).

Entry points:

- ``python -m mpail2.train.train --env Ant-v5 learner.replay_size=500000`` (unified launcher; ``--env`` is stripped for Hydra).

MPAIL differs from MGAIL by using model-based planning (MPPI) instead of
model-free policy optimization.

Requires Gymnasium / MuJoCo; does not require Isaac Sim.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # Use EGL for headless GPU rendering

import dataclasses
import random
from datetime import datetime
from typing import Any

import torch
from omegaconf import OmegaConf

from mpail2.envs.gym_mujoco import (
    setup_environment,
    get_env_dimensions,
    load_demonstrations,
)
from mpail2.runner import MPAIL2Runner
from mpail2.train.configs.gym_train_config import GymHydraTrainConfig
from mpail2.train.gym_tasks.gym_assembly import (
    apply_gym_learner_overrides,
    build_gym_runner_cfg,
    default_gym_learner_cfg,
)
from mpail2.train.gym_tasks.utils import find_demo_file
from mpail2.train.utils.runtime import (
    ensure_wandb_settings,
    finish_wandb_if_running,
    make_runner_log_cfg,
    maybe_disable_wandb_if_unavailable,
    maybe_init_wandb,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available. Install with: pip install wandb")


def run_gym_training(cfg: Any) -> None:
    """Run gym MPAIL training from a Hydra-composed config (DictConfig or ``GymHydraTrainConfig``)."""
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_object(cfg)
    if not isinstance(cfg, GymHydraTrainConfig):
        raise TypeError(f"Expected GymHydraTrainConfig after conversion, got {type(cfg)}")

    env_id = cfg.env_id

    demo_path = cfg.demo_path
    if demo_path is None:
        demo_path = find_demo_file(env_id, cfg.demo_dir)
    elif not os.path.exists(demo_path):
        print(f"[WARN] Demo file not found at: {demo_path}")
        demo_path = find_demo_file(env_id, cfg.demo_dir)

    run_name = cfg.log.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{env_id}_{timestamp}"
    cfg.log.run_name = run_name

    log_dir = os.path.join(cfg.log.run_log_dir, env_id, run_name)
    cfg.log.run_log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)

    print("=" * 60)
    print("MPAIL Training on Gymnasium Environment")
    print("=" * 60)
    print(f"Environment: {env_id}")
    print(f"Num envs: {cfg.num_envs}")
    print(f"Demonstrations: {demo_path}")
    print(f"Iterations: {cfg.runner.num_learning_iterations}")
    print(f"Log directory: {log_dir}")
    print("=" * 60)

    random.seed(cfg.runner.seed)
    torch.manual_seed(cfg.runner.seed)

    use_wandb = maybe_disable_wandb_if_unavailable(cfg.log, wandb_available=WANDB_AVAILABLE)
    if use_wandb:
        ensure_wandb_settings(cfg.log)
    maybe_init_wandb(log_cfg=cfg.log, run_name=run_name, config_obj=dataclasses.asdict(cfg))
    if use_wandb:
        print(f"[INFO] Wandb logging enabled: {cfg.log.wandb_entity}/{cfg.log.wandb_project}/{run_name}")

    render_mode = "rgb_array" if cfg.log.video else None
    video_dir = os.path.join(log_dir, "videos") if cfg.log.video else None

    env = setup_environment(
        env_id=env_id,
        num_envs=cfg.num_envs,
        max_episode_length=cfg.max_episode_length,
        device=cfg.device,
        render_mode=render_mode,
        video_folder=video_dir if cfg.log.video else None,
        video_step_trigger=lambda step: step % cfg.log.video_interval == 0,
        video_length=cfg.log.video_length,
        enable_wandb=use_wandb,
        video_fps=cfg.log.video_fps,
        terminate_when_unhealthy=not cfg.no_termination,
    )

    obs_dim, action_dim = get_env_dimensions(env, cfg.num_envs)
    print(f"[INFO] Observation dim: {obs_dim}")
    print(f"[INFO] Action dim: {action_dim}")

    print(f"[INFO] Loading demonstrations from: {demo_path}")
    demonstrations, metadata = load_demonstrations(
        demo_path, device=cfg.device, num_demos=cfg.num_demos
    )

    for key, tensor in demonstrations.items():
        print(f"  {key}: {tensor.shape}")
    if metadata:
        print(f"  Metadata: {metadata}")

    print("[INFO] Creating MPAIL configuration...")
    learner_cfg = default_gym_learner_cfg(
        state_dim=obs_dim,
        action_dim=action_dim,
        use_terminations=not cfg.no_termination,
    )
    apply_gym_learner_overrides(learner_cfg, cfg.learner)

    logger = "wandb" if use_wandb else None
    runner_log_cfg = make_runner_log_cfg(cfg.log, logger=logger)
    runner_cfg = build_gym_runner_cfg(
        learner_cfg=learner_cfg,
        num_learning_iterations=cfg.runner.num_learning_iterations,
        path_to_demonstrations=demo_path,
        seed=cfg.runner.seed,
        logger=logger,
        log_cfg=runner_log_cfg,
    )

    if use_wandb:
        import wandb

        wandb.config.update(dataclasses.asdict(runner_cfg))

    print("[INFO] Creating MPAIL2 runner...")
    runner = MPAIL2Runner(
        demonstrations=demonstrations,
        env=env,
        runner_cfg=runner_cfg,
        device=cfg.device,
    )

    env.seed(cfg.runner.seed)

    print("[INFO] Starting training...")
    try:
        runner.learn()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    finally:
        print("[INFO] Saving final model...")
        runner.save(postfix="final")

        finish_wandb_if_running(log_cfg=cfg.log)
        env.close()

    print("[INFO] Training complete!")
    print(f"[INFO] Models saved to: {os.path.join(log_dir, 'models')}")


if __name__ == "__main__":
    from mpail2.train.train import main as unified_main

    unified_main(default_env=GymHydraTrainConfig.env_id)
