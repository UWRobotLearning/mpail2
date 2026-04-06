"""Isaac Lab image training loop: env first, learner from observation spaces, then runner."""

from __future__ import annotations

import os
import traceback
from typing import Any

import gymnasium as gym
import torch

from mpail2.envs.isaac.launcher import isaac_session
from mpail2.envs.utils import CustomRecordVideo
from mpail2.runner import MPAIL2Runner

from mpail2.train.registry import TrainEnvSpec
from mpail2.train.utils.demos import load_demonstrations
from mpail2.train.utils.hydra_learner import learner_config_from_train_cfg
from mpail2.train.utils.isaac_runner_assembly import assemble_isaac_runner
from mpail2.train.utils.learner_from_spaces import sync_learner_dims_from_env
from mpail2.train.utils.runtime import ensure_wandb_settings, finish_wandb_if_running


def run_isaac_image_training(cfg: Any, launcher_args, spec: TrainEnvSpec) -> None:
    """OmegaConf or dataclass cfg; must expose .task, .log, .runner, .device, etc."""
    ensure_wandb_settings(cfg.log)

    with isaac_session(
        launcher_args,
        enable_cameras=cfg.enable_cameras,
        log_level=cfg.log.log_level,
    ):
        try:
            import mpail2.envs.isaac.franka  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg

            log_cfg = cfg.log

            env_cfg = parse_env_cfg(cfg.task, device=cfg.device, num_envs=cfg.num_envs)
            env = gym.make(cfg.task, cfg=env_cfg, render_mode="rgb_array")

            learner_cfg = learner_config_from_train_cfg(cfg)
            if cfg.sync_learner_dims_from_env:
                learner_cfg = sync_learner_dims_from_env(learner_cfg, env)
            runner_cfg = assemble_isaac_runner(
                cfg,
                learner_cfg,
                demo_env_var=spec.demo_env_var,
                default_demo_rel=spec.default_demo_rel,
                default_num_iterations=spec.default_num_iterations,
            )

            if log_cfg.video:
                if log_cfg.video_length:
                    vid_length = log_cfg.video_length
                else:
                    vid_length = env.unwrapped.max_episode_length

                video_kwargs = {
                    "video_folder": os.path.join(log_cfg.run_log_dir, "videos"),
                    "step_trigger": lambda step: step % log_cfg.video_interval == 0,
                    "video_length": vid_length,
                    "disable_logger": True,
                    "enable_wandb": not log_cfg.no_wandb,
                    "video_resolution": log_cfg.video_resolution,
                    "video_crf": log_cfg.video_crf,
                }
                print("[INFO] Recording videos during training.")
                env = CustomRecordVideo(env, **video_kwargs)

            demonstrations = load_demonstrations(runner_cfg.path_to_demonstrations, device=cfg.device)

            runner = MPAIL2Runner(
                demonstrations=demonstrations,
                env=env,
                runner_cfg=runner_cfg,
                device=cfg.device,
                dtype=torch.float32,
            )
            try:
                runner.learn()
            finally:
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    finish_wandb_if_running(log_cfg=log_cfg)
                except Exception:
                    pass

        except Exception:
            print("\n[ERROR] Exception occurred in training script:")
            traceback.print_exc()
            raise
