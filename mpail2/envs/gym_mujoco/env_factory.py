"""Build vectorized Gymnasium + MPAIL tensor wrappers (mirrors ``real/*/env_factory`` pattern)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym

from .wrappers import VectorizedGymnasiumWrapper, VideoRecordingWrapper


def make_env(
    env_id,
    idx,
    max_episode_length,
    device,
    render_mode=None,
    terminate_when_unhealthy=True,
):
    """
    Return a thunk that builds one sub-environment (for ``SyncVectorEnv``).

    ``idx`` is unused but kept for call-site compatibility with vectorized factory patterns.
    """
    del idx, max_episode_length, device

    def thunk():
        try:
            env = gym.make(
                env_id,
                render_mode=render_mode,
                terminate_when_unhealthy=terminate_when_unhealthy,
            )
        except TypeError:
            env = gym.make(env_id, render_mode=render_mode)
            if not terminate_when_unhealthy:
                print(
                    f"[WARN] Environment {env_id} does not support terminate_when_unhealthy"
                )

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


@dataclass
class GymMujocoEnvArgs:
    """Arguments for :func:`make_mujoco_env` / :func:`setup_environment`."""

    env_id: str
    num_envs: int = 1
    max_episode_length: int = 1000
    device: str = "cuda"
    render_mode: Optional[str] = None
    video_folder: Optional[str] = None
    video_step_trigger: Optional[Callable[[int], bool]] = None
    video_length: int = 1000
    enable_wandb: bool = False
    video_fps: int = 50
    terminate_when_unhealthy: bool = True


def setup_environment(
    env_id: str,
    num_envs: int = 1,
    max_episode_length: int = 1000,
    device: str = "cuda",
    render_mode: Optional[str] = None,
    video_folder: Optional[str] = None,
    video_step_trigger: Optional[Callable[[int], bool]] = None,
    video_length: int = 1000,
    enable_wandb: bool = False,
    video_fps: int = 50,
    terminate_when_unhealthy: bool = True,
):
    """Create SyncVectorEnv + :class:`VectorizedGymnasiumWrapper` [+ optional video]."""
    return make_mujoco_env(
        GymMujocoEnvArgs(
            env_id=env_id,
            num_envs=num_envs,
            max_episode_length=max_episode_length,
            device=device,
            render_mode=render_mode,
            video_folder=video_folder,
            video_step_trigger=video_step_trigger,
            video_length=video_length,
            enable_wandb=enable_wandb,
            video_fps=video_fps,
            terminate_when_unhealthy=terminate_when_unhealthy,
        )
    )


def make_mujoco_env(args: GymMujocoEnvArgs | Any):
    """Preferred entry point: dataclass-driven env construction."""
    env_id = args.env_id
    num_envs = args.num_envs
    max_episode_length = args.max_episode_length
    device = args.device
    render_mode = args.render_mode
    video_folder = args.video_folder
    video_step_trigger = args.video_step_trigger
    video_length = args.video_length
    enable_wandb = args.enable_wandb
    video_fps = args.video_fps
    terminate_when_unhealthy = args.terminate_when_unhealthy

    term_str = "enabled" if terminate_when_unhealthy else "DISABLED"
    print(
        f"[INFO] Creating {num_envs} environment(s): {env_id} "
        f"render_mode={render_mode}, termination: {term_str}"
    )

    env_fns = [
        make_env(
            env_id,
            i,
            max_episode_length,
            device,
            render_mode=render_mode,
            terminate_when_unhealthy=terminate_when_unhealthy,
        )
        for i in range(num_envs)
    ]
    env = gym.vector.SyncVectorEnv(env_fns)

    env = VectorizedGymnasiumWrapper(
        env,
        num_envs=num_envs,
        max_episode_length=max_episode_length,
        device=device,
    )

    if video_folder is not None:
        import os

        os.makedirs(video_folder, exist_ok=True)
        env = VideoRecordingWrapper(
            env,
            video_folder=video_folder,
            step_trigger=video_step_trigger,
            video_length=video_length,
            enable_wandb=enable_wandb,
            fps=video_fps,
        )
        print(f"[INFO] Video recording enabled: {video_folder}")

    return env


# Backward-compat aliases (older call-sites may still use these names).
GymEnvArgs = GymMujocoEnvArgs
make_gym_env = make_mujoco_env
