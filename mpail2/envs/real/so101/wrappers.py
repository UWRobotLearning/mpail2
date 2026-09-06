"""MPAIL-shaped wrapper for SO101RobotEnv (mirrors FrankaRealWrapper / KinovaRealWrapper)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from .robot_limits import MAX_EPISODE_STEPS

logger = logging.getLogger("so101_env")


class SO101RealWrapper(gym.Wrapper):
    """Converts SO101RobotEnv numpy observations to torch tensors.

    Sets ``num_envs`` and ``max_episode_length`` on the *inner* env so that
    ``env.unwrapped`` exposes both attributes as required by MPAIL2Runner.
    """

    def __init__(self, env: gym.Env, device: str = "cuda"):
        super().__init__(env)
        self.device = device
        self.step_count = 0
        self.episode_count = 0

        # Optional callback(prev_obs_torch, obs_torch) -> float, set by the caller after
        # building the learner (e.g. train_so101_local.py) to compute + log the actual
        # GAIL discriminator reward per step, the same way grpc_policy_server.py does
        # inline via learner._encoder/_reward. runner.learn() itself never sees this —
        # the env's own step() reward stays 0.0, since that's purely a logging value
        # (learner.py: "THESE ARE NOT TO BE USED IN LEARNING - JUST FOR LOGGING").
        self.reward_fn: Optional[Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], float]] = None
        self._prev_obs_torch: Optional[Dict[str, torch.Tensor]] = None

        # Expose these on unwrapped for MPAIL2Runner
        self.env.unwrapped.num_envs = 1
        self.env.unwrapped.device = device
        if not hasattr(self.env.unwrapped, "max_episode_length"):
            self.env.unwrapped.max_episode_length = MAX_EPISODE_STEPS
        self.max_episode_length = self.env.unwrapped.max_episode_length

    # ─── observation helpers ──────────────────────────────────────────────────

    def _obs_to_torch(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in obs.items()
        }

    # ─── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        self.step_count = 0
        self.episode_count += 1
        self._prev_obs_torch = None  # discriminator reward needs a real (obs_t, obs_t+1) pair
        t0 = time.time()
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["reset_time"] = info.get("reset_time", round(time.time() - t0, 3))
        obs_torch = self._obs_to_torch(obs)
        self._prev_obs_torch = obs_torch
        return obs_torch, info

    def step(
        self, action: torch.Tensor | np.ndarray
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        t0 = time.time()

        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action, dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(action_np)
        obs_torch = self._obs_to_torch(obs)

        self.step_count += 1
        truncated = truncated or (self.step_count >= self.max_episode_length)

        info = dict(info)
        info["action_executed"] = action_np.flatten().tolist()
        info["mpail_env/step_time"] = round(time.time() - t0, 3)

        if self.reward_fn is not None and self._prev_obs_torch is not None:
            disc_reward = self.reward_fn(self._prev_obs_torch, obs_torch)
            info["disc_reward"] = disc_reward
            logger.info(f"[step {self.step_count - 1}] disc_reward={disc_reward:.4f}")
        self._prev_obs_torch = obs_torch

        return (
            obs_torch,
            torch.tensor([float(reward)], dtype=torch.float32, device=self.device),
            torch.tensor([bool(terminated)], dtype=torch.bool, device=self.device),
            torch.tensor([bool(truncated)], dtype=torch.bool, device=self.device),
            info,
        )

    def seed(self, seed: int | None = None) -> None:
        pass
