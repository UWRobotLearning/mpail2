"""MPAIL-shaped torch observations/rewards (matches ``FrankaRealWrapper`` pattern)."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch


class KinovaRealWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, device: str = "cuda"):
        super().__init__(env)
        self.device = device
        self.step_count = 0
        self.max_episode_length = getattr(env, "max_episode_steps", 700)

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        self.step_count = 0
        t0 = time.time()
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["reset_time"] = info.get("reset_time", time.time() - t0)
        return self._obs_to_torch(obs), info

    def step(self, action):
        t0 = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        terminated = bool(terminated) or self.step_count >= self.max_episode_length
        info = dict(info)
        info["action_executed"] = (
            action.detach().cpu().numpy()
            if torch.is_tensor(action)
            else np.asarray(action)
        )
        info["mpail_env/step_time"] = round(time.time() - t0, 3)
        return (
            self._obs_to_torch(obs),
            torch.tensor([reward], dtype=torch.float32, device=self.device),
            torch.tensor([terminated], dtype=torch.bool, device=self.device),
            torch.tensor([truncated], dtype=torch.bool, device=self.device),
            info,
        )

    def _obs_to_torch(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in obs.items()}
