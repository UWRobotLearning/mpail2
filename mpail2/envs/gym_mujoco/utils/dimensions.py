"""Observation / action dimension helpers for wrapped Gymnasium envs."""

import gymnasium as gym


def get_env_dimensions(env, num_envs: int = 1) -> tuple[int, int]:
    """
    Extract flattened ``obs`` dim and action dim from a (possibly wrapped) env.

    Uses ``single_observation_space`` / ``single_action_space`` when present
    (vectorized stacks).
    """
    del num_envs  # reserved for API compatibility
    base_env = env.env if hasattr(env, "env") else env

    obs_space = getattr(base_env, "single_observation_space", None)
    act_space = getattr(base_env, "single_action_space", None)

    if obs_space is None:
        obs_space = env.observation_space
    if act_space is None:
        act_space = env.action_space

    if isinstance(obs_space, gym.spaces.Dict):
        obs_dim = sum(
            space.shape[0] if len(space.shape) == 1 else space.shape[-1]
            for space in obs_space.spaces.values()
        )
    else:
        obs_dim = (
            obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape[-1]
        )

    action_dim = (
        act_space.shape[0] if len(act_space.shape) == 1 else act_space.shape[-1]
    )

    return obs_dim, action_dim
