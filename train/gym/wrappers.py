"""
Wrappers for Gymnasium environments to make them compatible with MPAIL
"""

import os
import torch
import numpy as np
import gymnasium as gym

import av
from gymnasium import logger
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers.rendering import RecordVideo
from typing import Callable



class GymnasiumWrapper(gym.Wrapper):
    """
    Wrapper to add Isaac Lab-compatible attributes to Gymnasium environments.
    
    Isaac Lab environments have:
    - env.unwrapped.num_envs
    - env.unwrapped.max_episode_length
    - env.unwrapped.device
    - env.unwrapped.seed()
    - env.unwrapped.current_iteration
    
    Also handles tensor conversions (numpy -> torch) and adds batch dimension
    for single environments to match vectorized environment interface.
    
    For vectorized environments (SyncVectorEnv), use VectorizedGymnasiumWrapper instead.
    """
    
    def __init__(self, env, num_envs=1, max_episode_length=1000, device="cpu", dtype=torch.float32):
        # Only wrap single environments, not vectorized ones
        if isinstance(env, gym.vector.VectorEnv):
            raise ValueError("GymnasiumWrapper cannot wrap VectorEnv. Use VectorizedGymnasiumWrapper or handle vectorized envs separately.")
        
        super().__init__(env)
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device
        self.dtype = dtype
        self.current_iteration = 0
        self._seed = None
        
        # Check if this is a vectorized environment (SyncVectorEnv)
        # Vectorized envs already return batched observations/actions
        self._is_vectorized = False
        
        # For single env, set on unwrapped base environment
        self.unwrapped.num_envs = num_envs
        self.unwrapped.max_episode_length = max_episode_length
        self.unwrapped.device = device
        self.unwrapped.current_iteration = 0
        
        # Add seed method to unwrapped environment if it doesn't exist
        if not hasattr(self.unwrapped, 'seed'):
            def seed_method(seed):
                self._seed = seed
                return [seed]
            self.unwrapped.seed = seed_method
        
        # Also set on self.env for backward compatibility
        self.env.num_envs = num_envs
        self.env.max_episode_length = max_episode_length
        self.env.device = device
        self.env.current_iteration = 0
        
        # Wrap observation_space in Dict format to match MPAIL's expected format
        # This ensures runner.py extracts actor_obs_shape as a dict
        if not isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({
                "obs": env.observation_space
            })
        else:
            # If already a Dict, ensure "obs" key exists
            if "obs" not in env.observation_space.spaces:
                first_key = list(env.observation_space.spaces.keys())[0]
                self.observation_space = gym.spaces.Dict({
                    "obs": env.observation_space[first_key]
                })
            else:
                self.observation_space = env.observation_space
        
    def seed(self, seed):
        """Set seed for environment"""
        # Gymnasium uses reset(seed=seed) instead of seed()
        # Store seed for use in reset
        self._seed = seed
        # Also update unwrapped environment's seed if it has the method
        if hasattr(self.unwrapped, 'seed'):
            self.unwrapped.seed(seed)
        return [seed]
    
    def _to_tensor(self, x, add_batch_dim=True, dtype=None):
        """Convert numpy array to torch tensor with proper dtype"""
        if isinstance(x, dict):
            return {k: self._to_tensor(v, add_batch_dim, dtype) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(self._to_tensor(item, add_batch_dim, dtype) for item in x)
        else:
            tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            if not isinstance(tensor, torch.Tensor):
                tensor_dtype = dtype if dtype is not None else self.dtype
                tensor = torch.tensor(tensor, dtype=tensor_dtype)
            # Convert to specified dtype (or default dtype) and move to device
            target_dtype = dtype if dtype is not None else self.dtype
            tensor = tensor.to(dtype=target_dtype, device=self.device)
            if add_batch_dim and not self._is_vectorized:
                # Only add batch dimension for single environments
                # Vectorized envs already return batched tensors
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                elif len(tensor.shape) == 1:
                    # For 1D arrays, add batch dim: [N] -> [1, N]
                    tensor = tensor.unsqueeze(0)
            return tensor
    
    def _ensure_dict_format(self, obs):
        """Convert observation to dict format expected by MPAIL"""
        if isinstance(obs, dict):
            # If already dict but doesn't have "obs" key, add it
            if "obs" not in obs:
                # Use first value or create new key
                first_key = list(obs.keys())[0]
                obs = {"obs": obs[first_key]}
            return obs
        else:
            # Convert tensor/array to dict format
            return {"obs": obs}
    
    def reset(self, seed=None, options=None):
        """Reset with seed support and tensor conversion"""
        if seed is None and self._seed is not None:
            seed = self._seed
        
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Convert to tensor and add batch dimension for single env
        obs_tensor = self._to_tensor(obs, add_batch_dim=True)
        
        # Convert to dict format expected by MPAIL
        obs_tensor = self._ensure_dict_format(obs_tensor)
        
        # Handle info dict
        if isinstance(info, dict):
            info_tensor = {k: self._to_tensor(v, add_batch_dim=False) if isinstance(v, (np.ndarray, list)) else v 
                          for k, v in info.items()}
        else:
            info_tensor = info
        
        return obs_tensor, info_tensor
    
    def step(self, actions):
        """Step environment with tensor conversion"""
        # Convert actions from torch to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
            # For single env, remove batch dimension if present
            if self.num_envs == 1 and len(actions_np.shape) > 1:
                actions_np = actions_np[0]
        else:
            actions_np = actions
        
        obs, reward, terminated, truncated, info = self.env.step(actions_np)
        
        # Convert to tensors and add batch dimensions for single env
        obs_tensor = self._to_tensor(obs, add_batch_dim=True)
        # Convert to dict format expected by MPAIL
        obs_tensor = self._ensure_dict_format(obs_tensor)
        
        reward_tensor = self._to_tensor(reward, add_batch_dim=True)
        # Convert terminated and truncated to bool tensors (not float32)
        terminated_tensor = self._to_tensor(terminated, add_batch_dim=True, dtype=torch.bool)
        truncated_tensor = self._to_tensor(truncated, add_batch_dim=True, dtype=torch.bool)
        
        # Handle info dict
        if isinstance(info, dict):
            info_tensor = {k: self._to_tensor(v, add_batch_dim=False) if isinstance(v, (np.ndarray, list)) else v 
                          for k, v in info.items()}
        else:
            info_tensor = info
        
        return obs_tensor, reward_tensor, terminated_tensor, truncated_tensor, info_tensor


class VectorizedGymnasiumWrapper:
    """
    Wrapper for vectorized environments (SyncVectorEnv) to add Isaac Lab-compatible attributes.
    
    Unlike GymnasiumWrapper, this doesn't inherit from gym.Wrapper since VectorEnv
    cannot be wrapped with gym.Wrapper. Instead, it uses composition and adds
    the necessary attributes and tensor conversion methods.
    """
    
    def __init__(self, env, num_envs=1, max_episode_length=1000, device="cpu", dtype=torch.float32):
        if not isinstance(env, gym.vector.VectorEnv):
            raise ValueError("VectorizedGymnasiumWrapper can only wrap VectorEnv instances.")
        
        self.env = env
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device
        self.dtype = dtype
        self.current_iteration = 0
        self._seed = None
        
        # Set attributes on unwrapped environment for compatibility with MGAILRunner
        # For vectorized envs, unwrapped is the VectorEnv itself
        if not hasattr(env.unwrapped, 'num_envs'):
            env.unwrapped.num_envs = num_envs
        if not hasattr(env.unwrapped, 'max_episode_length'):
            env.unwrapped.max_episode_length = max_episode_length
        if not hasattr(env.unwrapped, 'device'):
            env.unwrapped.device = device
        if not hasattr(env.unwrapped, 'current_iteration'):
            env.unwrapped.current_iteration = 0
        if not hasattr(env.unwrapped, 'seed'):
            def seed_method(seed):
                self._seed = seed
                return [seed]
            env.unwrapped.seed = seed_method
        
        # Delegate to wrapped environment for standard attributes
        self.action_space = env.action_space
        
        # Wrap observation_space in Dict format to match MPAIL's expected format
        # This ensures runner.py extracts actor_obs_shape as a dict
        if not isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({
                "obs": env.observation_space
            })
        else:
            # If already a Dict, ensure "obs" key exists
            if "obs" not in env.observation_space.spaces:
                first_key = list(env.observation_space.spaces.keys())[0]
                self.observation_space = gym.spaces.Dict({
                    "obs": env.observation_space[first_key]
                })
            else:
                self.observation_space = env.observation_space
    
    def _to_tensor(self, x, add_batch_dim=False, dtype=None):
        """Convert numpy array to torch tensor with proper dtype"""
        if isinstance(x, dict):
            return {k: self._to_tensor(v, add_batch_dim, dtype) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(self._to_tensor(item, add_batch_dim, dtype) for item in x)
        else:
            tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            if not isinstance(tensor, torch.Tensor):
                tensor_dtype = dtype if dtype is not None else self.dtype
                tensor = torch.tensor(tensor, dtype=tensor_dtype)
            target_dtype = dtype if dtype is not None else self.dtype
            tensor = tensor.to(dtype=target_dtype, device=self.device)
            # Don't add batch dim for vectorized envs - they're already batched
            return tensor
    
    def _ensure_dict_format(self, obs):
        """Convert observation to dict format expected by MPAIL"""
        if isinstance(obs, dict):
            if "obs" not in obs:
                first_key = list(obs.keys())[0]
                obs = {"obs": obs[first_key]}
            return obs
        else:
            return {"obs": obs}
    
    def reset(self, seed=None, options=None):
        """Reset with seed support and tensor conversion"""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Convert to tensor (already batched for vectorized envs)
        obs_tensor = self._to_tensor(obs, add_batch_dim=False)
        obs_tensor = self._ensure_dict_format(obs_tensor)
        
        # Handle info dict
        if isinstance(info, dict):
            info_tensor = {k: self._to_tensor(v, add_batch_dim=False) if isinstance(v, (np.ndarray, list)) else v 
                          for k, v in info.items()}
        else:
            info_tensor = info
        
        return obs_tensor, info_tensor
    
    def step(self, actions):
        """Step environment with tensor conversion"""
        # Convert actions from torch to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions
        
        obs, reward, terminated, truncated, info = self.env.step(actions_np)
        
        # Convert to tensors (already batched for vectorized envs)
        obs_tensor = self._to_tensor(obs, add_batch_dim=False)
        obs_tensor = self._ensure_dict_format(obs_tensor)
        
        reward_tensor = self._to_tensor(reward, add_batch_dim=False)
        terminated_tensor = self._to_tensor(terminated, add_batch_dim=False, dtype=torch.bool)
        truncated_tensor = self._to_tensor(truncated, add_batch_dim=False, dtype=torch.bool)
        
        # Handle info dict
        if isinstance(info, dict):
            info_tensor = {k: self._to_tensor(v, add_batch_dim=False) if isinstance(v, (np.ndarray, list)) else v 
                          for k, v in info.items()}
        else:
            info_tensor = info
        
        return obs_tensor, reward_tensor, terminated_tensor, truncated_tensor, info_tensor
    
    def seed(self, seed):
        """Set seed for environment"""
        self._seed = seed
        if hasattr(self.env.unwrapped, 'seed'):
            self.env.unwrapped.seed(seed)
        return [seed]
    
    def close(self):
        """Close the environment"""
        return self.env.close()
    
    def render(self):
        """Render the environment - delegates to underlying env"""
        return self.env.render()
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment for compatibility with MPAIL runners.
        
        This provides access to the base environment, which is needed by MGAILRunner
        and other MPAIL components to access attributes like num_envs, device, etc.
        """
        # Return self since we have all the necessary attributes set
        # The underlying env.unwrapped would not have our custom attributes
        return self
    
    # Delegate other methods to wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)

