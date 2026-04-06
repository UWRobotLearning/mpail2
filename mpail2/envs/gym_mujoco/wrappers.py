"""Gymnasium → MPAIL tensor/dict adapters and optional vectorized video recording."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch


class GymnasiumWrapper(gym.Wrapper):
    """Single-env wrapper: Isaac-lab-like attributes, dict ``obs`` space, torch I/O."""

    def __init__(
        self, env, num_envs=1, max_episode_length=1000, device="cpu", dtype=torch.float32
    ):
        if isinstance(env, gym.vector.VectorEnv):
            raise ValueError(
                "GymnasiumWrapper cannot wrap VectorEnv; use VectorizedGymnasiumWrapper."
            )

        super().__init__(env)
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device
        self.dtype = dtype
        self.current_iteration = 0
        self._seed = None
        self._is_vectorized = False

        self.unwrapped.num_envs = num_envs
        self.unwrapped.max_episode_length = max_episode_length
        self.unwrapped.device = device
        self.unwrapped.current_iteration = 0

        if not hasattr(self.unwrapped, "seed"):

            def seed_method(seed):
                self._seed = seed
                return [seed]

            self.unwrapped.seed = seed_method

        self.env.num_envs = num_envs
        self.env.max_episode_length = max_episode_length
        self.env.device = device
        self.env.current_iteration = 0

        if not isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({"obs": env.observation_space})
        elif "obs" not in env.observation_space.spaces:
            first_key = list(env.observation_space.spaces.keys())[0]
            self.observation_space = gym.spaces.Dict(
                {"obs": env.observation_space[first_key]}
            )
        else:
            self.observation_space = env.observation_space

    def seed(self, seed):
        self._seed = seed
        if hasattr(self.unwrapped, "seed"):
            self.unwrapped.seed(seed)
        return [seed]

    def _to_tensor(self, x, add_batch_dim=True, dtype=None):
        if isinstance(x, dict):
            return {k: self._to_tensor(v, add_batch_dim, dtype) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_tensor(item, add_batch_dim, dtype) for item in x)
        tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        if not isinstance(tensor, torch.Tensor):
            td = dtype if dtype is not None else self.dtype
            tensor = torch.tensor(tensor, dtype=td)
        target_dtype = dtype if dtype is not None else self.dtype
        tensor = tensor.to(dtype=target_dtype, device=self.device)
        if add_batch_dim and not self._is_vectorized:
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            elif len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
        return tensor

    def _ensure_dict_format(self, obs):
        if isinstance(obs, dict):
            if "obs" not in obs:
                first_key = list(obs.keys())[0]
                return {"obs": obs[first_key]}
            return obs
        return {"obs": obs}

    def reset(self, seed=None, options=None):
        if seed is None and self._seed is not None:
            seed = self._seed

        obs, info = self.env.reset(seed=seed, options=options)

        obs_tensor = self._ensure_dict_format(self._to_tensor(obs, add_batch_dim=True))

        if isinstance(info, dict):
            info_tensor = {
                k: self._to_tensor(v, add_batch_dim=False)
                if isinstance(v, (np.ndarray, list))
                else v
                for k, v in info.items()
            }
        else:
            info_tensor = info

        return obs_tensor, info_tensor

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
            if self.num_envs == 1 and len(actions_np.shape) > 1:
                actions_np = actions_np[0]
        else:
            actions_np = actions

        obs, reward, terminated, truncated, info = self.env.step(actions_np)

        obs_tensor = self._ensure_dict_format(self._to_tensor(obs, add_batch_dim=True))
        reward_tensor = self._to_tensor(reward, add_batch_dim=True)
        terminated_tensor = self._to_tensor(
            terminated, add_batch_dim=True, dtype=torch.bool
        )
        truncated_tensor = self._to_tensor(truncated, add_batch_dim=True, dtype=torch.bool)

        if isinstance(info, dict):
            info_tensor = {
                k: self._to_tensor(v, add_batch_dim=False)
                if isinstance(v, (np.ndarray, list))
                else v
                for k, v in info.items()
            }
        else:
            info_tensor = info

        return obs_tensor, reward_tensor, terminated_tensor, truncated_tensor, info_tensor


class VectorizedGymnasiumWrapper:
    """Composition wrapper for ``SyncVectorEnv`` + torch + dict ``obs``."""

    def __init__(
        self, env, num_envs=1, max_episode_length=1000, device="cpu", dtype=torch.float32
    ):
        if not isinstance(env, gym.vector.VectorEnv):
            raise ValueError("VectorizedGymnasiumWrapper can only wrap VectorEnv instances.")

        self.env = env
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device
        self.dtype = dtype
        self.current_iteration = 0
        self._seed = None

        if not hasattr(env.unwrapped, "num_envs"):
            env.unwrapped.num_envs = num_envs
        if not hasattr(env.unwrapped, "max_episode_length"):
            env.unwrapped.max_episode_length = max_episode_length
        if not hasattr(env.unwrapped, "device"):
            env.unwrapped.device = device
        if not hasattr(env.unwrapped, "current_iteration"):
            env.unwrapped.current_iteration = 0
        if not hasattr(env.unwrapped, "seed"):

            def seed_method(seed):
                self._seed = seed
                return [seed]

            env.unwrapped.seed = seed_method

        self.action_space = env.action_space

        if not isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({"obs": env.observation_space})
        elif "obs" not in env.observation_space.spaces:
            first_key = list(env.observation_space.spaces.keys())[0]
            self.observation_space = gym.spaces.Dict(
                {"obs": env.observation_space[first_key]}
            )
        else:
            self.observation_space = env.observation_space

    def _to_tensor(self, x, add_batch_dim=False, dtype=None):
        if isinstance(x, dict):
            return {k: self._to_tensor(v, add_batch_dim, dtype) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_tensor(item, add_batch_dim, dtype) for item in x)
        tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        if not isinstance(tensor, torch.Tensor):
            td = dtype if dtype is not None else self.dtype
            tensor = torch.tensor(tensor, dtype=td)
        target_dtype = dtype if dtype is not None else self.dtype
        return tensor.to(dtype=target_dtype, device=self.device)

    def _ensure_dict_format(self, obs):
        if isinstance(obs, dict):
            if "obs" not in obs:
                first_key = list(obs.keys())[0]
                return {"obs": obs[first_key]}
            return obs
        return {"obs": obs}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        obs_tensor = self._ensure_dict_format(self._to_tensor(obs, add_batch_dim=False))

        if isinstance(info, dict):
            info_tensor = {
                k: self._to_tensor(v, add_batch_dim=False)
                if isinstance(v, (np.ndarray, list))
                else v
                for k, v in info.items()
            }
        else:
            info_tensor = info

        return obs_tensor, info_tensor

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions

        obs, reward, terminated, truncated, info = self.env.step(actions_np)

        obs_tensor = self._ensure_dict_format(self._to_tensor(obs, add_batch_dim=False))
        reward_tensor = self._to_tensor(reward, add_batch_dim=False)
        terminated_tensor = self._to_tensor(
            terminated, add_batch_dim=False, dtype=torch.bool
        )
        truncated_tensor = self._to_tensor(
            truncated, add_batch_dim=False, dtype=torch.bool
        )

        if isinstance(info, dict):
            info_tensor = {
                k: self._to_tensor(v, add_batch_dim=False)
                if isinstance(v, (np.ndarray, list))
                else v
                for k, v in info.items()
            }
        else:
            info_tensor = info

        return obs_tensor, reward_tensor, terminated_tensor, truncated_tensor, info_tensor

    def seed(self, seed):
        self._seed = seed
        if hasattr(self.env.unwrapped, "seed"):
            self.env.unwrapped.seed(seed)
        return [seed]

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    @property
    def unwrapped(self):
        return self

    def __getattr__(self, name):
        return getattr(self.env, name)


class VideoRecordingWrapper:
    """Record first sub-env RGB frames (vectorized MPAIL setup) with optional wandb."""

    def __init__(
        self,
        env,
        video_folder: str,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int | None = None,
        enable_wandb: bool = False,
        fps: int = 50,
    ):
        self.env = env
        self.video_folder = video_folder
        self.step_trigger = step_trigger
        self.enable_wandb = enable_wandb
        self.fps = fps

        self.num_envs = env.num_envs
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.video_length = (
            video_length if video_length is not None else self.max_episode_length
        )

        self._step_count = 0
        self._iteration_step = 0
        self._recording = False
        self._frames: list = []
        self._video_count = 0

        print("[VIDEO] Video recording wrapper initialized:")
        print(f"        Folder: {video_folder}")
        print(
            f"        Video length: {self.video_length} (max_episode_length: {self.max_episode_length})"
        )
        print(f"        Step trigger: {step_trigger}")

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self._step_count += 1
        self._iteration_step += 1

        if self._iteration_step == 1:
            self._maybe_start_recording()

        if self._recording:
            self._capture_frame()

        if self._iteration_step >= self.max_episode_length:
            if self._recording and len(self._frames) > 0:
                self._save_video()
            self._iteration_step = 0

        return obs, reward, terminated, truncated, info

    def _maybe_start_recording(self):
        if self.step_trigger is None:
            return
        if not self._recording and self.step_trigger(self._step_count - 1):
            self._recording = True
            self._frames = []
            print(f"[VIDEO] Started recording at step {self._step_count - 1}")

    def _capture_frame(self):
        try:
            frame = self.env.render()
            if frame is None:
                return
            if isinstance(frame, tuple):
                frame = frame[0]
            elif hasattr(frame, "shape") and len(frame.shape) == 4:
                frame = frame[0]
            self._frames.append(frame)
        except Exception as e:
            print(f"[VIDEO DEBUG] Exception in _capture_frame: {e}")

    def _save_video(self):
        if len(self._frames) == 0:
            self._recording = False
            return

        try:
            import av as _av
        except ImportError:
            print("[WARN] PyAV not installed, skipping video save")
            self._recording = False
            self._frames = []
            return

        video_path = (
            f"{self.video_folder}/video_{self._video_count:04d}_step_{self._step_count}.mp4"
        )
        frames = np.array(self._frames)
        height, width = frames.shape[1:3]

        container = _av.open(video_path, mode="w")
        stream = container.add_stream("h264", rate=self.fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for frame_data in frames:
            frame = _av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

        print(f"[INFO] Saved video: {video_path} ({len(frames)} frames)")

        if self.enable_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({"video": wandb.Video(video_path, fps=self.fps)}, commit=False)
            except Exception as e:
                print(f"[WARN] Failed to log video to wandb: {e}")

        self._video_count += 1
        self._recording = False
        self._frames = []

    def seed(self, seed):
        return self.env.seed(seed)

    def close(self):
        if self._recording and len(self._frames) > 0:
            self._save_video()
        return self.env.close()

    def render(self):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)
