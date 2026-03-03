"""
Utility functions and configurations for Gymnasium MPAIL

This module contains:
- Environment setup utilities (make_env, setup_environment, get_env_dimensions)
"""

import gymnasium as gym
from dataclasses import dataclass
from typing import Callable, Optional

from .wrappers import VectorizedGymnasiumWrapper

def make_env(env_id, idx, max_episode_length, device, render_mode=None, terminate_when_unhealthy=True):
    """
    Create a single environment with wrappers.
    Video recording is handled separately with CustomRecordVideo wrapper.
    
    Args:
        env_id: Gymnasium environment ID
        idx: Environment index (for vectorized environments)
        max_episode_length: Maximum episode length
        device: Device to use (cpu or cuda)
        render_mode: Render mode for the environment (e.g., "rgb_array" for video recording)
        terminate_when_unhealthy: Whether to terminate episode when agent is unhealthy.
            Set to False for imitation learning to allow agent to recover from bad states.
    
    Returns:
        A callable that creates the environment when called
    """
    def thunk():
        # Create environment with render_mode if specified (needed for video recording)
        # Pass terminate_when_unhealthy for MuJoCo environments that support it
        try:
            env = gym.make(env_id, render_mode=render_mode, terminate_when_unhealthy=terminate_when_unhealthy)
        except TypeError:
            # Environment doesn't support terminate_when_unhealthy parameter
            env = gym.make(env_id, render_mode=render_mode)
            if not terminate_when_unhealthy:
                print(f"[WARN] Environment {env_id} does not support terminate_when_unhealthy parameter")
        
        # Add standard wrappers
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


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
    """
    Create and wrap a gymnasium environment for MPAIL/MGAIL training.
    
    This function handles both single and vectorized environments and wraps
    them with the appropriate MPAIL-compatible wrapper.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "Ant-v5", "Hopper-v5")
        num_envs: Number of parallel environments (use >1 for vectorized)
        max_episode_length: Maximum steps per episode
        device: Device for tensor operations ("cuda" or "cpu")
        render_mode: Render mode for video recording (e.g., "rgb_array")
        video_folder: Directory to save videos (enables video recording if set)
        video_step_trigger: Function that takes step count and returns True to start recording
        video_length: Length of each video recording
        enable_wandb: Whether to log videos to wandb
        terminate_when_unhealthy: Whether to terminate when agent is unhealthy.
            Set to False for imitation learning to disable early termination.
    
    Returns:
        Wrapped environment compatible with MPAIL/MGAIL runners
    
    Example:
        >>> env = setup_environment("Ant-v5", num_envs=4, device="cuda")
        >>> obs = env.reset()
        >>> print(obs["obs"].shape)  # torch.Size([4, 27])
    """
    from .video_utils import CustomRecordVideo
    
    term_str = "enabled" if terminate_when_unhealthy else "DISABLED"
    print(f"[INFO] Creating {num_envs} environment(s): {env_id} with render mode: {render_mode}, termination: {term_str}")
    
    # Create environment functions
    env_fns = [
        make_env(env_id, i, max_episode_length, device, render_mode=render_mode, 
                 terminate_when_unhealthy=terminate_when_unhealthy)
        for i in range(num_envs)
    ]
    env = gym.vector.SyncVectorEnv(env_fns)
    
    # Wrap for MPAIL compatibility
    env = VectorizedGymnasiumWrapper(
        env,
        num_envs=num_envs,
        max_episode_length=max_episode_length,
        device=device,
    )
    
    # Apply video recording wrapper if enabled
    # Note: CustomRecordVideo needs a gym.Env, so we wrap the VectorizedGymnasiumWrapper
    # to make it compatible
    if video_folder is not None:
        import os
        os.makedirs(video_folder, exist_ok=True)
        
        # Make VectorizedGymnasiumWrapper compatible with gym.Wrapper by adding inheritance
        # We do this dynamically to avoid modifying the class definition
        env = _VideoRecordingWrapper(
            env,
            video_folder=video_folder,
            step_trigger=video_step_trigger,
            video_length=video_length,
            enable_wandb=enable_wandb,
            fps=video_fps,
        )
        print(f"[INFO] Video recording enabled: {video_folder}")

    return env


class _VideoRecordingWrapper:
    """
    Video recording wrapper that works with VectorizedGymnasiumWrapper.
    
    This wrapper handles video recording for vectorized environments by
    capturing frames from the first environment.
    
    Video recording is aligned with episode boundaries:
    - Recording starts at the beginning of an iteration (when step_trigger returns True)
    - Recording continues for exactly max_episode_length steps (one full iteration)
    - Video is saved at the end of the iteration
    """
    
    def __init__(
        self,
        env,
        video_folder: str,
        step_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = None,  # If None, uses max_episode_length
        enable_wandb: bool = False,
        fps: int = 50,
    ):
        self.env = env
        self.video_folder = video_folder
        self.step_trigger = step_trigger
        self.enable_wandb = enable_wandb
        self.fps = fps
        
        # Delegate attributes
        self.num_envs = env.num_envs
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Video length defaults to max_episode_length (one full iteration)
        self.video_length = video_length if video_length is not None else self.max_episode_length
        
        # Video state
        self._step_count = 0
        self._iteration_step = 0  # Steps within current iteration (0 to max_episode_length-1)
        self._recording = False
        self._frames = []
        self._video_count = 0
        
        print(f"[VIDEO] Video recording wrapper initialized:")
        print(f"        Folder: {video_folder}")
        print(f"        Video length: {self.video_length} (max_episode_length: {self.max_episode_length})")
        print(f"        Step trigger: {step_trigger}")

    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset iteration step counter at the start of each reset
        # (reset is called at the beginning of training)
        return obs, info
    
    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self._step_count += 1
        self._iteration_step += 1
        
        # Check if we should start recording at the beginning of an iteration
        # An iteration starts when iteration_step == 1 (just after incrementing from 0)
        if self._iteration_step == 1:
            self._maybe_start_recording()
        
        # Capture frame if recording
        if self._recording:
            self._capture_frame()
        
        # Check if we've reached the end of an iteration (max_episode_length steps)
        if self._iteration_step >= self.max_episode_length:
            # End of iteration - save video if recording
            if self._recording and len(self._frames) > 0:
                self._save_video()
            # Reset iteration step counter for next iteration
            self._iteration_step = 0
        
        return obs, reward, terminated, truncated, info
    
    def _maybe_start_recording(self):
        """Check if we should start recording at the beginning of this iteration."""
        if self.step_trigger is None:
            return
        # Use step_count - 1 since we want to check at the START of the iteration
        # (step_count has already been incremented)
        if not self._recording and self.step_trigger(self._step_count - 1):
            self._recording = True
            self._frames = []
            print(f"[VIDEO] Started recording at step {self._step_count - 1}")
    
    def _capture_frame(self):
        try:
            frame = self.env.render()
            if frame is None:
                print(f"[VIDEO DEBUG] render() returned None")
                return
            
            # SyncVectorEnv.render() returns different formats:
            # - tuple of frames (one per env) 
            # - numpy array with shape (num_envs, H, W, C)
            # We only record from the first environment
            if isinstance(frame, tuple):
                frame = frame[0]  # Take first env's frame
            elif hasattr(frame, 'shape') and len(frame.shape) == 4:
                frame = frame[0]  # Take first env's frame from batched array
            
            self._frames.append(frame)
        except Exception as e:
            print(f"[VIDEO DEBUG] Exception in _capture_frame: {e}")
    
    def _save_video(self):
        if len(self._frames) == 0:
            self._recording = False
            return
        
        import numpy as np
        try:
            import av
        except ImportError:
            print("[WARN] PyAV not installed, skipping video save")
            self._recording = False
            self._frames = []
            return
        
        video_path = f"{self.video_folder}/video_{self._video_count:04d}_step_{self._step_count}.mp4"
        
        # Save video using PyAV
        frames = np.array(self._frames)
        height, width = frames.shape[1:3]
        
        container = av.open(video_path, mode='w')
        stream = container.add_stream('h264', rate=self.fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        
        for frame_data in frames:
            frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        
        print(f"[INFO] Saved video: {video_path} ({len(frames)} frames)")
        
        # Log to wandb if enabled
        # IMPORTANT: Use commit=False to avoid creating extra wandb steps
        # The runner's wandb.log() call will commit these values along with training stats
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


def get_env_dimensions(
    env,
    num_envs: int = 1,
) -> tuple:
    """
    Extract observation and action dimensions from an environment.
    
    Handles both single and vectorized environments, including wrapped ones.
    
    Args:
        env: Gymnasium environment (can be wrapped)
        num_envs: Number of parallel environments
    
    Returns:
        Tuple of (obs_dim, action_dim)
    
    Example:
        >>> env = setup_environment("Ant-v5", num_envs=4)
        >>> obs_dim, action_dim = get_env_dimensions(env, num_envs=4)
        >>> print(f"Obs: {obs_dim}, Action: {action_dim}")  # Obs: 27, Action: 8
    """
    # Access the underlying SyncVectorEnv to get single (non-batched) observation space
    base_env = env.env if hasattr(env, 'env') else env
    
    # Use single_observation_space to get unbatched dimensions
    # This works for both SyncVectorEnv and VectorizedGymnasiumWrapper
    obs_space = getattr(base_env, 'single_observation_space', None)
    act_space = getattr(base_env, 'single_action_space', None)
    
    # Fallback to observation_space if single_* not available
    if obs_space is None:
        obs_space = env.observation_space
    if act_space is None:
        act_space = env.action_space
    
    # Get dimensions from the unbatched space
    if isinstance(obs_space, gym.spaces.Dict):
        # Sum all dimensions in dict observation space
        obs_dim = sum(
            space.shape[0] if len(space.shape) == 1 else space.shape[-1]
            for space in obs_space.spaces.values()
        )
    else:
        obs_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape[-1]
    
    action_dim = act_space.shape[0] if len(act_space.shape) == 1 else act_space.shape[-1]
    
    return obs_dim, action_dim

