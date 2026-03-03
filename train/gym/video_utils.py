"""
Video recording utilities for Gymnasium environments.

This module provides CustomRecordVideo for recording videos with optional wandb integration.
Users must initialize wandb before using enable_wandb=True.

Example:
    >>> import wandb
    >>> wandb.init(project="my-project")
    >>>
    >>> from gymnasium_mpail.video_utils import CustomRecordVideo, make_video_env
    >>> env = make_video_env("Ant-v5", "./videos", enable_wandb=True)
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import av
import gymnasium as gym
from gymnasium import logger
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers.rendering import RecordVideo


class CustomRecordVideo(RecordVideo):
    """
    Custom video recorder that extends gymnasium's RecordVideo with:
    - PyAV-based encoding for better compression
    - Optional wandb video logging
    - Configurable video resolution and quality (CRF)

    Note: If enable_wandb=True, wandb must be initialized before creating this wrapper.

    Args:
        env: The gymnasium environment to wrap
        video_folder: Directory to save videos
        episode_trigger: Function that returns True for episodes to record
        step_trigger: Function that returns True for steps to start recording
        video_length: Maximum length of each video (0 for full episodes)
        name_prefix: Prefix for video filenames
        fps: Frames per second (None uses env's metadata)
        disable_logger: Whether to disable gymnasium's video logger
        enable_wandb: Whether to log videos to wandb (requires wandb to be initialized)
        video_resolution: Output video resolution (width, height)
        video_crf: Constant Rate Factor for video quality (lower = better, 18-28 typical)
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        enable_wandb: bool = True,
        video_resolution: tuple[int, int] = (1280, 720),
        video_crf: int = 30,
    ):
        # Check wandb if enabled
        if enable_wandb:
            try:
                import wandb
                if wandb.run is None or wandb.run.name is None:
                    raise ValueError("wandb must be initialized before wrapping with enable_wandb=True.")
            except ImportError:
                raise ImportError("wandb is required when enable_wandb=True. Install with: pip install wandb")

        super().__init__(
            env,
            video_folder,
            episode_trigger,
            step_trigger,
            video_length,
            name_prefix,
            fps,
            disable_logger,
        )
        self.enable_wandb = enable_wandb
        self.video_resolution = video_resolution
        self.video_crf = video_crf

    def stop_recording(self):
        """Stop current recording and save the video using PyAV for better compression."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            output = av.open(path, "w")
            output_stream = output.add_stream(
                "libx264",
                rate=round(self.frames_per_sec),
            )
            output_stream.width, output_stream.height = self.video_resolution
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": str(self.video_crf), "preset": "veryslow"}

            for frame in self.recorded_frames:
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame = video_frame.reformat(
                    width=self.video_resolution[0], height=self.video_resolution[1]
                )
                packet = output_stream.encode(video_frame)
                output.mux(packet)

            # Flush encoder
            packet = output_stream.encode()
            output.mux(packet)
            output.close()

            # Log to wandb if enabled
            if self.enable_wandb:
                import wandb
                wandb.log({"Video": wandb.Video(path)}, commit=False)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None


def make_video_env(
    env_id: str,
    video_folder: str,
    episode_trigger: Optional[Callable[[int], bool]] = None,
    render_mode: str = "rgb_array",
    enable_wandb: bool = False,
    video_resolution: tuple[int, int] = (1280, 720),
    video_crf: int = 30,
    **env_kwargs
) -> gym.Env:
    """
    Create a gymnasium environment with video recording enabled.

    This is a convenience function that:
    1. Creates the environment with rgb_array render mode
    2. Wraps it with CustomRecordVideo for video recording

    Note: If enable_wandb=True, you must initialize wandb before calling this function.

    Args:
        env_id: Gymnasium environment ID (e.g., "Ant-v5")
        video_folder: Directory to save videos
        episode_trigger: Function(episode_id) -> bool, returns True for episodes to record
                        Default: record every 100th episode
        render_mode: Render mode (should be "rgb_array" for video recording)
        enable_wandb: Whether to log videos to wandb (requires wandb.init() first)
        video_resolution: Output video resolution (width, height)
        video_crf: Video quality (lower = better quality, larger file)
        **env_kwargs: Additional arguments passed to gym.make()

    Returns:
        Wrapped environment with video recording

    Example:
        >>> import wandb
        >>> wandb.init(project="my-project")
        >>> env = make_video_env(
        ...     "Ant-v5",
        ...     video_folder="./videos",
        ...     episode_trigger=lambda ep: ep % 50 == 0,
        ...     enable_wandb=True
        ... )
    """
    # Default episode trigger: every 100 episodes
    if episode_trigger is None:
        episode_trigger = lambda ep: ep % 100 == 0

    # Create output directory
    os.makedirs(video_folder, exist_ok=True)

    # Create base environment
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)

    # Wrap with video recorder
    env = CustomRecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=episode_trigger,
        enable_wandb=enable_wandb,
        video_resolution=video_resolution,
        video_crf=video_crf,
    )

    return env


__all__ = [
    "CustomRecordVideo",
    "make_video_env",
]
