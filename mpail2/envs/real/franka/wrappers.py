"""Adapted from `frankz/perception/wrapper.py` in the frankz repository.

Source: https://github.com/memmelma/frankz/tree/mpail2
Author: memmelma

Modifications have been made to fit this project.
"""

import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from .hardware import gather_realsense_cameras
from .robot_limits import MAX_Z_FORCE


def low_pass_filter(value, prev_value, alpha=0.5):
    if prev_value is None:
        return value
    return alpha * value + (1 - alpha) * prev_value


class RealSenseWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        center_crop: bool = False,
        resize=None,
        normalize: bool = False,
        rename_camera_keys: Callable = None,
        buffer_size: int = 60,
        serial_numbers: list = None,
    ):
        super().__init__(env)
        if serial_numbers is None:
            serial_numbers = []

        self.center_crop = center_crop
        self.resize = resize
        self.normalize = normalize
        self.rename_camera_keys = rename_camera_keys

        self.cameras = gather_realsense_cameras(
            rgb=True, depth=False, align=None, serial_numbers=serial_numbers
        )
        assert len(self.cameras) > 0, "No RealSense cameras found"

        print(f"[RealSenseWrapper] {len(self.cameras)} camera(s) detected")

        self.buffers: Dict[str, deque] = {}
        self.threads = []
        self._stop_event = threading.Event()

        for cam in self.cameras:
            serial = cam._serial_number
            self.buffers[serial] = deque(maxlen=buffer_size)

            t = threading.Thread(
                target=self._camera_loop,
                args=(cam,),
                daemon=True,
            )
            t.start()
            self.threads.append(t)

        sample = self._wait_for_first_frame()
        img = self._process_image(sample["image"])

        low, high, dtype = self._obs_space_params(img)

        cam_spaces = {
            self._cam_key(k): gym.spaces.Box(
                low=low,
                high=high,
                shape=img.shape,
                dtype=dtype,
            )
            for k in self.buffers.keys()
        }

        combined = dict(self.observation_space.spaces)
        combined.update(cam_spaces)
        self.observation_space = gym.spaces.Dict(combined)

        print("[RealSenseWrapper] Warming up cameras...")
        for _ in range(3):
            self._get_synced_images(time.time())
            time.sleep(0.1)
        print("[RealSenseWrapper] Cameras ready.")

    def _camera_loop(self, camera):
        serial = camera._serial_number

        while not self._stop_event.is_set():
            data = camera.read_camera()

            rgb = data["rgb"]

            self.buffers[serial].append(
                {
                    "t": time.time(),
                    "image": rgb,
                }
            )
            time.sleep(0.001)

    def _wait_for_first_frame(self) -> Dict[str, Any]:
        while True:
            for buf in self.buffers.values():
                if len(buf) > 0:
                    return buf[-1]
            time.sleep(0.005)

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)

        if self.center_crop:
            size = min(img.shape[:2])
            y = (img.shape[0] - size) // 2
            x = (img.shape[1] - size) // 2
            img = img[y : y + size, x : x + size]

        if self.resize:
            img = cv2.resize(img, self.resize)

        if self.normalize:
            img = img / 255.0

        return img

    def _obs_space_params(self, img):
        if self.normalize:
            return 0.0, 1.0, np.float32
        return 0.0, 255.0, np.float32

    def _cam_key(self, serial: str) -> str:
        if self.rename_camera_keys:
            return self.rename_camera_keys(serial)
        return f"{serial}_image"

    def _get_synced_images(self, retrieve_time: float):
        obs = {}
        capture_time = {}

        for serial, buf in self.buffers.items():
            if not buf:
                continue

            best = min(buf, key=lambda x: abs(x["t"] - retrieve_time))
            obs[self._cam_key(serial)] = self._process_image(best["image"])
            capture_time[self._cam_key(serial)] = best["t"]

        return obs, capture_time

    def reset(self, seed=None, options=None):
        del seed, options  # FrankaClient reset has no seed API
        obs, info = self.env.reset()

        retrieve_time = time.time()
        cam_obs, capture_time = self._get_synced_images(retrieve_time)
        obs.update(cam_obs)

        info = dict(info)
        info["camera/capture_time"] = capture_time
        info["camera/retrieve_time"] = retrieve_time
        info["camera/delays"] = np.round([retrieve_time - t for t in capture_time.values()], 2)

        return obs, info

    def step(self, action):
        start_time = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)

        retrieve_time = time.time()
        cam_obs, capture_time = self._get_synced_images(time.time())
        obs.update(cam_obs)

        info = dict(info)
        info["camera/retrieve_time"] = retrieve_time
        info["camera/delays"] = np.round([retrieve_time - t for t in capture_time.values()], 2)
        info["camera/step_time"] = np.round(time.time() - start_time, 3)

        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop_event.set()
        for t in self.threads:
            t.join(timeout=2.0)

        print("[RealSenseWrapper] Closing cameras...")
        for cam in self.cameras:
            try:
                if hasattr(cam, "disable_camera"):
                    cam.disable_camera()
            except Exception as e:
                print(f"[WARNING] Error closing camera {cam._serial_number}: {e}")

        if hasattr(self.env, "close"):
            self.env.close()


class FrankaRealWrapper(gym.Wrapper):
    def __init__(
        self,
        env: Optional[gym.Env] = None,
        control_frequency: float = 15.0,
        state_dim: int = 14,
        action_dim: int = 3,
        device: str = "cuda",
        max_episode_length: int = 120,
        use_image: bool = True,
        lower_limits=None,
        upper_limits=None,
        use_low_pass_filter: bool = False,
    ):
        super().__init__(env)
        self.step_count = 0
        self.episode_count = 0
        self.control_frequency = control_frequency
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.num_envs = 1
        self.env.unwrapped.max_episode_length = max_episode_length
        self.max_episode_length = max_episode_length
        self.use_low_pass_filter = use_low_pass_filter

        self.lower_limits = np.array(lower_limits)
        self.upper_limits = np.array(upper_limits)
        self.max_vel_xy = 1.0
        self.max_vel_z = 1.0
        self.prev_pos = None

        self._cartesian_delta = np.zeros(7, dtype=np.float32)

        self.action_scale_xy = 0.5
        self.action_scale_z = 0.5

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, action_dim), dtype=np.float32
        )
        if use_image:
            self.observation_space = spaces.Dict(
                {
                    "proprioception": spaces.Box(
                        -np.inf, np.inf, (self.num_envs, state_dim), np.float32
                    ),
                    "table_cam": spaces.Box(0.0, 1.0, (self.num_envs, 64, 64, 3), np.float32),
                    "wrist_cam": spaces.Box(0.0, 1.0, (self.num_envs, 64, 64, 3), np.float32),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "proprioception": spaces.Box(
                        -np.inf, np.inf, (self.num_envs, state_dim), np.float32
                    ),
                }
            )

    def __del__(self):
        self.close()

    def _get_state(self, obs=None) -> dict:
        if obs is None:
            obs = self.env.get_obs()

        q_pos = obs["qpos"]
        q_vel = obs["qvel"]
        ee_pos = obs["ee_pose"][:3]
        ee_rot_vec = obs["ee_pose"][3:6]
        ee_quat = R.from_rotvec(ee_rot_vec).as_quat(scalar_first=False)
        gripper_state = np.array([obs.get("gripper_state", [0.0])[0]], dtype=np.float32)
        state = np.concatenate(
            [
                q_pos,
                q_vel,
                ee_pos,
                ee_quat,
                gripper_state,
            ]
        ).astype(np.float32)

        table_cam = obs.get("table_cam", np.zeros((64, 64, 3), dtype=np.float32))
        wrist_cam = obs.get("wrist_cam", np.zeros((64, 64, 3), dtype=np.float32))

        return {
            "proprioception": torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
            "table_cam": torch.tensor(table_cam, dtype=torch.float32, device=self.device).unsqueeze(0),
            "wrist_cam": torch.tensor(wrist_cam, dtype=torch.float32, device=self.device).unsqueeze(0),
        }

    def _process_action(self, action_vec: np.ndarray) -> np.ndarray:
        assert len(action_vec) == self.action_dim

        obs = self.env.unwrapped.get_latest_obs()
        ee_pos = obs["ee_pose"][:3]
        z_force = obs["ee_contact"][2]

        pos = action_vec[:3].copy()

        if z_force > MAX_Z_FORCE:
            pos[2] = max(0.2, pos[2])

        pos[:2] *= self.action_scale_xy
        pos[2] *= self.action_scale_z

        np.clip(pos[:2], -self.max_vel_xy, self.max_vel_xy, out=pos[:2])
        pos[2] = np.clip(pos[2], -self.max_vel_z, self.max_vel_z)

        lower_violation = (ee_pos <= self.lower_limits) & (pos < 0)
        upper_violation = (ee_pos >= self.upper_limits) & (pos > 0)

        mask = lower_violation | upper_violation
        if mask.any():
            pos[mask] = 0.0

        if self.use_low_pass_filter:
            pos = low_pass_filter(pos, self.prev_pos, alpha=0.75)
            self.prev_pos = pos.copy()

        pos[1] *= -1
        pos[2] *= -1

        cd = self._cartesian_delta
        cd[:] = 0.0
        cd[:3] = pos
        if self.action_dim == 3:
            cd[6] = 1.0
        else:
            assert len(action_vec) in (4, 7)
            cd[6] = action_vec[-1]

        return cd

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        del seed, options
        self.step_count = 0
        self.episode_count += 1

        reset_start = time.time()

        obs, info = self.env.reset()
        info["obs"] = obs
        time.sleep(1.0)

        obs = self._get_state(obs)

        info["reset_time"] = time.time() - reset_start

        return obs, info

    def step(self, action):
        start_time = time.time()

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()
        else:
            action = np.array(action).flatten()

        cartesian_delta = self._process_action(action)

        obs, reward, terminated, truncated, info = self.env.step(cartesian_delta)
        info["obs"] = obs

        obs_dict = self._get_state(obs)

        self.step_count += 1
        terminated = terminated or self.step_count >= self.max_episode_length

        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        terminated = torch.tensor([bool(terminated)], dtype=torch.bool, device=self.device)
        truncated = torch.tensor([truncated], dtype=torch.bool, device=self.device)

        info["action_executed"] = action
        info["mpail_env/step_time"] = np.round(time.time() - start_time, 3)

        return obs_dict, reward, terminated, truncated, info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
