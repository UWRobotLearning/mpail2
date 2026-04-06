"""Adapted from `frankz/network/client.py` in the frankz repository.

Source: https://github.com/memmelma/frankz/tree/mpail2
Author: memmelma

Modifications have been made to fit this project.
"""

import pickle
import time

import gymnasium as gym
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R

def _unnormalize(action, min_action, max_action):
    # Unnormalize from [-1, 1] back to [min_action, max_action]
    return (action + 1) / 2 * (max_action - min_action) + min_action


class FrankaClient(gym.Env):
    def __init__(
        self,
        server_address="tcp://localhost:5555",
        dynamics_factor=0.2,
        control_hz=10,
        reset_qpos=None,
        delta_action_min=np.ones(6) * -0.1,
        delta_action_max=np.ones(6) * 0.1,
        max_attempt_reset=5,
        device="cuda",
        max_episode_length=100,
    ):

        super().__init__()

        self.server_address = server_address
        self.dynamics_factor = dynamics_factor
        assert dynamics_factor >= 0 and dynamics_factor <= 1, "Invalid dynamics factor"
        self.control_hz = control_hz
        assert control_hz > 0, "Invalid control hz"
        self.reset_qpos = reset_qpos

        self.num_envs = 1
        self.delta_action_min = delta_action_min
        self.delta_action_max = delta_action_max
        self.max_episode_length = max_episode_length
        self.device = device
        self.latest_obs = None
        self.max_attempt_reset = max_attempt_reset

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_address)

        self.init_server()

        self.observation_space = gym.spaces.Dict(
            {k: gym.spaces.Box(low=-np.inf, high=np.inf, shape=v.shape) for k, v in self.get_obs().items()}
        )
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))

        print(f"Connected to Franka server at {server_address}")

    def init_server(self):
        self._send_command(
            "init",
            {
                "control_mode": "cartesian_delta",
                "dynamics_factor": self.dynamics_factor,
                "reset_qpos": self.reset_qpos,
            },
        )

    def _send_command(self, command, data=None):
        assert command in ["step", "reset", "get_obs", "init"], "Invalid command"

        message = {
            "command": command,
            "data": data,
        }

        self.socket.send(pickle.dumps(message))
        response = pickle.loads(self.socket.recv())

        if response.get("error"):
            raise RuntimeError(f"Server error: {response['error']}")

        return response.get("result")

    def step(self, action, blocking=False):
        start_time = time.perf_counter()

        action = action.copy()

        assert len(action) == 7, "Invalid action length, expected 7, got " + str(len(action))
        # map gripper from [-1., 1.] to [0., 1.] [close, open]
        gripper_action = (action[6] + 1) / 2

        if self.delta_action_min is not None and self.delta_action_max is not None:
            action[:6] = _unnormalize(action[:6], self.delta_action_min, self.delta_action_max)

        action = action[:6]  # only use first 6 dims for Cartesian delta

        # send action to server
        self._send_command(
            "step",
            {
                "action": np.concatenate([action, [gripper_action]]),
                "blocking": blocking,
            },
        )
        obs = self.get_obs()

        # ensure control loop runs at target hz
        end_time = time.perf_counter()
        sleep_time = 1.0 / self.control_hz - (end_time - start_time)
        if sleep_time < 0:
            print(f"Warning: Control loop running slower than target {self.control_hz}Hz (behind by {sleep_time:.4f}s)")
        time.sleep(max(0, sleep_time))

        info = {}
        info["client/step_time"] = np.round(time.perf_counter() - start_time, 3)

        return obs, 0.0, False, False, info

    def reset(self, seed=None, options=None):
        info = {}
        for _ in range(self.max_attempt_reset):
            try:
                self._send_command("reset")
                break
            except RuntimeError as e:
                info["error_in_reset"] = True
                print("RuntimeError in reset: ", e)
                print("Resetting environment")
                self.init_server()
        return self.get_obs(), info

    def get_obs(self):
        """
        Get the current observation from the robot.

        Returns:
            dict containing:
                - qpos: joint positions (7,)
                - qvel: joint velocities (7,)
                - ee_pose: end-effector pose as [x, y, z, rx, ry, rz] (6,)
                - gripper_state: gripper state (1,)
        """
        obs = self._send_command("get_obs")
        ee_rot_vec = R.from_quat(obs["ee_pose"][3:7], scalar_first=False).as_rotvec()
        obs["ee_pose"] = np.concatenate([obs["ee_pose"][:3], ee_rot_vec])
        self.latest_obs = obs
        return obs

    def get_latest_obs(self):
        if self.latest_obs is None:
            return self.get_obs()
        return self.latest_obs

    def close(self):
        self.socket.close()
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
