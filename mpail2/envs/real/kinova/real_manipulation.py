"""ROS 2 twist + TF Kinova backend as a Gymnasium environment (numpy obs for inner API)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .robot_limits import (
    ACTION_DIM,
    FALLBACK_EE_POSE,
    FALLBACK_EE_VEL,
    LOWER_LIMITS,
    MAX_COMMAND_SPEED,
    MAX_EPISODE_STEPS,
    RESET_CONTROL_HZ,
    RESET_MAX_TIME_S,
    RESET_TARGET_XYZ,
    RESET_TOLERANCE,
    STATE_DIM,
    TF_FRAME_BASE,
    TF_FRAME_EE,
    TWIST_TOPIC,
    UPPER_LIMITS,
    Z_THRESHOLD,
)

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node

    ROS2_AVAILABLE = True
except ImportError as e:
    ROS2_AVAILABLE = False
    rclpy = None  # type: ignore
    Node = None  # type: ignore
    Twist = None  # type: ignore
    print(f"[WARNING] ROS2 import failed: {e}")


@dataclass(kw_only=True)
class ManipulationObservation:
    end_effector_pose: np.ndarray
    end_effector_velocity: np.ndarray


@dataclass(kw_only=True)
class ManipulationAction:
    end_effector_velocity: Optional[np.ndarray] = None
    gripper_command: Optional[float] = None
    control_mode: str = "twist"


class KinovaManipulationEnv(gym.Env):
    """Gym env: TF EE pose + twist commands (Kinova / similar)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        control_frequency: float = 10.0,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        device: str = "cuda",
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        super().__init__()
        self.control_frequency = control_frequency
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.max_episode_steps = max_episode_steps

        self.num_envs = 1
        self.episode_count = 0

        self.observation_space = spaces.Dict(
            {
                "proprioception": spaces.Box(
                    -np.inf, np.inf, shape=(self.num_envs, state_dim), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, action_dim), dtype=np.float32
        )

        print("[INFO] Initialize ROS2 resources...")
        self._init_all_ros2_resources()

        print("[INFO] KinovaManipulationEnv initialized")
        print(f"[INFO] - Control frequency: {control_frequency} Hz")
        print(f"[INFO] - State / action dim: {state_dim} / {action_dim}")

    def _init_all_ros2_resources(self):
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from rclpy.node import Node
            from tf2_ros import Buffer, TransformListener
        except ImportError:
            self._current_pose = FALLBACK_EE_POSE.copy()
            self._current_velocity = FALLBACK_EE_VEL.copy()
            self._twist_pub = None
            self._main_node = None
            self._tf_thread = None
            self._tf_reader_running = False
            return

        if not rclpy.ok():
            rclpy.init()

        self._main_node = Node("real_env_unified_node")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._main_node)

        self._twist_pub = self._main_node.create_publisher(Twist, TWIST_TOPIC, 10)

        self._current_pose = FALLBACK_EE_POSE.copy()
        self._current_velocity = FALLBACK_EE_VEL.copy()
        self._last_pose = FALLBACK_EE_POSE.copy()
        self._last_time = time.time()

        self._tf_reader_running = True
        self._start_ros2_background_worker()

    def _start_ros2_background_worker(self):
        import threading

        def ros2_worker():
            while self._tf_reader_running:
                try:
                    transform = self._tf_buffer.lookup_transform(
                        TF_FRAME_BASE, TF_FRAME_EE, rclpy.time.Time()
                    )

                    translation = transform.transform.translation
                    x, y, z = translation.x, translation.y, translation.z

                    current_time = time.time()
                    new_pose = np.array([x, y, z], dtype=np.float32)
                    dt = current_time - self._last_time

                    if dt > 0:
                        velocity = (new_pose - self._last_pose) / dt
                        velocity = np.clip(velocity, -2.0, 2.0)
                        self._current_velocity = velocity

                    self._last_pose = self._current_pose.copy()
                    self._current_pose = new_pose
                    self._last_time = current_time

                except Exception:
                    pass

                try:
                    rclpy.spin_once(self._main_node, timeout_sec=0.01)
                except Exception:
                    pass

                time.sleep(0.01)

        self._tf_thread = threading.Thread(target=ros2_worker, daemon=True)
        self._tf_thread.start()

    def _init_twist_publisher(self):
        if getattr(self, "_twist_pub", None) is not None or getattr(
            self, "_main_node", None
        ) is not None:
            return
        self._init_all_ros2_resources()

    def close(self):
        self._tf_reader_running = False
        if hasattr(self, "_tf_thread") and self._tf_thread is not None:
            self._tf_thread.join(timeout=2.0)
        if hasattr(self, "_main_node") and self._main_node is not None:
            try:
                self._main_node.destroy_node()
            except Exception:
                pass
            self._main_node = None
        super().close()

    def read_robot_state(self) -> ManipulationObservation:
        return ManipulationObservation(
            end_effector_pose=self._current_pose.copy(),
            end_effector_velocity=self._current_velocity.copy(),
        )

    def execute_robot_action(self, action: ManipulationAction):
        if not ROS2_AVAILABLE or Twist is None:
            print("[WARNING] ROS2 not available, skipping action execution")
            return

        if getattr(self, "_twist_pub", None) is None:
            print("[ERROR] Twist publisher not initialized")
            return

        if action.control_mode == "twist" and action.end_effector_velocity is not None:
            linear_vel = (
                action.end_effector_velocity[:3]
                if len(action.end_effector_velocity) >= 3
                else action.end_effector_velocity
            )

            twist_msg = Twist()
            twist_msg.linear.x = float(linear_vel[0]) if len(linear_vel) > 0 else 0.0
            twist_msg.linear.y = float(linear_vel[1]) if len(linear_vel) > 1 else 0.0
            twist_msg.linear.z = float(linear_vel[2]) if len(linear_vel) > 2 else 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = 0.0

            self._twist_pub.publish(twist_msg)

    def send_stop_command(self):
        if not ROS2_AVAILABLE or Twist is None:
            print("[WARNING] ROS2 not available, skipping stop command")
            return

        if getattr(self, "_twist_pub", None) is None:
            print("[ERROR] Twist publisher not initialized")
            return

        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0

        self._twist_pub.publish(twist_msg)
        print("[INFO] Stop command sent")

    def obs_to_state_vector(self, obs: ManipulationObservation) -> np.ndarray:
        state_vector = np.concatenate(
            [obs.end_effector_pose.astype(np.float32), np.zeros(3, dtype=np.float32)]
        )
        if hasattr(self, "_current_velocity_for_next_step"):
            state_vector[3:6] = self._current_velocity_for_next_step
        return state_vector.astype(np.float32)

    def action_vector_to_command(
        self, action_vector: np.ndarray, current_state: Optional[np.ndarray] = None
    ) -> ManipulationAction:
        if len(action_vector) < 3:
            raise ValueError(f"Need at least 3 action dims, got {len(action_vector)}")

        end_effector_velocity = action_vector[:3].copy().astype(np.float32)

        if current_state is not None and len(current_state) >= 3:
            current_xyz = current_state[:3]
            if current_xyz[2] > Z_THRESHOLD and end_effector_velocity[2] > 0:
                end_effector_velocity[2] *= -1.0

            lower_violation = (current_xyz <= LOWER_LIMITS) & (end_effector_velocity < 0)
            upper_violation = (current_xyz >= UPPER_LIMITS) & (end_effector_velocity > 0)
            end_effector_velocity[lower_violation | upper_violation] = 0.0

        end_effector_velocity = np.clip(
            end_effector_velocity, -MAX_COMMAND_SPEED, MAX_COMMAND_SPEED
        )
        full_velocity = np.zeros(6, dtype=np.float32)
        full_velocity[:3] = end_effector_velocity
        gripper_command = 0.0 if len(action_vector) >= 7 else None
        return ManipulationAction(
            end_effector_velocity=full_velocity,
            gripper_command=gripper_command,
            control_mode="twist",
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if hasattr(self, "_current_velocity_for_next_step"):
            delattr(self, "_current_velocity_for_next_step")

        target_position = RESET_TARGET_XYZ.copy()
        tolerance = RESET_TOLERANCE
        control_frequency = RESET_CONTROL_HZ
        max_velocity = MAX_COMMAND_SPEED

        self.episode_count += 1

        axis_reached = [False, False, False]
        axis_side: list[Optional[int]] = [None, None, None]

        if getattr(self, "_twist_pub", None) is None:
            self._init_twist_publisher()

        reset_start_time = time.time()

        while not all(axis_reached):
            if time.time() - reset_start_time > RESET_MAX_TIME_S:
                self.send_stop_command()
                break

            obs = self.read_robot_state()
            current_position = obs.end_effector_pose
            position_error = target_position - current_position
            velocity_command = np.zeros(3, dtype=np.float32)

            for axis in range(3):
                if axis_reached[axis]:
                    continue

                error = position_error[axis]

                if axis_side[axis] is None:
                    if abs(error) > tolerance:
                        axis_side[axis] = 1 if error > 0 else 0

                current_error_sign = 1 if error > 0 else 0

                if abs(error) <= tolerance:
                    axis_reached[axis] = True
                elif axis_side[axis] is not None and current_error_sign != axis_side[axis]:
                    axis_reached[axis] = True
                else:
                    velocity_command[axis] = np.sign(error) * min(max_velocity, abs(error))

            if not all(axis_reached):
                action = ManipulationAction(
                    end_effector_velocity=np.concatenate(
                        [velocity_command, np.zeros(3, dtype=np.float32)]
                    ),
                    control_mode="twist",
                )
                self.execute_robot_action(action)

            time.sleep(1.0 / control_frequency)

        self.send_stop_command()

        final_obs = self.read_robot_state()
        final_state = self.obs_to_state_vector(final_obs)

        obs_dict = {"proprioception": final_state.reshape(1, -1).astype(np.float32)}
        info: Dict[str, Any] = {
            "final_position": final_obs.end_effector_pose.tolist(),
            "target_position": target_position.tolist(),
            "reset_time": time.time() - reset_start_time,
        }
        return obs_dict, info

    def step(
        self, action: np.ndarray | torch.Tensor
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        current_obs = self.read_robot_state()
        current_state = self.obs_to_state_vector(current_obs)

        if isinstance(action, torch.Tensor):
            action_vector = action.detach().cpu().numpy().reshape(-1)
        else:
            action_vector = np.asarray(action, dtype=np.float32).reshape(-1)

        robot_action = self.action_vector_to_command(action_vector, current_state)
        self.execute_robot_action(robot_action)

        time.sleep(1.0 / self.control_frequency)

        next_obs = self.read_robot_state()
        next_state = self.obs_to_state_vector(next_obs)

        pos_change = (next_state[:3] - current_state[:3]) * 10.0
        next_state[3:6] = pos_change.copy()
        self._current_velocity_for_next_step = pos_change.copy()

        reward = 0.0

        obs_dict = {"proprioception": next_state.reshape(1, -1).astype(np.float32)}
        info = {
            "current_position": current_obs.end_effector_pose.tolist(),
            "next_position": next_obs.end_effector_pose.tolist(),
            "action_executed": action_vector.tolist(),
        }

        return obs_dict, float(reward), False, False, info
