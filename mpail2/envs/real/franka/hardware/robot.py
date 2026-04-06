"""Adapted from `frankz/hardware/robot.py` in the frankz repository.

Source: https://github.com/memmelma/frankz/tree/mpail2
Author: memmelma

Modifications have been made to fit this project.
"""

import numpy as np
from franky import Affine, CartesianMotion, JointMotion, ReferenceType, RelativeDynamicsFactor, Robot
from scipy.spatial.transform import Rotation as R

from .grippers import FrankaGripper


class Franka:
    def __init__(
        self,
        robot_ip="172.16.0.2",
        control_mode="joint_position",
        dynamics_factor=0.1,
        reset_qpos=None,
    ):
        """
        Initialize the Franka robot server.

        Args:
            robot_ip: IP address of the Franka robot.
            control_mode: Control mode for the robot.
            dynamics_factor: Relative dynamics factor for velocity/acceleration/jerk (0-1).
            reset_qpos: Joint configuration used by ``reset()``.
        """

        assert control_mode in ["joint_position", "joint_delta", "cartesian_delta"], "Invalid control mode"
        self.control_mode = control_mode

        self.robot = Robot(robot_ip)
        self.robot.recover_from_errors()

        assert 0 <= dynamics_factor <= 1, "Invalid dynamics factor"
        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(
            velocity=dynamics_factor,
            acceleration=dynamics_factor / 2,
            jerk=dynamics_factor / 4,
        )

        self.gripper = FrankaGripper(robot_ip=robot_ip)

        self.reset_qpos = (
            np.array(
                [
                    0.08535511,
                    -0.8716217,
                    -0.11309249,
                    -2.92283648,
                    -0.13878459,
                    2.05753043,
                    0.88281583,
                ]
            )
            if reset_qpos is None
            else reset_qpos
        )

        self.min_qpos = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=np.float32,
        )
        self.max_qpos = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=np.float32,
        )

    def step(self, action, blocking=False):
        """Execute a joint-space action and optional gripper command."""

        self.robot.recover_from_errors()

        if self.control_mode == "cartesian_delta":
            robot_action = action[:6]
            gripper_action = action[6] if len(action) > 6 else None
        else:
            robot_action = action[:7]
            gripper_action = action[7] if len(action) > 7 else None

        if self.control_mode == "joint_position":
            motion = JointMotion(robot_action.tolist())
            self.robot.move(motion, asynchronous=not blocking)
        elif self.control_mode == "joint_delta":
            current_state = self.robot.current_joint_state
            current_qpos = np.array(current_state.position)
            target_qpos = np.clip(current_qpos + robot_action, self.min_qpos, self.max_qpos)
            motion = JointMotion(target_qpos.tolist())
            self.robot.move(motion, asynchronous=not blocking)
        elif self.control_mode == "cartesian_delta":
            motion = CartesianMotion(
                Affine(
                    robot_action[:3],
                    R.from_euler("xyz", robot_action[3:6]).as_quat(),
                ),
                ReferenceType.Relative,
            )
            self.robot.move(motion, asynchronous=not blocking)

        if gripper_action is not None:
            self.gripper.move(gripper_action, blocking=blocking)

    def reset(self):
        try:
            self.robot.recover_from_errors()
            motion = JointMotion(self.reset_qpos.tolist())
            self.robot.move(motion)
            self.gripper.reset()
        except Exception as e:
            print(f"Error during reset: {e}")
            raise

    def get_obs(self):
        """Get the current robot observation."""

        state = self.robot.state
        qpos = np.array(state.q, dtype=np.float32)
        qvel = np.array(state.dq, dtype=np.float32)

        cartesian_state = self.robot.current_cartesian_state
        end_effector_pose = cartesian_state.pose.end_effector_pose
        ee_pose = np.concatenate(
            [end_effector_pose.translation, end_effector_pose.quaternion],
            axis=0,
            dtype=np.float32,
        )

        gripper_state = np.array([self.gripper.get_position()], dtype=np.float32)

        return {
            "qpos": qpos,
            "qvel": qvel,
            "ee_pose": ee_pose,
            "gripper_state": gripper_state,
        }

    def close(self):
        """Clean up resources and close connections to the robot."""
        try:
            if hasattr(self, "gripper"):
                del self.gripper
                self.gripper = None
        except Exception as e:
            print(f"Warning: Error closing gripper: {e}")

        try:
            if hasattr(self, "robot"):
                del self.robot
                self.robot = None
        except Exception as e:
            print(f"Warning: Error closing robot: {e}")
