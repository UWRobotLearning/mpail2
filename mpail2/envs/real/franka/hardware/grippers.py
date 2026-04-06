"""Adapted from `frankz/hardware/grippers.py` in the frankz repository.

Source: https://github.com/memmelma/frankz/tree/mpail2
Author: memmelma

Modifications have been made to fit this project.
"""

class FrankaGripper:
    """
    Franka gripper implementation using Franky.

    Close: 0. Open: 1.
    """

    def __init__(self, robot_ip="172.16.0.2"):
        super().__init__()
        from franky import Gripper as FrankyGripper

        self.gripper = FrankyGripper(robot_ip)
        self.gripper_speed = 0.1  # m/s
        self.gripper_force = 20.0  # N
        self.max_width = 0.08

    def reset(self):
        self.gripper.move(0.08, 0.1)

    def move(self, action, blocking=False):
        if blocking:
            if action < 0.5:
                self.gripper.grasp(
                    0.0,
                    self.gripper_speed,
                    self.gripper_force,
                    epsilon_outer=0.08,
                )
            else:
                self.gripper.move(0.08, self.gripper_speed)
        else:
            if action < 0.5:
                self.gripper.grasp_async(
                    0.0,
                    self.gripper_speed,
                    self.gripper_force,
                    epsilon_outer=0.08,
                )
            else:
                self.gripper.move_async(0.08, self.gripper_speed)

    def get_position(self):
        return self.gripper.width / self.max_width
