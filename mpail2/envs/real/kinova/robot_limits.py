"""Defaults for Kinova / Gen3 ROS twist + TF stack."""

import numpy as np

STATE_DIM = 6
ACTION_DIM = 7

LOWER_LIMITS = np.array([0.2, -0.45, 0.145], dtype=np.float32)
UPPER_LIMITS = np.array([0.6, 0.2, 0.3], dtype=np.float32)
SAFE_HEIGHT = 0.16
Z_THRESHOLD = SAFE_HEIGHT + 0.01
MAX_COMMAND_SPEED = 0.1

MAX_EPISODE_STEPS = 700

RESET_TARGET_XYZ = np.array([0.461, -0.041, 0.17], dtype=np.float32)
RESET_TOLERANCE = 0.03
RESET_CONTROL_HZ = 10.0
RESET_MAX_TIME_S = 5.0

TWIST_TOPIC = "/twist_controller/commands"
TF_FRAME_BASE = "base_link"
TF_FRAME_EE = "end_effector_link"

FALLBACK_EE_POSE = np.array([0.480, -0.008, 0.209], dtype=np.float32)
FALLBACK_EE_VEL = np.zeros(3, dtype=np.float32)
