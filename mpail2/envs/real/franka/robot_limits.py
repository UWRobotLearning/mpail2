"""Robot and camera constants for the Franka real stack (no training / Isaac deps)."""

import numpy as np

RESET_QPOS = np.array(
    [0.09908148, -0.08514409, -0.09948055, -2.39459327, -0.01271878, 2.30942455, 0.79499885]
)
LOWER_LIMITS = np.array([0.4, -0.25, 0.06])
UPPER_LIMITS = np.array([0.65, 0.25, 0.25])
MAX_Z_FORCE = 3.7

TABLE_CAM_SERIAL = "825312072800"
WRIST_CAM_SERIAL = "825312070097"

STATE_DIM = 22
ACTION_DIM = 3
