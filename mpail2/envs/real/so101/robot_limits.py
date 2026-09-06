"""Physical constants and defaults for the SO-101 (SO-ARM101) 6-DOF robot arm.

Action space: Cartesian end-effector (x, y, z) + wrist_roll + gripper — 5-dim.
State space:  6 joint positions in degrees (unchanged).
IK converts the 5-dim policy action back to 6 joint targets before sending to the robot.
"""

import numpy as np

# ─── dimensions ───────────────────────────────────────────────────────────────
STATE_DIM      = 6   # 6 joint positions reported by LeRobot (raw, used for IK only)
ACTION_DIM     = 5   # [x, y, z, wrist_roll, gripper] — x,y,z is an ABSOLUTE target
                     # position (mapped from [-1,1] onto the EE_LOWER_M/EE_UPPER_M box),
                     # not a delta. wrist_roll and gripper are still delta/increment style.
EE_PROPRIO_DIM = 13  # proprioception: [ee_x, ee_y, ee_z, qx, qy, qz, qw, j0..j5]

# Joint names in the order LeRobot reports / planner_server.py parses them
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint position bounds (degrees).  Tune to your calibration.
JOINT_LOWER_DEG = np.array([
    -51.16,   # shoulder_pan
    -26.02,   # shoulder_lift — home(-6.022) - 20
    -90.0,    # elbow_flex   — demo reaches -82.9°; extended from -17.1
    -66.77,   # wrist_flex
    -60.0,    # wrist_roll
    3.0,      # gripper — current demo.pt reaches 4.02° (1st pct 4.02°); lowered from 9.0,
              # which was clipping ~5° of the demo's full-closed range every step.
], dtype=np.float32)

JOINT_UPPER_DEG = np.array([
    35.0,     # shoulder_pan
    65.0,     # shoulder_lift — demo reaches 62.2°; extended from 13.98
    40.0,     # elbow_flex   — current demo.pt reaches 38.68° (99th pct 31.05°); raised from
              # 27.0, which was clipping ~12° off the demo's most-bent elbow poses.
    104.0,    # wrist_flex   — demo reaches 103.30°; raised from 102.5
    60.0,     # wrist_roll
    120.0,    # gripper
], dtype=np.float32)

# Home position (degrees) — calibrated from physical robot.
# Order matches JOINT_NAMES: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
HOME_POSITION_DEG = np.array([-21.19, -5.41, 1.58, 99.47, -14.20, 28.66], dtype=np.float32)

# ─── Cartesian workspace ──────────────────────────────────────────────────────
# Home EE position computed from FK at HOME_POSITION_DEG (arm joints only).
HOME_EE_M = np.array([0.1861, 0.0592, 0.0878], dtype=np.float32)  # metres

# NOTE: so101_env.py's xyz action is now an ABSOLUTE position (mapped from [-1,1]
# onto the EE_LOWER_M/EE_UPPER_M box below), not a delta — MAX_DELTA_M is unused
# there. Still used by grpc_policy_server.py's (separate, untouched) delta-style
# action mapping: action_norm ∈ [-1,1]^3 is a per-step displacement scaled by
# MAX_DELTA_M * speed_scale, action=0 → hold current position.
MAX_DELTA_M = np.float32(0.03)   # metres per step at speed_scale=1.0

# Hard workspace bounds — EE is clipped to this box (xyz is an absolute position target,
# see so101_env.py's _action_norm_to_joints). Tightened to the current demo.pt's actual
# reach (recomputed via FK over all 5970 transitions, 2026-08-10 recording):
#   x: min=0.1619  1st pct=0.1666  99th pct=0.2618  max=0.2782
#   y: min=-0.0726 1st pct=-0.0657 99th pct=0.0586   max=0.0598
#   z: min=-0.0141 1st pct=-0.0098 99th pct=0.0905   max=0.0923
# x upper raised 0.24 -> 0.28 (demo max 0.2782 + margin) — the previous bound was
# clipping ~9.6% of this demo's transitions before the target EE position was ever
# reached, misaligning the demo's target with what the live env could actually hit.
EE_LOWER_M = np.array([0.14,  -0.08,  -0.02], dtype=np.float32)
EE_UPPER_M = np.array([0.28,   0.09,   0.10], dtype=np.float32)

# Kept for reference (not used in action mapping anymore).
EE_HALF_RANGE_M = (EE_UPPER_M - EE_LOWER_M) / 2

# Per-joint speed limit in degrees (legacy, used by so101_env.py joint-space path).
MAX_DELTA_DEG = np.array([10, 10, 10, 10, 10, 30], dtype=np.float32)

# Wrist roll action component (index 3): direct degree control.
HOME_WRIST_ROLL_DEG   = np.float32(-14.20)    # matches HOME_POSITION_DEG[4]
WRIST_ROLL_HALF_RANGE = np.float32(5.0)        # max °/step at speed_scale=1.0 (delta action)

# Gripper action component (index 4): direct degree control.
# Not read by the actual clip (so101_env.py clips against JOINT_LOWER_DEG[5]/
# JOINT_UPPER_DEG[5] instead) — kept in sync here so this isn't a stale/misleading
# duplicate for anything that does reference it.
GRIPPER_LOWER_DEG  = np.float32(3.0)
GRIPPER_UPPER_DEG  = np.float32(120.0)
HOME_GRIPPER_DEG   = np.float32(28.66)   # matches HOME_POSITION_DEG[5]
GRIPPER_HALF_RANGE = np.float32(30.0)   # max °/step at speed_scale=1.0 (delta action) — demo closes ~44° in 20 steps

# ─── timing ───────────────────────────────────────────────────────────────────
MAX_EPISODE_STEPS     = 200  # 300 client steps → 299 transitions, fills storage exactly
CONTROL_FREQUENCY_HZ  = 10.0   # env step rate

# ─── LeRobot robot-control HTTP server ────────────────────────────────────────
# The env talks to a lightweight HTTP server running on the LeRobot side.
# Implement the server (see so101_env.py docstring for the expected API).
DEFAULT_ROBOT_HOST = "127.0.0.1"
DEFAULT_ROBOT_PORT = 7070       # intentionally different from ExternalPlannerServer (8080)

# ─── Cameras ──────────────────────────────────────────────────────────────────
# CAM_KEY  = Web camera  (opencv, index_or_path=1, /dev/video1)
# CAM2_KEY = RealSense D435i  (serial 327743060231)
CAM_KEY  = "cam"    # Wrist camera — opencv /dev/video6
# 64x48 preserves the native 640x480 (4:3) capture aspect ratio exactly (10x isotropic
# downsample) — unlike 84x84, which stretches x/y by different factors (84/640 vs 84/480).
CAM_H    = 48
CAM_W    = 64
CAM_C    = 3

CAM2_KEY = "cam2"   # RealSense D435i — /dev/video0, serial 317422074482
CAM2_H   = 48
CAM2_W   = 64
CAM2_C   = 3
