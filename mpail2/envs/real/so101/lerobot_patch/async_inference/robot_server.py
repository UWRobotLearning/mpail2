"""
Minimal FastAPI robot server (port 7070) for the LeRobot side of the RL pipeline.

The policy server (mpail2 env) calls these endpoints to drive the robot:

  GET  /state              → {"joints": [float, ...], "joint_names": [...]}
  POST /step  {"joints": [float, ...]}  → execute joint targets (degrees)
  POST /reset              → move slowly to HOME_POSITION_DEG and return state

Joint order is fixed by robot.action_features at startup and printed to the log.
The policy server must use the same order for every /step call.

Usage:
    python -m lerobot.async_inference.robot_server \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM1 \\
        --robot.id=Kid_right \\
        --host=0.0.0.0 \\
        --port=7070 \\
        --home_joints="16.44 -104.44 95.91 72.75 -0.04 1.96" \\
        --reset_steps=80 \\
        --reset_hz=50
"""

import logging
import time
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass

import draccus
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robot_server")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RobotServerConfig:
    robot: RobotConfig
    host: str = "0.0.0.0"
    port: int = 7070
    # Space-separated home joint positions in degrees (order matches joint_names at startup).
    home_joints: str = "16.44 -104.44 95.91 72.75 -0.04 1.96"
    # Number of interpolation steps when moving to home (more = slower/smoother).
    reset_steps: int = 80
    # Frequency (Hz) of each interpolation step during reset.
    reset_hz: float = 50.0


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    joints: list[float]

class StateResponse(BaseModel):
    joints: list[float]
    joint_names: list[str]


# ---------------------------------------------------------------------------
# Global robot state (set during lifespan startup)
# ---------------------------------------------------------------------------

_robot = None
_joint_names: list[str] = []        # internal keys: "shoulder_pan.pos", ...
_joint_names_bare: list[str] = []   # bare names for API: "shoulder_pan", ...
_home_joints: list[float] = []
_reset_steps: int = 80
_reset_hz: float = 50.0
_lock = threading.Lock()             # serialize robot bus access


def _obs_to_joints(raw_obs: dict) -> list[float]:
    return [float(raw_obs[k]) for k in _joint_names]


def _joints_to_action(joints: list[float]) -> dict:
    return {k: v for k, v in zip(_joint_names, joints)}


def _move_to_slowly(target: list[float], steps: int, hz: float) -> list[float]:
    """Interpolate from current position to target over `steps` steps at `hz` Hz."""
    dt = 1.0 / hz
    raw_obs = _robot.get_observation()
    current = _obs_to_joints(raw_obs)

    for i in range(1, steps + 1):
        t = i / steps
        interp = [c + t * (g - c) for c, g in zip(current, target)]
        _robot.send_action(_joints_to_action(interp))
        time.sleep(dt)

    raw_obs = _robot.get_observation()
    return _obs_to_joints(raw_obs)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _robot is not None:
        _robot.disconnect()
        logger.info("Robot disconnected.")


app = FastAPI(lifespan=lifespan)


@app.get("/state", response_model=StateResponse)
def get_state():
    """Return current joint positions in degrees."""
    with _lock:
        try:
            raw_obs = _robot.get_observation()
        except (ConnectionError, TimeoutError) as e:
            raise HTTPException(status_code=503, detail=f"Robot read failed: {e}")

    joints = _obs_to_joints(raw_obs)
    return StateResponse(joints=joints, joint_names=_joint_names_bare)


@app.post("/step", response_model=StateResponse)
def post_step(req: StepRequest):
    """Execute joint targets (degrees) and return the resulting state."""
    if len(req.joints) != len(_joint_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(_joint_names)} joints, got {len(req.joints)}. "
                   f"Joint order: {_joint_names_bare}",
        )

    with _lock:
        try:
            _robot.send_action(_joints_to_action(req.joints))
            raw_obs = _robot.get_observation()
        except (ConnectionError, TimeoutError) as e:
            raise HTTPException(status_code=503, detail=f"Robot error: {e}")

    joints = _obs_to_joints(raw_obs)
    return StateResponse(joints=joints, joint_names=_joint_names_bare)


@app.post("/reset", response_model=StateResponse)
def post_reset():
    """Move smoothly to home position and return resulting state."""
    with _lock:
        try:
            joints = _move_to_slowly(_home_joints, _reset_steps, _reset_hz)
        except (ConnectionError, TimeoutError) as e:
            raise HTTPException(status_code=503, detail=f"Robot error: {e}")

    logger.info(f"Reset to home: {dict(zip(_joint_names_bare, joints))}")
    return StateResponse(joints=joints, joint_names=_joint_names_bare)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: RobotServerConfig):
    global _robot, _joint_names, _joint_names_bare, _home_joints, _reset_steps, _reset_hz

    register_third_party_plugins()

    _robot = make_robot_from_config(cfg.robot)
    _robot.connect()

    _joint_names = list(_robot.action_features.keys())
    _joint_names_bare = [k.removesuffix(".pos") for k in _joint_names]

    home_values = [float(v) for v in cfg.home_joints.split()]
    if len(home_values) != len(_joint_names):
        raise ValueError(
            f"--home_joints has {len(home_values)} values but robot has {len(_joint_names)} joints: "
            f"{_joint_names_bare}"
        )
    _home_joints = home_values
    _reset_steps = cfg.reset_steps
    _reset_hz = cfg.reset_hz

    logger.info(f"Joint order:   {list(enumerate(_joint_names_bare))}")
    logger.info(f"Home position: {dict(zip(_joint_names_bare, _home_joints))}")
    logger.info(f"Reset motion:  {cfg.reset_steps} steps at {cfg.reset_hz} Hz "
                f"({cfg.reset_steps / cfg.reset_hz:.1f}s total)")
    logger.info(f"Robot server starting on http://{cfg.host}:{cfg.port}")

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
