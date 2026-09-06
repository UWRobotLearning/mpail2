"""
Teleoperate a robot while streaming observations to an external HTTP planner.

The human stays in control (leader arm drives the follower). Observations are
forwarded to the planner at every step so the planner can observe what is
happening (e.g. for logging, imitation learning, or advisory output).
The planner's returned actions are logged but NOT applied to the robot.

Usage (single arm):
    python -m lerobot.async_inference.teleop_with_planner \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyUSB0 \
        --robot.id=my_robot \
        --teleop.type=so100_leader \
        --teleop.port=/dev/ttyUSB1 \
        --teleop.id=my_leader \
        --planner_url=http://localhost:9090/plan \
        --task="pick up the cup" \
        --fps=30

Notes:
  - No bridge server (external_planner_server) is needed.
  - Observations are POST-ed as JSON identical to the format external_planner_server uses.
  - HTTP calls are fire-and-forget (non-blocking) so they never stall the control loop.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass

import draccus
import numpy as np
import requests

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
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so_leader,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


@dataclass
class TeleopWithPlannerConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    planner_url: str = "http://localhost:9090/plan"
    task: str = "teleoperation"
    fps: int = 30
    # Seconds before giving up on a planner HTTP response
    http_timeout: float = 2.0
    # Whether to include camera images in the JSON payload (large!)
    include_images: bool = False


def _build_payload(
    raw_obs: dict,
    timestep: int,
    task: str,
    include_images: bool,
    teleop_action: dict | None = None,
) -> dict:
    observation_payload: dict = {"task": task}
    for key, value in raw_obs.items():
        if isinstance(value, np.ndarray):
            if not include_images and value.ndim == 3:
                continue
            observation_payload[key] = value.tolist()
        elif hasattr(value, "tolist"):
            observation_payload[key] = value.tolist()
        else:
            observation_payload[key] = value

    payload: dict = {
        "timestep": timestep,
        "timestamp": time.time(),
        "must_go": False,
        "observation": observation_payload,
    }
    # Include actual teleop action so planner_server records what the human did,
    # not the planner's own prediction.
    if teleop_action is not None:
        payload["teleop_action"] = [float(v) for v in teleop_action.values()]
    return payload


def _post_to_planner(url: str, payload: dict, timeout: float) -> None:
    """Called in a background thread — never blocks the control loop."""
    try:
        resp = requests.post(
            url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        actions = data.get("actions", [])
        logger.debug(f"Planner returned {len(actions)} action(s) for step #{payload['timestep']}")
    except requests.RequestException as e:
        logger.warning(f"Planner HTTP error at step #{payload['timestep']}: {e}")


@draccus.wrap()
def main(cfg: TeleopWithPlannerConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    robot.connect()
    teleop.connect()

    logger.info(f"Teleoperating robot — streaming observations to {cfg.planner_url}")
    logger.info("Press Ctrl+C to stop.")

    environment_dt = 1.0 / cfg.fps
    timestep = 0

    try:
        while True:
            loop_start = time.perf_counter()

            # 1. Capture observation from follower arm
            try:
                raw_obs = robot.get_observation()
                raw_obs["task"] = cfg.task
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Observation read failed at step #{timestep}, skipping: {e}")
                timestep += 1
                precise_sleep(max(0.0, environment_dt - (time.perf_counter() - loop_start)))
                continue

            # 2. Get teleop action from leader arm and apply it to the follower
            action = teleop.get_action()
            robot.send_action(action)

            # 3. Fire-and-forget POST to the planner (non-blocking)
            payload = _build_payload(raw_obs, timestep, cfg.task, cfg.include_images, action)
            threading.Thread(
                target=_post_to_planner,
                args=(cfg.planner_url, payload, cfg.http_timeout),
                daemon=True,
            ).start()

            timestep += 1
            precise_sleep(max(0.0, environment_dt - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        robot.disconnect()
        teleop.disconnect()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
