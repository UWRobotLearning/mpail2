"""
LeRobot RL client — runs the episode loop between the robot server and the policy server.

Data flow per timestep:
  1. GET  robot_server:7070/state          → obs  (6 joint floats)
  2. POST policy_server:8765/act           ← {"obs": [6 floats]}
                                           → {"action": [6 floats]}
  3. POST robot_server:7070/step           ← {"joints": [6 floats]}
                                           → next_obs
  4. POST policy_server:8765/step_done     ← {"obs":      [6 floats],
                                               "action":   [6 floats],
                                               "next_obs": [6 floats],
                                               "done":     bool}

At episode end:
  5. POST robot_server:7070/reset          → go to home position
  6. POST policy_server:8765/reset         → triggers gradient update, clears storage

Usage:
    python -m lerobot.async_inference.rl_client \\
        --robot_url=http://127.0.0.1:7070 \\
        --policy_url=http://127.0.0.1:8765 \\
        --num_episodes=20 \\
        --max_steps=200 \\
        --control_hz=10.0
"""

import logging
import time
from dataclasses import dataclass, field

import draccus
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rl_client")


@dataclass
class RLClientConfig:
    robot_url: str = "http://127.0.0.1:7070"
    policy_url: str = "http://127.0.0.1:8765"
    num_episodes: int = 20
    max_steps: int = 200
    control_hz: float = 10.0
    # Seconds before giving up on an HTTP call
    timeout: float = 5.0
    # Home position (degrees) sent to the robot on every episode reset.
    # Override with --home_joints='[16.44,-104.44,95.91,72.75,-0.04,1.96]'
    home_joints: list = field(
        default_factory=lambda: [16.44, -104.44, 95.91, 72.75, -0.04, 1.96]
    )


# ---------------------------------------------------------------------------
# Thin HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: float) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _post(url: str, body: dict, timeout: float) -> dict:
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Per-server calls
# ---------------------------------------------------------------------------

def robot_get_state(robot_url: str, timeout: float) -> list[float]:
    data = _get(f"{robot_url}/state", timeout)
    return data["joints"]


def robot_step(robot_url: str, joints: list[float], timeout: float) -> list[float]:
    data = _post(f"{robot_url}/step", {"joints": joints}, timeout)
    return data["joints"]


def robot_reset(robot_url: str, timeout: float, home_joints: list[float] | None = None) -> list[float]:
    body = {"joints": home_joints} if home_joints else {}
    data = _post(f"{robot_url}/reset", body, timeout)
    return data["joints"]


def policy_act(policy_url: str, obs: list[float], timeout: float) -> list[float]:
    data = _post(f"{policy_url}/act", {"obs": obs}, timeout)
    return data["action"]


def policy_step_done(
    policy_url: str,
    obs: list[float],
    action: list[float],
    next_obs: list[float],
    done: bool,
    timeout: float,
    reward: float = 0.0,
) -> None:
    # reward=0.0 — MPAIL learns reward from demonstrations, not from the env signal
    _post(
        f"{policy_url}/step_done",
        {"obs": obs, "action": action, "next_obs": next_obs, "done": done, "reward": reward},
        timeout,
    )


def policy_reset(policy_url: str, timeout: float) -> None:
    _post(f"{policy_url}/reset", {}, timeout)


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(cfg: RLClientConfig, episode: int) -> int:
    """Run one episode. Returns the number of steps taken."""
    dt = 1.0 / cfg.control_hz

    # Start from a clean robot state
    obs = robot_reset(cfg.robot_url, cfg.timeout, cfg.home_joints)
    logger.info(f"Episode {episode} start | obs={[round(v,2) for v in obs]}")

    for step in range(cfg.max_steps):
        step_start = time.perf_counter()

        # 1. Get action from policy server
        try:
            action = policy_act(cfg.policy_url, obs, cfg.timeout)
        except requests.RequestException as e:
            logger.error(f"  /act failed at step {step}: {e}")
            break

        # 2. Send action to robot, get next observation
        try:
            next_obs = robot_step(cfg.robot_url, action, cfg.timeout)
        except requests.RequestException as e:
            logger.error(f"  /step failed at step {step}: {e}")
            break

        done = (step == cfg.max_steps - 1)

        # 3. Report transition to policy server
        try:
            policy_step_done(cfg.policy_url, obs, action, next_obs, done, cfg.timeout)
        except requests.RequestException as e:
            logger.warning(f"  /step_done failed at step {step}: {e}")

        logger.debug(
            f"  step {step:3d} | "
            f"action={[round(a,1) for a in action]} | "
            f"next_obs={[round(v,2) for v in next_obs]}"
        )

        obs = next_obs

        if done:
            break

        # Pace the loop to control_hz
        elapsed = time.perf_counter() - step_start
        time.sleep(max(0.0, dt - elapsed))

    steps_taken = step + 1
    logger.info(f"Episode {episode} done after {steps_taken} steps — sending /reset to policy server")

    # 4. Reset policy server (triggers gradient update)
    try:
        policy_reset(cfg.policy_url, cfg.timeout)
    except requests.RequestException as e:
        logger.warning(f"  /reset (policy) failed: {e}")

    return steps_taken


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: RLClientConfig):
    # Health check
    try:
        _get(f"{cfg.policy_url}/health", timeout=3.0)
        logger.info(f"Policy server reachable at {cfg.policy_url}")
    except requests.RequestException as e:
        logger.error(f"Policy server not reachable at {cfg.policy_url}: {e}")
        return

    try:
        robot_get_state(cfg.robot_url, timeout=3.0)
        logger.info(f"Robot server reachable at {cfg.robot_url}  home={cfg.home_joints}")
    except requests.RequestException as e:
        logger.error(f"Robot server not reachable at {cfg.robot_url}: {e}")
        return

    total_steps = 0
    for ep in range(1, cfg.num_episodes + 1):
        steps = run_episode(cfg, ep)
        total_steps += steps
        logger.info(f"Total steps so far: {total_steps}")

    logger.info(f"Done — {cfg.num_episodes} episodes, {total_steps} total steps.")


if __name__ == "__main__":
    main()
