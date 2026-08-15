# ExternalPlannerServer
#
# A drop-in replacement for PolicyServer that, instead of running a local neural-network policy,
# forwards every observation to an EXTERNAL planner over HTTP (JSON) and returns whatever
# actions that planner sends back.
#
# Usage:
#   python -m lerobot.async_inference.external_planner_server \
#       --host=0.0.0.0 \
#       --port=8080 \
#       --planner_url=http://192.168.1.50:9090/plan
#
# The robot-side RobotClient is UNCHANGED — point it at this server's host:port as usual.

import dataclasses
import json
import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from queue import Empty, Queue

import grpc
import numpy as np
import requests

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks

from .helpers import TimedAction, TimedObservation, get_logger

logger = get_logger("external_planner_server")


# ---------------------------------------------------------------------------
# What is SENT to the external planner  (HTTP POST body, JSON)
# ---------------------------------------------------------------------------
#
# {
#   "timestep": int,           -- which control step this observation belongs to
#   "timestamp": float,        -- unix time (seconds) when the robot captured this obs
#   "must_go": bool,           -- True when the robot's action queue is empty and needs actions NOW
#   "observation": {
#       "task": "string",               -- task description passed from robot client
#       "shoulder_pan.pos":  float,     -- raw motor position keys, one per joint
#       "shoulder_lift.pos": float,     --   key names match robot.action_features exactly
#       "elbow_flex.pos":    float,     --   (actual names depend on your robot model)
#       "wrist_flex.pos":    float,
#       "wrist_roll.pos":    float,
#       "gripper.pos":       float,
#       "front": [...],                 -- camera image as nested list (H, W, 3) uint8,
#                                       --   only included when --include_images is set
#       ... (one key per camera)
#   }
# }
#
# IMPORTANT: the order of motor keys in "observation" matches robot.action_features.
#            Your planner must return actions in that SAME order.
#
# ---------------------------------------------------------------------------
# What is EXPECTED back from the external planner  (HTTP response body, JSON)
# ---------------------------------------------------------------------------
#
# {
#   "actions": [
#       [float, float, ..., float],   -- action vector for timestep N   (one float per joint,
#       [float, float, ..., float],   --   in the same order as the motor keys above)
#       ...                           -- return as many steps as your planner wants
#   ]
# }
#
# Each inner list must have exactly action_dim floats (= number of motors).
# The first element is executed at the CURRENT timestep; the rest are queued in order.
# ---------------------------------------------------------------------------


def _timed_obs_to_json(obs: TimedObservation, include_images: bool = False) -> dict:
    """Serialize a TimedObservation to a plain JSON-safe dict."""
    raw = obs.get_observation()
    observation_payload = {}

    for key, value in raw.items():
        if key == "task":
            observation_payload["task"] = value
            continue

        if isinstance(value, np.ndarray):
            if not include_images and value.ndim == 3:
                # Skip image tensors by default to keep payload small.
                # Set include_images=True if your planner needs them.
                continue
            observation_payload[key] = value.tolist()
        elif hasattr(value, "tolist"):
            observation_payload[key] = value.tolist()
        else:
            observation_payload[key] = value

    return {
        "timestep": obs.get_timestep(),
        "timestamp": obs.get_timestamp(),
        "must_go": bool(obs.must_go),
        "observation": observation_payload,
    }


def _json_to_timed_actions(
    response: dict,
    base_timestep: int,
    base_timestamp: float,
    environment_dt: float,
) -> list[TimedAction]:
    """Parse the planner's JSON response into a list of TimedAction."""
    import torch

    raw_actions = response["actions"]  # list[list[float]]
    timed_actions = []
    for i, action_vec in enumerate(raw_actions):
        timed_actions.append(
            TimedAction(
                timestamp=base_timestamp + i * environment_dt,
                timestep=base_timestep + i,
                action=torch.tensor(action_vec, dtype=torch.float32),
            )
        )
    return timed_actions


class ExternalPlannerServer(services_pb2_grpc.AsyncInferenceServicer):
    """gRPC AsyncInference servicer that delegates planning to an external HTTP server."""

    def __init__(
        self,
        planner_url: str,
        environment_dt: float = 1.0 / 30,
        obs_queue_timeout: float = 1.0,
        include_images: bool = False,
        http_timeout: float = 5.0,
    ):
        """
        Args:
            planner_url:        Full URL of the external planner endpoint,
                                e.g. "http://192.168.1.50:9090/plan"
            environment_dt:     Control period in seconds (used to timestamp returned actions).
            obs_queue_timeout:  How long GetActions blocks waiting for an observation (seconds).
            include_images:     Whether to include camera images in the JSON payload.
                                Images are large; only enable if your planner needs them.
            http_timeout:       Requests timeout for the planner HTTP call (seconds).
        """
        self.planner_url = planner_url
        self.environment_dt = environment_dt
        self.obs_queue_timeout = obs_queue_timeout
        self.include_images = include_images
        self.http_timeout = http_timeout

        self.shutdown_event = threading.Event()
        self.observation_queue: Queue[TimedObservation] = Queue(maxsize=1)

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    # ------------------------------------------------------------------
    # gRPC handlers (called by the robot's RobotClient)
    # ------------------------------------------------------------------

    def Ready(self, request, context):  # noqa: N802
        """Handshake: robot checks the server is alive."""
        logger.info(f"Robot client connected: {context.peer()}")
        self.shutdown_event.clear()
        self.observation_queue = Queue(maxsize=1)
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Robot sends policy metadata on startup.

        We log it but do not load any local model — the external planner owns inference.
        If your external planner needs these details, forward them here.
        """
        policy_specs = pickle.loads(request.data)  # nosec
        logger.info(
            f"Policy instructions received (not used locally): "
            f"policy_type={policy_specs.policy_type}, "
            f"pretrained={policy_specs.pretrained_name_or_path}, "
            f"actions_per_chunk={policy_specs.actions_per_chunk}"
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Robot streams its latest observation here.

        The observation is deserialized and placed in the single-slot queue,
        replacing any stale observation that has not been processed yet.
        """
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, logger
        )
        timed_obs: TimedObservation = pickle.loads(received_bytes)  # nosec

        logger.debug(f"Received observation #{timed_obs.get_timestep()} (must_go={timed_obs.must_go})")

        # Replace stale observation if queue is full
        if self.observation_queue.full():
            try:
                self.observation_queue.get_nowait()
            except Empty:
                pass

        self.observation_queue.put(timed_obs)
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Robot polls here to receive its next action chunk.

        1. Wait for the latest observation.
        2. Forward it to the external planner as JSON.
        3. Parse the planner's response and return it to the robot.
        """
        try:
            obs: TimedObservation = self.observation_queue.get(timeout=self.obs_queue_timeout)
        except Empty:
            logger.debug("No observation available yet, returning empty actions.")
            return services_pb2.Actions(data=b"")

        logger.info(f"Forwarding observation #{obs.get_timestep()} to external planner at {self.planner_url}")

        payload = _timed_obs_to_json(obs, include_images=self.include_images)

        try:
            t0 = time.perf_counter()
            response = requests.post(
                self.planner_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.http_timeout,
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - t0
            logger.info(f"External planner responded in {elapsed * 1000:.1f}ms")
        except requests.RequestException as e:
            logger.error(f"External planner request failed: {e}")
            return services_pb2.Actions(data=b"")

        timed_actions = _json_to_timed_actions(
            response.json(),
            base_timestep=obs.get_timestep(),
            base_timestamp=obs.get_timestamp(),
            environment_dt=self.environment_dt,
        )

        actions_bytes = pickle.dumps(timed_actions)  # nosec
        return services_pb2.Actions(data=actions_bytes)

    def Plan(self, request_iterator, context):  # noqa: N802
        """Direct synchronous plan RPC: receive observation stream, call planner, return actions.

        Unlike the SendObservations + GetActions two-step, this is a single blocking call:
        the client streams observation chunks, the server forwards them to the HTTP planner,
        and returns the resulting actions directly in the response.
        """
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, logger
        )
        if not received_bytes:
            return services_pb2.Actions(data=b"")

        timed_obs: TimedObservation = pickle.loads(received_bytes)  # nosec
        logger.debug(f"Plan RPC: observation #{timed_obs.get_timestep()} (must_go={timed_obs.must_go})")

        payload = _timed_obs_to_json(timed_obs, include_images=self.include_images)

        try:
            t0 = time.perf_counter()
            response = requests.post(
                self.planner_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.http_timeout,
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - t0
            logger.info(f"Plan RPC: planner responded in {elapsed * 1000:.1f}ms for obs #{timed_obs.get_timestep()}")
        except requests.RequestException as e:
            logger.error(f"Plan RPC: planner request failed: {e}")
            return services_pb2.Actions(data=b"")

        timed_actions = _json_to_timed_actions(
            response.json(),
            base_timestep=timed_obs.get_timestep(),
            base_timestamp=timed_obs.get_timestamp(),
            environment_dt=self.environment_dt,
        )

        actions_bytes = pickle.dumps(timed_actions)  # nosec
        return services_pb2.Actions(data=actions_bytes)

    def stop(self):
        self.shutdown_event.set()
        logger.info("ExternalPlannerServer stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(
    host: str = "0.0.0.0",
    port: int = 8080,
    planner_url: str = "http://localhost:9090/plan",
    environment_dt: float = 1.0 / 30,
    obs_queue_timeout: float = 1.0,
    include_images: bool = False,
    http_timeout: float = 5.0,
):
    server_instance = ExternalPlannerServer(
        planner_url=planner_url,
        environment_dt=environment_dt,
        obs_queue_timeout=obs_queue_timeout,
        include_images=include_images,
        http_timeout=http_timeout,
    )

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_instance, grpc_server)
    grpc_server.add_insecure_port(f"{host}:{port}")

    logger.info(f"ExternalPlannerServer listening on {host}:{port}")
    logger.info(f"Forwarding observations to external planner at: {planner_url}")
    grpc_server.start()

    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        server_instance.stop()
        grpc_server.stop(grace=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--planner_url", default="http://localhost:9090/plan")
    parser.add_argument("--environment_dt", type=float, default=1.0 / 30)
    parser.add_argument("--obs_queue_timeout", type=float, default=1.0)
    parser.add_argument("--include_images", action="store_true")
    parser.add_argument("--http_timeout", type=float, default=5.0)
    args = parser.parse_args()

    serve(
        host=args.host,
        port=args.port,
        planner_url=args.planner_url,
        environment_dt=args.environment_dt,
        obs_queue_timeout=args.obs_queue_timeout,
        include_images=args.include_images,
        http_timeout=args.http_timeout,
    )
