"""
Minimal gRPC server that returns hardcoded actions derived from the robot's own
current joint positions.  No external planner, no neural network.

Purpose: verify that the full pipeline (RobotClient -> gRPC -> robot.send_action)
actually moves the robot before connecting a real planner.

HOW IT WORKS
------------
1. Robot sends its current joint positions as an observation.
2. This server reads those positions and builds a small action chunk:
     step 0 : current position          (hold still)
     step 1 : current position + DELTA  (tiny movement)
     step 2 : current position          (return)
3. RobotClient executes those actions in order -> robot should wiggle slightly.

USAGE
-----
Terminal 1 (this server):
    python -m lerobot.async_inference.hardcoded_action_server \
        --port=8080 \
        --delta=2.0 \
        --joint_index=0

Terminal 2 (real robot):
    python -m lerobot.async_inference.robot_client \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyUSB0 \
        --robot.id=my_robot \
        --server_address=127.0.0.1:8080 \
        --task="test" \
        --policy_type=act \
        --pretrained_name_or_path=test/model \
        --actions_per_chunk=3 \
        --policy_device=cpu \
        --client_device=cpu

SAFETY
------
- DELTA default is 2.0 degrees. Start small.
- The robot moves to current+DELTA then returns. Very short wiggle.
- Stop anytime with Ctrl+C on Terminal 2.
"""

import argparse
import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from queue import Empty, Queue

import grpc
import numpy as np
import torch

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks

from .helpers import TimedAction, TimedObservation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hardcoded_action_server")


class HardcodedActionServer(services_pb2_grpc.AsyncInferenceServicer):
    """
    Returns actions built from the robot's own current state.
    Proves the data pipeline works without any external planner.
    """

    def __init__(self, delta: float = 2.0, joint_index: int = 0, environment_dt: float = 1 / 30):
        """
        Args:
            delta:          How many degrees to move joint `joint_index` (default 2.0).
            joint_index:    Which joint to wiggle (0 = first joint in action_features order).
            environment_dt: Control period in seconds.
        """
        self.delta = delta
        self.joint_index = joint_index
        self.environment_dt = environment_dt

        self.shutdown_event = threading.Event()
        self.observation_queue: Queue[TimedObservation] = Queue(maxsize=1)

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    # ------------------------------------------------------------------
    # gRPC handlers
    # ------------------------------------------------------------------

    def Ready(self, request, context):  # noqa: N802
        logger.info(f"Robot client connected: {context.peer()}")
        self.shutdown_event.clear()
        self.observation_queue = Queue(maxsize=1)
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        policy_specs = pickle.loads(request.data)  # nosec
        logger.info(
            f"Policy instructions received | "
            f"policy_type={policy_specs.policy_type} | "
            f"actions_per_chunk={policy_specs.actions_per_chunk}"
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, logger
        )
        timed_obs: TimedObservation = pickle.loads(received_bytes)  # nosec

        raw_obs = timed_obs.get_observation()

        # Raw observation uses motor names as keys (e.g. "shoulder_pan.pos"),
        # not the dataset key "observation.state".
        motor_keys = [k for k in raw_obs if k.endswith(".pos")]
        logger.info(
            f"Obs #{timed_obs.get_timestep()} | "
            f"must_go={timed_obs.must_go} | "
            f"motor_keys={motor_keys} | "
            f"values={[round(float(raw_obs[k]), 2) for k in motor_keys]}"
        )

        if self.observation_queue.full():
            try:
                self.observation_queue.get_nowait()
            except Empty:
                pass
        self.observation_queue.put(timed_obs)
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        try:
            obs: TimedObservation = self.observation_queue.get(timeout=1.0)
        except Empty:
            logger.debug("No observation in queue, returning empty.")
            return services_pb2.Actions(data=b"")

        raw_obs = obs.get_observation()

        # Collect motor positions in sorted key order (must match robot.action_features order)
        motor_keys = [k for k in raw_obs if k.endswith(".pos")]

        if not motor_keys:
            logger.warning(f"No '.pos' keys found in observation. Keys present: {list(raw_obs.keys())}")
            return services_pb2.Actions(data=b"")

        current = np.array([float(raw_obs[k]) for k in motor_keys], dtype=np.float32)
        action_dim = len(current)

        # Build 3-step chunk:
        #   step 0: hold current position
        #   step 1: move joint[joint_index] by +delta
        #   step 2: return to current position
        step0 = current.copy()
        step1 = current.copy()
        step1[self.joint_index % action_dim] += self.delta
        step2 = current.copy()

        timed_actions = [
            TimedAction(
                timestamp=obs.get_timestamp() + i * self.environment_dt,
                timestep=obs.get_timestep() + i,
                action=torch.tensor(a, dtype=torch.float32),
            )
            for i, a in enumerate([step0, step1, step2])
        ]

        logger.info(
            f"Sending 3 actions for obs #{obs.get_timestep()} | "
            f"joint[{self.joint_index}]: "
            f"{current[self.joint_index % action_dim]:.2f} -> "
            f"{step1[self.joint_index % action_dim]:.2f} -> "
            f"{step2[self.joint_index % action_dim]:.2f}"
        )

        return services_pb2.Actions(data=pickle.dumps(timed_actions))  # nosec

    def stop(self):
        self.shutdown_event.set()
        logger.info("Server stopped.")


def serve(port: int = 8080, delta: float = 2.0, joint_index: int = 0, fps: int = 30):
    server_instance = HardcodedActionServer(
        delta=delta,
        joint_index=joint_index,
        environment_dt=1.0 / fps,
    )

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_instance, grpc_server)
    grpc_server.add_insecure_port(f"0.0.0.0:{port}")

    logger.info(f"HardcodedActionServer listening on port {port}")
    logger.info(f"Will wiggle joint[{joint_index}] by ±{delta} degrees each observation")
    grpc_server.start()

    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        server_instance.stop()
        grpc_server.stop(grace=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--delta", type=float, default=2.0,
                        help="Degrees to move the test joint (keep small!)")
    parser.add_argument("--joint_index", type=int, default=0,
                        help="Which joint to wiggle (0-indexed)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    serve(port=args.port, delta=args.delta, joint_index=args.joint_index, fps=args.fps)
