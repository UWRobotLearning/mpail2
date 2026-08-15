# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    bi_so_leader,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so_leader,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        self.teleop = make_teleoperator_from_config(config.teleop) if config.teleop else None
        if self.teleop:
            self.teleop.connect()
            self.logger.info("Teleoperator (leader arm) connected — running in teleoperation mode.")

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # Episode step counter (incremented in control_loop_action)
        self._step_count = 0

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    # ------------------------------------------------------------------
    # Home position / smooth reset  (mirrors HTTP robot_server feature)
    # ------------------------------------------------------------------

    def move_to_home_slowly(self) -> None:
        """Linearly interpolate from current joint positions to home over
        reset_steps steps at reset_hz Hz.  No-op if home_joints is empty."""
        if not self.config.home_joints:
            return

        joint_names = list(self.robot.action_features.keys())
        home_values = [float(v) for v in self.config.home_joints.split()]
        if len(home_values) != len(joint_names):
            raise ValueError(
                f"home_joints has {len(home_values)} values but robot has "
                f"{len(joint_names)} joints: {joint_names}"
            )

        raw_obs = self.robot.get_observation()
        current = [float(raw_obs[k]) for k in joint_names]

        dt = 1.0 / self.config.reset_hz
        self.logger.info(
            f"Moving to home over {self.config.reset_steps} steps "
            f"at {self.config.reset_hz} Hz "
            f"({self.config.reset_steps / self.config.reset_hz:.1f}s)"
        )
        for i in range(1, self.config.reset_steps + 1):
            t = i / self.config.reset_steps
            interp = {k: c + t * (g - c) for k, c, g in zip(joint_names, current, home_values)}
            self.robot.send_action(interp)
            time.sleep(dt)
        self.logger.info("Home position reached.")

    # ------------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------------

    def _stop_episode(self) -> None:
        """End the current episode loop without disconnecting the robot."""
        self.shutdown_event.set()

    def reset_for_new_episode(self) -> None:
        """Reset all in-memory state so a new episode can start cleanly.
        The robot stays connected; call move_to_home_slowly() separately."""
        self.shutdown_event.clear()
        self.action_queue = Queue()
        self.action_queue_size = []
        with self.latest_action_lock:
            self.latest_action = -1
        self.action_chunk_size = -1
        self.must_go = threading.Event()
        self.must_go.set()
        self._step_count = 0
        self.start_barrier = threading.Barrier(2)
        self.fps_tracker.reset()

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        if self.teleop:
            self.teleop.disconnect()
            self.logger.debug("Teleoperator disconnected")

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def plan_direct(self, obs: TimedObservation) -> list[TimedAction]:
        """Send an observation and receive actions in one synchronous gRPC call (Plan RPC).

        Alternative to the SendObservations + GetActions two-step.  Blocks until the
        server returns the planned actions.  Returns an empty list on failure.
        """
        observation_bytes = pickle.dumps(obs)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Plan",
            silent=True,
        )
        try:
            actions_chunk = self.stub.Plan(observation_iterator)
            if len(actions_chunk.data) == 0:
                return []
            timed_actions: list[TimedAction] = pickle.loads(actions_chunk.data)  # nosec
            return timed_actions
        except grpc.RpcError as e:
            self.logger.error(f"Plan RPC error: {e}")
            return []

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        # Episode step counter — stop after max_episode_steps if set
        self._step_count += 1
        if self.config.max_episode_steps and self._step_count >= self.config.max_episode_steps:
            self.logger.info(f"Episode done: reached {self._step_count} steps.")
            self._stop_episode()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        if not self.running:
            return
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action + 1, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except RuntimeError as e:
            msg = str(e)
            if "read thread is not running" in msg or "camera" in msg.lower():
                self.logger.warning(f"Camera error — attempting reconnect: {e}")
                try:
                    self.robot.disconnect()
                    time.sleep(1.0)
                    self.robot.connect()
                    self.logger.info("Robot reconnected successfully.")
                except Exception as reconnect_err:
                    self.logger.error(f"Reconnect failed — stopping episode: {reconnect_err}")
                    self._stop_episode()
            else:
                self.logger.error(f"Error in observation sender: {e}")
        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def _teleop_control_loop(self, task: str, verbose: bool = False) -> None:
        """Control loop for teleoperation mode.

        The leader arm drives the follower.  Every step the observation
        (including the actual teleop action under key 'teleop_action') is
        streamed to the gRPC server so planner_server.py can record it.
        No start_barrier is used — there is no action-receiver thread.
        """
        self.logger.info("Teleoperation control loop starting")

        while self.running:
            loop_start = time.perf_counter()

            # 1. Read follower observation
            try:
                raw_obs: RawObservation = self.robot.get_observation()
            except (ConnectionError, TimeoutError) as e:
                self.logger.warning(f"Observation read failed, skipping: {e}")
                time.sleep(self.config.environment_dt)
                continue

            # 2. Get and apply leader action
            teleop_action = self.teleop.get_action()
            self.robot.send_action(teleop_action)

            # 3. Attach task + teleop action so the server can record them
            raw_obs["task"] = task
            raw_obs["teleop_action"] = list(teleop_action.values())

            with self.latest_action_lock:
                latest = self.latest_action

            obs = TimedObservation(
                timestamp=time.time(),
                observation=raw_obs,
                timestep=max(latest + 1, 0),
                must_go=True,  # always send — no action queue to pace against
            )
            self.send_observation(obs)

            with self.latest_action_lock:
                self.latest_action += 1

            # 4. Episode step counter
            self._step_count += 1
            if self.config.max_episode_steps and self._step_count >= self.config.max_episode_steps:
                self.logger.info(f"Teleop episode done: reached {self._step_count} steps.")
                self._stop_episode()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(obs.get_timestamp())
                self.logger.info(
                    f"Teleop step #{self._step_count} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f}"
                )

            time.sleep(max(0.0, self.config.environment_dt - (time.perf_counter() - loop_start)))

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


def _run_one_episode(client: "RobotClient") -> None:
    """Run one episode — teleop or policy mode."""
    if client.teleop:
        # Teleop: single thread, no action receiver needed
        client._teleop_control_loop(task=client.config.task)
    else:
        # Policy: action receiver + control loop threads
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        client.control_loop(task=client.config.task)
        action_receiver_thread.join(timeout=2.0)


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = RobotClient(cfg)

    if not client.start():
        return

    episode_mode = cfg.max_episode_steps is not None or cfg.num_episodes is not None
    num_episodes = cfg.num_episodes  # None = run forever

    if not episode_mode:
        # ── Original continuous behaviour (unchanged) ──────────────────
        client.logger.info("Running in continuous mode (no episode limit).")
        if client.teleop:
            try:
                client._teleop_control_loop(task=cfg.task)
            finally:
                client.stop()
                client.logger.info("Client stopped")
        else:
            action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
            action_receiver_thread.start()
            try:
                client.control_loop(task=cfg.task)
            finally:
                client.stop()
                action_receiver_thread.join()
                if cfg.debug_visualize_queue_size:
                    visualize_action_queue_size(client.action_queue_size)
                client.logger.info("Client stopped")

    else:
        # ── Episode mode (new) ─────────────────────────────────────────
        ep = 0
        try:
            while num_episodes is None or ep < num_episodes:
                ep += 1
                client.logger.info(f"── Episode {ep} ──────────────────────────")
                if ep == 1:
                    client.move_to_home_slowly()
                _run_one_episode(client)
                if num_episodes is None or ep < num_episodes:
                    client.reset_for_new_episode()
                    # Block until the server finishes training on the completed episode.
                    client.logger.info("Waiting for server to complete training+reset...")
                    client.stub.Ready(services_pb2.Empty())
                    client.move_to_home_slowly()
                    # Pause after arm is home — robot holds still, no images sent.
                    if cfg.reset_pause_seconds > 0:
                        client.logger.info(f"Scene reset pause: {cfg.reset_pause_seconds:.0f}s ...")
                        time.sleep(cfg.reset_pause_seconds)
        finally:
            client.stop()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info(f"Done — {ep} episode(s) completed.")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
