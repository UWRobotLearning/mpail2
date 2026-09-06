"""
demo_recording_server.py — gRPC demo-recording server for SO-101.

Returns home-position actions so the robot holds still while you move it manually
or via a leader arm. Every (obs_t, obs_t+1) pair is saved to .npz files.

Run this in the `mpail2` conda env:
    conda activate mpail2
    python -m mpail2.envs.real.so101.training.demo_recording_server --collect_dir ./raw_demos2 --flush_every 50

After collecting, convert to .pt:
    python -m mpail2.envs.real.so101.training.convert --dirs raw_demos2 --out raw_demos2_master.pt

Robot client (lerobot env) — see mpail2/envs/real/so101/README.md for the full connect +
collect walkthrough and an explanation of each flag:
    python -m lerobot.async_inference.robot_client \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=Kid \\
        --robot.cameras="{cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, cam2: {type: intelrealsense, serial_number_or_name: 317422074482, width: 640, height: 480, fps: 30}}" \\
        --teleop.type=so100_leader \\
        --teleop.port=/dev/ttyACM1 \\
        --teleop.id=Mom \\
        --server_address=127.0.0.1:8080 \\
        --policy_type=act \\
        --pretrained_name_or_path=dummy \\
        --actions_per_chunk=1 \\
        --task="pick up the cup"
"""

import argparse
import io
import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional

import grpc
import numpy as np
import torch

from mpail2.envs.real.so101.transport import services_pb2, services_pb2_grpc
from lerobot.async_inference.helpers import TimedObservation, TimedAction

from mpail2.envs.real.so101 import OBS_KEY, HOME_POSITION_DEG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("demo_recording_server")


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA DISPLAY — live OpenCV window updated from a background thread
# ─────────────────────────────────────────────────────────────────────────────
_CAM_SAVE_PATH = "cameras_latest.png"

def _update_camera_display(image_arrays: dict) -> None:
    """Save both camera images side-by-side to cameras_latest.png.

    Open this file in VSCode — it auto-refreshes as the file updates.
    Layout: RealSense (cam2) on the left, wrist cam (cam) on the right.
    """
    try:
        import cv2
    except ImportError:
        return
    frames = []
    for key in ("cam2", "cam"):
        img = image_arrays.get(key)
        if img is None:
            continue
        img = np.array(img, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
        if img_u8.shape[2] == 3:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        img_u8 = cv2.resize(img_u8, (320, 240))
        frames.append(img_u8)
    if not frames:
        return
    composed = np.concatenate(frames, axis=1)
    cv2.imwrite(_CAM_SAVE_PATH, composed)


# ─────────────────────────────────────────────────────────────────────────────
# PICKLE — deserialize lerobot TimedObservation
# ─────────────────────────────────────────────────────────────────────────────

def _loads(data: bytes):
    return pickle.loads(data)  # nosec


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_raw_obs(raw_obs: dict) -> tuple[np.ndarray, dict]:
    joint_vals = []
    image_arrays = {}
    for key, value in raw_obs.items():
        if key in ("task", "teleop_action"):
            continue
        arr = np.array(value, dtype=np.float32)
        if arr.ndim == 3:
            # Camera image (H, W, C); strip lerobot prefix if present
            cam_key = key.removeprefix("observation.images.")
            image_arrays[cam_key] = arr
        else:
            # Scalar motor key ("shoulder_pan.pos", ...) or full "observation.state" array
            joint_vals.extend(arr.flatten().tolist())
    return np.array(joint_vals, dtype=np.float32), image_arrays


# ─────────────────────────────────────────────────────────────────────────────
# DEMO RECORDER  (same logic as planner_server.py _record_step / _flush)
# ─────────────────────────────────────────────────────────────────────────────

class DemoRecorder:
    """Buffers (obs_t, obs_t+1) pairs and flushes to .npz files."""

    def __init__(self, collect_dir: Path, flush_every: int = 200):
        self.collect_dir = collect_dir
        self.flush_every = flush_every
        collect_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._obs_buf:         List[np.ndarray] = []
        self._next_obs_buf:    List[np.ndarray] = []
        self._image_bufs:      Dict[str, List[np.ndarray]] = {}
        self._image_next_bufs: Dict[str, List[np.ndarray]] = {}
        self._pending_obs:     Optional[np.ndarray] = None
        self._pending_images:  Dict[str, np.ndarray] = {}
        self._file_idx = 0

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.max() <= 1.0:
            return (img * 255).clip(0, 255).astype(np.uint8)
        return img.clip(0, 255).astype(np.uint8)

    def record(self, obs: np.ndarray, images: Dict[str, np.ndarray]):
        images_u8 = {k: self._to_uint8(v) for k, v in images.items()}
        with self._lock:
            if self._pending_obs is not None:
                self._obs_buf.append(self._pending_obs)
                self._next_obs_buf.append(obs.copy())
                for cam, img in self._pending_images.items():
                    self._image_bufs.setdefault(cam, []).append(img)
                    self._image_next_bufs.setdefault(cam, []).append(
                        images_u8.get(cam, img).copy()
                    )

            self._pending_obs    = obs.copy()
            self._pending_images = images_u8

            if len(self._obs_buf) >= self.flush_every:
                self._flush()

    def flush(self):
        with self._lock:
            self._flush()

    def reset_pending(self):
        """Discard any half-open transition so the next call to record() starts fresh."""
        with self._lock:
            self._pending_obs    = None
            self._pending_images = {}

    def _flush(self):
        if not self._obs_buf:
            return
        obs_arr  = np.stack(self._obs_buf)
        nobs_arr = np.stack(self._next_obs_buf)
        save_data = {OBS_KEY: np.stack([obs_arr, nobs_arr], axis=1)}  # (N, 2, state_dim)
        for cam in self._image_bufs:
            imgs  = np.stack(self._image_bufs[cam])
            nimgs = np.stack(self._image_next_bufs[cam])
            save_data[cam] = np.stack([imgs, nimgs], axis=1)           # (N, 2, H, W, C)

        path = self.collect_dir / f"traj_{self._file_idx:04d}.npz"
        np.savez(str(path), **save_data)
        n = len(self._obs_buf)
        cams = list(self._image_bufs.keys())
        logger.info(f"[recorder] Saved {n} steps → {path}  cameras={cams or 'none'}")

        self._obs_buf.clear(); self._next_obs_buf.clear()
        self._image_bufs.clear(); self._image_next_bufs.clear()
        self._file_idx += 1
        self._pending_obs    = None
        self._pending_images = {}


# ─────────────────────────────────────────────────────────────────────────────
# gRPC SERVICER
# ─────────────────────────────────────────────────────────────────────────────

class DemoRecordingServicer(services_pb2_grpc.AsyncInferenceServicer):

    def __init__(
        self,
        recorder: DemoRecorder,
        max_episode_steps: int = 200,
        show_cameras: bool = False,
        episode_pause_seconds: float = 5.0,
    ):
        self.recorder    = recorder
        self.max_episode_steps = max_episode_steps
        self.show_cameras = show_cameras
        # Auto-advance episodes by step count instead of waiting for a client-driven
        # Ready() call (see _episode_end).
        self.episode_pause_seconds = episode_pause_seconds
        self._going_home = False
        if show_cameras:
            logger.info(f"[cameras] Saving live frames to {_CAM_SAVE_PATH} — open in VSCode to watch")

        self._obs_queue: Queue = Queue(maxsize=1)
        self.shutdown_event = threading.Event()

        self._episode_steps  = 0
        self._episode_count  = 0
        self._prev_timestep  = -1
        self._recording_paused = False   # True while waiting for Enter between episodes

    def Ready(self, request, context):
        # Kick off end-of-episode work (save, wait for Enter) in a background thread so
        # the client gets this response immediately and the arm can keep moving.
        # _recording_paused is set to True inside _episode_end() before any slow work
        # starts, so incoming observations are discarded until the next episode is
        # ready to record.
        if self._episode_steps > 0:
            logger.info(f"Ready: starting episode-end work in background (ep {self._episode_count + 1}, {self._episode_steps} steps)...")
            threading.Thread(target=self._episode_end, daemon=True).start()
        logger.info(f"Robot client connected: {context.peer()}")
        self.shutdown_event.clear()
        self._obs_queue = Queue(maxsize=1)
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        # No-op — demo recording always returns a single-action chunk regardless of
        # the client's requested actions_per_chunk. Implemented only so the RPC isn't
        # left unimplemented against lerobot's async_inference client protocol.
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):
        try:
            buf = io.BytesIO()
            n_chunks = 0
            for chunk in request_iterator:
                state = chunk.transfer_state
                if state == services_pb2.TransferState.TRANSFER_BEGIN:
                    buf.seek(0); buf.truncate(0)
                buf.write(chunk.data)
                n_chunks += 1
                if state == services_pb2.TransferState.TRANSFER_END:
                    break
            data = buf.getvalue()
            logger.debug(f"SendObservations: received {n_chunks} chunk(s), {len(data)} bytes")
        except Exception as e:
            logger.exception(f"SendObservations chunk read error: {e}")
            return services_pb2.Empty()

        # While paused between episodes, discard immediately — arm movement is
        # handled client-side (leader→follower), so we don't need to queue or record.
        if self._recording_paused:
            logger.debug("SendObservations: discarded (recording paused)")
            return services_pb2.Empty()

        try:
            timed_obs: TimedObservation = _loads(data)
        except Exception as e:
            logger.exception(f"SendObservations deserialize error: {e}  data[:64]={data[:64]!r}")
            return services_pb2.Empty()

        try:
            timestep = timed_obs.get_timestep()
            joint_state, image_arrays = _parse_raw_obs(timed_obs.get_observation())

            if self._episode_steps == 0:
                cam_info = {k: v.shape for k, v in image_arrays.items()}
                logger.info(f"[obs check] joints={joint_state.shape} cameras={cam_info}")

            if self.show_cameras:
                _update_camera_display(image_arrays)

            if self._episode_steps % 10 == 0:
                from mpail2.envs.real.so101.robot_limits import JOINT_NAMES
                joint_str = "  ".join(f"{n}={v:.2f}" for n, v in zip(JOINT_NAMES, joint_state))
                logger.info(f"[joints] {joint_str}")

            self._prev_timestep = timestep

            self.recorder.record(joint_state, image_arrays)

            # Auto-advance to the next episode once max_episode_steps is reached
            # rather than waiting on a client-driven Ready() call.
            if not self._recording_paused:
                self._episode_steps += 1
                logger.info(f"[ep {self._episode_count + 1} step {self._episode_steps}/{self.max_episode_steps}]")
                if self._episode_steps >= self.max_episode_steps:
                    self._recording_paused = True
                    threading.Thread(target=self._auto_episode_end, daemon=True).start()

            # Push to obs queue so GetActions can respond
            if self._obs_queue.full():
                try: self._obs_queue.get_nowait()
                except Empty: pass
            self._obs_queue.put((timestep, timed_obs.get_timestamp(), joint_state))

        except Exception as e:
            logger.exception(f"SendObservations processing error: {e}")

        return services_pb2.Empty()

    def GetActions(self, request, context):
        try:
            timestep, timestamp, joint_state = self._obs_queue.get(timeout=1.0)
        except Empty:
            return services_pb2.Actions(data=b"")

        if self._going_home:
            # Episode just ended (max_episode_steps reached) — drive to home instead
            # of holding in place, for the duration of _auto_episode_end's pause.
            action_deg = HOME_POSITION_DEG.copy()
        else:
            # Echo current joints back — robot actively holds wherever it already is.
            action_deg = joint_state.copy()

        timed_actions = [TimedAction(
            timestamp=timestamp, timestep=timestep,
            action=torch.tensor(action_deg, dtype=torch.float32),
        )]
        return services_pb2.Actions(data=pickle.dumps(timed_actions))  # nosec

    def _auto_episode_end(self):
        """Auto-triggered once _episode_steps reaches max_episode_steps — saves the
        episode, drives the arm home (via GetActions checking _going_home), pauses
        episode_pause_seconds, then resumes recording. Runs in a background thread so
        the request/response loop keeps moving while the arm drives home and the
        recorder flushes to disk.
        """
        ep = self._episode_count + 1
        self._episode_count = ep
        logger.info(
            f"[ep {ep}] {self.max_episode_steps} steps reached — saving and returning home "
            f"(pausing {self.episode_pause_seconds:.1f}s before episode {ep + 1})..."
        )
        self.recorder.flush()  # slow disk write — runs in background
        self._going_home = True
        time.sleep(self.episode_pause_seconds)
        self._going_home = False
        self._episode_steps = 0
        self.recorder.reset_pending()
        self._recording_paused = False
        logger.info(f"[ep {ep + 1}] recording started")

    def _episode_end(self):
        """Triggered when the client calls Ready() with an episode already in
        progress (i.e. ended early, before max_episode_steps) — saves and waits for
        an Enter keypress before resuming, instead of auto-resuming immediately.
        """
        self._episode_count += 1
        # Pause recording immediately so observations sent while saving are discarded.
        self._recording_paused = True
        ep = self._episode_count

        def _save_and_wait():
            self.recorder.flush()  # slow disk write — runs in background
            print(
                f"\n── Episode {ep} saved ──────────────────────────\n"
                f"   Press Enter to start recording episode {ep + 1} "
                f"(Ctrl-C to stop)...",
                flush=True,
            )
            self._wait_for_enter()

        threading.Thread(target=_save_and_wait, daemon=True).start()

    def _wait_for_enter(self):
        """Block in a background thread until the user presses Enter, then resume recording."""
        try:
            input()
        except EOFError:
            pass
        self.recorder.reset_pending()   # clean slate for the new episode
        self._recording_paused = False
        print(f"   Recording episode {self._episode_count + 1} ...", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def serve():
    parser = argparse.ArgumentParser(description="Demo-recording gRPC server for SO-101")
    parser.add_argument("--collect_dir",  required=True,
                        help="Directory to save recorded (obs, next_obs) .npz files.")
    parser.add_argument("--flush_every",  type=int, default=200,
                        help="Flush buffer to disk every N steps (default: 200).")
    parser.add_argument("--port",         type=int, default=8080)
    parser.add_argument("--max_episode_steps", type=int, default=200,
                        help="Hard cap on transitions recorded per episode — auto-ends the "
                             "episode (save, return home, pause, resume) once reached.")
    parser.add_argument("--episode_pause_seconds", type=float, default=5.0,
                        help="Pause this long (arm held at home) after each episode's "
                             "max_episode_steps is reached, before auto-resuming recording "
                             "for the next episode.")
    parser.add_argument("--show_cameras", action="store_true",
                        help="Save live camera frames to cameras_latest.png during recording "
                             "(open in an editor that auto-refreshes to watch).")
    args = parser.parse_args()

    recorder = DemoRecorder(Path(args.collect_dir), flush_every=args.flush_every)
    logger.info(f"Demo recording enabled → {args.collect_dir}  (flush every {args.flush_every} steps)")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(
        DemoRecordingServicer(
            recorder=recorder,
            max_episode_steps=args.max_episode_steps,
            show_cameras=args.show_cameras,
            episode_pause_seconds=args.episode_pause_seconds,
        ),
        server,
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    logger.info(f"gRPC server listening on port {args.port}  mode=demo-recording")
    logger.info("Waiting for robot_client.py ...")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        recorder.flush()
        logger.info("Final demo flush complete.")
        server.stop(grace=2.0)


if __name__ == "__main__":
    serve()
