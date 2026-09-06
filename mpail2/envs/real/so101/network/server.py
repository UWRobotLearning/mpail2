"""
so101_robot_server.py — gRPC control server exposing the real SO-101 arm + cameras
to mpail2/envs/real/so101/so101_env.py.

Runs directly against lerobot's SOFollower driver — no LeRobot async_inference
layer involved, no separate client process required on this side. Mirrors
mpail2/envs/real/franka/network/server.py's role (thin RPC wrapper around the
hardware driver), but gRPC instead of ZMQ, and combines action+observation into
one round-trip per step instead of separate reads.

Run in the `lerobot` conda env (needs `mpail2` installed there too, e.g. `pip install -e ".[so101]"`):

    conda activate lerobot
    python -m mpail2.envs.real.so101.network.server \
        --robot_port /dev/ttyACM0 \
        --robot_id Kid \
        --cam_index /dev/video0 \
        --cam2_serial 317422074482 \
        --grpc_port 7070

Then point the mpail2-side env at it (mock=False):

    from mpail2.envs.real.so101 import SO101RealEnvArgs, make_so101_env
    env = make_so101_env(SO101RealEnvArgs(host="<this machine's IP>", port=7070, mock=False))
"""

import argparse
import logging
import time
from concurrent import futures

import grpc
import numpy as np

from mpail2.envs.real.so101.transport import so101_robot_pb2, so101_robot_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("so101_robot_server")

# Must match mpail2/envs/real/so101/robot_limits.py — JOINT_NAMES, HOME_POSITION_DEG,
# CAM_KEY/CAM2_KEY, CAM_H/CAM_W/CAM2_H/CAM2_W. Kept as local constants here since this
# script runs in the separate `lerobot` conda env, which doesn't have mpail2 installed.
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME_POSITION_DEG = [-21.19, -5.41, 1.58, 99.47, -14.20, 28.66]
CAM_KEY, CAM2_KEY = "cam", "cam2"
# 64x48 preserves the native 640x480 (4:3) aspect ratio exactly (10x isotropic downsample,
# no stretch) — unlike 84x84, which squashes x/y by different factors (84/640 vs 84/480).
CAM_W, CAM_H = 64, 48
CAM2_W, CAM2_H = 64, 48


def _joints_dict_to_array(obs: dict) -> np.ndarray:
    return np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)


def _to_bytes(img: np.ndarray, h: int, w: int) -> bytes:
    """uint8 (H0, W0, 3) camera frame -> raw bytes at (h, w, 3), resizing if needed."""
    img = np.asarray(img)
    if img.shape[0] != h or img.shape[1] != w:
        import cv2
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img, dtype=np.uint8).tobytes()


class SO101RobotServicer(so101_robot_pb2_grpc.SO101RobotServicer):
    def __init__(
        self, robot, settle_tolerance_deg: float = 0.5, settle_timeout_s: float = 0.5,
        gripper_settle_tolerance_deg: float = 3.0,
        reset_duration_s: float = 4.0, reset_hz: float = 20.0,
    ):
        self.robot = robot
        self.settle_tolerance_deg = settle_tolerance_deg
        self.settle_timeout_s = settle_timeout_s
        self.reset_duration_s = reset_duration_s
        self.reset_hz = reset_hz
        # Gripper is a much slower/weaker actuator than the 5 arm joints (and often fights real
        # resistance — squeezing an object, brushing the table) — holding it to the same 1° arm
        # tolerance means it never settles, so every Step() burns the full timeout every time,
        # silently undoing whatever control rate was requested. It still gets commanded and its
        # readback is still returned; it's just not what the wait blocks on.
        self.per_joint_tolerance_deg = {name: settle_tolerance_deg for name in JOINT_NAMES}
        self.per_joint_tolerance_deg["gripper"] = gripper_settle_tolerance_deg

    def _state_from_obs(self, obs: dict) -> "so101_robot_pb2.RobotState":
        joints = _joints_dict_to_array(obs)
        cam = _to_bytes(obs[CAM_KEY], CAM_H, CAM_W) if CAM_KEY in obs else b""
        cam2 = _to_bytes(obs[CAM2_KEY], CAM2_H, CAM2_W) if CAM2_KEY in obs else b""
        return so101_robot_pb2.RobotState(joints_deg=joints.tolist(), cam=cam, cam2=cam2)

    def _read_present_position(self, retries: int = 5, backoff_s: float = 0.005) -> dict:
        """bus.sync_read('Present_Position') with retry-on-transient-error.

        The half-duplex serial bus occasionally returns a corrupted status packet under
        load (raises ConnectionError) — with the settle-wait polling this every 10ms,
        that's a "when", not "if". Absorb a few transient glitches here rather than
        letting one crash the whole Step()/Reset() RPC; if it's a real disconnect
        (not transient), it'll still raise after exhausting retries.
        """
        for attempt in range(retries):
            try:
                return self.robot.bus.sync_read("Present_Position")
            except ConnectionError:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff_s)

    def _get_observation_sync(self) -> dict:
        """Like robot.get_observation(), but calls cam.read() (blocking — waits for the
        NEXT frame captured after this call starts) instead of cam.read_latest()
        (non-blocking peek at whatever's already sitting in the background capture
        thread's buffer, which can be up to ~1/fps stale). Guarantees the returned
        frames were captured after the joint read below, at the cost of blocking for
        up to ~1/fps per camera instead of returning instantly. Always used — see
        so101_env.py/grpc_policy_server.py, which both expect the policy to act on a
        freshly-captured observation rather than a possibly-stale buffered one.
        """
        obs = {f"{name}.pos": val for name, val in self._read_present_position().items()}
        for cam_key, cam in self.robot.cameras.items():
            obs[cam_key] = cam.read()
        return obs

    def _get_observation_safe(self, retries: int = 3, backoff_s: float = 0.005) -> dict:
        """_get_observation_sync() with retry-on-transient-error."""
        for attempt in range(retries):
            try:
                return self._get_observation_sync()
            except ConnectionError:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff_s)

    def _wait_until_settled(self, goal_pos: dict) -> tuple[bool, dict]:
        """Block until every joint's readback is within tolerance of its goal, or timeout.

        send_action()/sync_write() is fire-and-forget — it returns as soon as the goal
        register is written, not once the servo arrives. At high step rates the very
        next Step() would otherwise overwrite the goal before heavier joints (shoulder/
        elbow/wrist, real inertia) finish traveling, so get_observation() reads a stale,
        frozen-looking position. Poll Present_Position directly (cheaper than a full
        get_observation(), which also grabs both camera frames) until settled.

        settle_timeout_s <= 0 means wait indefinitely — no deadline is set at all. Only
        safe if per_joint_tolerance_deg is loose enough for every joint to actually reach;
        a joint with a persistent steady-state offset (e.g. gravity droop) that never
        closes will otherwise block this call forever.

        Returns (settled, last_errors) — last_errors is {joint: |present-goal|} from the
        final poll, so callers can log how far off a timeout actually was (a fraction of
        a degree off vs. stuck 20 degrees away are very different problems).
        """
        deadline = time.time() + self.settle_timeout_s if self.settle_timeout_s > 0 else None
        while True:
            present = self._read_present_position()
            errors = {name: abs(present[name] - goal_pos[name]) for name in goal_pos}
            if all(err <= self.per_joint_tolerance_deg[name] for name, err in errors.items()):
                return True, errors
            if deadline is not None and time.time() >= deadline:
                return False, errors
            time.sleep(0.01)

    def Reset(self, request, context):
        # Ease into home instead of one big instant jump: send a ramp of intermediate
        # targets (linear interpolation, present -> home) at reset_hz, over reset_duration_s.
        # A single send_action(HOME_POSITION_DEG) commands the servos at full speed/torque
        # from wherever they currently are, which can be a large, abrupt, jerky move if the
        # previous episode ended far from home.
        present = self._read_present_position()
        start = {name: present[name] for name in JOINT_NAMES}
        goal = dict(zip(JOINT_NAMES, HOME_POSITION_DEG))

        n_steps = max(1, int(self.reset_duration_s * self.reset_hz))
        dt = 1.0 / self.reset_hz
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            interp = {name: start[name] + alpha * (goal[name] - start[name]) for name in JOINT_NAMES}
            self.robot.send_action({f"{name}.pos": val for name, val in interp.items()})
            time.sleep(dt)

        self._wait_until_settled(goal)
        obs = self._get_observation_safe()
        logger.info(f"Reset -> home position (ramped over {self.reset_duration_s:.1f}s)")
        return self._state_from_obs(obs)

    def Step(self, request, context):
        joints = list(request.joints_deg)
        if len(joints) != len(JOINT_NAMES):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Expected {len(JOINT_NAMES)} joint targets, got {len(joints)}",
            )
        action = dict(zip([f"{n}.pos" for n in JOINT_NAMES], joints))
        sent = self.robot.send_action(action)
        goal_pos = {k.removesuffix(".pos"): v for k, v in sent.items()}
        settled, errors = self._wait_until_settled(goal_pos)
        if not settled:
            over_tol = {
                name: err for name, err in errors.items()
                if err > self.per_joint_tolerance_deg[name]
            }
            worst = sorted(over_tol.items(), key=lambda kv: -kv[1])
            worst_str = ", ".join(f"{name}={err:.2f}°" for name, err in worst)
            logger.warning(
                f"Step: arm did not settle within {self.settle_timeout_s:.2f}s — "
                f"reading present position anyway. Over tolerance: {worst_str}"
            )
        obs = self._get_observation_safe()
        return self._state_from_obs(obs)


def serve():
    parser = argparse.ArgumentParser(description="gRPC control server for the real SO-101 arm")
    parser.add_argument("--robot_port", default="/dev/ttyACM0", help="Serial port the arm is connected to")
    parser.add_argument("--robot_id", default="Kid", help="Calibration id (matches the LeRobot calibration file)")
    parser.add_argument("--cam_index", default="/dev/video0", help="Wrist cam (opencv) index or device path")
    parser.add_argument("--cam2_serial", default=None, help="RealSense serial number for cam2 (omit to disable)")
    parser.add_argument("--cam_fps", type=int, default=30,
                         help="Capture fps for both cameras (default: 30).")
    parser.add_argument("--grpc_host", default="[::]")
    parser.add_argument("--grpc_port", type=int, default=7070)
    parser.add_argument(
        "--max_relative_target", type=float, default=None,
        help="Per-step delta clamp passed straight to lerobot's SOFollowerRobotConfig (default: no clamp).",
    )
    parser.add_argument(
        "--settle_tolerance_deg", type=float, default=2.0,
        help="Step()/Reset() block until every arm joint (all but gripper) is within this many "
             "degrees of its goal (or --settle_timeout_s elapses) before reading the observation back.",
    )
    parser.add_argument(
        "--gripper_settle_tolerance_deg", type=float, default=15.0,
        help="Separate, looser settle tolerance for the gripper — it's a slower/weaker actuator "
             "than the arm joints (and often fights real resistance), so holding it to the arm's "
             "tolerance means it never settles and every step burns the full timeout for nothing.",
    )
    parser.add_argument(
        "--settle_timeout_s", type=float, default=0.5,
        help="Max time to wait for the arm to settle before giving up and reading present position "
             "anyway. <= 0 disables the timeout entirely (wait indefinitely) — only safe if "
             "settle/gripper tolerances are loose enough for every joint to actually be reachable, "
             "or a joint with a persistent steady-state offset will hang Step() forever.",
    )
    parser.add_argument(
        "--reset_duration_s", type=float, default=4.0,
        help="Reset() ramps from the current position to home over this many seconds "
             "(linear interpolation), instead of commanding home in one instant jump.",
    )
    parser.add_argument(
        "--reset_hz", type=float, default=20.0,
        help="Update rate of the Reset() ramp's intermediate targets.",
    )
    parser.add_argument(
        "--p_coefficient", type=int, default=16,
        help="Position-loop proportional gain applied to every joint after connect, overriding "
             "LeRobot's own default of 16 ('to avoid shakiness', tuned for smooth human-paced "
             "teleop). RL-driven targets change faster/less smoothly than a teleop operator's hand, "
             "so the softer gain isn't enough torque to close small residual errors in time — this "
             "was the confirmed root cause of the persistent settle-timeout warnings. 32 restores "
             "Feetech's own factory default. Set higher for tighter tracking at the risk of "
             "oscillation/shakiness; set to 16 (or omit this override) to match stock teleop behavior.",
    )
    parser.add_argument(
        "--gripper_torque_limit", type=int, default=500,
        help="Max_Torque_Limit for the gripper only (0-1000 = 0-100%%), overriding LeRobot's own "
             "default of 500 (50%%, 'to avoid burnout'). 800 = 80%%, a meaningful increase while "
             "leaving headroom below full torque given the gripper is the joint most prone to "
             "sustained resistance (squeezing objects, hitting the table). Raise further only with "
             "awareness of overheating risk under prolonged stall.",
    )
    parser.add_argument(
        "--i_coefficient", type=int, default=4,
        help="Position-loop integral gain applied to every joint after connect, overriding "
             "LeRobot's own default of 0 (pure P+D, no integral term at all). A P+D-only loop "
             "cannot drive steady-state error to zero under any constant load (gravity on "
             "shoulder_lift, static friction, gripper squeezing an object) — it settles into a "
             "stable equilibrium where the P term's output exactly balances the load and simply "
             "stops there, which is why raising --p_coefficient alone and/or waiting longer "
             "(--settle_timeout_s) doesn't help: the system already reached its (wrong) equilibrium "
             "and isn't converging further at all. A small nonzero I term lets the controller "
             "accumulate that persistent error over time and keep increasing output until it's "
             "actually eliminated. Start small and increase cautiously — too high an I gain causes "
             "integral windup / overshoot / oscillation. 0 restores stock teleop behavior.",
    )
    parser.add_argument(
        "--goal_velocity", type=int, default=0,
        help="Goal_Velocity register (address 46) written to every motor after connect — a "
             "hardware-level speed cap: the servo's own control loop will not move faster than "
             "this toward whatever Goal_Position it's given, no matter how large the commanded "
             "jump is. This is the same kind of protection Kinova/Franka get for free from their "
             "native Cartesian controllers; these Feetech servos don't have it unless set "
             "explicitly. 0 = unlimited (stock/factory behavior, no cap). Units are raw register "
             "ticks, not degrees/sec directly — tune empirically by watching actual motion "
             "smoothness, there's no documented tick-to-deg/s conversion in this codebase yet.",
    )
    args = parser.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

    cameras = {CAM_KEY: OpenCVCameraConfig(index_or_path=args.cam_index, width=640, height=480, fps=args.cam_fps)}
    if args.cam2_serial:
        from lerobot.cameras.realsense import RealSenseCameraConfig
        cameras[CAM2_KEY] = RealSenseCameraConfig(
            serial_number_or_name=args.cam2_serial, width=640, height=480, fps=args.cam_fps
        )

    config = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
    )
    robot = SO101Follower(config)
    logger.info(f"Connecting to SO-101 on {args.robot_port} (id={args.robot_id})...")
    robot.connect()
    logger.info("Connected.")

    # robot.connect() -> configure() just wrote LeRobot's own teleop-tuned defaults
    # (P_Coefficient=16, I_Coefficient=0, gripper Max_Torque_Limit=500) — override for this
    # RL-driven training/eval server, where per-step targets change faster/less smoothly than
    # a teleop operator's hand. I_Coefficient=0 (pure P+D) is the confirmed root cause of the
    # persistent settle-timeout warnings: with no integral term the loop settles into a stable
    # equilibrium under any constant load (gravity, friction, gripper resistance) instead of
    # ever fully closing the error — no amount of extra P gain or settle_timeout_s fixes that,
    # since the system isn't still converging, it's already stopped.
    for motor in robot.bus.motors:
        robot.bus.write("P_Coefficient", motor, args.p_coefficient)
        robot.bus.write("I_Coefficient", motor, args.i_coefficient)
        robot.bus.write("Goal_Velocity", motor, args.goal_velocity)
    robot.bus.write("Max_Torque_Limit", "gripper", args.gripper_torque_limit)
    logger.info(
        f"Overrode servo tuning for training: P_Coefficient={args.p_coefficient}, "
        f"I_Coefficient={args.i_coefficient}, Goal_Velocity={args.goal_velocity} (all joints), "
        f"gripper Max_Torque_Limit={args.gripper_torque_limit}/1000"
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = SO101RobotServicer(
        robot, settle_tolerance_deg=args.settle_tolerance_deg, settle_timeout_s=args.settle_timeout_s,
        gripper_settle_tolerance_deg=args.gripper_settle_tolerance_deg,
        reset_duration_s=args.reset_duration_s, reset_hz=args.reset_hz,
    )
    so101_robot_pb2_grpc.add_SO101RobotServicer_to_server(servicer, server)
    server.add_insecure_port(f"{args.grpc_host}:{args.grpc_port}")
    server.start()
    logger.info(f"so101_robot_server listening on {args.grpc_host}:{args.grpc_port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        robot.disconnect()
        server.stop(grace=2.0)


if __name__ == "__main__":
    serve()
