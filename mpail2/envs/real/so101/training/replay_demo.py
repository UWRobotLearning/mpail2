"""
Replay a recorded .npz demo on the physical SO-101 follower arm,
optionally showing the recorded camera images in a live window.

Usage:
    # Direct joint replay (original):
    python replay_demo.py pick_changed/traj_0001.npz --port /dev/ttyACM0 --hz 10

    # IK replay — runs demo joints through FK→IK to verify the Cartesian pipeline:
    python replay_demo.py pick_changed/traj_0001.npz --port /dev/ttyACM0 --hz 10 --use_ik

    # Visual check only (no robot):
    python replay_demo.py pick_changed/traj_0001.npz --no_robot --show_cameras

--state_cols selects which columns of observation.state to use as joint targets.
  Old 12-dim recordings: use 0 1 2 3 4 5 (follower, first 6).
  New 6-dim recordings:  use 0 1 2 3 4 5 (default, all 6).
"""

import argparse
import time

import numpy as np


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert stored image (float32, any range) to uint8 RGB."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to .npz demo file")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Follower arm serial port")
    parser.add_argument("--robot_id", default="Kid")
    parser.add_argument("--hz", type=float, default=10.0, help="Replay speed in Hz")
    parser.add_argument(
        "--state_cols", type=int, nargs="+", default=list(range(6)),
        help="Which columns of observation.state to use (default: 0-5)"
    )
    parser.add_argument(
        "--show_cameras", action="store_true",
        help="Display recorded camera images in a window during replay"
    )
    parser.add_argument(
        "--no_robot", action="store_true",
        help="Skip robot connection — only show camera images (useful for quick visual check)"
    )
    parser.add_argument(
        "--use_ik", action="store_true",
        help="Replay via FK→IK: convert demo joint positions to EE xyz then back to joints. "
             "Verifies the Cartesian IK pipeline before live RL."
    )
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    state = data["observation.state"]  # (N, 2, D)
    joint_positions = state[:, 0, args.state_cols]  # (N, 6) — obs side
    n_steps = len(joint_positions)

    # Collect camera arrays present in the file
    cam_keys = [k for k in data.keys() if k not in ("observation.state", "actions")]
    cameras = {k: data[k][:, 0, :, :, :] for k in cam_keys}  # (N, H, W, C)

    # Pre-compute IK targets if requested
    ik_joint_positions = None
    if args.use_ik:
        import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from mpail2.envs.real.so101.ik_utils import fk, ik
        print("Pre-computing IK targets from demo joint positions...")
        ik_joints = []
        prev_arm = joint_positions[0, :5].copy()
        for step_joints in joint_positions:
            arm = step_joints[:5]
            ee_xyz = fk(arm)
            ik_arm = ik(ee_xyz, initial_arm_deg=prev_arm)
            ik_joints.append(np.append(ik_arm, step_joints[5]))  # keep original gripper
            prev_arm = ik_arm
        ik_joint_positions = np.array(ik_joints, dtype=np.float32)

        # Show round-trip error
        ee_errors = []
        for orig, ik_j in zip(joint_positions, ik_joint_positions):
            ee_orig = fk(orig[:5])
            ee_ik   = fk(ik_j[:5])
            ee_errors.append(np.linalg.norm(ee_ik - ee_orig) * 1000)
        print(f"  IK EE round-trip error: mean={np.mean(ee_errors):.1f}mm  max={np.max(ee_errors):.1f}mm")

    print(f"Loaded {args.npz}")
    print(f"  Steps: {n_steps}, mode={'IK' if args.use_ik else 'direct joints'}")
    print(f"  Cameras: {cam_keys if cam_keys else 'none'}")
    print(f"  Joint ranges:")
    for i in range(len(args.state_cols)):
        name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"joint_{i}"
        print(f"    {name}: {joint_positions[:,i].min():.2f} to {joint_positions[:,i].max():.2f} deg")

    show = args.show_cameras and bool(cam_keys)
    if args.show_cameras and not cam_keys:
        print("  (no camera data in file — --show_cameras ignored)")

    fig = ax = im_handle = None
    if show:
        import matplotlib
        matplotlib.use("TkAgg")  # use TkAgg; falls back gracefully if unavailable
        import matplotlib.pyplot as plt
        n_cams = len(cam_keys)
        fig, axes = plt.subplots(1, n_cams, figsize=(6 * n_cams, 5))
        if n_cams == 1:
            axes = [axes]
        first_frames = [_to_uint8(cameras[k][0]) for k in cam_keys]
        im_handles = [ax.imshow(f) for ax, f in zip(axes, first_frames)]
        titles = [ax.set_title(k) for ax, k in zip(axes, cam_keys)]
        step_text = fig.suptitle("step 0")
        plt.tight_layout()
        plt.pause(0.001)

    input(f"\nPress Enter to start replay at {args.hz} Hz ...")

    robot = None
    if not args.no_robot:
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots import make_robot_from_config
        cfg = SOFollowerRobotConfig(port=args.port, id=args.robot_id, use_degrees=True)
        robot = make_robot_from_config(cfg)
        robot.connect()
        print("Robot connected. Starting replay...")
    else:
        print("--no_robot: skipping robot connection, showing images only.")

    replay_positions = ik_joint_positions if args.use_ik else joint_positions

    dt = 1.0 / args.hz
    try:
        for step, joints in enumerate(replay_positions):
            t0 = time.perf_counter()

            if robot is not None:
                action = {f"{name}.pos": float(joints[i]) for i, name in enumerate(JOINT_NAMES)}
                robot.send_action(action)

            if show:
                for im_h, k in zip(im_handles, cam_keys):
                    im_h.set_data(_to_uint8(cameras[k][step]))
                step_text.set_text(f"step {step}/{n_steps}  {'[IK]' if args.use_ik else ''}")
                plt.pause(0.001)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))
            if step % 20 == 0:
                print(f"  step {step}/{n_steps}  joints={joints.round(1)}")

    except KeyboardInterrupt:
        print("\nReplay interrupted.")
    finally:
        if show:
            import matplotlib.pyplot as plt
            plt.close("all")
        if robot is not None:
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
