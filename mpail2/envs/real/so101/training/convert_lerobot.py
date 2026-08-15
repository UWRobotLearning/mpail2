"""
convert_lerobot.py — Convert a LeRobot recorded dataset to MPAIL2 .pt format.

Run in the LEROBOT conda env (lerobot must be installed):
    conda activate lerobot
    python convert_lerobot.py \
        --repo_id local/my_demos \
        --root ./raw_lerobot_demos \
        --out raw_demos2_master.pt \
        --cam_name cam \
        --img_w 64 --img_h 48

What it does
────────────
  For every consecutive frame pair (t, t+1) within each episode it builds:
    observation.state : (N, 2, STATE_DIM)   joint positions
    cam               : (N, 2, 48, 64, 3)   camera frames, resized & in [0, 1]

  N = total transitions across all episodes (last frame of each episode is dropped
  because it has no t+1 counterpart).

The output file is ready for:
    python train_so101_local.py --demo_path raw_demos2_master.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def convert(repo_id: str, root: str | None, out: str, cam_name: str, img_w: int, img_h: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset  repo_id={repo_id}  root={root or 'hub'} ...")
    ds = LeRobotDataset(repo_id=repo_id, root=root)
    print(f"  {ds.num_episodes} episodes, {ds.num_frames} frames total")
    print(f"  features: {list(ds.features.keys())}")

    lerobot_cam_key = f"observation.images.{cam_name}"
    has_camera = lerobot_cam_key in ds.features
    if not has_camera:
        print(f"[WARN] Camera key '{lerobot_cam_key}' not found. Available image keys:")
        for k in ds.features:
            if "image" in k:
                print(f"  {k}")
        print("  Continuing without camera data.")

    state_pairs = []
    cam_pairs   = []

    # Iterate episode by episode so we never pair frames across boundaries
    hf = ds.hf_dataset
    episode_indices = sorted(hf.unique("episode_index"))
    print(f"Converting {len(episode_indices)} episodes ...")

    for ep_idx in episode_indices:
        ep_rows = hf.filter(lambda row: row["episode_index"] == ep_idx)
        n_frames = len(ep_rows)
        if n_frames < 2:
            print(f"  episode {ep_idx}: only {n_frames} frame(s), skipping")
            continue

        # Pull state — shape (T, STATE_DIM)
        states = np.array(ep_rows["observation.state"], dtype=np.float32)  # (T, D)

        # Build consecutive pairs (T-1, 2, D)
        state_t  = states[:-1]   # (T-1, D)
        state_t1 = states[1:]    # (T-1, D)
        state_pairs.append(np.stack([state_t, state_t1], axis=1))   # (T-1, 2, D)

        if has_camera:
            # Frames are decoded tensors; lerobot returns (C, H, W) uint8 or float
            cam_frames = []
            for i in range(n_frames):
                frame = ds[ep_rows[i]["frame_index"]]
                img = frame[lerobot_cam_key]   # tensor (C, H, W)
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                cam_frames.append(img)         # (C, H, W) float [0,1]

            # Stack → (T, C, H, W), resize (cv2.INTER_AREA, matching live camera
            # downsampling in so101_robot_server.py), convert to (T, H, W, C)
            cam_t  = torch.stack(cam_frames[:-1]).permute(0, 2, 3, 1).numpy()   # (T-1, H, W, C)
            cam_t1 = torch.stack(cam_frames[1:]).permute(0, 2, 3, 1).numpy()    # (T-1, H, W, C)

            cam_t  = np.stack([
                cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA) for img in cam_t
            ])
            cam_t1 = np.stack([
                cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA) for img in cam_t1
            ])

            cam_pairs.append(np.stack([cam_t, cam_t1], axis=1))   # (T-1, 2, H, W, C)

        print(f"  episode {ep_idx}: {n_frames} frames → {n_frames - 1} transitions")

    # Concatenate across episodes
    all_states = np.concatenate(state_pairs, axis=0)   # (N, 2, D)
    demos = {"observation.state": torch.from_numpy(all_states)}

    if cam_pairs:
        all_cams = np.concatenate(cam_pairs, axis=0)   # (N, 2, H, W, C)
        demos["cam"] = torch.from_numpy(all_cams)

    # Save
    torch.save(demos, out)
    print(f"\nSaved → {out}")
    for k, v in demos.items():
        mb = v.numel() * 4 / 1e6
        print(f"  {k}: {tuple(v.shape)}  ({mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset → MPAIL2 .pt")
    parser.add_argument("--repo_id", required=True,
                        help="LeRobot repo ID, e.g. 'local/my_demos'")
    parser.add_argument("--root",    default=None,
                        help="Local root directory where the dataset is saved "
                             "(e.g. ./raw_lerobot_demos). Omit to load from HF Hub.")
    parser.add_argument("--out",     default="raw_demos2_master.pt",
                        help="Output .pt file path (default: raw_demos2_master.pt)")
    parser.add_argument("--cam_name", default="cam",
                        help="Camera name used during recording (default: cam). "
                             "This is the name after 'observation.images.' in the dataset.")
    parser.add_argument("--img_w", type=int, default=64,
                        help="Resize camera frames to this width (default: 64 — matches 640x480 aspect ratio)")
    parser.add_argument("--img_h", type=int, default=48,
                        help="Resize camera frames to this height (default: 48 — matches 640x480 aspect ratio)")
    args = parser.parse_args()

    convert(
        repo_id=args.repo_id,
        root=args.root,
        out=args.out,
        cam_name=args.cam_name,
        img_w=args.img_w,
        img_h=args.img_h,
    )
