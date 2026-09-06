"""Convert raw_demos*/*.npz trajectory files to a single MPAIL-format .pt file.

Usage:
    python convert.py                          # merges raw_demos2 only (default)
    python convert.py --dirs raw_demos2        # same
    python convert.py --dirs raw_demos raw_demos2 raw_demos3   # all demos
    python convert.py --out my_demos.pt        # custom output path
    python convert.py --img_w 64 --img_h 48    # resize cameras to 64x48 (default, matches 640x480 aspect ratio)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

_IMG_KEY_PREFIXES = ("observation.images.",)


def _normalize_key(key: str) -> str:
    """Strip lerobot image key prefix: 'observation.images.cam' → 'cam'."""
    for prefix in _IMG_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _resize_cam(tensor: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    """Resize camera tensor (N, 2, H, W, C) → (N, 2, out_h, out_w, C).

    Uses cv2.INTER_AREA to match the downsampling used for live camera frames
    in so101_robot_server.py — keeping demo and online observations consistent.
    out_w/out_h should match the source aspect ratio (e.g. 64x48 for 640x480
    native capture) to avoid the anisotropic stretch a square resize would cause.
    """
    N, two, H, W, C = tensor.shape
    if H == out_h and W == out_w:
        return tensor
    arr = tensor.reshape(N * two, H, W, C).numpy()
    if arr.max() > 1.0:
        arr = arr / 255.0
    resized = np.stack([
        cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        for frame in arr
    ])
    return torch.from_numpy(resized).reshape(N, two, out_h, out_w, C)


def convert(source_dirs: list[str], out_path: str, img_w: int = 64, img_h: int = 48, state_dim: int | None = None) -> None:
    all_data: dict = {}
    total_files = 0

    for dir_name in source_dirs:
        src = Path(dir_name)
        if not src.exists():
            print(f"[WARN] {src} does not exist — skipping")
            continue
        npz_files = sorted(src.glob("*.npz"))
        if not npz_files:
            print(f"[WARN] {src} contains no .npz files — skipping")
            continue
        for npz_file in npz_files:
            d = np.load(str(npz_file), allow_pickle=True)
            for raw_key in d.files:
                key = _normalize_key(raw_key)
                arr = torch.from_numpy(d[raw_key].astype(np.float32))
                # Resize cameras immediately to avoid holding 480×640 frames in RAM
                if arr.dim() == 5:
                    arr = _resize_cam(arr, img_w, img_h)
                all_data.setdefault(key, []).append(arr)
            total_files += 1
            print(f"  loaded {npz_file}  keys={list(d.files)}")

    if not all_data:
        raise RuntimeError("No data found in any of the source directories.")

    merged = {k: torch.cat(vs, dim=0) for k, vs in all_data.items()}

    # Keep only MPAIL-format keys: shape (N, 2, *obs_shape) — drop "actions" etc.
    demos = {k: v for k, v in merged.items() if v.dim() >= 3 and v.shape[1] == 2}

    if not demos:
        raise RuntimeError(
            "No MPAIL-format tensors found (expected shape [N, 2, *obs_shape]). "
            "Keys found: " + str(list(merged.keys()))
        )

    # Trim state vector if it has extra dims (e.g. follower+leader concatenated → follower only)
    state_key = "observation.state"
    if state_dim is not None and state_key in demos and demos[state_key].shape[-1] != state_dim:
        before = demos[state_key].shape[-1]
        demos[state_key] = demos[state_key][..., :state_dim]
        print(f"  trimmed {state_key}: dim {before} → {state_dim}")

    torch.save(demos, out_path)
    print(f"\nSaved {out_path}")
    print(f"  Source files : {total_files}")
    for k, v in demos.items():
        print(f"  {k}: {tuple(v.shape)}  ({v.numel() * 4 / 1e6:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs", nargs="+", default=["raw_demos2"],
        help="Source directories containing .npz trajectory files (default: raw_demos2)",
    )
    parser.add_argument(
        "--out", default="raw_demos2_master.pt",
        help="Output .pt file path (default: raw_demos2_master.pt)",
    )
    parser.add_argument(
        "--img_w", type=int, default=64,
        help="Resize camera images to this width (default: 64 — matches 640x480 native aspect ratio).",
    )
    parser.add_argument(
        "--img_h", type=int, default=48,
        help="Resize camera images to this height (default: 48 — matches 640x480 native aspect ratio).",
    )
    parser.add_argument(
        "--state_dim", type=int, default=None,
        help="Trim observation.state to this many dims (e.g. 6 if data has follower+leader=12)",
    )
    args = parser.parse_args()
    convert(args.dirs, args.out, img_w=args.img_w, img_h=args.img_h, state_dim=args.state_dim)
