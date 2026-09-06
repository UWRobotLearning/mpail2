"""check_encoder_collapse.py — Diagnose whether the encoder's latent space has
collapsed (encoding most/all observations to a low-variance region), which a
low dynamics (JEP) loss alone cannot rule out: an encoder that maps everything
to a near-constant vector also gets near-zero prediction loss, without having
learned anything useful.

Loads a checkpoint + demo.pt, runs the encoder over a sample of real (obs,
next_obs) pairs (through the same CamOnlyObsNormalizer pipeline used during
training), and reports:
  - per-dimension std (how many latent dims are effectively "dead")
  - effective rank of the latent covariance (participation ratio + cumulative
    explained-variance curve): a healthy, non-collapsed representation should
    spread variance across many dimensions, not concentrate it in a handful.

Usage:
    python check_encoder_collapse.py --demo_path demo.pt \\
        --checkpoint logs/so101_local/models/model_65.pt
"""

import argparse
import glob
import os

import numpy as np
import torch

from mpail2.envs.real.so101 import ik_utils
from mpail2.envs.real.so101 import OBS_KEY, EE_PROPRIO_DIM
from mpail2.envs.real.so101.robot_limits import (
    CAM_KEY, CAM_H, CAM_W, CAM_C, CAM2_KEY, CAM2_H, CAM2_W, CAM2_C,
)
from mpail2.configs.defs import MultiCoderConfig, CNNCoderConfig, PlannerConfig, PolicySamplingConfig
from mpail2.planner import Planner
from mpail2.utils.obs_normalizer import ObsNormalizerFactory


def latest_checkpoint(models_dir: str) -> str:
    candidates = glob.glob(os.path.join(models_dir, "model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {models_dir}")
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", default="demo.pt")
    parser.add_argument("--checkpoint", default=None,
                         help="Path to a model_*.pt checkpoint. Defaults to the most "
                              "recently modified one under logs/so101_local/models.")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--joint_dim", type=int, default=256)
    parser.add_argument("--no_wrist_cam", action="store_true")
    parser.add_argument("--num_pairs", type=int, default=4096,
                         help="Number of (obs, next_obs) demo transitions to sample "
                              "(yields 2x this many latent vectors).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random_init", action="store_true",
                         help="Skip checkpoint loading, use freshly initialized weights "
                              "(architecture-only sanity check, not a trained-model diagnosis).")
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    checkpoint = None if args.random_init else (args.checkpoint or latest_checkpoint("logs/so101_local/models"))
    print(f"Checkpoint: {checkpoint if checkpoint else '(random init — no checkpoint loaded)'}")
    print(f"Demo file : {args.demo_path}")

    demonstrations = torch.load(args.demo_path, map_location="cpu", weights_only=False)
    demo_keys = {OBS_KEY, CAM_KEY, CAM2_KEY}
    demonstrations = {k: v.float() for k, v in demonstrations.items() if k in demo_keys}
    if CAM2_KEY not in demonstrations:
        demonstrations[CAM2_KEY] = torch.zeros(
            demonstrations[CAM_KEY].shape[:-1] + (CAM2_C,), dtype=torch.float32
        )

    print("Converting demo joint states to EE proprioception (running FK)...")
    demo_joints = demonstrations[OBS_KEY].numpy()
    N = demo_joints.shape[0]
    ee_proprio = np.zeros((N, 2, EE_PROPRIO_DIM), dtype=np.float32)
    for i in range(N):
        for j in range(2):
            ee_proprio[i, j] = ik_utils.joints_to_ee_proprio(demo_joints[i, j])
    demonstrations[OBS_KEY] = torch.from_numpy(ee_proprio)

    num_pairs = min(args.num_pairs, N)
    idx = np.random.choice(N, size=num_pairs, replace=False)

    _coder_list = [
        MultiCoderConfig.ProprioCoderConfig(obs_key=OBS_KEY, input_dim=EE_PROPRIO_DIM, output_dim=args.joint_dim),
        CNNCoderConfig(obs_key=CAM2_KEY, H=CAM2_H, W=CAM2_W, C=CAM2_C),
    ]
    if not args.no_wrist_cam:
        _coder_list.append(CNNCoderConfig(obs_key=CAM_KEY, H=CAM_H, W=CAM_W, C=CAM_C))
    encoder_cfg = MultiCoderConfig(coder_list=_coder_list)

    planner_cfg = PlannerConfig(
        encoder_cfg=encoder_cfg, action_dim=5, latent_dim=args.latent_dim,
        sampling_cfg=PolicySamplingConfig(),
    )

    planner = Planner(policy_config=planner_cfg, num_envs=1, device=device, dtype=torch.float32)
    if checkpoint is not None:
        saved_dict = torch.load(checkpoint, map_location=device, weights_only=False)
        planner.load_state_dict(saved_dict["model_state_dict"])
    planner.eval()

    obs_normalizer = ObsNormalizerFactory.create_normalizer(normalization_type="cam_only", device=device)

    # Stack obs (t) and next_obs (t+1) together for more diverse samples.
    obs = {
        OBS_KEY: torch.cat([demonstrations[OBS_KEY][idx, 0], demonstrations[OBS_KEY][idx, 1]], dim=0).to(device),
        CAM_KEY: torch.cat([demonstrations[CAM_KEY][idx, 0], demonstrations[CAM_KEY][idx, 1]], dim=0).to(device),
        CAM2_KEY: torch.cat([demonstrations[CAM2_KEY][idx, 0], demonstrations[CAM2_KEY][idx, 1]], dim=0).to(device),
    }
    obs = obs_normalizer(obs)

    latents = []
    batch_size = 512
    total = obs[OBS_KEY].shape[0]
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = {k: v[start:end] for k, v in obs.items()}
            latents.append(planner.encoder(batch).cpu())
    Z = torch.cat(latents, dim=0)  # [2*num_pairs, latent_dim]

    print(f"\nEncoded {Z.shape[0]} latent vectors, dim={Z.shape[1]}")

    mean = Z.mean(dim=0)
    std = Z.std(dim=0)
    print(f"\nPer-dim latent std: min={std.min():.4g}  median={std.median():.4g}  "
          f"mean={std.mean():.4g}  max={std.max():.4g}")

    for thresh_frac in (0.01, 0.05, 0.10):
        thresh = thresh_frac * std.max()
        n_dead = (std < thresh).sum().item()
        print(f"  dims with std < {thresh_frac*100:.0f}% of max std ({thresh:.4g}): "
              f"{n_dead} / {std.numel()} ({100*n_dead/std.numel():.1f}%)")

    centered = Z - mean
    # SVD of centered latents for the variance spectrum (avoids materializing the
    # full [dim, dim] covariance matrix).
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    eigvals = (S ** 2) / (Z.shape[0] - 1)
    total_var = eigvals.sum()
    explained = torch.cumsum(eigvals, dim=0) / total_var

    participation_ratio = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    print(f"\nEffective rank (participation ratio): {participation_ratio:.1f} / {Z.shape[1]} dims")

    for frac in (0.50, 0.90, 0.99):
        n_dims = int((explained >= frac).nonzero()[0].item()) + 1
        print(f"  dims needed for {frac*100:.0f}% of variance: {n_dims} / {Z.shape[1]}")

    print(f"\nTop-5 eigenvalue share of total variance: "
          f"{[f'{(e/total_var).item():.3f}' for e in eigvals[:5]]}")

    print("\n--- Interpretation ---")
    print(f"latent_dim={Z.shape[1]}. If effective rank is a small fraction of latent_dim, "
          f"or a handful of dims explain >90% of variance, or many dims have near-zero std, "
          f"that's the signature of representation collapse (encoder ignoring most of what "
          f"it's given, dynamics loss looking good for the wrong reason).")


if __name__ == "__main__":
    main()
