"""
Train MPAIL on Gymnasium Environments

A clean training script for MPAIL (Model-based Planning with Adversarial Imitation Learning)
on standard gymnasium environments like Ant-v5, Hopper-v5, Humanoid-v5, etc.

MPAIL differs from MGAIL by using model-based planning (MPPI) instead of
model-free policy optimization. This allows it to:
- Plan multiple steps ahead using the learned dynamics model
- Often achieves better sample efficiency
- Requires more computation per action

This script:
- Uses gymnasium environments with MPAIL-compatible wrappers
- Loads expert demonstrations in MPAIL format
- Creates MPAIL configuration programmatically
- Supports wandb logging and video recording
- Does NOT require Isaac Sim

Usage:
    python train_mpail_gym.py --env Ant-v5
    python train_mpail_gym.py --env Humanoid-v5
    python train_mpail_gym.py --env Hopper-v5 --num-envs 8
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")  # Use EGL for headless GPU rendering

import dataclasses
import sys
import argparse
import random
from datetime import datetime

import torch

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Add train directory for local imports
sys.path.insert(0, os.path.dirname(__file__))

# Gymnasium MPAIL imports - using common utilities
from gym import (
    setup_environment,
    get_env_dimensions,
    load_demonstrations,
)
from gym.gym_configs import create_mpail_runner_cfg, GymLogConfig
# Video recording is handled by setup_environment() via _VideoRecordingWrapper

# MPAIL2 imports
from mpail2.runner import MPAIL2Runner

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available. Install with: pip install wandb")

# Local utilities
from train.utils import find_demo_file


def main():
    parser = argparse.ArgumentParser(
        description="Train MPAIL on Gymnasium environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Environment
    parser.add_argument("--env", type=str, default="Ant-v5",
                       help="Gymnasium environment ID")
    parser.add_argument("--num-envs", type=int, default=1,
                       help="Number of parallel environments")
    parser.add_argument("--max-episode-length", type=int, default=1000,
                       help="Maximum episode length")

    # Demonstrations
    parser.add_argument("--demo-path", type=str, default=None,
                       help="Path to demonstration file (.pt). Auto-detected if not provided.")
    parser.add_argument("--demo-dir", type=str, default=None,
                       help="Base directory to search for demo files")
    parser.add_argument("--num-demos", type=int, default=None,
                       help="Number of demonstrations to use (None = all)")

    # Training
    parser.add_argument("--num-iterations", type=int, default=200,
                       help="Number of training iterations")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")

    # Model architecture
    parser.add_argument("--latent-dim", type=int, default=256,
                       help="Latent dimension for dynamics model")
    parser.add_argument("--loss-horizon", type=int, default=7,
                       help="Horizon for trajectory-based losses")
    parser.add_argument("--replay-size", type=int, default=1_000_000,
                       help="Replay buffer size")
    parser.add_argument("--replay-batch-size", type=int, default=256,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")

    # MPPI Planner (specific to MPAIL)
    parser.add_argument("--num-rollouts", type=int, default=512,
                       help="Number of MPPI rollouts")
    parser.add_argument("--planning-horizon", type=int, default=7,
                       help="MPPI planning horizon (timesteps)")
    parser.add_argument("--opt-iters", type=int, default=5,
                       help="MPPI optimization iterations")

    # Logging
    parser.add_argument("--log-dir", type=str, default="./logs/mpail",
                       help="Base directory for logs")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Run name (auto-generated if not provided)")
    parser.add_argument("--wandb", type=bool, default=False,
                        help="Use wandb for logging (requires WANDB_PROJECT and WANDB_ENTITY env vars)")
    # Note: WANDB_PROJECT and WANDB_ENTITY must be set as environment variables
    parser.add_argument("--checkpoint-every", type=int, default=50,
                       help="Checkpoint frequency (iterations)")

    # Video
    parser.add_argument("--video", action="store_true",
                       help="Record training videos")
    parser.add_argument("--video-interval", type=int, default=5000,
                       help="Video recording interval (steps). Videos are recorded at iteration boundaries.")
    parser.add_argument("--video-length", type=int, default=None,
                       help="Video length (steps). Default: max_episode_length (full iteration)")
    parser.add_argument("--video-fps", type=int, default=30,
                       help="Video playback fps (default 50 for MuJoCo)")

    # Environment options
    parser.add_argument("--no-termination", action="store_true",
                       help="Disable early termination (terminate_when_unhealthy=False). "
                            "Recommended for imitation learning to let agent recover from bad states.")

    args = parser.parse_args()

    # Auto-detect demo file if not provided
    if args.demo_path is None:
        args.demo_path = find_demo_file(args.env, args.demo_dir)
    elif not os.path.exists(args.demo_path):
        # Try to find the file relative to common locations
        print(f"[WARN] Demo file not found at: {args.demo_path}")
        args.demo_path = find_demo_file(args.env, args.demo_dir)

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.env}_{timestamp}"

    # Create log directory
    log_dir = os.path.join(args.log_dir, args.env, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)

    print("=" * 60)
    print("MPAIL Training on Gymnasium Environment")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Num envs: {args.num_envs}")
    print(f"Demonstrations: {args.demo_path}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Planning horizon: {args.planning_horizon}")
    print(f"MPPI rollouts: {args.num_rollouts}")
    print(f"Log directory: {log_dir}")
    print("=" * 60)

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and args.wandb
    wandb_project = None
    wandb_entity = None
    if use_wandb:
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_entity = os.environ.get("WANDB_ENTITY")
        if not wandb_project or not wandb_entity:
            raise ValueError(
                "WANDB_PROJECT and WANDB_ENTITY environment variables must be set. "
                "Set them with: export WANDB_PROJECT=<project> WANDB_ENTITY=<entity>\n"
                "Or disable wandb with --no-wandb"
            )
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        print(f"[INFO] Wandb logging enabled: {wandb_entity}/{wandb_project}/{args.run_name}")

    # Create log config
    log_cfg = GymLogConfig(
        run_log_dir=log_dir,
        run_name=args.run_name,
        wandb=use_wandb,
        checkpoint_every=args.checkpoint_every,
        video=args.video,
        video_interval=args.video_interval,
        video_length=args.video_length,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        termination_when_unhealthy=not args.no_termination,
    )

    # Setup environment using common utility
    render_mode = "rgb_array" if args.video else None
    video_dir = os.path.join(log_dir, "videos") if args.video else None

    env = setup_environment(
        env_id=args.env,
        num_envs=args.num_envs,
        max_episode_length=args.max_episode_length,
        device=args.device,
        render_mode=render_mode,
        video_folder=video_dir if args.video else None,
        video_step_trigger=lambda step: step % args.video_interval == 0,
        video_length=args.video_length,  # None = use max_episode_length (full iteration)
        enable_wandb=use_wandb,  # Only log to wandb if wandb is active
        video_fps=args.video_fps,
        terminate_when_unhealthy=not args.no_termination,
    )

    # Get dimensions using common utility
    obs_dim, action_dim = get_env_dimensions(env, args.num_envs)
    print(f"[INFO] Observation dim: {obs_dim}")
    print(f"[INFO] Action dim: {action_dim}")

    # Note: Video recording is already handled by setup_environment() via video_folder parameter

    # Load demonstrations
    print(f"[INFO] Loading demonstrations from: {args.demo_path}")
    demonstrations, metadata = load_demonstrations(args.demo_path, device=args.device, num_demos=args.num_demos)

    for key, tensor in demonstrations.items():
        print(f"  {key}: {tensor.shape}")
    if metadata:
        print(f"  Metadata: {metadata}")

    # Create MPAIL config
    print("[INFO] Creating MPAIL configuration...")
    runner_cfg = create_mpail_runner_cfg(
        state_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        num_learning_iterations=args.num_iterations,
        path_to_demonstrations=args.demo_path,
        num_demos=args.num_demos,
        seed=args.seed,
        logger="wandb" if use_wandb else None,
        replay_size=args.replay_size,
        replay_batch_size=args.replay_batch_size,
        loss_horizon=args.loss_horizon,
        num_rollouts=args.num_rollouts,
        opt_iters=args.opt_iters,
        lr=args.lr,
        use_terminations=not args.no_termination,
        log_cfg=log_cfg,  # Pass the GymLogConfig
    )

    if log_cfg.wandb:
        wandb.config.update(dataclasses.asdict(runner_cfg))

    # Create runner
    print("[INFO] Creating MPAIL2 runner...")
    runner = MPAIL2Runner(
        demonstrations=demonstrations,
        env=env,
        runner_cfg=runner_cfg,
        device=args.device,
    )

    # Set environment seed
    env.seed(args.seed)

    # Train
    print("[INFO] Starting training...")
    try:
        runner.learn()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    finally:
        # Save final model
        print("[INFO] Saving final model...")
        runner.save(postfix="final")

        # Cleanup
        if use_wandb:
            wandb.finish()
        env.close()

    print("[INFO] Training complete!")
    print(f"[INFO] Models saved to: {os.path.join(log_dir, 'models')}")


if __name__ == "__main__":
    main()
