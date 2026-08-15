"""
train_so101_local.py — Train MPAIL2 on the real SO-101 via the local, non-dropping
gRPC env (mpail2/envs/real/so101), instead of the LeRobot async_inference /
demo_recording_server.py RPC-handler path.

Unlike demo_recording_server.py (which reacts to inbound SendObservations/GetActions
RPCs from a separately-running lerobot robot_client), this script drives the
robot itself, in-process, via MPAIL2Runner.learn()'s standard rollout loop —
env.step()/env.reset() block until the real action has been sent and the fresh
post-action state read back, so no observation is ever dropped.

Requires mpail2.envs.real.so101.network.server to be running first (separate
`lerobot` conda env, connected to the real arm + cameras). See
mpail2/envs/real/so101/README.md for the full setup.

    conda activate lerobot
    python -m mpail2.envs.real.so101.network.server --robot_port /dev/ttyACM0 --robot_id Kid \\
        --cam_index /dev/video0 --cam2_serial <realsense_serial> --grpc_port 7070

Then, in the `mpail2` conda env:

    conda activate mpail2
    python -m mpail2.envs.real.so101.training.train_so101_local --demo_path demo.pt --robot_host 127.0.0.1 \\
        --robot_port 7070 --device cuda --speed_scale 0.4 --wandb
"""

import argparse
import logging
import os

import numpy as np
import torch

from mpail2.envs.real.so101 import ik_utils
from mpail2.envs.real.so101 import (
    SO101RealEnvArgs, make_so101_env, OBS_KEY, ACTION_DIM, EE_PROPRIO_DIM,
)
from mpail2.envs.real.so101.robot_limits import (
    CAM_KEY, CAM_H, CAM_W, CAM_C, CAM2_KEY, CAM2_H, CAM2_W, CAM2_C,
)
from mpail2.runner import MPAIL2Runner
from mpail2.configs.cfgs import MPAIL2RunnerCfg, ObsNormalizerCfg
from mpail2.configs.defs import (
    MultiCoderConfig, CNNCoderConfig, PlannerConfig, PolicySamplingConfig, LearnerConfig,
)

# so101_env.py logs per-step action/observation details via logging.getLogger("so101_env");
# this handler is what actually makes them visible on the console.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Train MPAIL2 on the real SO-101 via the local gRPC env (no dropped observations)"
    )
    parser.add_argument("--demo_path", required=True, help="Path to .pt demo file (same format as grpc_policy_server.py)")
    parser.add_argument("--robot_host", default="127.0.0.1", help="so101_robot_server.py host")
    parser.add_argument("--robot_port", type=int, default=7070, help="so101_robot_server.py gRPC port")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_dir", default="logs/so101_local")
    parser.add_argument("--num_rollouts", type=int, default=512)
    parser.add_argument("--num_elites", type=int, default=64)
    parser.add_argument("--opt_iters", type=int, default=5)
    parser.add_argument("--min_std", type=float, default=0.1,
                         help="MPPI sampling std floor. Doesn't help early on — every reset/first "
                              "planning iter reinitializes _iter_std to max_std regardless.")
    parser.add_argument("--max_std", type=float, default=3.0,
                         help="MPPI sampling std ceiling on the tanh-squashed [-1,1] action — this is "
                              "the actual knob for 'how large is the raw action', independent of "
                              "speed_scale/lpf_alpha (which only rescale/smooth it afterward). An "
                              "untrained value fn won't shrink this much via CEM refinement, so a fresh "
                              "model will keep sampling near-saturated actions until it's had some "
                              "training. Lower this (e.g. 0.5-1.0) for gentler motion pre-training.")
    parser.add_argument("--policy_proportion", type=float, default=0.05,
                         help="Fraction of MPPI rollouts drawn from the learned policy net vs pure "
                              "random exploration (default 0.05 = 95%% random, matches grpc_policy_server.py).")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--joint_dim", type=int, default=256)
    parser.add_argument("--speed_scale", type=float, default=1.0)
    parser.add_argument("--lpf_alpha", type=float, default=1.0,
                         help="Low-pass filter weight on the new action (0=frozen, 1=no filter). "
                              "Tames jerky/saturated MPPI output; speed_scale alone won't fix that "
                              "since it only rescales the envelope, not how often the action sits at it.")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--gripper_hold_steps", type=int, default=1,
                         help="Only accept a new gripper command every Nth step (holding the "
                              "current target fixed the rest of the time), so gripper motion is a "
                              "deliberate chunked open/close instead of raw per-step MPPI jitter. "
                              "1 = update every step (default, unchanged behavior).")
    parser.add_argument("--max_episode_steps", type=int, default=200,
                         help="Steps per learn() iteration (one rollout before each update()).")
    parser.add_argument("--reset_pause_seconds", type=float, default=3.0,
                         help="Pause after homing before the next episode's actions start, so the "
                              "arm actually settles instead of immediately being commanded again.")
    parser.add_argument("--loss_horizon", type=int, default=7)
    parser.add_argument("--replay_size", type=int, default=40_000)
    parser.add_argument("--replay_batch_size", type=int, default=256)
    parser.add_argument("--num_episodes", type=int, default=200,
                         help="Number of episodes to run. Each episode is one runner.learn() call "
                              "(max_episode_steps env steps + one update()), reset to home first.")
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--load_checkpoint", default=None, help="Resume from a .pt checkpoint saved by MPAIL2Runner.save().")
    parser.add_argument("--no_wrist_cam", action="store_true", help="Exclude wrist cam from the encoder.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="so101-mpail2-local")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--eval", action="store_true",
                         help="Roll the loaded checkpoint out on the robot with no training updates, "
                              "no replay storage, no checkpoint saving — just act() + env.step(). "
                              "Requires --load_checkpoint.")
    parser.add_argument("--num_eval_episodes", type=int, default=10,
                         help="Number of episodes to roll out when --eval is set.")
    parser.add_argument("--eval_policy_only", action="store_true",
                         help="During --eval, bypass CEM/MPPI entirely and use the policy network's "
                              "own deterministic mean action every step (Planner.act_policy_only) — "
                              "100%% policy-driven, no noise rollouts, no elite re-weighting. Default "
                              "eval behavior otherwise still runs the full CEM search with "
                              "deterministic=True on the final iteration.")
    args = parser.parse_args()

    if args.eval and not args.load_checkpoint:
        parser.error("--eval requires --load_checkpoint.")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print(f"Loading demonstrations from {args.demo_path} ...")
    demonstrations = torch.load(args.demo_path, map_location="cpu", weights_only=False)
    demo_keys = {OBS_KEY, CAM_KEY, CAM2_KEY}
    demonstrations = {k: v.float() for k, v in demonstrations.items() if k in demo_keys}
    if CAM_KEY not in demonstrations:
        raise RuntimeError(f"Demo file missing '{CAM_KEY}'. Keys: {list(demonstrations)}")
    if CAM2_KEY not in demonstrations:
        print(f"[WARN] Demo file missing '{CAM2_KEY}' — training without RealSense camera.")
        demonstrations[CAM2_KEY] = torch.zeros(
            demonstrations[CAM_KEY].shape[:-1] + (CAM2_C,), dtype=torch.float32
        )

    # No re-pairing/subsampling: at the current control rate + speed_scale, online
    # per-step EE motion (MAX_DELTA_M * speed_scale) is already the same single-digit-mm
    # order of magnitude as the demo's raw per-frame motion, so there's no longer a large
    # scale gap for the discriminator to trivially exploit — unlike when this was tuned
    # against a 1Hz online control loop. Using demos in their raw recorded form also
    # avoids re-pairing groups ever straddling two independently-recorded demo takes.
    print("  Converting demo joint states to EE proprioception (running FK)...")
    demo_joints = demonstrations[OBS_KEY].numpy()
    N = demo_joints.shape[0]
    ee_proprio = np.zeros((N, 2, EE_PROPRIO_DIM), dtype=np.float32)
    for i in range(N):
        for j in range(2):
            ee_proprio[i, j] = ik_utils.joints_to_ee_proprio(demo_joints[i, j])
    demonstrations[OBS_KEY] = torch.from_numpy(ee_proprio)
    for k, v in demonstrations.items():
        print(f"  {k}: {tuple(v.shape)}")

    print(f"Connecting to so101_robot_server.py at {args.robot_host}:{args.robot_port} ...")
    env = make_so101_env(SO101RealEnvArgs(
        device=device, host=args.robot_host, port=args.robot_port,
        max_episode_length=args.max_episode_steps, speed_scale=args.speed_scale,
        lpf_alpha=args.lpf_alpha, reset_pause_seconds=args.reset_pause_seconds,
        gripper_hold_steps=args.gripper_hold_steps, mock=False,
    ))

    _coder_list = [
        MultiCoderConfig.ProprioCoderConfig(obs_key=OBS_KEY, input_dim=EE_PROPRIO_DIM, output_dim=args.joint_dim),
        CNNCoderConfig(obs_key=CAM2_KEY, H=CAM2_H, W=CAM2_W, C=CAM2_C),
    ]
    if not args.no_wrist_cam:
        _coder_list.append(CNNCoderConfig(obs_key=CAM_KEY, H=CAM_H, W=CAM_W, C=CAM_C))
    encoder_cfg = MultiCoderConfig(coder_list=_coder_list)

    planner_cfg = PlannerConfig(
        encoder_cfg=encoder_cfg, action_dim=ACTION_DIM, latent_dim=args.latent_dim,
        sampling_cfg=PolicySamplingConfig(
            num_rollouts=args.num_rollouts, num_timesteps=args.loss_horizon,
            policy_proportion=args.policy_proportion, min_std=args.min_std, max_std=args.max_std,
        ),
        opt_iters=args.opt_iters, num_elites=args.num_elites,
    )
    learner_cfg = LearnerConfig(
        planner_cfg=planner_cfg, replay_size=args.replay_size, replay_batch_size=args.replay_batch_size,
        loss_horizon=args.loss_horizon, use_terminations=False,
        obs_normalizer_cfg=ObsNormalizerCfg(normalization_type="cam_only"),
    )

    if args.wandb:
        import wandb as _wandb
        _wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        print(f"W&B run: {_wandb.run.url}")

    log_cfg = MPAIL2RunnerCfg.LogCfg(
        log_dir=args.log_dir, checkpoint_every=args.checkpoint_every,
        no_wandb=not args.wandb, logger="wandb" if args.wandb else None, video_interval=999_999,
    )
    # num_learning_iterations is fixed at 1: runner.learn() calls env.reset() once at its
    # own start (runner.py:108), then rolls out max_episode_steps steps and updates. We call
    # learn() once per episode from an outer loop below, so each call's own reset() gives us
    # per-episode homing "for free" without touching runner.py's shared learn() loop (which
    # also backs Isaac-sim training, where individual sub-envs auto-reset internally — a real
    # robot doesn't, so calling learn() once for many iterations would never re-home the arm).
    runner_cfg = MPAIL2RunnerCfg(
        learner_cfg=learner_cfg, log_cfg=log_cfg,
        num_learning_iterations=1, logger="wandb" if args.wandb else None, vis_rollouts=False,
    )
    os.makedirs(os.path.join(args.log_dir, "models"), exist_ok=True)
    runner = MPAIL2Runner(demonstrations=demonstrations, env=env, runner_cfg=runner_cfg, device=device)

    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint} ...")
        runner.load(args.load_checkpoint)
        ckpt = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        runner.current_learning_iteration = int(ckpt.get("iter", 0))
        print(f"  Resuming from iteration {runner.current_learning_iteration}")

    runner.learner._reward_scale = float(args.reward_scale)

    # Per-step reward logging: env.step()'s own reward stays 0.0 (learner.py: "NOT TO BE USED
    # IN LEARNING - JUST FOR LOGGING") — the actual GAIL discriminator reward comparing
    # consecutive latent embeddings is what grpc_policy_server.py computes inline for its
    # [ep N step M] disc_reward= log lines. Wire the same computation into the wrapper here
    # so it prints every step without touching runner.py's shared learn() loop.
    def _disc_reward_fn(obs, next_obs):
        with torch.no_grad():
            z = runner.learner._encoder(obs)
            zn = runner.learner._encoder(next_obs)
            return float(runner.learner._reward(z, zn).item())
    env.reward_fn = _disc_reward_fn

    print(
        f"Runner ready  device={device}  proprio_dim={EE_PROPRIO_DIM}  action_dim={ACTION_DIM}  "
        f"latent_dim={planner_cfg.latent_dim}"
    )

    if args.eval:
        print(f"Evaluating checkpoint: {args.num_eval_episodes} episodes of {args.max_episode_steps} "
              f"steps each, no training updates "
              f"({'policy-net only, no CEM' if args.eval_policy_only else 'full CEM, deterministic final iter'}) ...")
        runner.learner.eval()
        for episode in range(args.num_eval_episodes):
            print(f"[eval episode {episode + 1}/{args.num_eval_episodes}] resetting to home, then rolling out ...")
            obs, info = env.reset()
            runner.learner.planner.reset()
            for _ in range(args.max_episode_steps):
                with torch.no_grad():
                    if args.eval_policy_only:
                        action = runner.learner.planner.act_policy_only(obs)
                    else:
                        action = runner.learner.planner.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action.to(env.unwrapped.device))
                if terminated or truncated:
                    break
        return

    print(f"Starting training: {args.num_episodes} episodes of {args.max_episode_steps} steps each ...")

    start_episode = runner.current_learning_iteration
    try:
        for episode in range(start_episode, start_episode + args.num_episodes):
            print(f"[episode {episode + 1}/{start_episode + args.num_episodes}] resetting to home, then rolling out ...")
            runner.learn()
            # runner.learn() always runs exactly one iteration (runner_cfg.num_learning_iterations=1
            # above) and its internal current_learning_iteration bookkeeping is a no-op for that
            # case (it just resets to whatever it already was) — advance it here instead, or every
            # checkpoint keeps overwriting "model_1.pt" and wandb's it/tot_iter never move.
            runner.current_learning_iteration = episode + 1
    except KeyboardInterrupt:
        print("Interrupted — saving checkpoint before exit...")
        runner.save(postfix=f"ep{runner.current_learning_iteration}_interrupted")
        raise


if __name__ == "__main__":
    main()
