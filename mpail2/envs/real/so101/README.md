# SO-101 real-robot environment

Real SO-101 (SO-ARM101) 6-DOF arm support for MPAIL2 — ported from the
`mpail-lerobot` fork. Structured to mirror `mpail2/envs/real/franka/`: the
gym-env-facing code lives directly in this package, the hardware-owning gRPC
server lives under `network/` (like Franka's `network/server.py` /
`network/client.py`), and the standalone training/data scripts live under
`training/`.

Unlike Franka (custom hardware driver in `mpail2/envs/real/franka/hardware/`),
SO-101 drives the arm + cameras through LeRobot's own driver stack
(`SO101Follower`, OpenCV/RealSense camera classes) via a patched copy of
LeRobot's `async_inference` protocol — hence `lerobot_patch/`, which has no
Python role inside `mpail2` itself; it's a deploy target for a separate
`lerobot` clone (see below).

## Layout

```
so101/
  __init__.py            # exports: SO101RobotEnv, SO101RealWrapper, SO101RealEnvArgs, make_so101_env, ...
  so101_env.py            # gym.Env client (gRPC) — analogous to Franka's network/client.py's FrankaClient
  env_factory.py           # make_so101_env() factory + args dataclass wiring
  wrappers.py              # SO101RealWrapper (MPAIL-shaped obs/action)
  robot_limits.py          # joint/EE bounds, home pose, dims, camera specs
  ik_utils.py               # FK/IK (ikpy + soa.urdf), used by so101_env.py
  soa.urdf                  # arm description used by ik_utils.py
  transport/                 # gRPC stubs (generated from so101_robot.proto / lerobot's services.proto)
  network/
    server.py                # gRPC server owning the physical arm + cameras
  training/
    demo_recording_server.py  # gRPC server for recording (obs, next_obs) demo pairs
    train_so101_local.py       # main MPAIL2 training entry point (in-process env loop)
    convert.py                 # raw_demos*/*.npz -> demo.pt
    convert_lerobot.py          # LeRobot-recorded dataset -> demo.pt
    replay_demo.py               # replay/sanity-check a recorded .npz trajectory
    check_encoder_collapse.py     # offline diagnostic: encoder latent effective-rank check
  lerobot_patch/
    async_inference/               # patched lerobot/src/lerobot/async_inference/ — deploy into your lerobot clone
```

## Install

```bash
conda create -n mpail2 python=3.10
conda activate mpail2
pip install -e ".[so101]"
```

You'll also need a separate `lerobot` conda env (Python 3.12) with the
[`lerobot`](https://github.com/huggingface/lerobot) repo installed — it owns
the low-level Feetech servo + camera drivers this package talks to:

```bash
conda create -n lerobot python=3.12
conda activate lerobot
cd <path-to-lerobot-clone> && pip install -e .
pip install pyrealsense2   # RealSense camera support
pip install -e "<path-to-this-mpail2-clone>[so101]"   # so mpail2.envs.real.so101.network.server is importable here too
```

Deploy the LeRobot patch once (re-run after pulling upstream `lerobot`
changes, or after editing anything under `lerobot_patch/async_inference/`
here — it fixes a camera-reconnect bug, a timestep off-by-one, and adds a
block-until-server-ready sync point between episodes/training updates):

```bash
cp -r mpail2/envs/real/so101/lerobot_patch/async_inference/* <lerobot-clone>/src/lerobot/async_inference/
```

### Known issue: the `lerobot` env needs mpail2's *full* dependency stack

Because `network/server.py` is a submodule of the `mpail2` package, importing
it (even via `-m`) runs `mpail2/__init__.py` first — which eagerly imports the
whole algorithm stack (`configs`, `encoder`, `learner`, ...), pulling in
`torch`/`hydra-core`/`wandb`/`scipy`/`matplotlib`, not just the
grpc/opencv/pyrealsense2 that `network/server.py` itself actually needs. This
is inherent to nesting the server inside the package (Franka's
`network/server.py` has the exact same coupling) — it isn't specific to
SO-101. Concretely, this means the `lerobot` conda env needs a `matplotlib`
version compatible with `mpail2/utils/rollout_vis.py`'s `matplotlib.cm.get_cmap`
usage (removed in matplotlib ≥ 3.9) — pin an older matplotlib there if you hit
`ImportError: cannot import name 'get_cmap' from 'matplotlib.cm'`.

## Running scripts

Everything here is a proper submodule of the installed `mpail2` package, so
invoke with `-m` from anywhere (no need to `cd` to a particular directory,
unlike the original `mpail-lerobot` fork's top-level scripts):

```bash
python -m mpail2.envs.real.so101.network.server --help
python -m mpail2.envs.real.so101.training.train_so101_local --help
```

## Workflow

### 1. Calibrate (once per arm, `lerobot` env)

```bash
conda activate lerobot
lerobot-find-port
lerobot-calibrate --robot.type=so_follower --robot.port=/dev/ttyACM0 --robot.id=<robot_id>
lerobot-calibrate --teleop.type=so_leader --teleop.port=/dev/ttyACM1 --teleop.id=<leader_id>
```

### 2. Start the robot-side server (`lerobot` env, keep running)

```bash
conda activate lerobot
python -m mpail2.envs.real.so101.network.server \
    --robot_port /dev/ttyACM0 --robot_id <robot_id> \
    --cam_index /dev/video0 --cam2_serial <realsense_serial> --grpc_port 7070
```

Owns the arm + both cameras; backs `training/train_so101_local.py`'s online
training loop. See its `--help` for servo-tuning flags
(`--p_coefficient`, `--i_coefficient`, `--goal_velocity`, ...) if motion is
shaky or not settling.

### 3. Collect demonstrations

Start the recording server (`mpail2` env):

```bash
conda activate mpail2
python -m mpail2.envs.real.so101.training.demo_recording_server --collect_dir ./raw_demos2 --flush_every 200
```

It holds the arm still (or drives home between episodes) and records every
`(obs_t, obs_t+1)` pair it receives. Drive the arm via LeRobot's standard
teleop client (`lerobot` env, separate terminal):

```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=<robot_id> \
    --robot.cameras="{cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, cam2: {type: intelrealsense, serial_number_or_name: <realsense_serial>, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=<leader_id> \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=dummy \
    --actions_per_chunk=1 \
    --task="pick up the cup"
```

Flag notes:
- `--robot.cameras`: `cam` (wrist, OpenCV) and `cam2` (RealSense, by serial number).
- `--server_address`: must match `demo_recording_server.py`'s `--port` (default 8080).
- `--policy_type` / `--pretrained_name_or_path`: required by `robot_client`'s
  CLI even here, where the server ignores them and just echoes joint state
  back — `act` / `dummy` are placeholders, not a real policy.
- Move the leader arm to demonstrate the task; the follower mirrors it and
  every step gets recorded. Each episode auto-ends (saves, homes, pauses,
  resumes) after `--max_episode_steps` (default 200) steps.

### 4. Convert to training format

```bash
conda activate mpail2
python -m mpail2.envs.real.so101.training.convert --dirs raw_demos2 --out demo.pt --img_w 64 --img_h 48
```

(Recorded via a LeRobot dataset instead? Use `convert_lerobot.py` in the
`lerobot` env instead.)

Sanity-check a trajectory by replaying it on the real arm:

```bash
conda activate lerobot
python -m mpail2.envs.real.so101.training.replay_demo raw_demos2/traj_0000.npz --port /dev/ttyACM0 --robot_id <robot_id>
```

### 5. Train

With the robot-side server (step 2) still running:

```bash
conda activate mpail2
python -m mpail2.envs.real.so101.training.train_so101_local \
    --demo_path demo.pt --robot_host 127.0.0.1 --robot_port 7070 \
    --device cuda --speed_scale 0.4 --lpf_alpha 0.5 --wandb
```

See `--help` for the full flag list (MPPI sampling, gripper hold steps,
checkpointing, eval mode, ...). `--eval --load_checkpoint <path>` rolls out a
trained checkpoint with no training updates; add `--eval_policy_only` to
bypass CEM/MPPI and use the policy network's own deterministic action.

### 6. Diagnose encoder collapse (optional)

```bash
conda activate mpail2
python -m mpail2.envs.real.so101.training.check_encoder_collapse \
    --demo_path demo.pt --checkpoint logs/so101_local/models/model_N.pt
```

Reports per-dimension latent std and effective rank (participation ratio) — a
low effective rank relative to `latent_dim` is the signature of
representation collapse even when the JEP/dynamics loss looks fine.
