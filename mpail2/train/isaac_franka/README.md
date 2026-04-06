# Isaac Franka Training

Training is driven by [`mpail2/train/train.py`](../train.py) (`python -m mpail2.train.train` or `mpail2-train`; `mpail2-train-isaac` is the same entry point).

## Demo Files

If you need the Isaac Franka demonstration files, download them with Git LFS:

```bash
git lfs pull --include="mpail2/train/isaac_franka/demos/*"
```

## Installation

- Install Isaac Sim / IsaacLab before using these training scripts.
- See the [installation guide](../../../docs/INSTALL.md#isaaclab--isaac-sim) for the full environment setup instructions.

## Training Example

With `mpail2` installed in your environment:

```bash
python -m mpail2.train.train --env push --headless log.no_wandb=True
python -m mpail2.train.train --env pick_place --headless log.no_wandb=True
```

Equivalent using the console entry point:

```bash
mpail2-train --env push --headless log.no_wandb=True
```

Notes:

- For IsaacSim GUI, remove `--headless`.
- Remove `log.no_wandb=True` if you want W&B logging enabled.
- If wandb is enabled, set `WANDB_PROJECT` and `WANDB_ENTITY` (or provide `log.wandb_project`/`log.wandb_entity` via Hydra overrides).
- Hydra options live on the shared base [`configs/isaac_base_config.py`](../configs/isaac_base_config.py) and the per-env nodes in [`envs/franka_push.py`](../envs/franka_push.py) / [`envs/franka_pick_place.py`](../envs/franka_pick_place.py). The MPAIL `defs.LearnerConfig` preset is **not** a direct Hydra node (OmegaConf cannot represent its `Type[]` fields); it is chosen from `cfg.task` in [`utils/hydra_learner.py`](../utils/hydra_learner.py), then patched by `cfg.learner` dict overrides. CLI examples: `learner.replay_size=100000` or `learner.planner_cfg.latent_dim=256`. With `sync_learner_dims_from_env=true` (default), action and encoder input shapes are aligned to the live env after `gym.make`; set `sync_learner_dims_from_env=false` to use only the preset dimensions.

## Adding a new Isaac task

Use [`envs/franka_push.py`](../envs/franka_push.py) as a template.

1. **Create one env module** — Add `mpail2/train/envs/<your_env>.py` with a `@dataclass` train config subclass of [`IsaacHydraTrainConfig`](../configs/isaac_base_config.py), `ConfigStore.instance().store(...)`, and module constants `ENV_NAME`, optional `ENV_ALIASES`, and `ENV_SPEC` ([field definitions](adding_env_reference.md)).

2. **Set task + demos** — Set `cfg.task` to your Isaac Lab / Gymnasium registration id; provide demonstrations where resolution can find them.

3. **Run** — `python -m mpail2.train.train --env <ENV_NAME-or-alias> ...`

You do not need to edit [`registry.py`](../registry.py) or [`envs/__init__.py`](../envs/__init__.py); env modules are discovered automatically.

## Gymnasium (MuJoCo) training

Use `python -m mpail2.train.train --env Ant-v5 ...` (any Gym id not in the Isaac registry).
