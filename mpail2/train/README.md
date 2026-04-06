# Training

With the package installed (`pip install -e .` from the repository root), use **`python -m mpail2.train.train`** or **`mpail2-train`** (`mpail2-train-isaac` is the same).

## Routing

- **`--env`** is inspected first. If it matches an Isaac registry name or alias (including the default **`push`** when `--env` is omitted), the **Isaac Lab + Hydra** path runs.
- Otherwise the **Gymnasium / MuJoCo** path runs with the same argv (e.g. `--env Ant-v5`).

Avoid naming a future Isaac alias exactly the same as a Gymnasium id you intend for MuJoCo, or the Isaac path would win.

## Isaac Lab (Hydra)

- **Example:** `python -m mpail2.train.train --env pick_place --headless log.no_wandb=True`
- **How to add a task:** See [Adding a new Isaac task](isaac_franka/README.md#adding-a-new-isaac-task) in [`isaac_franka/README.md`](isaac_franka/README.md); field definitions: [`adding_env_reference.md`](isaac_franka/adding_env_reference.md).

## Gymnasium / MuJoCo

- **Example:** `python -m mpail2.train.train --env Ant-v5 learner.replay_size=100000 log.no_wandb=True`.
