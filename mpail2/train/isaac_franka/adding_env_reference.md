# Reference: env module metadata and demo resolution

Used when adding an Isaac image task (see the [three-step guide](README.md#adding-a-new-isaac-task) in `README.md`).

`ConfigStore.instance().store(name=..., ...)` **must** use the same string as `ENV_SPEC["config_name"]` (that is the Hydra `config_name` passed to `compose`).

| Constant | Purpose |
|----------|---------|
| `ENV_NAME` | Primary CLI token for `--env` (e.g. `push`). |
| `ENV_ALIASES` | Optional extra strings accepted by `--env` (e.g. full Gym id). All map to the same env as `ENV_NAME`. |
| `ENV_SPEC` | Dict (or [`TrainEnvSpec`](../registry.py)) with the keys below. |

| `ENV_SPEC` key | Purpose |
|----------------|---------|
| `suite` | Training suite dispatcher id (currently `isaac_image` for Isaac image tasks). |
| `config_name` | Hydra top-level config name; must match `ConfigStore.store(name=...)`. |
| `demo_env_var` | Optional shell env var name for a demo file path when `runner.path_to_demonstrations` is unset (e.g. `MPAIL_PUSH_DEMO`). |
| `default_demo_rel` | Default demo path when Hydra and `demo_env_var` are unset; resolved under search dirs in [`utils/demos.py`](../utils/demos.py). |
| `default_num_iterations` | Fallback when `runner.num_learning_iterations` is `None` in Hydra. |

## Demo path resolution (first match wins)

1. Hydra `runner.path_to_demonstrations` if set.
2. Else path from the environment variable `ENV_SPEC["demo_env_var"]`, if set.
3. Else `ENV_SPEC["default_demo_rel"]` via [`utils/demos.py`](../utils/demos.py) and [`mpail2/envs/utils/paths.py`](../../envs/utils/paths.py).

## mpail2 side

Register the Isaac Lab / Gymnasium task and ship or download demonstrations so the resolved path exists.
