"""Unified training entry: Isaac Lab (Hydra) or Gymnasium / MuJoCo (Hydra).

Routing uses ``--env``: values registered for Isaac (default ``push`` if omitted) run
the Isaac + Hydra pipeline; any other ``--env`` (e.g. ``Ant-v5``) composes the ``gym``
Hydra config and runs Gym training. ``--env`` is removed before Hydra parses overrides.

Examples::

    python -m mpail2.train.train --env push --headless log.no_wandb=True
    python -m mpail2.train.train --env Ant-v5 learner.replay_size=500000 log.no_wandb=True

Requires the package installed (``pip install -e .``) or ``PYTHONPATH`` at the repo root.
"""

from __future__ import annotations

import argparse
import sys

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import mpail2.train.envs  # noqa: F401 - register Hydra ConfigStore nodes
from mpail2.envs.isaac.launcher import parse_launcher_args
from mpail2.train.loops.isaac_image import run_isaac_image_training
from mpail2.train.registry import resolve_train_env


def _parse_selected_env(default_env: str = "push") -> str:
    """Parse and remove ``--env`` from ``sys.argv`` for both backends."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", type=str, default=default_env)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return args.env


def _compose_config(config_name: str, overrides: list[str]):
    """Compose a Hydra config with a clean GlobalHydra state."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def main(default_env: str = "push") -> None:
    env_token = _parse_selected_env(default_env=default_env)
    try:
        spec = resolve_train_env(env_token)
    except ValueError:
        import mpail2.train.configs.gym_train_config  # noqa: F401 - register ``gym`` ConfigStore node

        cfg = _compose_config(
            config_name="gym",
            overrides=sys.argv[1:] + [f"env_id={env_token}"],
        )

        from mpail2.train.gym_tasks.train_gym import run_gym_training

        run_gym_training(cfg)
        return

    if spec.suite == "isaac_image":
        launcher_args = parse_launcher_args("Train MPAIL2 on Isaac Franka image tasks.")
        cfg = _compose_config(config_name=spec.config_name, overrides=sys.argv[1:])

        run_isaac_image_training(cfg, launcher_args, spec)
        return

    raise ValueError(
        f"Registered env {env_token!r} has unsupported training suite {spec.suite!r}. "
        "Add a dispatcher branch in mpail2.train.train.main()."
    )


if __name__ == "__main__":
    main()
