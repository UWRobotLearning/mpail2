"""Shared Isaac Sim launcher utilities for Hydra-based training scripts."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager


def parse_launcher_args(
    description: str = "Isaac training script",
) -> argparse.Namespace:
    """Parse CLI with IsaacLab AppLauncher args; forward remainder to ``sys.argv`` for Hydra."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=description)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli


@contextmanager
def isaac_session(
    args_cli: argparse.Namespace,
    enable_cameras: bool = True,
    log_level: str = "error",
):
    """Launch Isaac Sim with Kit log verbosity from config.

    Usage::

        launcher_args = parse_launcher_args()

        @hydra_main(config_name="...")
        def main(cfg):
            with isaac_session(launcher_args, log_level=cfg.log.log_level):
                ...
    """
    from isaaclab.app import AppLauncher

    args_cli.enable_cameras = enable_cameras

    launcher = AppLauncher(args_cli)
    try:
        yield launcher.app
    finally:
        if launcher.app is not None:
            launcher.app.close()
