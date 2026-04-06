"""Shared runtime helpers for train loops."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from mpail2.configs.cfgs import MPAIL2RunnerCfg


def ensure_wandb_settings(log_cfg: Any) -> None:
    """Fill wandb project/entity from env and validate when wandb is enabled."""
    if getattr(log_cfg, "no_wandb", False):
        return

    log_cfg.wandb_project = getattr(log_cfg, "wandb_project", None) or os.environ.get("WANDB_PROJECT")
    log_cfg.wandb_entity = getattr(log_cfg, "wandb_entity", None) or os.environ.get("WANDB_ENTITY")

    if not log_cfg.wandb_project or not log_cfg.wandb_entity:
        raise ValueError(
            "\033[33m"
            + """
WANDB_PROJECT and WANDB_ENTITY environment variables must be set when wandb logging is enabled.
Set them with:

conda env config vars set WANDB_PROJECT=<project>
conda env config vars set WANDB_ENTITY=<entity>

Or disable wandb with: log.no_wandb=True
            """
            + "\033[0m"
        )


def maybe_disable_wandb_if_unavailable(log_cfg: Any, *, wandb_available: bool) -> bool:
    """Return effective wandb usage, disabling it when the package is unavailable."""
    if getattr(log_cfg, "no_wandb", False):
        return False
    if wandb_available:
        return True
    print("[WARNING] wandb not available. Logging disabled. Install with: pip install wandb")
    log_cfg.no_wandb = True
    return False


def make_runner_log_cfg(log_cfg: Any, *, logger: str | None) -> MPAIL2RunnerCfg.LogCfg:
    """Convert backend-specific log config to ``MPAIL2RunnerCfg.LogCfg``."""
    return MPAIL2RunnerCfg.LogCfg(
        logger=logger,
        checkpoint_every=log_cfg.checkpoint_every,
        no_wandb=log_cfg.no_wandb,
        log_dir=log_cfg.run_log_dir,
        video_interval=log_cfg.video_interval,
    )


def maybe_init_wandb(*, log_cfg: Any, run_name: str, config_obj: Any, run_dir: str | None = None) -> None:
    """Initialize wandb if enabled."""
    if getattr(log_cfg, "no_wandb", False):
        return
    import wandb

    kwargs = {
        "project": log_cfg.wandb_project,
        "entity": log_cfg.wandb_entity,
        "name": run_name,
        "config": asdict(config_obj) if hasattr(config_obj, "__dataclass_fields__") else config_obj,
    }
    if run_dir:
        kwargs["dir"] = run_dir
    wandb.init(**kwargs)


def finish_wandb_if_running(*, log_cfg: Any) -> None:
    """Finish wandb run safely when enabled."""
    if getattr(log_cfg, "no_wandb", False):
        return
    import wandb

    if wandb.run is not None:
        wandb.finish()

