import os
import unittest
from unittest.mock import patch

from mpail2.train.configs.gym_train_config import GymLogHydraCfg
from mpail2.train.configs.isaac_base_config import LogConfig
from mpail2.train.utils.runtime import (
    ensure_wandb_settings,
    make_runner_log_cfg,
    maybe_disable_wandb_if_unavailable,
)


class SharedTrainRuntimeTest(unittest.TestCase):
    def test_make_runner_log_cfg_works_for_isaac_and_gym(self):
        isaac_log = LogConfig(run_log_dir="/tmp/isaac", checkpoint_every=7, no_wandb=False, video_interval=11)
        gym_log = GymLogHydraCfg(
            run_log_dir="/tmp/gym",
            checkpoint_every=5,
            no_wandb=True,
            video_interval=13,
        )

        isaac_runner_log = make_runner_log_cfg(isaac_log, logger="wandb")
        gym_runner_log = make_runner_log_cfg(gym_log, logger=None)

        self.assertEqual(isaac_runner_log.log_dir, "/tmp/isaac")
        self.assertFalse(isaac_runner_log.no_wandb)
        self.assertEqual(isaac_runner_log.checkpoint_every, 7)
        self.assertEqual(isaac_runner_log.video_interval, 11)

        self.assertEqual(gym_runner_log.log_dir, "/tmp/gym")
        self.assertTrue(gym_runner_log.no_wandb)
        self.assertEqual(gym_runner_log.checkpoint_every, 5)
        self.assertEqual(gym_runner_log.video_interval, 13)

    def test_ensure_wandb_settings_fills_from_env(self):
        log_cfg = GymLogHydraCfg(no_wandb=False, wandb_entity=None, wandb_project=None)
        with patch.dict(os.environ, {"WANDB_PROJECT": "proj", "WANDB_ENTITY": "ent"}, clear=False):
            ensure_wandb_settings(log_cfg)
        self.assertEqual(log_cfg.wandb_project, "proj")
        self.assertEqual(log_cfg.wandb_entity, "ent")

    def test_ensure_wandb_settings_raises_when_enabled_and_missing(self):
        log_cfg = GymLogHydraCfg(no_wandb=False, wandb_entity=None, wandb_project=None)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                ensure_wandb_settings(log_cfg)

    def test_maybe_disable_wandb_if_unavailable(self):
        log_cfg = GymLogHydraCfg(no_wandb=False)
        use_wandb = maybe_disable_wandb_if_unavailable(log_cfg, wandb_available=False)
        self.assertFalse(use_wandb)
        self.assertTrue(log_cfg.no_wandb)


if __name__ == "__main__":
    unittest.main()
