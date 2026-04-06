import unittest

import mpail2.train.configs.gym_train_config  # noqa: F401 - ConfigStore
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from mpail2.train.configs.gym_train_config import GymHydraTrainConfig
from mpail2.train.gym_tasks.gym_assembly import (
    apply_gym_learner_overrides,
    default_gym_learner_cfg,
)
import mpail2.configs.defs as defs


class GymHydraConfigTest(unittest.TestCase):
    def tearDown(self):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

    def test_compose_default(self):
        with initialize(version_base=None):
            cfg = compose(config_name="gym")
        obj = OmegaConf.to_object(cfg)
        self.assertIsInstance(obj, GymHydraTrainConfig)
        self.assertEqual(obj.env_id, "Ant-v5")

    def test_cli_overrides(self):
        with initialize(version_base=None):
            cfg = compose(
                config_name="gym",
                overrides=[
                    "env_id=Hopper-v5",
                    "learner.replay_size=999",
                    "runner.num_learning_iterations=3",
                    "learner.planner_cfg.sampling_cfg.num_timesteps=11",
                ],
            )
        obj = OmegaConf.to_object(cfg)
        self.assertEqual(obj.env_id, "Hopper-v5")
        self.assertEqual(obj.learner["replay_size"], 999)
        self.assertEqual(obj.runner.num_learning_iterations, 3)
        self.assertEqual(obj.learner["planner_cfg"]["sampling_cfg"]["num_timesteps"], 11)

    def test_apply_learner_overrides(self):
        learner = default_gym_learner_cfg(state_dim=8, action_dim=4, use_terminations=True)
        overrides = {
            "replay_size": 111,
            "planner_cfg": {
                "latent_dim": 128,
                "sampling_cfg": {"num_rollouts": 100, "num_timesteps": 9},
                "opt_iters": 2,
            },
            "policy_learner_cfg": {"opt_params": {"lr": 1e-3}},
            "replay_batch_size": 32,
            "loss_horizon": 5,
        }
        apply_gym_learner_overrides(learner, overrides)
        self.assertEqual(learner.replay_size, 111)
        self.assertEqual(learner.planner_cfg.latent_dim, 128)
        self.assertEqual(learner.planner_cfg.sampling_cfg.num_rollouts, 100)
        self.assertEqual(learner.planner_cfg.sampling_cfg.num_timesteps, 9)
        self.assertEqual(learner.planner_cfg.opt_iters, 2)
        self.assertEqual(learner.policy_learner_cfg.opt_params["lr"], 1e-3)

    def test_unknown_override_key_raises(self):
        learner = default_gym_learner_cfg(state_dim=8, action_dim=4, use_terminations=True)
        with self.assertRaises(KeyError):
            apply_gym_learner_overrides(
                learner,
                {
                    "planner_cfg": {"not_a_real_field": 1},
                },
            )

    def test_default_gym_learner_cfg_returns_learner_cfg(self):
        learner = default_gym_learner_cfg(state_dim=8, action_dim=4, use_terminations=True)
        self.assertIsInstance(learner, defs.LearnerConfig)
        self.assertEqual(learner.planner_cfg.obs_dim, 8)
        self.assertEqual(learner.planner_cfg.action_dim, 4)
        self.assertTrue(learner.use_terminations)

    def test_gym_train_cfg_log_defaults(self):
        cfg = GymHydraTrainConfig()
        self.assertTrue(cfg.log.no_wandb)
        self.assertEqual(cfg.log.checkpoint_every, 50)


if __name__ == "__main__":
    unittest.main()
