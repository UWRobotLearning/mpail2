import unittest

import mpail2.train.envs  # noqa: F401 - register ConfigStore nodes
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from mpail2.train.utils.hydra_learner import learner_config_from_train_cfg


class HydraLearnerOverridesTest(unittest.TestCase):
    def tearDown(self):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

    def test_push_cli_overrides_apply_to_learner(self):
        with initialize(version_base=None):
            cfg = compose(
                config_name="push_image",
                overrides=[
                    "learner.replay_size=12345",
                    "learner.planner_cfg.latent_dim=321",
                ],
            )
        learner = learner_config_from_train_cfg(cfg)
        self.assertEqual(learner.replay_size, 12345)
        self.assertEqual(learner.planner_cfg.latent_dim, 321)

    def test_pick_place_default_preset_selected(self):
        with initialize(version_base=None):
            cfg = compose(config_name="pick_place_image")
        learner = learner_config_from_train_cfg(cfg)
        self.assertEqual(learner.planner_cfg.action_dim, 4)
        self.assertEqual(learner.replay_size, 50000)

    def test_unknown_override_key_raises(self):
        with initialize(version_base=None):
            cfg = compose(
                config_name="push_image",
                overrides=["+learner.not_a_real_field=1"],
            )
        with self.assertRaises(KeyError):
            learner_config_from_train_cfg(cfg)


if __name__ == "__main__":
    unittest.main()
