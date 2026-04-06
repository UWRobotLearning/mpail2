import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mpail2.train.envs as train_envs
from mpail2.train import train as train_entry


class TrainDispatchTest(unittest.TestCase):
    def _patch_hydra_runtime(self):
        # Train entry expects Hydra's global singleton + initialize() context.
        # Stub both so tests can exercise dispatch logic without real Hydra state.
        global_hydra_inst = SimpleNamespace(
            is_initialized=lambda: False,
            clear=lambda: None,
        )
        return patch.multiple(
            train_entry,
            initialize=lambda **kwargs: contextlib.nullcontext(),
            GlobalHydra=SimpleNamespace(instance=lambda: global_hydra_inst),
        )

    def _registered_isaac_env_tokens(self):
        # Read tokens from env module metadata so this test stays in sync as
        # aliases are added/removed.
        tokens = []
        for module in train_envs.discover_env_modules():
            env_name = getattr(module, "ENV_NAME", None)
            spec = getattr(module, "ENV_SPEC", None)
            if env_name is None or spec is None:
                continue
            tokens.extend([env_name, *tuple(getattr(module, "ENV_ALIASES", ()))])
        return tokens

    def test_routes_gym_when_env_not_registered_in_isaac(self):
        # If env resolution fails for Isaac registry, main() should route to Gym.
        with patch.object(train_entry, "sys", SimpleNamespace(argv=["prog", "--env", "Ant-v5", "log.no_wandb=True"])):
            with patch.object(train_entry, "resolve_train_env", side_effect=ValueError("unknown")):
                with patch.object(train_entry, "compose", return_value="gym_cfg") as compose_mock:
                    with self._patch_hydra_runtime():
                        with patch("mpail2.train.gym_tasks.train_gym.run_gym_training") as run_gym:
                            train_entry.main()

        run_gym.assert_called_once_with("gym_cfg")
        compose_mock.assert_called_once()
        self.assertEqual(
            # --env is consumed by argparse; remaining CLI args are passed through
            # and env_id is injected for Gym config composition.
            compose_mock.call_args.kwargs["overrides"],
            ["log.no_wandb=True", "env_id=Ant-v5"],
        )

    def test_routes_isaac_when_env_registered(self):
        # Registered Isaac envs should compose the env-specific Hydra config and
        # run Isaac training with parsed launcher args.
        spec = SimpleNamespace(config_name="push_image", suite="isaac_image")
        launcher_args = SimpleNamespace(headless=True)

        with patch.object(train_entry, "sys", SimpleNamespace(argv=["prog", "--env", "push", "log.no_wandb=True"])):
            with patch.object(train_entry, "resolve_train_env", return_value=spec):
                with patch.object(train_entry, "parse_launcher_args", return_value=launcher_args):
                    with patch.object(train_entry, "compose", return_value="isaac_cfg") as compose_mock:
                        with self._patch_hydra_runtime():
                            with patch.object(train_entry, "run_isaac_image_training") as run_isaac:
                                train_entry.main()

        compose_mock.assert_called_once_with(config_name="push_image", overrides=["log.no_wandb=True"])
        run_isaac.assert_called_once_with("isaac_cfg", launcher_args, spec)

    def test_registered_unknown_suite_raises(self):
        # Guardrail: unknown registered suites must fail loudly until a dispatcher
        # branch is added in train.main().
        spec = SimpleNamespace(config_name="real_push", suite="real")
        with patch.object(train_entry, "sys", SimpleNamespace(argv=["prog", "--env", "real_push"])):
            with patch.object(train_entry, "resolve_train_env", return_value=spec):
                with self.assertRaises(ValueError):
                    train_entry.main()

    def test_all_registered_isaac_env_tokens_route_to_isaac_training(self):
        # Smoke test dispatch coverage across every registered Isaac token
        # (canonical ENV_NAME + all aliases).
        launcher_args = SimpleNamespace(headless=True)
        for token in self._registered_isaac_env_tokens():
            with self.subTest(env_token=token):
                spec = train_entry.resolve_train_env(token)
                with patch.object(
                    train_entry,
                    "sys",
                    SimpleNamespace(argv=["prog", "--env", token, "log.no_wandb=True"]),
                ):
                    with patch.object(train_entry, "parse_launcher_args", return_value=launcher_args):
                        with patch.object(train_entry, "compose", return_value="isaac_cfg") as compose_mock:
                            with self._patch_hydra_runtime():
                                with patch.object(train_entry, "run_isaac_image_training") as run_isaac:
                                    train_entry.main()

                compose_mock.assert_called_once_with(
                    config_name=spec.config_name,
                    overrides=["log.no_wandb=True"],
                )
                run_isaac.assert_called_once_with("isaac_cfg", launcher_args, spec)


if __name__ == "__main__":
    unittest.main()
