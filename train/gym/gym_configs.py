"""
Gymnasium-specific configurations for MPAIL2 algorithm.

This module provides configuration builders that create proper config objects
compatible with the mpail2 library, tailored for gymnasium environments.

Key differences from Isaac Lab configs:
- No image encoders (gymnasium MuJoCo envs use state observations)
- Simpler observation structure (single "obs" key)
- Direct state-based latent dynamics
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# Import MPAIL2 configs - use defs for defaults, cfgs for base types
from mpail2.configs.cfgs import MPAIL2RunnerCfg, ObsNormalizerCfg
from mpail2.configs.defs import (
    # Default configs with sensible values
    RewardConfig,
    EnsembleValueConfig,
    MLPCoderConfig,
    DynamicsConfig,
    PolicySamplingConfig,
    PlannerConfig,
    ValueLearnerConfig,
    RewardLearnerConfig,
    DynamicsLearnerConfig,
    PolicyLearnerConfig,
    LearnerConfig,
)


# =============================================================================
# Model Architecture Defaults for Gymnasium
# =============================================================================

DEFAULT_MODEL_KWARGS = {
    "hidden_dims": [256, 256],
    "activation": "silu",
    "use_layer_norm": True,
}

DEFAULT_LARGE_MODEL_KWARGS = {
    **DEFAULT_MODEL_KWARGS,
    "hidden_dims": [512, 512],
}


# =============================================================================
# Log Configuration for Gymnasium
# =============================================================================

@dataclass
class GymLogConfig:
    """Logging configuration for gymnasium training."""
    run_log_dir: str = "./logs"
    run_name: str = "gymnasium_run"
    wandb: bool = False
    no_log: bool = False
    log_every: int = 10
    video: bool = False
    video_interval: int = 5000
    video_length: Optional[int] = None  # None = use max_episode_length (full iteration)
    video_resolution: tuple = (1280, 720)
    video_crf: int = 30
    no_checkpoints: bool = False
    checkpoint_every: int = 100
    model_save_dirname: str = "models"
    wandb_entity: Optional[str] = None
    wandb_project: str = "gymnasium-mpail"
    save_state_data: bool = False
    save_state_data_every: int = 5
    dynamics_data: bool = False
    use_dynamics_absolute_path: bool = False
    test_mode: bool = False
    termination_when_unhealthy: bool = True

    # Required property for MPAIL2Runner
    @property
    def log_dir(self) -> str:
        return self.run_log_dir

    @property
    def model_save_path(self):
        return f"{self.run_log_dir}/{self.model_save_dirname}"


# =============================================================================
# MPAIL2 Configuration Builder for Gymnasium
# =============================================================================

def create_mpail_runner_cfg(
    state_dim: int,
    action_dim: int,
    latent_dim: int = 256,
    num_learning_iterations: int = 1000,
    path_to_demonstrations: str = "demonstrations.pt",
    num_demos: Optional[int] = None,
    seed: int = 42,
    logger: Optional[str] = "wandb",
    # Training params
    replay_size: int = 100_000,
    replay_batch_size: int = 256,
    loss_horizon: int = 7,
    replay_ratio: float = 1.0,
    use_terminations: bool = False,
    # MPPI planner params
    num_rollouts: int = 512,
    opt_iters: int = 5,
    opt: str = "adam",
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    # Model architecture
    model_kwargs: Optional[Dict] = None,
    # Optional log config (use GymLogConfig or similar)
    log_cfg: Optional['GymLogConfig'] = None,
) -> MPAIL2RunnerCfg:
    """
    Create MPAIL2 runner configuration for gymnasium environments.

    MPAIL2 uses model-based planning (MPPI) for imitation learning.

    Args:
        state_dim: Observation dimension
        action_dim: Action dimension
        latent_dim: Latent space dimension
        num_learning_iterations: Number of training iterations
        path_to_demonstrations: Path to demonstration file
        num_demos: Number of demonstrations to use
        seed: Random seed
        logger: Logger type
        replay_size: Size of replay buffer
        replay_batch_size: Batch size for training
        loss_horizon: Horizon for loss computation
        replay_ratio: Gradient updates per env step
        use_terminations: Whether to use terminal mask in value updates.
            When True, next state value is zeroed for terminal transitions.
        num_rollouts: Number of MPPI rollouts
        num_timesteps: Planning horizon for MPPI
        opt_iters: Number of MPPI optimization iterations
        lr: Learning rate
        gamma: Discount factor
        lam: GAE lambda
        model_kwargs: Model architecture settings

    Returns:
        Configured MPAIL2RunnerCfg
    """
    if model_kwargs is None:
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()

    # Encoder configuration (MLP for state-based observations)
    encoder_cfg = MLPCoderConfig(
        obs_key="obs",  # Match wrapper's dict key
        input_dim=state_dim,
        output_dim=latent_dim,
        model_kwargs={
            **model_kwargs,
            "override_last_layer_norm": True,
        }
    )

    # Dynamics configuration
    dynamics_cfg = DynamicsConfig(
        latent_dim=latent_dim,
        action_dim=action_dim,
        model_kwargs={
            **model_kwargs,
            "override_last_layer_norm": True,
        }
    )

    # Reward configuration (for AIRL-style reward)
    reward_cfg = RewardConfig(
        state_dim=latent_dim,
        model_kwargs={
            **model_kwargs,
            "use_layer_norm": False,
            "disable_output_bias": True,
        }
    )

    # Value function configuration
    value_cfg = EnsembleValueConfig(
        state_dim=latent_dim,
        action_dim=action_dim,
        model_kwargs=model_kwargs.copy(),
    )

    # Policy sampling configuration
    sampling_cfg = PolicySamplingConfig(
        action_dim=action_dim,
        state_dim=latent_dim,
        policy_proportion=0.1,
        action_lims=[(-1.0, 1.0)] * action_dim,
        num_rollouts=num_rollouts,
        num_timesteps=loss_horizon,
        model_kwargs=model_kwargs.copy(),
    )

    # Planner configuration
    planner_cfg = PlannerConfig(
        obs_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        opt_iters=opt_iters,
        encoder_cfg=encoder_cfg,
        dynamics_cfg=dynamics_cfg,
        reward_cfg=reward_cfg,
        value_cfg=value_cfg,
        sampling_cfg=sampling_cfg,
        gamma=gamma,
        decoder_cfg=None,
    )

    # Learner configurations
    reward_learner_cfg = RewardLearnerConfig(
        opt=opt,
        opt_params={"lr": lr},
    )

    value_learner_cfg = ValueLearnerConfig(
        opt=opt,
        opt_params={"lr": lr},
        gamma=gamma,
        lam=lam,
    )

    policy_learner_cfg = PolicyLearnerConfig(
        opt=opt,
        opt_params={"lr": lr},
        target_entropy=-action_dim,
        alpha_lr=lr,
        lam=lam,
    )

    dynamics_learner_cfg = DynamicsLearnerConfig(
        opt=opt,
        opt_params={"lr": lr},
    )

    obs_normalizer_cfg = ObsNormalizerCfg(
        normalization_type="fixed",
    )

    # MPAIL2 Learner configuration
    learner_cfg = LearnerConfig(
        planner_cfg=planner_cfg,
        reward_learner_cfg=reward_learner_cfg,
        value_learner_cfg=value_learner_cfg,
        policy_learner_cfg=policy_learner_cfg,
        dynamics_learner_cfg=dynamics_learner_cfg,
        obs_normalizer_cfg=obs_normalizer_cfg,
        replay_size=replay_size,
        replay_batch_size=replay_batch_size,
        loss_horizon=loss_horizon,
        replay_ratio=replay_ratio,
        use_terminations=use_terminations,
    )

    # Log configuration for runner - use provided or create default
    if log_cfg is None:
        runner_log_cfg = MPAIL2RunnerCfg.LogCfg(
            logger=logger,
            checkpoint_every=100,
            no_wandb=(logger != "wandb"),
            log_dir="./logs",
            video_interval=5000,
        )
    else:
        # Use provided GymLogConfig (it has compatible interface)
        runner_log_cfg = log_cfg

    # Runner configuration
    runner_cfg = MPAIL2RunnerCfg(
        learner_cfg=learner_cfg,
        num_learning_iterations=num_learning_iterations,
        path_to_demonstrations=path_to_demonstrations,
        seed=seed,
        logger=logger,
        log_cfg=runner_log_cfg,
        vis_rollouts=False,
    )

    return runner_cfg


__all__ = [
    "create_mpail_runner_cfg",
    "GymLogConfig",
    "DEFAULT_MODEL_KWARGS",
    "DEFAULT_LARGE_MODEL_KWARGS",
]
