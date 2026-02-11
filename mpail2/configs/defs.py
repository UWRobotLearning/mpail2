from typing import List, Tuple, Literal, Optional
from dataclasses import dataclass, field, MISSING
import mpail2.configs.cfgs as cfgs

'''
Default configurations for MPAIL.

The following settings MUST be set for each task:
1) encoder_cfg
2) action_dim
3) latent_dim

'''

#### SHARED CONSTANTS ####
OPT = "adam"
LR = 3e-4
HORIZON = 7
OPT_ITERS = 5
GAMMA = 0.99
LAM = 0.95

MODEL_KWARGS = {
    "hidden_dims": [512, 512],
    "activation": "silu",
    "use_layer_norm": True,
}
CNN_MODEL_KWARGS = {
    "out_channels": [32, 32, 32, 32],
    "kernel_sizes": [3, 3, 3, 3],
    "strides": [2, 2, 2, 1],
    "activation": "silu",
}
OPT_PARAMS = {"lr": LR}

# Add these settings:
# 1) ACTION_DIM
# 2) LATENT_DIM
# 3) Encoder Configs

##############################
### Planner Configurations ###
##############################

#
# Reward and Value
#

@dataclass(kw_only=True)
class RewardConfig(cfgs.RewardCfg):
    state_dim: int = None
    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
        "use_layer_norm": False,
        "disable_output_bias": True,
    })

@dataclass(kw_only=True)
class EnsembleValueConfig(cfgs.EnsembleValueCfg):
    state_dim: int = None
    action_dim: int = None
    num_q: int = 5
    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
    })

@dataclass(kw_only=True)
class TDReturnConfig(cfgs.TDReturnCfg):
    reward_cfg: RewardConfig = RewardConfig()
    value_cfg: EnsembleValueConfig = EnsembleValueConfig()
    gamma: float = GAMMA

#
# Encoders
#

@dataclass(kw_only=True)
class MLPCoderConfig(cfgs.MLPCoderCfg):

    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
        ## OVERRIDES ##
        "hidden_dims": [256, 256],
        "override_last_layer_norm": True,
    })

@dataclass(kw_only=True)
class CNNCoderConfig(cfgs.CNNCoderCfg):
    model_kwargs: dict = field(default_factory=lambda: {
        **CNN_MODEL_KWARGS,
        ## OVERRIDES ##
        "override_last_layer_activation": True,
    })

@dataclass(kw_only=True)
class MultiCoderConfig(cfgs.MultiCoderCfg):

    @dataclass(kw_only=True)
    class ProprioCoderConfig(cfgs.MLPCoderCfg):
        model_kwargs:dict = field(default_factory=lambda: {
            **MODEL_KWARGS,
            ## OVERRIDES ##
            "hidden_dims": [256],
            "use_layer_norm": False,
            "override_last_layer_norm": True,
            "override_last_layer_activation": True,
        })

    model_kwargs:dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
        ## OVERRIDES ##
        "override_last_layer_norm": True,
    })

# FOR VISUALIZATION PURPOSES ONLY
@dataclass(kw_only=True)
class MLPDecoderConfig(cfgs.MLPDecoderCfg):
    '''Configuration for MLP Decoder'''

    model_kwargs: dict = field(default_factory=lambda: {
        "activation": "silu",
        "hidden_dims": [256, 256],
        "use_layer_norm": False,
        "override_last_layer_norm": False,
    })

#
# Dynamics
#

@dataclass(kw_only=True)
class DynamicsConfig(cfgs.DynamicsCfg):

    x_dim: int = None
    '''agent state dimension'''

    action_dim: int = None
    '''action dimension'''

    latent_dim: int = None
    '''dimension of the latent space'''

    model_kwargs={
        **MODEL_KWARGS,
        "override_last_layer_norm": True,
    },

#
# Sampling/Policy
#

@dataclass(kw_only=True)
class PolicySamplingConfig(cfgs.PolicySamplingCfg):

    action_dim: int = None
    state_dim: int = None

    action_lims: list[Tuple[float, float]] | Tuple[float, float] = (-1., 1.)
    num_rollouts: int = 512
    num_timesteps: int = HORIZON
    policy_proportion: float = 0.05
    min_std: float = 0.05
    max_std: float = 2.0
    model_kwargs: dict = field(default_factory=lambda: {
        **MODEL_KWARGS,
    })

#
# Planner
#

@dataclass(kw_only=True)
class PlannerConfig(cfgs.PlannerCfg):

    # THESE MUST BE SET FOR EACH TASK
    encoder_cfg: cfgs.CoderCfg = MISSING
    action_dim: int = MISSING

    latent_dim: int = 512
    num_elites: int = 64
    temperature: float = 2.0
    opt_iters: int = OPT_ITERS

    reward_cfg: TDReturnConfig = TDReturnConfig()
    sampling_cfg: PolicySamplingConfig = PolicySamplingConfig()
    dynamics_cfg: DynamicsConfig = DynamicsConfig()

    seed: int = 42
    u_per_command: int = 1
    debug: bool = False

##############################
### Learner Configurations ###
##############################

@dataclass(kw_only=True)
class ValueLearnerConfig(cfgs.ValueLearnerCfg):

    opt: str = OPT

    opt_params: dict = field(default_factory=lambda: OPT_PARAMS)

    gamma: float = GAMMA

    lam: float = LAM

    max_grad_norm: float = 5.0

    polyak_tau: float = 0.01

@dataclass(kw_only=True)
class DiscLearnerConfig(cfgs.DiscLearnerCfg):

    opt: str = OPT

    opt_params: dict = field(default_factory=lambda: OPT_PARAMS)

    gp_coeff: float = 0.1

    gp_target_gradient: float = 1.0

@dataclass(kw_only=True)
class PolicyLearnerConfig(cfgs.PolicyLearnerCfg):

    opt: str = OPT

    opt_params: dict = field(default_factory=lambda: OPT_PARAMS)

    max_grad_norm: float = 1.0

    target_entropy: float = -3.0  # -ACTION_DIM

    alpha_lr: float = LR

    lam: float = LAM

@dataclass(kw_only=True)
class DynamicsLearnerConfig(cfgs.DynamicsLearnerCfg):

    opt: str = OPT

    opt_params: dict = field(default_factory=lambda: OPT_PARAMS)

    enc_lr_scale: float = 0.1

    rho: float = 0.95

    recon_coeff: float = 1.0 # Exact coefficient doesn't matter if not using recon loss

@dataclass(kw_only=True)
class LearnerConfig(cfgs.MPAIL2LearnerCfg):

    # Policy
    planner_cfg: cfgs.PlannerCfg = MISSING

    # Training parameters
    replay_ratio: float = 1.0

    # Replay buffer settings
    replay_size: int = 100_000
    replay_batch_size: int = 256
    use_stack_buffer: bool = True

    # Loss horizon for trajectory-based updates
    loss_horizon: int = HORIZON  # Match num_timesteps in sampling_cfg

    # Dynamics - ENABLED for training
    dynamics_learner_cfg: cfgs.DynamicsLearnerCfg = DynamicsLearnerConfig()

    # Reward (adversarial training)
    reward_learner_cfg: cfgs.DiscLearnerCfg = DiscLearnerConfig()

    # Value
    value_learner_cfg: cfgs.ValueLearnerCfg = ValueLearnerConfig()

    # Policy Learner
    policy_learner_cfg: cfgs.PolicyLearnerCfg = PolicyLearnerConfig()
