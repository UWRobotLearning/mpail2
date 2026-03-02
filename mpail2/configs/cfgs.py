"""All configurations for MPAIL."""

import torch
import torch.nn as nn
from dataclasses import dataclass, field, MISSING
from typing import Literal, Callable, Any, Optional, Type, List, Tuple, TYPE_CHECKING

from mpail2.layers import mlp_factory, cnn_factory, identity_factory

# Import actual classes for default class_type values
from mpail2.encoder import Coder, Decoder, CNNCoder, MultiCoder
from mpail2.dynamics import Dynamics
from mpail2.sampling import PolicySampling
from mpail2.reward import Reward
from mpail2.value import EnsembleValue
from mpail2.learner import MPAIL2Learner

if TYPE_CHECKING:
    from ..utils.rollout_vis import RolloutsVisualization

###############################################################################
# ENCODER CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class CoderCfg:

    input_dim: int = None
    '''input dimension'''

    output_dim: int = None
    '''output dimension'''

    obs_key: str = None

    class_type: Type['Coder'] = Coder
    '''class type for the encoder/decoder model'''


@dataclass(kw_only=True)
class IdentityCoderCfg(CoderCfg):
    '''Configuration for Identity Encoder/Decoder that returns input as is'''

    model_factory: callable = identity_factory
    '''model factory that creates an identity model'''

    model_kwargs: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class MLPCoderCfg(CoderCfg):

    model_factory: callable = mlp_factory

    model_kwargs: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class MLPDecoderCfg(MLPCoderCfg):
    '''Configuration for MLP Decoder'''

    model_kwargs: dict = field(default_factory=dict)

    class_type: Type['Decoder'] = Decoder


@dataclass(kw_only=True)
class CNNCoderCfg(CoderCfg):

    H: int = None
    W: int = None
    C: int = None
    '''height and width of the input image'''

    model_kwargs: dict = field(default_factory=dict)

    model_factory: callable = cnn_factory

    class_type: Type = CNNCoder


@dataclass(kw_only=True)
class MultiCoderCfg(MLPCoderCfg):
    '''Configuration for MultiEncoder that handles both images and state'''

    output_dim: int = None

    coder_list: List[CoderCfg] = None
    '''Dictionary mapping observation keys to their respective Encoder configurations'''

    model_kwargs: dict = field(default_factory=dict)

    model_factory: callable = mlp_factory

    class_type: Type['MultiCoder'] = MultiCoder

    def __post_init__(self):
        # Propagate output_dim to individual coders if not set
        if self.coder_list is not None and self.output_dim is not None:
            for coder_cfg in self.coder_list:
                if coder_cfg.output_dim is None:
                    coder_cfg.output_dim = self.output_dim


###############################################################################
# DYNAMICS CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class DynamicsCfg:
    """Configuration for latent dynamics models (operates on encoded states)."""

    class_type: Type['Dynamics'] = Dynamics
    '''class type for the dynamics model'''

    latent_dim: int = None
    '''latent space dimension (input/output dimension)'''

    action_dim: int = None
    '''action dimension'''

    model_kwargs: dict = None

    model_factory: callable = mlp_factory

###############################################################################
# SAMPLING CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class SamplingCfg:
    """Base configuration for sampling strategies."""

    action_dim: int = None
    """ Control dimension """

    noise: list[float] | float = None
    """ Noise """

    action_lims: List[Tuple[float, float]] | Tuple[float, float] = None
    """ Max control values """

    num_rollouts: int = None
    """ Number of rollouts """

    num_timesteps: int = None
    """ Number of timesteps. Equal to Horizon minus 1 """


@dataclass(kw_only=True)
class PolicySamplingCfg(SamplingCfg):
    """Configuration for PolicySampling - extends base SamplingCfg"""

    action_dim: int = None

    state_dim: int = None

    class_type: Type['PolicySampling'] = PolicySampling

    model_factory: callable = mlp_factory

    model_kwargs: dict = None

    policy_proportion: float = None
    '''Fraction of controls from policy vs noise'''

    min_std: float = None
    '''Minimum standard deviation for policy action distribution'''

    max_std: float = None
    '''Maximum standard deviation for policy action distribution'''

    noise: None = None
    '''NOT USED. NO NEED TO SPECIFY. DEFINED FOR BASE CLASS'''


###############################################################################
# REWARD CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class RewardCfg:

    class_type: Type['Reward'] = Reward

    state_dim: int = None

    model_kwargs: dict = None

    model_factory: Callable[[Any], torch.nn.Sequential] = mlp_factory


@dataclass(kw_only=True)
class EnsembleValueCfg:
    """Configuration for ensemble value function with multiple Q-networks"""

    class_type: Type['EnsembleValue'] = EnsembleValue

    state_dim: int = None

    action_dim: int = None

    num_q: int = None  # Number of Q-networks in ensemble (default from TD-MPC2)

    model_kwargs: dict = None

    model_factory: Callable[[Any], torch.nn.Sequential] = mlp_factory

###############################################################################
# LEARNER CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class ValueLearnerCfg:

    opt: str = None
    '''Optimizer type'''

    opt_params: dict = None
    '''Optimizer parameters'''

    gamma: float = None
    '''Discount factor'''

    lam: float = None
    '''Lambda return parameter'''

    max_grad_norm: float = None
    '''Clips the gradient norm of the value function parameters to this value'''

    polyak_tau: float = None
    '''Polyak averaging coefficient: target = (1-tau) * target + tau * source'''


@dataclass(kw_only=True)
class RewardLearnerCfg:

    opt: str = None
    '''Optimizer type'''

    opt_params: dict = None
    '''Optimizer parameters'''

    gp_coeff: float = None
    '''Coefficient for gradient penalty term in WGAN loss'''

    gp_target_gradient: float = None
    '''Target gradient norm for gradient penalty'''

@dataclass(kw_only=True)
class DynamicsLearnerCfg:

    opt: str = None
    '''optimizer to use for training the dynamics model. Can be "adam", "sgd", etc.'''

    opt_params: dict = None
    '''Optimizer parameters for the dynamics model'''

    rho: float = None
    '''Temporal decay in JEP loss computation.'''

    recon_coeff: float = None
    '''Coefficient for the reconstruction loss term in the overall loss function. Only matters
    if reconstruction loss is used in the final loss. Otherwise decoder graph is detached from dynamics.'''

    enc_lr_scale: float = None
    '''Scaling factor for the encoder learning rate relative to the dynamics model learning rate.'''


@dataclass(kw_only=True)
class PolicyLearnerCfg:

    opt: str = None
    '''Optimizer type'''

    opt_params: dict = None
    '''Optimizer parameters'''

    max_grad_norm: float = None
    '''Clips the gradient norm of the policy parameters to this value'''

    target_entropy: float = None
    '''Target entropy for automatic tuning. Common choice: -dim(action_space), e.g., -2 for 2D actions'''

    alpha_lr: float = None
    '''Learning rate for entropy coefficient (alpha) optimization'''

    lam: float = None
    '''Lambda parameter for Model-based vs Model-free trade-off'''


@dataclass(kw_only=True)
class ObsNormalizerCfg:
    """Configuration for observation normalization"""

    normalization_type: Literal["none", "cam_only"] = None
    '''Type of normalization: none or cam_only'''

###############################################################################
# VISUALIZATION CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class RolloutVisConfig:

    class_type: Type['RolloutsVisualization'] = None

    vis_rollouts: bool = None

    vis_n_envs: int = None

    vis_n_rollouts: int = None

    xlim: tuple = None

    ylim: tuple = None

    show_velocity: bool = None

    show_elevation: bool = None

    cost_range: tuple = None

    show_trajectory_trace: bool = None

    # State index configuration for visualization
    # Indices for current position (x0)
    pos_x_idx: int = 0  # ee_x for x-axis of plot
    pos_y_idx: int = 1  # ee_y for y-axis of plot
    pos_z_idx: int = 2  # ee_z for z-axis of plot

    # Indices for cube/object trajectory rollouts (right plot)
    cube_x_idx: int = 7  # obj_x
    cube_y_idx: int = 8  # obj_y
    cube_z_idx: int = 9  # obj_z

    # Indices for velocity
    vel_x_idx: int = 6
    vel_y_idx: int = 7

    # Index for yaw
    yaw_idx: int = 5

    def __post_init__(self):
        # Import here to avoid circular imports
        from ..utils.rollout_vis import RolloutsVisualization
        if self.class_type is None:
            self.class_type = RolloutsVisualization

        self.control_scatter_1_indices: tuple = (self.pos_x_idx, self.pos_z_idx)
        self.control_scatter_cube_left_indices: tuple = (self.cube_x_idx, self.cube_z_idx)
        self.control_scatter_2_indices: tuple = (self.cube_x_idx, self.cube_y_idx)
        self.control_scatter_3_indices: tuple = (self.pos_x_idx, self.pos_y_idx)
        self.rollout_x_idx: int = self.pos_x_idx
        self.rollout_y_idx: int = self.pos_y_idx
        self.rollout_z_idx: int = self.pos_z_idx
        self.cube_pos_x_idx: int = self.cube_x_idx
        self.cube_pos_y_idx: int = self.cube_y_idx

###############################################################################
# PLANNER CONFIGURATION
###############################################################################

@dataclass(kw_only=True)
class PlannerCfg:
    """Configuration for MPAIL planner."""

    obs_dim: Optional[int] = None
    '''Propagates observation dimension to lower level modules'''

    action_dim: int = None
    '''Propagates action dimension to lower level modules'''

    latent_dim: int = None
    '''Propagates latent dimension to lower level modules'''

    temperature: float = None
    '''Temperature of optimization step'''

    opt_iters: int = None
    '''Number of optimization iterations per action'''

    u_per_command: int = None
    '''Number of control commands per action'''

    num_elites: int = None
    '''Number of elite samples for CEM-style optimization'''

    vis_cfg: RolloutVisConfig = None
    '''Visualization configuration. None disables visualization'''

    encoder_cfg: CoderCfg = None
    '''Configuration for the encoder (obs -> latent)'''

    decoder_cfg: CoderCfg = None
    '''Configuration for the decoder (latent -> obs), optional'''

    reward_cfg: RewardCfg = None
    '''Configuration for reward function'''

    value_cfg: EnsembleValueCfg = None
    '''Configuration for value function'''

    gamma: Optional[float] = None
    '''Discount factor for reward computation during planning'''

    sampling_cfg: PolicySamplingCfg = None
    '''Configuration for sampling module'''

    dynamics_cfg: DynamicsCfg = None
    '''Configuration for dynamics model'''

    def __post_init__(self):
        # Set default encoder config if not provided
        if self.encoder_cfg is None:
            self.encoder_cfg = MLPCoderCfg()

        # If obs_dim, action_dim, latent_dim are provided, propagate to lower level configs
        if self.obs_dim is not None:
            self.encoder_cfg.input_dim = self.obs_dim  # Encoder input
            if self.decoder_cfg is not None:
                self.decoder_cfg.output_dim = self.obs_dim  # Decoder output
        if self.action_dim is not None:
            if self.value_cfg is not None:
                self.value_cfg.action_dim = self.action_dim  # Value
            self.dynamics_cfg.action_dim = self.action_dim  # Dynamics
            self.sampling_cfg.action_dim = self.action_dim  # Policy
        if self.latent_dim is not None:
            self.sampling_cfg.state_dim = self.latent_dim  # Policy
            if self.reward_cfg is not None:
                self.reward_cfg.state_dim = self.latent_dim  # Reward
            if self.value_cfg is not None:
                self.value_cfg.state_dim = self.latent_dim  # Value
            self.dynamics_cfg.latent_dim = self.latent_dim  # Dynamics
            self.encoder_cfg.output_dim = self.latent_dim  # Encoder output
            if hasattr(self.encoder_cfg, '__post_init__'):
                self.encoder_cfg.__post_init__()  # Run encoder propagation if needed
            if self.decoder_cfg is not None:
                self.decoder_cfg.input_dim = self.latent_dim  # Decoder input


###############################################################################
# TOP-LEVEL CONFIGURATIONS
###############################################################################

@dataclass(kw_only=True)
class MPAIL2LearnerCfg:

    class_type: Type['MPAIL2Learner'] = MPAIL2Learner
    '''Type of the MPAIL learner class'''

    replay_ratio: float = None
    '''Number of gradient updates per new environment step collected'''

    replay_size: int = None
    '''Size of the replay buffer'''

    replay_batch_size: int = None
    '''Batch size for replay buffer sampling'''

    loss_horizon: int = None
    '''Number of steps to consider for loss computation'''

    use_terminations: bool = None
    '''Whether to use terminal mask in value function updates.
    When True, the next state value is set to 0 for terminal transitions,
    implementing the standard RL (1 - done) * V(s') term.'''

    # Planner
    planner_cfg: PlannerCfg = None
    '''Configuration for MPPI module'''

    # Dynamics
    dynamics_learner_cfg: DynamicsLearnerCfg = None

    # Reward
    reward_learner_cfg: RewardLearnerCfg = None
    '''Configuration for reward learning algorithm (WGAN-style adversarial training)'''

    # Value
    value_learner_cfg: ValueLearnerCfg = None
    '''Configuration for value function'''

    # Policy
    policy_learner_cfg: PolicyLearnerCfg = None

    # Observation Normalizer
    obs_normalizer_cfg: ObsNormalizerCfg = None


@dataclass(kw_only=True)
class MPAIL2RunnerCfg:

    path_to_demonstrations: str = None
    '''Number of steps per environment'''

    learner_cfg: MPAIL2LearnerCfg = None
    '''Configuration for MPAIL learner'''

    num_learning_iterations: int = None
    '''Number of learning iterations'''

    logger: Optional[Literal["wandb"]] = None
    '''Logger type'''

    @dataclass(kw_only=True)
    class LogCfg:

        logger: Optional[Literal["wandb"]] = None
        '''Logger type'''

        checkpoint_every: int = None
        '''Frequency of saving model checkpoints during training (in learning iterations)'''

        no_wandb: bool = False
        '''Whether to disable Weights & Biases logging even if logger is set to "wandb"'''

        log_dir: str = None
        '''Directory to save logs and checkpoints'''

        video_interval: int = None
        '''Frequency of logging rollout videos during training (in learning iterations)'''

    log_cfg: LogCfg = None

    seed: int = None
    '''Random seed'''

    vis_rollouts: bool = False
    '''Whether to visualize rollouts during training'''
