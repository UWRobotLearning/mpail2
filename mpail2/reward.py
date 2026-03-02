import torch
import warnings
from typing import TYPE_CHECKING

from mpail2.utils import resolve_obj  # Accommodate both yaml and Configclass
from mpail2.value import EnsembleValue

if TYPE_CHECKING:
    from .configs.cfgs import RewardCfg


class Reward(torch.nn.Module):
    '''Adversarial Inverse Reinforcement Learning reward function. Evaluates state transitions
    using a learned discriminator to classify expert vs generator transitions'''

    def __init__(
        self,
        cfg : 'RewardCfg',
        num_envs : int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.cfg = cfg
        self.state_dim = cfg.state_dim

        self.model = resolve_obj(self.cfg.model_factory)(
            input_dim = cfg.state_dim * 2, # for s and s'
            output_dim = 1,
            **cfg.model_kwargs,
        ).to(device=device, dtype=dtype)

    @torch.compile
    def forward(self, state, next_state, action=None):
        '''state shape: (num_envs, state_dim)'''
        _input = torch.cat([state, next_state], dim=-1)
        return self.model(_input).squeeze(-1)


class TDReturn(torch.nn.Module):
    '''DEPRECATED: Temporal Difference return function.

    This class is deprecated. Rollout evaluation functionality has been moved to Planner.
    The Planner now directly holds Reward and EnsembleValue instances.
    '''
    def __init__(
        self,
        cfg,
        num_envs : int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        warnings.warn(
            "TDReturn is deprecated. Rollout evaluation has been moved to Planner. "
            "Use PlannerCfg.reward_cfg and PlannerCfg.value_cfg directly.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.cfg = cfg

        self.reward_cfg = self.cfg.reward_cfg
        self.reward: Reward = resolve_obj(self.reward_cfg.class_type)(
            self.reward_cfg, device=device, num_envs=num_envs
        )
        self.value_cfg = self.cfg.value_cfg
        self.value: EnsembleValue = resolve_obj(self.value_cfg.class_type)(
            self.value_cfg, device=device, num_envs=num_envs
        )

        # Check state dimensions
        assert self.reward.state_dim == self.value.state_dim, \
            f"Reward class state dim {self.reward.state_dim} does not match value \
            class state dim {self.value.state_dim}"

        self.state_dim = self.reward.state_dim

        if self.cfg.feature_inds is not None:
            assert len(self.cfg.feature_inds) == self.reward.state_dim, \
                f"Length of feature indices {self.cfg.feature_inds} do not match state dimension {self.reward.state_dim}"
            self.feature_inds = torch.tensor(self.cfg.feature_inds, device=device, dtype=torch.long)
        else:
            self.feature_inds = None

    def forward(self, rollouts: torch.Tensor, actions: torch.Tensor):
        '''rollouts shape: (num_envs, rollouts, horizon + 1, state_dim)'''

        rewards = torch.zeros_like(actions[..., 0]) # [num_envs, rollouts, horizon]

        # Evaluate single step rewards
        states = rollouts[..., :-1, :]
        next_states = rollouts[..., 1:, :]
        rewards[...] = self.reward(state=states, action=actions, next_state=next_states)

        # Evaluate terminal state rewards
        ts_states = states[..., -1, :]
        ts_actions = actions[..., -1, :]
        # Override last step reward with terminal state reward
        rewards[..., -1] = self.value(state=ts_states, action=ts_actions, return_type='avg')

        if self.cfg.gamma is not None:
            _gam_factors = self.cfg.gamma ** torch.arange(
                rewards.shape[-1], device=rollouts.device, dtype=rollouts.dtype
            )
            rewards *= _gam_factors

        return rewards

    def get_reward(self):
        return self.reward

    def get_value_function(self):
        return self.value
