import torch
from typing import TYPE_CHECKING

from mpail2.utils import resolve_obj # Accommodate both yaml and Configclass

if TYPE_CHECKING:
    from .configs.cfgs import RewardCfg, EnsembleValueCfg
    from .configs.cfgs import TDReturnCfg


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
        return self.model(_input)


class EnsembleValue(torch.nn.Module):
    """
    Ensemble of Q-networks that samples two at random during forward pass,
    similar to TD-MPC2 implementation.
    """

    class Q(torch.nn.Module):
        '''Q-network wrapper'''

        def __init__(self, model):
            super().__init__()
            self.model = model

        @torch.compile
        def forward(self, state, action):
            '''Returns Q-value for the given state-action pair'''
            _input = torch.cat([state, action], dim=-1)
            return self.model(_input).squeeze(-1)

    def __init__(
        self,
        cfg: 'EnsembleValueCfg',
        num_envs: int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.num_q = cfg.num_q if hasattr(cfg, 'num_q') else 2

        # Create ensemble of Q-networks
        self.cfg.model_kwargs["input_dim"] = cfg.state_dim + cfg.action_dim
        self.cfg.model_kwargs["output_dim"] = 1

        self.Qs = torch.nn.ModuleList([
            self.Q(resolve_obj(self.cfg.model_factory)(**self.cfg.model_kwargs))
            for _ in range(self.num_q)
        ]).to(device=device, dtype=dtype)

    def track_q_grad(self, enable: bool = True):
        for q_net in self.Qs:
            for param in q_net.model.parameters():
                param.requires_grad_(enable)

    def forward(self, state, action, return_type='min'):
        """
        Forward pass through ensemble.

        Args:
            state: State tensor
            action: Action tensor
            return_type: 'min' returns minimum of 2 sampled Q-values,
                        'avg' returns average of 2 sampled Q-values,
                        'all' returns all Q-values

        Returns:
            Q-value(s) based on return_type
        """
        assert return_type in {'min', 'avg', 'all'}

        # Compute all Q-values
        q_values = torch.stack([q_net(state, action) for q_net in self.Qs], dim=0)

        if return_type == 'all':
            return q_values

        # Sample two random Q-networks
        qidx = torch.randperm(self.num_q, device=q_values.device)[:2]
        sampled_qs = q_values[qidx]

        if return_type == 'min':
            return sampled_qs.min(0).values.squeeze(-1)
        else:  # avg
            return sampled_qs.mean(0).squeeze(-1)


class TDReturn(torch.nn.Module):
    '''Temporal Difference return function. Evaluates full trajectories by evaluating
    using a single step class and a terminal state class'''
    def __init__(
        self,
        cfg : 'TDReturnCfg',
        num_envs : int,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
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
        # stack s and s' for discriminator input
        states = rollouts[..., :-1, :] # last state for value
        next_states = rollouts[..., 1:, :]
        rewards[...] = self.reward(state=states, action=actions, next_state=next_states)

        # Evaluate terminal state rewards
        ts_states = states[..., -1, :] # rollouts is H+1
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
