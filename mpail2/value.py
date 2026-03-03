import torch
from typing import TYPE_CHECKING

from mpail2.utils import resolve_obj  # Accommodate both yaml and Configclass

if TYPE_CHECKING:
    from .configs.cfgs import EnsembleValueCfg


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
        self.num_q = cfg.num_q

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
