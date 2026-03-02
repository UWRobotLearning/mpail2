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
