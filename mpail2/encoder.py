"""Encoders and Decoders for MPAIL dynamics models."""

import torch
from typing import Dict, List, TYPE_CHECKING

from mpail2.utils import resolve_obj

if TYPE_CHECKING:
    from .configs.cfgs import CoderCfg, CNNCoderCfg, MultiCoderCfg


class Coder(torch.nn.Module):
    '''Base Encoder class'''

    def __init__(self, cfg: 'CoderCfg'):
        super(Coder, self).__init__()
        self.cfg = cfg
        self.obs_key = self.cfg.obs_key

        self.coder = resolve_obj(self.cfg.model_factory)(
            input_dim=self.cfg.input_dim,
            output_dim=self.cfg.output_dim,
            **self.cfg.model_kwargs
        )

    @torch.compile
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.coder(obs[self.obs_key])


class Decoder(Coder):
    '''Base Decoder class'''

    def __init__(self, cfg: 'CoderCfg'):
        super(Decoder, self).__init__(cfg)

    @torch.compile
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {self.obs_key: self.coder(latent)}


class CNNCoder(Coder):
    '''CNN Encoder'''

    cfg: 'CNNCoderCfg'

    def __init__(self, cfg: 'CNNCoderCfg'):
        super(Coder, self).__init__()
        self.cfg = cfg
        self.obs_key = self.cfg.obs_key

        self.coder = resolve_obj(self.cfg.model_factory)(
            H=self.cfg.H, W=self.cfg.W, C=self.cfg.C,
            output_dim=self.cfg.output_dim,
            **self.cfg.model_kwargs
        )

    @torch.compile
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''Handles generic batch shapes and IsaacLab's HWC standard:
        obs: [..., H, W, C]'''

        # reshape to (N, C, H, W) for torch Conv2d
        _obs = obs[self.obs_key]
        _obs_shape = _obs.shape
        H, W, C = _obs_shape[-3:]
        _obs_reshaped = _obs.reshape(-1, H, W, C).permute(0, 3, 1, 2)
        encoded = self.coder(_obs_reshaped)
        encoded_shape = encoded.shape[1:]
        return encoded.view(*_obs_shape[:-3], *encoded_shape)


class MultiCoder(torch.nn.Module):

    def __init__(self, cfg: 'MultiCoderCfg'):
        super(MultiCoder, self).__init__()

        self.cfg = cfg

        self.coders: List[Coder] = torch.nn.ModuleList(
            resolve_obj(coder_cfg.class_type)(
                cfg=coder_cfg,
            ) for coder_cfg in self.cfg.coder_list
        )

        self._embedding_dim = sum(
            coder_cfg.output_dim for coder_cfg in self.cfg.coder_list
        )

        self.latent_coder = resolve_obj(self.cfg.model_factory)(
            input_dim=self._embedding_dim,
            output_dim=self.cfg.output_dim,
            **self.cfg.model_kwargs,
        )

    @torch.compile
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for coder in self.coders:
            embeddings.append(coder(obs_dict))

        latent = self.latent_coder(torch.cat(embeddings, dim=-1))
        return latent
