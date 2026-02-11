import torch
import torch.nn as nn
from typing import Dict, Optional, Literal, Union


class CamOnlyObsNormalizer(nn.Module):
    """Normalizer that only applies to camera observations, passes through all others."""

    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__()
        self.device = device

    def forward(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(obs, dict):
            # Handle dict observations - only normalize camera tensors
            normalized_obs = {}
            for key, value in obs.items():
                if isinstance(value, torch.Tensor) and "cam" in key.lower():
                    # Camera observations: normalize to [-0.5, 0.5]
                    normalized_obs[key] = value / 255.0 - 0.5
                else:
                    # Pass through all other observations unchanged
                    normalized_obs[key] = value
            return normalized_obs
        else:
            # For single tensor, pass through unchanged
            return obs

    def inverse(self, normalized_obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(normalized_obs, dict):
            # Handle dict observations - only denormalize camera tensors
            denormalized_obs = {}
            for key, value in normalized_obs.items():
                if isinstance(value, torch.Tensor) and "cam" in key.lower():
                    # Camera observations: denormalize from [-0.5, 0.5]
                    denormalized_obs[key] = (value + 0.5) * 255.0
                else:
                    # Pass through all other observations unchanged
                    denormalized_obs[key] = value
            return denormalized_obs
        else:
            # For single tensor, pass through unchanged
            return normalized_obs


class ObsNormalizerFactory:

    @staticmethod
    def create_normalizer(
        normalization_type: Literal["none", "cam_only"],
        device: str = "cuda",
        **kwargs  # Accept but ignore extra params for backwards compatibility
    ) -> Optional[nn.Module]:
        if normalization_type == "none":
            return None
        elif normalization_type == "cam_only":
            return CamOnlyObsNormalizer(device=device)
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")