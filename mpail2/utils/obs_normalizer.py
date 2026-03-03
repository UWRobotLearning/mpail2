import torch
import torch.nn as nn
from typing import Dict, Optional, Literal, Union


class FixedObsNormalizer(nn.Module):
    """Normalizer that uses fixed statistics computed from expert demonstrations.

    Normalizes observations to zero mean and unit variance using statistics
    computed once at initialization from expert data.
    """

    def __init__(
        self,
        demonstrations: Dict[str, torch.Tensor],
        device: str = "cuda",
        eps: float = 1e-8,
        clip_obs: Optional[float] = 10.0,
        **kwargs
    ):
        """
        Args:
            demonstrations: Dictionary of expert demonstrations with shape {key: [N, 2, *obs_shape]}
                           where N is number of transitions and 2 is (obs, next_obs)
            device: Device to store statistics on
            eps: Small constant for numerical stability
            clip_obs: Optional clipping value for normalized observations
        """
        super().__init__()
        self.device = device
        self.eps = eps
        self.clip_obs = clip_obs

        # Compute and register statistics from demonstrations
        self._compute_statistics(demonstrations)

    def _compute_statistics(self, demonstrations: Dict[str, torch.Tensor]):
        """Compute mean and std from expert demonstrations."""
        self._obs_keys = list(demonstrations.keys())

        for key, data in demonstrations.items():
            # Skip camera observations - they have their own normalization
            if "cam" in key.lower():
                print(f"[FixedObsNormalizer] Skipping camera obs: {key}")
                continue

            # data shape: [N, 2, *obs_shape] - both obs and next_obs
            # Flatten to [N*2, *obs_shape] to compute statistics over all observations
            flat_data = data.reshape(-1, *data.shape[2:]).float()

            # Compute statistics along batch dimension
            mean = flat_data.mean(dim=0)
            std = flat_data.std(dim=0)

            # Register as buffers so they're saved/loaded with model and moved with .to()
            self.register_buffer(f"{key}_mean", mean.to(self.device))
            self.register_buffer(f"{key}_std", std.to(self.device))

            print(f"[FixedObsNormalizer] {key}: mean range [{mean.min():.3f}, {mean.max():.3f}], "
                  f"std range [{std.min():.3f}, {std.max():.3f}]")

    def forward(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(obs, dict):
            normalized_obs = {}
            for key, value in obs.items():
                if hasattr(self, f"{key}_mean"):
                    mean = getattr(self, f"{key}_mean")
                    std = getattr(self, f"{key}_std")
                    normalized_obs[key] = (value - mean) / (std + self.eps)
                    if self.clip_obs is not None:
                        normalized_obs[key] = torch.clamp(normalized_obs[key], min=-self.clip_obs, max=self.clip_obs)
                else:
                    # Pass through observations not in demo stats
                    normalized_obs[key] = value
            return normalized_obs
        else:
            # For single tensor, use first key's statistics (assumes single obs type)
            if len(self._obs_keys) > 0:
                key = self._obs_keys[0]
                mean = getattr(self, f"{key}_mean")
                std = getattr(self, f"{key}_std")
                return (obs - mean) / (std + self.eps)
            return obs

    def inverse(self, normalized_obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(normalized_obs, dict):
            denormalized_obs = {}
            for key, value in normalized_obs.items():
                if hasattr(self, f"{key}_mean"):
                    mean = getattr(self, f"{key}_mean")
                    std = getattr(self, f"{key}_std")
                    denormalized_obs[key] = value * (std + self.eps) + mean
                else:
                    denormalized_obs[key] = value
            return denormalized_obs
        else:
            if len(self._obs_keys) > 0:
                key = self._obs_keys[0]
                mean = getattr(self, f"{key}_mean")
                std = getattr(self, f"{key}_std")
                return normalized_obs * (std + self.eps) + mean
            return normalized_obs


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
        normalization_type: Literal["none", "cam_only", "fixed"],
        device: str = "cuda",
        demonstrations: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs  # Accept but ignore extra params for backwards compatibility
    ) -> Optional[nn.Module]:
        """
        Create an observation normalizer based on the specified type.

        Args:
            normalization_type: Type of normalization to apply
                - "none": No normalization
                - "cam_only": Only normalize camera observations to [-0.5, 0.5]
                - "fixed": Normalize using fixed statistics from expert demonstrations
            device: Device to create normalizer on
            demonstrations: Expert demonstrations dict (required for "fixed" type)
                           Shape: {key: [N, 2, *obs_shape]}

        Returns:
            Normalizer module or None
        """
        if normalization_type == "none":
            return None
        elif normalization_type == "cam_only":
            return CamOnlyObsNormalizer(device=device)
        elif normalization_type == "fixed":
            if demonstrations is None:
                raise ValueError(
                    "demonstrations must be provided for 'fixed' normalization type"
                )
            return FixedObsNormalizer(
                demonstrations=demonstrations,
                device=device,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")