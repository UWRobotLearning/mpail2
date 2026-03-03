"""
Data utilities for expert demonstration format conversion and handling.

This module supports two data formats:

1. **Universal Expert Dataset Format** (recommended for new data):
   - Flat structure with shape (num_transitions, feature_dim)
   - Contains: obs, next_obs, actions, rewards, dones, terminated, truncated
   - Suitable for behavior cloning, RL, and imitation learning

2. **MPAIL Format** (legacy, for backward compatibility):
   - Dictionary with shape {key: tensor[N, 2, *obs_shape]}
   - N = number of transitions, 2 = (obs, next_obs) pairs
   - Only contains observation pairs

When loading data for MPAIL/IRL baselines, the universal format is automatically
converted to MPAIL format.

Example for Ant-v5 (Universal Format):
    {
        "obs": tensor[10000, 27],
        "next_obs": tensor[10000, 27],
        "actions": tensor[10000, 8],
        "rewards": tensor[10000],
        "dones": tensor[10000],
        "terminated": tensor[10000],
        "truncated": tensor[10000],
    }

Example for Ant-v5 (MPAIL Format):
    {"obs": tensor[10000, 2, 27]}  # 10000 transitions, 27-dim observations
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def create_transition_pairs(
    observations: Union[List[np.ndarray], np.ndarray, torch.Tensor],
    episode_boundaries: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Create (obs, next_obs) pairs from a sequence of observations.
    
    This handles both single-episode and multi-episode data correctly by
    not creating transitions across episode boundaries.
    
    Args:
        observations: Observations, shape [T, *obs_shape] where T is total timesteps
        episode_boundaries: List of indices where episodes end (exclusive).
                          If None, treats all observations as one episode.
    
    Returns:
        Tensor of shape [N, 2, *obs_shape] where N is number of valid transitions
    
    Example:
        >>> obs = np.random.randn(1000, 27)  # 1000 timesteps, 27-dim obs
        >>> pairs = create_transition_pairs(obs)  # Shape: [999, 2, 27]
        
        >>> # With episode boundaries at 100, 250, 1000
        >>> pairs = create_transition_pairs(obs, episode_boundaries=[100, 250, 1000])
        >>> # Shape: [997, 2, 27] (3 fewer due to episode boundaries)
    """
    # Convert to numpy if tensor
    if isinstance(observations, torch.Tensor):
        observations = observations.numpy()
    elif isinstance(observations, list):
        observations = np.array(observations)
    
    # If no boundaries provided, treat as single episode
    if episode_boundaries is None:
        episode_boundaries = [len(observations)]
    
    pairs = []
    prev_boundary = 0
    
    for boundary in episode_boundaries:
        # Get observations for this episode
        ep_obs = observations[prev_boundary:boundary]
        
        if len(ep_obs) > 1:
            # Create (obs, next_obs) pairs within episode
            obs = ep_obs[:-1]  # All but last
            next_obs = ep_obs[1:]  # All but first
            
            # Stack to [num_transitions, 2, *obs_shape]
            ep_pairs = np.stack([obs, next_obs], axis=1)
            pairs.append(ep_pairs)
        
        prev_boundary = boundary
    
    if not pairs:
        raise ValueError("No valid transitions found. Check that observations have at least 2 timesteps.")
    
    # Concatenate all episodes
    all_pairs = np.concatenate(pairs, axis=0)
    
    return torch.tensor(all_pairs, dtype=torch.float32)


def package_mpail_demos(
    observations: Union[List[np.ndarray], np.ndarray, torch.Tensor],
    episode_boundaries: Optional[List[int]] = None,
    obs_key: str = "obs",
) -> Dict[str, torch.Tensor]:
    """
    Package observations into MPAIL demonstration format.
    
    Args:
        observations: Observations array/tensor of shape [T, *obs_shape]
        episode_boundaries: Optional list of episode end indices
        obs_key: Key to use in the output dictionary (default: "obs")
    
    Returns:
        Dictionary {"obs": tensor[N, 2, *obs_shape]} in MPAIL format
    
    Example:
        >>> obs = np.random.randn(1000, 27)  # 1000 timesteps, 27-dim
        >>> demos = package_mpail_demos(obs)
        >>> print(demos["obs"].shape)  # torch.Size([999, 2, 27])
    """
    pairs = create_transition_pairs(observations, episode_boundaries)
    return {obs_key: pairs}


def package_mpail_demos_multimodal(
    observation_dict: Dict[str, Union[List[np.ndarray], np.ndarray, torch.Tensor]],
    episode_boundaries: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Package multi-modal observations into MPAIL demonstration format.
    
    Use this when you have multiple observation modalities (e.g., state + images).
    
    Args:
        observation_dict: Dictionary mapping modality names to observations
                         Each value should have shape [T, *obs_shape]
        episode_boundaries: Optional list of episode end indices (same for all modalities)
    
    Returns:
        Dictionary {modality: tensor[N, 2, *obs_shape]} for each modality
    
    Example:
        >>> obs_dict = {
        ...     "obs": np.random.randn(1000, 27),         # State
        ...     "camera": np.random.randn(1000, 64, 64),  # Images
        ... }
        >>> demos = package_mpail_demos_multimodal(obs_dict)
        >>> print(demos["obs"].shape)  # torch.Size([999, 2, 27])
        >>> print(demos["camera"].shape)  # torch.Size([999, 2, 64, 64])
    """
    result = {}
    for key, observations in observation_dict.items():
        pairs = create_transition_pairs(observations, episode_boundaries)
        result[key] = pairs
    
    return result


def save_demonstrations(
    demonstrations: Dict[str, torch.Tensor],
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save demonstrations to a .pt file.
    
    Args:
        demonstrations: Dictionary of demonstration tensors
        path: Output file path (should end with .pt)
        metadata: Optional metadata to save alongside demonstrations
    
    Example:
        >>> demos = {"obs": torch.randn(1000, 2, 27)}
        >>> save_demonstrations(demos, "expert_demos.pt", metadata={"env": "Ant-v5"})
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # Validate format
    for key, tensor in demonstrations.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Value for key '{key}' must be a torch.Tensor, got {type(tensor)}")
        if tensor.dim() < 3 or tensor.shape[1] != 2:
            raise ValueError(
                f"Tensor for key '{key}' must have shape [N, 2, *obs_shape], got {tensor.shape}"
            )
    
    # Save with optional metadata
    if metadata is not None:
        save_dict = {"demonstrations": demonstrations, "metadata": metadata}
    else:
        save_dict = demonstrations
    
    torch.save(save_dict, path)
    
    # Print summary
    print(f"[INFO] Saved demonstrations to {path}")
    for key, tensor in demonstrations.items():
        print(f"  {key}: {tuple(tensor.shape)} ({tensor.numel() * 4 / 1e6:.2f} MB)")


def load_demonstrations(
    path: str,
    device: str = "cpu",
    num_demos: Optional[int] = None,
    obs_key: str = "obs",
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
    """
    Load demonstrations from a .pt file and return in MPAIL format.
    
    This function handles multiple formats with auto-detection:
    1. Universal format: Automatically converted to MPAIL format
    2. Dictionary of tensors: {"obs": tensor[N, 2, obs_dim]}
    3. Dictionary with metadata: {"demonstrations": {...}, "metadata": {...}}
    4. Raw tensor: tensor[N, 2, obs_dim] (wrapped as {"obs": tensor})
    
    Args:
        path: Path to the .pt file
        device: Device to load tensors to
        num_demos: Optional maximum number of demonstrations to load
        obs_key: Key to use for observations in output (default: "obs")
    
    Returns:
        Tuple of (demonstrations_dict, metadata_dict or None)
        demonstrations_dict is always in MPAIL format: {key: tensor[N, 2, *obs_shape]}
    
    Example:
        >>> demos, metadata = load_demonstrations("expert_demos.pt")
        >>> print(demos["obs"].shape)  # torch.Size([999, 2, 27])
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Demonstration file not found: {path}")
    
    data = torch.load(path, map_location=device, weights_only=False)
    
    # Auto-detect format and handle accordingly
    metadata = None
    
    # Check if universal format (new)
    if is_universal_format(data):
        print(f"[INFO] Detected universal format, converting to MPAIL format...")
        metadata = data.get("metadata")
        
        # Extract obs and next_obs
        obs = data["obs"].to(device=device)
        next_obs = data["next_obs"].to(device=device)
        
        if num_demos is not None:
            obs = obs[:num_demos]
            next_obs = next_obs[:num_demos]
        
        # Convert to MPAIL format
        demonstrations = convert_to_mpail_format(
            {"obs": obs, "next_obs": next_obs},
            obs_key=obs_key,
        )
        return demonstrations, metadata
    
    # Handle MPAIL/legacy formats
    if isinstance(data, dict):
        if "demonstrations" in data:
            # Format: {"demonstrations": {...}, "metadata": {...}}
            demonstrations = data["demonstrations"]
            metadata = data.get("metadata")
        elif all(isinstance(v, torch.Tensor) for v in data.values()):
            # Format: {"obs": tensor, ...}
            demonstrations = data
        else:
            # Assume it's demonstrations with some metadata mixed in
            demonstrations = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
            metadata = {k: v for k, v in data.items() if not isinstance(v, torch.Tensor)}
    elif isinstance(data, torch.Tensor):
        # Raw tensor format
        demonstrations = {"obs": data}
    else:
        raise ValueError(f"Unknown demonstration format: {type(data)}")
    
    # Move to device and optionally limit number of demos
    result = {}
    for key, tensor in demonstrations.items():
        tensor = tensor.to(device=device)
        if num_demos is not None:
            tensor = tensor[:num_demos]
        result[key] = tensor
    
    # Validate MPAIL format
    for key, tensor in result.items():
        if tensor.dim() < 3 or tensor.shape[1] != 2:
            raise ValueError(
                f"Invalid demonstration format for key '{key}': "
                f"expected shape [N, 2, *obs_shape], got {tensor.shape}"
            )
    
    return result, metadata


def merge_demonstrations(
    demo_paths: List[str],
    output_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Merge multiple demonstration files into one.
    
    Args:
        demo_paths: List of paths to demonstration files
        output_path: Optional path to save merged demonstrations
    
    Returns:
        Merged demonstrations dictionary
    
    Example:
        >>> merged = merge_demonstrations([
        ...     "demos_episode_1_100.pt",
        ...     "demos_episode_101_200.pt",
        ... ], output_path="demos_merged.pt")
    """
    all_demos = {}
    
    for path in demo_paths:
        demos, _ = load_demonstrations(path)
        
        for key, tensor in demos.items():
            if key not in all_demos:
                all_demos[key] = [tensor]
            else:
                all_demos[key].append(tensor)
    
    # Concatenate along the first (transition) dimension
    merged = {key: torch.cat(tensors, dim=0) for key, tensors in all_demos.items()}
    
    if output_path is not None:
        save_demonstrations(merged, output_path)
    
    return merged


def get_demo_stats(demonstrations: Dict[str, torch.Tensor]) -> Dict:
    """
    Compute statistics for demonstration data in MPAIL format.
    
    Args:
        demonstrations: Demonstration dictionary in MPAIL format [N, 2, *obs_shape]
    
    Returns:
        Dictionary with statistics for each modality
    """
    stats = {}
    
    for key, tensor in demonstrations.items():
        # tensor shape: [N, 2, *obs_shape]
        obs = tensor[:, 0]  # Current observations
        next_obs = tensor[:, 1]  # Next observations
        
        stats[key] = {
            "num_transitions": tensor.shape[0],
            "obs_shape": tuple(tensor.shape[2:]),
            "obs_mean": obs.mean(dim=0).tolist() if obs.numel() < 1000 else obs.mean().item(),
            "obs_std": obs.std(dim=0).tolist() if obs.numel() < 1000 else obs.std().item(),
            "obs_min": obs.min().item(),
            "obs_max": obs.max().item(),
            "delta_mean": (next_obs - obs).mean().item(),
            "delta_std": (next_obs - obs).std().item(),
        }
    
    return stats


# =============================================================================
# Universal Expert Dataset Format (New)
# =============================================================================

def save_expert_dataset(
    obs: Union[np.ndarray, torch.Tensor],
    next_obs: Union[np.ndarray, torch.Tensor],
    actions: Union[np.ndarray, torch.Tensor],
    rewards: Union[np.ndarray, torch.Tensor],
    dones: Union[np.ndarray, torch.Tensor],
    terminated: Union[np.ndarray, torch.Tensor],
    truncated: Union[np.ndarray, torch.Tensor],
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save expert dataset in the universal flattened format.
    
    This format stores all transition data needed for behavior cloning, RL,
    and imitation learning algorithms.
    
    Args:
        obs: Current observations, shape [N, obs_dim]
        next_obs: Next observations, shape [N, obs_dim]
        actions: Expert actions, shape [N, action_dim]
        rewards: Step rewards, shape [N]
        dones: Episode done flags (terminated | truncated), shape [N]
        terminated: True termination flags, shape [N]
        truncated: Truncation flags (e.g., time limit), shape [N]
        path: Output file path (should end with .pt)
        metadata: Optional metadata dictionary
    
    Example:
        >>> save_expert_dataset(
        ...     obs=obs_array,
        ...     next_obs=next_obs_array,
        ...     actions=actions_array,
        ...     rewards=rewards_array,
        ...     dones=dones_array,
        ...     terminated=terminated_array,
        ...     truncated=truncated_array,
        ...     path="expert_data.pt",
        ...     metadata={"env": "Ant-v5", "num_episodes": 100}
        ... )
    """
    # Convert to tensors if needed
    def to_tensor(x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype)
        elif isinstance(x, torch.Tensor):
            return x.to(dtype)
        else:
            return torch.tensor(x, dtype=dtype)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # Package data
    dataset = {
        "format": "universal_v1",  # Format identifier for auto-detection
        "obs": to_tensor(obs),
        "next_obs": to_tensor(next_obs),
        "actions": to_tensor(actions),
        "rewards": to_tensor(rewards).squeeze(),
        "dones": to_tensor(dones, dtype=torch.bool).squeeze(),
        "terminated": to_tensor(terminated, dtype=torch.bool).squeeze(),
        "truncated": to_tensor(truncated, dtype=torch.bool).squeeze(),
    }
    
    # Add metadata if provided
    if metadata is not None:
        dataset["metadata"] = metadata
    
    # Validate shapes
    n_transitions = dataset["obs"].shape[0]
    assert dataset["next_obs"].shape[0] == n_transitions, "next_obs length mismatch"
    assert dataset["actions"].shape[0] == n_transitions, "actions length mismatch"
    assert dataset["rewards"].shape[0] == n_transitions, "rewards length mismatch"
    assert dataset["dones"].shape[0] == n_transitions, "dones length mismatch"
    assert dataset["terminated"].shape[0] == n_transitions, "terminated length mismatch"
    assert dataset["truncated"].shape[0] == n_transitions, "truncated length mismatch"
    
    torch.save(dataset, path)
    
    # Print summary
    print(f"[INFO] Saved expert dataset to {path}")
    print(f"  Transitions: {n_transitions}")
    print(f"  obs: {tuple(dataset['obs'].shape)}")
    print(f"  actions: {tuple(dataset['actions'].shape)}")
    print(f"  Total size: {sum(v.numel() * 4 for v in dataset.values() if isinstance(v, torch.Tensor)) / 1e6:.2f} MB")


def load_expert_dataset(
    path: str,
    device: str = "cpu",
    num_demos: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
    """
    Load expert dataset from a .pt file in universal format.
    
    Args:
        path: Path to the .pt file
        device: Device to load tensors to
        num_demos: Optional maximum number of transitions to load
    
    Returns:
        Tuple of (dataset_dict, metadata_dict or None)
        dataset_dict contains: obs, next_obs, actions, rewards, dones, terminated, truncated
    
    Example:
        >>> dataset, metadata = load_expert_dataset("expert_data.pt")
        >>> print(dataset["obs"].shape)  # torch.Size([10000, 27])
        >>> print(dataset["actions"].shape)  # torch.Size([10000, 8])
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    data = torch.load(path, map_location=device, weights_only=False)
    
    # Check format
    if not isinstance(data, dict) or data.get("format") != "universal_v1":
        raise ValueError(
            f"File is not in universal format. "
            f"Use load_demonstrations() for MPAIL format files."
        )
    
    metadata = data.get("metadata")
    
    # Extract dataset fields
    fields = ["obs", "next_obs", "actions", "rewards", "dones", "terminated", "truncated"]
    dataset = {}
    
    for field in fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in dataset")
        tensor = data[field].to(device=device)
        if num_demos is not None:
            tensor = tensor[:num_demos]
        dataset[field] = tensor
    
    return dataset, metadata


def convert_to_mpail_format(
    dataset: Dict[str, torch.Tensor],
    obs_key: str = "obs",
) -> Dict[str, torch.Tensor]:
    """
    Convert universal dataset format to MPAIL format.
    
    Takes flat obs and next_obs arrays and stacks them into MPAIL's
    [N, 2, *obs_shape] format.
    
    Args:
        dataset: Dictionary with "obs" and "next_obs" keys
        obs_key: Key to use in output dictionary (default: "obs")
    
    Returns:
        Dictionary {obs_key: tensor[N, 2, *obs_shape]} in MPAIL format
    
    Example:
        >>> dataset = {"obs": torch.randn(1000, 27), "next_obs": torch.randn(1000, 27)}
        >>> mpail_demos = convert_to_mpail_format(dataset)
        >>> print(mpail_demos["obs"].shape)  # torch.Size([1000, 2, 27])
    """
    obs = dataset["obs"]
    next_obs = dataset["next_obs"]
    
    # Stack to [N, 2, *obs_shape]
    paired = torch.stack([obs, next_obs], dim=1)
    
    return {obs_key: paired}


def is_universal_format(data: Union[Dict, torch.Tensor]) -> bool:
    """
    Check if loaded data is in universal format.
    
    Args:
        data: Loaded data from torch.load()
    
    Returns:
        True if data is in universal format, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    # Check for format identifier
    if data.get("format") == "universal_v1":
        return True
    
    # Check for universal format fields (without explicit format tag)
    universal_fields = {"obs", "next_obs", "actions", "rewards", "dones"}
    if universal_fields.issubset(data.keys()):
        # Additional check: obs should be 2D [N, obs_dim], not 3D [N, 2, obs_dim]
        obs = data.get("obs")
        if isinstance(obs, torch.Tensor) and obs.dim() == 2:
            return True
    
    return False


def is_mpail_format(data: Union[Dict, torch.Tensor]) -> bool:
    """
    Check if loaded data is in MPAIL format.
    
    Args:
        data: Loaded data from torch.load()
    
    Returns:
        True if data is in MPAIL format, False otherwise
    """
    if isinstance(data, torch.Tensor):
        # Raw tensor: check if shape is [N, 2, *obs_shape]
        return data.dim() >= 3 and data.shape[1] == 2
    
    if not isinstance(data, dict):
        return False
    
    # Check for "demonstrations" key (wrapped format)
    if "demonstrations" in data:
        demos = data["demonstrations"]
        if isinstance(demos, dict):
            for v in demos.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == 2:
                    return True
        return False
    
    # Check tensor values for MPAIL shape [N, 2, *obs_shape]
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dim() >= 3 and v.shape[1] == 2:
                return True
    
    return False


def get_dataset_stats(dataset: Dict[str, torch.Tensor]) -> Dict:
    """
    Compute statistics for expert dataset in universal format.
    
    Args:
        dataset: Dataset dictionary with obs, actions, rewards, etc.
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "num_transitions": dataset["obs"].shape[0],
        "obs_shape": tuple(dataset["obs"].shape[1:]),
        "action_shape": tuple(dataset["actions"].shape[1:]),
        "obs_mean": dataset["obs"].mean().item(),
        "obs_std": dataset["obs"].std().item(),
        "obs_min": dataset["obs"].min().item(),
        "obs_max": dataset["obs"].max().item(),
        "action_mean": dataset["actions"].mean().item(),
        "action_std": dataset["actions"].std().item(),
        "action_min": dataset["actions"].min().item(),
        "action_max": dataset["actions"].max().item(),
        "reward_mean": dataset["rewards"].mean().item(),
        "reward_std": dataset["rewards"].std().item(),
        "reward_min": dataset["rewards"].min().item(),
        "reward_max": dataset["rewards"].max().item(),
        "num_dones": dataset["dones"].sum().item(),
        "num_terminated": dataset["terminated"].sum().item(),
        "num_truncated": dataset["truncated"].sum().item(),
    }
    
    return stats


__all__ = [
    # MPAIL format functions (legacy)
    "create_transition_pairs",
    "package_mpail_demos",
    "package_mpail_demos_multimodal",
    "save_demonstrations",
    "load_demonstrations",
    "merge_demonstrations",
    "get_demo_stats",
    # Universal format functions (new)
    "save_expert_dataset",
    "load_expert_dataset",
    "convert_to_mpail_format",
    "is_universal_format",
    "is_mpail_format",
    "get_dataset_stats",
]
