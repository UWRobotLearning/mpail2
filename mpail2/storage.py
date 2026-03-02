import math
import torch


class DictDemoDataset(torch.utils.data.Dataset):
    def __init__(self, demo_dict, horizon=1, device='cuda', obs_normalizer=None):
        """
        Dataset for sampling consecutive subtrajectories from flattened expert demonstrations.

        Args:
            demo_dict: Dictionary of demonstrations with shape [N, 2, *obs_shape]
                      where N is flattened trajectories and 2 is (obs, next_obs)
            horizon: Length of subtrajectories to sample
            device: Device for sampling operations
            obs_normalizer: Optional observation normalizer to apply to demonstrations
        """
        self.demo_dict = demo_dict
        self.horizon = horizon
        self.device = device
        self.obs_normalizer = obs_normalizer

        # All groups should have same N
        first_value = next(iter(demo_dict.values()))
        self.num_transitions = first_value.shape[0]

        # Calculate number of valid subtrajectories
        # We can sample starting from indices 0 to (num_transitions - horizon)
        self.num_samples = max(1, self.num_transitions - horizon + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Return a consecutive subtrajectory of length horizon starting at idx.

        Returns:
            obs_traj: Dictionary {group: [horizon, *obs_shape]}
            next_obs_traj: Dictionary {group: [horizon, *obs_shape]}
        """
        # Sample consecutive transitions from idx to idx+horizon
        end_idx = idx + self.horizon

        obs_traj = {group: data[idx:end_idx, 0] for group, data in self.demo_dict.items()}
        next_obs_traj = {group: data[idx:end_idx, 1] for group, data in self.demo_dict.items()}

        # Apply observation normalization if available
        if self.obs_normalizer is not None:
            obs_traj = self.obs_normalizer(obs_traj)
            next_obs_traj = self.obs_normalizer(next_obs_traj)

        return obs_traj, next_obs_traj

    def mini_traj_batch_generator(self, batch_size, num_epochs=1):
        """
        Generate batches of subtrajectories, sampling without replacement within each epoch.

        Args:
            batch_size: Number of subtrajectories per batch
            num_epochs: Number of epochs to iterate

        Yields:
            Tuple of (obs_batch, next_obs_batch)
            Shape of each batch element: (batch_size, horizon, *feature_dims)
        """
        if self.num_samples == 0:
            for _ in range(num_epochs):
                yield (None, None)
            return

        # Clamp batch size to available samples
        actual_batch_size = min(batch_size, self.num_samples)

        for epoch in range(num_epochs):
            if actual_batch_size <= 0:
                yield (None, None)
                continue

            # Sample unique start indices without replacement using direct index computation
            sample_indices = torch.randperm(self.num_samples, device=self.device)[:actual_batch_size]

            # Build sequences using tensor indexing
            # Create time indices: (horizon, batch_size) where each column is start_idx + [0,1,2,...,horizon-1]
            time_indices = sample_indices.unsqueeze(0) + torch.arange(self.horizon, device=self.device).unsqueeze(1)

            # Index into demonstrations: [N, 2, *obs_shape]
            # time_indices has shape [horizon, batch_size], use advanced indexing
            obs_batch = {k: v[time_indices, 0].transpose(0, 1) for k, v in self.demo_dict.items()}
            next_obs_batch = {k: v[time_indices, 1].transpose(0, 1) for k, v in self.demo_dict.items()}

            # Apply observation normalization if available
            if self.obs_normalizer is not None:
                obs_batch = self.obs_normalizer(obs_batch)
                next_obs_batch = self.obs_normalizer(next_obs_batch)

            # Result shape: [batch_size, horizon, *obs_shape]
            yield (obs_batch, next_obs_batch)


class RolloutStorage:
    """Storage for MPAIL training with dictionary observations.

    Maintains episode storage and a replay buffer for off-policy learning.
    Uses stack (FIFO) replacement strategy for the replay buffer.
    """

    class Transition:
        """Container for environment transitions."""
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None

        def clear(self):
            self.__init__()

    def __init__(self,
        num_envs,
        num_steps_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        replay_size=None,
        replay_batch_size=None,
        device=torch.device("cuda"),
        obs_normalizer=None
    ):
        # Store basic parameters
        self.device = device
        self.num_steps_per_env = num_steps_per_env
        self.num_envs = num_envs
        self.action_shape = action_shape
        self.replay_batch_size = replay_batch_size

        # Observations are always dictionaries
        self.actor_obs_shape = actor_obs_shape

        # Initialize episode storage
        self.observations = {
            k: torch.zeros(num_steps_per_env, num_envs, *v, device=self.device)
            for k, v in actor_obs_shape.items()
        }

        self.actions = torch.zeros(num_steps_per_env, num_envs, *action_shape, device=self.device)
        self.rewards = torch.zeros(num_steps_per_env, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_steps_per_env, num_envs, 1, device=self.device).byte()

        # Privileged observations (not used in MPAIL but kept for compatibility)
        self.privileged_observations = None

        # Initialize replay buffer
        self._replay = replay_size is not None
        if self._replay:

            self.replay_dim_size = math.ceil(replay_size / num_steps_per_env) # number of trajectories

            self.replay_data = {
                "obs": {k: torch.zeros(num_steps_per_env, self.replay_dim_size, *v, device=self.device)
                        for k, v in actor_obs_shape.items()},
                "actions": torch.zeros(num_steps_per_env, self.replay_dim_size, *action_shape, device=self.device),
                "dones": torch.zeros(num_steps_per_env, self.replay_dim_size, 1, device=self.device),
            }

        self.num_trajectories_stored = 0
        self.obs_normalizer = obs_normalizer

        # Counter for number of transitions stored
        self.step = 0

    def add_transitions(self, transition: 'RolloutStorage.Transition'):

        # check if the transition is valid
        if self.step >= self.num_steps_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core - observations are always dict
        for k in self.actor_obs_shape.keys():
            self.observations[k][self.step].copy_(transition.observations[k])

        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        self.step += 1

    def clear(self):
        """Clear the episode storage."""
        self.step = 0

    def _get_obs(self, obs_storage, key=None):
        """Get observations from dict storage"""
        if key is not None:
            return obs_storage[key]
        # Return the first key's observations for shape inference
        return next(iter(obs_storage.values()))

    def _get_num_envs(self):
        """Get number of envs from dict observations"""
        return self._get_obs(self.observations).shape[1]

    def _roll_obs(self, obs_storage):
        """Roll observations to get next_obs"""
        return {k: torch.roll(v, shifts=-1, dims=0) for k, v in obs_storage.items()}

    @property
    def has_replay_data(self):
        """Check if replay buffer has sufficient data for training"""
        if not self._replay:
            return False
        # Check if we have at least some data in the replay buffer
        return self.num_trajectories_stored > 0

    def _add_new_data_to_buffer(self):
        """Helper to add new trajectory data to replay buffer.
        Returns the number of samples that were added (for overflow handling)."""
        num_envs = self._get_num_envs()
        space_remaining = self.replay_dim_size - self.num_trajectories_stored

        if space_remaining <= 0:
            return 0  # Buffer already full, no samples added

        # Only add as much as will fit
        num_to_add = min(num_envs, space_remaining)
        end_indx = self.num_trajectories_stored + num_to_add

        obs = self.observations
        actions = self.actions
        dones = self.dones

        for k in self.actor_obs_shape.keys():
            self.replay_data["obs"][k][:, self.num_trajectories_stored:end_indx] = obs[k][:, :num_to_add]

        self.replay_data["actions"][:, self.num_trajectories_stored:end_indx] = actions[:, :num_to_add]
        self.replay_data["dones"][:, self.num_trajectories_stored:end_indx] = dones[:, :num_to_add].to(torch.float32)

        self.num_trajectories_stored = end_indx

        # Return number of samples added
        return num_to_add

    def process_replay_buffer(self):
        """Main entry point for processing replay buffer using FIFO strategy."""
        if not self._replay:
            return

        # Try to add new data to buffer first
        num_added = self._add_new_data_to_buffer()

        # If buffer has space and we added all data, we're done
        if self.num_trajectories_stored <= self.replay_dim_size:
            return

        # Buffer is full - use FIFO to add remaining data
        num_envs = self._get_num_envs()
        num_samples = num_envs - num_added

        # Shift existing data left to make room for new data
        shift_size = num_samples
        for k in self.actor_obs_shape.keys():
            self.replay_data["obs"][k][:, :-shift_size] = self.replay_data["obs"][k][:, shift_size:]
        self.replay_data["actions"][:, :-shift_size] = self.replay_data["actions"][:, shift_size:]
        self.replay_data["dones"][:, :-shift_size] = self.replay_data["dones"][:, shift_size:]

        # Add new data at the end
        obs = self.observations
        actions = self.actions
        dones = self.dones

        start_idx = self.replay_dim_size - num_samples
        for k in self.actor_obs_shape.keys():
            self.replay_data["obs"][k][:, start_idx:] = obs[k][:, num_added:num_added + num_samples]
        self.replay_data["actions"][:, start_idx:] = actions[:, num_added:num_added + num_samples]
        self.replay_data["dones"][:, start_idx:] = dones[:, num_added:num_added + num_samples].to(torch.float32)

    def _compute_valid_starts(self, dones_data, horizon, max_start):
        """
        Compute valid starting positions for trajectory sampling that don't cross episode boundaries.

        A start position t in trajectory i is valid if:
        - dones[t, i], dones[t+1, i], ..., dones[t+H-2, i] are all 0
        - dones[t+H-1, i] can be 0 or 1 (terminal at last step is okay)

        This ensures the sampled subsequence doesn't cross an episode boundary,
        since a done=1 means the next observation is from a reset (different episode).

        Args:
            dones_data: [T, num_trajs, 1] tensor of done flags
            horizon: Length of subsequences to sample
            max_start: Maximum valid start index (T - horizon - 1)

        Returns:
            valid_indices: [N, 2] tensor of (traj_idx, start_time) pairs, or None if all valid
        """
        T, num_trajs, _ = dones_data.shape
        dones = dones_data.squeeze(-1)  # [T, num_trajs]

        # For horizon=1, all positions are valid (no internal transitions to cross)
        if horizon <= 1:
            return None  # Signal that all positions are valid

        # We need to check that dones[t:t+H-1] has no 1s for each start position t
        # Using efficient cumsum approach:
        # - cumsum gives running count of dones
        # - For window [t, t+H-1), count = cumsum[t+H-1] - cumsum[t] (with t=0 edge case)

        # Compute cumsum along time dimension
        cumsum = torch.cumsum(dones.float(), dim=0)  # [T, num_trajs]

        # Pad with zeros at the start for easier indexing
        padded = torch.cat([torch.zeros(1, num_trajs, device=self.device), cumsum], dim=0)  # [T+1, num_trajs]

        # For start position t, window is [t, t+H-1) (H-1 elements, excluding last step)
        # Number of dones in window = padded[t+H-1] - padded[t]
        num_valid_starts = max_start + 1

        # Compute done counts for each window [t, t+H-1) for t in [0, max_start]
        # window_dones[t, i] = number of dones in dones[t:t+H-1, i]
        end_indices = torch.arange(horizon - 1, horizon - 1 + num_valid_starts, device=self.device)
        start_indices = torch.arange(num_valid_starts, device=self.device)

        window_dones = padded[end_indices, :] - padded[start_indices, :]  # [num_valid_starts, num_trajs]

        # A position is valid if window has no dones (count == 0)
        valid_mask = (window_dones == 0)  # [num_valid_starts, num_trajs]

        # Check if all positions are valid (common case for fixed-length episodes)
        if valid_mask.all():
            return None  # Signal that all positions are valid

        # Get indices of valid (start_time, traj_idx) pairs
        valid_positions = valid_mask.nonzero(as_tuple=False)  # [N, 2] with (start_time, traj_idx)

        # Swap columns to get (traj_idx, start_time) format
        valid_indices = valid_positions[:, [1, 0]]  # [N, 2] with (traj_idx, start_time)

        return valid_indices

    def mini_traj_batch_generator(self, horizon, num_epochs=8):
        """Generate batches of trajectory subsequences from replay buffer.

        Only samples subsequences that stay within episode boundaries (don't cross dones).
        This ensures all sampled data is valid for dynamics and value learning.

        Args:
            horizon: Length of trajectory subsequences to sample
            num_epochs: Number of epochs to iterate over the data
        Yields:
            Tuple of (obs_batch, actions_batch, next_obs_batch, terminal_mask)
            - obs_batch, actions_batch, next_obs_batch: Shape (batch_size, horizon, feature_dims)
            - terminal_mask: Shape (batch_size,) - True if last timestep is a terminal state
        """

        if not self._replay or self.num_trajectories_stored == 0:
            # No replay data available, yield empty batches
            for epoch in range(num_epochs):
                yield (None, None, None, None)
            return

        # Keep reference to replay data without materializing large slices
        # This avoids allocating huge tensors for image observations
        replay_obs_data = self.replay_data["obs"]
        replay_actions_traj = self.replay_data["actions"][:, :self.num_trajectories_stored]
        replay_dones_traj = self.replay_data["dones"][:, :self.num_trajectories_stored]

        trajectory_length = self.num_steps_per_env
        num_trajectories = self.num_trajectories_stored

        # Clamp horizon to available trajectory length
        horizon = min(horizon, trajectory_length)

        # Determine batch size (number of subsequences to sample per epoch)
        if self.replay_batch_size is not None:
            batch_size = self.replay_batch_size
        else:
            batch_size = num_trajectories

        # Calculate max start index
        # Exclude last timestep since it has invalid next_obs (wraps to first)
        max_start = trajectory_length - horizon - 1
        if max_start < 0:
            max_start = 0

        # Compute valid start positions that don't cross episode boundaries
        valid_indices = self._compute_valid_starts(replay_dones_traj, horizon, max_start)

        if valid_indices is None:
            # All positions are valid - use original flat indexing (more efficient)
            num_starts_per_traj = max_start + 1
            max_subsequences = num_trajectories * num_starts_per_traj
            use_valid_indices = False
        else:
            # Only some positions are valid - sample from valid_indices
            max_subsequences = valid_indices.shape[0]
            use_valid_indices = True

        if max_subsequences == 0:
            # No valid subsequences (can happen if all trajectories have early terminations)
            for epoch in range(num_epochs):
                yield (None, None, None, None)
            return

        # Determine actual batch size based on available subsequences
        actual_batch_size = min(batch_size, max_subsequences)

        for epoch in range(num_epochs):
            if actual_batch_size <= 0:
                yield (None, None, None, None)
                continue

            # Sample unique subsequences without replacement
            sample_indices = torch.randperm(max_subsequences, device=self.device)[:actual_batch_size]

            if use_valid_indices:
                # Index into precomputed valid (traj_idx, start_time) pairs
                sampled_pairs = valid_indices[sample_indices]  # [batch, 2]
                traj_indices = sampled_pairs[:, 0]
                start_times = sampled_pairs[:, 1]
            else:
                # Convert flat indices to (traj_idx, start_time) using modular arithmetic
                num_starts_per_traj = max_start + 1
                traj_indices = sample_indices // num_starts_per_traj
                start_times = sample_indices % num_starts_per_traj

            # Build sequences using tensor indexing
            # Create time indices: (horizon, batch) where each column is start_time + [0,1,2,...,horizon-1]
            time_indices = start_times.unsqueeze(0) + torch.arange(horizon, device=self.device).unsqueeze(1)
            # For next_obs, shift time indices by 1 (avoids expensive torch.roll on full buffer)
            next_time_indices = time_indices + 1

            # Index directly from replay data - only allocates batch_size elements, not full buffer
            obs_seq = {k: v[time_indices, traj_indices].transpose(0, 1)
                       for k, v in replay_obs_data.items()}
            next_obs_seq = {k: v[next_time_indices, traj_indices].transpose(0, 1)
                           for k, v in replay_obs_data.items()}

            # Apply observation normalization on the batch (not the full buffer)
            if self.obs_normalizer is not None:
                obs_seq = self.obs_normalizer(obs_seq)
                next_obs_seq = self.obs_normalizer(next_obs_seq)

            actions_seq = replay_actions_traj[time_indices, traj_indices].transpose(0, 1)

            # Get terminal mask for the last timestep of each subsequence
            # This tells the learner if the last transition ends in a terminal state
            # Shape: [batch_size] - True if done[t+H-1] == 1
            last_time_indices = time_indices[-1]  # [batch]
            terminal_mask = replay_dones_traj[last_time_indices, traj_indices, 0].bool()  # [batch]

            yield (obs_seq, actions_seq, next_obs_seq, terminal_mask)

