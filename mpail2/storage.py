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
    Supports both stack (FIFO) and random replacement strategies for the replay buffer.
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
        use_stack_buffer=False,
        obs_normalizer=None
    ):
        # Store basic parameters
        self.device = device
        self.num_steps_per_env = num_steps_per_env
        self.num_envs = num_envs
        self.action_shape = action_shape

        self.use_stack_buffer = use_stack_buffer
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

    def _process_replay_buffer_stack(self, start_env_idx=0, num_samples=None):
        """Stack-like behavior: new data appended, old data removed FIFO

        Args:
            start_env_idx: Which env index to start sampling from (for overflow handling)
            num_samples: How many samples to add (None = all remaining from start_env_idx)
        """
        num_envs = self._get_num_envs()
        if num_samples is None:
            num_samples = num_envs - start_env_idx

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
            self.replay_data["obs"][k][:, start_idx:] = obs[k][:, start_env_idx:start_env_idx + num_samples]
        self.replay_data["actions"][:, start_idx:] = actions[:, start_env_idx:start_env_idx + num_samples]
        self.replay_data["dones"][:, start_idx:] = dones[:, start_env_idx:start_env_idx + num_samples].to(torch.float32)

    def _process_replay_buffer_random(self, start_env_idx=0, num_samples=None):
        """Random replacement behavior

        Args:
            start_env_idx: Which env index to start sampling from (for overflow handling)
            num_samples: How many samples to add (None = all remaining from start_env_idx)
        """
        num_envs = self._get_num_envs()
        if num_samples is None:
            num_samples = num_envs - start_env_idx

        obs = self.observations
        actions = self.actions
        dones = self.dones

        # Permute indices across envs for adding new traj data
        permute_obs_inds = torch.randperm(num_samples, device=self.device)[:num_samples] + start_env_idx

        add_obs = {k: v[:, permute_obs_inds] for k, v in obs.items()}
        add_actions = actions[:, permute_obs_inds]
        add_dones = dones[:, permute_obs_inds]

        # Overwrite random elements of the replay buffer with new data
        replace_inds = torch.randperm(self.replay_dim_size, device=self.device)[:num_samples]

        for k in self.actor_obs_shape.keys():
            self.replay_data["obs"][k][:, replace_inds] = add_obs[k]
        self.replay_data["actions"][:, replace_inds] = add_actions
        self.replay_data["dones"][:, replace_inds] = add_dones.to(torch.float32)

    def process_replay_buffer(self):
        """Main entry point for processing replay buffer."""
        if not self._replay:
            return

        # Try to add new data to buffer first
        num_added = self._add_new_data_to_buffer()

        # If buffer has space and we added all data, we're done
        if self.num_trajectories_stored <= self.replay_dim_size:
            return

        # Buffer is full - process any remaining data or all data if buffer was already full
        if self.use_stack_buffer:
            self._process_replay_buffer_stack(start_env_idx=num_added)
        else:
            self._process_replay_buffer_random(start_env_idx=num_added)

    def mini_traj_batch_generator(self, horizon, num_epochs=8):
        """Generate batches of trajectory subsequences from replay buffer.

        Args:
            horizon: Length of trajectory subsequences to sample
            num_epochs: Number of epochs to iterate over the data
        Yields:
            Tuple of (obs_batch, actions_batch, next_obs_batch, dones_batch)
            Shape of each batch element: (batch_size, horizon, feature_dims)
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

        # Calculate max number of valid subsequences across all trajectories
        # Exclude last timestep since it has invalid next_obs (wraps to first)
        max_start = trajectory_length - horizon - 1
        if max_start < 0:
            max_start = 0
        max_subsequences = num_trajectories * (max_start + 1)

        # Determine actual batch size based on available subsequences
        actual_batch_size = min(batch_size, max_subsequences)

        for epoch in range(num_epochs):
            if actual_batch_size <= 0:
                yield (None, None, None, None)
                continue

            # Sample unique subsequences without replacement using direct index computation
            sample_indices = torch.randperm(max_subsequences, device=self.device)[:actual_batch_size]

            # Convert flat indices to (traj_idx, start_time) using modular arithmetic
            # Each trajectory has (max_start + 1) valid start positions
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
            dones_seq = replay_dones_traj[time_indices, traj_indices].to(torch.float32).transpose(0, 1)

            yield (obs_seq, actions_seq, next_obs_seq, dones_seq)

