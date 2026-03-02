import torch
from typing import TYPE_CHECKING, Any, Dict, Optional

from mpail2.sampling import PolicySampling
from mpail2.dynamics import Dynamics
from mpail2.reward import Reward
from mpail2.value import EnsembleValue
from mpail2.encoder import Coder, Decoder
from mpail2.utils import resolve_obj

if TYPE_CHECKING:
    from .configs.cfgs import PlannerCfg


class Planner(torch.nn.Module):
    '''MPAIL Planner with learnable dynamics, reward, and policy sampling.

    Combines model-predictive control with learned components for imitation learning.
    '''

    encoder: Coder
    decoder: Optional[Decoder]
    dynamics: Dynamics
    reward: Reward
    value: EnsembleValue
    sampling: PolicySampling

    def __init__(self,
        policy_config: 'PlannerCfg',
        num_envs: int,
        device: torch.device = "cuda",
        dtype = torch.float32,
    ):
        super().__init__()
        self.device, self.dtype = device, dtype

        self.num_envs = num_envs

        self.cfg = policy_config

        # Create encoder and decoder at planner level
        self.encoder = resolve_obj(self.cfg.encoder_cfg.class_type)(cfg=self.cfg.encoder_cfg)
        if self.cfg.decoder_cfg is not None:
            self.decoder = resolve_obj(self.cfg.decoder_cfg.class_type)(cfg=self.cfg.decoder_cfg)
        else:
            self.decoder = None

        self.dynamics = resolve_obj(self.cfg.dynamics_cfg.class_type)(
            self.cfg.dynamics_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.reward = resolve_obj(self.cfg.reward_cfg.class_type)(
            self.cfg.reward_cfg,
            self.num_envs,
            device=self.device
        )
        self.value = resolve_obj(self.cfg.value_cfg.class_type)(
            self.cfg.value_cfg,
            self.num_envs,
            device=self.device
        )
        self.gamma = self.cfg.gamma
        self.sampling = resolve_obj(self.cfg.sampling_cfg.class_type)(
            self.cfg.sampling_cfg,
            self.num_envs,
            device=self.device
        )

        self.to(device=device, dtype=dtype)
        self.u_per_command = self.cfg.u_per_command
        self.num_envs = num_envs

        ## Initialize buffers (rollouts, costs, weights, optimal controls)
        _decode_obs_dim = self.decoder.cfg.output_dim if self.decoder is not None else 0
        self._rollouts_shape = (self.num_envs, self.sampling.K, self.sampling.T + 1, _decode_obs_dim)
        self._opt_controls_shape = (self.num_envs, self.sampling.T, self.sampling.nu)

        self._z_rollouts = torch.zeros(
            (*(self._rollouts_shape[:-1]), self.dynamics.cfg.latent_dim),
            device=self.device, dtype=self.dtype # For storing latent rollouts
        )

        # For debugging, algorithmic data, and visualization
        # Updated upon calling optimize()
        self._current_obs = None
        self._opt_controls = torch.zeros(self._opt_controls_shape, device=self.device, dtype=self.dtype) # Optimal controls
        self._returns = torch.zeros_like(self._z_rollouts[..., :-1, 0]) # [num_envs, K, T]; transitions is T - 1
        self._weights = torch.zeros_like(self._returns[..., 0]) # [num_envs, K]
        self._opt_states = torch.zeros_like(self._z_rollouts[:, 0, ...])

        self._pred_obs = self.decoder is not None
        if self._pred_obs:
            self._rollouts = torch.zeros(self._rollouts_shape, dtype=self.dtype, device=self.device) # Sampled rollouts
        else:
            self._rollouts = None

        # For action selection wrappers
        self._prior_controls = torch.zeros((self.num_envs, self.sampling.K, self.sampling.T, self.sampling.nu),
                                                  device=self.device, dtype=self.dtype) # [num_envs, K, T, nu]

        # Initialize visualization if configured
        self.vis = None
        vis_cfg = getattr(self.cfg, "vis_cfg", None)
        if vis_cfg:
            self.vis = resolve_obj(vis_cfg.class_type)(
                vis_cfg,
            )

        # Previous latent encoding: (num_envs, latent_dim)
        self._prev_z = torch.zeros(
            (num_envs, self.dynamics.cfg.latent_dim),
            device=self.device, dtype=self.dtype
        )

        # Last actions: (num_envs, action_dim)
        self._last_actions = torch.zeros(
            (num_envs, self.dynamics.cfg.action_dim),
            device=self.device, dtype=self.dtype
        )

        self.cfg = policy_config
        self.to(device=device, dtype=dtype)
        self.num_envs = num_envs
        self.device, self.dtype = device, dtype

        # Make temperature learnable
        self._temp_exp = torch.nn.Parameter(torch.log(torch.tensor(self.cfg.temperature, device=self.device, dtype=self.dtype)))

        self._obs_normalizer = None

        print(f"[INFO] MPAILPlanner initialized. Total number of params: {sum(p.numel() for p in self.parameters())}")
        print(f"[INFO] \tEncoder: {sum(p.numel() for p in self.encoder.parameters())}")
        if self.decoder is not None:
            print(f"[INFO] \tDecoder: {sum(p.numel() for p in self.decoder.parameters())}")
        print(f"[INFO] \tDynamics: {sum(p.numel() for p in self.dynamics.parameters())}")
        print(f"[INFO] \tSampling: {sum(p.numel() for p in self.sampling.parameters())}")
        print(f"[INFO] \tReward: {sum(p.numel() for p in self.reward.parameters())}")
        print(f"[INFO] \tValue: {sum(p.numel() for p in self.value.parameters())}")
        print(self) # Prints torch summary of the model

    def td_return(self, rollouts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        '''Evaluate rollouts using learned reward and value functions.

        Computes TD-style returns: single-step rewards for intermediate steps,
        and value estimates for the terminal state.

        Args:
            rollouts: Latent state trajectories, shape (num_envs, K, horizon + 1, latent_dim)
            actions: Action sequences, shape (num_envs, K, horizon, action_dim)

        Returns:
            Returns tensor of shape (num_envs, K, horizon) where:
            - [:, :, :-1] contains single-step rewards r(s, s')
            - [:, :, -1] contains terminal value estimates V(s_T, a_T)
        '''
        rewards = torch.zeros_like(actions[..., 0])  # [num_envs, K, horizon]

        # Evaluate single step rewards
        states = rollouts[..., :-1, :]
        next_states = rollouts[..., 1:, :]
        rewards[...] = self.reward(state=states, action=actions, next_state=next_states)

        # Evaluate terminal state value
        ts_states = states[..., -1, :]
        ts_actions = actions[..., -1, :]
        # Override last step reward with terminal state value
        rewards[..., -1] = self.value(state=ts_states, action=ts_actions, return_type='avg')

        if self.gamma is not None:
            _gam_factors = self.gamma ** torch.arange(
                rewards.shape[-1], device=rollouts.device, dtype=rollouts.dtype
            )
            rewards *= _gam_factors

        return rewards

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self._temp_exp)

    @property
    def action_dim(self):
        return self.sampling.nu

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        use_prev_opt: bool = True,
        vis_rollouts: bool = False,
        deterministic:bool=False,
    ) -> torch.Tensor:

        if self._obs_normalizer is not None:
            observations = self._obs_normalizer(observations)

        actions = self.step(obs=observations, deterministic=deterministic).squeeze(-2)  # Update belief

        if vis_rollouts:
            self.create_vis()

        self._last_actions = actions
        with torch.no_grad():
            self._prev_z = self.encoder(self._current_obs)

        return actions

    def update(self, obs: torch.Tensor, map: torch.Tensor=None):
        '''
        Update the internal belief state of the agent.
        '''
        self._current_obs = obs

    def reset(self, reset_inds: torch.Tensor=None):
        '''
        Clear controller state after finishing a trial.
        '''
        if reset_inds is None:
            self._opt_controls[:] = 0.
            self._prev_z[:] = 0.
            self._last_actions[:] = 0.
        else:
            self._opt_controls[reset_inds] = 0.
            self._prev_z[reset_inds] = 0.
            self._last_actions[reset_inds] = 0.

        self.sampling.reset_iter_state()

    def step(self, obs, use_prev_opt:bool=True, deterministic:bool=False) -> torch.Tensor:
        '''
        Perform update and forward pass of the MPPI controller in immediate sequence.
        Returns next best controls. Seeds optimization with previous optimal controls.
        '''
        self.update(obs) # Update belief

        # Seeds samples with previous mean if use_prev_opt is True
        if use_prev_opt:
            # Fills in the last u_per_command controls with 0
            self._opt_controls[:] = torch.roll(self._opt_controls, shifts=-self.u_per_command, dims=-2)
            self._opt_controls[:, -self.u_per_command:, :] = 0.
        else:
            self._opt_controls[:] = 0.

        self.sampling.reset_iter_state()

        for i in range(self.cfg.opt_iters - 1):
            # Subsequent optimization uses previous optimal controls
            plan = self.optimize(obs, iter=i, deterministic=True)

        plan = self.optimize(
            obs, iter=self.cfg.opt_iters - 1, deterministic=deterministic
        )

        return plan[:, :self.u_per_command, :] # Forward pass and next best control

    def optimize(self, obs, iter=None, deterministic:bool=False) -> torch.Tensor:
        """
        Perform forward pass of the MPPI controller.
        :param: x0

        Uses current self._opt_controls as the mean for sampling.

        Weight computation adapted from:
        https://github.com/UM-ARM-Lab/pytorch_mppi/blob/bfcc9150ec9066fb5a0f01b65ddb603c49c66867/src/pytorch_mppi/mppi.py#L197
        """

        # Sample rollouts and compute rewards
        with torch.no_grad():
            z0 = self.encoder(obs)
            self._prior_controls[:] = self.sampling.sample(prev_controls=self._opt_controls, iter=iter, state=z0) # [num_envs, K, T, nu]
            self._z_rollouts[:] = self.dynamics(z0, self._prior_controls)
            if self._pred_obs:
                self._rollouts[:] = self.decoder(self._z_rollouts)
            self._returns[:] = self.td_return(rollouts=self._z_rollouts, actions=self._prior_controls)  # [num_envs, K, T]

        # Update weights and optimal controls via CEM-MPPI
        _elite_idxs = torch.topk(self._returns.sum(dim=-1), k=self.cfg.num_elites).indices
        _elite_values = self._returns.gather(
            dim=-2,
            index=_elite_idxs.unsqueeze(-1)
        ) # [num_envs, num_elites, T]

        elite_rewards = _elite_values.sum(dim=-1)  # [num_envs, num_elites]
        max_value = elite_rewards.max(dim=-1, keepdim=True).values  # [num_envs, 1]
        score = torch.exp((1. / self.temperature) * (elite_rewards - max_value))  # [num_envs, num_elites]
        score = score / (score.sum(dim=-1, keepdim=True) + 1e-9)  # normalize

        # Get elite actions: [num_envs, num_elites, T, nu]
        _elite_actions = self._prior_controls.gather(
            dim=1,
            index=_elite_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self._prior_controls.shape[-2], self._prior_controls.shape[-1])
        )
        self.sampling.update_plan_dist_from_elites(_elite_actions, score)

        # Set weights to zero except for elites
        self._weights[:] = 0.
        self._weights.scatter_(dim=1, index=_elite_idxs, src=score)

        if not deterministic:
            self._opt_controls[:] = self._sample_plan()
        else:
            self._opt_controls[:] = torch.sum(self._prior_controls * self._weights[..., None, None], dim=-3)

        return self._opt_controls

    def _sample_plan(self) -> torch.Tensor:
        '''
        Uses current weights and rollouts to sample new actions plans.

        :param sample_from_gaussian: If True, samples from the last fitted Gaussian
            (iter_mean + iter_std * noise) instead of multinomial sampling from rollouts.
        :return: sampled plan [..., T, nu]
        '''
        # Sample from the fitted Gaussian
        noise = torch.randn_like(self.sampling._iter_mean)
        sampled_plan = self.sampling._iter_mean + self.sampling._iter_std * noise
        sampled_plan = torch.tanh(sampled_plan)  # Ensure actions are in [-1, 1]
        return sampled_plan  # [num_envs, T, nu]

    def compute_policy_influence(self) -> torch.Tensor:
        '''
        Compute the influence of the learned action distribution on the sampled actions
        :param actions: [num_envs, K, T, nu] - sampled actions
        :return: [num_envs, K, T] - log probabilities of sampled actions under the action distribution
        '''
        _K_policy = self.sampling._K_policy
        policy_weights = self._weights[..., -_K_policy:].detach().clone()  # [num_envs, K_policy]
        influence = policy_weights.sum(dim=-1)  # [num_envs, K_policy]
        return influence

    def set_obs_normalizer(self, obs_normalizer):
        self._obs_normalizer = obs_normalizer

    def compute_stats(self) -> Dict[str, Any]:
        _rewards = self._returns.detach().clone()
        mean_ss_rewards = _rewards[..., :-1].mean()
        min_ss_rewards = _rewards[..., :-1].min().detach().clone()
        max_ss_rewards = _rewards[..., :-1].max().detach().clone()
        mean_ts_rewards = _rewards[..., -1].mean()
        max_ts_rewards = _rewards[..., -1].max().detach().clone()
        min_ts_rewards = _rewards[..., -1].min().detach().clone()
        std_ss_rewards = _rewards[..., :-1].std()
        std_ts_rewards = _rewards[..., -1].std()
        _traj_rewards = _rewards.sum(dim=-1) # [num_envs, K]
        min_traj_reward = _traj_rewards.min(dim=-1).values.mean()
        max_traj_reward = _traj_rewards.max(dim=-1).values.mean()
        mean_std_traj_reward = _traj_rewards.std(dim=-1).mean()
        policy_influence = self.compute_policy_influence().mean()

        # Compute std of actions
        action_std = self.sampling._iter_std[:, 0, :].detach().clone()  # [num_envs, T, nu]

        stats = {
            "MPPI/Temperature": self.temperature.detach().item(),
            "MPPI/Average Value": mean_ts_rewards,
            "MPPI/Average Reward": mean_ss_rewards,
            "MPPI/Min Reward": min_ss_rewards.item(),
            "MPPI/Max Reward": max_ss_rewards.item(),
            "MPPI/Min Value": min_ts_rewards.item(),
            "MPPI/Max Value": max_ts_rewards.item(),
            "MPPI/Std Reward": std_ss_rewards,
            "MPPI/Std Value": std_ts_rewards,
            "MPPI/Min Return": min_traj_reward,
            "MPPI/Max Return": max_traj_reward,
            "MPPI/Std Return": mean_std_traj_reward,
            "MPPI/Mean Policy Influence": policy_influence.item(),
        }

        for i in range(action_std.shape[-1]):
            stats[f"MPPI/Mean Action Dist Std{i}"] = action_std[:, i].mean().item()

        return stats

    def create_vis(self):

        with torch.no_grad():

            if not self.vis:
                raise ValueError("Debug visualization is not enabled. Enable visualization by providing " +
                                "a visualizer config to the MPPI configuration.")

            vis_env_ids = list(range(self.vis.vis_n_envs))
            vis_rollouts = self._rollouts[vis_env_ids, :, :-1, :] # [n_envs, n_rollouts, horizon, state_dim]
            vis_rewards = self._returns[vis_env_ids] # [n_envs, n_rollouts, horizon]
            horizon, state_dim = vis_rollouts.shape[2:]

            # Get topk rollouts
            topk_reward_inds = torch.topk(vis_rewards.sum(dim=-1), k=self.vis.vis_n_rollouts, largest=True).indices
            topk_rollout_inds = topk_reward_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, horizon, state_dim)
            topk_vis_rollouts = vis_rollouts.gather(dim=1, index=topk_rollout_inds)
            topk_reward_rollout_inds = topk_reward_inds.unsqueeze(-1).expand(-1, -1, horizon)
            topk_vis_rewards = vis_rewards.gather(dim=1, index=topk_reward_rollout_inds)

            # Get k random rollouts
            rand_inds = torch.randperm(vis_rollouts.shape[1])[:self.vis.vis_n_rollouts]
            rand_vis_rollouts = vis_rollouts[:, rand_inds]
            rand_vis_rewards = vis_rewards[:, rand_inds]

            x0 = self._current_obs
            vis_rollouts = torch.cat([topk_vis_rollouts, rand_vis_rollouts], dim=1)
            vis_rewards = torch.cat([topk_vis_rewards, rand_vis_rewards], dim=1)
            z0 = self.encoder(x0)
            z_rollouts = self.dynamics(z0, self._opt_controls)
            opt_states = self.decoder(z_rollouts) if self.decoder is not None else None  # [num_envs, T, state_dim]

            if self._obs_normalizer is not None:
                vis_rollouts = self._obs_normalizer.inverse(vis_rollouts)
                opt_states = self._obs_normalizer.inverse(opt_states)
                x0 = self._obs_normalizer.inverse(x0)

            # convert to numpy and send to vis
            x0 = x0['proprioception'].cpu().numpy()
            vis_rollouts = vis_rollouts.cpu().numpy()
            opt_states = opt_states.cpu().numpy()
            vis_rewards = vis_rewards.cpu().numpy()

            self.vis.update(
                x0,
                vis_rollouts,
                rollout_rewards=vis_rewards,
                elevation_map=None,
                optimal_control=opt_states
            )
