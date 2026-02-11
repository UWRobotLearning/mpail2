import copy
import math
import time
import torch
from collections import defaultdict
import torch.nn.functional as F
from typing import TYPE_CHECKING, Dict, Optional

from mpail2.encoder import Coder
from mpail2.dynamics import Dynamics
from mpail2.reward import Reward, EnsembleValue
from mpail2.sampling import PolicyNetwork
from mpail2.planner import Planner
from .storage import RolloutStorage as Storage, DictDemoDataset
from .utils import resolve_optim, compute_disc_train_stats
from .utils.obs_normalizer import ObsNormalizerFactory

if TYPE_CHECKING:
    from .configs.cfgs import  MPAIL2LearnerCfg, ValueLearnerCfg


class MPAIL2Learner:
    '''Composes MPPIPlanner to include buffer population for MPAIL training as well as exploratory
    action selection.'''

    planner: Planner
    _encoder: Coder
    _dynamics: Dynamics
    _reward: Reward
    _value: EnsembleValue
    _policy: PolicyNetwork

    def __init__(self,
        demonstrations: torch.Tensor | Storage,
        num_envs: int,
        learner_config: 'MPAIL2LearnerCfg',
        device: str = "cuda",
        dtype=torch.float32
    ):

        self.device = device
        self.dtype = dtype

        self.cfg = learner_config
        self.value_learner_cfg = self.cfg.value_learner_cfg
        self.reward_learner_cfg = self.cfg.reward_learner_cfg

        self.transition = Storage.Transition()

        # Learning params
        self._demonstrations = demonstrations
        self.replay_ratio = self.cfg.replay_ratio

        #
        # PLANNER SETUP
        #

        self.planner:Planner = Planner(self.cfg.planner_cfg, num_envs, device=device, dtype=dtype)

        self._encoder:Coder = self.planner.encoder
        self._dynamics:Dynamics = self.planner.dynamics
        self._reward:Reward = self.planner.td_return.get_reward()
        self._value:EnsembleValue = self.planner.td_return.get_value_function()
        self._policy:PolicyNetwork = self.planner.sampling.policy

        self._decoder:Optional[Coder] = self.planner.decoder

        #
        ## REWARD SETUP
        #

        self._reward_opt = resolve_optim(self.reward_learner_cfg.opt)(
            self._reward.parameters(),
            **self.reward_learner_cfg.opt_params
        )

        #
        ## VALUE SETUP
        #

        self._value_opt = resolve_optim(self.value_learner_cfg.opt)(
            self._value.parameters(),
            **self.value_learner_cfg.opt_params
        )

        # Setup Polyak averaging for value networks
        self._polyak_tau = self.value_learner_cfg.polyak_tau

        # Create target networks as deep copies of the main networks
        self._value_target = self._create_target_value_network()

        #
        # DYNAMICS SETUP
        #

        self.dynamics_learner_cfg = self.cfg.dynamics_learner_cfg

        _enc_lr = self.dynamics_learner_cfg.opt_params['lr'] * self.dynamics_learner_cfg.enc_lr_scale
        dyn_params = [
            {"params": self._encoder.parameters(), "lr": _enc_lr},
            {"params": self._dynamics.parameters()},
        ]
        if self._decoder is not None:
            dyn_params.append({"params": self._decoder.parameters()})
        self._dyn_opt = resolve_optim(self.dynamics_learner_cfg.opt)(
            dyn_params,
            **self.dynamics_learner_cfg.opt_params
        )

        #
        # POLICY SAMPLING SETUP
        #

        self.policy_learner_cfg = self.cfg.policy_learner_cfg
        self._policy_opt = resolve_optim(self.policy_learner_cfg.opt)(
            self._policy.parameters(),
            **self.policy_learner_cfg.opt_params
        )

        # Initialize log_alpha as a learnable parameter
        self._log_alpha = torch.tensor(math.log(1.0), requires_grad=True, device=self.device)
        self._alpha_opt = torch.optim.Adam([self._log_alpha], lr=self.policy_learner_cfg.alpha_lr)
        self._target_entropy = self.policy_learner_cfg.target_entropy

        #
        # OBSERVATION NORMALIZER SETUP (CAMERA ONLY)
        #

        self._obs_normalizer = None
        if self.cfg.obs_normalizer_cfg is not None:
            self._obs_normalizer = ObsNormalizerFactory.create_normalizer(
                normalization_type=self.cfg.obs_normalizer_cfg.normalization_type,
                device=self.device
            )

            self.planner.set_obs_normalizer(self._obs_normalizer)

        self.planner.reset()

    def init_storage(
        self,
        num_envs,
        num_steps_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        # create rollout storage
        self.storage = Storage(
            num_envs,
            num_steps_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            replay_size=self.cfg.replay_size,
            replay_batch_size=self.cfg.replay_batch_size,
            device=self.device,
            use_stack_buffer=self.cfg.use_stack_buffer,
            obs_normalizer=self._obs_normalizer
        )

    def eval(self):
        self.planner.eval()

    def train(self):
        self.planner.train()

    def act(self, obs, critic_obs=None, vis_rollouts=False):
        '''Performs forward pass on MPPI and collects discriminator logits for MPAIL
        TODO: Critic obs?'''

        # Do NOT normalize obs for planner - planner uses raw observations
        # Normalization is only applied when feeding to value function

        # Compute the actions and values
        if critic_obs is None:
            critic_obs = obs

        with torch.no_grad():
            self.transition.actions = self.planner.act(obs, vis_rollouts=vis_rollouts)

        # need to record obs and critic_obs before env.step()
        # Store observations (both dict and tensor cases)
        self.transition.observations = obs
        self.transition.critic_observations = obs

        actions_to_return = self.transition.actions
        return actions_to_return

    def process_env_step(self, rewards, dones, infos, next_obs) -> Dict[str, float]:

        with torch.no_grad():

            # Record the rewards and dones
            # Note: we clone here because later on we bootstrap the rewards based on timeouts
            self.transition.dones = dones

            #################################################################################################
            self.transition.rewards = rewards # NOTE: THESE ARE NOT TO BE USED IN LEARNING - JUST FOR LOGGING
            #################################################################################################

            # Compute policy stats
            stats = self.planner.compute_stats()

            # Record the transition
            self.storage.add_transitions(self.transition)
            self.transition.clear()
            # Convert dones to indices for reset
            reset_inds = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if reset_inds.numel() > 0:
                self.planner.reset(reset_inds)
            # If no environments are done, don't reset anything

        return stats

    @property
    def _total_batch_size(self):
        return self.storage.num_steps_per_env * self.storage.num_envs

    def _create_target_value_network(self):
        """Create target value network as a deep copy of the main value network"""
        target_network = copy.deepcopy(self._value)
        # Disable gradients for target network
        for param in target_network.parameters():
            param.requires_grad_(False)

        target_network.eval()  # Set to eval mode

        for target_param, main_param in zip(target_network.parameters(), self._value.parameters()):
            target_param.data.copy_(main_param.data)

        return target_network

    def _soft_update_target_value(self):
        """Soft update using Polyak averaging: target = (1-tau) * target + tau * main"""
        for target_param, main_param in zip(self._value_target.parameters(), self._value.parameters()):
            target_param.data.copy_(
                (1.0 - self._polyak_tau) * target_param.data + self._polyak_tau * main_param.data
            )

    def update(self, iteration: int) -> Dict[str, float]:

        self.storage.process_replay_buffer() # Move episode storage to replay buffer

        # Demonstrations are dictionaries where each group has [N, 2, *obs_shape]
        # Sample consecutive subtrajectories of length loss_horizon
        demo_dataset = DictDemoDataset(
            self._demonstrations,
            horizon=self.cfg.loss_horizon,
            device=self.device,
            obs_normalizer=self._obs_normalizer
        )

        _num_flat_trajs = self.storage.replay_batch_size // self.cfg.loss_horizon
        demo_batch_size = _num_flat_trajs  # Number of subtrajectories to sample

        # Calculate number of updates based on replay ratio and new data collected
        num_updates = max(1, int(self.replay_ratio * self._total_batch_size))

        # Use custom batch generator to avoid duplicate sampling
        demo_generator = demo_dataset.mini_traj_batch_generator(
            batch_size=demo_batch_size,
            num_epochs=num_updates
        )

        traj_generator = self.storage.mini_traj_batch_generator(
            self.cfg.loss_horizon,
            num_epochs=num_updates
        )

        self._mean_stats  = defaultdict(float)
        self._tot_stats = defaultdict(float)

        actual_updates = 0

        for (
            (replay_obs_traj, replay_actions_traj, replay_next_obs_traj, _),
            (demo_obs_traj, demo_next_obs_traj)
        ) in zip(traj_generator, demo_generator):

            # Flatten dict observations
            demo_obs_flat = {k: v.flatten(start_dim=0, end_dim=1) for k, v in demo_obs_traj.items()}
            demo_next_obs_flat = {k: v.flatten(start_dim=0, end_dim=1) for k, v in demo_next_obs_traj.items()}

            # (1) UPDATE DYNAMICS
            _start = time.time()
            self._mean_stats['Dyn/mean_loss'] += self.update_dynamics( # TODO: dont duplicate encoding
                obs_batch_traj=replay_obs_traj,
                next_obs_batch_traj=replay_next_obs_traj, # For reconstruction loss
                actions_batch_traj=replay_actions_traj,
            )
            self._tot_stats['Dyn/learn_time'] += time.time() - _start

            # Encode observations to latent space
            with torch.no_grad():
                replay_latent_traj = self._encoder(replay_obs_traj)
                replay_next_latent_traj = self._encoder(replay_next_obs_traj)
                demo_latent_flat = self._encoder(demo_obs_flat)
                demo_next_latent_flat = self._encoder(demo_next_obs_flat)

            # Flatten trajectories for non-trajectory-based updates
            _num_flat_trajs = self.storage.replay_batch_size // self.cfg.loss_horizon
            latent_batch_flat = replay_latent_traj[:_num_flat_trajs].flatten(start_dim=0, end_dim=1)
            next_latent_batch_flat = replay_next_latent_traj[:_num_flat_trajs].flatten(start_dim=0, end_dim=1)
            actions_batch_flat = replay_actions_traj[:_num_flat_trajs].flatten(start_dim=0, end_dim=1)

            # (2) UPDATE REWARD
            _start = time.time()
            _reward_stats = self.update_reward(
                demo_latent_batch=demo_latent_flat,
                demo_next_latent_batch=demo_next_latent_flat,
                latent_batch=latent_batch_flat,
                next_latent_batch=next_latent_batch_flat,
            )
            for key, value in _reward_stats.items():
                self._mean_stats[key] += value
            self._tot_stats['Reward/learn_time'] += time.time() - _start

            # (3) UPDATE VALUE
            _start = time.time()
            self._mean_stats['Value/mean_loss'] += self.update_value(
                latent_batch=latent_batch_flat,
                action_batch=actions_batch_flat,
                next_latent_batch=next_latent_batch_flat,
                learner_cfg=self.value_learner_cfg,
            )
            self._tot_stats['Value/learn_time'] += time.time() - _start

            # (4) UPDATE POLICY
            _start = time.time()
            self._mean_stats['Policy/mean_loss'] += self.update_policy( # TODO: dont duplicate encoding
                latent_batch=latent_batch_flat,
            )
            self._tot_stats['Policy/learn_time'] += time.time() - _start

            actual_updates += 1
            self._mean_stats['Data/mean_latent_norm'] += latent_batch_flat.norm(dim=-1).mean().item()

        # Average stats over updates
        for key in self._mean_stats.keys():
            self._mean_stats[key] /= max(1, actual_updates)

        # Store training info
        self._tot_stats['Training/num_updates'] = actual_updates
        self._tot_stats['Training/replay_ratio'] = self.replay_ratio

        # Log how much data is in the replay buffer
        self._tot_stats['Storage/replay_size'] = self.storage.num_trajectories_stored * self.storage.num_steps_per_env

        # -- Clear the storage
        self.storage.clear()

        self._mean_stats.update(self._tot_stats)

        return self._mean_stats

    def update_reward(
        self,
        latent_batch: torch.Tensor,
        next_latent_batch: torch.Tensor,
        demo_latent_batch: torch.Tensor,
        demo_next_latent_batch: torch.Tensor,
    ):
        '''Trains reward network using WGAN loss (discriminator-style adversarial training)'''

        _demo_rewards = self._reward(demo_latent_batch, demo_next_latent_batch)
        _gen_rewards = self._reward(latent_batch, next_latent_batch)

        # WGAN loss: maximize demo rewards, minimize generator rewards
        loss = -_demo_rewards.mean() + _gen_rewards.mean()

        train_stats = {}
        # add gradient penalty if coefficient is specified
        if self.reward_learner_cfg.gp_coeff is not None and self.reward_learner_cfg.gp_coeff > 0:
            gp_loss = self.reward_learner_cfg.gp_coeff * self._compute_gradient_penalty(
                expert_state=demo_latent_batch.detach(),
                expert_next_state=demo_next_latent_batch.detach(),
                gen_state=latent_batch.detach(),
                gen_next_state=next_latent_batch.detach(),
                stats_dict=train_stats,
            )
            loss += gp_loss

        with torch.no_grad():
            # FOR STATS LOGGING
            _output = torch.cat((
                _demo_rewards,
                _gen_rewards
            ))
            _target = torch.cat(( # [ 1 \\ 0 ]
                torch.ones_like(_demo_rewards),
                torch.zeros_like(_gen_rewards)
            ))

        # do gradient step
        self._reward_opt.zero_grad()
        loss.backward()
        self._reward_opt.step()

        # Compute training stats
        with torch.no_grad():

            train_stats.update(compute_disc_train_stats(
                _output, _target, loss, from_probs=False
            ))

            train_stats.update({
                "mean_demo_reward": _demo_rewards.mean().item(),
                "mean_gen_reward": _gen_rewards.mean().item(),
                "demo_std": _demo_rewards.std().item(),
            })

            if self.reward_learner_cfg.gp_coeff is not None and self.reward_learner_cfg.gp_coeff > 0:
                train_stats["gp_loss"] = gp_loss.item()

        _train_stats = {}
        for key, value in train_stats.items():
            _train_stats[f"Reward/{key}"] = value

        _train_stats['Reward/learner_latent_std'] = latent_batch.std(-2).mean().item()
        _train_stats['Reward/demo_latent_std'] = demo_latent_batch.std(-2).mean().item()

        return _train_stats

    def _compute_gradient_penalty(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        gen_state: torch.Tensor,
        gen_next_state: torch.Tensor,
        stats_dict: dict = {},
    ):
        """
        Computes the gradient penalty term for WGAN-GP using interpolation between
        expert and generated samples.
        """
        batch_size = gen_state.size(0)

        # Sample interpolation factors
        alpha_shape = [batch_size] + [1] * (expert_state.ndim - 1)
        alpha = torch.rand(alpha_shape, device=self.device)

        # Interpolation between expert and generated inputs
        _expert_input = torch.cat((expert_state, expert_next_state), dim=-1)
        _gen_input = torch.cat((gen_state, gen_next_state), dim=-1)
        interpolated = torch.lerp(_expert_input, _gen_input, alpha).requires_grad_(True)

        # Reward model output on interpolated samples
        interpolated_scores = self._reward.model(interpolated)

        # Compute gradients w.r.t. interpolated inputs
        gp_gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gp_gradients = gp_gradients.view(batch_size, -1)
        gp_gradient_norm = gp_gradients.norm(2, dim=1)
        penalty = ((gp_gradient_norm - 1.0) ** 2).mean()
        stats_dict['gp_gradient_mean'] = gp_gradient_norm.mean().item()

        return penalty

    def update_value( # TODO: share policy-planned trajectories with policy update
        self,
        latent_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_latent_batch: torch.Tensor,
        learner_cfg : 'ValueLearnerCfg',
    ):

        with torch.no_grad():
            rewards_batch = self._reward(latent_batch, next_latent_batch, action=action_batch)
            _next_plans = self._policy.plan(next_latent_batch) # [batch_size, H, action_dim]
            pred_next_latent_batch_traj = self._dynamics(
                z0=next_latent_batch, controls=_next_plans,
            ) # [batch_size, H+1, latent_dim]
            next_vals = self._n_step_return_lambda(
                z_traj=pred_next_latent_batch_traj[..., :-1, :],
                actions_traj=_next_plans,
                next_z_traj=pred_next_latent_batch_traj[..., 1:, :],
                lam=learner_cfg.lam,
                value=self._value_target,
                reward_fn=self._reward,
            )
            td_target = rewards_batch + learner_cfg.gamma * next_vals

        # Use normalized states for value function
        all_q_values = self._value(latent_batch, action_batch, return_type='all')  # [num_q, batch]

        loss = sum(F.mse_loss(q_vals, td_target) for q_vals in all_q_values) / len(all_q_values)

        self._value_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._value.parameters(), self.value_learner_cfg.max_grad_norm
        )
        self._value_opt.step()

        self._mean_stats['Value/td_target_mean'] += td_target.mean().item()
        self._mean_stats['Value/mean_q_value'] += all_q_values.mean().item()

        # Update target networks using Polyak averaging
        self._soft_update_target_value()

        return loss.item()

    def update_policy(
        self,
        latent_batch: torch.Tensor,
    ): # [batch_size, H, state_dim]

        self._value.track_q_grad(enable=False)

        plans = self._policy.plan(latent_batch)
        log_prob_plans = self._policy._log_prob_plans # [batch_size, H, 1]
        pred_latents = self._dynamics(
            z0=latent_batch, controls=plans,
        )

        alpha = self._log_alpha.exp().squeeze()

        # Multi-step model-based TDLambda
        qs = self._n_step_return_lambda( # [batch_size]
            z_traj=pred_latents[..., :-1, :],
            actions_traj=plans,
            next_z_traj=pred_latents[..., 1:, :],
            value=self._value,
            lam=self.policy_learner_cfg.lam,
            reward_fn=self._reward,
            log_probs=log_prob_plans.squeeze(-1),
            alpha=alpha.detach(),
        )
        loss = -qs.mean()

        self._policy_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._policy.parameters(), self.policy_learner_cfg.max_grad_norm
        )
        self._policy_opt.step()

        # Update alpha
        self._alpha_opt.zero_grad()
        alpha_loss = alpha * (-log_prob_plans.mean(-2).detach() - self._target_entropy).mean()
        alpha_loss.backward()
        self._alpha_opt.step()

        # Log stats
        self._mean_stats['Policy/alpha'] += alpha.item()
        self._mean_stats['Policy/alpha_loss'] += alpha_loss.item()
        self._mean_stats['Policy/entropy'] += -log_prob_plans.mean().item()
        self._mean_stats['Policy/mean_std'] += self._policy.std.mean().detach().item()

        self._value.track_q_grad(enable=True)
        return loss.item()

    def update_dynamics(
        self,
        obs_batch_traj: torch.Tensor,
        next_obs_batch_traj: torch.Tensor, # For optional decoding loss
        actions_batch_traj: torch.Tensor,
    ):

        with torch.no_grad():
            latent_batch_traj = self._encoder(obs_batch_traj)
            next_latent_batch_traj = self._encoder(next_obs_batch_traj)

        _H = actions_batch_traj.shape[1]
        rho = self.dynamics_learner_cfg.rho
        _rhos = rho ** torch.arange(_H, device=self.device)

        # Get initial latent state z0
        z0 = latent_batch_traj[:, 0]

        pred_latents = self._dynamics( # [batch_size, H+1, latent_dim]
            z0=z0, controls=actions_batch_traj,
        )

        # JEP loss: L = sum_t rho^t || z_{t+1} - f(z_t, a_t) ||_2^2
        _jep_se = (pred_latents[:, 1:, :] - next_latent_batch_traj.detach()).pow(2)
        jep_loss = (_rhos[..., None] * _jep_se).mean()

        # Decoder for visualization if specified
        if self._decoder:
            _recon_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for obs_key, obs in next_obs_batch_traj.items():
                _recon_loss += F.mse_loss(
                    self._decoder(next_latent_batch_traj.detach())[obs_key],
                    next_obs_batch_traj[obs_key],
                )
        else:
            _recon_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        dyn_loss = jep_loss

        loss = dyn_loss

        self._mean_stats['Dyn/jep_loss'] += jep_loss.item()
        self._mean_stats['Dyn/recon_loss'] += _recon_loss.item()

        self._dyn_opt.zero_grad()
        loss.backward()
        if self._decoder:
            _recon_loss.backward() # Backprop through decoder
        self._dyn_opt.step()

        # Compute validation losses
        # TODO

        return loss.item()

    def _n_step_return_lambda(
            self, z_traj, actions_traj, next_z_traj, value, lam: float, reward_fn,
            alpha:float=None, log_probs:torch.Tensor=None,
            value_return_type: str='min'
        ):
        '''
        Computes the n-step lambda return. Defined recursively as:
        G_t = lam * Q(z_t, a_t) + (1 - lam) * [r_t + gamma * G_{t+1}]
        Tensors are shape [batch, H, dim]
        Args:
            z_traj: Latent trajectory [batch, H, latent_dim]
            actions_traj: Action trajectory [batch, H, action_dim]
            next_z_traj: Next latent trajectory [batch, H, latent_dim]
            value: Value function to use for bootstrapping
            lam: Lambda parameter for return computation
            reward_fn: Reward function to use
            alpha: Entropy regularization coefficient
            log_probs: Log probabilities of actions [batch, H] (for entropy regularization)
            value_return_type: Return type for value function ('min', 'mean', 'all')
        Returns:
            _return: Lambda return [batch]
        '''

        _gam, _H = self.value_learner_cfg.gamma, self.cfg.loss_horizon
        _all_rewards = reward_fn.reward( # [batch, H]
            z_traj, next_z_traj, action=actions_traj
        )
        _all_vals = value(z_traj, actions_traj, return_type=value_return_type)  # [batch, H] or [num_q, batch, H]
        if log_probs is not None:
            # Entropy regularization adjustment
            _all_rewards = _all_rewards - alpha * log_probs
            _all_vals = _all_vals - alpha * log_probs

        steps = torch.arange(_H - 1, device=self.device, dtype=self.dtype)
        W = (_gam * lam) ** steps

        b = _all_rewards[..., :-1] + _gam * (1 - lam) * _all_vals[..., 1:]
        b[..., -1] += _gam * lam * _all_vals[..., -1]

        _lambda_return = b @ W.view(-1, 1)

        _return = lam * _all_vals[..., 0] + (1 - lam) * _lambda_return.squeeze(-1)

        return _return
