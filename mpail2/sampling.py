import torch
import math
from typing import TYPE_CHECKING, Optional
from mpail2.utils import resolve_obj

from torch.nn import functional as F

if TYPE_CHECKING:
    from .configs.cfgs import PolicySamplingCfg

class PolicyNetwork(torch.nn.Module):
    '''Policy Network for MPAIL
    Predicts action distribution given state input.
    Outputs mean and std deviation for action distribution.
    '''
    def __init__(
        self,
        model: torch.nn.Sequential,
        min_std: float,
        max_std: float,
        horizon: int,
        action_dim: int,
        device="cuda", dtype=torch.float32
    ):
        super().__init__()
        self.to(device=device, dtype=dtype)

        self.model = model
        self.horizon, self.action_dim = horizon, action_dim

        self._min_logstd = math.log(min_std)
        self._max_logstd = math.log(max_std)
        self._log_prob_plans = None
        self.std = None
        self.mu = None

    def act(self, observation, shape:Optional[torch.Size]=None):
        '''Sample first action from the policy given observation'''
        return self.plan(observation, shape=shape)[..., 0, :]

    def plan(self, observation, shape:Optional[torch.Size]=None):
        '''Sample action sequences from the policy given observation
        :param observation: torch.Tensor of shape (..., state_dim)
        :param shape: torch.Size for sampling multiple sequences
        :return: torch.Tensor of shape (..., T, nu)'''

        model_output = self.model(observation)

        # Model outputs both mean and log_std
        # Split the output into mean and log_std
        _batch_shape = model_output.shape[:-1]
        _mu, _log_std = torch.chunk(model_output, 2, dim=-1)
        _mu = _mu.view(*_batch_shape, self.horizon, self.action_dim)
        _log_std = _log_std.view(*_batch_shape, self.horizon, self.action_dim)

        # Convert log_std to std with safe exponential
        _log_std = self._log_std(_log_std, self._min_logstd, self._max_logstd - self._min_logstd)
        self.std = torch.exp(_log_std)
        self.mu = _mu  # Store mean for action_mean property

        # Sample actions
        _shape = _mu.shape if shape is None else shape + _mu.shape
        eps = torch.randn(_shape, device=_mu.device, dtype=_mu.dtype)
        log_prob = self._gaussian_logprob(eps, _log_std)
        plans = _mu + self.std * eps
        self.mu, plans, _log_prob = self._squash(_mu, plans, log_prob)
        self._log_prob_plans = _log_prob

        return plans

    @property
    def action_mean(self) -> torch.Tensor:
        return self.mu[..., 0, :]

    @property
    def action_stddev(self) -> torch.Tensor:
        return self.std[..., 0, :]

    @property
    def log_prob(self) -> torch.Tensor:
        return self._log_prob_plans[..., 0, :]

    @property
    def entropy(self) -> torch.Tensor:
        return -self.log_prob

    def _log_std(self, x, low, dif):
        return low + 0.5 * dif * (torch.tanh(x) + 1)

    def _gaussian_logprob(self, eps, log_std):
        """Compute Gaussian log probability."""
        residual = -0.5 * eps.pow(2) - log_std
        log_prob = residual - 0.9189385175704956
        return log_prob.sum(-1, keepdim=True)

    def _squash(self, mu, pi, log_pi):
        """Apply squashing function."""
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
        log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
        return mu, pi, log_pi


class PolicySampling(torch.nn.Module):
    """
    Policy-based sampling that combines learned policy with Gaussian noise.
    Similar to DeltaSampling but includes a neural network policy component.

    Supports TD-MPC2-style iterative std computation where std is updated
    based on weighted variance of elite samples across MPPI iterations.
    """

    def __init__(
        self,
        sampling_cfg: 'PolicySamplingCfg',
        num_envs: int,
        dtype = torch.float32,
        device = torch.device("cuda"),
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.cfg = sampling_cfg

        self.nu = self.cfg.action_dim
        self.K = self.cfg.num_rollouts
        self._K_policy = int(self.K * self.cfg.policy_proportion)
        self._K_noise = self.K - self._K_policy
        self.T = self.cfg.num_timesteps
        self.num_envs = num_envs

        self._noise_std = self.cfg.noise

        self._iter_mean = torch.zeros((num_envs, self.T, self.nu), device=device, dtype=dtype)
        self._iter_std = torch.full((num_envs, self.T, self.nu), self.cfg.max_std, device=device, dtype=dtype)

        self.max_control = self.cfg.action_lims
        if self.max_control is not None:
            self.max_control = torch.tensor(self.max_control, device=self.device, dtype=self.dtype)

        # Create policy network
        output_dim = self.T * self.nu * 2  # mean and log std for each timestep and control dimension
        self.policy = PolicyNetwork(
            model=resolve_obj(self.cfg.model_factory)(
                input_dim=self.cfg.state_dim,
                output_dim=output_dim,  # mean and log std
                **self.cfg.model_kwargs,
            ),
            min_std=self.cfg.min_std,
            max_std=self.cfg.max_std,
            horizon=self.T,
            action_dim=self.nu,
            device=device,
            dtype=dtype
        )
        self.policy.to(device=device, dtype=dtype)

    def reset_iter_state(self, prev_controls: Optional[torch.Tensor] = None, reset_idx: Optional[torch.Tensor] = None):
        """
        Reset iterative state (mean/std) at the start of a new MPC step.
        Called at iter=0 of each optimization cycle.

        :param prev_controls: Initial mean from previous timestep [num_envs, T, nu]
        :param reset_idx: Optional indices to reset (for partial resets)
        """
        if reset_idx is None:
            self._iter_std[:] = self.cfg.max_std
            if prev_controls is not None:
                self._iter_mean[:] = prev_controls
            else:
                self._iter_mean[:] = 0.
        else:
            self._iter_std[reset_idx] = self.cfg.max_std
            if prev_controls is not None:
                self._iter_mean[reset_idx] = prev_controls[reset_idx]
            else:
                self._iter_mean[reset_idx] = 0.

    def update_plan_dist_from_elites(
        self,
        elite_actions: torch.Tensor,
        score: torch.Tensor,
    ):
        """
        Update iterative mean and std based on elite samples (TD-MPC2 style).

        :param elite_actions: Elite action sequences [num_envs, num_elites, T, nu]
        :param score: Normalized weights for elites [num_envs, num_elites]
        """
        # Compute weighted mean: [num_envs, T, nu]
        # score: [num_envs, num_elites] -> [num_envs, num_elites, 1, 1]
        score_expanded = score.unsqueeze(-1).unsqueeze(-1)

        # Weighted mean across elites
        mean = (score_expanded * elite_actions).sum(dim=1)  # [num_envs, T, nu]

        # Weighted variance: E[(x - mean)^2]
        # elite_actions: [num_envs, num_elites, T, nu], mean: [num_envs, T, nu]
        variance = (score_expanded * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)  # [num_envs, T, nu]
        std = variance.sqrt()

        self._iter_mean[:] = mean
        self._iter_std[:] = std.clamp(min=self.cfg.min_std, max=self.cfg.max_std)

    @torch.compile
    def sample(
        self, state: torch.Tensor, iter: int,
        prev_controls: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Sample controls combining policy network and Gaussian noise.

        When use_iterative_std=True (default), uses TD-MPC2-style iterative std:
        - On iter=0: resets _iter_std to max_std and _iter_mean to prev_controls
        - Uses _iter_std (updated by update_std_from_elites) for noise scaling

        When use_iterative_std=False, uses the original annealing schedule.

        :param state: torch.Tensor of shape (num_envs, state_dim)
        :param prev_controls: torch.Tensor of shape (num_envs, T, nu)
        :param iter: current optimization iteration for annealing
        :param use_iterative_std: whether to use TD-MPC2-style iterative std
        :return: controls tensor of shape (num_envs, K, T, nu)
        """
        assert self.policy is not None, "Policy model not set. Call set_policy() first."
        if prev_controls is None:
            prev_controls = torch.zeros((self.num_envs, self.T, self.nu), device=self.device, dtype=self.dtype)

        #  Added Noise
        _noise = torch.randn((self.num_envs, self._K_noise, self.T, self.nu),
                             device=self.device, dtype=self.dtype)

        # _iter_std: [num_envs, T, nu] -> [num_envs, 1, T, nu] for broadcasting
        _noise = _noise * self._iter_std.unsqueeze(1)  # [num_envs, K_noise, T, nu]

        # Use _iter_mean as the center for sampling
        noise_controls = self._iter_mean.unsqueeze(1) + _noise  # [num_envs, K_noise, T, nu]

        if self.max_control is not None:
            noise_controls = torch.tanh(noise_controls)

        # Sample from policy network (already squashed)
        policy_plans = self.policy.plan(state, shape=torch.Size([self._K_policy]))  # [_K_policy, num_envs, T, nu]
        policy_plans = policy_plans.transpose(0, 1)  # [num_envs, _K_policy, T, nu]

        controls = torch.cat([noise_controls, policy_plans], dim=-3)  # [num_envs, K, T, nu]

        return controls
