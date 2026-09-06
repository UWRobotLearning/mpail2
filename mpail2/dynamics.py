"""Dynamics models for MPAIL."""

import torch
from typing import TYPE_CHECKING

from mpail2.utils import resolve_obj

if TYPE_CHECKING:
    from .configs.cfgs import DynamicsCfg


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Projects a batch of latents onto random 1D directions and penalizes deviation
    of each projection's empirical characteristic function from a standard
    Gaussian's (via a differentiable normality test, quadrature-integrated over
    `knots`). Minimizing this pushes variance to spread across many random
    directions instead of concentrating in a few (representation/dimensional
    collapse) — ported from mpail-research fix-main #18.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: Tensor of shape (T, B, D).
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (proj @ A).unsqueeze(-1) * self.t.to(dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - self.phi.to(dtype=proj.dtype)).square()
        err = err + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(dtype=proj.dtype)) * proj.size(-2)
        return statistic.mean()


class Dynamics(torch.nn.Module):
    """
    Latent dynamics model. Propagates latent states through a feedforward network.
    Expects already-encoded latent states as input.
    """
    def __init__(
        self,
        dynamics_cfg: 'DynamicsCfg',
        num_envs: int,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        super(Dynamics, self).__init__()
        self.dtype = dtype
        self.d = device
        self.cfg = dynamics_cfg
        self.num_envs = num_envs

        # Feedforward dynamics network
        self.model = resolve_obj(self.cfg.model_factory)(
            input_dim=self.cfg.latent_dim + self.cfg.action_dim,
            output_dim=self.cfg.latent_dim,
            **self.cfg.model_kwargs,
        )

    def step(self, current_latent: torch.Tensor, control: torch.Tensor):
        """
        Compute next latent state using feedforward network.

        Args:
            current_latent: Current latent state of shape (..., latent_dim)
            control: Control input of shape (..., action_dim)

        Returns:
            next_latent: Next latent state of shape (..., latent_dim)
        """
        model_input = torch.cat((current_latent, control), dim=-1)
        return self.model(model_input)

    @torch.compile
    def forward(
        self,
        z0: torch.Tensor,
        controls: torch.Tensor,
    ):
        """
        Forward pass for dynamics model. Rolls out latent trajectory from initial latent state.

        Args:
            z0: initial latent state of shape (B, ..., latent_dim)
            controls: controls of shape (B, ..., T, a_dim)
            tf_mask: mask for teacher forcing of shape (B, ..., T)
            latent_gt: ground truth latent states for teacher forcing of shape (B, ..., T, latent_dim)

        Returns:
            zs: latent trajectory of shape (B, ..., T+1, latent_dim)
        """
        controls = controls.to(device=self.d, dtype=self.dtype)
        T = controls.shape[-2]

        # Match batch dimensions: if controls is (B, ..., T, a_dim), expand z0 to (B, ..., latent_dim)
        batch_shape = controls.shape[:-2]  # (B, ...)

        num_extra_dims = len(batch_shape) - (z0.ndim - 1)
        z0_expanded = z0.view(*z0.shape[:-1], *([1] * max(0, num_extra_dims)), z0.shape[-1])
        z0_expanded = z0_expanded.expand(*batch_shape, z0.shape[-1])

        zs_list = [z0_expanded]

        for i in range(T):
            current_latent = zs_list[-1]

            # Step dynamics in latent space
            next_latent = self.step(current_latent, controls[..., i, :])
            zs_list.append(next_latent)

        # Stack all latent states
        return torch.stack(zs_list, dim=-2)
