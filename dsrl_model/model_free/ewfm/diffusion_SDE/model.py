import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from diffusion_SDE import dpm_solver_pytorch
from diffusion_SDE import schedule
from scipy.special import softmax

from diffusion_SDE.riemannian_ode_solver import RiemannianODESolver
from diffusion_SDE.utils.manifolds.manifold import Euclidean
from diffusion_SDE.utils.model_wrapper import ModelWrapper

class FMWrapper(ModelWrapper):
    """Wraps the ScoreNet (flow model) into the ModelWrapper interface expected by RiemannianODESolver."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def forward(self, x, t, **kwargs):
        # Expect x: [B, D], t: scalar tensor or tensor shape [B]
        return self.model(x, t)

def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)


def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class Residual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim=128, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim),SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim),SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    def forward(self, x, t):
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)

class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [state_dim + action_dim, 256, 256, 256, 1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))

class QGPO_Critic(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        # is sdim is 0  means unconditional guidance
        assert sdim > 0
        # only apply to conditional sampling here
        self.q0 = TwinQ(adim, sdim).to(args.device)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False).to(args.device)
        self.discount = 0.99
        
        self.args = args
        self.alpha = args.alpha
class ScoreBase(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.device = args.device if (args is not None and hasattr(args, 'device')) else ('cuda' if torch.cuda.is_available() else 'cpu')


# DPM solver components are optional; only used in diffusion mode.
        self.noise_schedule = None
        self.dpm_solver = None
        if dpm_solver_pytorch is not None and args is not None and hasattr(args, 'schedule'):
            try:
                self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule=args.schedule)
            except Exception:
                self.noise_schedule = None


# marginal_prob_std: callable(t) -> (alpha_t, std_t) OR None for Flow Matching mode
        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.condition = None


    def maybe_init_dpm_solver(self):
# initialize dpm solver only if marginal_prob_std is provided and dpm solver library available
        if self.marginal_prob_std is None:
            return
        if dpm_solver_pytorch is None:
            return
        if self.noise_schedule is None and self.args is not None and hasattr(self.args, 'schedule'):
            self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule=self.args.schedule)
        if self.dpm_solver is None and self.noise_schedule is not None:
# create solver using a wrapper that expects diffusion-mode score
            self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule, predict_x0=True)


    def forward_dmp_wrapper_fn(self, x, t):
# This wrapper is used by the DPM_Solver and expects the diffusion-mode score
        if self.marginal_prob_std is None:
            raise RuntimeError("DPM wrapper called but marginal_prob_std is None. DPM-Solver available only in diffusion mode.")
# In diffusion mode, self(x,t) must return the score s = h / sigma
        score = self(x, t)
        sigma = self.marginal_prob_std(t)[1]
    # return -score * sigma => -h (consistent with earlier wrapper behavior)
        return - score * sigma[..., None]


    def dpm_wrapper_sample(self, dim, batch_size, is_numpy=True, **kwargs):
# Only valid in diffusion mode
        if self.marginal_prob_std is None:
            raise RuntimeError("dpm_wrapper_sample() called in Flow Matching mode. DPM-Solver sampling is only valid in diffusion mode.")
        self.maybe_init_dpm_solver()
        if self.dpm_solver is None:
            raise RuntimeError("DPM solver not available or not initialized.")
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            sample = self.dpm_solver.sample(init_x, **kwargs)
            return sample.cpu().numpy() if is_numpy else sample


    def forward(self, x, t, condition=None):
        raise NotImplementedError


    def select_actions(self, states, diffusion_steps=15):
        multiple_input = True
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            else:
                states = states.to(self.device)


            if states.dim() == 1:
                states = states.unsqueeze(0)
                multiple_input = False
            num_states = states.shape[0]


# If in diffusion mode, use DPM-Solver
            if self.marginal_prob_std is not None:
                self.condition = states
                self.maybe_init_dpm_solver()
                if self.dpm_solver is None:
                    raise RuntimeError("DPM solver not initialized; cannot sample.")
                results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=2)
                actions = results.reshape(num_states, self.output_dim).copy()
                self.condition = None
            else:
# Flow Matching mode: perform simple Euler integration of the learned vector field.
# IMPORTANT: This is a simple fallback sampler for quick testing. For accurate sampling, integrate the ODE using
# a proper ODE solver (e.g., RK4 or adaptive solvers) with v_theta as the vector field.
                self.condition = states
                B = states.shape[0]
                D = self.output_dim  # flattened trajectory dimension (H * act_dim)

                # initial condition: zeros (deterministic conditional policy)
                x0 = torch.zeros(B, D, device=self.device)

                # wrap model and construct solver
                velocity = FMWrapper(self)
                manifold = Euclidean()
                solver = RiemannianODESolver(manifold=manifold, velocity_model=velocity)

                # integration hyperparams
                # diffusion_steps used as number of ODE steps; you pass it from call
                step_size = 1.0 / float(diffusion_steps)
                time_grid = torch.tensor([0.0, 1.0], device=self.device)

                # call solver: returns x(t=1) shape [B, D]
                x1 = solver.sample(
                    x_init=x0,
                    step_size=step_size,
                    projx=False,
                    proju=False,
                    method="euler",
                    time_grid=time_grid,
                    return_intermediates=False,
                    verbose=False,
                    enable_grad=False,
                )

                # reshape to actions; ensure numpy for compatibility with existing code
                actions = x1.reshape(B, D).detach().cpu().numpy()
                self.condition = None


        out_actions = [actions[i] for i in range(actions.shape[0])] if multiple_input else actions[0]
        return out_actions


    def sample_and_logprob(self, states, diffusion_steps=15, hutchinson_samples=1):
        """
        Sample actions and compute log-probability for SAC-style entropy.
        Uses OT flow matching with Hutchinson trace estimator for divergence.
        
        Args:
            states: [B, obs_dim] conditioning states
            diffusion_steps: number of Euler steps for ODE integration
            hutchinson_samples: number of random vectors for trace estimation
        
        Returns:
            actions: [B, act_dim] sampled actions (detached)
            logp: [B] log-probabilities (detached)
        """
        # Detach states to avoid graph issues
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        else:
            states = states.to(self.device).detach()
        
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        B = states.shape[0]
        D = self.output_dim
        
        # Initial sample from base distribution: x0 ~ N(0, I)
        x = torch.randn(B, D, device=self.device)
        
        # Log-prob of base distribution: log p(x0) = -0.5 * ||x0||^2 - D/2 * log(2π)
        log_p = -0.5 * (x ** 2).sum(dim=-1) - 0.5 * D * np.log(2 * np.pi)
        
        dt = 1.0 / diffusion_steps
        self.condition = states
        
        # Enable gradients for Hutchinson estimator even if called inside no_grad context
        with torch.enable_grad():
            for step in range(diffusion_steps):
                t_val = step * dt
                t = torch.full((B,), t_val, device=self.device)
                
                # Detach and clone to create fresh leaf tensor with gradients
                x_grad = x.detach().clone().requires_grad_(True)
                v = self.forward(x_grad, t)
                
                # Hutchinson trace estimator: div(v) ≈ E[ε^T ∇v ε]
                div_v = torch.zeros(B, device=self.device)
                for h_idx in range(hutchinson_samples):
                    eps = torch.randn_like(x_grad)
                    vjp = torch.autograd.grad(
                        outputs=v, inputs=x_grad,
                        grad_outputs=eps,
                        create_graph=False, 
                        retain_graph=(h_idx < hutchinson_samples - 1)
                    )[0]
                    div_v = div_v + (vjp * eps).sum(dim=-1)
                div_v = div_v / hutchinson_samples
                
                # Update log-prob
                log_p = log_p - div_v.detach() * dt
                x = x + v.detach() * dt
        
        self.condition = None
        return x.detach(), log_p.detach()

    def sample_and_logprob_fast(self, states, diffusion_steps=10):
        """
        Fast approximate sampling with log-prob.
        Uses simple approximation: logp ≈ -0.5 * ||x0||^2 (ignores Jacobian)
        Much faster - use for training, full version for evaluation.
        
        Args:
            states: [B, obs_dim] conditioning states
            diffusion_steps: number of Euler steps
        
        Returns:
            actions: [B, act_dim] sampled actions
            logp: [B] approximate log-probabilities
        """
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            else:
                states = states.to(self.device)
            
            if states.dim() == 1:
                states = states.unsqueeze(0)
            
            B = states.shape[0]
            D = self.output_dim
            
            # Sample from base distribution
            x0 = torch.randn(B, D, device=self.device)
            
            # Track initial log-prob (OT flows are approximately volume-preserving)
            log_p0 = -0.5 * (x0 ** 2).sum(dim=-1)
            
            x = x0
            dt = 1.0 / diffusion_steps
            self.condition = states
            
            for step in range(diffusion_steps):
                t = torch.full((B,), step * dt, device=self.device)
                v = self.forward(x, t)
                x = x + v * dt
            
            self.condition = None
            return x, log_p0

    def sample_actions_fast(self, states, diffusion_steps=15):
        """
        Fast action sampling without log-prob computation.
        Uses Euler integration for speed.
        
        Args:
            states: [B, obs_dim] conditioning states
            diffusion_steps: number of Euler steps
        
        Returns:
            actions: [B, act_dim] sampled actions
        """
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            else:
                states = states.to(self.device)
            
            if states.dim() == 1:
                states = states.unsqueeze(0)
            
            B = states.shape[0]
            D = self.output_dim
            
            # Start from noise
            x = torch.randn(B, D, device=self.device)
            
            dt = 1.0 / diffusion_steps
            self.condition = states
            
            for step in range(diffusion_steps):
                t = torch.full((B,), step * dt, device=self.device)
                v = self.forward(x, t)
                x = x + v * dt
            
            self.condition = None
            return x

    @torch.no_grad()
    def sample(self, states, sample_per_state=16, diffusion_steps=15, is_numpy=True):
        num_states = states.shape[0]

        # Convert to tensor
        states = torch.FloatTensor(states).to(self.device) if is_numpy else states.to(self.device)

        # Repeat conditioning for sampling multiple trajectories
        states = torch.repeat_interleave(states, sample_per_state, dim=0)

        # Store conditioning for the model
        self.condition = states

        # Case 1: Diffusion mode (legacy)
        if self.marginal_prob_std is not None:
            results = self.dpm_wrapper_sample(
                self.output_dim,
                batch_size=states.shape[0],
                steps=diffusion_steps,
                order=2,
                is_numpy=is_numpy
            )

        # Case 2: Flow Matching ODE mode
        else:
            # Build Euclidean manifold & wrapped velocity field
            manifold = Euclidean()
            velocity = FMWrapper(self)
            solver = RiemannianODESolver(manifold, velocity)

            # Initial x0 (deterministic FM policy)
            x0 = torch.zeros(states.shape[0], self.output_dim, device=self.device)

            # ODE step size
            step_size = 1.0 / diffusion_steps

            # Integrate from t=0 → t=1
            xt = solver.sample(
                x_init=x0,
                step_size=step_size,
                method="rk4",         # best stability
                projx=True,
                proju=True,
                time_grid=torch.tensor([0., 1.], device=self.device),
            )

            results = xt.detach().cpu().numpy() if is_numpy else xt

        # reshape to [num_states, sample_per_state, output_dim]
        actions = results.reshape(num_states, sample_per_state, self.output_dim)

        if is_numpy:
            actions = actions.copy()

        self.condition = None
        return actions


class ScoreNet(ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim-output_dim, 32), SiLU())
        self.sort_t = nn.Sequential(
            nn.Linear(64, 128),
            SiLU(),
            nn.Linear(128, 128),
        )
        self.down_block1 = Residual_Block(output_dim, 512)
        self.down_block2 = Residual_Block(512, 256)
        self.down_block3 = Residual_Block(256, 128)
        self.middle1 = Residual_Block(128, 128)
        self.up_block3 = Residual_Block(256, 256)
        self.up_block2 = Residual_Block(512, 512)
        self.last = nn.Linear(1024, output_dim)
    def forward(self, x, t, condition=None):
        embed = self.embed(t)
        if condition is not None:
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        else:
            if self.condition is None:
                raise RuntimeError("No condition available for ScoreNet. Provide `condition` or set `self.condition` before calling forward.")
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition]*x.shape[0])
            else:
                raise RuntimeError("Condition batch-size mismatch")
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        embed = self.sort_t(embed)
        d1 = self.down_block1(x, embed)
        d2 = self.down_block2(d1, embed)
        d3 = self.down_block3(d2, embed)
        u3 = self.middle1(d3, embed)
        u2 = self.up_block3(torch.cat([d3, u3], dim=-1), embed)
        u1 = self.up_block2(torch.cat([d2, u2], dim=-1), embed)
        u0 = torch.cat([d1, u1], dim=-1)
        h = self.last(u0)
        self.h = h
# Mode switch: if marginal_prob_std is None -> Flow Matching mode (return vector-field directly)
        if self.marginal_prob_std is None:
            return h
# Otherwise diffusion mode: return score = h / sigma_t
        sigma = self.marginal_prob_std(t)[1]
        return h / sigma[..., None]
    @torch.no_grad()
    
    @torch.no_grad()
    def sample_trajectory(self, obs, diffusion_steps=15, horizon=None, act_dim=None):
    
        self.eval()

        # Convert obs to tensor and batch it
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)
        else:
            obs = obs.to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        B = obs.shape[0]
        self.condition = obs

        # Determine trajectory dimension
        traj_dim = self.output_dim

        # Initial x0 (deterministic FM policy)
        x0 = torch.zeros(B, traj_dim, device=self.device)

        # Build solver
        manifold = Euclidean()
        velocity = FMWrapper(self)
        solver = RiemannianODESolver(manifold, velocity)

        # Integrate ODE from t=0 → 1
        x1 = solver.sample(
            x_init=x0,
            step_size=1.0 / diffusion_steps,
            method="rk4",
            projx=True,
            proju=True,
            time_grid=torch.tensor([0., 1.], device=self.device)
        )

        self.condition = None
        return x1        # shape [B, traj_dim]

    @torch.no_grad()
    def select_trajectory_actions(self, states, diffusion_steps=15, horizon=5, act_dim=None, use_first_action=True):

        traj_flat = self.sample_trajectory(
            states,
            diffusion_steps=diffusion_steps,
            horizon=horizon,
            act_dim=act_dim
        )

        if act_dim is None:
            act_dim = self.output_dim // horizon

        B = traj_flat.shape[0]
        traj = traj_flat.reshape(B, horizon, act_dim)

        if use_first_action:
            return traj[:, 0, :].cpu().numpy()
        else:
            return traj.cpu().numpy()
