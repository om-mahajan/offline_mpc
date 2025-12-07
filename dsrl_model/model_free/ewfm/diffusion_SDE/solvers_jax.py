"""
JAX implementations of ODE solvers for Flow Matching.
Replaces RiemannianODESolver from riemannian_ode_solver.py
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional
from functools import partial


# =============================================================================
# Euclidean Manifold Operations
# =============================================================================

def euclidean_proj_x(x: jnp.ndarray) -> jnp.ndarray:
    """Project point onto Euclidean manifold (identity)."""
    return x


# =============================================================================
# ODE Step Functions
# =============================================================================

def euler_step(
    velocity_fn: Callable,
    x: jnp.ndarray,
    t: float,
    dt: float
) -> jnp.ndarray:
    """
    Single Euler integration step.
    
    Args:
        velocity_fn: (x, t) -> v, vector field function
        x: [B, D] current state
        t: Current time (scalar)
        dt: Time step (scalar)
        
    Returns:
        x_next: [B, D] state at t + dt
    """
    v = velocity_fn(x, t)
    return x + dt * v


def rk4_step(
    velocity_fn: Callable,
    x: jnp.ndarray,
    t: float,
    dt: float
) -> jnp.ndarray:
    """
    Single RK4 integration step.
    
    Args:
        velocity_fn: (x, t) -> v, vector field function
        x: [B, D] current state
        t: Current time (scalar)
        dt: Time step (scalar)
        
    Returns:
        x_next: [B, D] state at t + dt
    """
    k1 = velocity_fn(x, t)
    k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = velocity_fn(x + dt * k3, t + dt)
    
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def heun_step(
    velocity_fn: Callable,
    x: jnp.ndarray,
    t: float,
    dt: float
) -> jnp.ndarray:
    """
    Single Heun integration step (2nd order).
    
    Args:
        velocity_fn: (x, t) -> v, vector field function
        x: [B, D] current state
        t: Current time (scalar)
        dt: Time step (scalar)
        
    Returns:
        x_next: [B, D] state at t + dt
    """
    v1 = velocity_fn(x, t)
    x_euler = x + dt * v1
    v2 = velocity_fn(x_euler, t + dt)
    
    return x + 0.5 * dt * (v1 + v2)


# =============================================================================
# Main ODE Solver
# =============================================================================

@partial(jax.jit, static_argnames=['num_steps', 'method'])
def integrate_ode(
    velocity_fn: Callable,
    x_init: jnp.ndarray,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 15,
    method: str = 'rk4'
) -> jnp.ndarray:
    """
    Integrate ODE from t_start to t_end.
    
    Args:
        velocity_fn: (x, t) -> v, the vector field
        x_init: [B, D] initial state at t=t_start
        t_start: Start time (typically 0)
        t_end: End time (typically 1)
        num_steps: Number of integration steps
        method: 'euler', 'heun', or 'rk4'
        
    Returns:
        x_final: [B, D] state at t=t_end
    """
    dt = (t_end - t_start) / num_steps
    
    if method == 'euler':
        step_fn = euler_step
    elif method == 'heun':
        step_fn = heun_step
    elif method == 'rk4':
        step_fn = rk4_step
    else:
        raise ValueError(f"Unknown method: {method}")
    
    timesteps = jnp.linspace(t_start, t_end - dt, num_steps)
    
    def body_fn(x, t):
        x_next = step_fn(velocity_fn, x, t, dt)
        x_next = euclidean_proj_x(x_next)
        return x_next, None
    
    x_final, _ = jax.lax.scan(body_fn, x_init, timesteps)
    
    return x_final


# =============================================================================
# Flow Matching Sampling Functions
# =============================================================================

def sample_flow_trajectory(
    apply_fn: Callable,
    params: dict,
    condition: jnp.ndarray,
    output_dim: int,
    num_steps: int = 15,
    method: str = 'rk4',
    rng: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Sample trajectory using Flow Matching ODE integration.
    
    Args:
        apply_fn: Model's apply function
        params: Model parameters
        condition: [B, obs_dim] conditioning observations
        output_dim: Trajectory dimension (horizon * act_dim)
        num_steps: Number of ODE integration steps
        method: Integration method ('euler', 'heun', 'rk4')
        rng: Optional RNG key (not used for deterministic sampling)
        
    Returns:
        x1: [B, output_dim] sampled trajectories at t=1
    """
    B = condition.shape[0]
    
    # Initial condition: zeros
    # Shape: [B, output_dim]
    x0 = jnp.zeros((B, output_dim))
    
    def velocity_fn(x: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Velocity field v_theta(x, t | condition).
        
        Args:
            x: [B, output_dim] current state
            t: scalar time
            
        Returns:
            v: [B, output_dim] velocity
        """
        t_batch = jnp.full((x.shape[0],), t)  # [B]
        return apply_fn(params, x, t_batch, condition=condition, train=False)
    
    # Integrate from t=0 to t=1
    # Output: [B, output_dim]
    x1 = integrate_ode(
        velocity_fn=velocity_fn,
        x_init=x0,
        t_start=0.0,
        t_end=1.0,
        num_steps=num_steps,
        method=method
    )
    
    return x1


def select_actions_jax(
    apply_fn: Callable,
    params: dict,
    states: jnp.ndarray,
    horizon: int,
    act_dim: int,
    diffusion_steps: int = 15,
    method: str = 'rk4',
    use_first_action: bool = True
) -> jnp.ndarray:
    """
    Select actions for given states using trained flow model.
    
    Args:
        apply_fn: Model's apply function
        params: Model parameters
        states: [B, obs_dim] current observations
        horizon: Planning horizon H
        act_dim: Action dimension
        diffusion_steps: Number of ODE steps
        method: Integration method
        use_first_action: If True, return only first action
        
    Returns:
        If use_first_action: [B, act_dim]
        Else: [B, horizon, act_dim]
    """
    output_dim = horizon * act_dim
    
    # Sample trajectory: [B, horizon * act_dim]
    traj_flat = sample_flow_trajectory(
        apply_fn=apply_fn,
        params=params,
        condition=states,
        output_dim=output_dim,
        num_steps=diffusion_steps,
        method=method
    )
    
    # Reshape: [B, horizon * act_dim] -> [B, horizon, act_dim]
    B = traj_flat.shape[0]
    traj = traj_flat.reshape(B, horizon, act_dim)
    
    if use_first_action:
        return traj[:, 0, :]  # [B, act_dim]
    else:
        return traj  # [B, horizon, act_dim]