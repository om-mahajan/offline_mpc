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

#@partial(jax.jit, static_argnames=['num_steps', 'method'])
def integrate_ode(velocity_fn, x, t0, t1, num_steps, method="rk4"):
    """Integrate dx/dt = v(x,t) from t0 → t1 in num_steps."""
    dt = (t1 - t0) / num_steps
    t = t0

    for _ in range(num_steps):
        if method == "euler":
            x = x + dt * velocity_fn(x, t)

        elif method == "heun":
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + dt * k1, t + dt)
            x = x + 0.5 * dt * (k1 + k2)

        else:  # rk4
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        t = t + dt

    return x




# =============================================================================
# Flow Matching Sampling Functions
# =============================================================================

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
"""
def sample_flow_trajectory(
    apply_fn: Callable,
    params: dict,
    condition: jnp.ndarray,
    output_dim: int,
    num_steps: int = 15,
    method: str = 'rk4',
    rng: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:

    B = condition.shape[0]

    # Initial trajectory x0 = zeros
    x0 = jnp.zeros((B, output_dim))

    def velocity_fn(x: jnp.ndarray, t: float) -> jnp.ndarray:
        #'''Compute v_theta(x_t, t | condition)'''

        t_batch = jnp.full((x.shape[0],), t)

        # THIS WAS THE BUG — now fixed:
        return apply_fn(
            {'params': params},   # MUST wrap params
            x,                    # noisy actions: [B, output_dim]
            t_batch,              # [B]
            condition,            # [B, obs_dim]
            False                 # train=False
        )

    # Correct ODE call using fixed signature
    x1 = integrate_ode(
        velocity_fn=velocity_fn,
        x_init=x0,
        t_start=0.0,
        t_end=1.0,
        num_steps=num_steps,
        method=method
    )

    return x1        # [B, output_dim]
"""
def sample_flow_trajectory(apply_fn, params, condition, output_dim,
                           num_steps=15, method="rk4"):

    B = condition.shape[0]
    x0 = jnp.zeros((B, output_dim))
 
    def velocity_fn(x, t):
       
        t_batch = jnp.full((x.shape[0],), t)
        return apply_fn(
            params,
            x,
            t_batch,
            condition,
            False
        )

    # CALL WITH POSITIONAL ARGUMENTS ONLY
    x1 = integrate_ode(velocity_fn, x0, 0.0, 1.0, num_steps, method)

    return x1



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
'''
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

    B = states.shape[0]
    output_dim = horizon * act_dim     # flatten trajectory dim

    traj_flat = sample_flow_trajectory(
        apply_fn=apply_fn,
        params=params,
        condition=states,              # [B, obs_dim]
        output_dim=output_dim,         # H * act_dim
        num_steps=diffusion_steps,
        method=method
    )

    # reshape: [B, H*act_dim] → [B, H, act_dim]
    traj = traj_flat.reshape(B, horizon, act_dim)

    if use_first_action:
        return traj[:, 0, :]           # [B, act_dim]
    
    return traj                        # [B, H, act_dim]
'''

def select_actions_jax(apply_fn, params, states,
                       horizon, act_dim,
                       diffusion_steps=15, method="rk4",
                       use_first_action=True):

    B = states.shape[0]
    output_dim = horizon * act_dim
    
    traj_flat = sample_flow_trajectory(
        apply_fn, params, states, output_dim,
        num_steps=diffusion_steps,
        method=method
    )

    traj = traj_flat.reshape(B, horizon, act_dim)

    return traj[:, 0] if use_first_action else traj
