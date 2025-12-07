"""
JAX utility functions for training.
"""

import jax
import jax.numpy as jnp
from typing import Any

PyTree = Any


def update_target_params(
    online_params: PyTree,
    target_params: PyTree,
    tau: float
) -> PyTree:
    """
    Soft update of target network parameters.
    
    target = tau * online + (1 - tau) * target
    
    Args:
        online_params: Current online network parameters
        target_params: Current target network parameters
        tau: Soft update coefficient (typically 0.005)
        
    Returns:
        Updated target parameters (same structure as input)
    """
    return jax.tree_map(
        lambda online, target: tau * online + (1 - tau) * target,
        online_params,
        target_params
    )


def normalize_observation_jax(
    obs: jnp.ndarray,
    mu_obs: jnp.ndarray,
    std_obs: jnp.ndarray,
    eps: float = 1e-6
) -> jnp.ndarray:
    """
    Normalize observations.
    
    Args:
        obs: Observations of any shape [..., obs_dim]
        mu_obs: Mean [obs_dim]
        std_obs: Std [obs_dim]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized observations, same shape as input
    """
    if mu_obs is None:
        return obs
    return (obs - mu_obs) / (std_obs + eps)