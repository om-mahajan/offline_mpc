"""
Custom TrainState classes for JAX training.
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
from typing import Any
import optax


class QTrainState(train_state.TrainState):
    """
    TrainState for Q-network with target parameters.
    
    Attributes:
        params: Online network parameters
        target_params: Target network parameters (for stable Q-learning)
    """
    target_params: Any = None


class FlowTrainState(train_state.TrainState):
    """TrainState for flow model."""
    pass


def create_q_train_state(
    rng: jax.random.PRNGKey,
    model,
    obs_dim: int,
    act_dim: int,
    learning_rate: float,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 1.0
) -> QTrainState:
    """
    Initialize Q-network with target parameters.
    
    Args:
        rng: JAX random key
        model: FlaxTwinQ model instance
        obs_dim: Observation dimension
        act_dim: Action dimension
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for AdamW
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        QTrainState with initialized params and target_params
    """
    # Dummy inputs for initialization
    # Shape: [1, obs_dim], [1, act_dim]
    dummy_obs = jnp.ones((1, obs_dim))
    dummy_act = jnp.ones((1, act_dim))
    
    params = model.init(rng, dummy_obs, dummy_act)
    
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay)
    )
    
    return QTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        target_params=params  # Initialize target = online
    )


def create_v_train_state(
    rng: jax.random.PRNGKey,
    model,
    obs_dim: int,
    learning_rate: float,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 1.0
) -> train_state.TrainState:
    """
    Initialize V-network.
    
    Args:
        rng: JAX random key
        model: FlaxVNetwork model instance
        obs_dim: Observation dimension
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_grad_norm: Gradient clipping threshold
        
    Returns:
        Standard TrainState for V-network
    """
    # Shape: [1, obs_dim]
    dummy_obs = jnp.ones((1, obs_dim))
    params = model.init(rng, dummy_obs)
    
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def create_flow_train_state(
    rng: jax.random.PRNGKey,
    model,
    obs_dim: int,
    traj_dim: int,
    learning_rate: float,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 1.0
) -> FlowTrainState:
    """
    Initialize flow model.
    
    Args:
        rng: JAX random key
        model: FlaxScoreNet model instance
        obs_dim: Observation/condition dimension
        traj_dim: Trajectory dimension (horizon * act_dim)
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_grad_norm: Gradient clipping threshold
        
    Returns:
        FlowTrainState for flow model
    """
    # Shapes: x=[1, traj_dim], t=[1], condition=[1, obs_dim]
    dummy_x = jnp.ones((1, traj_dim))
    dummy_t = jnp.ones((1,))
    dummy_cond = jnp.ones((1, obs_dim))
    
    params = model.init(rng, dummy_x, dummy_t, dummy_cond)
    
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay)
    )
    
    return FlowTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )