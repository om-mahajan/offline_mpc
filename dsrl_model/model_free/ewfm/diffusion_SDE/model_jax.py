"""
Complete JAX/Flax implementations of:
- FlaxTwinQ (replaces TwinQ)
- FlaxVNetwork (replaces VNetwork)  
- FlaxScoreNet (replaces ScoreNet)

All models follow the same API as PyTorch versions in model.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple, Optional
from functools import partial


# =============================================================================
# Utility Modules
# =============================================================================

class FlaxGaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    
    Input:  x: [B] time values in [0, 1]
    Output: [B, embed_dim] Fourier features
    """
    embed_dim: int
    scale: float = 30.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # W: [embed_dim // 2] - fixed random weights
        W = self.param(
            'W',
            lambda rng, shape: jax.random.normal(rng, shape) * self.scale,
            (self.embed_dim // 2,)
        )
        W = jax.lax.stop_gradient(W)  # Non-trainable
        
        # x_proj: [B, embed_dim // 2]
        x_proj = x[..., None] * W[None, :] * 2 * jnp.pi
        # Output: [B, embed_dim]
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class FlaxResidualBlock(nn.Module):
    """
    Residual block with time conditioning.
    
    Input:  x: [B, input_dim], t_embed: [B, t_dim]
    Output: [B, output_dim]
    """
    output_dim: int
    t_dim: int = 128
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t_embed: jnp.ndarray) -> jnp.ndarray:
        input_dim = x.shape[-1]
        
        # Time MLP: [B, t_dim] -> [B, output_dim]
        t_out = nn.silu(t_embed)
        t_out = nn.Dense(self.output_dim, name='time_dense')(t_out)
        
        # Dense layers: [B, input_dim] -> [B, output_dim]
        h1 = nn.Dense(self.output_dim, name='dense1')(x)
        h1 = nn.silu(h1)
        h1 = h1 + t_out  # Add time conditioning
        
        h2 = nn.Dense(self.output_dim, name='dense2')(h1)
        h2 = nn.silu(h2)
        
        # Residual connection
        if input_dim != self.output_dim:
            x = nn.Dense(self.output_dim, name='modify_x')(x)
        
        return h2 + x


# =============================================================================
# FlaxTwinQ - Twin Q-network
# =============================================================================

class FlaxTwinQ(nn.Module):
    """
    Twin Q-network for robust Q-value estimation.
    Returns min(Q1, Q2) and both Q values.
    
    Input:  state: [B, obs_dim], action: [B, act_dim]
    Output: (q_min: [B], (q1: [B], q2: [B]))
    """
    hidden_dims: Sequence[int] = (256, 256, 256)
    
    @nn.compact
    def __call__(
        self, 
        state: jnp.ndarray, 
        action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        # Concatenate: [B, obs_dim + act_dim]
        x = jnp.concatenate([state, action], axis=-1)
        
        # Q1 network: [B, obs_dim + act_dim] -> [B]
        q1 = x
        for i, dim in enumerate(self.hidden_dims):
            q1 = nn.Dense(dim, name=f'q1_fc{i}')(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(1, name='q1_out')(q1).squeeze(-1)  # [B, 1] -> [B]
        
        # Q2 network: [B, obs_dim + act_dim] -> [B]
        q2 = x
        for i, dim in enumerate(self.hidden_dims):
            q2 = nn.Dense(dim, name=f'q2_fc{i}')(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(1, name='q2_out')(q2).squeeze(-1)  # [B, 1] -> [B]
        
        # Conservative estimate: [B]
        q_min = jnp.minimum(q1, q2)
        return q_min, (q1, q2)


# =============================================================================
# FlaxVNetwork - State Value Network
# =============================================================================

class FlaxVNetwork(nn.Module):
    """
    State value network V(s).
    
    Input:  state: [B, obs_dim]
    Output: v: [B]
    """
    hidden_size: int = 256
    
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        # [B, obs_dim] -> [B, hidden_size]
        h = nn.Dense(self.hidden_size)(state)
        h = nn.relu(h)
        # [B, hidden_size] -> [B, hidden_size]
        h = nn.Dense(self.hidden_size)(h)
        h = nn.relu(h)
        # [B, hidden_size] -> [B]
        v = nn.Dense(1)(h).squeeze(-1)
        return v


# =============================================================================
# FlaxScoreNet - Flow/Score Network for Trajectory Generation
# =============================================================================

class FlaxScoreNet(nn.Module):
    """
    Score/Flow network for trajectory generation.
    Predicts vector field v_theta(x_t, t | condition).
    
    Input:
        x: [B, output_dim] noisy trajectory (flattened actions)
        t: [B] time values in [0, 1]
        condition: [B, obs_dim] conditioning observation (optional)
        
    Output: [B, output_dim] predicted vector field
    """
    output_dim: int
    embed_dim: int = 32
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        t: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        train: bool = True
    ) -> jnp.ndarray:
        # Time embedding: [B] -> [B, embed_dim]
        embed = FlaxGaussianFourierProjection(self.embed_dim)(t)
        embed = nn.Dense(self.embed_dim)(embed)
        
        # Condition embedding (if provided): [B, obs_dim] -> [B, 32]
        if condition is not None:
            cond_embed = nn.Dense(32, name='pre_sort_condition')(condition)
            cond_embed = nn.silu(cond_embed)
            # Concatenate: [B, 32 + embed_dim]
            embed = jnp.concatenate([cond_embed, embed], axis=-1)
        
        # Process time embedding: -> [B, 128]
        embed = nn.Dense(128, name='sort_t_1')(embed)
        embed = nn.silu(embed)
        embed = nn.Dense(128, name='sort_t_2')(embed)
        
        # U-Net style architecture
        # x: [B, output_dim]
        # Down path
        d1 = FlaxResidualBlock(512, name='down_block1')(x, embed)      # [B, 512]
        d2 = FlaxResidualBlock(256, name='down_block2')(d1, embed)     # [B, 256]
        d3 = FlaxResidualBlock(128, name='down_block3')(d2, embed)     # [B, 128]
        
        # Middle
        u3 = FlaxResidualBlock(128, name='middle1')(d3, embed)         # [B, 128]
        
        # Up path with skip connections
        # [B, 128 + 128] -> [B, 256]
        u2 = FlaxResidualBlock(256, name='up_block3')(
            jnp.concatenate([d3, u3], axis=-1), embed
        )
        # [B, 256 + 256] -> [B, 512]
        u1 = FlaxResidualBlock(512, name='up_block2')(
            jnp.concatenate([d2, u2], axis=-1), embed
        )
        
        # Final: [B, 512 + 512] -> [B, output_dim]
        u0 = jnp.concatenate([d1, u1], axis=-1)
        h = nn.Dense(self.output_dim, name='last')(u0)
        
        return h