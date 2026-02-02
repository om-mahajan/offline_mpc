#!/usr/bin/env python3
"""
JAX/Flax version of ipltwin_V5.py
Flow Matching (FM-OT) training with TwinQ/V preference learning.

This script uses:
- diffusion_SDE/model_jax.py: FlaxTwinQ, FlaxVNetwork, FlaxScoreNet
- diffusion_SDE/solvers_jax.py: ODE integration for sampling
- utils/train_state.py: QTrainState, FlowTrainState
- utils/jax_utils.py: update_target_params, normalize_observation_jax
"""

import os
import os.path as osp
import sys
import time
from functools import partial
from typing import Dict, Tuple, Optional, Any

import numpy as np

# JAX imports
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from tqdm import tqdm

# Add the repo root to path
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../..')))

# DSRL imports
import gymnasium as gym
import dsrl
import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium

# Local JAX modules
from diffusion_SDE.model_jax import FlaxTwinQ, FlaxVNetwork, FlaxScoreNet
from diffusion_SDE.solvers_jax import select_actions_jax
from utils.train_state import (
    QTrainState, FlowTrainState,
    create_q_train_state, create_v_train_state, create_flow_train_state
)
from utils.jax_utils import update_target_params, normalize_observation_jax

# Utilities
from dsrl_model.utils.logger import EpochLogger

# Dataset functions
from dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
    get_normalized_data
)

EP = 1e-6

# =============================================================================
# Default Configuration
# =============================================================================
default_cfg = {
    # Logging / checkpoint
    "log_freq": int(1e3),
    "save_freq": int(2e4),
    "eval_episode_freq": 10,
    "hidden_sizes": [256, 256],
    "max_grad_norm": 1.0,
    # Optimization
    "lr": 3e-4,
    "weight_decay": 1e-5,
    # Diffusion
    "diffusion_steps": 15,
    "train_horizon": 5,
    # IPL / Q pretrain
    "q_pretrain_iterations": int(1.5e5),
    "q_lr": 3e-4,
    "q_hidden": 256,
    # Value network (V)
    "v_lr": 3e-4,
    "v_hidden": 256,
    "v_updates_per_q_update": 1,
    # Q regularizer
    "lambda_q_reg": 1e-2,
    # Weight temperature
    "energy_alpha": 3.0,
    "cost_weight_temp": 1.0,
    # Target network update
    "target_update_freq": 10,
    "target_tau": 0.005,
    # Iterations
    "flow_train_iterations": int(1e5),
    "batch_size": 128,
    # Gamma
    "gamma": 0.99,
    # V updates
    "v_use_neg_in_updates": True,
    # DSRL dataset config
    "density": 1.0,
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
    "non_pref_noise": 0.0,
    "num_folds": 1,
    # Flow Matching specific
    "sigma_min": 0.01,
}


# =============================================================================
# Data Sampling (JAX-compatible, uses numpy for indexing)
# =============================================================================

def sample_trajectory_batch_jax(
    neg_data: dict,
    union_data: dict,
    batch_size: int,
    train_horizon: int,
    rng: jax.random.PRNGKey,
    mu_obs: Optional[jnp.ndarray] = None,
    std_obs: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """
    Sample trajectory batches from negative and union data.
    
    Args:
        neg_data: Dict with 'observations', 'actions' of shape [num_traj, ep_len, dim]
        union_data: Dict with 'observations', 'actions', 'rewards'
        batch_size: Number of trajectory segments to sample (B)
        train_horizon: Length of each trajectory segment (H)
        rng: JAX random key
        mu_obs: Observation mean [obs_dim] for normalization
        std_obs: Observation std [obs_dim] for normalization
        
    Returns:
        neg_obs: [H, B, obs_dim]
        neg_acts: [H, B, act_dim]
        union_obs: [H, B, obs_dim]
        union_acts: [H, B, act_dim]
        union_rew: [H, B, 1]
        rng: Updated random key
    """
    # Convert to numpy for efficient random access
    neg_obs_arr = np.asarray(neg_data['observations'])
    neg_act_arr = np.asarray(neg_data['actions'])
    union_obs_arr = np.asarray(union_data['observations'])
    union_act_arr = np.asarray(union_data['actions'])
    union_rew_arr = np.asarray(union_data['rewards'])
    
    n_neg, neg_len = neg_obs_arr.shape[0], neg_obs_arr.shape[1]
    n_union, union_len = union_obs_arr.shape[0], union_obs_arr.shape[1]
    
    # Validate trajectory lengths
    assert neg_len >= train_horizon, f"Negative trajectory length {neg_len} < horizon {train_horizon}"
    assert union_len >= train_horizon, f"Union trajectory length {union_len} < horizon {train_horizon}"
    
    # Split RNG
    rng, *keys = jax.random.split(rng, 5)
    
    # Sample indices (convert to numpy for indexing)
    neg_idx = np.array(jax.random.randint(keys[0], (batch_size,), 0, n_neg))
    union_idx = np.array(jax.random.randint(keys[1], (batch_size,), 0, n_union))
    neg_starts = np.array(jax.random.randint(keys[2], (batch_size,), 0, max(1, neg_len - train_horizon)))
    union_starts = np.array(jax.random.randint(keys[3], (batch_size,), 0, max(1, union_len - train_horizon)))
    
    # Extract segments
    neg_obs_batch, neg_act_batch = [], []
    union_obs_batch, union_act_batch, union_rew_batch = [], [], []
    
    for i in range(batch_size):
        neg_obs_batch.append(neg_obs_arr[neg_idx[i], neg_starts[i]:neg_starts[i]+train_horizon])
        neg_act_batch.append(neg_act_arr[neg_idx[i], neg_starts[i]:neg_starts[i]+train_horizon])
        union_obs_batch.append(union_obs_arr[union_idx[i], union_starts[i]:union_starts[i]+train_horizon])
        union_act_batch.append(union_act_arr[union_idx[i], union_starts[i]:union_starts[i]+train_horizon])
        union_rew_batch.append(union_rew_arr[union_idx[i], union_starts[i]:union_starts[i]+train_horizon])
    
    # Stack and transpose: [B, H, D] -> [H, B, D]
    neg_obs = jnp.array(np.stack(neg_obs_batch)).transpose(1, 0, 2)
    neg_acts = jnp.array(np.stack(neg_act_batch)).transpose(1, 0, 2)
    union_obs = jnp.array(np.stack(union_obs_batch)).transpose(1, 0, 2)
    union_acts = jnp.array(np.stack(union_act_batch)).transpose(1, 0, 2)
    # Rewards are scalars: [B, H] -> [H, B] -> [H, B, 1]
    union_rew = jnp.array(np.stack(union_rew_batch)).transpose(1, 0)
    union_rew = union_rew[..., None]  # Add trailing dimension: [H, B, 1]
    
    # Normalize observations if stats provided
    if mu_obs is not None:
        neg_obs = normalize_observation_jax(neg_obs, mu_obs, std_obs)
        union_obs = normalize_observation_jax(union_obs, mu_obs, std_obs)
    
    return neg_obs, neg_acts, union_obs, union_acts, union_rew, rng


# =============================================================================
# Training Step Functions
# =============================================================================

@partial(jax.jit, static_argnames=['gamma', 'lambda_q_reg'])
def q_preference_step(
    q_state: QTrainState,
    neg_obs: jnp.ndarray,
    neg_acts: jnp.ndarray,
    union_obs: jnp.ndarray,
    union_acts: jnp.ndarray,
    gamma: float = 0.99,
    lambda_q_reg: float = 1e-2
) -> Tuple[QTrainState, Dict[str, jnp.ndarray]]:
    """
    Q preference learning step: union preferred over negative.
    
    Args:
        q_state: QTrainState with params and target_params
        neg_obs: [H, B, obs_dim] negative observations
        neg_acts: [H, B, act_dim] negative actions
        union_obs: [H, B, obs_dim] union observations
        union_acts: [H, B, act_dim] union actions
        gamma: Discount factor
        lambda_q_reg: Q regularization coefficient
        
    Returns:
        q_state: Updated QTrainState
        metrics: Dict with 'pref_loss', 'reg_loss', 'q1_mean', 'q2_mean', 'q_diff'
    """
    horizon, batch = union_obs.shape[0], union_obs.shape[1]
    
    # Flatten: [H, B, D] -> [H*B, D]
    flat_union_obs = union_obs.reshape(-1, union_obs.shape[-1])  # [H*B, obs_dim]
    flat_union_acts = union_acts.reshape(-1, union_acts.shape[-1])  # [H*B, act_dim]
    flat_neg_obs = neg_obs.reshape(-1, neg_obs.shape[-1])  # [H*B, obs_dim]
    flat_neg_acts = neg_acts.reshape(-1, neg_acts.shape[-1])  # [H*B, act_dim]
    
    def loss_fn(params):
        # Forward pass: returns (q_min, (q1, q2))
        # q1, q2: [H*B]
        _, (q1_union, q2_union) = q_state.apply_fn(params, flat_union_obs, flat_union_acts)
        _, (q1_neg, q2_neg) = q_state.apply_fn(params, flat_neg_obs, flat_neg_acts)
        
        # Conservative estimate: min(Q1, Q2)
        q_union_flat = jnp.minimum(q1_union, q2_union)  # [H*B]
        q_neg_flat = jnp.minimum(q1_neg, q2_neg)  # [H*B]
        
        # Reshape to [H, B]
        q_union = q_union_flat.reshape(horizon, batch)
        q_neg = q_neg_flat.reshape(horizon, batch)
        
        # Trajectory scores (discounted sum over horizon)
        discounts = jnp.array([gamma ** i for i in range(horizon)])[:, None]  # [H, 1]
        s_union = (q_union * discounts).sum(axis=0)  # [B]
        s_neg = (q_neg * discounts).sum(axis=0)  # [B]
        
        # Preference loss: sigmoid BCE, union should have higher score
        logits = s_union - s_neg  # [B]
        labels = jnp.ones_like(logits)  # [B]
        pref_loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        
        # Twin Q regularization
        q1_reg = 0.5 * (jnp.mean(q1_union**2) + jnp.mean(q1_neg**2))
        q2_reg = 0.5 * (jnp.mean(q2_union**2) + jnp.mean(q2_neg**2))
        q_reg = 0.5 * (q1_reg + q2_reg)
        
        score_reg = 0.5 * (jnp.mean(s_union**2) + jnp.mean(s_neg**2))
        reg_loss = lambda_q_reg * (q_reg + 0.1 * score_reg)
        
        total_loss = pref_loss + reg_loss
        
        metrics = {
            "pref_loss": pref_loss,
            "reg_loss": reg_loss,
            "total_q_loss": total_loss,
            "q1_mean": jnp.mean(q1_union),
            "q2_mean": jnp.mean(q2_union),
            "q_diff": jnp.mean(jnp.abs(q1_union - q2_union))
        }
        
        return total_loss, metrics
    
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
    q_state = q_state.apply_gradients(grads=grads)
    
    return q_state, metrics


@partial(jax.jit, static_argnames=['gamma', 'v_use_neg'])
def v_update_step(
    v_state: train_state.TrainState,
    q_state: QTrainState,
    union_obs: jnp.ndarray,
    union_acts: jnp.ndarray,
    neg_obs: jnp.ndarray,
    neg_acts: jnp.ndarray,
    gamma: float = 0.99,
    v_use_neg: bool = True
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    """
    V-network update: V(s) ≈ Q(s,a) - γ V(s').
    
    Args:
        v_state: V-network TrainState
        q_state: Q-network TrainState (frozen)
        union_obs: [H, B, obs_dim]
        union_acts: [H, B, act_dim]
        neg_obs: [H, B, obs_dim]
        neg_acts: [H, B, act_dim]
        gamma: Discount factor
        v_use_neg: Whether to include negative data in V updates
        
    Returns:
        v_state: Updated V-network TrainState
        metrics: Dict with 'v_loss'
    """
    # Build next observations by shifting
    # union_next_obs[t] = union_obs[t+1], union_next_obs[-1] = union_obs[-1]
    union_next_obs = jnp.roll(union_obs, -1, axis=0)
    union_next_obs = union_next_obs.at[-1].set(union_obs[-1])
    
    # Flatten
    flat_u_obs = union_obs.reshape(-1, union_obs.shape[-1])  # [H*B, obs_dim]
    flat_u_acts = union_acts.reshape(-1, union_acts.shape[-1])  # [H*B, act_dim]
    flat_u_next = union_next_obs.reshape(-1, union_next_obs.shape[-1])  # [H*B, obs_dim]
    
    if v_use_neg:
        neg_next_obs = jnp.roll(neg_obs, -1, axis=0)
        neg_next_obs = neg_next_obs.at[-1].set(neg_obs[-1])
        
        flat_n_obs = neg_obs.reshape(-1, neg_obs.shape[-1])
        flat_n_acts = neg_acts.reshape(-1, neg_acts.shape[-1])
        flat_n_next = neg_next_obs.reshape(-1, neg_next_obs.shape[-1])
        
        # Concatenate union and negative
        flat_obs = jnp.concatenate([flat_u_obs, flat_n_obs], axis=0)  # [2*H*B, obs_dim]
        flat_acts = jnp.concatenate([flat_u_acts, flat_n_acts], axis=0)  # [2*H*B, act_dim]
        flat_next = jnp.concatenate([flat_u_next, flat_n_next], axis=0)  # [2*H*B, obs_dim]
    else:
        flat_obs = flat_u_obs
        flat_acts = flat_u_acts
        flat_next = flat_u_next
    
    # Compute targets (no gradient through Q and V_next)
    q_flat, _ = q_state.apply_fn(q_state.params, flat_obs, flat_acts)  # [N]
    v_next = v_state.apply_fn(v_state.params, flat_next)  # [N]
    target = jax.lax.stop_gradient(q_flat - gamma * v_next)  # [N]
    
    def v_loss_fn(v_params):
        v_pred = v_state.apply_fn(v_params, flat_obs)  # [N]
        loss = jnp.mean((v_pred - target) ** 2)
        return loss
    
    loss, grads = jax.value_and_grad(v_loss_fn)(v_state.params)
    v_state = v_state.apply_gradients(grads=grads)
    
    return v_state, {"v_loss": loss}


@jax.jit
def compute_weights_from_qv(
    q_state: QTrainState,
    v_state: train_state.TrainState,
    union_obs: jnp.ndarray,
    union_acts: jnp.ndarray,
    cost_weight_temp: float = 1.0
) -> jnp.ndarray:
    """
    Compute advantage-based energy weights from Q-V.
    
    Args:
        q_state: Q-network TrainState
        v_state: V-network TrainState
        union_obs: [H, B, obs_dim]
        union_acts: [H, B, act_dim]
        cost_weight_temp: Temperature for scaling advantages
        
    Returns:
        energy: [H, B] energy weights (centered advantages * temp)
    """
    horizon, batch = union_obs.shape[0], union_obs.shape[1]
    
    flat_obs = union_obs.reshape(-1, union_obs.shape[-1])  # [H*B, obs_dim]
    flat_acts = union_acts.reshape(-1, union_acts.shape[-1])  # [H*B, act_dim]
    
    # Get Q and V values
    q_flat, _ = q_state.apply_fn(q_state.params, flat_obs, flat_acts)  # [H*B]
    v_flat = v_state.apply_fn(v_state.params, flat_obs)  # [H*B]
    
    # Reshape to [H, B]
    q_vals = q_flat.reshape(horizon, batch)
    v_vals = v_flat.reshape(horizon, batch)
    
    # Compute advantages and center
    adv = q_vals - v_vals  # [H, B]
    adv_centered = adv - jnp.mean(adv)
    
    # Scale by temperature
    energy = adv_centered * cost_weight_temp
    
    return energy  # [H, B]


# =============================================================================
# Flow Matching OT Functions
# =============================================================================

def psi_t_ot(
    x0: jnp.ndarray, 
    x1: jnp.ndarray, 
    t: jnp.ndarray, 
    sigma_min: float
) -> jnp.ndarray:
    """
    OT linear interpolation.
    
    Args:
        x0: [B, D] noise
        x1: [B, D] data trajectory
        t: [B] time in [0, 1]
        sigma_min: Minimum sigma
        
    Returns:
        x_t: [B, D] interpolated state
    """
    sigma_t = (1.0 - (1.0 - sigma_min) * t)[:, None]  # [B, 1]
    t_expanded = t[:, None]  # [B, 1]
    return sigma_t * x0 + t_expanded * x1


def u_t_ot(
    x_t: jnp.ndarray, 
    x1: jnp.ndarray, 
    t: jnp.ndarray, 
    sigma_min: float
) -> jnp.ndarray:
    """
    OT vector field.
    
    Args:
        x_t: [B, D] current state
        x1: [B, D] target state
        t: [B] time
        sigma_min: Minimum sigma
        
    Returns:
        u_t: [B, D] target vector field
    """
    one_minus_sigma_min = 1.0 - sigma_min
    denom = (1.0 - one_minus_sigma_min * t)[:, None]  # [B, 1]
    denom = jnp.maximum(denom, 1e-6)
    return (x1 - one_minus_sigma_min * x_t) / denom


@partial(jax.jit, static_argnames=['sigma_min', 'energy_alpha', 'use_guidance'])
def train_flow_step(
    flow_state: FlowTrainState,
    union_obs: jnp.ndarray,
    union_acts: jnp.ndarray,
    weights: Optional[jnp.ndarray],
    rng: jax.random.PRNGKey,
    sigma_min: float = 0.01,
    energy_alpha: float = 3.0,
    use_guidance: bool = True
) -> Tuple[FlowTrainState, jnp.ndarray, jax.random.PRNGKey]:
    """
    Flow Matching training step with OT path.
    
    Args:
        flow_state: FlowTrainState
        union_obs: [H, B, obs_dim]
        union_acts: [H, B, act_dim]
        weights: [H, B] energy weights (optional)
        rng: JAX random key
        sigma_min: OT path sigma_min
        energy_alpha: Guidance temperature
        use_guidance: Whether to use energy-weighted guidance
        
    Returns:
        flow_state: Updated FlowTrainState
        loss: Scalar loss value
        rng: Updated random key
    """
    horizon, batch, obs_dim = union_obs.shape
    act_dim = union_acts.shape[-1]
    
    # Condition on first observation: [B, obs_dim]
    cond_obs = union_obs[0]
    
    # Prepare data: [H, B, act_dim] -> [B, H, act_dim] -> [B, H*act_dim]
    acts_reshaped = union_acts.transpose(1, 0, 2)
    x1 = acts_reshaped.reshape(batch, -1)  # [B, D] where D = H * act_dim
    
    # Split RNG
    rng, noise_key, time_key = jax.random.split(rng, 3)
    
    # Sample x0 ~ N(0, I): [B, D]
    x0 = jax.random.normal(noise_key, x1.shape)
    
    # Sample t ~ Uniform(eps, 1-eps): [B]
    eps = 1e-6
    random_t = jax.random.uniform(time_key, (batch,), minval=eps, maxval=1.0 - eps)
    
    def loss_fn(params):
        # OT interpolation
        x_t = psi_t_ot(x0, x1, random_t, sigma_min)  # [B, D]
        u_t = u_t_ot(x_t, x1, random_t, sigma_min)    # [B, D]
        
        # Model prediction (positional args: params, x, t, condition, train)
        v_theta = flow_state.apply_fn(
            params, x_t, random_t, cond_obs, True
        )  # [B, D]
        
        # Reshape to per-timestep: [B, D] -> [B, H, act_dim]
        v_theta_ts = v_theta.reshape(batch, horizon, act_dim)
        u_t_ts = u_t.reshape(batch, horizon, act_dim)
        
        # Per-timestep squared errors: [B, H]
        per_step_err = jnp.sum((v_theta_ts - u_t_ts)**2, axis=2)
        
        # Guidance weights
        if use_guidance and weights is not None:
            # weights: [H, B] -> [B, H]
            energy = weights.T
            # Clip for stability
            energy_clipped = jnp.clip(energy, -50.0, 50.0)
            # Softmax to get per-timestep weights: [B, H]
            guidance = jax.nn.softmax(energy_alpha * energy_clipped, axis=1)
            guidance = jax.lax.stop_gradient(guidance)
        else:
            # Uniform weights
            guidance = jnp.ones((batch, horizon)) / float(horizon)
        
        # Weighted loss
        loss = jnp.mean(jnp.sum(per_step_err * guidance, axis=1))
        
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(flow_state.params)
    flow_state = flow_state.apply_gradients(grads=grads)
    
    return flow_state, loss, rng


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_policy_jax(
    eval_env,
    flow_state: FlowTrainState,
    obs_dim: int,
    act_dim: int,
    horizon: int,
    mu_obs: Optional[jnp.ndarray],
    std_obs: Optional[jnp.ndarray],
    diffusion_steps: int = 15
) -> Tuple[float, float, int]:
    """
    Evaluate flow policy in environment.
    
    Args:
        eval_env: Gymnasium environment
        flow_state: Trained FlowTrainState
        obs_dim: Observation dimension
        act_dim: Action dimension
        horizon: Planning horizon
        mu_obs: Observation mean for normalization [obs_dim]
        std_obs: Observation std for normalization [obs_dim]
        diffusion_steps: Number of ODE integration steps
        
    Returns:
        total_reward: Episode return
        total_cost: Episode cost
        total_len: Episode length
    """
    obs, _ = eval_env.reset()
    obs = jnp.array(obs).reshape(1, -1)  # [1, obs_dim]
    
    if mu_obs is not None:
        obs = normalize_observation_jax(obs, mu_obs, std_obs)
    
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    
    while not done:
        # Select action using flow model
        action = select_actions_jax(
            apply_fn=flow_state.apply_fn,
            params=flow_state.params,
            states=obs,
            horizon=horizon,
            act_dim=act_dim,
            diffusion_steps=diffusion_steps,
            method='rk4',
            use_first_action=True
        )  # [1, act_dim]
        
        # Step environment
        action_np = np.array(action[0])
        next_obs, reward, terminated, truncated, info = eval_env.step(action_np)
        
        # Update observation
        obs = jnp.array(next_obs).reshape(1, -1)
        if mu_obs is not None:
            obs = normalize_observation_jax(obs, mu_obs, std_obs)
        
        total_reward += reward
        total_cost += info.get("cost", 0.0)
        total_len += 1
        done = terminated or truncated
    
    return total_reward, total_cost, total_len


# =============================================================================
# Main Training Function
# =============================================================================

def main(args):
    """
    Main training loop for IPL + Flow Matching.
    
    Phase 1: TwinQ and V pretraining with preference learning
    Phase 2: Flow model training with energy-weighted guidance
    """
    # Merge user args with default config
    config = {**default_cfg}
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v

    # Initialize JAX RNG
    rng = jax.random.PRNGKey(args.seed)

    # =========================================================================
    # Setup logging & experiment directories
    # =========================================================================
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = f"seed-{str(args.seed).zfill(3)}"
    relpath = f"{subfolder}-{relpath}"
    algo = "ipl_flow_twinq_fm_jax"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config({**config, **vars(args)})

    # =========================================================================
    # Build environment
    # =========================================================================
    print(f"Creating environment: {args.task}")
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)

    # =========================================================================
    # Load dataset
    # =========================================================================
    print(f"\nLoading DSRL dataset...")
    
    dataset_config = {
        "density": config.get("density", 1.0),
        "inpaint_ranges": config.get("inpaint_ranges", []),
        "num_negative_trajectories": config.get("num_negative_trajectories", 50),
        "num_union_trajectories": config.get("num_union_trajectories", -1),
        "non_pref_noise": config.get("non_pref_noise", 0.0),
    }
    
    raw_data = eval_env.get_dataset()
    
    # Compute trajectory lengths
    dones_idx = np.where((raw_data["terminals"] == 1) | (raw_data["timeouts"] == 1))[0]
    traj_lengths = []
    start = 0
    for end_idx in dones_idx:
        traj_lengths.append(end_idx - start + 1)
        start = end_idx + 1
    
    max_traj_len = max(traj_lengths)
    mean_traj_len = np.mean(traj_lengths)
    print(f"Trajectory length statistics: mean={mean_traj_len:.1f}, max={max_traj_len}, min={min(traj_lengths)}")
    
    ep_len = max_traj_len
    num_folds = config.get("num_folds", 1)
    
    d4rl_data = get_dataset_in_d4rl_format(
        env=eval_env,
        config=dataset_config,
        task=args.task,
        ep_len=ep_len,
        num_folds=num_folds
    )
    
    print(f"D4RL data loaded. Shape: {d4rl_data['observations'].shape}")
    
    neg_data, union_data = get_neg_and_union_data_2(d4rl_data, dataset_config)
    
    # Normalize observations if requested
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        print("Normalizing observations...")
        neg_data, union_data, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
        mu_obs = jnp.array(mu_obs)
        std_obs = jnp.array(std_obs)
    
    print(f"\nDataset statistics:")
    print(f"  Negative set: {neg_data['observations'].shape}")
    print(f"  Union set: {union_data['observations'].shape}")
    print(f"  Observation dim: {neg_data['observations'].shape[-1]}")
    print(f"  Action dim: {neg_data['actions'].shape[-1]}")
    
    # =========================================================================
    # Get dimensions
    # =========================================================================
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    train_horizon = config["train_horizon"]
    traj_dim = train_horizon * act_dim
    
    print(f"\nModel dimensions:")
    print(f"  obs_dim: {obs_dim}")
    print(f"  act_dim: {act_dim}")
    print(f"  train_horizon: {train_horizon}")
    print(f"  traj_dim: {traj_dim}")
    
    # =========================================================================
    # Initialize models
    # =========================================================================
    print("\nInitializing JAX models...")
    rng, q_rng, v_rng, flow_rng = jax.random.split(rng, 4)
    
    q_model = FlaxTwinQ(hidden_dims=(config["q_hidden"], config["q_hidden"], config["q_hidden"]))
    v_model = FlaxVNetwork(hidden_size=config["v_hidden"])
    flow_model = FlaxScoreNet(output_dim=traj_dim)
    
    q_state = create_q_train_state(
        q_rng, q_model, obs_dim, act_dim,
        learning_rate=config["q_lr"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"]
    )
    
    v_state = create_v_train_state(
        v_rng, v_model, obs_dim,
        learning_rate=config["v_lr"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"]
    )
    
    flow_state = create_flow_train_state(
        flow_rng, flow_model, obs_dim, traj_dim,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"]
    )
    
    # Training config
    q_pretrain_iters = config["q_pretrain_iterations"]
    flow_train_iters = config["flow_train_iterations"]
    batch_size = config["batch_size"]
    v_updates_per_q = config.get("v_updates_per_q_update", 1)
    use_guidance = getattr(args, 'use_guidance', True)

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    # =========================================================================
    # Phase 1: TwinQ and V pretraining
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: TwinQ & V Pretraining")
    print("=" * 60)
    
    pbar = tqdm(range(q_pretrain_iters), desc="Phase1: Q+V")
    
    for step in pbar:
        # Sample batch
        neg_obs, neg_acts, union_obs, union_acts, union_rew, rng = sample_trajectory_batch_jax(
            neg_data, union_data, batch_size, train_horizon, rng, mu_obs, std_obs
        )
        
        # Q preference update
        q_state, q_metrics = q_preference_step(
            q_state, neg_obs, neg_acts, union_obs, union_acts,
            gamma=config["gamma"],
            lambda_q_reg=config["lambda_q_reg"]
        )
        
        # V updates
        for _ in range(v_updates_per_q):
            v_state, v_metrics = v_update_step(
                v_state, q_state, union_obs, union_acts, neg_obs, neg_acts,
                gamma=config["gamma"],
                v_use_neg=config["v_use_neg_in_updates"]
            )
        
        # Target network soft update
        if (step + 1) % config["target_update_freq"] == 0:
            q_state = q_state.replace(
                target_params=update_target_params(
                    q_state.params,
                    q_state.target_params,
                    config["target_tau"]
                )
            )
        
        # Logging
        if (step + 1) % config["log_freq"] == 0:
            logger.store(
                Phase1_Step=step + 1,
                Q_PrefLoss=float(q_metrics["pref_loss"]),
                Q_RegLoss=float(q_metrics["reg_loss"]),
                Q1_Mean=float(q_metrics["q1_mean"]),
                Q2_Mean=float(q_metrics["q2_mean"]),
                Q_Diff=float(q_metrics["q_diff"]),
                V_Loss=float(v_metrics["v_loss"]),
            )
            pbar.set_postfix(
                pref=f"{q_metrics['pref_loss']:.4f}",
                v=f"{v_metrics['v_loss']:.4f}",
                q1=f"{q_metrics['q1_mean']:.2f}"
            )
            logger.dump_tabular()
        
        # Save checkpoint during Phase 1
        if (step + 1) % config["save_freq"] == 0:
            import pickle
            ckpt_dir = os.path.join(args.log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            ckpt_path = os.path.join(ckpt_dir, f"phase1_ckpt_step_{step+1}.pkl")
            checkpoint = {
                "phase": 1,
                "step": step + 1,
                "q_params": q_state.params,
                "q_target_params": q_state.target_params,
                "v_params": v_state.params,
                "config": config,
            }
            with open(ckpt_path, "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"\n[Phase1 Checkpoint saved @ step {step+1}] -> {ckpt_path}")
    
    print("Phase 1 complete.")

    # =========================================================================
    # Phase 2: Flow model training
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Flow Matching Training")
    print("=" * 60)
    
    pbar = tqdm(range(flow_train_iters), desc="Phase2: Flow")
    
    for step in pbar:
        # Sample batch
        neg_obs, neg_acts, union_obs, union_acts, union_rew, rng = sample_trajectory_batch_jax(
            neg_data, union_data, batch_size, train_horizon, rng, mu_obs, std_obs
        )
        
        # Compute energy weights from Q-V
        weights = None
        if use_guidance:
            weights = compute_weights_from_qv(
                q_state, v_state, union_obs, union_acts,
                cost_weight_temp=config["cost_weight_temp"]
            )
        
        # Flow training step
        flow_state, flow_loss, rng = train_flow_step(
            flow_state, union_obs, union_acts, weights, rng,
            sigma_min=config["sigma_min"],
            energy_alpha=config["energy_alpha"],
            use_guidance=use_guidance
        )
        
        # Logging
        if (step + 1) % config["log_freq"] == 0:
            logger.store(
                Phase2_Step=step + 1,
                Flow_Loss=float(flow_loss),
            )
            pbar.set_postfix(loss=f"{flow_loss:.6f}")
            logger.dump_tabular()
        
        # Evaluation
        if args.use_eval and (step + 1) % args.eval_freq == 0:
            eval_reward, eval_cost, eval_len = evaluate_policy_jax(
                eval_env, flow_state, obs_dim, act_dim, train_horizon,
                mu_obs, std_obs, config["diffusion_steps"]
            )
            logger.store(
                Eval_Reward=eval_reward,
                Eval_Cost=eval_cost,
                Eval_Length=eval_len,
            )
            print(f"\n[Eval @ {step+1}] Reward={eval_reward:.2f}, Cost={eval_cost:.2f}, Len={eval_len}")
            logger.dump_tabular()
        
        # Save checkpoint
        if (step + 1) % config["save_freq"] == 0:
            import pickle
            ckpt_dir = os.path.join(args.log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # Save Q, V, and Flow states
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{step+1}.pkl")
            checkpoint = {
                "step": step + 1,
                "q_params": q_state.params,
                "q_target_params": q_state.target_params,
                "v_params": v_state.params,
                "flow_params": flow_state.params,
                "config": config,
            }
            with open(ckpt_path, "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"\n[Checkpoint saved @ step {step+1}] -> {ckpt_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Final evaluation
    if args.use_eval:
        print("\nFinal evaluation...")
        rewards, costs, lengths = [], [], []
        for _ in range(10):
            r, c, l = evaluate_policy_jax(
                eval_env, flow_state, obs_dim, act_dim, train_horizon,
                mu_obs, std_obs, config["diffusion_steps"]
            )
            rewards.append(r)
            costs.append(c)
            lengths.append(l)
        
        print(f"Final Results (10 episodes):")
        print(f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
        print(f"  Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        
        logger.store(
            Final_Reward_Mean=np.mean(rewards),
            Final_Reward_Std=np.std(rewards),
            Final_Cost_Mean=np.mean(costs),
            Final_Cost_Std=np.std(costs),
            Final_Length_Mean=np.mean(lengths),
        )
        logger.dump_tabular()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IPL + Flow Matching Training (JAX)")
    
    # Task and experiment
    parser.add_argument("--task", type=str, default="OfflineSwimmerVelocityGymnasium-v1",
                        help="DSRL task name")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for logs")
    parser.add_argument("--experiment", type=str, default="jax_exp",
                        help="Experiment name")
    
    # Training settings
    parser.add_argument("--seed", type=int, default=3,
                        help="Random seed")
    parser.add_argument("--normalize_observation", action="store_true", default=False,
                        help="Normalize observations")
    parser.add_argument("--use_guidance", action="store_false", dest="no_guidance",
                        help="Disable energy-weighted guidance (default: enabled)")
    parser.set_defaults(use_guidance=True)
    
    # Evaluation
    parser.add_argument("--use_eval", action="store_true", default=False,
                        help="Run evaluation during training")
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="Evaluation frequency")
    
    # Hyperparameters (can override defaults)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_horizon", type=int, default=None)
    parser.add_argument("--q_pretrain_iterations", type=int, default=None)
    parser.add_argument("--flow_train_iterations", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--q_lr", type=float, default=None)
    parser.add_argument("--v_lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--energy_alpha", type=float, default=None)
    parser.add_argument("--cost_weight_temp", type=float, default=None)
    
    # Logging and saving frequencies
    parser.add_argument("--log_freq", type=int, default=None,
                        help="Logging frequency")
    parser.add_argument("--save_freq", type=int, default=None,
                        help="Checkpoint saving frequency")
    
    # Dataset config
    parser.add_argument("--num_negative_trajectories", type=int, default=None,
                        help="Number of negative trajectories")
    parser.add_argument("--num_union_trajectories", type=int, default=None,
                        help="Number of union trajectories (-1 for all)")
    
    args = parser.parse_args()
    main(args)