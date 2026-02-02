#!/usr/bin/env python3

import os, time, sys, pickle
import os.path as osp
from typing import Dict, Tuple, Any, Optional

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from flax.training import train_state
import optax
from tqdm import tqdm

import gymnasium as gym
import dsrl
import dsrl.offline_safety_gymnasium

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../..')))

from diffusion_SDE.model_jax import FlaxScoreNet
from utils.train_state import FlowTrainState, create_flow_train_state
from utils.jax_utils import normalize_observation_jax
from dsrl_model.utils.logger import EpochLogger

from dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
    get_normalized_data,
)

default_cfg = {
    "lr": 3e-4, 
    "batch_size": 64,
    "train_horizon": 15,
    "diffusion_steps": 15,
    "sigma_min": 0.01,
    "max_grad_norm": 1.0,
    "obs_hist_len": 4,
    "flow_train_iterations": int(3e4),
    "log_freq": int(2e3),
    "save_freq": int(2e4),
    "density": 1.0,
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
    "non_pref_noise": 0.0,
    "num_folds": 1,
    "union_cost_percentile": 100.0,  # Filter union to bottom X% cost (100 = no filter)
}


# =============================================================================
# Dataset preparation (NumPy -> padded, JAX-ready)
# =============================================================================
def make_obs_history(obs: np.ndarray, hist_len: int):
    """
    obs: [N, T, obs_dim] → [N, T, hist_len * obs_dim]
    """
    N, T, obs_dim = obs.shape
    pad = np.repeat(obs[:, :1], hist_len - 1, axis=1)
    padded = np.concatenate([pad, obs], axis=1)

    hist = []
    for t in range(T):
        hist.append(padded[:, t:t+hist_len].reshape(N, -1))

    return np.stack(hist, axis=1)

def prepare_dataset_for_jax(
    neg_data: Dict[str, np.ndarray],
    union_data: Dict[str, np.ndarray],
    train_horizon: int,
) -> Dict[str, jnp.ndarray]:
    """
    TODO : Observation history -------> done
    Prepare negative and union datasets for JAX sampling.

    We *truncate or pad* each trajectory to at least `train_horizon` time steps.

    Let:
        N_neg      = number of negative trajectories
        N_union    = number of union trajectories
        T_neg      = max length of negative traj in raw data
        T_union    = max length of union traj in raw data
        H          = train_horizon
        obs_dim    = observation dimension
        act_dim    = action dimension

    Input shapes (raw NumPy):
        neg_data['observations'] : [N_neg,   T_neg,   obs_dim]
        neg_data['actions']      : [N_neg,   T_neg,   act_dim]
        union_data['observations']: [N_union, T_union, obs_dim]
        union_data['actions']     : [N_union, T_union, act_dim]
        union_data['rewards']     : [N_union, T_union] or [N_union, T_union, 1]

    Output shapes (JAX arrays, still with full T_neg/T_union):
        neg_obs_all   : [N_neg,   T_neg,   obs_dim]
        neg_acts_all  : [N_neg,   T_neg,   act_dim]
        union_obs_all : [N_union, T_union, obs_dim]
        union_acts_all: [N_union, T_union, act_dim]
        union_rew_all : [N_union, T_union, 1]
    """
    # Basic conversions to float32 JAX arrays
    neg_obs_all = jnp.array(neg_data["observations"], dtype=jnp.float32)
    neg_acts_all = jnp.array(neg_data["actions"], dtype=jnp.float32)
    union_obs_all = jnp.array(union_data["observations"], dtype=jnp.float32)
    union_acts_all = jnp.array(union_data["actions"], dtype=jnp.float32)

    union_rew_np = np.asarray(union_data["rewards"])
    if union_rew_np.ndim == 2:
        union_rew_np = union_rew_np[..., None]
    union_rew_all = jnp.array(union_rew_np, dtype=jnp.float32)

    return {
        "neg_obs_all": neg_obs_all,
        "neg_acts_all": neg_acts_all,
        "union_obs_all": union_obs_all,
        "union_acts_all": union_acts_all,
        "union_rew_all": union_rew_all,
    }


# =============================================================================
# Fully JAX batch sampling (vmap + dynamic_slice)
# =============================================================================


def _slice_segment(
    traj: jnp.ndarray,  # [T, D]
    start: jnp.ndarray,  # []
    H: int,
) -> jnp.ndarray:
    """Slice a length-H segment from a single trajectory.

    traj : [T, D]
    start: scalar, 0 <= start <= T-H
    returns: [H, D]
    """
    T = traj.shape[0]
    D = traj.shape[1]
    start_clamped = jnp.minimum(start, T - H)
    return jax.lax.dynamic_slice(traj, (start_clamped, 0), (H, D))


def _sample_segments(
    trajs: jnp.ndarray,  # [N, T, D]
    idxs: jnp.ndarray,   # [B]
    starts: jnp.ndarray,  # [B]
    H: int,
) -> jnp.ndarray:
    """Vectorized segment sampling over batch.

    trajs : [N, T, D]
    idxs  : [B]  indices into N
    starts: [B]  start indices per batch element
    returns: [B, H, D]
    """
    def one(idx, st):
        return _slice_segment(trajs[idx], st, H)

    return jax.vmap(one)(idxs, starts)


@partial(jax.jit, static_argnames=("batch_size", "train_horizon"))
def sample_trajectory_batch_jax(
    neg_obs_all: jnp.ndarray,
    neg_acts_all: jnp.ndarray,
    union_obs_all: jnp.ndarray,
    union_acts_all: jnp.ndarray,
    union_rew_all: jnp.ndarray,
    batch_size: int,
    train_horizon: int,
    rng: jax.random.PRNGKey,
    mu_obs: Optional[jnp.ndarray] = None,
    std_obs: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """
    Sample a batch of trajectory segments (fully JAX, no Python loop).

    Let:
        N_neg, T_neg, obs_dim = neg_obs_all.shape
        N_union, T_union, obs_dim = union_obs_all.shape
        H = train_horizon
        B = batch_size

    Input:
        neg_obs_all   : [N_neg,   T_neg,   obs_dim]
        neg_acts_all  : [N_neg,   T_neg,   act_dim]
        union_obs_all : [N_union, T_union, obs_dim]
        union_acts_all: [N_union, T_union, act_dim]
        union_rew_all : [N_union, T_union, 1]

    Output:
        neg_obs   : [H, B, obs_dim]
        neg_acts  : [H, B, act_dim]
        union_obs : [H, B, obs_dim]
        union_acts: [H, B, act_dim]
        union_rew : [H, B, 1]
    """
    N_neg, T_neg, _ = neg_obs_all.shape
    N_union, T_union, _ = union_obs_all.shape

    rng, key_neg_idx, key_union_idx, key_neg_start, key_union_start = jax.random.split(rng, 5)

    # Sample trajectory indices
    neg_idx = jax.random.randint(key_neg_idx, (batch_size,), 0, N_neg)        # [B]
    union_idx = jax.random.randint(key_union_idx, (batch_size,), 0, N_union)  # [B]

    # Sample start indices; ensure room for H
    max_neg_start = jnp.maximum(1, T_neg - train_horizon)
    max_union_start = jnp.maximum(1, T_union - train_horizon)

    neg_starts = jax.random.randint(key_neg_start, (batch_size,), 0, max_neg_start)        # [B]
    union_starts = jax.random.randint(key_union_start, (batch_size,), 0, max_union_start)  # [B]

    # Sample segments: [B, H, D]
    neg_obs_BHD = _sample_segments(neg_obs_all, neg_idx, neg_starts, train_horizon)
    neg_acts_BHD = _sample_segments(neg_acts_all, neg_idx, neg_starts, train_horizon)
    union_obs_BHD = _sample_segments(union_obs_all, union_idx, union_starts, train_horizon)
    union_acts_BHD = _sample_segments(union_acts_all, union_idx, union_starts, train_horizon)
    union_rew_BH1 = _sample_segments(union_rew_all, union_idx, union_starts, train_horizon)

    # Transpose to [H, B, D]
    neg_obs = jnp.transpose(neg_obs_BHD, (1, 0, 2))
    neg_acts = jnp.transpose(neg_acts_BHD, (1, 0, 2))
    union_obs = jnp.transpose(union_obs_BHD, (1, 0, 2))
    union_acts = jnp.transpose(union_acts_BHD, (1, 0, 2))
    union_rew = jnp.transpose(union_rew_BH1, (1, 0, 2))

    if mu_obs is not None:
        neg_obs = normalize_observation_jax(neg_obs, mu_obs, std_obs)
        union_obs = normalize_observation_jax(union_obs, mu_obs, std_obs)

    return neg_obs, neg_acts, union_obs, union_acts, union_rew, rng


# =============================================================================
# OT path + flow loss
# =============================================================================


def psi_t_ot(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    sigma_min: float,
) -> jnp.ndarray:
    """OT linear interpolation.

    x0, x1: [B, D]
    t     : [B]
    -> x_t: [B, D]
    """
    sigma_t = (1.0 - (1.0 - sigma_min) * t)[:, None]
    return sigma_t * x0 + t[:, None] * x1


def u_t_ot(
    x_t: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    sigma_min: float,
) -> jnp.ndarray:
    """OT vector field.

    x_t, x1: [B, D]
    t      : [B]
    -> u_t : [B, D]
    """
    one_minus_sigma_min = 1.0 - sigma_min
    denom = (1.0 - one_minus_sigma_min * t)[:, None]
    denom = jnp.maximum(denom, 1e-6)
    return (x1 - one_minus_sigma_min * x_t) / denom


def train_flow_step(
    flow_state: FlowTrainState,
    rng: jax.random.PRNGKey,
    *,
    obs: jnp.ndarray,   # [B, obs_dim]
    traj: jnp.ndarray,  # [B, traj_dim]
):
    B = traj.shape[0]
    rng, t_rng, noise_rng = jax.random.split(rng, 3)

    t = jax.random.uniform(t_rng, (B,))
    noise = jax.random.normal(noise_rng, traj.shape)

    x_t = (1 - t[:, None]) * traj + t[:, None] * noise

    v_target = -traj + noise

    def loss_fn(params):
        v_pred = flow_state.apply_fn(
            params, x_t, t, condition=obs, train=True
        )
        return jnp.mean((v_pred - v_target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(flow_state.params)
    flow_state = flow_state.apply_gradients(grads=grads)

    return flow_state, loss, rng




def flow_matching_loss(
    flow_state,
    obs,        # [B, obs_dim]
    traj,       # [B, traj_dim]
    rng,
):
    """
    Standard linear-path flow matching loss.

    Returns:
        loss, new_rng
    """
    B = traj.shape[0]
    rng, t_rng, noise_rng = jax.random.split(rng, 3)

    # Sample time
    t = jax.random.uniform(t_rng, (B,))              # [B]
    noise = jax.random.normal(noise_rng, traj.shape) # [B, traj_dim]

    # Linear path
    alpha_t = 1.0 - t[:, None]   # [B, 1]
    sigma_t = t[:, None]         # [B, 1]

    x_t = alpha_t * traj + sigma_t * noise

    # Predict velocity
    v_pred = flow_state.apply_fn(
        flow_state.params,
        x_t,
        t,
        condition=obs,
        train=False,
    )

    # True target velocity
    d_alpha = -1.0
    d_sigma =  1.0
    v_target = d_alpha * traj + d_sigma * noise

    loss = jnp.mean((v_pred - v_target) ** 2)

    return loss, rng


# =============================================================================
# Euler ODE Sampler for Flow Inference
# =============================================================================


@partial(jax.jit, static_argnames=("num_steps", "traj_dim"))
def euler_sample_flow(
    flow_state: FlowTrainState,
    obs: jnp.ndarray,        # [B, obs_dim] or [obs_dim]
    rng: jax.random.PRNGKey,
    num_steps: int = 20,
    traj_dim: int = 60,
) -> jnp.ndarray:
    """
    Sample trajectory from flow model using Euler integration.
    
    Integrates from t=1 (noise) to t=0 (data).
    
    Args:
        flow_state: FlowTrainState with trained params
        obs: Conditioning observation [B, obs_dim] or [obs_dim]
        rng: JAX random key for initial noise
        num_steps: Number of Euler steps
        traj_dim: Trajectory dimension (train_horizon * act_dim)
        
    Returns:
        Sampled trajectory [B, traj_dim] or [traj_dim]
    """
    # Handle single obs
    single_obs = obs.ndim == 1
    if single_obs:
        obs = obs[None, :]  # [1, obs_dim]
    
    B = obs.shape[0]
    
    # Start from noise at t=1
    x_t = jax.random.normal(rng, (B, traj_dim))
    
    dt = 1.0 / num_steps
    
    def euler_step(carry, t_val):
        x_t = carry
        t = jnp.full((B,), t_val)
        
        # Predict velocity (flow goes from noise to data, so we use -v)
        v = flow_state.apply_fn(
            flow_state.params, x_t, t, condition=obs, train=False
        )
        
        # Euler update: x_{t-dt} = x_t - dt * v (integrating backwards)
        x_next = x_t - dt * v
        return x_next, None
    
    # Integrate from t=1 to t=0
    t_vals = jnp.linspace(1.0, dt, num_steps)
    x_final, _ = jax.lax.scan(euler_step, x_t, t_vals)
    
    if single_obs:
        x_final = x_final[0]  # Remove batch dim
    
    return x_final


def check_flow_reconstruction(
    flow_state: FlowTrainState,
    obs_sample: jnp.ndarray,   # [N, T, obs_dim]
    acts_sample: jnp.ndarray,  # [N, T, act_dim]
    traj_dim: int,
    train_horizon: int,
    rng: jax.random.PRNGKey,
    num_samples: int = 32,
) -> Tuple[float, float, float]:
    """
    Check if flow samples match training data distribution.
    
    Returns:
        mse: Mean squared error between sampled and actual trajectories
        action_range: Max absolute value in sampled actions
        correlation: Correlation between sampled and actual (rough quality metric)
    """
    # Take first num_samples trajectories, first observation as condition
    n = min(num_samples, obs_sample.shape[0])
    cond_obs = obs_sample[:n, 0, :]  # [n, obs_dim]
    
    # Sample from flow
    sampled_traj = euler_sample_flow(
        flow_state,
        cond_obs,
        rng,
        num_steps=20,
        traj_dim=traj_dim,
    )  # [n, traj_dim]
    
    # Get actual trajectory (flatten actions)
    actual_acts = acts_sample[:n, :train_horizon, :]  # [n, H, act_dim]
    actual_traj = actual_acts.reshape(n, -1)  # [n, H * act_dim]
    
    # Ensure same size
    min_dim = min(sampled_traj.shape[1], actual_traj.shape[1])
    sampled_traj = sampled_traj[:, :min_dim]
    actual_traj = actual_traj[:, :min_dim]
    
    mse = float(jnp.mean((sampled_traj - actual_traj) ** 2))
    action_range = float(jnp.max(jnp.abs(sampled_traj)))
    
    # Simple correlation check
    sampled_flat = sampled_traj.flatten()
    actual_flat = actual_traj.flatten()
    correlation = float(jnp.corrcoef(sampled_flat, actual_flat)[0, 1])
    
    return mse, action_range, correlation


class ObsHistoryBuffer:
    """
    Maintains observation history for evaluation.
    
    Stores the last `hist_len` observations and returns
    the flattened history for conditioning.
    """
    def __init__(self, hist_len: int, obs_dim: int):
        self.hist_len = hist_len
        self.obs_dim = obs_dim
        self.buffer = None
    
    def reset(self, first_obs: np.ndarray):
        """Reset buffer with first observation (repeated hist_len times)."""
        # first_obs: [obs_dim]
        self.buffer = np.tile(first_obs, (self.hist_len, 1))  # [hist_len, obs_dim]
    
    def add(self, obs: np.ndarray):
        """Add new observation and return flattened history."""
        # Shift buffer and add new obs
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs
    
    def get(self) -> np.ndarray:
        """Get flattened observation history."""
        return self.buffer.flatten()  # [hist_len * obs_dim]


def make_action_fn(
    flow_state: FlowTrainState,
    mu_obs: Optional[jnp.ndarray],
    std_obs: Optional[jnp.ndarray],
    traj_dim: int,
    act_dim: int,
    obs_hist_len: int = 1,
    num_steps: int = 20,
):
    """
    Create a JIT-compiled action function from flow state.
    
    Returns a function that takes (obs_flat, rng) and returns (action, new_rng).
    
    Args:
        flow_state: Trained FlowTrainState
        mu_obs: Observation mean for normalization [base_obs_dim] (or None)
        std_obs: Observation std for normalization [base_obs_dim] (or None)
        traj_dim: Trajectory dimension (train_horizon * act_dim)
        act_dim: Action dimension
        obs_hist_len: Observation history length (for tiling normalization stats)
        num_steps: Number of Euler integration steps
        
    Returns:
        action_fn: Callable[[jnp.ndarray, PRNGKey], Tuple[jnp.ndarray, PRNGKey]]
    """
    # Tile normalization stats to match observation history dimension
    # mu_obs: [base_obs_dim] -> [base_obs_dim * hist_len]
    if mu_obs is not None and obs_hist_len > 1:
        mu_obs_tiled = jnp.tile(mu_obs, obs_hist_len)
        std_obs_tiled = jnp.tile(std_obs, obs_hist_len)
    else:
        mu_obs_tiled = mu_obs
        std_obs_tiled = std_obs
    
    @jax.jit
    def action_fn(obs_flat: jnp.ndarray, rng: jax.random.PRNGKey):
        # Normalize observation if needed
        # obs_flat: [obs_dim * hist_len]
        if mu_obs_tiled is not None:
            obs_norm = (obs_flat - mu_obs_tiled) / (std_obs_tiled + 1e-6)
        else:
            obs_norm = obs_flat
        
        rng, sample_rng = jax.random.split(rng)
        
        # Sample trajectory from flow
        traj = euler_sample_flow(
            flow_state,
            obs_norm,
            sample_rng,
            num_steps=num_steps,
            traj_dim=traj_dim,
        )
        
        # Return first action (first act_dim elements of trajectory)
        action = traj[:act_dim]
        action = jnp.clip(action, -1.0, 1.0)  # Clip to valid action range
        
        return action, rng
    
    return action_fn


# =============================================================================
# Fused train_step (JIT) updating Q, V, Flow together
# =============================================================================



@partial(jax.jit, static_argnames=("train_horizon",))
def train_dual_flow_step(
    flow_pos: FlowTrainState,
    flow_neg: FlowTrainState,
    rng: jax.random.PRNGKey,
    *,
    union_obs: jnp.ndarray,   # [H, B, obs_dim] from sampler (transposed)
    union_acts: jnp.ndarray,  # [H, B, act_dim] from sampler (transposed)
    neg_obs: jnp.ndarray,     # [H, B, obs_dim] from sampler (transposed)
    neg_acts: jnp.ndarray,    # [H, B, act_dim] from sampler (transposed)
    train_horizon: int,
):
    """
    Train both positive and negative flow models on sampled batches.
    
    Input shapes (from sample_trajectory_batch_jax):
        union_obs:  [H, B, obs_dim]
        union_acts: [H, B, act_dim]
        neg_obs:    [H, B, obs_dim]
        neg_acts:   [H, B, act_dim]
    """
    # Transpose to [B, H, D] for easier handling
    u_obs = jnp.transpose(union_obs, (1, 0, 2))   # [B, H, obs_dim]
    u_acts = jnp.transpose(union_acts, (1, 0, 2)) # [B, H, act_dim]
    n_obs = jnp.transpose(neg_obs, (1, 0, 2))     # [B, H, obs_dim]
    n_acts = jnp.transpose(neg_acts, (1, 0, 2))   # [B, H, act_dim]
    
    # Slice to train_horizon (should already be correct, but be safe)
    u_acts = u_acts[:, :train_horizon, :]
    n_acts = n_acts[:, :train_horizon, :]

    Bu, Hu, Au = u_acts.shape
    Bn, Hn, An = n_acts.shape

    # Flatten actions to trajectory: [B, H * act_dim]
    u_traj = u_acts.reshape(Bu, Hu * Au)
    n_traj = n_acts.reshape(Bn, Hn * An)
    
    # Use first observation as conditioning: [B, obs_dim]
    u_cond = u_obs[:, 0, :]
    n_cond = n_obs[:, 0, :]
    
    flow_pos, pos_loss, rng = train_flow_step(
        flow_pos,
        rng,
        obs=u_cond,
        traj=u_traj,
    )

    flow_neg, neg_loss, rng = train_flow_step(
        flow_neg,
        rng,
        obs=n_cond,
        traj=n_traj,
    )

    metrics = {
        "flow_pos_loss": pos_loss,
        "flow_neg_loss": neg_loss,
    }

    return flow_pos, flow_neg, rng, metrics



# =============================================================================
# Evaluation (unchanged logic)
# =============================================================================


def evaluate_policy_jax(
    eval_env,
    action_fn,   # Callable[[jnp.ndarray, PRNGKey], Tuple[jnp.ndarray, PRNGKey]]
    rng: jax.random.PRNGKey,
    obs_hist_len: int = 1,
    base_obs_dim: int = None,
) -> Tuple[float, float, int, jax.random.PRNGKey]:
    """
    Evaluate flow policy in environment with observation history.
    
    Args:
        eval_env: Gymnasium environment
        action_fn: JIT-compiled action function from make_action_fn
        rng: JAX random key (will be updated and returned)
        obs_hist_len: Observation history length
        base_obs_dim: Base observation dimension (before history stacking)
        
    Returns:
        total_reward, total_cost, total_len, new_rng
    """
    obs, _ = eval_env.reset()
    obs = np.asarray(obs)
    
    # Setup observation history buffer
    if base_obs_dim is None:
        base_obs_dim = obs.shape[0]
    
    obs_buffer = ObsHistoryBuffer(obs_hist_len, base_obs_dim)
    obs_buffer.reset(obs)

    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False

    while not done:
        # Get flattened observation history
        obs_flat = obs_buffer.get()
        obs_jax = jnp.asarray(obs_flat)
        
        # Get action from flow policy (action_fn handles normalization)
        action, rng = action_fn(obs_jax, rng)
        action_np = np.asarray(action)

        next_obs, reward, terminated, truncated, info = eval_env.step(action_np)
        
        # Update observation buffer
        obs_buffer.add(np.asarray(next_obs))

        total_reward += reward
        total_cost += info.get("cost", 0.0)
        total_len += 1
        done = terminated or truncated

    return total_reward, total_cost, total_len, rng

def save_checkpoint(path, step, flow_pos, flow_neg, rng, mu_obs, std_obs):
    with open(path, "wb") as f:
        pickle.dump({
            "step": step,
            "flow_pos": flow_pos.params,
            "flow_neg": flow_neg.params,
            "rng": rng,
            "mu_obs": mu_obs,
            "std_obs": std_obs,
        }, f)

def load_checkpoint(ckpt_path, q_state, v_state, flow_state):
    """Restores training states from a pickle checkpoint."""

    if not os.path.exists(ckpt_path):
        print(f"[Checkpoint] No checkpoint found at {ckpt_path}")
        return None, q_state, v_state, flow_state, None, None

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    print(f"[Checkpoint] Loaded from: {ckpt_path}")

    # Restore all params & opt states
    q_state = q_state.replace(
        params=ckpt["q_params"],
        opt_state=ckpt["q_opt_state"]
    )

    if v_state is not None and ckpt["v_params"] is not None:
        v_state = v_state.replace(
            params=ckpt["v_params"],
            opt_state=ckpt["v_opt_state"]
        )

    flow_state = flow_state.replace(
        params=ckpt["flow_params"],
        opt_state=ckpt["flow_opt_state"]
    )

    # rng, normalization stats
    rng = ckpt["rng"]
    mu_obs = ckpt["mu_obs"]
    std_obs = ckpt["std_obs"]

    return ckpt["step"], q_state, v_state, flow_state, mu_obs, std_obs


# =============================================================================
# Main training loop
# =============================================================================


def main(args):
    """
    Dual-flow training main loop (professional version).

    Trains:
        - flow_pos : on union (preferred) trajectories
        - flow_neg : on negative trajectories

    Preserves:
        - logging
        - checkpointing
        - resume
        - normalization
        - obs history
        - JAX compilation discipline
    """

    # ------------------------------------------------------------------
    # 0. Merge CLI args with defaults (CRITICAL for reproducibility)
    # ------------------------------------------------------------------
    config = {**default_cfg}
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v

    # ------------------------------------------------------------------
    # 1. RNG + determinism
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 2. Logging directory (structured, reproducible)
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"seed-{args.seed:03d}-{timestamp}"
    algo = "dual_flow_fm"

    args.log_dir = os.path.join(
        args.log_dir,
        args.experiment,
        args.task,
        algo,
        run_name,
    )
    os.makedirs(args.log_dir, exist_ok=True)

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config({**config, **vars(args)})

    # ------------------------------------------------------------------
    # 3. Environment (ONLY used for dataset + optional eval)
    # ------------------------------------------------------------------
    print(f"Creating environment: {args.task}")
    env = gym.make(args.task)
    env.reset(seed=args.seed)

    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())

    # ------------------------------------------------------------------
    # 4. Load DSRL dataset
    # ------------------------------------------------------------------
    print("\nLoading offline dataset...")

    dataset_config = {
        "density": config["density"],
        "inpaint_ranges": config["inpaint_ranges"],
        "num_negative_trajectories": config["num_negative_trajectories"],
        "num_union_trajectories": config["num_union_trajectories"],
        "non_pref_noise": config["non_pref_noise"],
    }

    raw_data = env.get_dataset()

    # trajectory statistics (important sanity check)
    done_idxs = np.where(
        (raw_data["terminals"] == 1) | (raw_data["timeouts"] == 1)
    )[0]

    traj_lengths, start = [], 0
    for end in done_idxs:
        traj_lengths.append(end - start + 1)
        start = end + 1

    print(
        f"Trajectory stats | mean={np.mean(traj_lengths):.1f}, "
        f"max={np.max(traj_lengths)}, min={np.min(traj_lengths)}"
    )

    # Convert to D4RL-style dict
    d4rl_data = get_dataset_in_d4rl_format(
        env=env,
        config=dataset_config,
        task=args.task,
        ep_len=max(traj_lengths),
        num_folds=config["num_folds"],
    )

    neg_data, union_data = get_neg_and_union_data_2(
        d4rl_data, dataset_config
    )

    # ------------------------------------------------------------------
    # 4.5 Dataset statistics + optional union cost filtering
    # ------------------------------------------------------------------
    neg_costs = neg_data["costs"].sum(axis=1)
    union_costs = union_data["costs"].sum(axis=1)
    union_rewards = union_data["rewards"].sum(axis=1)
    
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Negative costs:  min={neg_costs.min():.2f}, max={neg_costs.max():.2f}, mean={neg_costs.mean():.2f}")
    print(f"Union costs:     min={union_costs.min():.2f}, max={union_costs.max():.2f}, mean={union_costs.mean():.2f}")
    print(f"Union rewards:   min={union_rewards.min():.2f}, max={union_rewards.max():.2f}, mean={union_rewards.mean():.2f}")
    print(f"Union contains {(union_costs > neg_costs.min()).sum()}/{len(union_costs)} trajectories with cost > min negative cost")
    
    # Optional: filter union to only best trajectories (low cost + high reward)
    union_pct = config.get("union_cost_percentile", 100.0)
    if union_pct < 100.0:
        # Target: select same number of trajectories as negative set
        num_neg_traj = len(neg_costs)
        
        # Compute composite score: high reward, low cost
        # Normalize both to [0, 1] range, then combine
        cost_normalized = (union_costs - union_costs.min()) / (union_costs.max() - union_costs.min() + 1e-8)
        reward_normalized = (union_rewards - union_rewards.min()) / (union_rewards.max() - union_rewards.min() + 1e-8)
        
        # Score = high reward - low cost
        # Higher score = better trajectory (low cost, high reward)
        composite_score = reward_normalized - cost_normalized  # Range: [-1, 1]
        
        # Select top num_neg_traj trajectories by composite score
        num_to_select = min(num_neg_traj, len(union_costs))
        top_indices = np.argsort(composite_score)[-num_to_select:]  # Top scores
        
        print(f"\nFiltering union to {num_to_select} best trajectories (matching neg count)")
        print(f"  Selection criteria: low cost + high reward (composite score)")
        
        keys = ["observations", "actions", "rewards", "costs", "terminals", "timeouts"]
        for k in keys:
            union_data[k] = union_data[k][top_indices]
        
        # Recompute stats after filtering
        union_costs_filtered = union_data["costs"].sum(axis=1)
        union_rewards_filtered = union_data["rewards"].sum(axis=1)
        print(f"  Filtered union costs:   min={union_costs_filtered.min():.2f}, max={union_costs_filtered.max():.2f}, mean={union_costs_filtered.mean():.2f}")
        print(f"  Filtered union rewards: min={union_rewards_filtered.min():.2f}, max={union_rewards_filtered.max():.2f}, mean={union_rewards_filtered.mean():.2f}")
    else:
        print(f"\nNo union filtering (union_cost_percentile={union_pct})")
    print("="*70 + "\n")

    # ------------------------------------------------------------------
    # 5. Optional observation normalization
    # ------------------------------------------------------------------
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        print("Normalizing observations...")
        neg_data, union_data, mu_np, std_np = get_normalized_data(
            neg_data, union_data
        )
        mu_obs = jnp.array(mu_np)
        std_obs = jnp.array(std_np)

    # ------------------------------------------------------------------
    # 6. Observation history stacking
    # ------------------------------------------------------------------
    hist_len = config["obs_hist_len"]
    print(f"Using observation history length = {hist_len}")

    neg_obs = neg_data["observations"]
    union_obs = union_data["observations"]

    if hist_len > 1:
        neg_obs = make_obs_history(neg_obs, hist_len)
        union_obs = make_obs_history(union_obs, hist_len)

    neg_obs = jnp.array(neg_obs, dtype=jnp.float32)
    union_obs = jnp.array(union_obs, dtype=jnp.float32)

    neg_acts = jnp.array(neg_data["actions"], dtype=jnp.float32)
    union_acts = jnp.array(union_data["actions"], dtype=jnp.float32)
    
    # Prepare rewards for sampler (shape: [N, T, 1])
    union_rew_np = np.asarray(union_data.get("rewards", np.zeros_like(union_data["actions"][:, :, 0])))
    if union_rew_np.ndim == 2:
        union_rew_np = union_rew_np[..., None]
    union_rew_all = jnp.array(union_rew_np, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # 7. Dimensions
    # ------------------------------------------------------------------
    base_obs_dim = env.observation_space.shape[0]
    obs_dim = base_obs_dim * hist_len
    act_dim = env.action_space.shape[0]
    traj_dim = config["train_horizon"] * act_dim

    print("\nModel dimensions:")
    print(f"  obs_dim   = {obs_dim}")
    print(f"  act_dim   = {act_dim}")
    print(f"  traj_dim  = {traj_dim}")

    # ------------------------------------------------------------------
    # 8. Initialize TWO flow models
    # ------------------------------------------------------------------
    rng, pos_rng, neg_rng = jax.random.split(rng, 3)

    flow_model = FlaxScoreNet(output_dim=traj_dim)

    flow_pos = create_flow_train_state(
        pos_rng,
        flow_model,
        obs_dim,
        traj_dim,
        learning_rate=config["lr"],
        max_grad_norm=config["max_grad_norm"],
    )

    flow_neg = create_flow_train_state(
        neg_rng,
        flow_model,
        obs_dim,
        traj_dim,
        learning_rate=config["lr"],
        max_grad_norm=config["max_grad_norm"],
    )

    # ------------------------------------------------------------------
    # 9. Resume from checkpoint (if requested)
    # ------------------------------------------------------------------
    start_step = 0
    latest_ckpt = os.path.join(args.log_dir, "ckpt_latest.pkl")

    if args.load_ckpt and os.path.exists(latest_ckpt):
        print(f"Resuming from checkpoint: {latest_ckpt}")
        with open(latest_ckpt, "rb") as f:
            ckpt = pickle.load(f)

        flow_pos = flow_pos.replace(params=ckpt["flow_pos"])
        flow_neg = flow_neg.replace(params=ckpt["flow_neg"])
        rng = ckpt["rng"]
        mu_obs = ckpt["mu_obs"]
        std_obs = ckpt["std_obs"]
        start_step = ckpt["step"]

    # ------------------------------------------------------------------
    # 10. Training loop
    # ------------------------------------------------------------------
    total_steps = config["flow_train_iterations"]
    batch_size = config["batch_size"]
    H = config["train_horizon"]

    print("=" * 70)
    print("Starting Dual-Flow Training")
    print("=" * 70)

    pbar = tqdm(range(start_step, total_steps), desc="Dual-Flow Training")

    for step in pbar:
        # ------ Sample random batch of trajectory segments ------
        (neg_obs_batch, neg_acts_batch, 
         union_obs_batch, union_acts_batch, 
         union_rew_batch, rng) = sample_trajectory_batch_jax(
            neg_obs,
            neg_acts,
            union_obs,
            union_acts,
            union_rew_all,
            batch_size=batch_size,
            train_horizon=H,
            rng=rng,
            mu_obs=None,  # Don't normalize here, flow handles it
            std_obs=None,
        )
        
        # ------ Train step with sampled batch ------
        flow_pos, flow_neg, rng, metrics = train_dual_flow_step(
            flow_pos,
            flow_neg,
            rng,
            union_obs=union_obs_batch,   # [H, B, obs_dim]
            union_acts=union_acts_batch, # [H, B, act_dim]
            neg_obs=neg_obs_batch,       # [H, B, obs_dim]
            neg_acts=neg_acts_batch,     # [H, B, act_dim]
            train_horizon=H,
        )

        # ---------------- Logging ----------------
        if (step + 1) % config["log_freq"] == 0:
            logger.log_tabular("Step", step + 1)
            logger.log_tabular("FlowPos_Loss", float(metrics["flow_pos_loss"]))
            logger.log_tabular("FlowNeg_Loss", float(metrics["flow_neg_loss"]))
            logger.dump_tabular()
        # eval
        if args.use_eval and (step + 1) % args.eval_freq == 0:
            # Create action functions for both flow models
            action_fn_pos = make_action_fn(
                flow_pos, mu_obs, std_obs, traj_dim, act_dim,
                obs_hist_len=hist_len,
                num_steps=config["diffusion_steps"]
            )
            action_fn_neg = make_action_fn(
                flow_neg, mu_obs, std_obs, traj_dim, act_dim,
                obs_hist_len=hist_len,
                num_steps=config["diffusion_steps"]
            )
            
            # Evaluate positive flow (union/preferred policy)
            eval_reward_p, eval_cost_p, eval_len_p, rng = evaluate_policy_jax(
                env,
                action_fn_pos,
                rng,
                obs_hist_len=hist_len,
                base_obs_dim=base_obs_dim,
            )

            # Evaluate negative flow
            eval_reward_n, eval_cost_n, eval_len_n, rng = evaluate_policy_jax(
                env,
                action_fn_neg,
                rng,
                obs_hist_len=hist_len,
                base_obs_dim=base_obs_dim,
            )

            print(
                f"\n[Eval @ {step+1}] "
                f"Pos: Reward={eval_reward_p:.2f}, Cost={eval_cost_p:.2f}, Len={eval_len_p} | "
                f"Neg: Reward={eval_reward_n:.2f}, Cost={eval_cost_n:.2f}, Len={eval_len_n}"
            )
            
            # Reconstruction quality check (ablation diagnostic)
            rng, recon_rng_p, recon_rng_n = jax.random.split(rng, 3)
            mse_pos, range_pos, corr_pos = check_flow_reconstruction(
                flow_pos, union_obs, union_acts, traj_dim, H, recon_rng_p
            )
            mse_neg, range_neg, corr_neg = check_flow_reconstruction(
                flow_neg, neg_obs, neg_acts, traj_dim, H, recon_rng_n
            )
            print(
                f"  Reconstruction: Pos MSE={mse_pos:.4f}, corr={corr_pos:.3f} | "
                f"Neg MSE={mse_neg:.4f}, corr={corr_neg:.3f}"
            )
            print(f"  Action ranges: Pos={range_pos:.3f}, Neg={range_neg:.3f}")


        # ---------------- Checkpoint ----------------
        if (step + 1) % config["save_freq"] == 0:
            ckpt = {
                "step": step + 1,
                "flow_pos": flow_pos.params,
                "flow_neg": flow_neg.params,
                "rng": rng,
                "mu_obs": mu_obs,
                "std_obs": std_obs,
            }
            ckpt_path = os.path.join(args.log_dir, "ckpt_latest.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(ckpt, f)

    print("\nTraining complete.")
   

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified IPL + Flow Matching Training (JAX)")

    parser.add_argument("--task", type=str, default="OfflineSwimmerVelocityGymnasium-v1")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--experiment", type=str, default="jax_exp")

    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--normalize_observation", action="store_true", default=False)
    parser.add_argument("--use_guidance", action="store_true", default=True)

    # Default eval == False (as requested)
    parser.add_argument("--use_eval", action="store_true", default=True)
    parser.add_argument("--eval_freq", type=int, default=3000)

    # Hyperparam overrides
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


    parser.add_argument(
    "--load_ckpt",
    action="store_true",
    help="Load latest checkpoint if available."
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        default=20000,
        help="Save checkpoint every N steps."
    )
    
    parser.add_argument(
        "--union_cost_percentile",
        type=float,
        default=100.0,
        help="Filter union data to bottom X%% cost trajectories (100 = no filter, 25 = bottom 25%% only)."
    )
    args = parser.parse_args()
    main(args)
