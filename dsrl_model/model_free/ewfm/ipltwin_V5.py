#!/usr/bin/env python3
"""
Modified training script to train a Flow Matching (FM-OT) model instead of score matching.
- Replaces diffusion denoising loss with Flow Matching OT loss
- Adds option `--use_guidance` to enable/disable energy-weighted guidance
- Keeps TwinQ / V pretraining unchanged; uses computed advantages as optional guidance

"""

import os
import os.path as osp
import random
import sys
import time
import functools
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm

# Add the repo root to path (adjust if needed)
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../..')))



# DSRL imports
import gymnasium as gym
import dsrl
import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # registers envs

# diffusion / previous ScoreNet imports (we reuse ScoreNet API but now it produces a vector field)
from diffusion_SDE.model import ScoreNet, TwinQ, update_target

# utilities
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.utils import get_params_norm

# Import dsrl_dataset functions
from dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
    get_normalized_data
)

EP = 1e-6

# -------------------------
# Default config
# -------------------------
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
    # Diffusion (not used for training now, but preserved for compatibility)
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
    # weight temperature
    "energy_alpha": 3.0,
    "cost_weight_temp": 1.0,
    # Target network update
    "target_update_freq": 10,
    "target_tau": 0.005,
    # Iterations
    "flow_train_iterations": int(1e5),
    "batch_size": 128,
    "device": "cuda",
    # gamma
    "gamma": 0.99,
    # V updates
    "v_use_neg_in_updates": True,
    # DSRL dataset config
    "density": 1.0,
    "inpaint_ranges": ((0.25, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
    "non_pref_noise": 0.0,
    "num_folds": 1,
    # Flow Matching specific
    "sigma_min": 0.01,  # OT path sigma_min
}


# -------------------------
# Utilities
# -------------------------
def normalize_observation(mu_obs, std_obs, obs):
    if mu_obs is None:
        return obs
    if isinstance(obs, torch.Tensor):
        mu = mu_obs.to(obs.device) if isinstance(mu_obs, torch.Tensor) else torch.tensor(mu_obs, device=obs.device, dtype=obs.dtype)
        std = std_obs.to(obs.device) if isinstance(std_obs, torch.Tensor) else torch.tensor(std_obs, device=obs.device, dtype=obs.dtype)
        return (obs - mu) / (std + EP)
    else:
        mu = mu_obs if not isinstance(mu_obs, torch.Tensor) else mu_obs.cpu().numpy()
        std = std_obs if not isinstance(std_obs, torch.Tensor) else std_obs.cpu().numpy()
        return (obs - mu) / (std + EP)


# -------------------------
# IPL TwinQ Critic (uses TwinQ from model.py)
# -------------------------
class IPL_TwinQ_Critic(nn.Module):
    """
    IPL Critic using TwinQ architecture from QGPO.
    Adds preference learning on top of twin Q-functions for robust Q-value estimation.
    """
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        # Use TwinQ from model.py
        self.q_network = TwinQ(action_dim=act_dim, state_dim=obs_dim)
        self.q_target = deepcopy(self.q_network).requires_grad_(False)
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.args = args
    
    def forward(self, obs, act):
        """Returns minimum of two Q values (conservative estimate, standard in SAC/TD3)"""
        return self.q_network(act, obs)
    
    def both(self, obs, act):
        """Returns both Q values for loss computation"""
        return self.q_network.both(act, obs)
    
    def get_target_q(self, obs, act):
        """Get target Q values"""
        return self.q_target(act, obs)


# -------------------------
# V-network
# -------------------------
class VNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


# -------------------------
# sample batch function (adapted for dsrl_dataset format)
# -------------------------
# (unchanged from original) -- keep sample_trajectory_batch_from_splits

# -------------------------
# IPL Q pretraining with TwinQ
# -------------------------
# (unchanged) q_preference_step

# -------------------------
# V update with TwinQ
# -------------------------
# (unchanged) v_update_step

# -------------------------
# Compute weights from frozen TwinQ & V
# -------------------------
@torch.no_grad()
def compute_weights_from_qv(q_critic, v_model, union_obs, union_acts, config):
    """
    Compute per-timestep energy weights from Q-V advantages.
    Uses minimum of twin Q values for conservative advantage estimation.
    
    Returns:
        energy: [horizon, batch] - raw energy values (not yet softmaxed)
    """
    horizon, batch = union_obs.shape[0], union_obs.shape[1]
    device = union_obs.device

    flat_obs = union_obs.reshape(-1, union_obs.shape[-1])
    flat_acts = union_acts.reshape(-1, union_acts.shape[-1])

    # Get Q value (minimum of twin Qs)
    q_flat = q_critic(flat_obs, flat_acts)
    v_flat = v_model(flat_obs).reshape(horizon, batch)
    # Ensure q_flat is 1D
    if q_flat.dim() > 1:
        q_flat = q_flat.squeeze(-1)
    q_flat = q_flat.reshape(horizon, batch)
    # Compute advantages
    adv = q_flat - v_flat
    # Center around MEAN (not max) for better stability
    adv_centered = adv - torch.mean(adv)
    
    # Scale by temperature
    temp = config.get("cost_weight_temp", 0.5)
    energy = adv_centered * temp
    
    # Optional: Add debug info (remove after debugging)
    if torch.isnan(energy).any() or torch.isinf(energy).any():
        print(f" WARNING: NaN or Inf in energy!")
        print(f"   Q range: [{q_flat.min():.3f}, {q_flat.max():.3f}]")
        print(f"   V range: [{v_flat.min():.3f}, {v_flat.max():.3f}]")
        print(f"   Adv range: [{adv.min():.3f}, {adv.max():.3f}]")
    
    return energy  # [horizon, batch]


# ===============================
# FLOW MATCHING (OT) TRAINING (trajectory-based)
# ===============================

def sample_trajectory_batch_from_splits(neg_data, union_data, batch_size, train_horizon, device, norm_fn=None, data_on_gpu=True):
    """
    Sample trajectory chunks from negative and union numpy arrays.
    neg_data/union_data: dict with keys ['observations', 'actions', 'rewards', 'costs', ...]
    Each is a numpy array of shape [num_trajectories, ep_len, dim] (or torch tensor if data_on_gpu=True)
    
    Returns: neg_obs, neg_acts, union_obs, union_acts, union_rewards
    Shapes: [horizon, batch, dim]
    """
    if data_on_gpu:
        # Data already on GPU as tensors
        neg_obs_array = neg_data['observations']
        neg_act_array = neg_data['actions']
        union_obs_array = union_data['observations']
        union_act_array = union_data['actions']
        union_rew_array = union_data['rewards']
        
        n_neg = neg_obs_array.shape[0]
        n_union = union_obs_array.shape[0]
        neg_len = neg_obs_array.shape[1]
        union_len = union_obs_array.shape[1]
        
        # Sample indices
        neg_indices = torch.randint(0, n_neg, (batch_size,), device=device)
        union_indices = torch.randint(0, n_union, (batch_size,), device=device)
        
        max_start_neg = max(1, neg_len - train_horizon)
        max_start_union = max(1, union_len - train_horizon)
        neg_starts = torch.randint(0, max_start_neg, (batch_size,), device=device)
        union_starts = torch.randint(0, max_start_union, (batch_size,), device=device)
        
        neg_obs_batch, neg_act_batch = [], []
        union_obs_batch, union_act_batch, union_rew_batch = [], [], []
        
        for i in range(batch_size):
            nidx = neg_indices[i].item()
            nstart = neg_starts[i].item()
            uidx = union_indices[i].item()
            ustart = union_starts[i].item()
            
            neg_obs_batch.append(neg_obs_array[nidx, nstart:nstart + train_horizon])
            neg_act_batch.append(neg_act_array[nidx, nstart:nstart + train_horizon])
            
            union_obs_batch.append(union_obs_array[uidx, ustart:ustart + train_horizon])
            union_act_batch.append(union_act_array[uidx, ustart:ustart + train_horizon])
            union_rew_batch.append(union_rew_array[uidx, ustart:ustart + train_horizon])
        
        # stack and transpose -> [horizon, batch, dim]
        neg_obs = torch.stack(neg_obs_batch).transpose(0, 1)
        neg_acts = torch.stack(neg_act_batch).transpose(0, 1)
        union_obs = torch.stack(union_obs_batch).transpose(0, 1)
        union_acts = torch.stack(union_act_batch).transpose(0, 1)
        union_rew = torch.stack(union_rew_batch).transpose(0, 1)
        
    else:
        # Original numpy path
        neg_obs_array = neg_data['observations']
        neg_act_array = neg_data['actions']
        union_obs_array = union_data['observations']
        union_act_array = union_data['actions']
        union_rew_array = union_data['rewards']
        
        n_neg = neg_obs_array.shape[0]
        n_union = union_obs_array.shape[0]
        neg_len = neg_obs_array.shape[1]
        union_len = union_obs_array.shape[1]

        # Sample trajectory indices
        neg_indices = np.random.randint(0, n_neg, size=batch_size)
        union_indices = np.random.randint(0, n_union, size=batch_size)

        # Sample starting points
        max_start_neg = max(1, neg_len - train_horizon)
        max_start_union = max(1, union_len - train_horizon)
        neg_starts = np.random.randint(0, max_start_neg, size=batch_size)
        union_starts = np.random.randint(0, max_start_union, size=batch_size)

        neg_obs_batch, neg_act_batch = [], []
        union_obs_batch, union_act_batch, union_rew_batch = [], [], []

        for i in range(batch_size):
            nidx = neg_indices[i]
            nstart = neg_starts[i]
            uidx = union_indices[i]
            ustart = union_starts[i]

            neg_obs_batch.append(neg_obs_array[nidx, nstart:nstart + train_horizon])
            neg_act_batch.append(neg_act_array[nidx, nstart:nstart + train_horizon])

            union_obs_batch.append(union_obs_array[uidx, ustart:ustart + train_horizon])
            union_act_batch.append(union_act_array[uidx, ustart:ustart + train_horizon])
            union_rew_batch.append(union_rew_array[uidx, ustart:ustart + train_horizon])

        # stack and transpose -> [horizon, batch, dim]
        neg_obs = torch.as_tensor(np.stack(neg_obs_batch), dtype=torch.float32).transpose(0, 1).to(device)
        neg_acts = torch.as_tensor(np.stack(neg_act_batch), dtype=torch.float32).transpose(0, 1).to(device)
        union_obs = torch.as_tensor(np.stack(union_obs_batch), dtype=torch.float32).transpose(0, 1).to(device)
        union_acts = torch.as_tensor(np.stack(union_act_batch), dtype=torch.float32).transpose(0, 1).to(device)
        union_rew = torch.as_tensor(np.stack(union_rew_batch), dtype=torch.float32).transpose(0, 1).to(device)

    if norm_fn is not None:
        neg_obs = norm_fn(neg_obs)
        union_obs = norm_fn(union_obs)

    return neg_obs, neg_acts, union_obs, union_acts, union_rew

def v_update_step(q_critic, v_model, v_optimizer, union_obs, union_acts, neg_obs, neg_acts, config):
    """
    Update V(s) to satisfy: V(s) ≈ Q(s,a) - γ V(s')
    Uses minimum of twin Q values for conservative estimation.
    """
    device = union_obs.device
    horizon, batch = union_obs.shape[0], union_obs.shape[1]

    # Build flat transitions from union set
    union_next_obs = torch.roll(union_obs, shifts=-1, dims=0)
    union_next_obs[-1] = union_obs[-1].clone()

    flat_union_obs = union_obs.reshape(-1, union_obs.shape[-1])
    flat_union_acts = union_acts.reshape(-1, union_acts.shape[-1])
    flat_union_next = union_next_obs.reshape(-1, union_next_obs.shape[-1])

    if config.get("v_use_neg_in_updates", True):
        neg_next_obs = torch.roll(neg_obs, shifts=-1, dims=0)
        neg_next_obs[-1] = neg_obs[-1].clone()
        flat_neg_obs = neg_obs.reshape(-1, neg_obs.shape[-1])
        flat_neg_acts = neg_acts.reshape(-1, neg_acts.shape[-1])
        flat_neg_next = neg_next_obs.reshape(-1, neg_next_obs.shape[-1])

        flat_obs = torch.cat([flat_union_obs, flat_neg_obs], dim=0)
        flat_acts = torch.cat([flat_union_acts, flat_neg_acts], dim=0)
        flat_next = torch.cat([flat_union_next, flat_neg_next], dim=0)
    else:
        flat_obs = flat_union_obs
        flat_acts = flat_union_acts
        flat_next = flat_union_next

    # Compute targets using minimum of twin Q
    with torch.no_grad():
        q_flat = q_critic(flat_obs, flat_acts)  # Uses min(Q1, Q2)
        # Ensure q_flat is 1D
        if q_flat.dim() > 1:
            q_flat = q_flat.squeeze(-1)
        v_next = v_model(flat_next)
        gamma = config.get("gamma", 1.0)
        target = q_flat - gamma * v_next

    v_pred = v_model(flat_obs)
    v_loss = F.mse_loss(v_pred, target)

    v_optimizer.zero_grad()
    v_loss.backward()
    clip_grad_norm_(v_model.parameters(), config["max_grad_norm"])
    v_optimizer.step()

    return {"v_loss": v_loss.detach().item()}

def q_preference_step(q_critic, q_optimizer, neg_obs, neg_acts, union_obs, union_acts, config):
    """
    Preference loss: union trajectory preferred over negative
    Uses TwinQ for robust Q-value estimation with conservative min(Q1, Q2).
    """
    horizon, batch = union_obs.shape[0], union_obs.shape[1]
    device = union_obs.device

    flat_union_obs = union_obs.reshape(-1, union_obs.shape[-1])
    flat_union_acts = union_acts.reshape(-1, union_acts.shape[-1])
    flat_neg_obs = neg_obs.reshape(-1, neg_obs.shape[-1])
    flat_neg_acts = neg_acts.reshape(-1, neg_acts.shape[-1])

    # Compute Q per time-step using BOTH Q networks
    # TwinQ.both returns (q1, q2) where each is shape [batch*horizon, 1] or [batch*horizon]
    q1_union, q2_union = q_critic.both(flat_union_obs, flat_union_acts)
    q1_neg, q2_neg = q_critic.both(flat_neg_obs, flat_neg_acts)
    
    # Ensure they are 1D tensors [batch*horizon]
    if q1_union.dim() > 1:
        q1_union = q1_union.squeeze(-1)
        q2_union = q2_union.squeeze(-1)
        q1_neg = q1_neg.squeeze(-1)
        q2_neg = q2_neg.squeeze(-1)
    
    # Take minimum (conservative estimate, like in SAC/TD3)
    q_union_flat = torch.min(q1_union, q2_union)
    q_neg_flat = torch.min(q1_neg, q2_neg)

    q_union = q_union_flat.reshape(horizon, batch)
    q_neg = q_neg_flat.reshape(horizon, batch)
    # Trajectory scores (discounted sum)
    gamma = config.get("gamma", 1.0)
    if gamma == 1.0:
        s_union = q_union.sum(dim=0)
        s_neg = q_neg.sum(dim=0)
    else:
        discounts = torch.tensor(
            [gamma ** i for i in range(horizon)], 
            device=device, 
            dtype=q_union.dtype
        ).unsqueeze(1)
        s_union = (q_union * discounts).sum(dim=0)
        s_neg = (q_neg * discounts).sum(dim=0)

    logits = s_union - s_neg
    labels = torch.ones_like(logits)
    pref_loss = F.binary_cross_entropy_with_logits(logits, labels)

    # Twin Q regularization (regularize both Q networks)
    lambda_q = config.get("lambda_q_reg", 1e-2)
    q1_reg = (q1_union.pow(2).mean() + q1_neg.pow(2).mean()) * 0.5
    q2_reg = (q2_union.pow(2).mean() + q2_neg.pow(2).mean()) * 0.5
    q_reg = (q1_reg + q2_reg) * 0.5
    
    score_reg = (s_union.pow(2).mean() + s_neg.pow(2).mean()) * 0.5
    reg_loss = lambda_q * (q_reg + 0.1 * score_reg)

    loss = pref_loss + reg_loss

    q_optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(q_critic.parameters(), config["max_grad_norm"])
    q_optimizer.step()

    return {
        "pref_loss": pref_loss.detach().item(),
        "reg_loss": reg_loss.detach().item(),
        "total_q_loss": loss.detach().item(),
        "q1_mean": q1_union.mean().item(),
        "q2_mean": q2_union.mean().item(),
        "q_diff": (q1_union - q2_union).abs().mean().item()
    }

def psi_t_ot(x0, x1, t, sigma_min):
    """OT linear interpolation (per paper Eq.20-22)
    x0: [B, D] noise
    x1: [B, D] data trajectory (flattened)
    t: [B] in [0,1]
    returns x_t: [B, D]
    """
    one_minus_sigma_min = 1.0 - sigma_min
    sigma_t = 1.0 - one_minus_sigma_min * t  # [B]
    sigma_t = sigma_t.view(-1, 1)
    t = t.view(-1, 1)
    return sigma_t * x0 + t * x1


def u_t_ot(x_t, x1, t, sigma_min):
    """OT vector field evaluated at x_t (paper Eq.21)
    u_t(x|x1) = (x1 - (1-sigma_min) * x) / (1 - (1-sigma_min) * t)
    x_t: [B, D]
    x1: [B, D]
    t: [B]
    returns u: [B, D]
    """
    one_minus_sigma_min = 1.0 - sigma_min
    denom = (1.0 - one_minus_sigma_min * t).view(-1, 1)
    denom = torch.clamp(denom, min=1e-6)
    return (x1 - one_minus_sigma_min * x_t) / denom


def train_flow_matching_step(flow_model, flow_optimizer, union_obs, union_acts, weights, args, config, use_guidance=True):
    """
    Train the model using Flow Matching with OT path.
    - union_acts: [H, B, act_dim]
    - union_obs:  [H, B, obs_dim] (used as condition)
    - weights: [H, B] or None (raw energy values)
    - use_guidance: bool whether to apply energy-guided weights
    Returns scalar loss
    """
    flow_model.train()
    horizon, batch, obs_dim = union_obs.shape
    _, _, act_dim = union_acts.shape

    device = union_obs.device

    # Conditioning on the first observation (like original script)
    cond_obs = union_obs[0]  # [B, obs_dim]
    flow_model.condition = cond_obs.to(device)

    # Prepare data: flatten trajectory to [B, D]
    acts_reshaped = union_acts.permute(1, 0, 2)  # [B, H, act_dim]
    x1 = acts_reshaped.reshape(batch, -1)       # [B, D]

    # Sample x0 ~ N(0, I) in trajectory space
    x0 = torch.randn_like(x1)

    # Sample t ~ Uniform(eps, 1-eps)
    eps = 1e-6
    random_t = torch.rand(batch, device=device) * (1.0 - 2 * eps) + eps

    sigma_min = config.get('sigma_min', 0.01)

    # Build OT interpolation x_t and target u_t
    x_t = psi_t_ot(x0, x1, random_t, sigma_min)   # [B, D]
    u_t = u_t_ot(x_t, x1, random_t, sigma_min)     # [B, D]

    # Model prediction: v_theta(x_t, t)
    # ScoreNet in original repo had a conditioning mechanism. We keep that.
    v_theta = flow_model(x_t, random_t)  # expected [B, D]

    # Reshape to per-timestep form: [B, H, act_dim]
    v_theta_ts = v_theta.reshape(batch, horizon, act_dim)
    u_t_ts = u_t.reshape(batch, horizon, act_dim)

    # Per-timestep squared errors
    per_step_err = torch.sum((v_theta_ts - u_t_ts)**2, dim=2)  # [B, H]

    # Guidance handling
    if use_guidance and (weights is not None):
        # weights given as [H, B] in compute_weights_from_qv; transpose to [B, H]
        energy = weights.transpose(0, 1)  # [B, H]
        alpha = config.get('energy_alpha', 3.0)
        # clip for numerical stability
        max_clip = 50.0
        energy_clipped = torch.clamp(energy, -max_clip, max_clip)
        guidance = F.softmax(alpha * energy_clipped, dim=1).detach()  # [B, H]
        if torch.isnan(guidance).any() or torch.isinf(guidance).any():
            print("⚠️  WARNING: NaN/Inf in guidance weights! Using uniform weights.")
            guidance = torch.ones_like(guidance) / horizon
    else:
        guidance = torch.ones(batch, horizon, device=device) / float(horizon)

    # Weighted loss across timesteps -> average over batch
    loss = torch.mean(torch.sum(per_step_err * guidance, dim=1))

    if torch.isnan(loss) or torch.isinf(loss):
        print("⚠️  WARNING: Invalid loss! Skipping this step.")
        flow_model.condition = None
        return 0.0

    flow_optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(flow_model.parameters(), config['max_grad_norm'])
    flow_optimizer.step()

    flow_model.condition = None
    return loss.item()


@torch.no_grad()
def evaluate_flow_policy(eval_env, score_model, device, norm_fn, 
                         diffusion_steps=15, horizon=5, act_dim=None):
    # using same evaluation as before - model.select_trajectory_actions should still work
    eval_obs, _ = eval_env.reset()
    eval_obs = torch.as_tensor(norm_fn(eval_obs), dtype=torch.float32, device=device).unsqueeze(0)
    
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    
    while not done:
        act = score_model.select_trajectory_actions(
            eval_obs, 
            diffusion_steps=diffusion_steps,
            horizon=horizon,
            act_dim=act_dim,
            use_first_action=True
        )
        
        next_obs, reward, terminated, truncated, info = eval_env.step(act[0])
        eval_obs = torch.as_tensor(norm_fn(next_obs), dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward += reward
        total_cost += info.get("cost", 0.0)
        total_len += 1
        done = terminated or truncated
    
    return total_reward, total_cost, total_len


# -------------------------
# Main training entrypoint
# -------------------------
def main(args):
    # Merge user args with default config
    config = {**default_cfg}
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v

    # device
    device = torch.device(args.device if isinstance(args.device, str) else f"{args.device}:{getattr(args, 'device_id', 0)}")
    args.device = device

    # Setup marginal_prob_std for compatibility (not used for FM training)
    # keep the function if model uses args.marginal_prob_std_fn internally
    try:
        from diffusion_SDE.schedule import marginal_prob_std
        marginal_prob_std_fn = functools.partial(marginal_prob_std, schedule=args.schedule, device=device)
        args.marginal_prob_std_fn = marginal_prob_std_fn
    except Exception:
        args.marginal_prob_std_fn = None

    # Setup logging & experiment dirs
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = "ipl_flow_twinq_fm"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config({**config, **vars(args)})

    # Build environment
    print(f"Creating environment: {args.task}")
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)

    # ============= Load dataset using dsrl_dataset.py =============
    print(f"\nLoading DSRL dataset using dsrl_dataset.py...")
    
    dataset_config = {
        "density": config.get("density", 1.0),
        "inpaint_ranges": config.get("inpaint_ranges", []),
        "num_negative_trajectories": config.get("num_negative_trajectories", 50),
        "num_union_trajectories": config.get("num_union_trajectories", -1),
        "non_pref_noise": config.get("non_pref_noise", 0.0),
    }
    
    raw_data = eval_env.get_dataset()
    
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
    
    print(f"Using ep_len={ep_len} for d4rl format conversion")
    
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
        mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32).to(device)
        std_obs = torch.as_tensor(std_obs, dtype=torch.float32).to(device)
    
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    
    # PRE-LOAD DATA TO GPU for faster training
    if args.preload_to_gpu:
        print("\nPre-loading entire dataset to GPU...")
        for key in neg_data.keys():
            neg_data[key] = torch.as_tensor(neg_data[key], dtype=torch.float32).to(device)
            union_data[key] = torch.as_tensor(union_data[key], dtype=torch.float32).to(device)
        print(f"✅ Dataset loaded to GPU. Using {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    else:
        print("Dataset kept in CPU memory (will transfer batches during training)")
    
    print(f"\nDataset statistics:")
    print(f"  Negative set: {neg_data['observations'].shape}")
    print(f"  Union set: {union_data['observations'].shape}")
    print(f"  Observation dim: {neg_data['observations'].shape[-1]}")
    print(f"  Action dim: {neg_data['actions'].shape[-1]}")
    
    # ============= End dataset loading =============

    # Get dimensions
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    
    # Create models
    print("\nInitializing models...")
    
    # Use TwinQ critic instead of simple QNetwork
    q_critic = IPL_TwinQ_Critic(
        obs_dim=obs_dim, 
        act_dim=act_dim, 
        args=args
    ).to(device)
    q_opt = Adam(
        q_critic.parameters(), 
        lr=config.get("q_lr", 1e-4), 
        weight_decay=config.get("weight_decay", 1e-5)
    )

    v_model = VNetwork(obs_dim=obs_dim, hidden_size=config["v_hidden"]).to(device)
    v_opt = Adam(
        v_model.parameters(), 
        lr=config.get("v_lr", 1e-4), 
        weight_decay=config.get("weight_decay", 1e-5)
    )

    # Flow model (ScoreNet used as a convenient net wrapper) -- now used as vector-field network
    train_horizon = config.get("train_horizon", 5)
    traj_dim = train_horizon * act_dim
    flow_model = ScoreNet(
        input_dim=obs_dim + traj_dim,
        output_dim=traj_dim,
        marginal_prob_std=None,
        args=args
    ).to(device)
    flow_opt = Adam(
        flow_model.parameters(), 
        lr=config.get("lr", 1e-4), 
        weight_decay=config.get("weight_decay", 1e-5)
    )
    
    assert flow_model.output_dim == train_horizon * act_dim, "flow_model.output_dim must equal H * act_dim"
    # keep pre_sort_condition expectation for compatibility
    assert (flow_model.pre_sort_condition is not None)

    # Training variables
    q_pretrain_iters = config["q_pretrain_iterations"]
    flow_train_iters = config["flow_train_iterations"]
    batch_size = config["batch_size"]
    train_horizon = config["train_horizon"]
    v_updates_per_q = config.get("v_updates_per_q_update", 1)

    use_guidance = getattr(args, 'use_guidance', True)

    print("=" * 60)
    print(f"TwinQ pretrain iterations: {q_pretrain_iters}")
    print(f"Flow (TwinQ+V-weighted) train iterations: {flow_train_iters}")
    print(f"Batch size (trajectories): {batch_size}, horizon: {train_horizon}")
    print(f"Use guidance: {use_guidance}")
    print("=" * 60)

    # ============= PHASE 1: Pretrain TwinQ & V =============
    print("\nPhase 1: TwinQ & V pretraining (IPL-style)")
    pbar = tqdm(range(q_pretrain_iters), desc="Phase1:TwinQ_V", unit="iter", dynamic_ncols=True)
    start_time = time.time()
    
    for step in pbar:
        # Sample batch using dsrl_dataset format
        neg_obs, neg_acts, union_obs, union_acts, union_rew = sample_trajectory_batch_from_splits(
            neg_data, union_data, batch_size, train_horizon, device, 
            norm_fn if args.normalize_observation else None,
            data_on_gpu=args.preload_to_gpu
        )

        # Q preference update (now with TwinQ)
        q_stats = q_preference_step(
            q_critic=q_critic,
            q_optimizer=q_opt,
            neg_obs=neg_obs,
            neg_acts=neg_acts,
            union_obs=union_obs,
            union_acts=union_acts,
            config=config
        )

        # V updates
        v_stats = {"v_loss": None}
        for _ in range(v_updates_per_q):
            v_stats = v_update_step(
                q_critic=q_critic,
                v_model=v_model,
                v_optimizer=v_opt,
                union_obs=union_obs,
                union_acts=union_acts,
                neg_obs=neg_obs,
                neg_acts=neg_acts,
                config=config
            )
        
        # Update target network (like in QGPO)
        if (step + 1) % config["target_update_freq"] == 0:
            update_target(q_critic.q_network, q_critic.q_target, tau=config["target_tau"])

        # Logging
        if (step + 1) % config["log_freq"] == 0 or (step + 1) == q_pretrain_iters:
            elapsed = time.time() - start_time
            pbar.set_description(
                f"Phase1 | Qpref={q_stats['pref_loss']:.4f} | "
                f"Vloss={v_stats['v_loss']:.4f} | "
                f"Q1={q_stats['q1_mean']:.2f} Q2={q_stats['q2_mean']:.2f}"
            )
            logger.log(
                f"Pretrain step {step+1}/{q_pretrain_iters} "
                f"Qpref={q_stats['pref_loss']:.4f} "
                f"Qreg={q_stats['reg_loss']:.6f} "
                f"Vloss={v_stats['v_loss']:.6f} "
                f"Q1_mean={q_stats['q1_mean']:.3f} "
                f"Q2_mean={q_stats['q2_mean']:.3f} "
                f"Q_diff={q_stats['q_diff']:.4f} "
                f"time={elapsed:.1f}s"
            )

        # Save checkpoints periodically
        if (step + 1) % config["save_freq"] == 0 or (step + 1) == q_pretrain_iters:
            logger.torch_save(itr=step+1, torch_saver_elements=q_critic, prefix="q_critic")
            logger.torch_save(itr=step+1, torch_saver_elements=v_model, prefix="v")

    print("TwinQ & V pretraining finished.")

    # Freeze Q and V
    for p in q_critic.parameters():
        p.requires_grad = False
    q_critic.eval()
    for p in v_model.parameters():
        p.requires_grad = False
    v_model.eval()

    # ============= PHASE 2: Train flow using TwinQ,V-derived weights ============
    print("\nPhase 2: Train flow (weighted by TwinQ-V advantages)")
    best_flow_reward = -float('inf')
    pbar = tqdm(range(flow_train_iters), desc="Phase2:Flow", unit="iter", dynamic_ncols=True)
    start_time = time.time()
    loss_history = []

    for step in pbar:
        # Sample batch
        neg_obs, neg_acts, union_obs, union_acts, union_rew = sample_trajectory_batch_from_splits(
            neg_data, union_data, batch_size, train_horizon, device,
            norm_fn if args.normalize_observation else None,
            data_on_gpu=args.preload_to_gpu
        )

        # Compute weights from frozen TwinQ & V (optional - used only if use_guidance True)
        weights = None
        if use_guidance:
            weights = compute_weights_from_qv(
                q_critic=q_critic, 
                v_model=v_model, 
                union_obs=union_obs, 
                union_acts=union_acts, 
                config=config
            )

        # Flow model update using Flow Matching OT objective
        flow_loss_value = train_flow_matching_step(
            flow_model=flow_model,
            flow_optimizer=flow_opt,
            union_obs=union_obs,
            union_acts=union_acts,
            weights=weights,
            args=args,
            config=config,
            use_guidance=use_guidance
        )
        
        loss_history.append(flow_loss_value)

        # Diagnostic logging every 1000 steps
        if (step + 1) % 1000 == 0:
            if weights is not None:
                energy_t = weights.transpose(0, 1)
                guidance_test = torch.softmax(config['energy_alpha'] * energy_t, dim=1)
                print(f"\n[Flow Diagnostics @ step {step+1}]")
                print(f"  Energy - min: {weights.min():.3f}, max: {weights.max():.3f}, mean: {weights.mean():.3f}")
                print(f"  Guidance - min: {guidance_test.min():.6f}, max: {guidance_test.max():.6f}")
                print(f"  Guidance entropy: {-(guidance_test * torch.log(guidance_test + 1e-8)).sum(dim=1).mean():.3f}")
            else:
                print(f"\n[Flow Diagnostics @ step {step+1}] Guidance disabled (uniform weights).")

            if len(loss_history) > 100:
                recent_losses = loss_history[-100:]
                recent_mean = np.mean(recent_losses)
                recent_std = np.std(recent_losses)
                print(f"  Recent 100 steps - Loss mean: {recent_mean:.6f}, std: {recent_std:.6f}")

        # Logging
        if (step + 1) % config["log_freq"] == 0 or (step + 1) == flow_train_iters:
            elapsed = time.time() - start_time
            pbar.set_description(f"Phase2 | Flow_loss={flow_loss_value:.6f}")
            logger.log(f"Flow step {step+1}/{flow_train_iters} flow_loss={flow_loss_value:.6f} time={elapsed:.1f}s")

        # Periodic evaluation
        if args.use_eval and ((step + 1) % (args.eval_freq or 1000) == 0):
            eval_reward, eval_cost, eval_len = 0.0, 0.0, 0
            
            for ep_idx in range(config["eval_episode_freq"]):
                try:
                    r, c, l = evaluate_flow_policy(
                        eval_env, flow_model, device, norm_fn,
                        diffusion_steps=args.diffusion_steps,
                        horizon=config["train_horizon"],
                        act_dim=act_dim
                    )
                    eval_reward += r
                    eval_cost += c
                    eval_len += l
                except Exception as e:
                    print(f"Warning: Evaluation episode {ep_idx} failed: {e}")
                    continue
                
            num_episodes = config["eval_episode_freq"]
            print(f"Completed {num_episodes} eval episodes.")
            eval_reward /= num_episodes
            eval_cost /= num_episodes
            eval_len /= num_episodes
            
            print(f"\n[Eval Step {step+1}] Reward: {eval_reward:.2f}, Cost: {eval_cost:.2f}, Len: {eval_len:.2f}")
            
            elapsed = time.time() - start_time
            logger.log_tabular("Flow/Step", step + 1)
            logger.log_tabular("Flow/Loss", flow_loss_value)
            logger.log_tabular("Eval/Reward", eval_reward)
            logger.log_tabular("Eval/Cost", eval_cost)
            logger.log_tabular("Eval/Length", eval_len)
            logger.log_tabular("Time/ElapsedSec", elapsed)
            logger.dump_tabular()
            
            # Save best model
            if eval_reward > best_flow_reward:
                best_flow_reward = eval_reward
                logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow_best")
                print(f"  -> New best reward: {best_flow_reward:.2f}")

        # Checkpoint saving
        if (step + 1) % config["save_freq"] == 0 or (step + 1) == flow_train_iters:
            logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow")
            logger.torch_save(itr=step+1, torch_saver_elements=q_critic, prefix="q_critic")
            logger.torch_save(itr=step+1, torch_saver_elements=v_model, prefix="v")

    # Final saves
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=flow_model, prefix="flow")
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=q_critic, prefix="q_critic")
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=v_model, prefix="v")
    logger.close()
    print("Training complete.")
    print(f"Best evaluation reward: {best_flow_reward:.2f}")


# -------------------------
# CLI args
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "--env", dest="task", default="OfflineSwimmerVelocityGymnasium-v1",
                        help="DSRL task name")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--experiment", type=str, default="twinq")
    parser.add_argument("--normalize_observation", action="store_true", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--diffusion_steps", type=int, default=15)
    parser.add_argument("--energy_alpha", type=float, default=3.0,
                        help="Temperature for energy-guided loss (default: 1.0)")
    parser.add_argument("--cost_weight_temp", type=float, default=1.0,
                        help="Temperature for advantage scaling (default: 0.5)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_horizon", type=int, default=5)
    parser.add_argument("--q_pretrain_iterations", type=int, default=150000)
    parser.add_argument("--flow_train_iterations", type=int, default=50000)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--use_eval", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=3)
    
    # TwinQ specific
    parser.add_argument("--target_update_freq", type=int, default=10,
                        help="Frequency of target network updates")
    parser.add_argument("--target_tau", type=float, default=0.005,
                        help="Soft update coefficient for target network")
    
    # DSRL dataset specific arguments
    parser.add_argument("--density", type=float, default=1.0,
                        help="Density for dataset preprocessing")
    parser.add_argument("--num_negative_trajectories", type=int, default=50,
                        help="Number of negative (non-preferred) trajectories")
    parser.add_argument("--num_union_trajectories", type=int, default=-1,
                        help="Number of union trajectories (-1 for all remaining)")
    parser.add_argument("--non_pref_noise", type=float, default=0.0,
                        help="Noise level in preference labels (0.0 to 1.0)")
    parser.add_argument("--num_folds", type=int, default=1,
                        help="Number of temporal folds for downsampling")
    
    # Guidance toggle for ablation
    parser.add_argument("--use_guidance", action="store_true", default=True,
                        help="Enable energy-weighted guidance for flow loss (ablation: toggle off to disable)")

    # GPU optimization
    parser.add_argument("--preload_to_gpu", action="store_true", default=True,
                        help="Pre-load entire dataset to GPU for faster training")
    
    args = parser.parse_args()
    main(args)
