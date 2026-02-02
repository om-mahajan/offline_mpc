#!/usr/bin/env python3
"""
Modified V5: Single-Step Flow Matching with Trajectory-Based Q/V Training
- Uses single-action flow policy like V6
- Keeps trajectory-based Q/V preference training from V5
- Fixed horizon issues and tensor shape handling
"""

import os
import os.path as osp
import sys
import time
import functools
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../..')))

import gymnasium as gym
import dsrl
import dsrl.offline_safety_gymnasium

from diffusion_SDE.model import ScoreNet, TwinQ, update_target
from dsrl_model.utils.logger import EpochLogger
from dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
    get_normalized_data
)

EP = 1e-6

default_cfg = {
    "log_freq": int(1e3),
    "save_freq": int(2e4),
    "eval_episode_freq": 10,
    "hidden_sizes": [256, 256],
    "max_grad_norm": 1.0,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "train_horizon": 15,
    # IPL / Q pretrain
    "q_pretrain_iterations": int(2.5e5),
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
    "flow_train_iterations": int(3e5),
    "batch_size": 64,
    "device": "cuda",
    "gamma": 0.99,
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


class IPL_TwinQ_Critic(nn.Module):
    """IPL Critic using TwinQ architecture from QGPO"""
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.q_network = TwinQ(action_dim=act_dim, state_dim=obs_dim)
        self.q_target = deepcopy(self.q_network).requires_grad_(False)
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.args = args
    
    def forward(self, obs, act):
        """Returns minimum of two Q values (conservative estimate)"""
        return self.q_network(act, obs)
    
    def both(self, obs, act):
        """Returns both Q values for loss computation"""
        return self.q_network.both(act, obs)
    
    def get_target_q(self, obs, act):
        """Get target Q values"""
        return self.q_target(act, obs)


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


def sample_trajectory_batch_from_splits(neg_data, union_data, batch_size, train_horizon, device, norm_fn=None, data_on_gpu=True):
    """
    Sample trajectory chunks from negative and union numpy arrays.
    Returns: neg_obs, neg_acts, union_obs, union_acts, union_rewards
    Shapes: [horizon, batch, dim]
    """
    if data_on_gpu:
        neg_obs_array = neg_data['observations']
        neg_act_array = neg_data['actions']
        union_obs_array = union_data['observations']
        union_act_array = union_data['actions']
        union_rew_array = union_data['rewards']
        
        n_neg = neg_obs_array.shape[0]
        n_union = union_obs_array.shape[0]
        neg_len = neg_obs_array.shape[1]
        union_len = union_obs_array.shape[1]
        
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
        
        neg_obs = torch.stack(neg_obs_batch).transpose(0, 1)
        neg_acts = torch.stack(neg_act_batch).transpose(0, 1)
        union_obs = torch.stack(union_obs_batch).transpose(0, 1)
        union_acts = torch.stack(union_act_batch).transpose(0, 1)
        union_rew = torch.stack(union_rew_batch).transpose(0, 1)
        
    else:
        neg_obs_array = neg_data['observations']
        neg_act_array = neg_data['actions']
        union_obs_array = union_data['observations']
        union_act_array = union_data['actions']
        union_rew_array = union_data['rewards']
        
        n_neg = neg_obs_array.shape[0]
        n_union = union_obs_array.shape[0]
        neg_len = neg_obs_array.shape[1]
        union_len = union_obs_array.shape[1]

        neg_indices = np.random.randint(0, n_neg, size=batch_size)
        union_indices = np.random.randint(0, n_union, size=batch_size)

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

        neg_obs = torch.as_tensor(np.stack(neg_obs_batch), dtype=torch.float32).transpose(0, 1).to(device)
        neg_acts = torch.as_tensor(np.stack(neg_act_batch), dtype=torch.float32).transpose(0, 1).to(device)
        union_obs = torch.as_tensor(np.stack(union_obs_batch), dtype=torch.float32).transpose(0, 1).to(device)
        union_acts = torch.as_tensor(np.stack(union_act_batch), dtype=torch.float32).transpose(0, 1).to(device)
        union_rew = torch.as_tensor(np.stack(union_rew_batch), dtype=torch.float32).transpose(0, 1).to(device)

    if norm_fn is not None:
        neg_obs = norm_fn(neg_obs)
        union_obs = norm_fn(union_obs)

    return neg_obs, neg_acts, union_obs, union_acts, union_rew


def sample_transition_batch(neg_data, union_data, batch_size, device, norm_fn=None, data_on_gpu=True):
    """
    Sample single-step transitions (s, a, s') for single-action flow matching.
    Returns: dict with 'neg' and 'union' keys, each containing s, a, s_next
    """
    if data_on_gpu:
        neg_obs = neg_data['observations']
        neg_act = neg_data['actions']
        union_obs = union_data['observations']
        union_act = union_data['actions']
        
        n_neg, len_neg = neg_obs.shape[0], neg_obs.shape[1]
        n_union, len_union = union_obs.shape[0], union_obs.shape[1]
        
        neg_traj_idx = torch.randint(0, n_neg, (batch_size,), device=device)
        neg_time_idx = torch.randint(0, len_neg-1, (batch_size,), device=device)
        union_traj_idx = torch.randint(0, n_union, (batch_size,), device=device)
        union_time_idx = torch.randint(0, len_union-1, (batch_size,), device=device)
        
        neg_s = neg_obs[neg_traj_idx, neg_time_idx]
        neg_a = neg_act[neg_traj_idx, neg_time_idx]
        neg_s_next = neg_obs[neg_traj_idx, neg_time_idx + 1]
        
        union_s = union_obs[union_traj_idx, union_time_idx]
        union_a = union_act[union_traj_idx, union_time_idx]
        union_s_next = union_obs[union_traj_idx, union_time_idx + 1]
        
    else:
        neg_obs = neg_data['observations']
        neg_act = neg_data['actions']
        union_obs = union_data['observations']
        union_act = union_data['actions']
        
        n_neg, len_neg = neg_obs.shape[0], neg_obs.shape[1]
        n_union, len_union = union_obs.shape[0], union_obs.shape[1]
        
        neg_traj_idx = np.random.randint(0, n_neg, size=batch_size)
        neg_time_idx = np.random.randint(0, len_neg - 1, size=batch_size)
        union_traj_idx = np.random.randint(0, n_union, size=batch_size)
        union_time_idx = np.random.randint(0, len_union - 1, size=batch_size)
        
        neg_s = torch.as_tensor(neg_obs[neg_traj_idx, neg_time_idx], dtype=torch.float32).to(device)
        neg_a = torch.as_tensor(neg_act[neg_traj_idx, neg_time_idx], dtype=torch.float32).to(device)
        neg_s_next = torch.as_tensor(neg_obs[neg_traj_idx, neg_time_idx + 1], dtype=torch.float32).to(device)
        
        union_s = torch.as_tensor(union_obs[union_traj_idx, union_time_idx], dtype=torch.float32).to(device)
        union_a = torch.as_tensor(union_act[union_traj_idx, union_time_idx], dtype=torch.float32).to(device)
        union_s_next = torch.as_tensor(union_obs[union_traj_idx, union_time_idx + 1], dtype=torch.float32).to(device)
    
    if norm_fn:
        neg_s, neg_s_next = norm_fn(neg_s), norm_fn(neg_s_next)
        union_s, union_s_next = norm_fn(union_s), norm_fn(union_s_next)
    
    return {
        'neg': {'s': neg_s, 'a': neg_a, 's_next': neg_s_next},
        'union': {'s': union_s, 'a': union_a, 's_next': union_s_next}
    }


def v_update_step(q_critic, v_model, v_optimizer, union_obs, union_acts, neg_obs, neg_acts, config):
    """
    Update V(s) to satisfy: V(s) ≈ Q(s,a) - γ V(s')
    Uses minimum of twin Q values for conservative estimation.
    """
    device = union_obs.device
    horizon, batch = union_obs.shape[0], union_obs.shape[1]

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

    with torch.no_grad():
        q_flat = q_critic(flat_obs, flat_acts)
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

    q1_union, q2_union = q_critic.both(flat_union_obs, flat_union_acts)
    q1_neg, q2_neg = q_critic.both(flat_neg_obs, flat_neg_acts)
    
    if q1_union.dim() > 1:
        q1_union = q1_union.squeeze(-1)
        q2_union = q2_union.squeeze(-1)
        q1_neg = q1_neg.squeeze(-1)
        q2_neg = q2_neg.squeeze(-1)
    
    q_union_flat = torch.min(q1_union, q2_union)
    q_neg_flat = torch.min(q1_neg, q2_neg)

    q_union = q_union_flat.reshape(horizon, batch)
    q_neg = q_neg_flat.reshape(horizon, batch)
    
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
    """OT linear interpolation"""
    one_minus_sigma_min = 1.0 - sigma_min
    sigma_t = 1.0 - one_minus_sigma_min * t
    sigma_t = sigma_t.view(-1, 1)
    t = t.view(-1, 1)
    return sigma_t * x0 + t * x1


def u_t_ot(x_t, x1, t, sigma_min):
    """OT vector field"""
    one_minus_sigma_min = 1.0 - sigma_min
    denom = (1.0 - one_minus_sigma_min * t).view(-1, 1)
    denom = torch.clamp(denom, min=1e-6)
    return (x1 - one_minus_sigma_min * x_t) / denom


@torch.no_grad()
def compute_advantage_energy(q_critic, v_model, states, actions, config):
    """
    Compute advantage A(s,a) = Q(s,a) - V(s) for energy guidance.
    Uses TwinQ for robust advantage estimation.
    """
    q_val = q_critic(states, actions)
    if q_val.dim() > 1:
        q_val = q_val.squeeze(-1)
    
    v_val = v_model(states)
    advantage = q_val - v_val
    
    return advantage


def train_flow_matching_step(flow_model, flow_optimizer, q_critic, v_model,
                             states, actions, config, use_guidance=True):
    """Single-step flow matching with advantage-based energy guidance"""
    flow_model.train()
    batch = states.shape[0]
    device = states.device
    
    flow_model.condition = states
    
    # Compute energy for guidance
    energy = None
    if use_guidance:
        energy = compute_advantage_energy(q_critic, v_model, states, actions, config)
    
    # OT path
    x1 = actions
    x0 = torch.randn_like(x1)
    eps = 1e-6
    t = torch.rand(batch, device=device) * (1.0 - 2 * eps) + eps
    sigma_min = config.get('sigma_min', 0.01)
    
    x_t = psi_t_ot(x0, x1, t, sigma_min)
    u_t = u_t_ot(x_t, x1, t, sigma_min)
    
    # Predict velocity
    v_theta = flow_model(x_t, t)
    
    # Per-sample error
    err = torch.sum((v_theta - u_t)**2, dim=1)
    
    # Apply guidance weighting
    if use_guidance and energy is not None:
        alpha = config.get('energy_alpha', 3.0)
        weights = F.softmax(alpha * energy, dim=0).detach()
        
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print("⚠️ WARNING: NaN/Inf in weights, using uniform")
            weights = torch.ones(batch, device=device) / batch
    else:
        weights = torch.ones(batch, device=device) / batch
    
    loss = torch.sum(err * weights)
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("⚠️ WARNING: Invalid loss, skipping")
        flow_model.condition = None
        return 0.0
    
    flow_optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(flow_model.parameters(), config['max_grad_norm'])
    flow_optimizer.step()
    
    flow_model.condition = None
    return loss.item()


@torch.no_grad()
def evaluate_flow_policy(eval_env, flow_model, device, norm_fn, diffusion_steps=15):
    """Evaluate single-step policy"""
    obs, _ = eval_env.reset()
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    
    while not done:
        act = flow_model.select_actions(obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(act)
        
        total_reward += reward
        total_cost += info.get('cost', 0.0)
        total_len += 1
        done = terminated or truncated
        obs = next_obs
    
    return total_reward, total_cost, total_len


def main(args):
    config = {**default_cfg}
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v

    device = torch.device(args.device if isinstance(args.device, str) else f"{args.device}:{getattr(args, 'device_id', 0)}")
    args.device = device

    # Setup logging
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = f"seed-{str(args.seed).zfill(3)}"
    relpath = f"{subfolder}-{relpath}"
    algo = "ipl_flow_twinq_fm_v5_fixed"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config({**config, **vars(args)})

    # Build environment
    print(f"Creating environment: {args.task}")
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)

    # Load dataset
    print(f"\nLoading DSRL dataset...")
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
    print(f"Trajectory length: mean={np.mean(traj_lengths):.1f}, max={max_traj_len}")
    
    d4rl_data = get_dataset_in_d4rl_format(
        env=eval_env,
        config=dataset_config,
        task=args.task,
        ep_len=max_traj_len,
        num_folds=config.get("num_folds", 1)
    )
    
    neg_data, union_data = get_neg_and_union_data_2(d4rl_data, dataset_config)
    
    # Normalize
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        print("Normalizing observations...")
        neg_data, union_data, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
        mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32).to(device)
        std_obs = torch.as_tensor(std_obs, dtype=torch.float32).to(device)
    
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    
    # Preload to GPU
    if args.preload_to_gpu:
        print("\nPre-loading to GPU...")
        for key in neg_data.keys():
            neg_data[key] = torch.as_tensor(neg_data[key], dtype=torch.float32).to(device)
            union_data[key] = torch.as_tensor(union_data[key], dtype=torch.float32).to(device)
        print(f"✅ GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    print(f"\nDataset: neg={neg_data['observations'].shape}, union={union_data['observations'].shape}")
    
    # Get dimensions
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    
    # Create models
    print("\nInitializing models...")
    
    q_critic = IPL_TwinQ_Critic(obs_dim=obs_dim, act_dim=act_dim, args=args).to(device)
    q_opt = Adam(q_critic.parameters(), lr=config.get("q_lr", 1e-4), weight_decay=config.get("weight_decay", 1e-5))

    v_model = VNetwork(obs_dim=obs_dim, hidden_size=config["v_hidden"]).to(device)
    v_opt = Adam(v_model.parameters(), lr=config.get("v_lr", 1e-4), weight_decay=config.get("weight_decay", 1e-5))

    # ✅ FIXED: Single-action flow model (NOT trajectory-based)
    flow_model = ScoreNet(
        input_dim=obs_dim + act_dim,  # Condition on obs, predict action
        output_dim=act_dim,            # Single action output
        marginal_prob_std=None,        # Flow matching mode
        args=args
    ).to(device)
    flow_opt = Adam(flow_model.parameters(), lr=config.get("lr", 1e-4), weight_decay=config.get("weight_decay", 1e-5))
    
    print(f"✅ TwinQ critic: obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"✅ Flow model: input_dim={obs_dim + act_dim}, output_dim={act_dim}")

    # Training variables
    q_pretrain_iters = config["q_pretrain_iterations"]
    flow_train_iters = config["flow_train_iterations"]
    batch_size = config["batch_size"]
    train_horizon = config["train_horizon"]
    v_updates_per_q = config.get("v_updates_per_q_update", 1)
    use_guidance = getattr(args, 'use_guidance', True)

    print("=" * 60)
    print(f"TwinQ pretrain iterations: {q_pretrain_iters}")
    print(f"Flow train iterations: {flow_train_iters}")
    print(f"Batch size: {batch_size}, horizon (for Q/V): {train_horizon}")
    print(f"Use guidance: {use_guidance}")
    print("=" * 60)

    # ============= PHASE 1: Pretrain TwinQ & V =============
    print("\nPhase 1: TwinQ & V pretraining (trajectory-based)")
    pbar = tqdm(range(q_pretrain_iters), desc="Phase1:TwinQ_V", unit="iter")
    start_time = time.time()
    
    for step in pbar:
        neg_obs, neg_acts, union_obs, union_acts, union_rew = sample_trajectory_batch_from_splits(
            neg_data, union_data, batch_size, train_horizon, device, 
            norm_fn if args.normalize_observation else None,
            data_on_gpu=args.preload_to_gpu
        )

        q_stats = q_preference_step(
            q_critic=q_critic,
            q_optimizer=q_opt,
            neg_obs=neg_obs,
            neg_acts=neg_acts,
            union_obs=union_obs,
            union_acts=union_acts,
            config=config
        )

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
        
        if (step + 1) % config["target_update_freq"] == 0:
            update_target(q_critic.q_network, q_critic.q_target, tau=config["target_tau"])

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

        if (step + 1) % config["save_freq"] == 0 or (step + 1) == q_pretrain_iters:
            logger.torch_save(itr=step+1, torch_saver_elements=q_critic, prefix="q_critic")
            logger.torch_save(itr=step+1, torch_saver_elements=v_model, prefix="v")

    print("TwinQ & V pretraining finished.")

    # ============= VISUALIZATION: Q, V, Advantage Analysis =============
    print("\n" + "="*60)
    print("Analyzing Q, V, and Advantage values across dataset...")
    print("="*60)
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    @torch.no_grad()
    def analyze_dataset_values(q_critic, v_model, neg_data, union_data, config, device, norm_fn):
        """Compute Q, V, Adv for all trajectories and visualize"""
        q_critic.eval()
        v_model.eval()
        
        # Sample trajectories from both sets
        n_samples = min(100, neg_data['observations'].shape[0], union_data['observations'].shape[0])
        
        neg_obs = neg_data['observations'][:n_samples]  # [n_samples, T, obs_dim]
        neg_act = neg_data['actions'][:n_samples]
        union_obs = union_data['observations'][:n_samples]
        union_act = union_data['actions'][:n_samples]
        
        if norm_fn is not None:
            neg_obs = norm_fn(neg_obs)
            union_obs = norm_fn(union_obs)
        
        def compute_trajectory_values(obs, act):
            """Compute Q, V, Adv for trajectory batch"""
            n_traj, T, obs_dim = obs.shape
            act_dim = act.shape[-1]
            
            # Flatten: [n_traj, T, dim] -> [n_traj*T, dim]
            obs_flat = obs.reshape(-1, obs_dim)
            act_flat = act.reshape(-1, act_dim)
            
            # Compute Q and V
            q_vals = q_critic(obs_flat, act_flat)
            if q_vals.dim() > 1:
                q_vals = q_vals.squeeze(-1)
            v_vals = v_model(obs_flat)
            
            # Reshape back: [n_traj*T] -> [n_traj, T]
            q_vals = q_vals.reshape(n_traj, T)
            v_vals = v_vals.reshape(n_traj, T)
            adv_vals = q_vals - v_vals
            
            # Compute trajectory-level statistics (sum over time)
            gamma = config.get("gamma", 1.0)
            if gamma == 1.0:
                q_traj = q_vals.sum(dim=1)  # [n_traj]
                v_traj = v_vals.sum(dim=1)
                adv_traj = adv_vals.sum(dim=1)
            else:
                T_actual = q_vals.shape[1]
                discounts = torch.tensor(
                    [gamma**i for i in range(T_actual)],
                    device=device,
                    dtype=q_vals.dtype
                ).unsqueeze(0)  # [1, T]
                q_traj = (q_vals * discounts).sum(dim=1)
                v_traj = (v_vals * discounts).sum(dim=1)
                adv_traj = (adv_vals * discounts).sum(dim=1)
            
            return q_traj, v_traj, adv_traj, q_vals, v_vals, adv_vals
        
        # Compute for both datasets
        neg_q, neg_v, neg_adv, neg_q_time, neg_v_time, neg_adv_time = compute_trajectory_values(neg_obs, neg_act)
        union_q, union_v, union_adv, union_q_time, union_v_time, union_adv_time = compute_trajectory_values(union_obs, union_act)
        
        # Move to CPU for plotting
        neg_q = neg_q.cpu().numpy()
        neg_v = neg_v.cpu().numpy()
        neg_adv = neg_adv.cpu().numpy()
        union_q = union_q.cpu().numpy()
        union_v = union_v.cpu().numpy()
        union_adv = union_adv.cpu().numpy()
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Q-values distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(neg_q, bins=30, alpha=0.6, label='Negative', color='red', density=True)
        ax1.hist(union_q, bins=30, alpha=0.6, label='Union (Preferred)', color='green', density=True)
        ax1.axvline(neg_q.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Neg Mean: {neg_q.mean():.2f}')
        ax1.axvline(union_q.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Union Mean: {union_q.mean():.2f}')
        ax1.set_xlabel('Q-value (trajectory sum)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Q-Value Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. V-values distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(neg_v, bins=30, alpha=0.6, label='Negative', color='red', density=True)
        ax2.hist(union_v, bins=30, alpha=0.6, label='Union (Preferred)', color='green', density=True)
        ax2.axvline(neg_v.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Neg Mean: {neg_v.mean():.2f}')
        ax2.axvline(union_v.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Union Mean: {union_v.mean():.2f}')
        ax2.set_xlabel('V-value (trajectory sum)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('V-Value Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Advantage distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(neg_adv, bins=30, alpha=0.6, label='Negative', color='red', density=True)
        ax3.hist(union_adv, bins=30, alpha=0.6, label='Union (Preferred)', color='green', density=True)
        ax3.axvline(neg_adv.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Neg Mean: {neg_adv.mean():.2f}')
        ax3.axvline(union_adv.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Union Mean: {union_adv.mean():.2f}')
        ax3.set_xlabel('Advantage (Q-V, trajectory sum)', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Advantage Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Scatter: Q vs V (colored by dataset)
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(neg_q, neg_v, c='red', alpha=0.5, s=30, label='Negative', edgecolors='darkred')
        ax4.scatter(union_q, union_v, c='green', alpha=0.5, s=30, label='Union (Preferred)', edgecolors='darkgreen')
        ax4.plot([min(neg_q.min(), union_q.min()), max(neg_q.max(), union_q.max())],
                 [min(neg_v.min(), union_v.min()), max(neg_v.max(), union_v.max())],
                 'k--', alpha=0.3, label='Q=V line')
        ax4.set_xlabel('Q-value', fontsize=12)
        ax4.set_ylabel('V-value', fontsize=12)
        ax4.set_title('Q vs V Scatter', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Gradient plot: trajectories sorted by Q-value
        ax5 = plt.subplot(2, 3, 5)
        
        # Sort trajectories by Q-value
        neg_sorted_idx = np.argsort(neg_q)
        union_sorted_idx = np.argsort(union_q)
        
        n_neg = len(neg_q)
        n_union = len(union_q)
        
        # Create color gradient
        neg_colors = cm.Reds(np.linspace(0.3, 0.9, n_neg))
        union_colors = cm.Greens(np.linspace(0.3, 0.9, n_union))
        
        # Plot sorted Q-values with gradient
        x_neg = np.arange(n_neg)
        x_union = np.arange(n_union) + n_neg + 5  # Offset for separation
        
        for i in range(n_neg):
            ax5.bar(x_neg[i], neg_q[neg_sorted_idx[i]], color=neg_colors[i], width=1.0)
        for i in range(n_union):
            ax5.bar(x_union[i], union_q[union_sorted_idx[i]], color=union_colors[i], width=1.0)
        
        ax5.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax5.axvline(n_neg + 2.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax5.text(n_neg/2, ax5.get_ylim()[1]*0.9, 'Negative\nTrajectories', 
                ha='center', fontsize=12, fontweight='bold', color='darkred')
        ax5.text(n_neg + 5 + n_union/2, ax5.get_ylim()[1]*0.9, 'Union (Preferred)\nTrajectories', 
                ha='center', fontsize=12, fontweight='bold', color='darkgreen')
        ax5.set_xlabel('Trajectory Index (sorted by Q-value)', fontsize=12)
        ax5.set_ylabel('Q-value', fontsize=12)
        ax5.set_title('Q-Value Gradient (Sorted Trajectories)', fontsize=14, fontweight='bold')
        ax5.grid(alpha=0.3, axis='y')
        
        # 6. Advantage gradient plot
        ax6 = plt.subplot(2, 3, 6)
        
        # Sort by advantage
        neg_adv_sorted_idx = np.argsort(neg_adv)
        union_adv_sorted_idx = np.argsort(union_adv)
        
        for i in range(n_neg):
            ax6.bar(x_neg[i], neg_adv[neg_adv_sorted_idx[i]], color=neg_colors[i], width=1.0)
        for i in range(n_union):
            ax6.bar(x_union[i], union_adv[union_adv_sorted_idx[i]], color=union_colors[i], width=1.0)
        
        ax6.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax6.axvline(n_neg + 2.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax6.text(n_neg/2, ax6.get_ylim()[1]*0.9, 'Negative', 
                ha='center', fontsize=12, fontweight='bold', color='darkred')
        ax6.text(n_neg + 5 + n_union/2, ax6.get_ylim()[1]*0.9, 'Union (Preferred)', 
                ha='center', fontsize=12, fontweight='bold', color='darkgreen')
        ax6.set_xlabel('Trajectory Index (sorted by Advantage)', fontsize=12)
        ax6.set_ylabel('Advantage (Q-V)', fontsize=12)
        ax6.set_title('Advantage Gradient (Sorted Trajectories)', fontsize=14, fontweight='bold')
        ax6.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        viz_path = os.path.join(args.log_dir, 'q_v_advantage_analysis.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {viz_path}")
        plt.close()
        
        # Print statistics
        print("\n" + "="*60)
        print("DATASET VALUE STATISTICS")
        print("="*60)
        print(f"\nNEGATIVE Trajectories (n={len(neg_q)}):")
        print(f"  Q-value:  mean={neg_q.mean():.3f}, std={neg_q.std():.3f}, min={neg_q.min():.3f}, max={neg_q.max():.3f}")
        print(f"  V-value:  mean={neg_v.mean():.3f}, std={neg_v.std():.3f}, min={neg_v.min():.3f}, max={neg_v.max():.3f}")
        print(f"  Advantage: mean={neg_adv.mean():.3f}, std={neg_adv.std():.3f}, min={neg_adv.min():.3f}, max={neg_adv.max():.3f}")
        
        print(f"\nUNION (Preferred) Trajectories (n={len(union_q)}):")
        print(f"  Q-value:  mean={union_q.mean():.3f}, std={union_q.std():.3f}, min={union_q.min():.3f}, max={union_q.max():.3f}")
        print(f"  V-value:  mean={union_v.mean():.3f}, std={union_v.std():.3f}, min={union_v.min():.3f}, max={union_v.max():.3f}")
        print(f"  Advantage: mean={union_adv.mean():.3f}, std={union_adv.std():.3f}, min={union_adv.min():.3f}, max={union_adv.max():.3f}")
        
        print(f"\nDIFFERENCE (Union - Negative):")
        print(f"  Δ Q-value:  {union_q.mean() - neg_q.mean():.3f} ({'✅ POSITIVE' if union_q.mean() > neg_q.mean() else '❌ NEGATIVE'})")
        print(f"  Δ V-value:  {union_v.mean() - neg_v.mean():.3f}")
        print(f"  Δ Advantage: {union_adv.mean() - neg_adv.mean():.3f}")
        
        # Preference accuracy
        preference_correct = (union_q > neg_q[:len(union_q)]).sum() if len(union_q) == len(neg_q) else None
        if preference_correct is not None:
            accuracy = preference_correct / len(union_q) * 100
            print(f"\nPreference Accuracy: {accuracy:.1f}% ({preference_correct}/{len(union_q)} trajectories)")
            print(f"  (How often Q(union) > Q(negative) for paired trajectories)")
        
        print("="*60 + "\n")
        
        return {
            'neg_q': neg_q, 'neg_v': neg_v, 'neg_adv': neg_adv,
            'union_q': union_q, 'union_v': union_v, 'union_adv': union_adv
        }
    
    # Run analysis
    analysis_results = analyze_dataset_values(
        q_critic, v_model, neg_data, union_data, config, device, 
        norm_fn if args.normalize_observation else None
    )
    
    # Freeze Q and V
    for p in q_critic.parameters():
        p.requires_grad = False
    q_critic.eval()
    for p in v_model.parameters():
        p.requires_grad = False
    v_model.eval()

    # ============= PHASE 2: Train single-action flow =============
    print("\nPhase 2: Train single-action flow with TwinQ-V advantages")
    best_flow_reward = -float('inf')
    pbar = tqdm(range(flow_train_iters), desc="Phase2:Flow", unit="iter")
    start_time = time.time()
    loss_history = []

    for step in pbar:
        # Sample single transitions for single-action flow matching
        batch_data = sample_transition_batch(
            neg_data, union_data, batch_size, device,
            norm_fn if args.normalize_observation else None,
            data_on_gpu=args.preload_to_gpu
        )
        
        states = batch_data['union']['s']
        actions = batch_data['union']['a']
        
        flow_loss_value = train_flow_matching_step(
            flow_model=flow_model,
            flow_optimizer=flow_opt,
            q_critic=q_critic,
            v_model=v_model,
            states=states,
            actions=actions,
            config=config,
            use_guidance=use_guidance
        )
        
        loss_history.append(flow_loss_value)

        if (step + 1) % 1000 == 0:
            if use_guidance:
                # Compute diagnostics
                with torch.no_grad():
                    energy = compute_advantage_energy(q_critic, v_model, states, actions, config)
                    guidance_weights = F.softmax(config['energy_alpha'] * energy, dim=0)
                print(f"\n[Flow Diagnostics @ step {step+1}]")
                print(f"  Energy - min: {energy.min():.3f}, max: {energy.max():.3f}, mean: {energy.mean():.3f}")
                print(f"  Guidance - min: {guidance_weights.min():.6f}, max: {guidance_weights.max():.6f}")
                print(f"  Guidance entropy: {-(guidance_weights * torch.log(guidance_weights + 1e-8)).sum():.3f}")
            else:
                print(f"\n[Flow Diagnostics @ step {step+1}] Guidance disabled.")

            if len(loss_history) > 100:
                recent_losses = loss_history[-100:]
                print(f"  Recent 100 steps - Loss mean: {np.mean(recent_losses):.6f}, std: {np.std(recent_losses):.6f}")

        if (step + 1) % config["log_freq"] == 0 or (step + 1) == flow_train_iters:
            elapsed = time.time() - start_time
            pbar.set_description(f"Phase2 | Flow_loss={flow_loss_value:.6f}")
            logger.log(f"Flow step {step+1}/{flow_train_iters} flow_loss={flow_loss_value:.6f} time={elapsed:.1f}s")

        if args.use_eval and ((step + 1) % (args.eval_freq or 1000) == 0):
            eval_reward, eval_cost, eval_len = 0.0, 0.0, 0
            
            for ep_idx in range(config["eval_episode_freq"]):
                try:
                    r, c, l = evaluate_flow_policy(eval_env, flow_model, device, norm_fn)
                    eval_reward += r
                    eval_cost += c
                    eval_len += l
                except Exception as e:
                    print(f"Warning: Evaluation episode {ep_idx} failed: {e}")
                    continue
                
            num_episodes = config["eval_episode_freq"]
            eval_reward /= num_episodes
            eval_cost /= num_episodes
            eval_len /= num_episodes
            
            print(f"\n[Eval Step {step+1}] Reward: {eval_reward:.2f}, Cost: {eval_cost:.2f}, Len: {eval_len:.2f}")
            
            logger.log_tabular("Flow/Step", step + 1)
            logger.log_tabular("Flow/Loss", flow_loss_value)
            logger.log_tabular("Eval/Reward", eval_reward)
            logger.log_tabular("Eval/Cost", eval_cost)
            logger.log_tabular("Eval/Length", eval_len)
            logger.dump_tabular()
            
            if eval_reward > best_flow_reward:
                best_flow_reward = eval_reward
                logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow_best")
                print(f"  -> New best reward: {best_flow_reward:.2f}")

        if (step + 1) % config["save_freq"] == 0 or (step + 1) == flow_train_iters:
            logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow")
            logger.torch_save(itr=step+1, torch_saver_elements=q_critic, prefix="q_critic")
            logger.torch_save(itr=step+1, torch_saver_elements=v_model, prefix="v")

    # Final saves
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=flow_model, prefix="flow_final")
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=q_critic, prefix="q_critic_final")
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=v_model, prefix="v_final")
    logger.close()
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best evaluation reward: {best_flow_reward:.2f}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "--env", dest="task", default="OfflineSwimmerVelocityGymnasium-v1")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--experiment", type=str, default="twinq_fixed_hist")
    parser.add_argument("--normalize_observation", action="store_true", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--schedule", type=str, default="Linear")
    parser.add_argument("--energy_alpha", type=float, default=3.0)
    parser.add_argument("--cost_weight_temp", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_horizon", type=int, default=15)
    parser.add_argument("--q_pretrain_iterations", type=int, default=250000)
    parser.add_argument("--flow_train_iterations", type=int, default=300000)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--use_eval", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--target_update_freq", type=int, default=10)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--num_negative_trajectories", type=int, default=50)
    parser.add_argument("--num_union_trajectories", type=int, default=-1)
    parser.add_argument("--non_pref_noise", type=float, default=0.0)
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--use_guidance", action="store_true", default=True)
    parser.add_argument("--preload_to_gpu", action="store_true", default=True)
    
    args = parser.parse_args()
    main(args)