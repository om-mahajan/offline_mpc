#!/usr/bin/env python3
"""
Unified IPL-weighted Flow Matching training script (revised)

Integrates IPL (Inverse Preference Learning) with Energy-Weighted Flow Matching (EWFM).
Phase 1: Alternate Q (preference) updates and V (value) updates.
Phase 2: Freeze Q & V, compute per-timestep advantages adv_t = Q(s,a) - V(s)
         and use them to compute weights for EWFM/ScoreNet updates.

Notes:
 - Designed to be dropped into your repo (uses ScoreNet, DSRLSafetyDataset, EpochLogger from your project)
 - Tune hyperparams (lambda_q_reg, q_pretrain_iterations, v_updates_per_q_update, etc.)
"""
import os
import os.path as osp
import random
import sys
from pathlib import Path
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

current_file = Path(__file__).resolve()
offline_mpc_dir = current_file.parents[3]  # Go up 3 levels to reach offline_mpc
sys.path.insert(0, str(offline_mpc_dir))

# Add the repo root to path (adjust if needed)
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../../..')))

# DSRL imports (assumes these packages/modules are installed/available)
import gymnasium as gym
import dsrl
import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # registers envs

# diffusion / EWFM imports (from your codebase)
from diffusion_SDE.loss import loss_fn as diffusion_loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, update_target  # ScoreNet: flow model

# small utilities from your repo (kept)
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.utils import get_params_norm

# Local dataset adapter (from Script A)
from dsrl_adapter import DSRLSafetyDataset

EP = 1e-6

# -------------------------
# Default config (merged and extended for IPL + V)
# -------------------------
default_cfg = {
    # Logging / checkpoint
    "log_freq": int(2e4),
    "save_freq": int(5e4),
    "eval_episode_freq": 3,
    "hidden_sizes": [256, 256],
    "max_grad_norm": 1.0,
    # Optimization
    "lr": 3e-4,
    "weight_decay": 0.0,
    # Diffusion
    "diffusion_steps": 15,
    "train_horizon": 5,
    # IPL / Q pretrain
    "q_pretrain_iterations": int(20000),
    "q_lr": 3e-4,
    "q_hidden": 256,
    # Value network (V)
    "v_lr": 3e-4,
    "v_hidden": 256,
    "v_updates_per_q_update": 1,
    # Q regularizer (prevents unbounded growth)
    "lambda_q_reg": 1e-3,
    # weight temperature and clipping
    "cost_weight_temp": 1.0,
    "weight_clip_min": 1e-6,
    "weight_clip_max": 100.0,
    # Iterations (defaults — override via CLI args)
    "flow_train_iterations": int(1e6),      # phase 2 total iterations
    "batch_size": 64,
    "device": "cuda",
    # gamma (temporal discounting inside trajectory sums / Bellman updates)
    "gamma": 0.99,
    # whether to include neg transitions in V updates for more coverage
    "v_use_neg_in_updates": True,
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
# Q-network (simple MLP mapping [s,a] -> scalar)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        input_dim = obs_dim + act_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [N, obs_dim + act_dim]
        return self.net(x).squeeze(-1)  # [N]


# -------------------------
# V-network (value: V(s) -> scalar)
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
        return self.net(s).squeeze(-1)  # [N]


# -------------------------
# sample batch function (adapted from your original)
# -------------------------
def sample_trajectory_batch_from_splits(dataset_splits, batch_size, train_horizon, device, norm_fn=None):
    """
    Sample trajectory chunks from negative and union sets
    Returns: neg_obs, neg_acts, union_obs, union_acts, union_rewards
    Shapes: [horizon, batch, dim]
    """
    neg_data = dataset_splits['negative']
    union_data = dataset_splits['union']
    neg_len = neg_data['observations'].shape[1]
    union_len = union_data['observations'].shape[1]

    neg_indices = torch.randint(0, len(neg_data['observations']), (batch_size,))
    union_indices = torch.randint(0, len(union_data['observations']), (batch_size,))

    max_start_neg = max(1, neg_len - train_horizon)
    max_start_union = max(1, union_len - train_horizon)
    neg_starts = torch.randint(0, max_start_neg, (batch_size,))
    union_starts = torch.randint(0, max_start_union, (batch_size,))

    neg_obs_batch, neg_act_batch = [], []
    union_obs_batch, union_act_batch, union_rew_batch = [], [], []

    for i in range(batch_size):
        nidx = neg_indices[i].item()
        nstart = neg_starts[i].item()
        uidx = union_indices[i].item()
        ustart = union_starts[i].item()

        neg_obs_batch.append(neg_data['observations'][nidx, nstart:nstart + train_horizon])
        neg_act_batch.append(neg_data['actions'][nidx, nstart:nstart + train_horizon])

        union_obs_batch.append(union_data['observations'][uidx, ustart:ustart + train_horizon])
        union_act_batch.append(union_data['actions'][uidx, ustart:ustart + train_horizon])
        union_rew_batch.append(union_data.get('rewards', torch.zeros_like(union_data['actions'][uidx, ustart:ustart + train_horizon])))

    # stack and transpose -> [horizon, batch, dim]
    neg_obs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in neg_obs_batch]).transpose(0, 1).to(device)
    neg_acts = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in neg_act_batch]).transpose(0, 1).to(device)
    union_obs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in union_obs_batch]).transpose(0, 1).to(device)
    union_acts = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in union_act_batch]).transpose(0, 1).to(device)
    union_rew = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in union_rew_batch]).transpose(0, 1).to(device)

    if norm_fn is not None:
        neg_obs = norm_fn(neg_obs)
        union_obs = norm_fn(union_obs)

    return neg_obs, neg_acts, union_obs, union_acts, union_rew


# -------------------------
# IPL Q pretraining (preference loss + Q reg)
# -------------------------
def q_preference_step(q_model, q_optimizer, neg_obs, neg_acts, union_obs, union_acts, config):
    """
    Preference loss: union trajectory preferred over negative
    Adds L2 regularizer on Q outputs to prevent unbounded solutions.
    """
    horizon, batch = union_obs.shape[0], union_obs.shape[1]
    device = union_obs.device

    flat_union_obs = union_obs.reshape(-1, union_obs.shape[-1])
    flat_union_acts = union_acts.reshape(-1, union_acts.shape[-1])
    flat_neg_obs = neg_obs.reshape(-1, neg_obs.shape[-1])
    flat_neg_acts = neg_acts.reshape(-1, neg_acts.shape[-1])

    # compute Q per time-step
    q_union_flat = q_model(torch.cat([flat_union_obs, flat_union_acts], dim=1))  # [H*B]
    q_neg_flat = q_model(torch.cat([flat_neg_obs, flat_neg_acts], dim=1))        # [H*B]

    q_union = q_union_flat.reshape(horizon, batch)
    q_neg = q_neg_flat.reshape(horizon, batch)

    # trajectory scores (discounted sum)
    gamma = config.get("gamma", 1.0)
    if gamma == 1.0:
        s_union = q_union.sum(dim=0)  # [B]
        s_neg = q_neg.sum(dim=0)
    else:
        discounts = torch.tensor([gamma ** i for i in range(horizon)], device=device, dtype=q_union.dtype).unsqueeze(1)
        s_union = (q_union * discounts).sum(dim=0)
        s_neg = (q_neg * discounts).sum(dim=0)

    logits = s_union - s_neg  # [B]
    labels = torch.ones_like(logits)

    pref_loss = F.binary_cross_entropy_with_logits(logits, labels)

    # Q regularizer: L2 on Q outputs (flat)
    lambda_q = config.get("lambda_q_reg", 1e-3)
    q_reg = (q_union_flat.pow(2).mean() + q_neg_flat.pow(2).mean()) * 0.5
    reg_loss = lambda_q * q_reg

    loss = pref_loss + reg_loss

    q_optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(q_model.parameters(), config["max_grad_norm"])
    q_optimizer.step()

    return {
        "pref_loss": pref_loss.detach().item(),
        "reg_loss": reg_loss.detach().item(),
        "total_q_loss": loss.detach().item()
    }


# -------------------------
# V update: Bellman-style regression using implied reward r ≈ Q - γ V(s')
# -------------------------
def v_update_step(q_model, v_model, v_optimizer, union_obs, union_acts, neg_obs, neg_acts, config):
    """
    Update V(s) to satisfy: V(s) ≈ Q(s,a) - γ V(s')
    We'll use sampled transitions from union (and optionally negative) chunks:
      - create flat (s, a, s_next) by shifting along horizon
      - use target = Q(s,a) - γ * V(s_next). Detach Q and V(s_next) as targets.
    Minimizes MSE(V(s), target)
    """
    device = union_obs.device
    horizon, batch = union_obs.shape[0], union_obs.shape[1]

    # build flat transitions from union set
    # union_next_obs: shift by -1 and copy last
    union_next_obs = torch.roll(union_obs, shifts=-1, dims=0)
    union_next_obs[-1] = union_obs[-1].clone()

    flat_union_obs = union_obs.reshape(-1, union_obs.shape[-1])
    flat_union_acts = union_acts.reshape(-1, union_acts.shape[-1])
    flat_union_next = union_next_obs.reshape(-1, union_next_obs.shape[-1])

    # Optionally include negatives to expand distribution for V learning
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

    # Compute targets: target = Q(s,a) - γ * V(s_next)
    with torch.no_grad():
        q_flat = q_model(torch.cat([flat_obs, flat_acts], dim=1))  # [N]
        v_next = v_model(flat_next)                                # [N]
        gamma = config.get("gamma", 1.0)
        target = q_flat - gamma * v_next                           # [N]

    v_pred = v_model(flat_obs)  # [N]
    v_loss = F.mse_loss(v_pred, target)

    v_optimizer.zero_grad()
    v_loss.backward()
    clip_grad_norm_(v_model.parameters(), config["max_grad_norm"])
    v_optimizer.step()

    return {"v_loss": v_loss.detach().item()}


# -------------------------
# Compute weights from frozen Q & V (phase 2)
# -------------------------
@torch.no_grad()
def compute_weights_from_qv(q_model, v_model, union_obs, union_acts, config):
    """
    Compute per-time-step advantage adv_t = Q(s_t,a_t) - V(s_t)
    Then convert to weights with temperature and normalization:
       w = exp(adv_t - 1)^temp ; then normalize by mean and clip.
    Returns weights with shape [horizon, batch]
    """
    horizon, batch = union_obs.shape[0], union_obs.shape[1]
    device = union_obs.device

    flat_obs = union_obs.reshape(-1, union_obs.shape[-1])
    flat_acts = union_acts.reshape(-1, union_acts.shape[-1])
    flat_inputs = torch.cat([flat_obs, flat_acts], dim=1)

    q_flat = q_model(flat_inputs).reshape(horizon, batch)  # [H,B]
    v_flat = v_model(flat_obs).reshape(horizon, batch)     # [H,B]

    adv = q_flat - v_flat  # [H,B]

    temp = config.get("cost_weight_temp", 1.0)
    weights = torch.exp(adv - 1.0) ** temp

    # normalize and clip
    mean_w = torch.mean(weights)
    weights = weights / (mean_w + EP)
    wmin, wmax = config.get("weight_clip_min", 1e-6), config.get("weight_clip_max", 100.0)
    weights = torch.clamp(weights, min=wmin, max=wmax)
    return weights


# -------------------------
# train flow with weights (unchanged)
# -------------------------
def train_weighted_flow_step(score_model, flow_optimizer, union_obs, union_acts, weights, args, config):
    """
    One optimizer step for the diffusion/score model using per-sample weights.
    union_obs: [horizon, batch, obs_dim]
    union_acts: [horizon, batch, act_dim]
    weights: [horizon, batch]
    """
    score_model.train()
    horizon, batch = union_obs.shape[0], union_obs.shape[1]
    flat_obs = union_obs.reshape(horizon * batch, -1).to(args.device)
    flat_acts = union_acts.reshape(horizon * batch, -1).to(args.device)
    flat_weights = weights.reshape(horizon * batch).to(args.device)

    # set condition (ScoreNet in Script A uses 'condition' attribute)
    score_model.condition = flat_obs

    eps = 1e-3
    random_t = torch.rand(flat_acts.shape[0], device=flat_acts.device, dtype=flat_acts.dtype) * (1.0 - eps) + eps
    z = torch.randn_like(flat_acts)
    alpha_t, std = args.marginal_prob_std_fn(random_t)  # returns [N], [N]
    alpha_t = alpha_t[:, None]
    std = std[:, None]
    perturbed_x = flat_acts * alpha_t + z * std
    score = score_model(perturbed_x, random_t)

    per_sample_losses = torch.sum((score * std + z) ** 2, dim=1)  # [N]
    weighted_loss = torch.mean(flat_weights * per_sample_losses)

    flow_optimizer.zero_grad()
    weighted_loss.backward()
    clip_grad_norm_(score_model.parameters(), config["max_grad_norm"])
    flow_optimizer.step()

    score_model.condition = None
    return weighted_loss.item()


# -------------------------
# Evaluate diffusion policy (reused)
# -------------------------
@torch.no_grad()
def evaluate_flow_policy(eval_env, score_model, device, norm_fn, diffusion_steps=15):
    eval_done = False
    eval_obs, _ = eval_env.reset()
    eval_obs = torch.as_tensor(norm_fn(eval_obs), dtype=torch.float32, device=device).unsqueeze(0)
    eval_reward, eval_cost, eval_len = 0.0, 0.0, 0
    while not eval_done:
        act = score_model.select_actions(eval_obs, diffusion_steps=diffusion_steps)
        if isinstance(act, list):
            act_np = act[0]
        elif isinstance(act, np.ndarray):
            act_np = act
        else:
            act_np = act.squeeze().cpu().numpy()
        next_obs, reward, terminated, truncated, info = eval_env.step(act_np)
        cost = info.get("cost", 0.0)
        next_obs = torch.as_tensor(norm_fn(next_obs), dtype=torch.float32, device=device).unsqueeze(0)
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_len += 1
        eval_done = terminated or truncated
    return eval_reward, eval_cost, eval_len


# -------------------------
# Main training entrypoint
# -------------------------
def main(args):
    # Merge user args with default config
    config = {**default_cfg}
    # allow args override
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v

    # device
    device = torch.device(args.device if isinstance(args.device, str) else f"{args.device}:{getattr(args, 'device_id', 0)}")
    args.device = device

    # Setup marginal_prob_std for diffusion ScoreNet
    marginal_prob_std_fn = functools.partial(marginal_prob_std, schedule=args.schedule, device=device)
    args.marginal_prob_std_fn = marginal_prob_std_fn

    # Setup logging & experiment dirs
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = "ipl_flow_matching_v2"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config({**config, **vars(args)})

    # Build environment for evaluation
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)

    # Load DSRL dataset via DSRLSafetyDataset (Script A)
    dsrl_dataset = DSRLSafetyDataset(env_name=args.task, num_negative=args.num_non_preferred, horizon=100, device=device)
    dataset_splits = dsrl_dataset.get_sets()

    # Normalization (optional)
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        all_obs = torch.cat([
            dataset_splits['negative']['observations'].reshape(-1, dsrl_dataset.obs_dim),
            dataset_splits['union']['observations'].reshape(-1, dsrl_dataset.obs_dim)
        ])
        mu_obs = all_obs.mean(dim=0)
        std_obs = all_obs.std(dim=0)
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    # config gamma may have been set by user
    config["gamma"] = config.get("gamma", 0.99)

    # Create models:
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    print("Initializing models...")
    q_model = QNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["q_hidden"]).to(device)
    q_opt = Adam(q_model.parameters(), lr=config.get("q_lr", 3e-4), weight_decay=config.get("weight_decay", 0.0))

    v_model = VNetwork(obs_dim=obs_dim, hidden_size=config["v_hidden"]).to(device)
    v_opt = Adam(v_model.parameters(), lr=config.get("v_lr", 3e-4), weight_decay=config.get("weight_decay", 0.0))

    # Flow model (ScoreNet) — same interface as Script A
    flow_model = ScoreNet(
        input_dim=obs_dim + act_dim,
        output_dim=act_dim,
        marginal_prob_std=marginal_prob_std_fn,
        args=args
    ).to(device)
    flow_opt = Adam(flow_model.parameters(), lr=config.get("lr", 3e-4), weight_decay=config.get("weight_decay", 0.0))

    # Training variables
    q_pretrain_iters = config["q_pretrain_iterations"]
    flow_train_iters = config["flow_train_iterations"]
    batch_size = config["batch_size"]
    train_horizon = config["train_horizon"]
    v_updates_per_q = config.get("v_updates_per_q_update", 1)

    print("=" * 60)
    print(f"Q pretrain iterations: {q_pretrain_iters}")
    print(f"Flow (Q+V-weighted) train iterations: {flow_train_iters}")
    print(f"Batch size (trajectories): {batch_size}, horizon: {train_horizon}")
    print("=" * 60)

    # ============= PHASE 1: Pretrain Q & V (IPL-style alternating updates) =============
    print("\nPhase 1: Q & V pretraining (IPL-style)")
    pbar = tqdm(range(q_pretrain_iters), desc="Phase1:Q_V", unit="iter", dynamic_ncols=True)
    start_time = time.time()
    for step in pbar:
        # sample a batch from dataset_splits
        neg_obs, neg_acts, union_obs, union_acts, union_rew = sample_trajectory_batch_from_splits(
            dataset_splits, batch_size, train_horizon, device, norm_fn if args.normalize_observation else None
        )

        # Q preference update (union preferred over negative)
        q_stats = q_preference_step(
            q_model=q_model,
            q_optimizer=q_opt,
            neg_obs=neg_obs,
            neg_acts=neg_acts,
            union_obs=union_obs,
            union_acts=union_acts,
            config=config
        )

        # V updates (may run multiple v updates per q update)
        v_stats = {"v_loss": None}
        for _ in range(v_updates_per_q):
            v_stats = v_update_step(
                q_model=q_model,
                v_model=v_model,
                v_optimizer=v_opt,
                union_obs=union_obs,
                union_acts=union_acts,
                neg_obs=neg_obs,
                neg_acts=neg_acts,
                config=config
            )

        # Logging
        if (step + 1) % config["log_freq"] == 0 or (step + 1) == q_pretrain_iters:
            elapsed = time.time() - start_time
            logger.log_tabular("Train/Step", step + 1)
            logger.log_tabular("Q/PrefLoss", q_stats["pref_loss"])
            logger.log_tabular("Q/RegLoss", q_stats["reg_loss"])
            logger.log_tabular("V/Loss", v_stats["v_loss"])
            logger.log_tabular("Flow/Step", 0.0)
            logger.log_tabular("Flow/Loss", 0.0)

            logger.log_tabular("Eval/Reward", 0.0)
            logger.log_tabular("Eval/Cost", 0.0)
            logger.log_tabular("Eval/Length", 0.0)
            logger.log_tabular("Time/ElapsedSec", elapsed)
            logger.dump_tabular()
            print(f"Pretrain step {step+1}/{q_pretrain_iters} Qpref={q_stats['pref_loss']:.4f} Qreg={q_stats['reg_loss']:.6f} Vloss={v_stats['v_loss']:.6f} time={elapsed:.1f}s")

        # Save checkpoints periodically
        if (step + 1) % config["save_freq"] == 0 or (step + 1) == q_pretrain_iters:
            logger.torch_save(itr=step+1, torch_saver_elements=q_model, prefix="q")
            logger.torch_save(itr=step+1, torch_saver_elements=v_model, prefix="v")

    print("Q & V pretraining finished.")

    # Freeze Q and V (use only for weights)
    for p in q_model.parameters():
        p.requires_grad = False
    q_model.eval()
    for p in v_model.parameters():
        p.requires_grad = False
    v_model.eval()

    # ============= PHASE 2: Train flow using Q,V-derived per-timestep weights ============
    print("\nPhase 2: Train flow (weighted by IPL Q-V advantages)")
    best_flow_reward = -float('inf')
    pbar = tqdm(range(flow_train_iters), desc="Phase2:Flow", unit="iter", dynamic_ncols=True)
    start_time = time.time()
    
    # Initialize eval metrics
    eval_reward, eval_cost, eval_len = 0.0, 0.0, 0.0

    for step in pbar:
        # sample a batch (trajectories)
        neg_obs, neg_acts, union_obs, union_acts, union_rew = sample_trajectory_batch_from_splits(
            dataset_splits, batch_size, train_horizon, device, norm_fn if args.normalize_observation else None
        )

        # Compute weights from frozen Q & V
        weights = compute_weights_from_qv(q_model=q_model, v_model=v_model, union_obs=union_obs, union_acts=union_acts, config=config)

        # Flow model update (weighted diffusion flow matching)
        flow_loss_value = train_weighted_flow_step(
            score_model=flow_model,
            flow_optimizer=flow_opt,
            union_obs=union_obs,
            union_acts=union_acts,
            weights=weights,
            args=args,
            config=config
        )

        # Periodic eval of flow policy
        if args.use_eval and ((step + 1) % (args.eval_freq or 1000) == 0):
            eval_reward, eval_cost, eval_len = 0.0, 0.0, 0.0
            for _ in range(config["eval_episode_freq"]):
                r, c, l = evaluate_flow_policy(eval_env, flow_model, device, norm_fn, diffusion_steps=args.diffusion_steps)
                eval_reward += r
                eval_cost += c
                eval_len += l
            eval_reward /= config["eval_episode_freq"]
            eval_cost /= config["eval_episode_freq"]
            eval_len /= config["eval_episode_freq"]
            print(f"\n[Eval Step {step+1}] Reward: {eval_reward:.2f}, Cost: {eval_cost:.2f}, Len: {eval_len:.2f}")
            
            # save best flow by reward
            if eval_reward > best_flow_reward:
                best_flow_reward = eval_reward
                logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow_best")

        # Logging - ALWAYS log all metrics (use last eval values or 0.0)
        if (step + 1) % config["log_freq"] == 0 or (step + 1) == flow_train_iters:
            elapsed = time.time() - start_time
            logger.log_tabular("Train/Step", q_pretrain_iters + step + 1)  # Continue from Phase 1
            logger.log_tabular("Q/PrefLoss", 0.0)  # Phase 1 metrics (not training anymore)
            logger.log_tabular("Q/RegLoss", 0.0)
            logger.log_tabular("V/Loss", 0.0)
            logger.log_tabular("Flow/Step", step + 1)
            logger.log_tabular("Flow/Loss", flow_loss_value)
            logger.log_tabular("Eval/Reward", eval_reward)  # Always log (will be 0.0 until first eval)
            logger.log_tabular("Eval/Cost", eval_cost)
            logger.log_tabular("Eval/Length", eval_len)
            logger.log_tabular("Time/ElapsedSec", elapsed)
            logger.dump_tabular()
            print(f"Flow step {step+1}/{flow_train_iters} flow_loss={flow_loss_value:.6f} eval_r={eval_reward:.2f} eval_c={eval_cost:.2f} time={elapsed:.1f}s")

        # Checkpoint saving
        if (step + 1) % config["save_freq"] == 0 or (step + 1) == flow_train_iters:
            logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow")
            logger.torch_save(itr=step+1, torch_saver_elements=q_model, prefix="q")
            logger.torch_save(itr=step+1, torch_saver_elements=v_model, prefix="v")

    # Final saves
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=flow_model, prefix="flow")
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=q_model, prefix="q")
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=v_model, prefix="v")
    logger.close()
    print("Training complete.")


# -------------------------
# CLI args
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "--env", dest="task", default="OfflinePointGoal1Gymnasium-v0",
                        help="DSRL task name")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--experiment", type=str, default="merged")
    parser.add_argument("--num_non_preferred", type=int, default=50)
    parser.add_argument("--num_union", type=int, default=-1)
    parser.add_argument("--normalize_observation", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--diffusion_steps", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_horizon", type=int, default=5)
    parser.add_argument("--q_pretrain_iterations", type=int, default=20000)
    parser.add_argument("--flow_train_iterations", type=int, default=24000)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--use_eval", action="store_true", default=True)
    parser.add_argument("--eval_freq", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    """
    # attach some fields required by ScoreNet/marginal_prob_std interface
    args.marginal_prob_std_fn = None  # filled in main
    args.device = args.device
    args.schedule = args.schedule
    args.diffusion_steps = args.diffusion_steps
    args.batch_size = args.batch_size
    args.train_horizon = args.train_horizon
    args.num_non_preferred = args.num_non_preferred
    args.num_union = args.num_union
    args.normalize_observation = args.normalize_observation
    args.log_freq = args.log_freq
    args.save_freq = args.save_freq
    args.use_eval = args.use_eval
    args.eval_freq = args.eval_freq
    args.experiment = args.experiment
    """
    main(args)
