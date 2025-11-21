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
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

# Add the offline_mpc directory to Python path
current_file = Path(__file__).resolve()
offline_mpc_dir = current_file.parents[3]  # Go up 3 levels to reach offline_mpc
sys.path.insert(0, str(offline_mpc_dir))

# DSRL imports (assumes these packages/modules are installed/available)
import gymnasium as gym
import dsrl
import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # type: ignore - registers envs

# diffusion/flow imports (from Script A)
from diffusion_SDE.loss import loss_fn as diffusion_loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, update_target  # ScoreNet: flow model

# SafeDICE/Script B imports (models / utils)
# Ensure these are available in python path (from your repo)
from dsrl_model.utils.models import gradient_panelty, minmax_discriminator_loss
from dsrl_model.utils.models import SafeDiceCritic  # critic class used by script B
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.utils import get_params_norm

# Local dataset adapter (from Script A)
from dsrl_adapter import DSRLSafetyDataset

# Energy-based filter network for cost model
from energy_filter_net import EnergyFilterNet, train_energy_filter_step

EP = 1e-6

# -------------------------
# Default config (merge)
# -------------------------
default_cfg = {
    # Logging / checkpoint
    "log_freq": int(2e4),
    "save_freq": int(5e4),
    "eval_episode_freq": 3,
    "hidden_sizes": [256, 256],
    "max_grad_norm": 1.0,
    # Optimization
    "lr": 0.0001,
    "weight_decay": 0.0,
    # Diffusion
    "diffusion_steps": 15,
    "train_horizon": 5,
    # SafeDICE / Safe training (Phase 2 only)
    "grad_reg_coeffs_nu": 1e-6,  # Gradient penalty for critic (nu function)
    "cost_weight_temp": 1.0,     # Temperature for advantage-based weighting
    # Iterations (defaults — override via CLI args)
    "cost_pretrain_iterations": int(5e5),   # phase 1: NU learning for cost model - 500,000
    "flow_train_iterations": int(1e6),      # phase 2: critic + flow training - 1,000,000
    "batch_size": 256,
    "device": "cuda",
    "gamma": 0.99,  # Discount factor for critic
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
# Reuse functions from Script B (find_alpha helper)
# -------------------------
@torch.no_grad()
def find_alpha(cost_model, union_obs, union_act, config):
    """
    Find alpha parameter for SafeDICE reward transformation.
    Uses energy model with separated obs/act interface.
    
    union_obs: [N, obs_dim], union_act: [N, act_dim]
    """
    device = union_obs.device
    data_size = union_obs.shape[0]
    batch_size = config["batch_size"]
    min_alpha = 1.0
    for i in range(max(1, data_size // batch_size)):
        start = i * batch_size
        end = min((i + 1) * batch_size, data_size)
        batch_obs = union_obs[start:end]
        batch_act = union_act[start:end]
        # Energy model uses separated interface
        cost = cost_model(batch_obs, batch_act)
        cost = (1 / torch.sigmoid(cost)) - 1
        alpha = torch.min(cost).item()
        if min_alpha > alpha:
            min_alpha = alpha
    min_alpha -= 1e-5
    return min_alpha


# -------------------------
# sample batch function (adapted from Script A)
# -------------------------
def sample_trajectory_batch_from_splits(dataset_splits, batch_size, train_horizon, device, norm_fn=None):
    """
    Sample trajectory chunks from negative and union sets.
    Supports both full trajectories (lists) and fixed-horizon tensors.
    
    Returns: neg_obs, neg_acts, union_obs, union_acts, union_rewards
    Shapes: [horizon, batch, dim]
    """
    # Check if using full trajectories (list) or fixed tensors
    if isinstance(dataset_splits['negative'], list):
        # Full trajectories mode
        neg_trajs = dataset_splits['negative']
        union_trajs = dataset_splits['union']
        
        neg_obs_batch, neg_act_batch = [], []
        union_obs_batch, union_act_batch, union_rew_batch = [], [], []
        
        for i in range(batch_size):
            # Sample random trajectories
            neg_traj = neg_trajs[np.random.randint(0, len(neg_trajs))]
            union_traj = union_trajs[np.random.randint(0, len(union_trajs))]
            
            # Sample random start points within each trajectory
            neg_len = len(neg_traj['observations'])
            union_len = len(union_traj['observations'])
            
            max_start_neg = max(0, neg_len - train_horizon)
            max_start_union = max(0, union_len - train_horizon)
            
            neg_start = np.random.randint(0, max_start_neg + 1) if max_start_neg > 0 else 0
            union_start = np.random.randint(0, max_start_union + 1) if max_start_union > 0 else 0
            
            # Extract chunks
            neg_obs_chunk = neg_traj['observations'][neg_start:neg_start + train_horizon]
            neg_act_chunk = neg_traj['actions'][neg_start:neg_start + train_horizon]
            union_obs_chunk = union_traj['observations'][union_start:union_start + train_horizon]
            union_act_chunk = union_traj['actions'][union_start:union_start + train_horizon]
            union_rew_chunk = union_traj.get('rewards', np.zeros(len(union_act_chunk)))[union_start:union_start + train_horizon]
            
            # Pad if necessary (for short trajectories)
            if len(neg_obs_chunk) < train_horizon:
                pad_len = train_horizon - len(neg_obs_chunk)
                neg_obs_chunk = np.pad(neg_obs_chunk, ((0, pad_len), (0, 0)), mode='edge')
                neg_act_chunk = np.pad(neg_act_chunk, ((0, pad_len), (0, 0)), mode='edge')
            if len(union_obs_chunk) < train_horizon:
                pad_len = train_horizon - len(union_obs_chunk)
                union_obs_chunk = np.pad(union_obs_chunk, ((0, pad_len), (0, 0)), mode='edge')
                union_act_chunk = np.pad(union_act_chunk, ((0, pad_len), (0, 0)), mode='edge')
                union_rew_chunk = np.pad(union_rew_chunk, (0, pad_len), mode='edge')
            
            neg_obs_batch.append(neg_obs_chunk)
            neg_act_batch.append(neg_act_chunk)
            union_obs_batch.append(union_obs_chunk)
            union_act_batch.append(union_act_chunk)
            union_rew_batch.append(union_rew_chunk)
        
        # Stack and transpose -> [horizon, batch, dim]
        neg_obs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in neg_obs_batch]).transpose(0, 1).to(device)
        neg_acts = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in neg_act_batch]).transpose(0, 1).to(device)
        union_obs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in union_obs_batch]).transpose(0, 1).to(device)
        union_acts = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in union_act_batch]).transpose(0, 1).to(device)
        union_rew = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in union_rew_batch]).transpose(0, 1).to(device)
        
    else:
        # Old fixed-horizon tensor mode
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
# compute SafeDICE-derived weights (use critic advantages)
# -------------------------
@torch.no_grad()
def compute_weights_from_critic(cost_model, critic_model, union_obs, union_acts, union_next_obs, alpha, config):
    """
    Given frozen cost_model and current critic_model, compute weights for union samples.
    Steps (from Script B):
     - sigmoid_cost = sigmoid(cost_model([s,a]))
     - reward_union = log( (1 - (1+alpha)*sigmoid) / ((1-alpha)*(1-sigmoid)) )
     - union_adv_nu = reward_union + gamma * nu(s') - nu(s)
     - weight = exp(union_adv_nu - 1) ** temp ; normalized by mean
    Input shapes:
      union_obs: [horizon, batch, obs_dim]
      union_acts: [horizon, batch, act_dim]
      union_next_obs: [horizon, batch, obs_dim]
    Returns:
      weights: [horizon, batch] (torch tensor, same device)
    """
    device = union_obs.device
    horizon, batch = union_obs.shape[0], union_obs.shape[1]
    flat_obs = union_obs.reshape(-1, union_obs.shape[-1])
    flat_acts = union_acts.reshape(-1, union_acts.shape[-1])
    flat_next_obs = union_next_obs.reshape(-1, union_next_obs.shape[-1])

    # cost model expects [s,a] concatenated for union samples (flat)
    with torch.no_grad():
        cost_inputs = torch.cat([flat_obs, flat_acts], dim=1)
        sigmoid_cost = torch.sigmoid(cost_model(cost_inputs)).squeeze()  # [N]
        # guard: clip sigmoid to avoid nan in log
        sigmoid_cost = torch.clamp(sigmoid_cost, 1e-6, 1.0 - 1e-6)

        # reward transform from Script B
        reward_union = torch.log((1 - (1 + alpha) * sigmoid_cost) / ((1 - alpha) * (1 - sigmoid_cost)))
        # normalize later across horizon*batch, or per-horizon? Use per-sample normalization later.

        # critic values
        flat_obs_val = critic_model(flat_obs)           # nu(s)
        flat_next_obs_val = critic_model(flat_next_obs) # nu(s')

        union_adv = reward_union + config['gamma'] * flat_next_obs_val.squeeze() - flat_obs_val.squeeze()  # [N]

        # reshape to [horizon, batch]
        union_adv = union_adv.reshape(horizon, batch)

        # weights: as in Script B
        temp = config.get("cost_weight_temp", 1.0)
        weight = torch.exp(union_adv - 1.0) ** temp
        # normalize
        weight = weight / (torch.mean(weight) + EP)

        return weight


# -------------------------
# train flow with weights (adapted from Script A)
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
# Critic update (one step) adapted from Script B train_critic_and_actor (nu part only)
# -------------------------
def critic_nu_loss_and_step(
    critic_model,
    critic_optimizer,
    target_init_obs,
    target_union_obs,
    target_union_next_obs,
    reward_union,
    config,
):
    """
    Compute nu loss (critic) with gradient penalty, following SafeDICE-style:
      nu_loss = linear_loss + non_linear_loss + grad_penalty
    Inputs:
        target_init_obs: [B, obs_dim]
        target_union_obs: [B_union, obs_dim]
        target_union_next_obs: [B_union, obs_dim]
        reward_union: [B_union]
        config: dictionary containing gamma, grad_reg_coeffs_nu, max_grad_norm
    Returns:
        nu_loss.item(): scalar loss
        union_adv_nu.detach(): detached advantage for weighting
    """

    device = target_init_obs.device
    B = target_init_obs.shape[0]
    batch_size = target_init_obs.shape[0]


    # -------------------------------
    # Use all union samples (no sub-sampling)
    union_obs_all = target_union_obs          # [horizon*batch, obs_dim]
    union_next_obs_all = target_union_next_obs  # [horizon*batch, obs_dim]
    reward_union_all = reward_union          # [horizon*batch]

    # -------------------------------
    # Compute nu values
    init_nu = critic_model(target_init_obs)               # [batch, 1]
    union_nu = critic_model(union_obs_all)               # [horizon*batch, 1]
    union_next_nu = critic_model(union_next_obs_all)     # [horizon*batch, 1]

    # advantage
    union_adv_nu = reward_union_all + config["gamma"] * union_next_nu.squeeze() - union_nu.squeeze()  # [horizon*batch]

    # loss terms
    non_linear_loss = torch.logsumexp(union_adv_nu, dim=0)
    linear_loss = (1 - config["gamma"]) * torch.mean(init_nu)
    nu_loss = linear_loss + non_linear_loss

    # -------------------------------
    # Gradient penalty
    # create interpolated inputs
    unif_rand = torch.rand(size=(union_obs_all.shape[0], 1), device=device)
    nu_inter = unif_rand * target_init_obs.repeat(union_obs_all.shape[0] // batch_size, 1) + (1 - unif_rand) * union_obs_all
    nu_next_inter = unif_rand * target_init_obs.repeat(union_next_obs_all.shape[0] // batch_size, 1) + (1 - unif_rand) * union_next_obs_all
    nu_inter_combined = torch.cat([union_obs_all, nu_inter, nu_next_inter], dim=0)
    nu_inter_combined = Variable(nu_inter_combined, requires_grad=True).to(device)
    nu_output = critic_model(nu_inter_combined)
    nu_loss += config["grad_reg_coeffs_nu"] * gradient_panelty(nu_inter_combined, nu_output)

    # -------------------------------
    # Step optimizer
    critic_optimizer.zero_grad()
    nu_loss.backward()
    clip_grad_norm_(critic_model.parameters(), config["max_grad_norm"])
    critic_optimizer.step()

    return nu_loss.item(), union_adv_nu.detach()

# -------------------------
# Evaluate diffusion policy (reused from Script A)
# Modified to generate horizon-length sequences and use only first action
# -------------------------
@torch.no_grad()
def evaluate_flow_policy(eval_env, score_model, device, norm_fn, diffusion_steps=15, eval_horizon=5):
    """
    Evaluate flow policy by generating horizon-length action sequences.
    Only the first action from each sequence is executed in the environment.
    
    Args:
        eval_horizon: Length of action sequence to generate (default: 5)
    """
    eval_done = False
    eval_obs, _ = eval_env.reset()
    eval_obs = torch.as_tensor(norm_fn(eval_obs), dtype=torch.float32, device=device).unsqueeze(0)
    eval_reward, eval_cost, eval_len = 0.0, 0.0, 0
    
    while not eval_done:
        # Generate horizon-length action sequence conditioned on current observation
        # Repeat observation for horizon length: [horizon, obs_dim] (without batch dimension)
        # ScoreNet.select_actions expects [batch, obs_dim] where batch = horizon in this case
        obs_horizon = eval_obs.squeeze(0).repeat(eval_horizon, 1)  # [horizon, obs_dim]
        
        # Generate action sequence
        # select_actions expects: states [batch, obs_dim], returns [batch, act_dim]
        act_sequence = score_model.select_actions(obs_horizon, diffusion_steps=diffusion_steps)
        
        # Extract first action only
        if isinstance(act_sequence, list):
            act_np = act_sequence[0]  # First action from the sequence
        elif isinstance(act_sequence, np.ndarray):
            act_np = act_sequence[0] if act_sequence.ndim > 1 else act_sequence
        else:
            # Tensor: [horizon, act_dim] -> take first timestep [act_dim]
            act_np = act_sequence[0].cpu().numpy()
        
        # Execute only the first action
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
# Main (integrated training)
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
    algo = "pubc_flow_matching"
    # Use offline_mpc/logs as base directory
    base_log_dir = os.path.join(str(offline_mpc_dir), "logs")
    args.log_dir = os.path.join(base_log_dir, args.experiment, args.task, algo, relpath)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    # Create separate loggers for Phase 1 (cost) and Phase 2 (flow)
    logger_phase1 = EpochLogger(log_dir=os.path.join(args.log_dir, "phase1_cost"), seed=str(args.seed))
    logger_phase1.save_config({**config, **vars(args)})
    
        # Build environment for evaluation
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)

    # Load DSRL dataset via DSRLSafetyDataset (Script A)
    # Filter trajectories: only keep those with reward > high_reward_ratio * max_reward
    # Then split into negative (high-cost) and union (remaining high-reward) sets
    # Store full trajectories (not truncated) - will sample chunks during training
    dsrl_dataset = DSRLSafetyDataset(
        env_name=args.task, 
        num_negative=args.num_non_preferred, 
        horizon=config["train_horizon"],  # Used for sampling chunks, not truncating
        device=device,
        high_reward_ratio=args.high_reward_ratio,
        store_full_trajectories=True  # Keep full trajectories
    )
    dataset_splits = dsrl_dataset.get_sets()

    # Normalization (optional)
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        # Handle full trajectories (list) vs fixed tensors
        if isinstance(dataset_splits['negative'], list):
            # Full trajectories: concatenate all observations
            all_obs_list = []
            for traj in dataset_splits['negative']:
                all_obs_list.append(torch.as_tensor(traj['observations'], dtype=torch.float32))
            for traj in dataset_splits['union']:
                all_obs_list.append(torch.as_tensor(traj['observations'], dtype=torch.float32))
            all_obs = torch.cat([obs.reshape(-1, dsrl_dataset.obs_dim) for obs in all_obs_list])
        else:
            # Fixed tensors
            all_obs = torch.cat([
                dataset_splits['negative']['observations'].reshape(-1, dsrl_dataset.obs_dim),
                dataset_splits['union']['observations'].reshape(-1, dsrl_dataset.obs_dim)
            ])
        mu_obs = all_obs.mean(dim=0)
        std_obs = all_obs.std(dim=0)
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)

    # Create models:
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    print("Initializing models...")
    # Energy-based cost model using NU learning
    cost_model = EnergyFilterNet(obs_dim=obs_dim, action_dim=act_dim, max_action=max_action).to(device)
    cost_opt = Adam(cost_model.parameters(), lr=config.get("lr", 3e-4), weight_decay=config.get("weight_decay", 0.0))

    critic_model = SafeDiceCritic(obs_dim=obs_dim, act_dim=0, hidden_size=config["hidden_sizes"][0]).to(device)  # nu(s) critic
    critic_opt = Adam(critic_model.parameters(), lr=config.get("lr", 3e-4), weight_decay=config.get("weight_decay", 0.0))

    # Flow model (ScoreNet) — same interface as Script A
    flow_model = ScoreNet(
        input_dim=obs_dim + act_dim,
        output_dim=act_dim,
        marginal_prob_std=marginal_prob_std_fn,
        args=args
    ).to(device)
    flow_opt = Adam(flow_model.parameters(), lr=config.get("lr", 3e-4), weight_decay=config.get("weight_decay", 0.0))

    # Optional: actor (behavior cloning) — omitted by default, but you can instantiate if you want
    # from dsrl_model.utils.models import SafeDiceTanhMixtureActor
    # actor = SafeDiceTanhMixtureActor(obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_sizes"][0]).to(device)
    # actor_opt = Adam(actor.parameters(), lr=config.get("lr", 3e-4))

    # Training variables
    cost_pretrain_iters = config["cost_pretrain_iterations"]
    flow_train_iters = config["flow_train_iterations"]
    batch_size = config["batch_size"]
    train_horizon = config["train_horizon"]

    print("=" * 60)
    print(f"Cost pretrain iterations: {cost_pretrain_iters}")
    print(f"Flow (critic+flow) train iterations: {flow_train_iters}")
    print(f"Batch size (trajectories): {batch_size}, horizon: {train_horizon}")
    print("=" * 60)

    # ============= PHASE 1: Pretrain cost/discriminator (SafeDICE-style) =============
    print("\nPhase 1: Cost (discriminator) pretraining")
    pbar = tqdm(range(cost_pretrain_iters), desc="Phase1:Cost", unit="iter", dynamic_ncols=True)
    start_time = time.time()
    for step in pbar:
        # sample a batch from dataset_splits
        neg_obs, neg_acts, union_obs, union_acts, union_rew = sample_trajectory_batch_from_splits(
            dataset_splits, batch_size, train_horizon, device, norm_fn if args.normalize_observation else None
        )
        # For discriminator pretrain, use flattened per-time-step samples
        # flatten across horizon x batch -> [N, dim]
        neg_obs_flat = neg_obs.reshape(-1, neg_obs.shape[-1])
        neg_acts_flat = neg_acts.reshape(-1, neg_acts.shape[-1])
        union_obs_flat = union_obs.reshape(-1, union_obs.shape[-1])
        union_acts_flat = union_acts.reshape(-1, union_acts.shape[-1])

        # NU learning with energy-based model
        loss_cost = train_energy_filter_step(
            model=cost_model,
            optimizer=cost_opt,
            neg_obs=neg_obs_flat,
            neg_acts=neg_acts_flat,
            union_obs=union_obs_flat,
            union_acts=union_acts_flat,
            config=config,
        )

        if (step + 1) % config["log_freq"] == 0 or (step + 1) == cost_pretrain_iters:
            elapsed = time.time() - start_time
            logger_phase1.log_tabular("Train/Step", step + 1)
            print(f"Cost pretrain step {step+1}/{cost_pretrain_iters} loss: {loss_cost.item():.4f} | time(s): {elapsed:.1f}")
            logger_phase1.log_tabular("Train/CostLoss", loss_cost.item())
            logger_phase1.log_tabular("Time/ElapsedSec", elapsed)
            logger_phase1.dump_tabular()

        if (step + 1) % config["save_freq"] == 0 or (step + 1) == cost_pretrain_iters:
            logger_phase1.torch_save(itr=step+1, torch_saver_elements=cost_model, prefix="cost")

    print("Cost pretraining finished. Computing alpha...")
    # compute alpha (on union flatten)
    if isinstance(dataset_splits['union'], list):
        # Full trajectories: concatenate all observations and actions
        union_obs_list = []
        union_acts_list = []
        for traj in dataset_splits['union']:
            union_obs_list.append(torch.as_tensor(traj['observations'], dtype=torch.float32, device=device))
            union_acts_list.append(torch.as_tensor(traj['actions'], dtype=torch.float32, device=device))
        union_obs_flat_all = torch.cat([obs.reshape(-1, dsrl_dataset.obs_dim) for obs in union_obs_list])
        union_acts_flat_all = torch.cat([acts.reshape(-1, act_dim) for acts in union_acts_list])
    else:
        # Fixed tensors
        union_obs_flat_all = dataset_splits['union']['observations'].reshape(-1, dsrl_dataset.obs_dim).to(device)
        union_acts_flat_all = dataset_splits['union']['actions'].reshape(-1, eval_env.action_space.shape[0]).to(device)
    
    alpha = find_alpha(cost_model=cost_model, union_obs=union_obs_flat_all, union_act=union_acts_flat_all, config=config)
    logger_phase1.log(f"Found alpha: {alpha:.6f}")
    print(f"Alpha = {alpha:.6f}")
    
    # Close Phase 1 logger
    logger_phase1.close()

    # Freeze cost model (use only for computing rewards)
    for p in cost_model.parameters():
        p.requires_grad = False
    cost_model.eval()

    # ============= PHASE 2: Train critic + flow with SafeDICE-derived weights ============
    print("\nPhase 2: Train critic + flow (weighted by critic advantage)")
    
    # Create Phase 2 logger
    logger_phase2 = EpochLogger(log_dir=os.path.join(args.log_dir, "phase2_flow"), seed=str(args.seed))
    logger_phase2.save_config({**config, **vars(args)})
    
    best_flow_reward = -float('inf')
    eval_deque = deque(maxlen=config["eval_episode_freq"])
    pbar = tqdm(range(flow_train_iters), desc="Phase2:Flow", unit="iter", dynamic_ncols=True)
    start_time = time.time()
    
    # Update critic less frequently to speed up training
    critic_update_freq = config.get("critic_update_freq", 10)  # Update critic every N steps
    # Option to disable weighting for maximum speed (use uniform weights)
    use_energy_weighting = config.get("use_energy_weighting", True)
    
    # Initialize all logging keys with NaN values to avoid assertion errors
    logger_phase2.log_tabular("Train/Step", 0)
    logger_phase2.log_tabular("Train/Loss/Nu", float('nan'))
    logger_phase2.log_tabular("Train/Loss/Flow", float('nan'))
    logger_phase2.log_tabular("Train/Norm/Critic", float('nan'))
    logger_phase2.log_tabular("Train/Norm/Flow", float('nan'))
    logger_phase2.log_tabular("Time/ElapsedSec", 0.0)
    logger_phase2.log_tabular("Eval/Reward", float('nan'))
    logger_phase2.log_tabular("Eval/Cost", float('nan'))
    logger_phase2.log_tabular("Eval/Length", float('nan'))
    logger_phase2.dump_tabular()

    for step in pbar:
        # sample a batch (trajectories)
        neg_obs, neg_acts, union_obs, union_acts, union_rew = sample_trajectory_batch_from_splits(
            dataset_splits, batch_size, train_horizon, device, norm_fn if args.normalize_observation else None
        )

        nu_loss_value = 0.0  # default if not updating critic
        
        if use_energy_weighting:
            # SLOW PATH: Compute energy-based weights (SafeDICE style)
            # We need next observations for critic (shifted by 1 along horizon). Build union_next_obs.
            # For simplicity, use next obs of same chunk (shift by 1, and last next -> copy last)
            union_next_obs = torch.roll(union_obs, shifts=-1, dims=0)
            # The rolled last step should be kept equal to last (no future), so copy last to last position
            union_next_obs[-1] = union_obs[-1].clone()

            # flatten union sequences for critic inputs (per time-step)
            flat_union_obs = union_obs.reshape(-1, union_obs.shape[-1])
            flat_union_acts = union_acts.reshape(-1, union_acts.shape[-1])
            flat_union_next_obs = union_next_obs.reshape(-1, union_next_obs.shape[-1])

            # Compute reward_union per flat sample using cost_model and alpha (as in Script B)
            with torch.no_grad():
                # Energy model uses separated obs/act interface
                sigmoid_cost = torch.sigmoid(cost_model(flat_union_obs, flat_union_acts)).squeeze()
                sigmoid_cost = torch.clamp(sigmoid_cost, 1e-6, 1.0 - 1e-6)
                reward_union_flat = torch.log((1 - (1 + alpha) * sigmoid_cost) / ((1 - alpha) * (1 - sigmoid_cost)))

            # Critic update: compute nu loss using flattened union samples grouped back into batches
            # For nu loss we expect batch inputs (we'll use the flat samples as the 'union' set)
            # For init states we sample initial obs from union sequences: take the first time-step observations of each trajectory
            target_init_obs = union_obs[0]  # shape [batch, obs_dim]
            target_union_obs = flat_union_obs  # shape [horizon*batch, obs_dim]
            target_union_next_obs = flat_union_next_obs

            # Update critic only periodically to speed up training
            if step % critic_update_freq == 0:
                # compute critic loss & step
                nu_loss_value, union_adv_flat = critic_nu_loss_and_step(
                    critic_model=critic_model,
                    critic_optimizer=critic_opt,
                    target_init_obs=target_init_obs,
                    target_union_obs=target_union_obs,
                    target_union_next_obs=target_union_next_obs,
                    reward_union=reward_union_flat,
                    config=config,
                )
            else:
                # Just compute advantages without updating (for weight computation)
                with torch.no_grad():
                    union_nu = critic_model(target_union_obs)
                    union_next_nu = critic_model(target_union_next_obs)
                    union_adv_flat = reward_union_flat + config["gamma"] * union_next_nu.squeeze() - union_nu.squeeze()

            # Convert union_adv_flat [N] back to [horizon, batch]
            horizon_len = union_obs.shape[0]
            union_adv = union_adv_flat.reshape(horizon_len, batch_size)  # still detached

            # Compute weights for flow training from union_adv (Script B style)
            temp = config.get("cost_weight_temp", 1.0)
            weights = torch.exp(union_adv - 1.0) ** temp
            weights = weights / (torch.mean(weights) + EP)
        else:
            # FAST PATH: Use uniform weights (standard behavior cloning / flow matching)
            weights = torch.ones(train_horizon, batch_size, device=device)

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

        # Optional: update actor (BC) with same weights (not implemented by default)
        # If you want actor updates, compute weighted BC loss using union_obs/union_acts and weights and step actor optimizer.

        # Periodic eval of flow policy
        eval_reward, eval_cost, eval_len = None, None, None
        if args.use_eval and ((step + 1) % (args.eval_freq or 1000) == 0):
            eval_reward, eval_cost, eval_len = 0.0, 0.0, 0
            # run a few episodes
            for _ in range(config["eval_episode_freq"]):
                r, c, l = evaluate_flow_policy(
                    eval_env, flow_model, device, norm_fn, 
                    diffusion_steps=args.diffusion_steps,
                    eval_horizon=config["train_horizon"]
                )
                eval_reward += r; eval_cost += c; eval_len += l
            eval_reward /= config["eval_episode_freq"]
            eval_cost /= config["eval_episode_freq"]
            eval_len /= config["eval_episode_freq"]
            
            # Print to console
            logger_phase2.log(f"[Eval] Step {step+1}: reward={eval_reward:.2f} cost={eval_cost:.2f} len={eval_len:.2f}")
            
            # save best flow by reward
            if eval_reward > best_flow_reward:
                best_flow_reward = eval_reward
                logger_phase2.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow_best")

        # Logging (always log all metrics, use current values or empty string for missing ones)
        if (step + 1) % config["log_freq"] == 0 or (step + 1) == flow_train_iters:
            elapsed = time.time() - start_time
            # Training metrics (always available)
            logger_phase2.log_tabular("Train/Step", step+1)
            logger_phase2.log_tabular("Train/Loss/Nu", nu_loss_value)
            logger_phase2.log_tabular("Train/Loss/Flow", flow_loss_value)
            logger_phase2.log_tabular("Train/Norm/Critic", get_params_norm(critic_model.parameters(), grads=False))
            logger_phase2.log_tabular("Train/Norm/Flow", get_params_norm(flow_model.parameters(), grads=False))
            logger_phase2.log_tabular("Time/ElapsedSec", elapsed)
            # Evaluation metrics (use NaN if eval hasn't run yet - works with CSV and TensorBoard)
            logger_phase2.log_tabular("Eval/Reward", eval_reward if eval_reward is not None else float('nan'))
            logger_phase2.log_tabular("Eval/Cost", eval_cost if eval_cost is not None else float('nan'))
            logger_phase2.log_tabular("Eval/Length", eval_len if eval_len is not None else float('nan'))
            logger_phase2.dump_tabular()

        # Checkpoint saving
        if (step + 1) % config["save_freq"] == 0 or (step + 1) == flow_train_iters:
            logger_phase2.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow")
            logger_phase2.torch_save(itr=step+1, torch_saver_elements=cost_model, prefix="cost")
            logger_phase2.torch_save(itr=step+1, torch_saver_elements=critic_model, prefix="critic")

    # Final saves
    logger_phase2.torch_save(itr=flow_train_iters, torch_saver_elements=flow_model, prefix="flow")
    logger_phase2.torch_save(itr=flow_train_iters, torch_saver_elements=cost_model, prefix="cost")
    logger_phase2.torch_save(itr=flow_train_iters, torch_saver_elements=critic_model, prefix="critic")
    logger_phase2.close()
    print("Training complete.")


# -------------------------
# CLI args: reuse get_args from your original script if available
# For now provide a minimal parser fallback
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SafeDICE Flow Matching with Energy-Based Cost Model")
    
    # Environment
    parser.add_argument("--task", "--env", dest="task", default="OfflinePointGoal1Gymnasium-v0", 
                        help="DSRL task name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Logging and experiment
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--experiment", type=str, default="safedice_energy", help="Experiment name")
    
    # Dataset configuration
    parser.add_argument("--num_non_preferred", type=int, default=50, 
                        help="Number of negative (unsafe) trajectories for NU learning")
    parser.add_argument("--num_union", type=int, default=-1, 
                        help="Number of union trajectories (-1 = all remaining)")
    parser.add_argument("--high_reward_ratio", type=float, default=0.5,
                        help="Filter trajectories: only keep those with reward >= high_reward_ratio * max_reward (default: 0.5 = 50%%)")
    parser.add_argument("--normalize_observation", action="store_true", 
                        help="Normalize observations")
    
    # Training configuration
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size (number of trajectories)")
    parser.add_argument("--train_horizon", type=int, default=5, 
                        help="Training horizon (timesteps per trajectory)")
    
    # Phase 1: Cost model training (NU learning)
    parser.add_argument("--cost_pretrain_iterations", type=int, default=500000, 
                        help="Phase 1: NU learning iterations for cost model")
    
    # Phase 2: Flow + critic training
    parser.add_argument("--flow_train_iterations", type=int, default=1000000, 
                        help="Phase 2: Flow and critic training iterations")
    
    # Optimization
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                        help="Max gradient norm for clipping")
    
    # Diffusion configuration
    parser.add_argument("--schedule", type=str, default="linear", 
                        choices=["linear", "cosine", "OT"], help="Diffusion noise schedule")
    parser.add_argument("--diffusion_steps", type=int, default=15, 
                        help="Number of diffusion steps for sampling")
    
    # SafeDICE parameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--grad_reg_coeffs_nu", type=float, default=1e-6, 
                        help="Gradient penalty coefficient for critic")
    parser.add_argument("--cost_weight_temp", type=float, default=1.0, 
                        help="Temperature for advantage-based weighting")
    parser.add_argument("--critic_update_freq", type=int, default=10,
                        help="Update critic every N steps in Phase 2 (higher=faster training, default=10)")
    parser.add_argument("--use_energy_weighting", action="store_true", default=False,
                        help="Use energy-based advantage weighting (slower but more accurate). If False, uses uniform weights (faster).")
    
    # Evaluation
    parser.add_argument("--use_eval", action="store_true", default=False, 
                        help="Enable periodic evaluation")
    parser.add_argument("--eval_freq", type=int, default=20000, 
                        help="Evaluation frequency (steps)")
    parser.add_argument("--eval_episode_freq", type=int, default=3, 
                        help="Number of episodes per evaluation")
    
    # Logging frequency
    parser.add_argument("--log_freq", type=int, default=20000, 
                        help="Logging frequency (steps)")
    parser.add_argument("--save_freq", type=int, default=20000, 
                        help="Model checkpoint frequency (steps)")
    
    args = parser.parse_args()

    # Attach fields required by ScoreNet/marginal_prob_std interface
    args.marginal_prob_std_fn = None  # filled in main()

    print("=" * 60)
    print("SafeDICE Flow Matching with Energy-Based Cost Model")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Dataset filter: Only trajectories with reward >= {args.high_reward_ratio*100:.0f}% of max reward")
    print(f"Dataset: {args.num_non_preferred} negative (high-cost) + union (high-reward) trajectories")
    print(f"Phase 1 (Cost): {args.cost_pretrain_iterations} iterations")
    print(f"Phase 2 (Flow): {args.flow_train_iterations} iterations")
    print(f"Batch size: {args.batch_size}, Horizon: {args.train_horizon}")
    print("=" * 60)

    main(args)
