#!/usr/bin/env python3
"""
Unified SafeDICE-weighted Flow Matching training script
Merges:
 - Script A: Diffusion / ScoreNet flow matching (energy-weighted flow)
 - Script B: SafeDICE-style cost/discriminator and critic (weights from critic advantages)

Two-phase training:
 Phase 1: Pretrain cost/discriminator (minmax + grad penalty)
 Phase 2: Freeze cost model; train Critic (nu) + Flow (ScoreNet) simultaneously,
          using critic-derived weights to bias the flow loss to safer samples.

Requirements: diffusion_SDE package (ScoreNet, marginal_prob_std, diffusion_loss_fn),
              DSRL dataset adapter (DSRLSafetyDataset) and SafeDiceCritic model.
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
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

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

EP = 1e-6

# -------------------------
# Default config (merge)
# -------------------------
default_cfg = {
    # Logging / checkpoint
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 5,
    "hidden_sizes": [256, 256],
    "max_grad_norm": 1.0,
    # Optimization
    "lr": 3e-4,
    "weight_decay": 0.0,
    # Diffusion
    "diffusion_steps": 15,
    "train_horizon": 5,
    # SafeDICE / Safe training
    "grad_reg_coeffs": 10.0,
    "grad_reg_coeffs_nu": 1e-6,
    "cost_weight_temp": 1.0,
    "act_train_use_logprob": True,
    # Iterations (defaults — override via CLI args)
    "cost_pretrain_iterations": int(5e4),   # phase 1
    "flow_train_iterations": int(1e4),      # phase 2 total iterations
    "batch_size": 64,
    "device": "cuda",
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
# Reuse functions from Script B (pretrain discriminator & helpers)
# -------------------------
def pretrain_discriminator(
    cost_model,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    config,
):
    """
    Re-uses Script B pretrain_discriminator:
    minmax_discriminator_loss + gradient penalty on interpolations.
    All inputs are [batch, dim]
    """
    batch_size = target_neg_obs.shape[0]
    device = target_neg_obs.device

    target_neg = torch.concat([target_neg_obs, target_neg_act], dim=1)
    target_union = torch.concat([target_union_obs, target_union_act], dim=1)

    # Create mixed samples for grad penalty
    unif_rand = torch.rand(size=(batch_size, 1)).to(device)
    target_mixed1 = unif_rand * target_neg + (1 - unif_rand) * target_union
    shuffle_idx = torch.randperm(batch_size).to(device)
    target_mixed2 = unif_rand * target_union[shuffle_idx] + (1 - unif_rand) * target_union
    target_mixed = torch.concat([target_mixed1, target_mixed2], dim=0)

    cost_neg = cost_model(target_neg)
    cost_union = cost_model(target_union)
    loss = minmax_discriminator_loss(cost_neg, cost_union)

    target_mixed = Variable(target_mixed, requires_grad=True).to(device=device)
    cost_mixed = cost_model(target_mixed)
    loss += config["grad_reg_coeffs"] * gradient_panelty(target_mixed, cost_mixed)
    return loss


@torch.no_grad()
def find_alpha(cost_model, union_obs, union_act, config):
    """
    Re-uses find_alpha from Script B. Returns a small lower bound alpha for transformation.
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
        cost = cost_model(torch.concat([batch_obs, batch_act], dim=1))
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
    algo = "safedice_flow_matching"
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
    config["gamma"] = 0.99

    # Create models:
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    print("Initializing models...")
    cost_model = SafeDiceCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_size=config["hidden_sizes"][0]).to(device)  # reuse SafeDiceCritic as cost network (it outputs scalar)
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

        loss_cost = pretrain_discriminator(
            cost_model=cost_model,
            target_neg_obs=neg_obs_flat,
            target_neg_act=neg_acts_flat,
            target_union_obs=union_obs_flat,
            target_union_act=union_acts_flat,
            config=config,
        )
        cost_opt.zero_grad()
        loss_cost.backward()
        cost_opt.step()

        if (step + 1) % config["log_freq"] == 0 or (step + 1) == cost_pretrain_iters:
            elapsed = time.time() - start_time
            logger.log_tabular("Train/Step", step + 1)
            print(f"Cost pretrain step {step+1}/{cost_pretrain_iters} loss: {loss_cost.item():.4f} | time(s): {elapsed:.1f}")
            logger.log_tabular("Train/CostLoss", loss_cost.item())
            logger.log_tabular("Time/ElapsedSec", elapsed)
            logger.dump_tabular()

        if (step + 1) % config["save_freq"] == 0 or (step + 1) == cost_pretrain_iters:
            logger.torch_save(itr=step+1, torch_saver_elements=cost_model, prefix="cost")

    print("Cost pretraining finished. Computing alpha...")
    # compute alpha (on union flatten)
    union_obs_flat_all = dataset_splits['union']['observations'].reshape(-1, dsrl_dataset.obs_dim).to(device)
    union_acts_flat_all = dataset_splits['union']['actions'].reshape(-1, eval_env.action_space.shape[0]).to(device)
    alpha = find_alpha(cost_model=cost_model, union_obs=union_obs_flat_all, union_act=union_acts_flat_all, config=config)
    logger.log(f"Found alpha: {alpha:.6f}")
    print(f"Alpha = {alpha:.6f}")

    # Freeze cost model (use only for computing rewards)
    for p in cost_model.parameters():
        p.requires_grad = False
    cost_model.eval()

    # ============= PHASE 2: Train critic + flow with SafeDICE-derived weights ============
    print("\nPhase 2: Train critic + flow (weighted by critic advantage)")
    best_flow_reward = -float('inf')
    eval_deque = deque(maxlen=config["eval_episode_freq"])
    pbar = tqdm(range(flow_train_iters), desc="Phase2:Flow", unit="iter", dynamic_ncols=True)
    start_time = time.time()

    for step in pbar:
        # sample a batch (trajectories)
        neg_obs, neg_acts, union_obs, union_acts, union_rew = sample_trajectory_batch_from_splits(
            dataset_splits, batch_size, train_horizon, device, norm_fn if args.normalize_observation else None
        )

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
            cost_inputs = torch.cat([flat_union_obs, flat_union_acts], dim=1)
            sigmoid_cost = torch.sigmoid(cost_model(cost_inputs)).squeeze()
            sigmoid_cost = torch.clamp(sigmoid_cost, 1e-6, 1.0 - 1e-6)
            reward_union_flat = torch.log((1 - (1 + alpha) * sigmoid_cost) / ((1 - alpha) * (1 - sigmoid_cost)))
            # reward normalization (optional): we keep as-is for now.

        # Critic update: compute nu loss using flattened union samples grouped back into batches
        # For nu loss we expect batch inputs (we'll use the flat samples as the 'union' set)
        # For init states we sample initial obs from union sequences: take the first time-step observations of each trajectory
        target_init_obs = union_obs[0]  # shape [batch, obs_dim]
        target_union_obs = flat_union_obs  # shape [horizon*batch, obs_dim]
        target_union_next_obs = flat_union_next_obs

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

        # Convert union_adv_flat [N] back to [horizon, batch]
        horizon_len = union_obs.shape[0]
        union_adv = union_adv_flat.reshape(horizon_len, batch_size)  # still detached

        # Compute weights for flow training from union_adv (Script B style)
        temp = config.get("cost_weight_temp", 1.0)
        weights = torch.exp(union_adv - 1.0) ** temp
        weights = weights / (torch.mean(weights) + EP)

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

        # Logging and evaluation
        if (step + 1) % config["log_freq"] == 0 or (step + 1) == flow_train_iters:
            elapsed = time.time() - start_time
            #logger.log_tabular("TrainStep", step+1)
            #logger.log_tabular("Loss/Nu", nu_loss_value)
            #logger.log_tabular("Loss/Flow", flow_loss_value)
            #logger.log_tabular("Norm/Params/critic", get_params_norm(critic_model.parameters(), grads=False))
            #logger.log_tabular("Norm/Params/flow", get_params_norm(flow_model.parameters(), grads=False))
            #logger.log_tabular("TimeElapsedSec", elapsed)
            #logger.dump_tabular()

        # Periodic eval of flow policy
        if args.use_eval and ((step + 1) % (args.eval_freq or 1000) == 0):
            eval_reward, eval_cost, eval_len = 0.0, 0.0, 0
            # run a few episodes
            for _ in range(config["eval_episode_freq"]):
                r, c, l = evaluate_flow_policy(eval_env, flow_model, device, norm_fn, diffusion_steps=args.diffusion_steps)
                eval_reward += r; eval_cost += c; eval_len += l
            eval_reward /= config["eval_episode_freq"]
            eval_cost /= config["eval_episode_freq"]
            eval_len /= config["eval_episode_freq"]
            logger.log(f"[Eval] Step {step+1}: reward={eval_reward:.2f} cost={eval_cost:.2f} len={eval_len:.2f}")
            # save best flow by reward
            if eval_reward > best_flow_reward:
                best_flow_reward = eval_reward
                logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow_best")

        # Checkpoint saving
        if (step + 1) % config["save_freq"] == 0 or (step + 1) == flow_train_iters:
            logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow")
            logger.torch_save(itr=step+1, torch_saver_elements=cost_model, prefix="cost")
            logger.torch_save(itr=step+1, torch_saver_elements=critic_model, prefix="critic")

    # Final saves
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=flow_model, prefix="flow")
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=cost_model, prefix="cost")
    logger.torch_save(itr=flow_train_iters, torch_saver_elements=critic_model, prefix="critic")
    logger.close()
    print("Training complete.")


# -------------------------
# CLI args: reuse get_args from your original script if available
# For now provide a minimal parser fallback
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
    parser.add_argument("--cost_pretrain_iterations", type=int, default=50000)
    parser.add_argument("--flow_train_iterations", type=int, default=10000)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--use_eval", action="store_true",default=True)
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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

    main(args)
