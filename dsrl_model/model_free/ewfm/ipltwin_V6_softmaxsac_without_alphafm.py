#!/usr/bin/env python3
"""
IPL-Style Flow Matching: Single-Step Policy with Inverse Bellman Operator
Fixed segment-based preference loss with proper tensor shapes
+ Added Q/V/Advantage visualization after Phase 1
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
from torch.utils.tensorboard import SummaryWriter

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
    "log_freq": 1000,
    "save_freq": 20000,
    "eval_episode_freq": 10,
    "max_grad_norm": 1.0,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "preference_iterations": 150000,
    "q_lr": 3e-4,
    "v_lr": 3e-4,
    "q_hidden": 256,
    "v_hidden": 256,
    "expectile_tau": 0.7,
    "chi2_coeff": 0.5,
    "target_clipping": True,
    "flow_train_iterations": 500000,
    "batch_size": 256,
    "device": "cuda",
    "gamma": 0.99,
    "target_update_freq": 2,
    "target_tau": 0.005,
    "lambda_pref": 1.0,
    "lambda_reg": 0.1,
    "energy_alpha": 3.0,
    "use_guidance": True,
    "warmup_iterations": 10000,
    "horizon": 25,
    "weight_from_q": False,
    "density": 1.0,
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
    "segment_length": 25,
    "sigma_min": 0.01,
    # SAC entropy regularization
    "alpha": 0.1,
    "target_entropy": None,  # Auto-set to -act_dim if None
    "auto_tune_alpha": True,
    "alpha_lr": 3e-4,
    "diffusion_steps": 10,  # ODE steps for sampling
    # Previous action conditioning
    "use_prev_action": False,  # If True, condition policy on [obs, prev_action]
}


# ============== JIT-compiled OT functions ==============
@torch.jit.script
def psi_t_ot_jit(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma_min: float) -> torch.Tensor:
    """OT linear interpolation - JIT compiled"""
    one_minus_sigma_min = 1.0 - sigma_min
    t_view = t.view(-1, 1)
    sigma_t = 1.0 - one_minus_sigma_min * t_view
    return sigma_t * x0 + t_view * x1


@torch.jit.script
def u_t_ot_jit(x_t: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma_min: float) -> torch.Tensor:
    """OT vector field - JIT compiled"""
    one_minus_sigma_min = 1.0 - sigma_min
    denom = (1.0 - one_minus_sigma_min * t).view(-1, 1).clamp(min=1e-6)
    return (x1 - one_minus_sigma_min * x_t) / denom


@torch.jit.script
def compute_segment_weights_jit(energy_flat: torch.Tensor, gamma_powers: torch.Tensor, 
                                B: int, H: int, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute segment-level softmax weights - JIT compiled"""
    energy_seg = energy_flat.view(B, H)
    A_seg = (energy_seg * gamma_powers).sum(dim=1)
    w_seg = F.softmax((alpha * A_seg).clamp(-20.0, 20.0), dim=0)
    return w_seg, A_seg


@torch.jit.script
def compute_weighted_loss_jit(v_theta: torch.Tensor, u_t: torch.Tensor, w_seg: torch.Tensor, B: int, H: int) -> torch.Tensor:
    """Compute weighted segment loss - JIT compiled"""
    err = ((v_theta - u_t) ** 2).sum(dim=1)  # [B*H]
    err_seg = err.view(B, H)
    return (w_seg.unsqueeze(1) * err_seg).sum() / H


class TrainingBuffers:
    """Pre-allocated buffers to avoid memory allocation in training loop"""
    __slots__ = ['noise', 't', 'offsets_flow', 'offsets_seg', 'device', 'batch_size', 'horizon', 'segment_len']
    
    def __init__(self, max_buf_size: int, max_dim: int, device: torch.device, segment_len: int, batch_size: int, horizon: int = 25):
        self.noise = torch.empty(max_buf_size, max_dim, device=device)
        self.t = torch.empty(max_buf_size, device=device)
        self.offsets_flow = torch.arange(horizon, device=device).unsqueeze(0)
        self.offsets_seg = torch.arange(segment_len, device=device).unsqueeze(0)
        self.device = device
        self.batch_size = batch_size
        self.horizon = horizon
        self.segment_len = segment_len


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
    """IPL Critic using TwinQ from model.py for robust Q-value estimation"""
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
    """Value network - optimized with inplace ReLU"""
    def __init__(self, obs_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


def sample_segment_batch_fast(neg_obs, neg_act, union_obs, union_act, buffers, return_prev_actions=False):
    """Optimized segment sampling with pre-computed offsets
    
    Args:
        return_prev_actions: If True, include 'prev_a' in returned dicts
    """
    B = buffers.batch_size
    k = buffers.segment_len
    offsets = buffers.offsets_seg
    device = buffers.device
    
    n_neg, T_neg = neg_obs.shape[:2]
    n_union, T_union = union_obs.shape[:2]
    max_neg = T_neg - k - 1
    max_union = T_union - k - 1

    idx_neg = torch.randint(0, n_neg, (B,), device=device)
    idx_union = torch.randint(0, n_union, (B,), device=device)
    start_neg = torch.randint(0, max_neg + 1, (B,), device=device)
    start_union = torch.randint(0, max_union + 1, (B,), device=device)

    neg_time_idx = start_neg.unsqueeze(1) + offsets
    neg_time_next = neg_time_idx + 1
    idx_neg_exp = idx_neg.unsqueeze(1).expand(-1, k)
    neg_s = neg_obs[idx_neg_exp, neg_time_idx]
    neg_a = neg_act[idx_neg_exp, neg_time_idx]
    neg_s_next = neg_obs[idx_neg_exp, neg_time_next]
    
    union_time_idx = start_union.unsqueeze(1) + offsets
    union_time_next = union_time_idx + 1
    idx_union_exp = idx_union.unsqueeze(1).expand(-1, k)
    union_s = union_obs[idx_union_exp, union_time_idx]
    union_a = union_act[idx_union_exp, union_time_idx]
    union_s_next = union_obs[idx_union_exp, union_time_next]

    result = {
        'neg': {'s': neg_s.transpose(0, 1), 'a': neg_a.transpose(0, 1), 's_next': neg_s_next.transpose(0, 1)},
        'union': {'s': union_s.transpose(0, 1), 'a': union_a.transpose(0, 1), 's_next': union_s_next.transpose(0, 1)}
    }
    
    if return_prev_actions:
        # Compute prev_a: shift actions, pad first with zeros
        neg_prev_a = torch.zeros_like(neg_a)
        neg_prev_a[:, 1:, :] = neg_a[:, :-1, :]
        union_prev_a = torch.zeros_like(union_a)
        union_prev_a[:, 1:, :] = union_a[:, :-1, :]
        result['neg']['prev_a'] = neg_prev_a.transpose(0, 1)
        result['union']['prev_a'] = union_prev_a.transpose(0, 1)
    
    return result


def sample_transitions_fast(union_obs, union_act, batch_size, device):
    """Fast transition sampling"""
    n, T = union_obs.shape[:2]
    idx = torch.randint(0, n, (batch_size,), device=device)
    t_idx = torch.randint(0, T - 1, (batch_size,), device=device)
    return union_obs[idx, t_idx], union_act[idx, t_idx]


def sample_flow_segments_fast(union_obs, union_act, buffers, horizon=None, return_prev_actions=False):
    """Optimized flow segment sampling with pre-computed offsets
    
    Args:
        return_prev_actions: If True, also return prev_actions [B, H, act_dim] where
                             prev_a[:, 0, :] = 0 and prev_a[:, 1:, :] = act[:, :-1, :]
    """
    B = buffers.batch_size
    H = horizon if horizon is not None else buffers.horizon
    offsets = buffers.offsets_flow[:, :H]
    device = buffers.device
    
    n, T = union_obs.shape[:2]
    max_start = T - H
    
    traj_idx = torch.randint(0, n, (B,), device=device)
    start_idx = torch.randint(0, max_start, (B,), device=device)
    
    time_idx = start_idx.unsqueeze(1) + offsets
    traj_exp = traj_idx.unsqueeze(1).expand(-1, H)
    
    seg_s = union_obs[traj_exp, time_idx]
    seg_a = union_act[traj_exp, time_idx]
    
    if not return_prev_actions:
        return seg_s, seg_a
    
    # Compute prev_actions: shift actions by 1, pad first with zeros
    # Shape: [B, H, act_dim]
    prev_a = torch.zeros_like(seg_a)
    prev_a[:, 1:, :] = seg_a[:, :-1, :]
    return seg_s, seg_a, prev_a


# ============== Loss Functions ==============

def ipl_preference_loss(q_critic, flow_model, batch_seg, gamma, chi2_coeff, target_clip, act_dim, 
                        alpha=0.1, diffusion_steps=10, use_prev_action=False):
    """
    Soft IPL preference loss with entropy-regularized inverse Bellman operator.
    Uses soft Q-target: Q(s',a') - α * log π(a'|s')
    """
    seg_u, seg_n = batch_seg["union"], batch_seg["neg"]
    s_neg, a_neg, s_neg_next = seg_n["s"], seg_n["a"], seg_n["s_next"]
    s_uni, a_uni, s_uni_next = seg_u["s"], seg_u["a"], seg_u["s_next"]
 
    k, B, obs_dim = s_neg.shape
    N = B * k

    neg_s_flat = s_neg.permute(1, 0, 2).reshape(N, obs_dim)
    neg_a_flat = a_neg.permute(1, 0, 2).reshape(N, act_dim)
    neg_s_next_flat = s_neg_next.permute(1, 0, 2).reshape(N, obs_dim)
    uni_s_flat = s_uni.permute(1, 0, 2).reshape(N, obs_dim)
    uni_a_flat = a_uni.permute(1, 0, 2).reshape(N, act_dim)
    uni_s_next_flat = s_uni_next.permute(1, 0, 2).reshape(N, obs_dim)

    obs_all = torch.cat([neg_s_flat, uni_s_flat], dim=0)
    act_all = torch.cat([neg_a_flat, uni_a_flat], dim=0)
    next_obs_all = torch.cat([neg_s_next_flat, uni_s_next_flat], dim=0)

    with torch.cuda.amp.autocast():
        q1, q2 = q_critic.both(obs_all, act_all)
        qs = torch.stack([q1.squeeze(-1), q2.squeeze(-1)], dim=0)

    # Soft inverse Bellman: sample next actions with log-prob
    # For next state, prev_action = current action (a_t becomes prev for s_{t+1})
    with torch.no_grad():
        if use_prev_action:
            next_actions, logp_next = flow_model.sample_and_logprob(
                next_obs_all, diffusion_steps=diffusion_steps, prev_actions=act_all
            )
        else:
            next_actions, logp_next = flow_model.sample_and_logprob(
                next_obs_all, diffusion_steps=diffusion_steps
            )
        q_next = q_critic.get_target_q(next_obs_all, next_actions).squeeze(-1)
        
        # Soft Q-target: Q(s',a') - α * log π(a'|s')
        soft_q_next = q_next - alpha * logp_next
        
        if target_clip:
            q_lim = 10.0 / (chi2_coeff * (gamma + 1e-6))
            soft_q_next = soft_q_next.clamp(-q_lim, q_lim)

    # Soft reward = Q(s,a) - γ * soft_Q(s',a')
    reward = qs - gamma * soft_q_next
    r_neg = reward[:, :N].view(2, B, k)
    r_uni = reward[:, N:].view(2, B, k)

    logits = r_uni.sum(dim=2) - r_neg.sum(dim=2)
    pref_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    chi2_loss = 0.5 * chi2_coeff * reward.pow(2).mean()
    
    return pref_loss + chi2_loss, r_uni.mean(), r_neg.mean(), logp_next.mean()

def v_expectile_loss(q_target, v_model, batch_seg, tau):
    """IQL-style expectile regression for V(s)"""
    seg_u, seg_n = batch_seg["union"], batch_seg["neg"]
    s_all = torch.cat([seg_u["s"].reshape(-1, seg_u["s"].shape[-1]),
                       seg_n["s"].reshape(-1, seg_n["s"].shape[-1])], dim=0)
    a_all = torch.cat([seg_u["a"].reshape(-1, seg_u["a"].shape[-1]),
                       seg_n["a"].reshape(-1, seg_n["a"].shape[-1])], dim=0)

    with torch.no_grad():
        q1, q2 = q_target.both(s_all, a_all)
        q_tgt = torch.min(q1, q2).squeeze(-1)

    v_pred = v_model(s_all)
    diff = q_tgt - v_pred
    weight = torch.abs(tau - (diff < 0).float())
    stats={
        "v_loss": (weight * diff.pow(2)).mean().item()
    }
    return (weight * diff.pow(2)).mean(), stats


def expectile_value_loss(v_model, q_critic, states, actions, tau):
    """Optimized expectile loss - returns tensors (no GPU sync)"""
    with torch.no_grad():
        q1, q2 = q_critic.q_target.both(actions, states)
        q_tgt = torch.min(q1, q2).squeeze(-1)
    
    v_pred = v_model(states)
    diff = q_tgt - v_pred
    weight = torch.abs(tau - (diff < 0).float())
    loss = (weight * diff.pow(2)).mean()
    return loss, v_pred.mean()


def psi_t_ot(x0, x1, t, sigma_min):
    """OT linear interpolation (fallback)"""
    return psi_t_ot_jit(x0, x1, t, sigma_min)


def u_t_ot(x_t, x1, t, sigma_min):
    """OT vector field (fallback)"""
    return u_t_ot_jit(x_t, x1, t, sigma_min)

@torch.no_grad()
def plot_q_energy_grid(q_critic, neg_data, union_data, device, save_path, energy_alpha=3.0):
    """
    Create a 1x2 grid:
      (1) Mean Q-value
      (2) Energy = softmax(alpha * Q)
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    def process(data_dict):
        obs = torch.as_tensor(data_dict["observations"], device=device, dtype=torch.float32)
        acts = torch.as_tensor(data_dict["actions"], device=device, dtype=torch.float32)

        B, T = obs.shape[:2]

        # Flatten
        obs_flat = obs.reshape(B * T, -1)
        act_flat = acts.reshape(B * T, -1)

        # Q critic
        q1, q2 = q_critic.both(obs_flat, act_flat)
        q = torch.min(q1, q2).squeeze(-1)

        # Mean Q per trajectory
        Q = q.reshape(B, T).mean(dim=1)

        # Reward & cost
        rewards = data_dict["rewards"].sum(axis=1)
        costs   = data_dict["costs"].sum(axis=1)

        R = rewards.cpu().numpy() if torch.is_tensor(rewards) else rewards
        C = costs.cpu().numpy()   if torch.is_tensor(costs)   else costs

        return Q, R, C

    # Process datasets
    Qn, Rn, Cn = process(neg_data)
    Qu, Ru, Cu = process(union_data)

    # Concatenate
    Q = torch.cat([Qn, Qu], dim=0)
    R = np.concatenate([Rn, Ru])
    C = np.concatenate([Cn, Cu])

    # --- ENERGY FROM Q ---
    E = F.softmax(energy_alpha * Q, dim=0).cpu().numpy()
    Q = Q.cpu().numpy()

    # --- PLOT ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        (Q, "Mean Q-value"),
        (E, f"Energy = softmax({energy_alpha} · Q)"),
    ]

    for ax, (colors, label) in zip(axes, panels):
        sc = ax.scatter(C, R, c=colors, cmap="viridis",
                        s=60, alpha=0.85, edgecolors="none")
        ax.set_xlabel("Total Cost")
        ax.set_ylabel("Total Reward")
        ax.set_title(label)

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(label)

        for i in range(len(colors)):
            ax.text(C[i], R[i], f"{i}", fontsize=6, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()

    print(f"🖼️ Saved Q/E grid → {save_path}")



@torch.no_grad()
def compute_advantage_energy(q_critic, states, actions, config):
    """
    Compute advantage A(s,a) = Q(s,a) - V(s) for energy guidance.
    """
    q_val = q_critic(states, actions)
    if q_val.dim() > 1:
        q_val = q_val.squeeze(-1)
    advantage = q_val
    energy = (advantage - advantage.mean()) / (advantage.std())
    return energy


def train_flow_matching_step(flow_model, flow_optimizer, q_critic,
                             states, actions, config, use_guidance=True):
    """Single-step flow matching with advantage-based energy guidance."""
    flow_model.train()
    batch = states.shape[0]
    device = states.device 
    energy = None
    if use_guidance:
        energy = compute_advantage_energy(q_critic, states, actions, config)
    
    x1 = actions
    x0 = torch.randn_like(x1)
    eps = 1e-6
    t = torch.rand(batch, device=device) * (1.0 - 2 * eps) + eps
    sigma_min = config.get('sigma_min', 0.01)
    
    x_t = psi_t_ot(x0, x1, t, sigma_min)
    u_t = u_t_ot(x_t, x1, t, sigma_min)
    
    v_theta = flow_model(x_t, t, condition=states)
    
    err = torch.sum((v_theta - u_t)**2, dim=1)
    
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


def train_flow_step_segment(flow_model, flow_optimizer, scaler, q_critic,
                            seg_s, seg_a, gamma_powers, buffers, config, use_guidance=True,
                            alpha=0.1, in_warmup=False, seg_prev_a=None):
    """
    Flow policy training with optional SAC entropy bonus.
    During warmup: Pure FM loss (fast, stable)
    After warmup: SAC policy loss + FM regularization
    
    Args:
        seg_prev_a: [B, H, act_dim] previous actions (optional, for prev_action conditioning)
    """
    flow_model.train()
    B, H, obs_dim = seg_s.shape
    act_dim = seg_a.shape[-1]
    device = seg_s.device
    sigma_min = config.get('sigma_min', 0.01)
    use_prev_action = config.get('use_prev_action', False)
    N = B * H
    
    # Flatten segments: [B*H, dim]
    s_flat = seg_s.reshape(N, obs_dim)
    a_flat = seg_a.reshape(N, act_dim)
    
    # Build condition: [obs] or [obs, prev_action]
    if use_prev_action and seg_prev_a is not None:
        prev_a_flat = seg_prev_a.reshape(N, act_dim)
        cond_flat = torch.cat([s_flat, prev_a_flat], dim=-1)
    else:
        cond_flat = s_flat
    
    # Compute segment weights (uniform during warmup)
    if use_guidance and not in_warmup:
        with torch.no_grad():
            q_val = q_critic(s_flat, a_flat).squeeze(-1)
            energy_flat = q_val
            w_seg, A_seg = compute_segment_weights_jit(
                energy_flat, gamma_powers, B, H, config.get('energy_alpha', 3.0)
            )
    else:
        w_seg = torch.full((B,), 1.0 / B, device=device)
        A_seg = torch.zeros(B, device=device)
    
    # ========== Flow Matching Loss (always computed) ==========
    noise = buffers.noise[:N, :act_dim]
    noise.normal_()
    t = buffers.t[:N]
    t.uniform_(1e-6, 1.0 - 1e-6)
    
    x_t = psi_t_ot_jit(noise, a_flat, t, sigma_min)
    u_t = u_t_ot_jit(x_t, a_flat, t, sigma_min)
    
    flow_optimizer.zero_grad(set_to_none=True)
    
    with torch.cuda.amp.autocast():
        v_theta = flow_model(x_t, t, condition=cond_flat)
        fm_loss = compute_weighted_loss_jit(v_theta, u_t, w_seg, B, H)
    
    
    scaler.scale(fm_loss).backward()
    scaler.unscale_(flow_optimizer)
    clip_grad_norm_(flow_model.parameters(), config['max_grad_norm'])
    scaler.step(flow_optimizer)
    scaler.update()
    
    flow_model.condition = None
    return fm_loss, A_seg.mean(), A_seg.std(), w_seg.max()


@torch.no_grad()
def evaluate_flow_policy(eval_env, flow_model, device, norm_fn, diffusion_steps=15, act_dim=None):
    """Evaluate single-step policy with optional prev_action conditioning"""
    obs, _ = eval_env.reset()
    obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
    obs = torch.as_tensor(norm_fn(obs), dtype=torch.float32, device=device).unsqueeze(0)
    
    # Track previous action for conditioning (if model uses it)
    use_prev_action = getattr(flow_model, 'use_prev_action', False)
    if act_dim is None:
        act_dim = flow_model.output_dim
    prev_action = torch.zeros(1, act_dim, device=device) if use_prev_action else None
    
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    
    while not done:
        act = flow_model.select_actions(obs, diffusion_steps=diffusion_steps, prev_actions=prev_action)
        next_obs, reward, terminated, truncated, info = eval_env.step(act)
        next_obs = np.array(next_obs) if not isinstance(next_obs, np.ndarray) else next_obs
        obs = torch.as_tensor(norm_fn(next_obs), dtype=torch.float32, device=device).unsqueeze(0)
        
        # Update prev_action for next step
        if use_prev_action:
            act_np = act if isinstance(act, np.ndarray) else np.array(act)
            prev_action = torch.as_tensor(act_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward += reward
        total_cost += info.get('cost', 0.0)
        total_len += 1
        done = terminated or truncated
    
    return total_reward, total_cost, total_len

def main(args):
    config = {**default_cfg}
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v
    
    device = torch.device(args.device if isinstance(args.device, str) 
                         else f"{args.device}:{getattr(args, 'device_id', 0)}")
    args.device = device
    
    # Enable performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = f"seed-{str(args.seed).zfill(3)}"
    relpath = f"{subfolder}-{relpath}"
    algo = "ipl_flow_joint_training_v7"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config({**config, **vars(args)})
    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))
    print(f"📊 TensorBoard logs: {tb_writer.log_dir}")
    print(f"   Run: tensorboard --logdir={args.log_dir}")
    
     
    # Environment & Dataset
    print(f"\nCreating environment: {args.task}")
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)
    
    dataset_config = {
        "density": config["density"],
        "inpaint_ranges": config.get("inpaint_ranges", ((0.0, 1.0, 0.0, 0.5),)),
        "num_negative_trajectories": config["num_negative_trajectories"],
        "num_union_trajectories": config["num_union_trajectories"],
        "non_pref_noise": 0.0,
    }
    
    raw_data = eval_env.get_dataset()
    dones_idx = np.where((raw_data["terminals"] == 1) | (raw_data["timeouts"] == 1))[0]
    traj_lengths = [dones_idx[0] + 1] + [dones_idx[i] - dones_idx[i-1] for i in range(1, len(dones_idx))]
    max_traj_len = max(traj_lengths)
    
    d4rl_data = get_dataset_in_d4rl_format(eval_env, dataset_config, args.task, max_traj_len, num_folds=1)
    neg_data, union_data = get_neg_and_union_data_2(d4rl_data, dataset_config)
    
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        neg_data, union_data, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
        mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32, device=device)
        std_obs = torch.as_tensor(std_obs, dtype=torch.float32, device=device)
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    
    # Preload to GPU
    print("\nPre-loading data to GPU...")
    neg_obs = torch.as_tensor(neg_data['observations'], dtype=torch.float32, device=device)
    neg_act = torch.as_tensor(neg_data['actions'], dtype=torch.float32, device=device)
    union_obs = torch.as_tensor(union_data['observations'], dtype=torch.float32, device=device)
    union_act = torch.as_tensor(union_data['actions'], dtype=torch.float32, device=device)
    print(f"📦 GPU: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB | neg={neg_obs.shape}, union={union_obs.shape}")
    
    # Models
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    
    print("\nInitializing models...")
    q_critic = IPL_TwinQ_Critic(obs_dim=obs_dim, act_dim=act_dim, args=args).to(device)

    use_prev_action = config.get('use_prev_action', False)
    flow_model = ScoreNet(
        input_dim=obs_dim + act_dim,
        output_dim=act_dim,
        marginal_prob_std=None,
        use_prev_action=use_prev_action,
        args=args
    ).to(device)
    
    # Fused optimizers for faster CUDA ops
    use_fused = device.type == 'cuda'
    q_opt = Adam(q_critic.parameters(), lr=config['q_lr'], weight_decay=config['weight_decay'], fused=use_fused)
    flow_opt = Adam(flow_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], fused=use_fused)
    
    # ========== SAC Alpha (Temperature) Infrastructure ==========
    target_entropy = config['target_entropy'] if config['target_entropy'] is not None else -act_dim


    auto_tune_alpha = config.get('auto_tune_alpha', True)
    
    # AMP GradScaler
    scaler = torch.cuda.amp.GradScaler()
    
    logger.log(f"TwinQ critic initialized: obs_dim={obs_dim}, act_dim={act_dim}")
    logger.log(f"Flow model: input_dim={obs_dim + act_dim}, output_dim={act_dim}, use_prev_action={use_prev_action}")
    logger.log(f"SAC alpha: init={config['alpha']:.4f}, target_entropy={target_entropy:.2f}, auto_tune={auto_tune_alpha}")
    logger.log(f"AMP enabled with fused={use_fused}")
    
    # ============================================================
    # Joint Training: Q + Flow together
    # ============================================================
    logger.log("\n" + "="*60)
    logger.log("Joint Training: IPL Q-learning + Flow Matching")
    logger.log("="*60)
    
    total_iters = config['preference_iterations']
    warmup_iters = config['warmup_iterations']
    flow_horizon = config['horizon']
    batch_size = config['batch_size']
    segment_len = config['segment_length']
    
    # Pre-allocate training buffers
    max_buf_size = batch_size * max(flow_horizon, segment_len)
    buffers = TrainingBuffers(max_buf_size, max(obs_dim, act_dim), device, segment_len, batch_size, flow_horizon)
    
    # Precompute gamma powers for segment advantage: [H]
    gamma_powers = (config['gamma'] ** torch.arange(flow_horizon, device=device, dtype=torch.float32))
    
    logger.log(f"📋 Config: warmup={warmup_iters}, horizon={flow_horizon}, weight_from_q={config['weight_from_q']}")
    logger.log(f"📋 Pre-allocated buffers: size={max_buf_size}")
    
    pbar = tqdm(range(total_iters), desc="JointTraining", unit="iter")
    start_time = time.time()
    
    # Tensor accumulators for logging (avoid GPU sync every step)
    acc_q_loss = torch.zeros(1, device=device)
    acc_r_union = torch.zeros(1, device=device)
    acc_r_neg = torch.zeros(1, device=device)
    acc_v_loss = torch.zeros(1, device=device)
    acc_v_mean = torch.zeros(1, device=device)
    acc_flow_loss = torch.zeros(1, device=device)
    acc_A_seg = torch.zeros(1, device=device)


    acc_count = 0

    for step in pbar:
        in_warmup = step < warmup_iters
        
        # ========== Sample Data ==========
        # Segment batch for IPL (only after warmup)
        if not in_warmup:
            batch_seg = sample_segment_batch_fast(neg_obs, neg_act, union_obs, union_act, buffers,
                                                   return_prev_actions=use_prev_action)
        
        # Flow segments from union data (use pre-allocated offsets)
        if use_prev_action:
            seg_s, seg_a, seg_prev_a = sample_flow_segments_fast(union_obs, union_act, buffers, flow_horizon,
                                                                  return_prev_actions=True)
        else:
            seg_s, seg_a = sample_flow_segments_fast(union_obs, union_act, buffers, flow_horizon)
            seg_prev_a = None
        
        # ========== Q-Learning Step (skip during warmup) ==========
        if not in_warmup:
            # Get current alpha value
            alpha = config['alpha']
            
            q_loss, r_union_mean, r_neg_mean, logp_q = ipl_preference_loss(
                q_critic, flow_model, batch_seg, 
                config['gamma'], config['chi2_coeff'], config['target_clipping'], act_dim,
                alpha=alpha, diffusion_steps=config.get('diffusion_steps', 10),
                use_prev_action=use_prev_action
            )

            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            clip_grad_norm_(q_critic.parameters(), config["max_grad_norm"])
            q_opt.step()
            
            # ========== V-Network Step ==========
            #REMOVED
            
            # Accumulate stats (no GPU sync)
            acc_q_loss += q_loss.detach()
            acc_r_union += r_union_mean.detach()
            acc_r_neg += r_neg_mean.detach()

        
        # ========== Flow Matching Step (segment-based) ==========
        use_flow_guidance = config['use_guidance'] and (not in_warmup)
        alpha_val = config['alpha']
        
        flow_loss, A_seg_mean, A_seg_std, w_seg_max = train_flow_step_segment(
            flow_model, flow_opt, scaler, q_critic,
            seg_s, seg_a, gamma_powers, buffers, config, use_guidance=use_flow_guidance,
            alpha=alpha_val, in_warmup=in_warmup, seg_prev_a=seg_prev_a
        )
        acc_flow_loss += flow_loss.detach()
        acc_A_seg += A_seg_mean.detach()
        acc_count += 1
        

        
        # ========== Target Update (after warmup) ==========
        if (not in_warmup) and ((step + 1) % config['target_update_freq'] == 0):
            update_target(q_critic.q_network, q_critic.q_target, config['target_tau'])
            
        # ========== Logging (sync here only) ==========
        if (step + 1) % config['log_freq'] == 0:
            elapsed = time.time() - start_time
            global_step = step + 1
            phase = "Warmup" if in_warmup else "Joint"
            
            # Sync accumulators (only GPU sync point in loop)
            n = max(acc_count, 1)
            q_loss_avg = (acc_q_loss / n).item() if not in_warmup else 0.0
            r_union_avg = (acc_r_union / n).item() if not in_warmup else 0.0
            r_neg_avg = (acc_r_neg / n).item() if not in_warmup else 0.0
            flow_loss_avg = (acc_flow_loss / n).item()
            A_seg_avg = (acc_A_seg / n).item()

            
            # Reset accumulators
            acc_q_loss.zero_()
            acc_r_union.zero_()
            acc_r_neg.zero_()
            acc_flow_loss.zero_()
            acc_A_seg.zero_()

        
            acc_count = 0
            
            # Progress bar
            pbar.set_description(
                f"{phase} | Q={q_loss_avg:.4f} | "
                f"Flow={flow_loss_avg:.6f} | α={alpha_val:.4f}"
            )
            
            # Text log
            logger.log(
                f"Step {step+1}/{total_iters} [{phase}] | "
                f"q_loss={q_loss_avg:.4f} "
                f"r_union={r_union_avg:.3f} "
                f"r_neg={r_neg_avg:.3f} | "
                f"flow_loss={flow_loss_avg:.6f} "
                f"A_seg={A_seg_avg:.3f} | "
                f"α={alpha_val:.4f} | "
                f"time={elapsed:.1f}s"
            )
            
            # TensorBoard
            tb_writer.add_scalar('Joint/q_loss', q_loss_avg, global_step)
            tb_writer.add_scalar('Joint/r_union_mean', r_union_avg, global_step)
            tb_writer.add_scalar('Joint/r_neg_mean', r_neg_avg, global_step)
            tb_writer.add_scalar('Joint/reward_gap', r_union_avg - r_neg_avg, global_step)
            tb_writer.add_scalar('Joint/flow_loss', flow_loss_avg, global_step)
            tb_writer.add_scalar('Flow/A_seg_mean', A_seg_avg, global_step)
            tb_writer.add_scalar('SAC/alpha', alpha_val, global_step)
            tb_writer.add_scalar('SAC/entropy', global_step)
            
        # ========== Checkpointing ==========
        if (step + 1) % config['save_freq'] == 0:
            logger.torch_save(itr=step+1, torch_saver_elements=q_critic, prefix="q_critic")
            logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow")
    
    # ========== Final Save ==========
    logger.torch_save(itr=total_iters, torch_saver_elements=flow_model, prefix="flow_final")
    logger.torch_save(itr=total_iters, torch_saver_elements=q_critic, prefix="q_critic_final")
    
    logger.log("✅ Joint training complete!")
    
    # ========== Visualization ==========
    plot_save_path = os.path.join(args.log_dir, "q_function_analysis.png")
    plot_q_energy_grid(
        q_critic, 
        neg_data, 
        union_data, 
        device, 
        plot_save_path, 
        energy_alpha=config['energy_alpha']
    )

    tb_writer.close()
    logger.close()
    
    logger.log("\n" + "="*60)
    logger.log("Training complete!")
    logger.log(f"📊 TensorBoard logs: {tb_writer.log_dir}")
    logger.log("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", "--env", dest="task", default="OfflineSwimmerVelocityGymnasium-v1")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default="/home/me22b018/safe_diff/offline_mpc/logs/merged/")
    parser.add_argument("--experiment", type=str, default="ipl_singlestagev2_smooth_woAFMloss")
    parser.add_argument("--normalize_observation", action="store_true", default=False)
    parser.add_argument("--preload_to_gpu", action="store_true", default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--schedule", type=str, default="Linear")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--preference_iterations", type=int, default=500000)
    parser.add_argument("--flow_train_iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=3e-4)
    parser.add_argument("--v_lr", type=float, default=3e-4)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--expectile_tau", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--energy_alpha", type=float, default=1)
    parser.add_argument("--use_guidance", action="store_true", default=True)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--target_update_freq", type=int, default=1)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=20000)
    parser.add_argument("--use_eval", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=4000)
    parser.add_argument("--eval_episode_freq", type=int, default=25)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--num_negative_trajectories", type=int, default=50)
    parser.add_argument("--num_union_trajectories", type=int, default=-1)
    parser.add_argument("--q_hidden", type=int, default=256)
    parser.add_argument("--v_hidden", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    # Flow warmup & segment config
    parser.add_argument("--warmup_iterations", type=int, default=50000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--weight_from_q", action="store_true", default=True,
                        help="If True, use Q(s,a) for weights; else use advantage Q(s,a)-V(s)")
    
    # SAC entropy regularization
    parser.add_argument("--alpha", type=float, default=0.1, help="Initial SAC temperature")
    parser.add_argument("--target_entropy", type=float, default=None, help="Target entropy (default: -act_dim)")
    parser.add_argument("--auto_tune_alpha", action="store_true", default=True, help="Auto-tune alpha")
    parser.add_argument("--alpha_lr", type=float, default=3e-4, help="Learning rate for alpha")
    parser.add_argument("--diffusion_steps", type=int, default=1, help="ODE steps for flow sampling")
    
    # Previous action conditioning
    parser.add_argument("--use_prev_action", action="store_true", default=False,
                        help="If True, condition policy on [obs, prev_action] instead of just obs")
    
    args = parser.parse_args()
    main(args)

