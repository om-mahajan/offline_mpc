#!/usr/bin/env python3
"""
IPL-Style Flow Matching V7: Unified Single-Phase Training


Architecture - Dhairan see this
  for iter = 1 to N:
      sample (s,a,s') ~ DU
      
      if iter % k_ipl == 0:  # IPL updates (slow)
          sample (σ+, σ−) ~ (DU, DN)
          update V via expectile regression
          update Q via preference loss + χ² regularization
      
      # Flow update (fast)
      A = stopgrad(Q(s,a) - V(s))
      w = softmax(α * A)
      L_flow = Σ w_i ||v_θ - u_t||²
      
      if iter % k_target == 0:  # Target updates (lazy)
          soft update Q_target, V_target
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
import matplotlib
matplotlib.use('Agg')
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
    # Unified training
    "total_iterations": int(5e5),
    "k_ipl": 1,                    # IPL update frequency (every k_ipl iterations)
    "k_target": 2,                 # Target network update frequency
    "warmup_iterations": 0,        # Flow-only warmup disabled by default
    # Logging/saving
    "log_freq": int(1e3),
    "save_freq": int(2e4),
    "max_grad_norm": 1.0,
    # Learning rates
    "lr": 3e-4,
    "q_lr": 3e-4,
    "v_lr": 3e-4,
    "weight_decay": 1e-5,
    # Network sizes
    "q_hidden": 256,
    "v_hidden": 256,
    # IPL config
    "expectile_tau": 0.5,
    "chi2_coeff": 0.5,
    "target_clipping": True,
    "segment_length": 25,
    # Flow config
    "batch_size": 256,
    "gamma": 0.99,
    "target_tau": 0.005,
    "energy_alpha": 10.0,
    "use_guidance": True,
    "sigma_min": 0.01,
    "weight_from_q": False,  # If True, use Q directly; else use advantage (Q-V)
    "segment_flow": False,   # If True, use segment-based advantage for flow weights
    "flow_segment_length": 25,  # Segment length for flow (can differ from IPL segment_length)
    # Dataset
    "density": 1.0,
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
    "device": "cuda",
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
    """IPL Critic using TwinQ for robust Q-value estimation"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q_network = TwinQ(action_dim=act_dim, state_dim=obs_dim)
        self.q_target = deepcopy(self.q_network).requires_grad_(False)
    
    def forward(self, obs, act):
        return self.q_network(act, obs)
    
    def both(self, obs, act):
        return self.q_network.both(act, obs)


class VNetwork(nn.Module):
    """Value network for state values"""
    def __init__(self, obs_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


# ============== Optimized Sampling Functions ==============

def sample_segment_batch_fast(neg_obs, neg_act, union_obs, union_act, 
                              segment_len, batch_size, device):
    """
    Fast segment sampling - assumes data already on GPU.
    Returns: dict with 'neg' and 'union' segments.
    """
    n_neg, T_neg = neg_obs.shape[:2]
    n_union, T_union = union_obs.shape[:2]
    max_neg = T_neg - segment_len - 1
    max_union = T_union - segment_len - 1

    idx_neg = torch.randint(0, n_neg, (batch_size,), device=device)
    idx_union = torch.randint(0, n_union, (batch_size,), device=device)
    start_neg = torch.randint(0, max_neg + 1, (batch_size,), device=device)
    start_union = torch.randint(0, max_union + 1, (batch_size,), device=device)

    # Vectorized slicing using gather
    offsets = torch.arange(segment_len, device=device).unsqueeze(0)  # [1, k]
    
    # Neg segments
    neg_time_idx = start_neg.unsqueeze(1) + offsets  # [B, k]
    neg_time_next = neg_time_idx + 1
    neg_s = neg_obs[idx_neg.unsqueeze(1).expand(-1, segment_len), neg_time_idx]
    neg_a = neg_act[idx_neg.unsqueeze(1).expand(-1, segment_len), neg_time_idx]
    neg_s_next = neg_obs[idx_neg.unsqueeze(1).expand(-1, segment_len), neg_time_next]
    
    # Union segments
    union_time_idx = start_union.unsqueeze(1) + offsets
    union_time_next = union_time_idx + 1
    union_s = union_obs[idx_union.unsqueeze(1).expand(-1, segment_len), union_time_idx]
    union_a = union_act[idx_union.unsqueeze(1).expand(-1, segment_len), union_time_idx]
    union_s_next = union_obs[idx_union.unsqueeze(1).expand(-1, segment_len), union_time_next]

    # Transpose to [k, B, dim] for IPL loss
    return {
        'neg': {'s': neg_s.transpose(0, 1), 'a': neg_a.transpose(0, 1), 's_next': neg_s_next.transpose(0, 1)},
        'union': {'s': union_s.transpose(0, 1), 'a': union_a.transpose(0, 1), 's_next': union_s_next.transpose(0, 1)}
    }


def sample_transitions_fast(union_obs, union_act, batch_size, device):
    """Fast transition sampling for flow - union data only, assumes GPU."""
    n_union, len_union = union_obs.shape[:2]
    traj_idx = torch.randint(0, n_union, (batch_size,), device=device)
    time_idx = torch.randint(0, len_union - 1, (batch_size,), device=device)
    return union_obs[traj_idx, time_idx], union_act[traj_idx, time_idx]


def sample_flow_segments_fast(union_obs, union_act, segment_len, batch_size, device):
    """Sample segments for flow training. Returns [B, H, dim] tensors."""
    n_union, T_union = union_obs.shape[:2]
    max_start = T_union - segment_len
    
    traj_idx = torch.randint(0, n_union, (batch_size,), device=device)
    start_idx = torch.randint(0, max_start, (batch_size,), device=device)
    
    offsets = torch.arange(segment_len, device=device)  # [H]
    time_idx = start_idx.unsqueeze(1) + offsets  # [B, H]
    traj_expanded = traj_idx.unsqueeze(1).expand(-1, segment_len)  # [B, H]
    
    seg_s = union_obs[traj_expanded, time_idx]  # [B, H, obs_dim]
    seg_a = union_act[traj_expanded, time_idx]  # [B, H, act_dim]
    return seg_s, seg_a


# ============== Loss Functions ==============

def ipl_preference_loss(q_critic, v_target, batch_seg, gamma, chi2_coeff, target_clip):
    """IPL preference loss over segments with chi² regularization"""
    seg_u, seg_n = batch_seg["union"], batch_seg["neg"]
    s_neg, a_neg, s_neg_next = seg_n["s"], seg_n["a"], seg_n["s_next"]
    s_uni, a_uni, s_uni_next = seg_u["s"], seg_u["a"], seg_u["s_next"]
 
    k, B, obs_dim = s_neg.shape
    act_dim = a_neg.shape[-1]
    N = B * k

    # Flatten all - use contiguous() to avoid slow strides
    neg_s_flat = s_neg.permute(1, 0, 2).contiguous().view(N, obs_dim)
    neg_a_flat = a_neg.permute(1, 0, 2).contiguous().view(N, act_dim)
    neg_s_next_flat = s_neg_next.permute(1, 0, 2).contiguous().view(N, obs_dim)
    uni_s_flat = s_uni.permute(1, 0, 2).contiguous().view(N, obs_dim)
    uni_a_flat = a_uni.permute(1, 0, 2).contiguous().view(N, act_dim)
    uni_s_next_flat = s_uni_next.permute(1, 0, 2).contiguous().view(N, obs_dim)

    obs_all = torch.cat([neg_s_flat, uni_s_flat], dim=0)
    act_all = torch.cat([neg_a_flat, uni_a_flat], dim=0)
    next_obs_all = torch.cat([neg_s_next_flat, uni_s_next_flat], dim=0)

    q1, q2 = q_critic.both(obs_all, act_all)
    qs = torch.stack([q1.squeeze(-1), q2.squeeze(-1)], dim=0)

    with torch.no_grad():
        v_next = v_target(next_obs_all).unsqueeze(0)
        if target_clip:
            q_lim = 1.0 / (chi2_coeff * (gamma + 1e-6))
            v_next = v_next.clamp(-q_lim, q_lim)

    reward = qs - gamma * v_next
    E = reward.shape[0]
    r_neg = reward[:, :N].view(E, B, k)
    r_uni = reward[:, N:].view(E, B, k)

    logits = r_uni.sum(dim=2) - r_neg.sum(dim=2)
    pref_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    chi2_loss = 0.5 * chi2_coeff * reward.pow(2).mean()

    # Return tensors - defer .item() to logging time to avoid GPU sync
    return pref_loss + chi2_loss, (pref_loss, chi2_loss, r_uni.mean(), r_neg.mean())


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
    return (weight * diff.pow(2)).mean()


# ============== Visualization ==============

@torch.no_grad()
def plot_q_v_adv_grid(q_critic, v_model, neg_data, union_data, device, save_path, energy_alpha=10.0):
    """Create 2x2 grid: Q, V, Advantage, Energy (softmax(alpha*A))."""
    import matplotlib.pyplot as plt

    def process(data_dict):
        obs = torch.as_tensor(data_dict["observations"], device=device, dtype=torch.float32)
        acts = torch.as_tensor(data_dict["actions"], device=device, dtype=torch.float32)
        B, T = obs.shape[:2]
        obs_flat, act_flat = obs.reshape(B * T, -1), acts.reshape(B * T, -1)

        q1, q2 = q_critic.both(obs_flat, act_flat)
        q = torch.min(q1, q2).squeeze(-1)
        v = v_model(obs_flat)
        a = q - v

        Q = q.reshape(B, T).mean(dim=1).cpu().numpy()
        V = v.reshape(B, T).mean(dim=1).cpu().numpy()
        A = a.reshape(B, T).mean(dim=1)

        rewards = data_dict["rewards"].sum(axis=1)
        costs = data_dict["costs"].sum(axis=1)
        R = rewards.cpu().numpy() if torch.is_tensor(rewards) else rewards
        C = costs.cpu().numpy() if torch.is_tensor(costs) else costs
        return Q, V, A, R, C

    Qn, Vn, An, Rn, Cn = process(neg_data)
    Qu, Vu, Au, Ru, Cu = process(union_data)

    Q = np.concatenate([Qn, Qu])
    V = np.concatenate([Vn, Vu])
    A = torch.cat([An, Au], dim=0)
    R = np.concatenate([Rn, Ru])
    C = np.concatenate([Cn, Cu])

    E = F.softmax(energy_alpha * A, dim=0).cpu().numpy()
    A = A.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    panels = [
        (Q, "Mean Q-value"),
        (V, "Mean V-value"),
        (A, "Advantage A = Q - V"),
        (E, f"Energy softmax({energy_alpha} * A)"),
    ]

    for ax, (colors, label) in zip(axes.flat, panels):
        sc = ax.scatter(C, R, c=colors, cmap="viridis", s=60, alpha=0.85, edgecolors="none")
        ax.set_xlabel("Total Cost")
        ax.set_ylabel("Total Reward")
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, label=label)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"Saved Q/V/A/E grid → {save_path}")


# ============== Flow Matching ==============

def psi_t_ot(x0, x1, t, sigma_min):
    """OT linear interpolation"""
    sigma_t = (1.0 - (1.0 - sigma_min) * t).view(-1, 1)
    return sigma_t * x0 + t.view(-1, 1) * x1


def u_t_ot(x_t, x1, t, sigma_min):
    """OT vector field"""
    denom = (1.0 - (1.0 - sigma_min) * t).view(-1, 1).clamp(min=1e-6)
    return (x1 - (1.0 - sigma_min) * x_t) / denom


def train_flow_step(flow_model, flow_opt, q_critic, v_model, states, actions, config, scaler, noise_buf, t_buf):
    """Single flow matching step with advantage weighting."""
    batch, device = states.shape[0], states.device
    
    # Compute weights (no grad through Q/V)
    with torch.no_grad():
        q_val = q_critic(states, actions)
        if q_val.dim() > 1:
            q_val = q_val.squeeze(-1)
        if config['weight_from_q']:
            energy = q_val
        else:
            energy = q_val - v_model(states)  # advantage
        weights = F.softmax((config['energy_alpha'] * energy).clamp(-20, 20), dim=0)

    # Flow matching - reuse pre-allocated buffers
    x1 = actions
    x0 = noise_buf[:batch].normal_()
    t = t_buf[:batch].uniform_(1e-6, 1.0 - 1e-6)
    sigma_min = config['sigma_min']
    
    x_t = psi_t_ot(x0, x1, t, sigma_min)
    u_t = u_t_ot(x_t, x1, t, sigma_min)
    
    # AMP forward pass
    with torch.cuda.amp.autocast():
        v_theta = flow_model(x_t, t, condition=states)
        err = ((v_theta - u_t) ** 2).sum(dim=1)
        loss = (err * weights).sum() if config['use_guidance'] else err.mean()

    flow_opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(flow_opt)
    clip_grad_norm_(flow_model.parameters(), config['max_grad_norm'])
    scaler.step(flow_opt)
    scaler.update()
    flow_model.condition = None
    
    return loss, energy.mean()  # Return tensors, defer .item()


def train_flow_step_segment(flow_model, flow_opt, q_critic, v_model, seg_s, seg_a, 
                            config, scaler, noise_buf, t_buf, gamma_powers):
    """Segment-based flow matching with discounted advantage weighting.
    
    Args:
        seg_s: [B, H, obs_dim]
        seg_a: [B, H, act_dim]
        gamma_powers: precomputed [H] tensor of gamma^t
    """
    B, H, obs_dim = seg_s.shape
    act_dim = seg_a.shape[-1]
    device = seg_s.device
    N = B * H
    
    # Flatten for batched Q/V forward pass
    s_flat = seg_s.reshape(N, obs_dim)
    a_flat = seg_a.reshape(N, act_dim)
    
    # Compute segment-level discounted advantage (no grad)
    with torch.no_grad():
        q_val = q_critic(s_flat, a_flat)
        if q_val.dim() > 1:
            q_val = q_val.squeeze(-1)
        if config['weight_from_q']:
            energy_flat = q_val
        else:
            energy_flat = q_val - v_model(s_flat)  # advantage
        
        # Reshape to [B, H] and apply gamma discounting
        energy_seg = energy_flat.view(B, H)
        A_seg = (energy_seg * gamma_powers).sum(dim=1)  # [B]
        w_seg = F.softmax((config['energy_alpha'] * A_seg).clamp(-20, 20), dim=0)  # [B]
    
    # Flow matching on all B*H transitions
    x1 = a_flat
    x0 = noise_buf[:N].normal_()
    t = t_buf[:N].uniform_(1e-6, 1.0 - 1e-6)
    sigma_min = config['sigma_min']
    
    x_t = psi_t_ot(x0, x1, t, sigma_min)
    u_t = u_t_ot(x_t, x1, t, sigma_min)
    
    # AMP forward pass
    with torch.cuda.amp.autocast():
        v_theta = flow_model(x_t, t, condition=s_flat)
        err = ((v_theta - u_t) ** 2).sum(dim=1)  # [N]
        
        if config['use_guidance']:
            # Reshape err to [B, H], apply segment weights, average
            err_seg = err.view(B, H)
            loss = (w_seg.unsqueeze(1) * err_seg).sum() / H  # weighted sum over B, mean over H
        else:
            loss = err.mean()
    
    flow_opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(flow_opt)
    clip_grad_norm_(flow_model.parameters(), config['max_grad_norm'])
    scaler.step(flow_opt)
    scaler.update()
    flow_model.condition = None
    
    return loss, A_seg.mean()  # Return tensors


# ============== Evaluation ==
@torch.no_grad()
def evaluate_policy(eval_env, flow_model, device, norm_fn):
    """Evaluate flow policy"""
    obs, _ = eval_env.reset()
    obs = torch.as_tensor(norm_fn(np.array(obs)), dtype=torch.float32, device=device).unsqueeze(0)
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    
    while not done:
        act = flow_model.select_actions(obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(act)
        obs = torch.as_tensor(norm_fn(np.array(next_obs)), dtype=torch.float32, device=device).unsqueeze(0)
        total_reward += reward
        total_cost += info.get('cost', 0.0)
        total_len += 1
        done = terminated or truncated
    
    return total_reward, total_cost, total_len


# ============== Main Training ==============

def main(args):
    config = {**default_cfg}
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v
    
    device = torch.device(args.device if isinstance(args.device, str) 
                         else f"{args.device}:{getattr(args, 'device_id', 0)}")
    
    # Setup logging
    relpath = f"seed-{str(args.seed).zfill(3)}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, "ipl_flow_v7_unified_lag5", relpath)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config({**config, **vars(args)})
    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))
    
    print(f" Logs: {args.log_dir}")
    print(f"   TensorBoard: tensorboard --logdir={args.log_dir}")
    
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
    
    # Keep full dicts on CPU for plotting (rewards/costs needed)
    neg_data_cpu, union_data_cpu = neg_data, union_data
    
    # Preload to GPU
    print("\nPre-loading data to GPU...")
    neg_obs = torch.as_tensor(neg_data['observations'], dtype=torch.float32, device=device)
    neg_act = torch.as_tensor(neg_data['actions'], dtype=torch.float32, device=device)
    union_obs = torch.as_tensor(union_data['observations'], dtype=torch.float32, device=device)
    union_act = torch.as_tensor(union_data['actions'], dtype=torch.float32, device=device)
    print(f" GPU: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB | neg={neg_obs.shape}, union={union_obs.shape}")
    
    # Models
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    
    q_critic = IPL_TwinQ_Critic(obs_dim, act_dim).to(device)
    v_model = VNetwork(obs_dim, config['v_hidden']).to(device)
    v_target = deepcopy(v_model).requires_grad_(False)
    flow_model = ScoreNet(obs_dim + act_dim, act_dim, marginal_prob_std=None, args=args).to(device)
    
    q_opt = Adam(q_critic.parameters(), lr=config['q_lr'], weight_decay=config['weight_decay'])
    v_opt = Adam(v_model.parameters(), lr=config['v_lr'], weight_decay=config['weight_decay'])
    flow_opt = Adam(flow_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    logger.log(f" Models: Q({obs_dim},{act_dim}), V({obs_dim}), Flow({obs_dim+act_dim}→{act_dim})")
    
    # Load checkpoint if provided
    if args.ckpt_dir and os.path.exists(args.ckpt_dir):
        logger.log(f" Loading Q critic from: {args.ckpt_dir}")
        q_critic.load_state_dict(torch.load(args.ckpt_dir, map_location=device, weights_only=True))
        q_critic.q_target.load_state_dict(q_critic.q_network.state_dict())
    
    # ============== UNIFIED TRAINING LOOP ==============
    logger.log("\n" + "="*60)
    logger.log("Unified IPL + Flow Training")
    logger.log(f"  Total iterations: {config['total_iterations']}")
    logger.log(f"  k_ipl: {config['k_ipl']} (IPL update freq)")
    logger.log(f"  k_target: {config['k_target']} (target update freq)")
    logger.log(f"  warmup: {config['warmup_iterations']} (flow-only warmup)")
    logger.log("="*60)
    
    total_iters = config['total_iterations']
    k_ipl = config['k_ipl']
    k_target = config['k_target']
    warmup = config['warmup_iterations']
    batch_size = config['batch_size']
    segment_len = config['segment_length']                      
    
    pbar = tqdm(range(1, total_iters + 1), desc="Training", unit="iter")
    start_time = time.time()
    
    # Stats accumulators for logging (use tensors to avoid GPU sync)
    flow_loss_acc = torch.tensor(0.0, device=device)
    adv_acc = torch.tensor(0.0, device=device)
    q_loss_acc = torch.tensor(0.0, device=device)
    v_loss_acc = torch.tensor(0.0, device=device)
    ipl_count = 0
    q_stats_last = None
    
    # AMP scaler and pre-allocated buffers
    scaler = torch.cuda.amp.GradScaler()
    use_segment_flow = config.get('segment_flow', False)
    flow_seg_len = config.get('flow_segment_length', segment_len)
    buf_size = batch_size * flow_seg_len if use_segment_flow else batch_size
    noise_buf = torch.empty(buf_size, act_dim, device=device)
    t_buf = torch.empty(buf_size, device=device)
    
    # Precompute gamma powers for segment advantage
    gamma_powers = (config['gamma'] ** torch.arange(flow_seg_len, device=device, dtype=torch.float32))
    
    for step in pbar:
        # ===== IPL UPDATE (slow) =====
        do_ipl = (step > warmup) and (step % k_ipl == 0)
        if do_ipl:
            batch_seg = sample_segment_batch_fast(
                neg_obs, neg_act, union_obs, union_act,
                segment_len, batch_size, device
            )
            
            # V update
            v_loss = v_expectile_loss(q_critic.q_target, v_model, batch_seg, config['expectile_tau'])
            v_opt.zero_grad()
            v_loss.backward()
            clip_grad_norm_(v_model.parameters(), config['max_grad_norm'])
            v_opt.step()
            
            # Q update
            q_loss, q_stats = ipl_preference_loss(
                q_critic, v_target, batch_seg,
                config['gamma'], config['chi2_coeff'], config['target_clipping']
            )
            q_opt.zero_grad()
            q_loss.backward()
            clip_grad_norm_(q_critic.parameters(), config['max_grad_norm'])
            q_opt.step()
            
            q_loss_acc += q_loss.detach()
            v_loss_acc += v_loss.detach()
            q_stats_last = q_stats
            ipl_count += 1
        
        # ===== FLOW UPDATE (fast) =====
        # Disable guidance during warmup (Q/V not trained yet, so advantages are meaningless)
        flow_config = {**config, 'use_guidance': config['use_guidance'] and (step > warmup)}
        
        if use_segment_flow:
            seg_s, seg_a = sample_flow_segments_fast(union_obs, union_act, flow_seg_len, batch_size, device)
            flow_loss, adv_mean = train_flow_step_segment(
                flow_model, flow_opt, q_critic, v_model, seg_s, seg_a,
                flow_config, scaler, noise_buf, t_buf, gamma_powers
            )
        else:
            states, actions = sample_transitions_fast(union_obs, union_act, batch_size, device)
            flow_loss, adv_mean = train_flow_step(flow_model, flow_opt, q_critic, v_model, states, actions, flow_config, scaler, noise_buf, t_buf)
        flow_loss_acc += flow_loss.detach()
        adv_acc += adv_mean.detach()
        
        # ===== TARGET UPDATE (lazy) =====
        if step % k_target == 0:
            update_target(v_model, v_target, config['target_tau'])
            update_target(q_critic.q_network, q_critic.q_target, config['target_tau'])
        
        # ===== LOGGING =====
        if step % config['log_freq'] == 0:
            elapsed = time.time() - start_time
            # Call .item() only at logging time - single GPU sync point
            avg_flow = (flow_loss_acc / config['log_freq']).item()
            avg_adv = (adv_acc / config['log_freq']).item()
            avg_q = (q_loss_acc / max(ipl_count, 1)).item() if ipl_count > 0 else 0.0
            avg_v = (v_loss_acc / max(ipl_count, 1)).item() if ipl_count > 0 else 0.0
            
            pbar.set_description(f"Flow={avg_flow:.4f} | Q={avg_q:.3f} | V={avg_v:.3f} | Adv={avg_adv:.2f}")
            logger.log(f"Step {step}/{total_iters} | flow={avg_flow:.5f} q={avg_q:.4f} v={avg_v:.4f} adv={avg_adv:.3f} | {elapsed:.0f}s")
            
            tb_writer.add_scalar('Train/flow_loss', avg_flow, step)
            tb_writer.add_scalar('Train/advantage_mean', avg_adv, step)
            if ipl_count > 0:
                tb_writer.add_scalar('Train/q_loss', avg_q, step)
                tb_writer.add_scalar('Train/v_loss', avg_v, step)
                if q_stats_last is not None:
                    tb_writer.add_scalar('Train/r_union', q_stats_last[2].item(), step)
                    tb_writer.add_scalar('Train/r_neg', q_stats_last[3].item(), step)
            
            # Reset accumulators
            flow_loss_acc = torch.tensor(0.0, device=device)
            adv_acc = torch.tensor(0.0, device=device)
            q_loss_acc = torch.tensor(0.0, device=device)
            v_loss_acc = torch.tensor(0.0, device=device)
            ipl_count = 0
        
        # ===== SAVE =====
        if step % config['save_freq'] == 0:
            logger.torch_save(itr=step, torch_saver_elements=flow_model, prefix="flow")
            logger.torch_save(itr=step, torch_saver_elements=q_critic, prefix="q_critic")
            logger.torch_save(itr=step, torch_saver_elements=v_model, prefix="v")
    
    # Final save
    logger.torch_save(itr=total_iters, torch_saver_elements=flow_model, prefix="flow_final")
    logger.torch_save(itr=total_iters, torch_saver_elements=q_critic, prefix="q_critic_final")
    logger.torch_save(itr=total_iters, torch_saver_elements=v_model, prefix="v_final")
    
    # Q/V/Advantage visualization
    plot_q_v_adv_grid(
        q_critic, v_model, neg_data_cpu, union_data_cpu, device,
        save_path=os.path.join(args.log_dir, "q_v_adv_energy_grid.png"),
        energy_alpha=config['energy_alpha']
    )
    
    tb_writer.close()
    logger.log("\n" + "="*60)
    logger.log(f" Training complete! Total time: {time.time() - start_time:.0f}s")
    logger.log("="*60)
    logger.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Task & logging
    parser.add_argument("--task", "--env", dest="task", default="OfflineSwimmerVelocityGymnasium-v1")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default="/home/me22b018/safe_diff/offline_mpc/logs/merged")
    parser.add_argument("--experiment", type=str, default="ipl_flow_unified_iplbaseline")
    
    # Training config
    parser.add_argument("--total_iterations", type=int, default=1000000)
    parser.add_argument("--k_ipl", type=int, default=1, help="IPL update frequency")
    parser.add_argument("--k_target", type=int, default=2, help="Target update frequency")
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Flow-only warmup iterations")
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Learning rates
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=3e-4)
    parser.add_argument("--v_lr", type=float, default=3e-4)
    
    # IPL config
    parser.add_argument("--expectile_tau", type=float, default=0.7)
    parser.add_argument("--chi2_coeff", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--segment_length", type=int, default=25)
    
    # Flow config
    parser.add_argument("--energy_alpha", type=float, default=10.0)
    parser.add_argument("--use_guidance", action="store_true", default=True)
    parser.add_argument("--weight_from_q", action="store_true", default=False, help="Use Q directly instead of advantage (Q-V) for flow weights")
    parser.add_argument("--segment_flow", action="store_true", default=False, help="Use segment-based discounted advantage for flow weights")
    parser.add_argument("--flow_segment_length", type=int, default=25, help="Segment length for segment-based flow")
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--target_tau", type=float, default=0.005)
    
    # Other
    parser.add_argument("--normalize_observation", action="store_true", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--schedule", type=str, default="Linear")
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=20000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--num_negative_trajectories", type=int, default=50)
    parser.add_argument("--num_union_trajectories", type=int, default=-1)
    parser.add_argument("--q_hidden", type=int, default=256)
    parser.add_argument("--v_hidden", type=int, default=256)
    parser.add_argument("--ckpt_dir", type=str, default="")
    
    args = parser.parse_args()
    main(args)
