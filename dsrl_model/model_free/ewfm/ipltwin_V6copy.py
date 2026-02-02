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
    "log_freq": int(1e3),
    "save_freq": int(2e4),
    "eval_episode_freq": 10,
    "max_grad_norm": 1.0,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    # IPL preference training
    "preference_iterations": int(1.5e5),
    "q_lr": 3e-4,
    "v_lr": 3e-4,
    "q_hidden": 256,
    "v_hidden": 256,
    "expectile_tau": 0.5,
    "chi2_coeff": 0.5,
    "target_clipping": True,
    # Flow training
    "flow_train_iterations": int(5e5),
    "batch_size": 256,
    "device": "cuda",
    "gamma": 0.99,
    "target_update_freq": 2,
    "target_tau": 0.005,
    # Preference loss weight
    "lambda_pref": 1.0,
    "lambda_reg": 0.1,
    # Energy guidance for flow
    "energy_alpha": 10.0,
    "use_guidance": True,
    # Dataset config
    "density": 1.0,
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
    "segment_length": 25,
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


def sample_segment_batch(neg_data, union_data, segment_len, batch_size, device,
                         norm_fn=None, data_on_gpu=True):
    """
    Returns paired preference segments σ⁺, σ⁻ with shapes:
      s, a, s_next : [segment_len, batch_size, dim]
    Exactly what IPL requires (Eq. 1 in paper).
    """
    if data_on_gpu:
        neg_obs = neg_data['observations']
        neg_act = neg_data['actions']
        union_obs = union_data['observations']
        union_act = union_data['actions']

        n_neg, T_neg = neg_obs.shape[:2]
        n_union, T_union = union_obs.shape[:2]

        max_neg = T_neg - segment_len - 1
        max_union = T_union - segment_len - 1

        idx_neg = torch.randint(0, n_neg, (batch_size,), device=device)
        idx_union = torch.randint(0, n_union, (batch_size,), device=device)
        start_neg = torch.randint(0, max_neg + 1, (batch_size,), device=device)
        start_union = torch.randint(0, max_union + 1, (batch_size,), device=device)

        def slice_segments(obs, act, idx, start):
            seg_s = []
            seg_a = []
            seg_s_NEXT = []
            for b in range(batch_size):
                i, st = idx[b].item(), start[b].item()
                seg_s.append(obs[i, st:st+segment_len])
                seg_a.append(act[i, st:st+segment_len])
                seg_s_NEXT.append(obs[i, st+1:st+segment_len+1])
            s = torch.stack(seg_s, 0).transpose(0, 1)
            a = torch.stack(seg_a, 0).transpose(0, 1)
            s_next = torch.stack(seg_s_NEXT, 0).transpose(0, 1)
            return s, a, s_next

        neg_s, neg_a, neg_s_next = slice_segments(neg_obs, neg_act, idx_neg, start_neg)
        union_s, union_a, union_s_next = slice_segments(union_obs, union_act, idx_union, start_union)

    else:
        raise NotImplementedError()

    if norm_fn:
        neg_s = norm_fn(neg_s); neg_s_next = norm_fn(neg_s_next)
        union_s = norm_fn(union_s); union_s_next = norm_fn(union_s_next)

    return {
        'neg':   {'s': neg_s, 'a': neg_a, 's_next': neg_s_next},
        'union': {'s': union_s, 'a': union_a, 's_next': union_s_next}
    }

def ipl_preference_loss_segments(q_critic, v_target, batch_seg, config):
    """
    IPL preference loss over segments with chi² regularization
    """
    gamma        = config.get("gamma", 0.99)
    chi2_coeff   = config.get("chi2_coeff", 0.5)
    target_clip  = config.get("target_clipping", True)

    seg_u = batch_seg["union"]
    seg_n = batch_seg["neg"]

    s_neg, a_neg, s_neg_next = seg_n["s"], seg_n["a"], seg_n["s_next"]
    s_uni, a_uni, s_uni_next = seg_u["s"], seg_u["a"], seg_u["s_next"]

    k, B, obs_dim = s_neg.shape
    act_dim = a_neg.shape[-1]

    s_neg_bk      = s_neg.permute(1, 0, 2).contiguous()
    a_neg_bk      = a_neg.permute(1, 0, 2).contiguous()
    s_negn_bk     = s_neg_next.permute(1, 0, 2).contiguous()

    s_uni_bk      = s_uni.permute(1, 0, 2).contiguous()
    a_uni_bk      = a_uni.permute(1, 0, 2).contiguous()
    s_unin_bk     = s_uni_next.permute(1, 0, 2).contiguous()

    N = B * k
    neg_s_flat      = s_neg_bk.reshape(N, obs_dim)
    neg_a_flat      = a_neg_bk.reshape(N, act_dim)
    neg_s_next_flat = s_negn_bk.reshape(N, obs_dim)

    uni_s_flat      = s_uni_bk.reshape(N, obs_dim)
    uni_a_flat      = a_uni_bk.reshape(N, act_dim)
    uni_s_next_flat = s_unin_bk.reshape(N, obs_dim)

    obs_all      = torch.cat([neg_s_flat,      uni_s_flat],      dim=0)
    act_all      = torch.cat([neg_a_flat,      uni_a_flat],      dim=0)
    next_obs_all = torch.cat([neg_s_next_flat, uni_s_next_flat], dim=0)

    q1, q2 = q_critic.both(obs_all, act_all)
    q1 = q1.squeeze(-1)
    q2 = q2.squeeze(-1)
    qs = torch.stack([q1, q2], dim=0)

    with torch.no_grad():
        v_next = v_target(next_obs_all).unsqueeze(0)
        if target_clip:
            q_lim = 1.0 / (chi2_coeff * (gamma + 1e-6))
            v_next = torch.clamp(v_next, -q_lim, q_lim)

    reward = qs - gamma * v_next

    E = reward.shape[0]
    r_neg = reward[:, :N].view(E, B, k)
    r_uni = reward[:, N:].view(E, B, k)

    R_neg = r_neg.sum(dim=2)
    R_uni = r_uni.sum(dim=2)
    #R_neg = R_neg.min(dim=0)[0]
    #R_uni = R_uni.min(dim=0)[0]
    logits = R_uni - R_neg
    labels = torch.ones_like(logits)
    pref_loss = F.binary_cross_entropy_with_logits(logits, labels)

    #chi2_loss = chi2_coeff * (F.smooth_l1_loss(reward, torch.zeros_like(reward)) + F.smooth_l1_loss(qs, torch.zeros_like(qs)))

    chi2_loss = 0.5 *chi2_coeff * (reward.pow(2).mean())

    total_loss = pref_loss + chi2_loss

    stats = {
        "pref_loss":    pref_loss.item(),
        "chi2_loss":    chi2_loss.item(),
        "total_loss":   total_loss.item(),
        "r_union_mean": r_uni.mean().item(),
        "r_neg_mean":   r_neg.mean().item(),
    }
    return stats, total_loss

def v_expectile_loss(q_critic, v_model, batch_seg, config):
    """
    IQL-style expectile regression for V(s) using min over TwinQ heads
    """
    tau = config.get("expectile_tau", 0.7)

    seg_u = batch_seg["union"]
    seg_n = batch_seg["neg"]

    s_u, a_u = seg_u["s"], seg_u["a"]
    s_n, a_n = seg_n["s"], seg_n["a"]

    s_all = torch.cat(
        [s_u.reshape(-1, s_u.shape[-1]),
         s_n.reshape(-1, s_n.shape[-1])],
        dim=0,
    )
    a_all = torch.cat(
        [a_u.reshape(-1, a_u.shape[-1]),
         a_n.reshape(-1, a_n.shape[-1])],
        dim=0,
    )

    with torch.no_grad():
        q1, q2 = q_critic.q_target.both(s_all, a_all)
        q_target = torch.min(q1, q2).squeeze(-1)

    v_pred = v_model(s_all)

    diff = q_target - v_pred
    weight = torch.abs(tau - (diff < 0).float())
    #v_loss = (weight * diff.pow(2)).mean() + 0.01 *(v_pred.pow(2).mean())
    v_loss = (weight * diff.pow(2)).mean()
    
    return {"v_loss": v_loss.item()}, v_loss


def sample_transition_batch(neg_data, union_data, batch_size, device, 
                            norm_fn=None, data_on_gpu=True):
    """Sample single-step transitions (s, a, s') for flow matching."""
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
def plot_q_v_adv_grid(q_critic, v_model, neg_data, union_data, device, save_path, energy_alpha=10.0):
    """
    Create a 2x2 grid: Q, V, Advantage, and Energy (softmax(alpha*A)).
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    def process(data_dict):
        obs = torch.as_tensor(data_dict["observations"], device=device, dtype=torch.float32)
        acts = torch.as_tensor(data_dict["actions"], device=device, dtype=torch.float32)
        T = obs.shape[1]
        B = obs.shape[0]

        # Flatten inputs
        obs_flat = obs.reshape(B*T, -1)
        act_flat = acts.reshape(B*T, -1)

        # Critic + V
        q1, q2 = q_critic.both(obs_flat, act_flat)
        q = torch.min(q1, q2).squeeze(-1)
        v = v_model(obs_flat)
        a = q - v                        # advantage

        Q = q.reshape(B, T).mean(dim=1).cpu().numpy()
        V = v.reshape(B, T).mean(dim=1).cpu().numpy()
        A = a.reshape(B, T).mean(dim=1)

        # reward & cost
        rewards = data_dict["rewards"].sum(axis=1)
        costs   = data_dict["costs"].sum(axis=1)
        R = rewards.cpu().numpy() if torch.is_tensor(rewards) else rewards
        C = costs.cpu().numpy()   if torch.is_tensor(costs)   else costs

        return Q, V, A, R, C

    # Process neg + union
    Qn, Vn, An, Rn, Cn = process(neg_data)
    Qu, Vu, Au, Ru, Cu = process(union_data)

    # Concatenate
    Q = np.concatenate([Qn, Qu])
    V = np.concatenate([Vn, Vu])
    A = torch.cat([An, Au], dim=0)     # ADV still torch
    R = np.concatenate([Rn, Ru])
    C = np.concatenate([Cn, Cu])

    # --- ENERGY from ADVANTAGE ---
    # Energy = softmax(alpha * A)
    E = F.softmax(energy_alpha * A, dim=0).cpu().numpy()

    # Convert A to numpy now
    A = A.cpu().numpy()

    # --- PLOT GRID ---
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

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(label)

        # annotate trajectory id
        for i in range(len(colors)):
            ax.text(C[i], R[i], f"{i}", fontsize=6, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"🖼️ Saved 4-panel Q/V/A/E grid → {save_path}")


@torch.no_grad()
def compute_advantage_energy(q_critic, v_model, states, actions, config):
    """
    Compute advantage A(s,a) = Q(s,a) - V(s) for energy guidance.
    """
    q_val = q_critic(states, actions)
    if q_val.dim() > 1:
        q_val = q_val.squeeze(-1)
    advantage = q_val 
    
    return advantage


def train_flow_matching_step(flow_model, flow_optimizer, q_critic, v_model,
                             states, actions, config, use_guidance=True):
    """Single-step flow matching with advantage-based energy guidance."""
    flow_model.train()
    batch = states.shape[0]
    device = states.device
    
    
    
    energy = None
    if use_guidance:
        energy = compute_advantage_energy(q_critic, v_model, states, actions, config)
    
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


@torch.no_grad()
def evaluate_flow_policy(eval_env, flow_model, device, norm_fn, diffusion_steps=15):
    """Evaluate single-step policy"""
    obs, _ = eval_env.reset()
    obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
    obs = torch.as_tensor(norm_fn(obs), dtype=torch.float32, device=device).unsqueeze(0)
    
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    
    while not done:
        act = flow_model.select_actions(obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(act)
        next_obs = np.array(next_obs) if not isinstance(next_obs, np.ndarray) else next_obs
        obs = torch.as_tensor(norm_fn(next_obs), dtype=torch.float32, device=device).unsqueeze(0)
        
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
    
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = f"seed-{str(args.seed).zfill(3)}"
    relpath = f"{subfolder}-{relpath}"
    algo = "ipl_flow_single_step_fixed_v6_with_viz"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config({**config, **vars(args)})
    tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))
    print(f"📊 TensorBoard logs: {tb_writer.log_dir}")
    print(f"   Run: tensorboard --logdir={args.log_dir}")
    
    
    print(f"Creating environment: {args.task}")
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)
    
    print("\nLoading DSRL dataset...")
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
    
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        print("Normalizing observations...")
        neg_data, union_data, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
        mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32).to(device)
        std_obs = torch.as_tensor(std_obs, dtype=torch.float32).to(device)
    
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    
    if args.preload_to_gpu:
        print("\nPre-loading to GPU...")
        for key in neg_data.keys():
            neg_data[key] = torch.as_tensor(neg_data[key], dtype=torch.float32).to(device)
            union_data[key] = torch.as_tensor(union_data[key], dtype=torch.float32).to(device)
        print(f"✅ GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    print(f"\nDataset: neg={neg_data['observations'].shape}, union={union_data['observations'].shape}")
    
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    
    print("\nInitializing models...")
    q_critic = IPL_TwinQ_Critic(obs_dim=obs_dim, act_dim=act_dim, args=args).to(device)
    q_opt = Adam(q_critic.parameters(), lr=config['q_lr'], weight_decay=config['weight_decay'])
    
    v_model = VNetwork(obs_dim=obs_dim, hidden_size=config['v_hidden']).to(device)
    v_target = deepcopy(v_model).requires_grad_(False).to(device)
    v_opt = Adam(v_model.parameters(), lr=config['v_lr'], weight_decay=config['weight_decay'])
    
    flow_model = ScoreNet(
        input_dim=obs_dim + act_dim,
        output_dim=act_dim,
        marginal_prob_std=None,
        args=args
    ).to(device)
    flow_opt = Adam(flow_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    logger.log(f"✅ TwinQ critic initialized: obs_dim={obs_dim}, act_dim={act_dim}")
    logger.log(f"✅ Flow model: input_dim={obs_dim + act_dim}, output_dim={act_dim}")
    
    # Phase 1: IPL Preference Learning
    logger.log("\n" + "="*60)
    logger.log("Phase 1: IPL Preference Learning (TwinQ + V)")
    logger.log("="*60)
    
    pref_iters = config['preference_iterations']
    pbar = tqdm(range(pref_iters), desc="Phase1:IPL", unit="iter")
    start_time = time.time()
    
    for step in pbar:
        batch_seg = sample_segment_batch(
            neg_data, union_data,
            segment_len=config['segment_length'],
            batch_size=config['batch_size'],
            device=device,
            norm_fn=norm_fn if args.normalize_observation else None,
            data_on_gpu=args.preload_to_gpu,
        )
        v_stats, v_loss = v_expectile_loss(q_critic, v_model, batch_seg, config)
        v_opt.zero_grad()
        v_loss.backward()
        clip_grad_norm_(v_model.parameters(), config["max_grad_norm"]) 
        v_opt.step()

        stats, q_loss = ipl_preference_loss_segments(
            q_critic, v_target, batch_seg, config
        )

        q_opt.zero_grad()
        q_loss.backward()
        clip_grad_norm_(q_critic.parameters(), config["max_grad_norm"])
        q_opt.step()

    # Update target networks
        
        
        if (step + 1) % config['target_update_freq'] == 0:
            update_target(v_model, v_target, config['target_tau'])
            update_target(q_critic.q_network, q_critic.q_target, config['target_tau'])
            
            
        
        if (step + 1) % config['log_freq'] == 0:
            elapsed = time.time() - start_time
            pbar.set_description(
                f"Phase1 | Pref={stats['pref_loss']:.4f} | "
                f"V={v_stats['v_loss']:.4f} | "
                f"total_loss={stats['total_loss']:.2f}"
            )
            logger.log(
                f"IPL step {step+1}/{pref_iters} "
                f"pref_loss={stats['pref_loss']:.4f} "
                f"chi2_loss={stats['chi2_loss']:.6f} "
                f"v_loss={v_stats['v_loss']:.6f} "
                f"total_loss={stats['total_loss']:.3f} "
                f"r_union_mean={stats['r_union_mean']:.3f} "
                f"r_neg_mean={stats['r_neg_mean']:.3f} "
                f"time={elapsed:.1f}s"
            )
            # TENSORBOARD SCALARS
            global_step = step + 1
            tb_writer.add_scalar('Phase1/pref_loss', stats['pref_loss'], global_step)
            tb_writer.add_scalar('Phase1/chi2_loss', stats['chi2_loss'], global_step)
            tb_writer.add_scalar('Phase1/v_loss', v_stats['v_loss'], global_step)
            tb_writer.add_scalar('Phase1/total_loss', stats['total_loss'], global_step)
            tb_writer.add_scalar('Phase1/r_union_mean', stats['r_union_mean'], global_step)
            tb_writer.add_scalar('Phase1/r_neg_mean', stats['r_neg_mean'], global_step)
            tb_writer.add_scalar('Phase1/reward_gap', 
                               stats['r_union_mean'] - stats['r_neg_mean'], global_step)
            

             
        if (step + 1) % config['save_freq'] == 0:
            logger.torch_save(itr=step+1, torch_saver_elements=q_critic, prefix="q_critic")
            logger.torch_save(itr=step+1, torch_saver_elements=v_model, prefix="v")
    
    logger.log("✅ IPL preference learning finished.")

    plot_q_v_adv_grid(
        q_critic, v_model,
        neg_data, union_data,
        device,
        save_path=os.path.join(args.log_dir, "q_v_adv_energy_grid.png"))


    # Freeze Q and V
    for p in q_critic.parameters():
        p.requires_grad = False
    for p in v_model.parameters():
        p.requires_grad = False
    q_critic.eval()
    v_model.eval()
    
    # Phase 2: Flow Matching
    logger.log("\n" + "="*60)
    logger.log("Phase 2: Flow Matching with Advantage Guidance (TwinQ)")
    logger.log("="*60)
    flow_start_step = pref_iters 
    flow_iters = config['flow_train_iterations']
    pbar = tqdm(range(flow_iters), desc="Phase2:Flow", unit="iter")
    start_time = time.time()
    
    for step in pbar:
        batch_data = sample_transition_batch(
            neg_data, union_data, config['batch_size'], device,
            norm_fn if args.normalize_observation else None,
            data_on_gpu=args.preload_to_gpu
        )
        
        states = batch_data['union']['s']
        actions = batch_data['union']['a']
        
        flow_loss = train_flow_matching_step(
            flow_model, flow_opt, q_critic, v_model,
            states, actions, config, use_guidance=config['use_guidance']
        )
        # ENHANCED LOGGING
        if (step + 1) % config['log_freq'] == 0:
            elapsed = time.time() - start_time
            global_step = flow_start_step + step + 1
            
            # Console
            pbar.set_description(f"Phase2 | Flow={flow_loss:.6f}")
            
            # Text log
            logger.log(f"Flow step {step+1}/{flow_iters} loss={flow_loss:.6f} time={elapsed:.1f}s")
            
            # TENSORBOARD
            tb_writer.add_scalar('Phase2/flow_loss', flow_loss, global_step)
        
        if (step + 1) % config['save_freq'] == 0:
            logger.torch_save(itr=step+1, torch_saver_elements=flow_model, prefix="flow")
    
    logger.torch_save(itr=flow_iters, torch_saver_elements=flow_model, prefix="flow_final")
    logger.torch_save(itr=flow_iters, torch_saver_elements=q_critic, prefix="q_critic_final")
    logger.torch_save(itr=flow_iters, torch_saver_elements=v_model, prefix="v_final")
    tb_writer.close()
    logger.close()
    
    logger.log("\n" + "="*60)
    logger.log("Training complete!")

    logger.log(f"TensorBoard logs: {tb_writer.log_dir}")
    logger.log("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", "--env", dest="task", default="OfflineSwimmerVelocityGymnasium-v1")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--experiment", type=str, default="ipl_twinq_fixed_v6")
    parser.add_argument("--normalize_observation", action="store_true", default=False)
    parser.add_argument("--preload_to_gpu", action="store_true", default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--schedule", type=str, default="Linear")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--preference_iterations", type=int, default=600000)
    parser.add_argument("--flow_train_iterations", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=3e-4)
    parser.add_argument("--v_lr", type=float, default=3e-4)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--expectile_tau", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--energy_alpha", type=float, default=10.0)
    parser.add_argument("--use_guidance", action="store_true", default=True)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--target_update_freq", type=int, default=1)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--use_eval", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=4000)
    parser.add_argument("--eval_episode_freq", type=int, default=10)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--num_negative_trajectories", type=int, default=50)
    parser.add_argument("--num_union_trajectories", type=int, default=-1)
    parser.add_argument("--q_hidden", type=int, default=256)
    parser.add_argument("--v_hidden", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    args = parser.parse_args()
    main(args)
