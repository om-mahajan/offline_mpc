"""
Weighted Flow Matching Policy with IQL Critic

This script replaces weighted behavioral cloning with OT-based weighted flow matching
for policy learning, while preserving IQL critic/value training and discriminator-based
cost estimation.

Based on train.py and ipltwin_V5.py

# =============================================================================
# TODOs / Implementation Plan:
# =============================================================================
# [x] TODO 1: Add OT flow matching helper functions (psi_t_ot, u_t_ot)
# [x] TODO 2: Import ScoreNet from diffusion_SDE/model.py with marginal_prob_std=None
# [x] TODO 3: Import DoubleQCritic, SingleV, Discriminator from agent/sac_models.py
# [x] TODO 4: Create FlowAgent class/namespace to hold all components inline
# [x] TODO 5: Implement update_actor_flow_matching (weighted flow matching loss)
# [x] TODO 6: Update eval_parallel_flow to use flow_model.select_actions()
# [x] TODO 7: Add TensorBoard logging with SummaryWriter
# [x] TODO 8: Add checkpoint saving for flow_model, critic, value, discriminators
# [x] TODO 9: Expose flow hyperparameters via config (sigma_min, diffusion_steps)
# [x] TODO 10: Keep existing infrastructure (Memory, pretrain_disc, update_critic, etc.)
# =============================================================================
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types
from types import SimpleNamespace

# Optional: safety_gymnasium for online evaluation (not needed for offline training)
try:
    import safety_gymnasium
    SAFETY_GYM_AVAILABLE = True
except ImportError:
    SAFETY_GYM_AVAILABLE = False
    print("Warning: safety_gymnasium not installed. Online evaluation disabled.")

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# Existing imports
from dataset.memory import Memory
from utils.utils import eval_mode, average_dicts, get_concat_samples, soft_update, hard_update, \
    load_dataset, merge_dataset

# DSRL dataset loading (alternative to hdf5 files)
try:
    import dsrl
    import dsrl.offline_safety_gymnasium  # registers envs
    from dsrl_dataset import (
        to_d4rl_format,
        get_neg_and_union_data_2,
        get_normalized_data
    )
    DSRL_AVAILABLE = True
except ImportError:
    DSRL_AVAILABLE = False
    print("Note: DSRL library not available. Will use hdf5 files for data loading.")

# TODO 2 & 3: Import models
from diffusion_SDE.model import ScoreNet
from agent.sac_models import DoubleQCritic, SingleV, Discriminator

torch.set_num_threads(2)


# =============================================================================
# TODO 1: OT Flow Matching Helper Functions
# =============================================================================

def psi_t_ot(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma_min: float = 0.01) -> torch.Tensor:
    """
    Optimal Transport linear interpolation (per FM paper Eq.20-22).
    
    Args:
        x0: Source samples (noise), shape [B, D]
        x1: Target samples (actions), shape [B, D]
        t: Time values in [0, 1], shape [B] or [B, 1]
        sigma_min: Minimum sigma for OT path (default: 0.01)
    
    Returns:
        x_t: Interpolated samples at time t, shape [B, D]
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # [B, 1]
    
    one_minus_sigma_min = 1.0 - sigma_min
    # sigma_t = 1 - (1 - sigma_min) * t
    sigma_t = 1.0 - one_minus_sigma_min * t  # [B, 1]
    
    # x_t = sigma_t * x0 + t * x1
    return sigma_t * x0 + t * x1


def u_t_ot(x_t: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma_min: float = 0.01) -> torch.Tensor:
    """
    OT vector field at x_t: u_t(x|x1) = (x1 - (1-σ_min)*x) / (1 - (1-σ_min)*t)
    
    Args:
        x_t: Current position, shape [B, D]
        x1: Target samples (actions), shape [B, D]
        t: Time values in [0, 1], shape [B] or [B, 1]
        sigma_min: Minimum sigma for OT path (default: 0.01)
    
    Returns:
        u_t: Target vector field, shape [B, D]
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # [B, 1]
    
    one_minus_sigma_min = 1.0 - sigma_min
    denom = 1.0 - one_minus_sigma_min * t  # [B, 1]
    
    # Clamp denominator to avoid division by zero near t=1
    denom = torch.clamp(denom, min=1e-6)
    
    return (x1 - one_minus_sigma_min * x_t) / denom


# =============================================================================
# TODO 4: FlowAgent Builder (replaces make_agent)
# =============================================================================

def make_flow_agent(env, args):
    """
    Build agent with flow matching policy instead of Gaussian actor.
    Returns a SimpleNamespace with all components.
    """
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    
    device = torch.device(args.device)
    
    # Ensure args has required attributes for model compatibility
    if not hasattr(args, 'gamma'):
        args.gamma = 0.99
    if not hasattr(args, 'method'):
        args.method = SimpleNamespace(loss='v0', tanh=False)
    elif not hasattr(args.method, 'tanh'):
        args.method.tanh = False
    
    # Flow hyperparameters (TODO 9: exposed via config)
    sigma_min = getattr(args, 'sigma_min', 0.01)
    diffusion_steps = getattr(args, 'diffusion_steps', 15)
    hidden_dim = getattr(args.agent, 'hidden_dim', 256)
    hidden_depth = getattr(args.agent, 'hidden_depth', 2)
    
    # TODO 2: Flow model (ScoreNet with marginal_prob_std=None for flow mode)
    # input_dim = obs_dim + action_dim (condition on obs, output action)
    # output_dim = action_dim (single-step flow)
    # Create a simple args namespace for ScoreNet that has required device attribute
    flow_args = SimpleNamespace(device=str(device))
    flow_model = ScoreNet(
        input_dim=obs_dim + action_dim,
        output_dim=action_dim,
        marginal_prob_std=None,  # Flow Matching mode (not diffusion)
        embed_dim=32,
        args=flow_args
    ).to(device)
    
    # TODO 3: Critic (DoubleQ)
    critic = DoubleQCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        args=args
    ).to(device)
    
    critic_target = DoubleQCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        args=args
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.requires_grad_(False)
    
    # Value network (if sep_V is enabled)
    value = SingleV(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        args=args
    ).to(device)
    
    # Discriminators (mix and bad)
    reward_factor = getattr(args.agent, 'reward_factor', 1.0)
    
    disc_mix = Discriminator(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        reward_factor=reward_factor
    ).to(device)
    
    disc_bad = Discriminator(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        reward_factor=reward_factor
    ).to(device)
    
    # Optimizers
    lr = getattr(args.agent, 'critic_lr', 3e-4)
    flow_lr = getattr(args.agent, 'actor_lr', 3e-4)
    disc_lr = getattr(args.agent, 'disc_lr', 3e-4)
    value_lr = getattr(args.agent, 'value_lr', 3e-4)
    
    flow_optimizer = torch.optim.Adam(flow_model.parameters(), lr=flow_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=value_lr)
    disc_mix_optimizer = torch.optim.Adam(disc_mix.parameters(), lr=disc_lr)
    disc_bad_optimizer = torch.optim.Adam(disc_bad.parameters(), lr=disc_lr)
    
    # Build agent namespace
    agent = SimpleNamespace(
        # Models
        flow_model=flow_model,
        critic=critic,
        critic_target=critic_target,
        value=value,
        disc_mix=disc_mix,
        disc_bad=disc_bad,
        
        # Optimizers
        flow_optimizer=flow_optimizer,
        critic_optimizer=critic_optimizer,
        value_optimizer=value_optimizer,
        disc_mix_optimizer=disc_mix_optimizer,
        disc_bad_optimizer=disc_bad_optimizer,
        
        # Config
        device=device,
        args=args,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=action_range,
        
        # Hyperparameters
        gamma=getattr(args, 'gamma', 0.99),
        batch_size=getattr(args.train, 'batch', 256),
        max_v=1.0 / (1.0 - getattr(args, 'gamma', 0.99)),
        reward_factor=reward_factor,
        sigma_min=sigma_min,
        diffusion_steps=diffusion_steps,
        
        # Target network update
        critic_tau=getattr(args.agent, 'critic_tau', 0.005),
        critic_target_update_frequency=getattr(args.agent, 'critic_target_update_frequency', 1),
        
        # For compatibility with existing code
        critic_net=critic,
        critic_target_net=critic_target,
        first_log=True,
    )
    
    print(f'--> Flow agent created: obs_dim={obs_dim}, action_dim={action_dim}')
    print(f'--> Flow config: sigma_min={sigma_min}, diffusion_steps={diffusion_steps}')
    
    return agent


# =============================================================================
# TODO 6: Evaluation with Flow Model
# =============================================================================

def eval_parallel_flow(agent, env, num_episodes, args):
    """
    Evaluate flow policy using ODE integration for action selection.
    """
    total_returns = []
    total_costs = []
    total_steps = 0
    
    diffusion_steps = agent.diffusion_steps
    
    while len(total_returns) < num_episodes:
        state, _ = env.reset()
        total_return = np.array([0.0 for _ in range(args.eval.n_envs)])
        total_cost = np.array([0.0 for _ in range(args.eval.n_envs)])
        
        with eval_mode(agent.flow_model):
            while True:
                # Use flow model for action selection (RK4 ODE integration)
                actions = agent.flow_model.select_actions(state, diffusion_steps=diffusion_steps)
                
                # Handle single vs batch actions
                if isinstance(actions, list):
                    actions = np.array(actions)
                elif isinstance(actions, np.ndarray) and actions.ndim == 1:
                    actions = actions.reshape(1, -1)
                
                # Clip actions to valid range
                actions = np.clip(actions, agent.action_range[0], agent.action_range[1])
                
                next_state, reward, cost, done, trunc, info = env.step(actions)
                state = next_state
                total_return += reward
                total_cost += cost
                total_steps += len(cost)
                
                if np.max(done) or np.max(trunc):
                    for idx in range(args.eval.n_envs):
                        total_returns.append(total_return[idx])
                        total_costs.append(total_cost[idx])
                    break
    
    return total_returns, total_costs, np.sum(total_costs) / total_steps


# =============================================================================
# TODO 5: Weighted Flow Matching Actor Update
# =============================================================================

def update_actor_flow_matching(self, obs, exp_action, next_obs, done, env_cost, is_constrained, step):
    """
    Update flow policy using weighted flow matching loss.
    
    Replaces update_actor_BC with:
    1. Same weight computation (Q-V advantage × disc_weight)
    2. OT flow matching loss instead of log-likelihood
    """
    args = self.args
    sigma_min = self.sigma_min
    eps = 1e-5  # Avoid t=0 or t=1
    
    # Compute weights (same as BC)
    with torch.no_grad():
        Qs = self.critic_target(obs, exp_action).clip(min=-self.max_v, max=self.max_v)
        if getattr(args.train, 'sep_V', True):
            current_V = self.value(obs).clip(min=-self.max_v, max=self.max_v)
        else:
            current_V = get_target_V(self, obs).clip(min=-self.max_v, max=self.max_v)
        
        # Discriminator weight
        cur_weight = get_weight(self, obs) + self.gamma * get_weight(self, next_obs)
        disc_weight = (1.0 / (cur_weight ** 2)).clip(min=0, max=100)
        
        # Q-V advantage weight
        Q_adv = (Qs - current_V).clip(max=10)
        adv_weight = torch.exp(Q_adv - Q_adv.max())
        adv_weight = adv_weight / adv_weight.mean()
        
        # Combined weight
        weight = (adv_weight * disc_weight).clip(max=100)
        weight = weight.squeeze(-1)  # [B]
    
    batch_size = obs.shape[0]
    device = obs.device
    
    # Sample noise x0 ~ N(0, I)
    x0 = torch.randn_like(exp_action)  # [B, action_dim]
    
    # Sample time t ~ Uniform(eps, 1-eps)
    t = torch.rand(batch_size, device=device) * (1.0 - 2 * eps) + eps  # [B]
    
    # Build OT interpolation x_t and target vector field u_t
    x_t = psi_t_ot(x0, exp_action, t, sigma_min)  # [B, action_dim]
    u_t = u_t_ot(x_t, exp_action, t, sigma_min)   # [B, action_dim]
    
    # Set condition for flow model and predict velocity
    self.flow_model.condition = obs
    v_theta = self.flow_model(x_t, t)  # [B, action_dim]
    self.flow_model.condition = None
    
    # Per-sample squared error
    per_sample_err = torch.sum((v_theta - u_t) ** 2, dim=-1)  # [B]
    
    # Weighted flow matching loss
    flow_loss = (weight.detach() * per_sample_err).mean()
    
    # Optimize
    self.flow_optimizer.zero_grad()
    flow_loss.backward()
    
    # Gradient clipping (optional, for stability)
    max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
    torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_grad_norm)
    
    self.flow_optimizer.step()
    
    # Logging
    loss_dict = {}
    if step % args.env.eval_interval == 0:
        with torch.no_grad():
            loss_dict = {
                'flow_loss/loss': round(flow_loss.item(), 5),
                'flow_loss/C_weight': round(weight[is_constrained.squeeze(-1)].mean().item(), 3),
                'flow_loss/U_weight': round(weight[~is_constrained.squeeze(-1)].mean().item(), 3),
                'flow_loss/mse': round(per_sample_err.mean().item(), 5),
            }
    
    return loss_dict


# =============================================================================
# Helper functions for agent (get_weight, get_target_V)
# =============================================================================

def get_weight(agent, obs, is_bad=None):
    """Get discriminator-based importance weight."""
    with torch.no_grad():
        mix_d = agent.disc_mix(obs)
        bad_d = agent.disc_bad(obs)
        
        # Weight based on discriminator outputs
        # Higher weight for samples that look "good" (high mix_d, low bad_d)
        weight = torch.log(mix_d / (1 - mix_d + 1e-8)) - torch.log(bad_d / (1 - bad_d + 1e-8))
    
    return weight


def get_target_V(agent, obs):
    """Get target value estimate."""
    with torch.no_grad():
        # Sample random actions and take min Q
        batch_size = obs.shape[0]
        random_actions = torch.rand(batch_size, agent.action_dim, device=agent.device) * 2 - 1
        V = agent.critic_target(obs, random_actions)
    return V


# =============================================================================
# Discriminator Training (same as original)
# =============================================================================

def pretrain_disc(self, mix_buffer, bad_buffer, disc_path, file_name, total_step=50000):
    """Pretrain discriminators to distinguish mix vs bad samples."""
    os.makedirs(disc_path, exist_ok=True)
    print(disc_path)
    print(file_name)
    
    mix_path = f'{disc_path}/mix_{file_name}'
    bad_path = f'{disc_path}/bad_{file_name}'
    
    # Only load if BOTH files exist, otherwise train both
    if os.path.isfile(mix_path) and os.path.isfile(bad_path):
        print('load mix disc from ', mix_path)
        self.disc_mix.load_state_dict(torch.load(mix_path, map_location=self.device))
        print('[mix disc]', update_disc(self, mix_buffer, bad_buffer, mix_path, 0))
        print('load bad disc from ', bad_path)
        self.disc_bad.load_state_dict(torch.load(bad_path, map_location=self.device))
        print('[bad disc]', update_disc(self, mix_buffer, bad_buffer, bad_path, 0))
    else:
        print('Pretrain disc mix')
        print(mix_path)
        for itr in range(total_step + 1):
            info = update_disc(self, mix_buffer, bad_buffer, mix_path, itr)
            if itr % 1000 == 0:
                print(f'{itr}/{total_step} dif = {abs(info["disc/bad"] - info["disc/mix"]):.3f}', info)
                if abs(info['disc/bad'] - info['disc/mix']) > 0.2:
                    break
        torch.save(self.disc_mix.state_dict(), mix_path)
        print('save disc mix to ', mix_path)
        print('-' * 20)
        
        print('Pretrain disc bad')
        print(bad_path)
        for itr in range(total_step + 1):
            info = update_disc(self, mix_buffer, bad_buffer, bad_path, itr)
            if itr % 1000 == 0:
                print(f'{itr}/{total_step} dif = {abs(info["disc/bad"] - info["disc/mix"]):.3f}', info)
                if abs(info['disc/bad'] - info['disc/mix']) > 0.2:
                    break
        torch.save(self.disc_bad.state_dict(), bad_path)
        print('save disc bad to ', bad_path)
        print('-' * 20)


def update_disc(agent, mix_buffer, bad_buffer, file_path, step):
    """Update discriminator for one step."""
    if 'mix' in file_path.split('/')[-1]:
        disc = agent.disc_mix
        disc_optimizer = agent.disc_mix_optimizer
    else:
        disc = agent.disc_bad
        disc_optimizer = agent.disc_bad_optimizer
    
    mix_batch = mix_buffer.get_samples(1024, agent.device, noise=0.0)
    bad_batch = bad_buffer.get_samples(1024, agent.device, noise=0.0)
    mix_obs, mix_next_obs, mix_action, _, _, _, mix_is_constrained = mix_batch
    bad_obs, bad_next_obs, bad_action, _, _, _, bad_is_constrained = bad_batch
    
    mix_d = disc(mix_obs)
    bad_d = disc(bad_obs)
    
    if 'bad' in file_path.split('/')[-1]:
        loss_bad = -torch.log(bad_d).mean()
        loss_mix = -torch.log(1 - mix_d).mean()
    else:
        loss_bad = -torch.log(1 - bad_d).mean()
        loss_mix = -torch.log(mix_d).mean()
    
    loss = loss_mix + loss_bad
    
    disc_optimizer.zero_grad()
    loss.backward()
    disc_optimizer.step()
    
    info = {}
    if step % 1000 == 0:
        with torch.no_grad():
            mix_batch = mix_buffer.get_samples(5000, agent.device)
            bad_batch = bad_buffer.get_samples(5000, agent.device)
            disc.eval()
            
            mix_obs, mix_next_obs, mix_action, _, _, _, mix_is_constrained = mix_batch
            bad_obs, bad_next_obs, bad_action, _, _, _, bad_is_constrained = bad_batch
            mix_d = disc(mix_obs)
            bad_d = disc(bad_obs)
            
            disc.train()
            
            info = {
                'disc/mix': round(mix_d.mean().item(), 3),
                'disc/bad': round(bad_d.mean().item(), 3),
            }
    return info


# =============================================================================
# Value Update (same as original)
# =============================================================================

def update_value(agent, obs, action, next_obs, step):
    """Update value network with IQL-style expectile loss."""
    def iql_loss(pred, target, expectile=0.7):
        err = target - pred
        weight = torch.abs(expectile - (err < 0).float())
        return (weight * torch.square(err)).mean()
    
    args = agent.args
    cur_V = agent.value(obs)
    with torch.no_grad():
        cur_Q = agent.critic_target(obs, action)
    
    loss = iql_loss(cur_V, cur_Q, 0.7)
    
    agent.value_optimizer.zero_grad()
    loss.backward()
    agent.value_optimizer.step()
    
    infos = {}
    if step % args.env.eval_interval == 0:
        infos = {
            'value/value_loss': round(loss.item(), 3),
            'value/V': round(cur_V.mean().item(), 3),
            'value/Q': round(cur_Q.mean().item(), 3),
        }
    
    return infos


# =============================================================================
# Critic Update (same as original)
# =============================================================================

def update_critic(agent, mix_batch, bad_batch, step):
    """Update critic with IQ-Learn style loss."""
    args = agent.args
    batch = get_concat_samples(mix_batch, bad_batch, args)
    obs, next_obs, action, reward, cost, done, is_constrained, is_bad = batch
    
    infos = {}
    
    # Update V
    if getattr(args.train, 'sep_V', True):
        infos = update_value(agent, obs, action, next_obs, step)
    
    # Update Q
    with torch.no_grad():
        if getattr(args.train, 'sep_V', True):
            next_V = agent.value(next_obs).clip(min=-agent.max_v, max=agent.max_v)
        else:
            next_V = get_target_V(agent, next_obs).clip(min=-agent.max_v, max=agent.max_v)
    
    current_Q1, current_Q2 = agent.critic(obs, action, both=True)
    q1_loss, loss_dict1 = critic_loss(agent, agent.critic.Q1, current_Q1, next_V, batch, step)
    q2_loss, loss_dict2 = critic_loss(agent, agent.critic.Q2, current_Q2, next_V, batch, step)
    c_loss = 0.5 * (q1_loss + q2_loss)
    loss_dict = average_dicts(loss_dict1, loss_dict2)
    
    infos.update(loss_dict)
    
    agent.critic_optimizer.zero_grad()
    c_loss.backward()
    agent.critic_optimizer.step()
    
    return infos


def critic_loss(agent, critic_Q, current_Q, next_v, batch, step):
    """Compute critic loss with IQ-Learn formulation."""
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, env_cost, done, is_constrained, is_bad = batch
    
    loss_dict = {}
    
    y = (1 - done) * gamma * next_v
    with torch.no_grad():
        cur_weight = get_weight(agent, obs, is_bad)
        next_weight = get_weight(agent, next_obs, is_bad)
        weight = (cur_weight + gamma * next_weight).clip(min=-agent.reward_factor, max=agent.reward_factor)
    
    reward = (current_Q - y)
    if not getattr(args.agent, 'pen_bad', False):
        reward = reward[~is_bad]
        weight = weight[~is_bad]
    else:
        if agent.first_log:
            print('[critic] penalize bad samples')
    
    reward_loss = -(weight * reward).mean()
    
    y = (1 - done) * gamma * next_v
    all_reward = (current_Q - y)
    chi2_loss = 0.5 * (all_reward ** 2).mean()
    
    loss = reward_loss + chi2_loss
    
    if getattr(args.method, 'loss', 'v0') == "v0":
        if agent.first_log:
            print('[critic] v0 value loss')
        if getattr(args.train, 'sep_V', True):
            v0 = agent.value(obs).mean()
        else:
            v0 = get_target_V(agent, obs).mean()
        v0_loss = (1 - gamma) * v0
        loss += v0_loss
    elif getattr(args.method, 'loss', 'v0') == "no":
        if agent.first_log:
            print('[critic] no value loss')
    
    return loss, loss_dict


# =============================================================================
# Main Update Loop
# =============================================================================

def iq_update(agent, mix_buffer, bad_buffer, step):
    """Main update step: critic + flow policy."""
    mix_batch = mix_buffer.get_samples(agent.batch_size, agent.device)
    bad_batch = bad_buffer.get_samples(agent.batch_size, agent.device)
    
    # Update critic
    info = update_critic(agent, mix_batch, bad_batch, step)
    
    # Update flow policy
    mix_batch = mix_buffer.get_samples(agent.batch_size, agent.device)
    obs, next_obs, action, env_reward, env_cost, done, is_constrained = mix_batch
    info.update(update_actor_flow_matching(agent, obs, action, next_obs, done, env_cost, is_constrained, step))
    
    # Target network update
    if step % agent.critic_target_update_frequency == 0:
        if getattr(agent.args.train, 'soft_update', True):
            soft_update(agent.critic_net, agent.critic_target_net, agent.critic_tau)
        else:
            hard_update(agent.critic_net, agent.critic_target_net)
    
    if agent.first_log:
        agent.first_log = False
    
    return info


# =============================================================================
# TODO 8: Checkpoint Saving
# =============================================================================

def save_checkpoint(agent, path, step):
    """Save all model checkpoints."""
    checkpoint = {
        'step': step,
        'flow_model': agent.flow_model.state_dict(),
        'critic': agent.critic.state_dict(),
        'critic_target': agent.critic_target.state_dict(),
        'value': agent.value.state_dict(),
        'disc_mix': agent.disc_mix.state_dict(),
        'disc_bad': agent.disc_bad.state_dict(),
        'flow_optimizer': agent.flow_optimizer.state_dict(),
        'critic_optimizer': agent.critic_optimizer.state_dict(),
        'value_optimizer': agent.value_optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')


def load_checkpoint(agent, path):
    """Load model checkpoints."""
    checkpoint = torch.load(path, map_location=agent.device)
    agent.flow_model.load_state_dict(checkpoint['flow_model'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.critic_target.load_state_dict(checkpoint['critic_target'])
    agent.value.load_state_dict(checkpoint['value'])
    agent.disc_mix.load_state_dict(checkpoint['disc_mix'])
    agent.disc_bad.load_state_dict(checkpoint['disc_bad'])
    agent.flow_optimizer.load_state_dict(checkpoint['flow_optimizer'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    agent.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
    print(f'Checkpoint loaded from {path}')
    return checkpoint.get('step', 0)


# =============================================================================
# Utility
# =============================================================================

def top_10_percent_numpy(lst):
    """Get top 10% values for CVaR computation."""
    np_array = np.array(lst)
    sorted_array = np.sort(np_array)[::-1]
    top_10_count = max(1, int(len(sorted_array) * 0.1))
    return sorted_array[:top_10_count]


def load_dsrl_dataset(env, args):
    """
    Load dataset using DSRL library directly from the environment.
    Returns (mix_dataset, bad_dataset) in the format expected by Memory.load_from_data().
    """
    if not DSRL_AVAILABLE:
        raise ImportError("DSRL library not available. Please install dsrl package.")
    
    print(f"\nLoading DSRL dataset from environment...", flush=True)
    
    # Get raw data from environment
    raw_data = env.get_dataset()
    print(f"Raw data loaded. Keys: {list(raw_data.keys())}", flush=True)
    
    # Compute trajectory lengths
    dones_idx = np.where((raw_data["terminals"] == 1) | (raw_data["timeouts"] == 1))[0]
    n_trajs = len(dones_idx)
    print(f"Found {n_trajs} trajectories", flush=True)
    
    if n_trajs == 0:
        raise ValueError("No trajectories found in dataset")
    
    traj_lengths = []
    start = 0
    for end_idx in dones_idx:
        traj_lengths.append(end_idx - start + 1)
        start = end_idx + 1
    
    max_traj_len = max(traj_lengths) if traj_lengths else 1000
    print(f"Trajectory length: mean={np.mean(traj_lengths):.1f}, max={max_traj_len}", flush=True)
    
    # Convert to trajectory format - using vectorized approach for speed
    print("Converting to trajectory format (this may take a moment)...", flush=True)
    
    # Faster approach: directly slice data into trajectories
    keys = ["observations", "actions", "rewards", "costs", "terminals", "timeouts"]
    d4rl_data = {}
    
    # Pre-allocate arrays for efficiency
    for k in keys:
        if k in raw_data:
            shape = (n_trajs, max_traj_len) + raw_data[k].shape[1:]
            d4rl_data[k] = np.zeros(shape, dtype=raw_data[k].dtype)
    
    # Fill in trajectory data
    start = 0
    for i, end_idx in enumerate(dones_idx):
        end = end_idx + 1
        traj_len = end - start
        for k in keys:
            if k in raw_data:
                d4rl_data[k][i, :traj_len] = raw_data[k][start:end]
                # Pad with last value if needed
                if traj_len < max_traj_len:
                    d4rl_data[k][i, traj_len:] = raw_data[k][end-1]
        start = end
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{n_trajs} trajectories...", flush=True)
    
    print(f"D4RL data shape: {d4rl_data['observations'].shape}", flush=True)
    
    # Dataset config for splitting
    dataset_config = {
        "num_negative_trajectories": getattr(args.expert, 'n_bad', 50),
        "num_union_trajectories": -1,  # Use all remaining
        "non_pref_noise": getattr(args, 'non_pref_noise', 0.0),
    }
    
    # Split into negative and union sets
    print("Splitting into negative and union sets...", flush=True)
    neg_data, union_data = get_neg_and_union_data_2(d4rl_data, dataset_config)
    
    # Convert trajectory format to flat format for Memory class
    def traj_to_flat(data, name="data"):
        """Convert trajectory format [N, T, D] to flat format [N*T, D]."""
        print(f"  Flattening {name}...", flush=True)
        n_traj, traj_len = data['observations'].shape[:2]
        
        # Flatten all arrays
        flat_data = {
            'states': data['observations'].reshape(-1, data['observations'].shape[-1]),
            'actions': data['actions'].reshape(-1, data['actions'].shape[-1]),
            'rewards': data['rewards'].reshape(-1),
            'costs': data['costs'].reshape(-1),
            'dones': data['terminals'].reshape(-1),
        }
        
        # Create next_states by shifting observations
        next_obs = np.zeros_like(data['observations'])
        next_obs[:, :-1] = data['observations'][:, 1:]
        next_obs[:, -1] = data['observations'][:, -1]  # Last step copies itself
        flat_data['next_states'] = next_obs.reshape(-1, data['observations'].shape[-1])
        
        return flat_data
    
    # Convert both datasets
    print("Converting to flat format...", flush=True)
    flat_neg = traj_to_flat(neg_data, "neg_data")
    flat_union = traj_to_flat(union_data, "union_data")
    
    # Add is_constrained flag (union=True for "good" data, neg=False for "bad" data)
    flat_union['is_constrained'] = np.ones(len(flat_union['states']), dtype=bool)
    flat_neg['is_constrained'] = np.zeros(len(flat_neg['states']), dtype=bool)
    
    print(f"Flat union size: {len(flat_union['states'])}", flush=True)
    print(f"Flat negative size: {len(flat_neg['states'])}", flush=True)
    
    return flat_union, flat_neg


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def namespace_to_dict(ns):
    """Convert SimpleNamespace back to dict recursively (for printing)."""
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return dict_to_namespace(cfg)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Weighted Flow Matching Training')
    
    # Get script directory for default config path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'conf/config.yaml')
    
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to config file')
    parser.add_argument('--env_name', type=str, default=None,
                        help='Environment name (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda:0, cpu)')
    parser.add_argument('--learn_steps', type=int, default=None,
                        help='Training steps (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for logs and checkpoints')
    return parser.parse_args()


def get_args():
    """Load config and apply CLI overrides."""
    cli_args = parse_args()
    
    # Load config from YAML
    args = load_config(cli_args.config)
    
    # Apply CLI overrides
    if cli_args.env_name is not None:
        args.env.name = cli_args.env_name
    if cli_args.seed is not None:
        args.seed = cli_args.seed
    if cli_args.learn_steps is not None:
        args.env.learn_steps = cli_args.learn_steps
    
    # Set device
    if cli_args.device is not None:
        args.device = cli_args.device
    else:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if cli_args.output_dir is not None:
        args.output_dir = cli_args.output_dir
    else:
        # Default output directory based on env name and timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(script_dir, 'outputs', f'{args.env.name}_{timestamp}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print config
    print(yaml.dump(namespace_to_dict(args), default_flow_style=False))
    
    return args, script_dir


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    args, script_dir = get_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and getattr(args, 'cuda_deterministic', False):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Determine if using DSRL offline environment or Safety Gymnasium
    env_name = args.env.name
    is_offline_env = "Offline" in env_name and DSRL_AVAILABLE
    
    print(f'env = {env_name} with num_envs = {args.eval.n_envs}')
    
    if is_offline_env:
        # Use DSRL offline environment (gymnasium-based)
        import gymnasium as gym
        # For evaluation, we need to convert offline env name to online equivalent
        # e.g., OfflinePointGoal1Gymnasium-v0 -> SafetyPointGoal1-v0
        online_env_name = env_name.replace("Offline", "Safety").replace("Gymnasium-v0", "-v0").replace("Gymnasium-v1", "-v1")
        print(f'Using DSRL offline environment, online eval with: {online_env_name}')
        
        # Create a single DSRL env for dataset loading
        single_dsrl_env = gym.make(env_name)
        single_dsrl_env.reset(seed=args.seed)
        
        # Create vectorized env for evaluation (using safety_gymnasium for online eval)
        try:
            eval_env = safety_gymnasium.vector.make(env_id=online_env_name, num_envs=args.eval.n_envs)
            eval_env.reset(seed=[i for i in range(args.eval.n_envs)])
        except Exception as e:
            print(f"Warning: Could not create vectorized eval env: {e}")
            print("Using single env for evaluation")
            # Fallback to single env wrapped as "vectorized"
            eval_env = gym.make(online_env_name)
            eval_env.reset(seed=args.seed)
    else:
        # Use Safety Gymnasium directly
        eval_env = safety_gymnasium.vector.make(env_id=env_name, num_envs=args.eval.n_envs)
        eval_env.reset(seed=[i for i in range(args.eval.n_envs)])
        single_dsrl_env = None
    
    # TODO 4: Create flow agent (replaces make_agent)
    agent = make_flow_agent(eval_env, args)
    
    # TODO 7: TensorBoard logging
    log_dir = os.path.join(args.output_dir, 'tb_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f'--> TensorBoard logs: {log_dir}')
    
    # Checkpoint directory
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    save_freq = getattr(args, 'save_freq', 20000)
    
    # Load datasets - try DSRL first, fall back to hdf5
    use_dsrl = getattr(args, 'use_dsrl', True) and DSRL_AVAILABLE and is_offline_env
    c_data_path = os.path.join(script_dir, f'experts/{args.env.name}/collect_C/full_data.hdf5')
    
    if use_dsrl and single_dsrl_env is not None:
        print("\n--> Loading data from DSRL library...")
        try:
            mix_dataset, bad_dataset = load_dsrl_dataset(single_dsrl_env, args)
            single_dsrl_env.close()
            
            mix_memory_replay = Memory(1, args.seed)
            mix_memory_replay.load_from_data(mix_dataset)
            print(f'--> mix memory size: {mix_memory_replay.size()}')
            
            bad_memory_replay = Memory(1, args.seed)
            bad_memory_replay.load_from_data(bad_dataset)
            print(f'--> bad memory size: {bad_memory_replay.size()}')
            
        except Exception as e:
            import traceback
            print(f"DSRL loading failed: {e}")
            traceback.print_exc()
            print("Falling back to hdf5 files...")
            use_dsrl = False
    
    if not use_dsrl:
        # Original hdf5 loading
        if not os.path.isfile(c_data_path):
            print(f"\n{'='*60}")
            print(f"ERROR: Expert data not found!")
            print(f"Expected path: {c_data_path}")
            print(f"\nPlease ensure you have collected/downloaded expert data for {args.env.name}")
            print(f"Or install the DSRL library: pip install dsrl")
            print(f"Expected directory structure:")
            print(f"  experts/{args.env.name}/collect_C/full_data.hdf5")
            print(f"  experts/{args.env.name}/collect_U/full_data.hdf5")
            print(f"  experts/{args.env.name}/collect_U/undesirable_data.hdf5")
            print(f"{'='*60}\n")
            raise FileNotFoundError(f"Expert data not found: {c_data_path}")
        
        c_dataset = load_dataset(c_data_path, num_trajectories=args.expert.n_mix_good, seed=0)
        u_data_path = os.path.join(script_dir, f'experts/{args.env.name}/collect_U/full_data.hdf5')
        u_dataset = load_dataset(u_data_path, num_trajectories=args.expert.n_mix_bad, seed=1)
        dataset = merge_dataset(c_dataset=c_dataset, u_dataset=u_dataset)
        del c_dataset
        del u_dataset
        
        mix_memory_replay = Memory(1, args.seed)
        mix_memory_replay.load_from_data(dataset)
        print(f'--> mix memory size: {mix_memory_replay.size()}')
        
        vio_data_path = os.path.join(script_dir, f'experts/{args.env.name}/collect_U/undesirable_data.hdf5')
        vio_dataset = load_dataset(vio_data_path, num_trajectories=args.expert.n_bad, seed=2)
        vio_dataset['is_constrained'] = np.zeros_like(vio_dataset['costs'], dtype=bool)
        
        bad_memory_replay = Memory(1, args.seed)
        bad_memory_replay.load_from_data(vio_dataset)
        print(f'--> bad memory size: {bad_memory_replay.size()}')
    
    # Pretrain discriminators
    disc_path = os.path.join(script_dir, f'experts/{args.env.name}/Disc')
    pretrain_disc(agent, mix_memory_replay, bad_memory_replay, disc_path,
                  f'state_disc_C({args.expert.n_mix_good})_U({args.expert.n_mix_bad})_B({args.expert.n_bad})')
    agent.disc_mix.eval()
    agent.disc_bad.eval()
    
    # Training loop
    LEARN_STEPS = int(args.env.learn_steps)
    
    print(f'\n{"="*60}')
    print(f'Starting training for {LEARN_STEPS} steps')
    print(f'{"="*60}\n')
    
    for learn_steps in range(LEARN_STEPS + 1):
        info = {}
        
        # Evaluation
        if learn_steps % args.env.eval_interval == 0:
            eval_returns, eval_costs, cost_rate = eval_parallel_flow(
                agent, eval_env, num_episodes=args.eval.eps, args=args
            )
            
            returns = np.mean(eval_returns)
            cvar_cost = np.mean(top_10_percent_numpy(eval_costs))
            costs = np.mean(eval_costs)
            
            print(f'[Eval {learn_steps/LEARN_STEPS*100:.1f}%] Returns: {returns:.2f}, '
                  f'Costs: {costs:.2f}, CVaR: {cvar_cost:.2f}, Cost rate: {cost_rate:.3f}')
            
            # TensorBoard logging
            writer.add_scalar('eval/returns', returns, learn_steps)
            writer.add_scalar('eval/costs', costs, learn_steps)
            writer.add_scalar('eval/cvar_cost', cvar_cost, learn_steps)
            writer.add_scalar('eval/cost_rate', cost_rate, learn_steps)
        
        # Progress logging
        if learn_steps % 1000 == 0:
            print(f'[{learn_steps}/{LEARN_STEPS}]')
        
        # Update
        info.update(iq_update(agent, mix_memory_replay, bad_memory_replay, learn_steps))
        
        # Log training metrics to TensorBoard
        if learn_steps % args.env.eval_interval == 0:
            for key, value in info.items():
                writer.add_scalar(f'train/{key}', value, learn_steps)
        
        # Save checkpoint
        if learn_steps > 0 and learn_steps % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{learn_steps}.pt')
            save_checkpoint(agent, ckpt_path, learn_steps)
    
    # Final checkpoint
    final_ckpt_path = os.path.join(ckpt_dir, 'checkpoint_final.pt')
    save_checkpoint(agent, final_ckpt_path, LEARN_STEPS)
    
    writer.close()
    print('\nTraining completed!')


if __name__ == "__main__":
    main()

