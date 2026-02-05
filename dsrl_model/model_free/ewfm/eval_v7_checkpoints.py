#!/usr/bin/env python3
"""
Evaluate flow model checkpoints from ipltwin_V7.py
Usage: python eval_v7_checkpoints.py --ckpt_dir /path/to/logs --num_episodes 10
"""

import os
import sys
import json
import glob
import argparse
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import gymnasium as gym
import dsrl
import dsrl.offline_safety_gymnasium

from diffusion_SDE.model import ScoreNet

EP = 1e-6


def find_checkpoints(ckpt_dir):
    """Find all flow checkpoints, sorted by iteration."""
    pattern = os.path.join(ckpt_dir, "torch_save", "flow*.pt")
    ckpts = []
    for f in glob.glob(pattern):
        try:
            itr = int(os.path.basename(f).split('_')[-1].replace('.pt', ''))
            ckpts.append((itr, f))
        except ValueError:
            continue
    return sorted(ckpts, key=lambda x: x[0])


@torch.no_grad()
def evaluate_episode(env, flow_model, device, mu_obs=None, std_obs=None, use_prev_action=False):
    """Run single episode, return (reward, cost, length)."""
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    
    total_reward, total_cost, length = 0.0, 0.0, 0
    done = False
    
    # Initialize prev_action tracking
    act_dim = flow_model.act_dim
    prev_action = torch.zeros(1, act_dim, device=device) if use_prev_action else None
    
    while not done:
        if mu_obs is not None:
            obs_norm = (obs - mu_obs) / (std_obs + EP)
        else:
            obs_norm = obs
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
        
        action = flow_model.select_actions(obs_t, prev_actions=prev_action)
        # Force flatten to 1D numpy array
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = action
        action_np = np.asarray(action_np).flatten()
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        
        # Update prev_action for next step
        if use_prev_action:
            prev_action = torch.as_tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        obs = np.asarray(next_obs, dtype=np.float32)
        total_reward += reward
        total_cost += info.get('cost', 0.0)
        length += 1
        done = terminated or truncated
    
    return total_reward, total_cost, length


def evaluate_checkpoint(ckpt_path, env, device, num_episodes, obs_dim, act_dim, args, mu_obs=None, std_obs=None, use_prev_action=False):
    """Evaluate single checkpoint."""
    # Get embed_dim and cond_dim from args (loaded from config)
    
    embed_dim = getattr(args, 'embed_dim', 512)
    cond_dim = getattr(args, 'cond_dim', 32)
    
    flow_model = ScoreNet(
        obs_dim + act_dim, act_dim, 
        marginal_prob_std=None, 
        embed_dim=embed_dim,
        cond_dim=cond_dim,
        use_prev_action=use_prev_action,
        args=args
    ).to(device)
    flow_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    flow_model.eval()
    
    rewards, costs, lengths = [], [], []
    for _ in range(num_episodes):
        r, c, l = evaluate_episode(env, flow_model, device, mu_obs, std_obs, use_prev_action=use_prev_action)
        rewards.append(r)
        costs.append(c)
        lengths.append(l)
    
    return {
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'length_mean': float(np.mean(lengths)),
        'rewards': rewards,
        'costs': costs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_prev_action", action="store_true", default=False, help="Condition flow on previous action")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load config and copy all training args
    config_path = os.path.join(args.ckpt_dir, "config.json")
    train_config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            train_config = json.load(f)
        for k, v in train_config.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
    
    # Set defaults if not in config
    if not hasattr(args, 'schedule'):
        args.schedule = "Linear"
    if not hasattr(args, 'normalize_observation'):
        args.normalize_observation = False
    # Override use_prev_action from config if not set via CLI
    if not args.use_prev_action:
        args.use_prev_action = train_config.get('use_prev_action', False)
    
    task = args.task or train_config.get('task')
    if not task:
        raise ValueError("Task not specified. Use --task")
    
    normalize = getattr(args, 'normalize_observation', False)
    
    print(f"Task: {task} | Device: {device} | Episodes: {args.num_episodes} | use_prev_action: {args.use_prev_action}")
    
    checkpoints = find_checkpoints(args.ckpt_dir)
    if not checkpoints:
        print(f"No checkpoints in {args.ckpt_dir}/torch_save/")
        return
    print(f"Found {len(checkpoints)} checkpoints")
    
    env = gym.make(task)
    env.reset(seed=42)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Normalization stats
    mu_obs, std_obs = None, None
    if normalize:
        from dsrl_dataset import get_dataset_in_d4rl_format, get_neg_and_union_data_2, get_normalized_data
        
        raw_data = env.get_dataset()
        dones_idx = np.where((raw_data["terminals"] == 1) | (raw_data["timeouts"] == 1))[0]
        traj_lengths = [dones_idx[0] + 1] + [dones_idx[i] - dones_idx[i-1] for i in range(1, len(dones_idx))]
        
        dataset_config = {
            "density": train_config.get("density", 1.0),
            "inpaint_ranges": train_config.get("inpaint_ranges", ((0.0, 1.0, 0.0, 0.5),)),
            "num_negative_trajectories": train_config.get("num_negative_trajectories", 50),
            "num_union_trajectories": train_config.get("num_union_trajectories", -1),
            "non_pref_noise": 0.0,
        }
        
        d4rl_data = get_dataset_in_d4rl_format(env, dataset_config, task, max(traj_lengths), num_folds=1)
        neg_data, union_data = get_neg_and_union_data_2(d4rl_data, dataset_config)
        _, _, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
        print("Loaded normalization stats")
    
    # Evaluate
    results = {}
    start = time.time()
    
    for i, (itr, path) in enumerate(checkpoints):
        res = evaluate_checkpoint(path, env, device, args.num_episodes, obs_dim, act_dim, args, mu_obs, std_obs, use_prev_action=args.use_prev_action)
        results[itr] = res
        print(f"[{i+1}/{len(checkpoints)}] iter={itr}: R={res['reward_mean']:.1f}±{res['reward_std']:.1f} C={res['cost_mean']:.1f} ±{res['cost_std']:.1f}")
    
    elapsed = time.time() - start
    best_itr = max(results, key=lambda k: results[k]['reward_mean'])
    best = results[best_itr]
    
    print(f"\nDone in {elapsed:.1f}s | Best: iter {best_itr} R={best['reward_mean']:.2f} C={best['cost_mean']:.2f}")
    
    # Save
    output = {
        'task': task, 'num_episodes': args.num_episodes,
        'best_iteration': best_itr, 'best_reward': best['reward_mean'], 'best_cost': best['cost_mean'],
        'results': results, 'eval_time': elapsed,
    }
    output_path = os.path.join(args.ckpt_dir, "eval_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {output_path}")
    
    env.close()


if __name__ == "__main__":
    main()
