#!/usr/bin/env python3
"""
Inference/Evaluation script for trained IPL+Flow models
Loads saved checkpoints and evaluates them on the environment

Place this file in: offline_mpc/dsrl_model/model_free/ewfm/inference.py
"""
import os
import sys
import argparse
import json
import functools
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
model_free_dir = current_dir.parent  # offline_mpc/dsrl_model/model_free/
dsrl_model_dir = model_free_dir.parent  # offline_mpc/dsrl_model/
offline_mpc_dir = dsrl_model_dir.parent  # offline_mpc/

sys.path.insert(0, str(model_free_dir))
sys.path.insert(0, str(dsrl_model_dir))
sys.path.insert(0, str(offline_mpc_dir))

import dsrl
import dsrl.offline_safety_gymnasium

from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet


def normalize_observation(mu_obs, std_obs, obs, eps=1e-6):
    """Normalize observations using saved statistics"""
    if mu_obs is None:
        return obs
    if isinstance(obs, torch.Tensor):
        mu = mu_obs.to(obs.device) if isinstance(mu_obs, torch.Tensor) else torch.tensor(mu_obs, device=obs.device, dtype=obs.dtype)
        std = std_obs.to(obs.device) if isinstance(std_obs, torch.Tensor) else torch.tensor(std_obs, device=obs.device, dtype=obs.dtype)
        return (obs - mu) / (std + eps)
    else:
        mu = mu_obs if not isinstance(mu_obs, torch.Tensor) else mu_obs.cpu().numpy()
        std = std_obs if not isinstance(std_obs, torch.Tensor) else std_obs.cpu().numpy()
        return (obs - mu) / (std + eps)


@torch.no_grad()
def evaluate_policy(env, score_model, device, norm_fn, 
                   diffusion_steps=15, horizon=5, act_dim=None,
                   max_steps=1000):
    """
    Run a single evaluation episode
    Returns: reward, cost, length
    """
    obs, _ = env.reset()
    obs = torch.as_tensor(norm_fn(obs), dtype=torch.float32, device=device).unsqueeze(0)
    
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        # Generate trajectory actions using the flow model
        act = score_model.select_trajectory_actions(
            obs, 
            diffusion_steps=diffusion_steps,
            horizon=horizon,
            act_dim=act_dim,
            use_first_action=True
        )
        
        # Execute first action
        next_obs, reward, terminated, truncated, info = env.step(act[0])
        obs = torch.as_tensor(norm_fn(next_obs), dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward += reward
        total_cost += info.get("cost", 0.0)
        total_len += 1
        steps += 1
        done = terminated or truncated
    
    return total_reward, total_cost, total_len


def load_model_checkpoint(checkpoint_path, obs_dim, act_dim, train_horizon, device, schedule='linear'):
    """
    Load a saved flow model checkpoint.
    Creates Args object matching the training script's configuration.
    """
    # Setup marginal_prob_std function
    marginal_prob_std_fn = functools.partial(marginal_prob_std, schedule=schedule, device=device)
    
    # Create args object matching training script
    class Args:
        def __init__(self):
            self.device = device
            self.marginal_prob_std_fn = marginal_prob_std_fn
            self.hidden_sizes = [256, 256]  # From default_cfg
            self.schedule = schedule
            # Add any other attributes that ScoreNet might need
            self.embed_dim = 128  # Common default
    
    args = Args()
    
    # Create model with same architecture as training
    traj_dim = train_horizon * act_dim
    model = ScoreNet(
        input_dim=obs_dim + traj_dim,
        output_dim=traj_dim,
        marginal_prob_std=marginal_prob_std_fn,
        args=args
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model


def find_latest_checkpoint(torch_save_dir, prefix='flow'):
    """
    Find the latest checkpoint with given prefix
    """
    checkpoint_files = list(Path(torch_save_dir).glob(f'{prefix}_model_*.pt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found with prefix '{prefix}' in {torch_save_dir}")
    
    # Extract iteration numbers and find max
    iterations = []
    for f in checkpoint_files:
        try:
            iter_num = int(f.stem.split('_')[-1])
            iterations.append((iter_num, f))
        except:
            continue
    
    if not iterations:
        raise ValueError(f"Could not parse iteration numbers from checkpoint files")
    
    latest = max(iterations, key=lambda x: x[0])
    return latest[1], latest[0]

def find_all_checkpoints(torch_save_dir, prefix='flow'):
    """
    Returns a sorted list of (iteration, checkpoint_path)
    for all checkpoints matching prefix_model_*.pt
    """
    checkpoint_files = list(Path(torch_save_dir).glob(f'{prefix}_model_*.pt'))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoints found with prefix '{prefix}' in {torch_save_dir}"
        )

    iterations = []
    for f in checkpoint_files:
        try:
            # filename like flow_model_20000.pt → iteration = 20000
            iter_num = int(f.stem.split('_')[-1])
            iterations.append((iter_num, f))
        except:
            continue

    iterations.sort(key=lambda x: x[0])   # ascending order
    return iterations


def main(args):
    # Setup device
    device = torch.device(args.device)

    # Create environment
    print(f"Creating environment: {args.task}")
    env = gym.make(args.task)
    env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"Environment: obs_dim={obs_dim}, act_dim={act_dim}")

    # Determine where checkpoints are stored
    torch_save_dir = Path(args.model_path) / "torch_save"
    if not torch_save_dir.exists():
        torch_save_dir = Path(args.model_path)

    print(f"Looking for checkpoints in: {torch_save_dir}")

    # Load ALL checkpoints
    all_checkpoints = find_all_checkpoints(torch_save_dir, prefix='flow')
    print(f"Found {len(all_checkpoints)} checkpoints.")

    # Normalization function (no-op unless training saved stats)
    norm_fn = lambda x: x

    # Store results for all checkpoints
    all_results = []

    # Loop over each checkpoint
    for iteration, checkpoint_path in all_checkpoints:
        print("\n" + "=" * 60)
        print(f" Evaluating checkpoint iteration {iteration}")
        print("=" * 60)

        # Load model for this checkpoint
        model = load_model_checkpoint(
            checkpoint_path,
            obs_dim,
            act_dim,
            args.train_horizon,
            device,
            schedule=args.schedule
        )

        rewards, costs, lengths = [], [], []

        # Run exactly 2 episodes for each checkpoint
        for ep in range(2):
            reward, cost, length = evaluate_policy(
                env,
                model,
                device,
                norm_fn,
                diffusion_steps=args.diffusion_steps,
                horizon=args.train_horizon,
                act_dim=act_dim,
                max_steps=args.max_steps
            )

            print(f"  Episode {ep+1}/2 → R={reward:.2f}, C={cost:.2f}, L={length}")

            rewards.append(reward)
            costs.append(cost)
            lengths.append(length)

        # Stats for this checkpoint
        stats = {
            'iteration': iteration,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'mean_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths)),
        }

        all_results.append(stats)

    # Save results for all checkpoints
    output_dir = Path(args.model_path) / "evaluation_results_all"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"all_checkpoint_results_seed{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(" All checkpoint evaluations completed!")
    print(f" Results saved to: {output_file}")
    print("=" * 60)

    env.close()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained IPL+Flow models')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model directory (should contain torch_save/ subdirectory)')
    parser.add_argument('--task', type=str, required=True,
                       help='Environment task name')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed for evaluation')
    
    # Evaluation settings
    parser.add_argument('--num-evals', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    # Model settings (should match training)
    parser.add_argument('--train-horizon', type=int, default=5,
                       help='Training horizon (trajectory length)')
    parser.add_argument('--diffusion-steps', type=int, default=15,
                       help='Number of diffusion sampling steps')
    parser.add_argument('--schedule', type=str, default='linear',
                       help='Diffusion schedule type')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run evaluation on (cpu or cuda)')
    
    args = parser.parse_args()
    main(args)
