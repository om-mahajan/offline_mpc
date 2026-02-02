#!/usr/bin/env python3
"""
Fast Evaluation Script for IPL-Style Flow Matching Models

Evaluates all flow model checkpoints in a training log directory.
Supports vectorized environments for fast parallel episode collection.

Usage:
    python eval_ipltwin.py --log_dir ./logs/experiment/task/algo/seed-001-...
    python eval_ipltwin.py --log_dir ./logs/... --num_episodes 50 --num_envs 8
    python eval_ipltwin.py --log_dir ./logs/... --checkpoint_filter "flow_best*"
"""

import os
import os.path as osp
import sys
import json
import glob
import re
import time
import functools
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Add parent paths for imports
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../..')))

import gymnasium as gym
import dsrl
import dsrl.offline_safety_gymnasium

from diffusion_SDE.model import ScoreNet
from dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
    get_normalized_data
)

EP = 1e-6


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""
    path: str
    prefix: str
    iteration: int
    filename: str


def discover_checkpoints(log_dir: str, pattern: str = "flow*_model_*.pt") -> List[CheckpointInfo]:
    """
    Discover all flow model checkpoints in the log directory.
    
    Args:
        log_dir: Path to training run directory
        pattern: Glob pattern for checkpoint files
        
    Returns:
        List of CheckpointInfo sorted by iteration number
    """
    torch_save_dir = osp.join(log_dir, "torch_save")
    if not osp.exists(torch_save_dir):
        raise FileNotFoundError(f"No torch_save directory found at: {torch_save_dir}")
    
    checkpoint_paths = glob.glob(osp.join(torch_save_dir, pattern))
    
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matching '{pattern}' found in {torch_save_dir}")
    
    checkpoints = []
    for path in checkpoint_paths:
        filename = osp.basename(path)
        # Extract iteration number from filename like "flow_model_100000.pt" or "flow_best_model_220000.pt"
        match = re.search(r'_model_(\d+)\.pt$', filename)
        if match:
            iteration = int(match.group(1))
            # Extract prefix (everything before _model_)
            prefix = filename.replace(f"_model_{iteration}.pt", "")
            checkpoints.append(CheckpointInfo(
                path=path,
                prefix=prefix,
                iteration=iteration,
                filename=filename
            ))
    
    # Sort by iteration
    checkpoints.sort(key=lambda x: x.iteration)
    return checkpoints


def load_config(log_dir: str) -> dict:
    """Load training config from log directory."""
    config_path = osp.join(log_dir, "config.json")
    if not osp.exists(config_path):
        raise FileNotFoundError(f"No config.json found at: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def normalize_observation(mu_obs, std_obs, obs):
    """Normalize observation using precomputed statistics."""
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


def compute_normalization_stats(task: str, config: dict, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute normalization statistics from dataset.
    
    Returns:
        (mu_obs, std_obs) tensors or (None, None) if normalization not needed
    """
    if not config.get("normalize_observation", False):
        return None, None
    
    print("Computing normalization statistics from dataset...")
    
    # Create temporary env to get dataset
    temp_env = gym.make(task)
    
    # Get dataset config
    dataset_config = {
        "density": config.get("density", 1.0),
        "inpaint_ranges": config.get("inpaint_ranges", []),
        "num_negative_trajectories": config.get("num_negative_trajectories", 50),
        "num_union_trajectories": config.get("num_union_trajectories", -1),
        "non_pref_noise": config.get("non_pref_noise", 0.0),
    }
    
    raw_data = temp_env.get_dataset()
    dones_idx = np.where((raw_data["terminals"] == 1) | (raw_data["timeouts"] == 1))[0]
    traj_lengths = []
    start = 0
    for end_idx in dones_idx:
        traj_lengths.append(end_idx - start + 1)
        start = end_idx + 1
    max_traj_len = max(traj_lengths)
    
    d4rl_data = get_dataset_in_d4rl_format(
        env=temp_env,
        config=dataset_config,
        task=task,
        ep_len=max_traj_len,
        num_folds=config.get("num_folds", 1)
    )
    
    neg_data, union_data = get_neg_and_union_data_2(d4rl_data, dataset_config)
    neg_data, union_data, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
    
    mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32).to(device)
    std_obs = torch.as_tensor(std_obs, dtype=torch.float32).to(device)
    
    temp_env.close()
    print("✅ Normalization statistics computed")
    
    return mu_obs, std_obs


def create_model(obs_dim: int, act_dim: int, config: dict, device: torch.device) -> ScoreNet:
    """Create ScoreNet model with correct architecture."""
    # Create args-like object for ScoreNet
    class Args:
        pass
    
    args = Args()
    args.device = device
    args.schedule = config.get("schedule", "Linear")
    
    model = ScoreNet(
        input_dim=obs_dim + act_dim,
        output_dim=act_dim,
        marginal_prob_std=None,  # Flow Matching mode
        args=args
    ).to(device)
    
    return model


@torch.no_grad()
def evaluate_single_episode(env, model, device, norm_fn, diffusion_steps: int = 10, seed: int = None) -> Tuple[float, float, int]:
    """
    Evaluate a single episode.
    
    Returns:
        (total_reward, total_cost, episode_length)
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
    obs = torch.as_tensor(norm_fn(obs), dtype=torch.float32, device=device).unsqueeze(0)
    
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    
    while not done:
        act = model.select_actions(obs, diffusion_steps=diffusion_steps)
        # select_actions returns a list when input has batch dim, extract first element
        if isinstance(act, list):
            act = act[0]
        # Also squeeze if still has extra dimension
        if isinstance(act, np.ndarray) and act.ndim == 2 and act.shape[0] == 1:
            act = act.squeeze(0)
        next_obs, reward, terminated, truncated, info = env.step(act)
        next_obs = np.array(next_obs) if not isinstance(next_obs, np.ndarray) else next_obs
        obs = torch.as_tensor(norm_fn(next_obs), dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward += reward
        total_cost += info.get('cost', 0.0)
        total_len += 1
        done = terminated or truncated
    
    return total_reward, total_cost, total_len


@torch.no_grad()
def evaluate_vectorized(envs, model, device, norm_fn, num_episodes: int, diffusion_steps: int = 10) -> Dict[str, np.ndarray]:
    """
    Evaluate using vectorized environments for speed.
    
    Args:
        envs: Vectorized environment (SyncVectorEnv)
        model: Flow model
        device: torch device
        norm_fn: Normalization function
        num_episodes: Total number of episodes to collect
        diffusion_steps: Number of ODE integration steps
        
    Returns:
        Dict with arrays of rewards, costs, lengths
    """
    num_envs = envs.num_envs
    
    # Track results
    all_rewards = []
    all_costs = []
    all_lengths = []
    
    # Per-env accumulators
    episode_rewards = np.zeros(num_envs)
    episode_costs = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=int)
    
    # Reset all envs
    obs, _ = envs.reset()
    obs = torch.as_tensor(norm_fn(obs), dtype=torch.float32, device=device)
    
    completed = 0
    pbar = tqdm(total=num_episodes, desc="Episodes", leave=False)
    
    while completed < num_episodes:
        # Get actions for all envs at once
        actions = model.select_actions(obs, diffusion_steps=diffusion_steps)
        
        # Step all envs
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # Accumulate
        episode_rewards += rewards
        episode_lengths += 1
        
        # Handle costs (may be in info dict)
        for i in range(num_envs):
            if 'cost' in infos:
                episode_costs[i] += infos['cost'][i] if hasattr(infos['cost'], '__getitem__') else infos['cost']
            elif 'final_info' in infos and infos['final_info'][i] is not None:
                episode_costs[i] += infos['final_info'][i].get('cost', 0.0)
        
        # Check for done envs
        dones = terminateds | truncateds
        for i in range(num_envs):
            if dones[i] and completed < num_episodes:
                all_rewards.append(episode_rewards[i])
                all_costs.append(episode_costs[i])
                all_lengths.append(episode_lengths[i])
                completed += 1
                pbar.update(1)
                
                # Reset accumulators for this env
                episode_rewards[i] = 0.0
                episode_costs[i] = 0.0
                episode_lengths[i] = 0
        
        # Prepare next obs
        next_obs = np.array(next_obs) if not isinstance(next_obs, np.ndarray) else next_obs
        obs = torch.as_tensor(norm_fn(next_obs), dtype=torch.float32, device=device)
    
    pbar.close()
    
    return {
        'rewards': np.array(all_rewards[:num_episodes]),
        'costs': np.array(all_costs[:num_episodes]),
        'lengths': np.array(all_lengths[:num_episodes])
    }


@torch.no_grad()
def evaluate_sequential(env, model, device, norm_fn, num_episodes: int, diffusion_steps: int = 10, base_seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Evaluate using a single environment sequentially.
    This is the reliable default that works with all models.
    
    Returns:
        Dict with arrays of rewards, costs, lengths
    """
    all_rewards = []
    all_costs = []
    all_lengths = []
    
    for i in tqdm(range(num_episodes), desc="Episodes", leave=False):
        seed = base_seed + i if base_seed is not None else None
        reward, cost, length = evaluate_single_episode(env, model, device, norm_fn, diffusion_steps, seed=seed)
        all_rewards.append(reward)
        all_costs.append(cost)
        all_lengths.append(length)
    
    return {
        'rewards': np.array(all_rewards),
        'costs': np.array(all_costs),
        'lengths': np.array(all_lengths)
    }


def evaluate_checkpoint(
    checkpoint: CheckpointInfo,
    model: ScoreNet,
    envs,
    device: torch.device,
    norm_fn,
    num_episodes: int,
    diffusion_steps: int,
    use_vectorized: bool = False,
    base_seed: int = 0
) -> Dict:
    """
    Evaluate a single checkpoint.
    
    Returns:
        Dict with checkpoint info and evaluation statistics
    """
    # Load weights
    state_dict = torch.load(checkpoint.path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Run evaluation (sequential only - vectorized not supported by model)
    results = evaluate_sequential(envs, model, device, norm_fn, num_episodes, diffusion_steps, base_seed=base_seed)
    
    return {
        'checkpoint': checkpoint.filename,
        'prefix': checkpoint.prefix,
        'iteration': checkpoint.iteration,
        'reward_mean': results['rewards'].mean(),
        'reward_std': results['rewards'].std(),
        'cost_mean': results['costs'].mean(),
        'cost_std': results['costs'].std(),
        'length_mean': results['lengths'].mean(),
        'length_std': results['lengths'].std(),
        'num_episodes': len(results['rewards'])
    }


def main(args):
    print("="*60)
    print("IPL Flow Matching - Checkpoint Evaluation")
    print("="*60)
    
    # Setup device
    device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load config
    print(f"\nLoading config from: {args.log_dir}")
    config = load_config(args.log_dir)
    
    # Override task if specified
    task = args.task if args.task else config.get("task", None)
    if task is None:
        raise ValueError("Task not specified and not found in config.json")
    print(f"Task: {task}")
    
    # Discover checkpoints
    print(f"\nDiscovering checkpoints...")
    checkpoints = discover_checkpoints(args.log_dir, args.checkpoint_filter)
    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.filename} (iter {ckpt.iteration})")
    
    # Create environment
    # NOTE: Vectorized environments are disabled because the flow model's ODE solver
    # doesn't properly handle batched time embeddings. Using sequential evaluation instead.
    print(f"\nCreating environment...")
    print("Using sequential evaluation (model doesn't support batched inference)")
    envs = gym.make(task)
    envs.reset(seed=args.seed)
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    use_vectorized = False
    
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    
    # Compute normalization stats
    mu_obs, std_obs = compute_normalization_stats(task, config, device)
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    
    # Create model (will load weights for each checkpoint)
    print("\nCreating model...")
    model = create_model(obs_dim, act_dim, config, device)
    model.eval()
    
    # Evaluate all checkpoints
    print(f"\nEvaluating {len(checkpoints)} checkpoints with {args.num_episodes} episodes each...")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print("-"*60)
    
    results = []
    for ckpt in tqdm(checkpoints, desc="Checkpoints"):
        tqdm.write(f"\nEvaluating: {ckpt.filename}")
        start_time = time.time()
        
        result = evaluate_checkpoint(
            checkpoint=ckpt,
            model=model,
            envs=envs,
            device=device,
            norm_fn=norm_fn,
            num_episodes=args.num_episodes,
            diffusion_steps=args.diffusion_steps,
            use_vectorized=use_vectorized,
            base_seed=args.seed
        )
        
        elapsed = time.time() - start_time
        result['eval_time_sec'] = elapsed
        results.append(result)
        
        tqdm.write(f"  Reward: {result['reward_mean']:.2f} ± {result['reward_std']:.2f}")
        tqdm.write(f"  Cost: {result['cost_mean']:.2f} ± {result['cost_std']:.2f}")
        tqdm.write(f"  Length: {result['length_mean']:.1f}")
        tqdm.write(f"  Time: {elapsed:.1f}s")
    
    # Close environments
    envs.close()
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save results
    output_path = osp.join(args.log_dir, "eval_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Find best checkpoint
    best_idx = df['reward_mean'].idxmax()
    best = df.iloc[best_idx]
    print(f"\n🏆 Best checkpoint: {best['checkpoint']}")
    print(f"   Reward: {best['reward_mean']:.2f} ± {best['reward_std']:.2f}")
    print(f"   Cost: {best['cost_mean']:.2f} ± {best['cost_std']:.2f}")
    
    # Find safest checkpoint (lowest cost with reasonable reward)
    if df['cost_mean'].min() < df['cost_mean'].max():
        safe_idx = df['cost_mean'].idxmin()
        safe = df.iloc[safe_idx]
        print(f"\n🛡️  Safest checkpoint: {safe['checkpoint']}")
        print(f"   Reward: {safe['reward_mean']:.2f} ± {safe['reward_std']:.2f}")
        print(f"   Cost: {safe['cost_mean']:.2f} ± {safe['cost_std']:.2f}")
    
    print("\n" + "="*60)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate IPL Flow Matching checkpoints")
    
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to training log directory containing torch_save/ and config.json")
    parser.add_argument("--task", type=str, default=None,
                        help="Environment task (overrides config.json if specified)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of evaluation episodes per checkpoint")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments (currently unused, sequential eval only)")
    parser.add_argument("--diffusion_steps", type=int, default=15,
                        help="Number of ODE integration steps (fewer = faster, default 10)")
    parser.add_argument("--checkpoint_filter", type=str, default="flow*_model_*.pt",
                        help="Glob pattern to filter checkpoints (default: flow*_model_*.pt)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for environment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    
    args = parser.parse_args()
    main(args)
