#!/usr/bin/env python3
"""
Fast Evaluation Script for IPL Flow Matching Checkpoints

Evaluates all flow model checkpoints in a torch_save folder.
Outputs results to CSV with mean ± std statistics for reward, cost, and episode length.

Usage:
    python eval_ipl_checkpoints.py --checkpoint_dir /path/to/torch_save --task OfflineSwimmerVelocityGymnasium-v1 --num_episodes 10

    # Or auto-detect task from config.json in parent directory:
    python eval_ipl_checkpoints.py --checkpoint_dir /path/to/experiment/torch_save --num_episodes 10
"""

import os
import sys
import glob
import json
import csv
import argparse
import time
import functools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckpointInfo:
    """Holds checkpoint metadata."""
    path: str
    step: int
    prefix: str  # 'flow', 'flow_final', etc.
    
    def __repr__(self):
        return f"Checkpoint(step={self.step}, prefix='{self.prefix}')"


@dataclass
class EvalResult:
    """Holds evaluation results for a single checkpoint."""
    step: int
    prefix: str
    mean_reward: float
    std_reward: float
    mean_cost: float
    std_cost: float
    mean_length: float
    std_length: float
    mean_action_norm: float
    eval_time: float
    num_episodes: int


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_observation(mu_obs: Optional[torch.Tensor], 
                          std_obs: Optional[torch.Tensor], 
                          obs: np.ndarray) -> np.ndarray:
    """Normalize observation using precomputed statistics."""
    if mu_obs is None:
        return obs
    mu = mu_obs.cpu().numpy() if isinstance(mu_obs, torch.Tensor) else mu_obs
    std = std_obs.cpu().numpy() if isinstance(std_obs, torch.Tensor) else std_obs
    return (obs - mu) / (std + EP)


def discover_checkpoints(checkpoint_dir: str, 
                         pattern: str = "flow*_model_*.pt") -> List[CheckpointInfo]:
    """
    Discover all flow model checkpoints in a directory.
    
    Args:
        checkpoint_dir: Path to torch_save directory
        pattern: Glob pattern for checkpoint files
        
    Returns:
        List of CheckpointInfo sorted by step number
    """
    search_pattern = os.path.join(checkpoint_dir, pattern)
    checkpoint_files = glob.glob(search_pattern)
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found matching {search_pattern}")
    
    checkpoints = []
    for path in checkpoint_files:
        basename = os.path.basename(path)
        # Parse: {prefix}_model_{step}.pt
        # Examples: flow_model_20000.pt, flow_final_model_300000.pt
        try:
            parts = basename.replace('.pt', '').split('_model_')
            if len(parts) == 2:
                prefix = parts[0]
                step = int(parts[1])
                checkpoints.append(CheckpointInfo(path=path, step=step, prefix=prefix))
        except (ValueError, IndexError):
            print(f"⚠️  Skipping unrecognized checkpoint: {basename}")
            continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x.step)
    return checkpoints


def load_config(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Load config.json from the experiment directory.
    
    Args:
        checkpoint_dir: Path to torch_save directory (config is in parent)
        
    Returns:
        Configuration dictionary
    """
    # Config is typically in parent of torch_save
    parent_dir = os.path.dirname(checkpoint_dir.rstrip('/'))
    config_path = os.path.join(parent_dir, 'config.json')
    
    if not os.path.exists(config_path):
        # Try current directory
        config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


def compute_normalization_stats(task: str, 
                                config: Dict, 
                                device: torch.device) -> Tuple[Optional[torch.Tensor], 
                                                                Optional[torch.Tensor]]:
    """
    Compute observation normalization statistics from dataset.
    
    Returns:
        (mu_obs, std_obs) tensors or (None, None) if normalization disabled
    """
    if not config.get('normalize_observation', False):
        return None, None
    
    print("📊 Computing normalization statistics from dataset...")
    
    # Create environment to get dataset
    env = gym.make(task)
    
    # Get dataset configuration
    dataset_config = {
        "density": config.get("density", 1.0),
        "inpaint_ranges": config.get("inpaint_ranges", []),
        "num_negative_trajectories": config.get("num_negative_trajectories", 50),
        "num_union_trajectories": config.get("num_union_trajectories", -1),
        "non_pref_noise": config.get("non_pref_noise", 0.0),
    }
    
    # Get trajectory length
    raw_data = env.get_dataset()
    dones_idx = np.where((raw_data["terminals"] == 1) | (raw_data["timeouts"] == 1))[0]
    traj_lengths = []
    start = 0
    for end_idx in dones_idx:
        traj_lengths.append(end_idx - start + 1)
        start = end_idx + 1
    max_traj_len = max(traj_lengths) if traj_lengths else 1000
    
    # Get formatted data
    d4rl_data = get_dataset_in_d4rl_format(
        env=env,
        config=dataset_config,
        task=task,
        ep_len=max_traj_len,
        num_folds=config.get("num_folds", 1)
    )
    
    neg_data, union_data = get_neg_and_union_data_2(d4rl_data, dataset_config)
    _, _, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
    
    mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32).to(device)
    std_obs = torch.as_tensor(std_obs, dtype=torch.float32).to(device)
    
    env.close()
    return mu_obs, std_obs


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def evaluate_single_episode(env: gym.Env,
                            flow_model: ScoreNet,
                            device: torch.device,
                            norm_fn,
                            diffusion_steps: int,
                            seed: int) -> Tuple[float, float, int, float]:
    """
    Evaluate a single episode.
    
    Returns:
        (total_reward, total_cost, episode_length, mean_action_norm)
    """
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    action_norms = []
    done = False
    
    while not done:
        # Normalize and convert to tensor
        obs_norm = norm_fn(obs)
        obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get action from flow model
        action = flow_model.select_actions(obs_tensor, diffusion_steps=diffusion_steps)
        
        # Handle list output
        if isinstance(action, list):
            action = action[0]
        action = np.asarray(action, dtype=np.float32)
        
        # Track action norm
        action_norms.append(np.linalg.norm(action))
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(next_obs, dtype=np.float32)
        
        total_reward += reward
        total_cost += info.get('cost', 0.0)
        total_len += 1
        done = terminated or truncated
    
    mean_action_norm = np.mean(action_norms) if action_norms else 0.0
    return total_reward, total_cost, total_len, mean_action_norm


def evaluate_checkpoint(checkpoint: CheckpointInfo,
                        flow_model: ScoreNet,
                        env: gym.Env,
                        device: torch.device,
                        norm_fn,
                        num_episodes: int,
                        diffusion_steps: int,
                        base_seed: int) -> EvalResult:
    """
    Evaluate a single checkpoint across multiple episodes.
    
    Args:
        checkpoint: Checkpoint metadata
        flow_model: Pre-initialized ScoreNet (state_dict will be loaded)
        env: Gymnasium environment
        device: Torch device
        norm_fn: Observation normalization function
        num_episodes: Number of evaluation episodes
        diffusion_steps: ODE integration steps for flow sampling
        base_seed: Base random seed for reproducibility
        
    Returns:
        EvalResult with statistics
    """
    start_time = time.time()
    
    # Load checkpoint weights
    state_dict = torch.load(checkpoint.path, map_location=device, weights_only=True)
    flow_model.load_state_dict(state_dict)
    flow_model.eval()
    
    # Run episodes
    rewards, costs, lengths, action_norms = [], [], [], []
    
    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        r, c, l, a_norm = evaluate_single_episode(
            env, flow_model, device, norm_fn, diffusion_steps, ep_seed
        )
        rewards.append(r)
        costs.append(c)
        lengths.append(l)
        action_norms.append(a_norm)
    
    eval_time = time.time() - start_time
    
    return EvalResult(
        step=checkpoint.step,
        prefix=checkpoint.prefix,
        mean_reward=np.mean(rewards),
        std_reward=np.std(rewards),
        mean_cost=np.mean(costs),
        std_cost=np.std(costs),
        mean_length=np.mean(lengths),
        std_length=np.std(lengths),
        mean_action_norm=np.mean(action_norms),
        eval_time=eval_time,
        num_episodes=num_episodes
    )


# =============================================================================
# Results Saving
# =============================================================================

def save_results_csv(results: List[EvalResult], output_path: str):
    """Save evaluation results to CSV."""
    fieldnames = [
        'step', 'prefix', 'mean_reward', 'std_reward', 
        'mean_cost', 'std_cost', 'mean_length', 'std_length',
        'mean_action_norm', 'eval_time', 'num_episodes'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'step': r.step,
                'prefix': r.prefix,
                'mean_reward': f"{r.mean_reward:.4f}",
                'std_reward': f"{r.std_reward:.4f}",
                'mean_cost': f"{r.mean_cost:.4f}",
                'std_cost': f"{r.std_cost:.4f}",
                'mean_length': f"{r.mean_length:.2f}",
                'std_length': f"{r.std_length:.2f}",
                'mean_action_norm': f"{r.mean_action_norm:.4f}",
                'eval_time': f"{r.eval_time:.2f}",
                'num_episodes': r.num_episodes
            })
    
    print(f"📄 Results saved to: {output_path}")


def save_summary_json(results: List[EvalResult], 
                      config: Dict,
                      output_path: str):
    """Save summary statistics to JSON."""
    if not results:
        return
    
    # Find best checkpoint by reward (with cost constraint consideration)
    best_by_reward = max(results, key=lambda r: r.mean_reward)
    best_by_cost = min(results, key=lambda r: r.mean_cost)
    
    # Find best trade-off (highest reward with cost < threshold)
    # Default: no cost constraint, just max reward
    best_tradeoff = best_by_reward
    
    summary = {
        "num_checkpoints": len(results),
        "num_episodes_per_checkpoint": results[0].num_episodes if results else 0,
        "total_eval_time": sum(r.eval_time for r in results),
        "best_by_reward": {
            "step": best_by_reward.step,
            "prefix": best_by_reward.prefix,
            "mean_reward": best_by_reward.mean_reward,
            "std_reward": best_by_reward.std_reward,
            "mean_cost": best_by_reward.mean_cost,
            "std_cost": best_by_reward.std_cost,
        },
        "best_by_cost": {
            "step": best_by_cost.step,
            "prefix": best_by_cost.prefix,
            "mean_reward": best_by_cost.mean_reward,
            "std_reward": best_by_cost.std_reward,
            "mean_cost": best_by_cost.mean_cost,
            "std_cost": best_by_cost.std_cost,
        },
        "config_used": {
            "task": config.get("task", "unknown"),
            "normalize_observation": config.get("normalize_observation", False),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Summary saved to: {output_path}")


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def main(args):
    print("=" * 60)
    print("IPL Flow Checkpoint Evaluator")
    print("=" * 60)
    
    device = torch.device(args.device)
    print(f"🖥️  Device: {device}")
    
    # Discover checkpoints
    print(f"\n📁 Scanning: {args.checkpoint_dir}")
    checkpoints = discover_checkpoints(args.checkpoint_dir, args.pattern)
    print(f"✅ Found {len(checkpoints)} checkpoints")
    
    # Apply checkpoint filtering
    if args.every_n > 1:
        checkpoints = checkpoints[::args.every_n]
        print(f"   Filtered to every {args.every_n}th: {len(checkpoints)} checkpoints")
    
    if args.max_checkpoints > 0 and len(checkpoints) > args.max_checkpoints:
        # Keep evenly spaced checkpoints
        indices = np.linspace(0, len(checkpoints) - 1, args.max_checkpoints, dtype=int)
        checkpoints = [checkpoints[i] for i in indices]
        print(f"   Limited to {args.max_checkpoints} evenly spaced checkpoints")
    
    # Load config
    config = load_config(args.checkpoint_dir)
    
    # Determine task
    task = args.task or config.get('task')
    if not task:
        raise ValueError("Task not specified and not found in config.json. Use --task argument.")
    print(f"\n🎮 Task: {task}")
    
    # Update config with command line overrides
    if args.normalize_observation is not None:
        config['normalize_observation'] = args.normalize_observation
    
    # Create environment
    print(f"\n🌍 Creating environment...")
    env = gym.make(task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"   obs_dim={obs_dim}, act_dim={act_dim}")
    
    # Compute normalization stats
    mu_obs, std_obs = compute_normalization_stats(task, config, device)
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    
    if mu_obs is not None:
        print(f"✅ Observation normalization enabled")
    else:
        print(f"ℹ️  Observation normalization disabled")
    
    # Create model (will load weights per checkpoint)
    print(f"\n🧠 Initializing flow model...")
    
    # Create args namespace for ScoreNet
    class ModelArgs:
        def __init__(self):
            self.device = device
            self.schedule = config.get('schedule', 'Linear')
    
    model_args = ModelArgs()
    
    flow_model = ScoreNet(
        input_dim=obs_dim + act_dim,
        output_dim=act_dim,
        marginal_prob_std=None,  # Flow matching mode
        args=model_args
    ).to(device)
    flow_model.eval()
    
    # Warm-up: load first checkpoint and do a dummy forward pass
    print(f"🔥 Warming up model...")
    if checkpoints:
        state_dict = torch.load(checkpoints[0].path, map_location=device, weights_only=True)
        flow_model.load_state_dict(state_dict)
        dummy_obs = torch.zeros(1, obs_dim, device=device)
        _ = flow_model.select_actions(dummy_obs, diffusion_steps=args.diffusion_steps)
    
    # Evaluate all checkpoints
    print(f"\n🚀 Evaluating {len(checkpoints)} checkpoints × {args.num_episodes} episodes each")
    print(f"   Diffusion steps: {args.diffusion_steps}")
    print("-" * 60)
    
    results = []
    total_start = time.time()
    
    pbar = tqdm(checkpoints, desc="Evaluating", unit="ckpt")
    for ckpt in pbar:
        result = evaluate_checkpoint(
            checkpoint=ckpt,
            flow_model=flow_model,
            env=env,
            device=device,
            norm_fn=norm_fn,
            num_episodes=args.num_episodes,
            diffusion_steps=args.diffusion_steps,
            base_seed=args.seed
        )
        results.append(result)
        
        # Update progress bar
        pbar.set_postfix({
            'step': ckpt.step,
            'R': f"{result.mean_reward:.1f}±{result.std_reward:.1f}",
            'C': f"{result.mean_cost:.1f}±{result.std_cost:.1f}"
        })
    
    total_time = time.time() - total_start
    
    # Close environment
    env.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/len(results):.2f}s per checkpoint)")
    
    if results:
        best = max(results, key=lambda r: r.mean_reward)
        print(f"\n🏆 Best checkpoint by reward:")
        print(f"   Step: {best.step} ({best.prefix})")
        print(f"   Reward: {best.mean_reward:.2f} ± {best.std_reward:.2f}")
        print(f"   Cost: {best.mean_cost:.2f} ± {best.std_cost:.2f}")
        print(f"   Length: {best.mean_length:.1f} ± {best.std_length:.1f}")
        
        lowest_cost = min(results, key=lambda r: r.mean_cost)
        print(f"\n🛡️  Lowest cost checkpoint:")
        print(f"   Step: {lowest_cost.step} ({lowest_cost.prefix})")
        print(f"   Reward: {lowest_cost.mean_reward:.2f} ± {lowest_cost.std_reward:.2f}")
        print(f"   Cost: {lowest_cost.mean_cost:.2f} ± {lowest_cost.std_cost:.2f}")
    
    # Save results
    output_dir = args.output_dir or os.path.dirname(args.checkpoint_dir.rstrip('/'))
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, args.output_csv)
    json_path = os.path.join(output_dir, args.output_csv.replace('.csv', '_summary.json'))
    
    config['task'] = task  # Ensure task is in config for summary
    save_results_csv(results, csv_path)
    save_summary_json(results, config, json_path)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate IPL Flow Matching checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint_dir", "-d", 
        type=str, 
        required=True,
        help="Path to torch_save directory containing checkpoints"
    )
    
    # Task configuration
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="Environment task name (auto-detected from config.json if not provided)"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--num_episodes", "-n",
        type=int,
        default=10,
        help="Number of evaluation episodes per checkpoint"
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=5,
        help="Number of ODE integration steps for flow sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for evaluation"
    )
    
    # Checkpoint filtering
    parser.add_argument(
        "--pattern",
        type=str,
        default="flow*_model_*.pt",
        help="Glob pattern for checkpoint files"
    )
    parser.add_argument(
        "--every_n",
        type=int,
        default=1,
        help="Evaluate every Nth checkpoint (for faster evaluation)"
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=0,
        help="Maximum number of checkpoints to evaluate (0 = all)"
    )
    
    # Normalization
    parser.add_argument(
        "--normalize_observation",
        type=lambda x: x.lower() == 'true',
        default=None,
        help="Override observation normalization (true/false, default: from config)"
    )  
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: parent of checkpoint_dir)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="eval_results.csv",
        help="Output CSV filename"
    )
    
    args = parser.parse_args()
    main(args)
