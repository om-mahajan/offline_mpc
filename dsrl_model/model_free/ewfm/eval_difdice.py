"""
Evaluation script for DifDICE trained models.
Loads saved checkpoints and evaluates them on the environment.
Runs completely on CPU.
"""

import os
import sys
from pathlib import Path
import argparse
import functools
import glob
import re

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add the offline_mpc directory to Python path
current_file = Path(__file__).resolve()
offline_mpc_dir = current_file.parents[3]
sys.path.insert(0, str(offline_mpc_dir))

# DSRL imports
import gymnasium as gym
import dsrl
import dsrl.offline_safety_gymnasium

# Model imports
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet
from dsrl_model.utils.models import SafeDiceCritic
from energy_filter_net import EnergyFilterNet

EP = 1e-6


def normalize_observation(mu_obs, std_obs, obs):
    """Normalize observations using precomputed mean and std."""
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


@torch.inference_mode()
def evaluate_flow_policy(eval_env, score_model, device, norm_fn, diffusion_steps=15, eval_horizon=5, max_steps=1000):
    """
    Evaluate flow policy by generating horizon-length action sequences.
    Only the first action from each sequence is executed.
    """
    eval_done = False
    eval_obs, _ = eval_env.reset()
    eval_obs = torch.as_tensor(norm_fn(eval_obs), dtype=torch.float32, device=device).unsqueeze(0)
    eval_reward, eval_cost, eval_len = 0.0, 0.0, 0
    
    while not eval_done and eval_len < max_steps:
        # Generate horizon-length action sequence
        obs_horizon = eval_obs.squeeze(0).repeat(eval_horizon, 1)  # [horizon, obs_dim]
        
        # Generate action sequence
        act_sequence = score_model.select_actions(obs_horizon, diffusion_steps=diffusion_steps)
        
        # Extract first action only - optimized to minimize conversions
        if isinstance(act_sequence, torch.Tensor):
            if device.type == 'cpu':
                act_np = act_sequence[0].numpy() if act_sequence.ndim > 1 else act_sequence.numpy()
            else:
                act_np = act_sequence[0].cpu().numpy() if act_sequence.ndim > 1 else act_sequence.cpu().numpy()
        elif isinstance(act_sequence, np.ndarray):
            act_np = act_sequence[0] if act_sequence.ndim > 1 else act_sequence
        else:  # list
            act_np = act_sequence[0]
        
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


def extract_iteration_number(filename):
    """Extract iteration number from checkpoint filename."""
    match = re.search(r'_(\d+)\.pt$', filename)
    if match:
        return int(match.group(1))
    return -1


def load_model_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint."""
    try:
        # Load on CPU regardless of where it was saved
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return False


def find_all_checkpoints(checkpoint_dir, model_prefix):
    """Find all checkpoints for a given model type."""
    pattern = os.path.join(checkpoint_dir, f"{model_prefix}_*.pt")
    checkpoints = glob.glob(pattern)
    
    # Sort by iteration number
    checkpoints_with_iter = [(cp, extract_iteration_number(cp)) for cp in checkpoints]
    checkpoints_with_iter = [(cp, it) for cp, it in checkpoints_with_iter if it >= 0]
    checkpoints_with_iter.sort(key=lambda x: x[1])
    
    return checkpoints_with_iter


def main(args):
    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Setup environment
    print(f"Setting up environment: {args.task}")
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)
    
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    
    # Setup marginal_prob_std for flow matching (always use linear schedule)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, schedule='linear', device=device)
    
    # Create dummy args object for ScoreNet
    class DummyArgs:
        pass
    
    model_args = DummyArgs()
    model_args.device = device
    model_args.marginal_prob_std_fn = marginal_prob_std_fn
    model_args.schedule = 'linear'  # Flow matching always uses linear schedule
    
    # Initialize models (architectures only)
    print("Initializing model architectures...")
    
    cost_model = EnergyFilterNet(obs_dim=obs_dim, action_dim=act_dim, max_action=max_action).to(device)
    cost_model.eval()
    
    critic_model = SafeDiceCritic(obs_dim=obs_dim, act_dim=0, hidden_size=256).to(device)
    critic_model.eval()
    
    flow_model = ScoreNet(
        input_dim=obs_dim + act_dim,
        output_dim=act_dim,
        marginal_prob_std=marginal_prob_std_fn,
        args=model_args
    ).to(device)
    flow_model.eval()
    
    # Normalization (if needed)
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        print("Warning: Normalization statistics not available. Running without normalization.")
        # In practice, you'd need to save these during training
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    
    # Find checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, "torch_save")
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    
    # Find all flow model checkpoints
    flow_checkpoints = find_all_checkpoints(checkpoint_dir, "flow_model")
    
    if not flow_checkpoints:
        print("No flow model checkpoints found!")
        return
    
    print(f"Found {len(flow_checkpoints)} flow model checkpoints")
    
    # Evaluate each checkpoint
    results = []
    
    # Progress bar for checkpoints
    checkpoint_pbar = tqdm(flow_checkpoints, desc="Evaluating checkpoints", unit="checkpoint")
    
    for checkpoint_path, iteration in checkpoint_pbar:
        checkpoint_pbar.set_description(f"Checkpoint iter={iteration}")
        
        # Load checkpoint
        success = load_model_checkpoint(checkpoint_path, flow_model, device)
        if not success:
            tqdm.write(f"Skipping checkpoint {iteration} due to loading error")
            continue
        
        # Run evaluation episodes
        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        
        # Progress bar for episodes
        episode_pbar = tqdm(range(args.num_eval_episodes), 
                           desc=f"  Episodes (iter={iteration})", 
                           leave=False, 
                           unit="ep")
        
        for ep in episode_pbar:
            reward, cost, length = evaluate_flow_policy(
                eval_env=eval_env,
                score_model=flow_model,
                device=device,
                norm_fn=norm_fn,
                diffusion_steps=args.diffusion_steps,
                eval_horizon=args.eval_horizon,
                max_steps=args.max_episode_steps
            )
            
            episode_rewards.append(reward)
            episode_costs.append(cost)
            episode_lengths.append(length)
            
            episode_pbar.set_postfix({'R': f'{reward:.1f}', 'C': f'{cost:.1f}', 'L': length})
        
        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_cost = np.mean(episode_costs)
        std_cost = np.std(episode_costs)
        mean_length = np.mean(episode_lengths)
        
        # Log results with tqdm
        tqdm.write(f"\nIteration {iteration}: "
                  f"R={mean_reward:.2f}±{std_reward:.2f}, "
                  f"C={mean_cost:.2f}±{std_cost:.2f}, "
                  f"L={mean_length:.1f}")
        
        results.append({
            'iteration': iteration,
            'checkpoint': checkpoint_path,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_cost': mean_cost,
            'std_cost': std_cost,
            'mean_length': mean_length,
            'episode_rewards': episode_rewards,
            'episode_costs': episode_costs,
            'episode_lengths': episode_lengths
        })
    
    # Save results to file
    results_file = os.path.join(args.checkpoint_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for {args.task}\n")
        f.write(f"{'='*80}\n\n")
        
        for result in results:
            f.write(f"Iteration: {result['iteration']}\n")
            f.write(f"Checkpoint: {os.path.basename(result['checkpoint'])}\n")
            f.write(f"Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}\n")
            f.write(f"Mean Cost: {result['mean_cost']:.2f} ± {result['std_cost']:.2f}\n")
            f.write(f"Mean Length: {result['mean_length']:.2f}\n")
            f.write(f"{'-'*80}\n")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {results_file}")
    print(f"{'='*60}")
    
    # Find best checkpoint
    if results:
        best_result = max(results, key=lambda x: x['mean_reward'])
        print(f"\nBest checkpoint by reward:")
        print(f"  Iteration: {best_result['iteration']}")
        print(f"  Reward: {best_result['mean_reward']:.2f} ± {best_result['std_reward']:.2f}")
        print(f"  Cost: {best_result['mean_cost']:.2f} ± {best_result['std_cost']:.2f}")
        
        # Also find best by reward/cost tradeoff (reward - lambda*cost)
        lambda_cost = 1.0  # Weight for cost penalty
        for result in results:
            result['score'] = result['mean_reward'] - lambda_cost * result['mean_cost']
        
        best_tradeoff = max(results, key=lambda x: x['score'])
        print(f"\nBest checkpoint by reward-cost tradeoff (λ={lambda_cost}):")
        print(f"  Iteration: {best_tradeoff['iteration']}")
        print(f"  Reward: {best_tradeoff['mean_reward']:.2f} ± {best_tradeoff['std_reward']:.2f}")
        print(f"  Cost: {best_tradeoff['mean_cost']:.2f} ± {best_tradeoff['std_cost']:.2f}")
        print(f"  Score: {best_tradeoff['score']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DifDICE trained models")
    
    # Environment
    parser.add_argument("--task", type=str, default="OfflineAntVelocityGymnasium-v1",
                        help="DSRL task name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Checkpoint directory
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing torch_save folder with checkpoints")
    
    # Evaluation settings
    parser.add_argument("--num_eval_episodes", type=int, default=3,
                        help="Number of evaluation episodes per checkpoint (default: 3 for speed)")
    parser.add_argument("--max_episode_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    
    # Flow matching settings
    parser.add_argument("--diffusion_steps", type=int, default=15,
                        help="Number of flow matching steps for sampling")
    parser.add_argument("--eval_horizon", type=int, default=5,
                        help="Horizon for action sequence generation")
    
    # Normalization
    parser.add_argument("--normalize_observation", action="store_true",
                        help="Normalize observations (requires saved statistics)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("DifDICE Flow Matching Evaluation Script")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Evaluation episodes: {args.num_eval_episodes}")
    print(f"Flow matching steps: {args.diffusion_steps}")
    print(f"Eval horizon: {args.eval_horizon}")
    print(f"Schedule: linear (flow matching)")
    print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    main(args)
