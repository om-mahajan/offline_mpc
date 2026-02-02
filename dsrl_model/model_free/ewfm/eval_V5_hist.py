#!/usr/bin/env python3
"""
Evaluation script for ipltwin_V5_hist.py checkpoints.

This script evaluates flow model checkpoints that use:
- Single-action output (NOT trajectory-based)
- ScoreNet with input_dim=obs_dim+act_dim, output_dim=act_dim
- Flow Matching mode (marginal_prob_std=None)
"""
import os
import os.path as osp
import sys
import glob
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import json

# Add repo root to path
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../..')))

import dsrl
import dsrl.offline_safety_gymnasium

from diffusion_SDE.model import ScoreNet


def evaluate_episode(env, model, device, diffusion_steps=15, max_steps=1000, debug=False):
    """
    Evaluate a single episode using the single-action flow model.
    
    Args:
        env: Gymnasium environment
        model: ScoreNet flow model
        device: torch device
        diffusion_steps: Number of ODE integration steps
        max_steps: Maximum episode length
        debug: Print debug info
        
    Returns:
        (total_reward, total_cost, episode_length)
    """
    model.eval()
    
    obs, _ = env.reset()
    
    total_reward = 0.0
    total_cost = 0.0
    episode_len = 0
    done = False
    
    if debug:
        print(f"\nDebug - Episode start:")
        print(f"  obs shape: {obs.shape}, obs[:5]: {obs[:5]}")
    
    while not done and episode_len < max_steps:
        try:
            # Use select_actions which handles conditioning and ODE integration
            # Returns single action directly (no trajectory extraction needed)
            action = model.select_actions(obs, diffusion_steps=diffusion_steps)
            
            # select_actions returns numpy array directly
            if isinstance(action, list):
                action = action[0]
            
            if debug and episode_len < 3:
                print(f"\nDebug - Step {episode_len}:")
                print(f"  action shape: {action.shape if hasattr(action, 'shape') else len(action)}")
                print(f"  action: {action}")
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if debug and episode_len < 3:
                print(f"  reward: {reward:.4f}, cost: {info.get('cost', 0.0):.4f}")
            
            # Update for next step
            obs = next_obs
            total_reward += reward
            total_cost += info.get("cost", 0.0)
            episode_len += 1
            done = terminated or truncated
            
        except Exception as e:
            print(f"Error during episode at step {episode_len}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if debug:
        print(f"\nDebug - Episode end: reward={total_reward:.2f}, cost={total_cost:.2f}, len={episode_len}")
    
    return total_reward, total_cost, episode_len


def load_model_checkpoint(checkpoint_path, obs_dim, act_dim, device):
    """
    Load a V5_hist flow model checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        obs_dim: Observation dimension
        act_dim: Action dimension
        device: torch device
        
    Returns:
        Loaded ScoreNet model
    """
    # Create dummy args object with necessary attributes
    class Args:
        def __init__(self):
            self.device = device
            self.schedule = 'linear'  # Required but unused in FM mode
    
    args = Args()
    
    # V5_hist uses single-action output (NOT trajectory-based)
    model = ScoreNet(
        input_dim=obs_dim + act_dim,  # Condition on obs, predict action
        output_dim=act_dim,            # Single action output
        marginal_prob_std=None,        # Flow Matching mode
        args=args
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full model and state_dict saves
    if isinstance(checkpoint, dict) and 'q_network.q1.0.weight' not in str(checkpoint.keys()):
        # It's a state_dict
        model.load_state_dict(checkpoint)
    elif hasattr(checkpoint, 'state_dict'):
        # It's a full model, extract state_dict
        model.load_state_dict(checkpoint.state_dict())
    else:
        # Try loading directly (might be state_dict already)
        try:
            model.load_state_dict(checkpoint)
        except:
            # Last resort: it might be the full model saved
            model = checkpoint
            model.to(device)
    
    model.eval()
    return model


def find_model_checkpoints(log_dir, step_interval=2000):
    """
    Find all flow model checkpoints at specified intervals.
    
    Args:
        log_dir: Directory containing torch_save folder
        step_interval: Evaluate models at this step interval
        
    Returns:
        List of (step_number, checkpoint_path) tuples sorted by step
    """
    # Check for torch_save subdirectory
    torch_save_dir = osp.join(log_dir, "torch_save")
    if osp.exists(torch_save_dir):
        search_dir = torch_save_dir
        print(f"Found torch_save directory: {torch_save_dir}")
    else:
        search_dir = log_dir
        print(f"No torch_save directory, searching in: {log_dir}")
    
    # Find flow model checkpoints
    patterns = [
        "flow_model_*.pt",
        "flow_*.pt",
    ]
    
    flow_models = []
    for pattern in patterns:
        found = glob.glob(osp.join(search_dir, pattern))
        flow_models.extend(found)
    
    # Remove duplicates
    flow_models = list(set(flow_models))
    
    if not flow_models:
        print(f"No flow models found in {search_dir}")
        print(f"Listing all .pt files:")
        all_pt_files = glob.glob(osp.join(search_dir, "*.pt"))
        for f in all_pt_files[:15]:
            print(f"  {osp.basename(f)}")
        return []
    
    checkpoints = []
    for model_path in flow_models:
        basename = osp.basename(model_path)
        
        # Skip best/final models for interval-based evaluation
        if 'best' in basename or 'final' in basename:
            continue
        
        try:
            # Extract step number: flow_model_XXXXXX.pt or flow_XXXXXX.pt
            name_without_ext = basename.replace(".pt", "")
            parts = name_without_ext.split("_")
            
            for part in reversed(parts):
                if part.isdigit():
                    step = int(part)
                    # Only include models at the specified interval
                    if step % step_interval == 0:
                        checkpoints.append((step, model_path))
                    break
        except (ValueError, IndexError):
            continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        print(f"Found {len(flow_models)} flow models but none match interval {step_interval}")
        print("Sample filenames:")
        for model_path in flow_models[:5]:
            print(f"  {osp.basename(model_path)}")
    
    return checkpoints


def evaluate_all_checkpoints(
    task_name,
    log_dir,
    num_episodes=10,
    diffusion_steps=15,
    step_interval=2000,
    device='cpu',
    max_episode_steps=1000
):
    """
    Evaluate all checkpoints for a given task.
    
    Args:
        task_name: DSRL task name
        log_dir: Directory containing checkpoints
        num_episodes: Episodes per checkpoint
        diffusion_steps: ODE integration steps
        step_interval: Checkpoint interval to evaluate
        device: torch device
        max_episode_steps: Max steps per episode
        
    Returns:
        List of result dictionaries
    """
    print(f"\n{'='*80}")
    print(f"Evaluating V5_HIST task: {task_name}")
    print(f"Log directory: {log_dir}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Create environment
    env = gym.make(task_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Environment: {task_name}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Diffusion steps: {diffusion_steps}\n")
    
    # Find checkpoints
    checkpoints = find_model_checkpoints(log_dir, step_interval=step_interval)
    
    if len(checkpoints) == 0:
        print(f"\n{'!'*80}")
        print(f"ERROR: No checkpoints found at interval {step_interval}")
        print(f"{'!'*80}\n")
        return []
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate:")
    for step, path in checkpoints:
        print(f"  Step {step}: {osp.basename(path)}")
    print()
    
    # Evaluate each checkpoint
    results = []
    
    for step, checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        print(f"\nEvaluating checkpoint at step {step}...")
        
        try:
            # Load model
            model = load_model_checkpoint(
                checkpoint_path=checkpoint_path,
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=device
            )
            
            # Run evaluation episodes
            rewards = []
            costs = []
            lengths = []
            
            for ep in range(num_episodes):
                env.reset(seed=ep)
                # Debug first episode of first checkpoint
                debug = (ep == 0 and step == checkpoints[0][0])
                
                reward, cost, length = evaluate_episode(
                    env=env,
                    model=model,
                    device=device,
                    diffusion_steps=diffusion_steps,
                    max_steps=max_episode_steps,
                    debug=debug
                )
                
                rewards.append(reward)
                costs.append(cost)
                lengths.append(length)
            
            # Compute statistics
            result = {
                'step': step,
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'mean_cost': np.mean(costs),
                'std_cost': np.std(costs),
                'mean_length': np.mean(lengths),
                'all_rewards': rewards,
                'all_costs': costs,
                'all_lengths': lengths
            }
            results.append(result)
            
            print(f"  Step {step}: Reward={result['mean_reward']:.2f}±{result['std_reward']:.2f}, "
                  f"Cost={result['mean_cost']:.2f}±{result['std_cost']:.2f}, "
                  f"Length={result['mean_length']:.1f}")
            
        except Exception as e:
            print(f"  Error evaluating checkpoint at step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    env.close()
    return results


def save_results(results, output_path):
    """Save evaluation results to JSON file."""
    results_serializable = []
    for result in results:
        result_copy = result.copy()
        for key in result_copy:
            if isinstance(result_copy[key], (np.ndarray, np.generic)):
                result_copy[key] = result_copy[key].tolist()
            elif isinstance(result_copy[key], list) and len(result_copy[key]) > 0:
                if isinstance(result_copy[key][0], (np.ndarray, np.generic)):
                    result_copy[key] = [float(x) for x in result_copy[key]]
        results_serializable.append(result_copy)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(results):
    """Print summary table of results."""
    if len(results) == 0:
        print("No results to display.")
        return
    
    print("\n" + "="*80)
    print("V5_HIST EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Step':<12} {'Mean Reward':<20} {'Mean Cost':<20} {'Length':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['step']:<12} "
              f"{result['mean_reward']:>8.2f} ± {result['std_reward']:<8.2f} "
              f"{result['mean_cost']:>8.2f} ± {result['std_cost']:<8.2f} "
              f"{result['mean_length']:>8.1f}")
    
    print("="*80)
    
    # Find best model
    if len(results) > 0:
        best_idx = np.argmax([r['mean_reward'] for r in results])
        best = results[best_idx]
        print(f"\nBest model: Step {best['step']} with reward {best['mean_reward']:.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate V5_hist Flow Matching model checkpoints")
    
    # Task selection
    parser.add_argument("--task", type=str, 
                       default="OfflineSwimmerVelocityGymnasium-v1",
                       choices=["OfflineSwimmerVelocityGymnasium-v1", 
                               "OfflinePointGoal1Gymnasium-v0"],
                       help="Task to evaluate")
    
    # Path configuration
    parser.add_argument("--base_log_dir", type=str,
                       default="~/safe_diff/offline_mpc/logs/merged",
                       help="Base log directory")
    
    parser.add_argument("--experiment_subdir", type=str,
                       default="twinq_fixed_hist",
                       help="Experiment subdirectory path")
    
    parser.add_argument("--seed_dir", type=str, default=None,
                       help="Specific seed directory to evaluate (overrides auto-detection)")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to evaluate per checkpoint")
    
    parser.add_argument("--diffusion_steps", type=int, default=15,
                       help="Number of ODE integration steps")
    
    parser.add_argument("--step_interval", type=int, default=2000,
                       help="Evaluate models at this step interval")
    
    parser.add_argument("--max_episode_steps", type=int, default=1000,
                       help="Maximum steps per episode")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run evaluation on (cpu or cuda)")
    
    # Output
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file for results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Expand home directory
    base_log_dir = osp.expanduser(args.base_log_dir)
    
    # Determine log directory
    if args.seed_dir:
        # Use provided seed directory directly
        run_dir = osp.expanduser(args.seed_dir)
        print(f"Using specified seed directory: {run_dir}")
    else:
        # Auto-detect: Construct full log directory path
        # V5_hist uses algorithm name: ipl_flow_twinq_fm_v5_fixed
        log_dir = osp.join(
            base_log_dir,
            args.task,
            args.experiment_subdir,
            args.task,
            "ipl_flow_twinq_fm_v5_fixed"
        )
        
        # Find seed directories
        run_dirs = glob.glob(osp.join(log_dir, "seed-*"))
        
        if len(run_dirs) == 0:
            print(f"Error: No run directories found in {log_dir}")
            print(f"\nSearching for alternative paths...")
            
            # Try without task repetition
            alt_log_dir = osp.join(
                base_log_dir,
                args.task,
                args.experiment_subdir,
                "ipl_flow_twinq_fm_v5_fixed"
            )
            run_dirs = glob.glob(osp.join(alt_log_dir, "seed-*"))
            
            if len(run_dirs) == 0:
                print(f"Also tried: {alt_log_dir}")
                print("No seed directories found. Please specify --seed_dir explicitly.")
                return
        
        # Use most recent run directory
        run_dirs.sort()
        run_dir = run_dirs[-1]
        print(f"Using run directory: {run_dir}")
    
    # Check if directory exists
    if not osp.exists(run_dir):
        print(f"Error: Directory does not exist: {run_dir}")
        return
    
    # Determine output file
    if args.output_file is None:
        output_file = osp.join(run_dir, f"eval_V5_hist_results_step{args.step_interval}.json")
    else:
        output_file = args.output_file
    
    # Run evaluation
    results = evaluate_all_checkpoints(
        task_name=args.task,
        log_dir=run_dir,
        num_episodes=args.num_episodes,
        diffusion_steps=args.diffusion_steps,
        step_interval=args.step_interval,
        device=args.device,
        max_episode_steps=args.max_episode_steps
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    if len(results) > 0:
        save_results(results, output_file)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
