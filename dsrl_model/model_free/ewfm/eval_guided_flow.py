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

from diffusion_SDE.model import ScoreNet, TwinQ


def normalize_observation(mu_obs, std_obs, obs, EP=1e-6):
    """Normalize observations using mean and std"""
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


@torch.no_grad()
def evaluate_episode(env, model, device, norm_fn, diffusion_steps=15, horizon=5, act_dim=None, max_steps=1000, debug=False):
    """
    Evaluate a single episode using the guided flow model.
    Returns: (total_reward, total_cost, episode_length)
    """
    model.eval()
    
    obs, _ = env.reset()
    obs_tensor = torch.as_tensor(norm_fn(obs), dtype=torch.float32, device=device).unsqueeze(0)
    
    total_reward = 0.0
    total_cost = 0.0
    episode_len = 0
    done = False
    
    if act_dim is None:
        act_dim = env.action_space.shape[0]
    
    if debug:
        print(f"\nDebug - Episode start:")
        print(f"  obs shape: {obs.shape}, obs: {obs[:5]}")
        print(f"  obs_tensor shape: {obs_tensor.shape}")
        print(f"  act_dim: {act_dim}, horizon: {horizon}")
    
    while not done and episode_len < max_steps:
        try:
            # Use model.sample() which uses proper RK4 ODE solver
            # Returns: [num_states, sample_per_state, output_dim]
            actions = model.sample(
                states=obs_tensor.cpu().numpy(),
                sample_per_state=1,
                diffusion_steps=diffusion_steps,
                is_numpy=True
            )
            
            # Extract single action: [1, 1, horizon*act_dim] -> [horizon, act_dim]
            traj_flat = actions[0, 0, :]  # shape: [horizon * act_dim]
            traj = traj_flat.reshape(horizon, act_dim)
            action = traj[0]  # First action in trajectory
            
            if debug and episode_len < 3:
                print(f"\nDebug - Step {episode_len}:")
                print(f"  traj_flat shape: {traj_flat.shape}")
                print(f"  traj shape: {traj.shape}")
                print(f"  action shape: {action.shape}")
                print(f"  action: {action}")
                print(f"  action range: [{action.min():.3f}, {action.max():.3f}]")
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if debug and episode_len < 3:
                print(f"  reward: {reward:.4f}, terminated: {terminated}, truncated: {truncated}")
                print(f"  cost: {info.get('cost', 0.0):.4f}")
            
            # Update observation
            obs_tensor = torch.as_tensor(norm_fn(next_obs), dtype=torch.float32, device=device).unsqueeze(0)
            
            # Accumulate metrics
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


def load_model_checkpoint(checkpoint_path, obs_dim, act_dim, horizon, device):
    """
    Load a guided flow model checkpoint.
    Note: Q and V networks were used during TRAINING to guide the loss,
    but are NOT needed during evaluation - the flow model learned the guided policy.
    """
    # Create model architecture
    traj_dim = horizon * act_dim
    
    # Create dummy args object with necessary attributes
    class Args:
        def __init__(self):
            self.device = device
            self.schedule = 'linear'
    
    args = Args()
    
    # Initialize model (Flow Matching mode: marginal_prob_std=None)
    model = ScoreNet(
        input_dim=obs_dim + traj_dim,
        output_dim=traj_dim,
        marginal_prob_std=None,  # Flow Matching mode
        args=args
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


def find_model_checkpoints(log_dir, step_interval=20000):
    """
    Find all flow model checkpoints at specified intervals.
    Returns list of (step_number, checkpoint_path) tuples.
    """
    # Check if there's a torch_save subdirectory
    torch_save_dir = osp.join(log_dir, "torch_save")
    if osp.exists(torch_save_dir):
        search_dir = torch_save_dir
        print(f"Found torch_save directory: {torch_save_dir}")
    else:
        search_dir = log_dir
    
    # Try different possible naming patterns
    patterns = [
        "flow_model_*.pt",
        "flow_*.pt", 
        "*flow*.pt"
    ]
    
    flow_models = []
    for pattern in patterns:
        found = glob.glob(osp.join(search_dir, pattern))
        flow_models.extend(found)
    
    # Remove duplicates
    flow_models = list(set(flow_models))
    
    if not flow_models:
        print(f"Debug: No flow models found in {search_dir}")
        print(f"Debug: Listing all .pt files in directory:")
        all_pt_files = glob.glob(osp.join(search_dir, "*.pt"))
        for f in all_pt_files[:10]:  # Show first 10 files
            print(f"  {osp.basename(f)}")
        return []
    
    checkpoints = []
    for model_path in flow_models:
        # Extract step number from filename
        basename = osp.basename(model_path)
        # Try different patterns: flow_model_XXXXXX.pt, flow_XXXXXX.pt, etc.
        try:
            # Remove .pt extension first
            name_without_ext = basename.replace(".pt", "")
            
            # Try to extract number from the end
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
        print(f"Debug: Found {len(flow_models)} flow models but none match interval {step_interval}")
        print(f"Debug: Sample filenames:")
        for model_path in flow_models[:5]:
            print(f"  {osp.basename(model_path)}")
    
    return checkpoints


def evaluate_all_checkpoints(
    task_name,
    log_dir,
    num_episodes=10,
    diffusion_steps=15,
    horizon=5,
    step_interval=20000,
    device='cpu',
    normalize_obs=False,
    max_episode_steps=1000
):
    """
    Evaluate all checkpoints for a given task.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating GUIDED FLOW task: {task_name}")
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
    print(f"  Horizon: {horizon}")
    print(f"  Diffusion steps: {diffusion_steps}\n")
    
    # Setup normalization (if needed)
    mu_obs, std_obs = None, None
    if normalize_obs:
        # Try to load normalization statistics from log directory
        norm_file = osp.join(log_dir, "norm_stats.json")
        if osp.exists(norm_file):
            with open(norm_file, 'r') as f:
                norm_stats = json.load(f)
                mu_obs = torch.tensor(norm_stats['mu'], dtype=torch.float32)
                std_obs = torch.tensor(norm_stats['std'], dtype=torch.float32)
            print(f"Loaded normalization stats from {norm_file}")
        else:
            print(f"Warning: Normalization requested but {norm_file} not found. Using unnormalized observations.")
    
    norm_fn = lambda obs: normalize_observation(mu_obs, std_obs, obs)
    
    # Find all checkpoints
    checkpoints = find_model_checkpoints(log_dir, step_interval=step_interval)
    
    if len(checkpoints) == 0:
        print(f"\n{'!'*80}")
        print(f"ERROR: No checkpoints found at interval {step_interval} in:")
        print(f"  {log_dir}")
        print(f"\nTrying to diagnose the issue...")
        print(f"{'!'*80}\n")
        
        # Check for torch_save subdirectory
        torch_save_dir = osp.join(log_dir, "torch_save")
        if osp.exists(torch_save_dir):
            search_dir = torch_save_dir
            print(f"Checking torch_save directory: {torch_save_dir}")
        else:
            search_dir = log_dir
            print(f"No torch_save directory found, checking main directory: {log_dir}")
        
        # List all .pt files to help debug
        all_pt_files = glob.glob(osp.join(search_dir, "*.pt"))
        print(f"\nTotal .pt files found: {len(all_pt_files)}")
        
        if len(all_pt_files) > 0:
            print(f"\nShowing all .pt files (up to 20):")
            for i, f in enumerate(all_pt_files[:20]):
                print(f"  [{i+1}] {osp.basename(f)}")
            
            # Try to extract step numbers from all files
            print(f"\nAttempting to extract step numbers:")
            step_info = []
            for f in all_pt_files:
                basename = osp.basename(f)
                name_without_ext = basename.replace(".pt", "")
                parts = name_without_ext.split("_")
                for part in reversed(parts):
                    if part.isdigit():
                        step_info.append((int(part), basename))
                        break
            
            if step_info:
                step_info.sort()
                print(f"\nExtracted steps from all models:")
                for step, name in step_info[:20]:
                    is_interval = "✓" if step % step_interval == 0 else "✗"
                    print(f"  {is_interval} Step {step:>7d}: {name}")
                
                # Find closest matching steps
                matching_steps = [s for s, _ in step_info if s % step_interval == 0]
                if matching_steps:
                    print(f"\nSteps that match interval {step_interval}: {matching_steps}")
                else:
                    print(f"\nNo steps match the interval {step_interval}")
                    print(f"Available steps: {[s for s, _ in step_info[:10]]}")
        else:
            print(f"\nNo .pt files found in directory!")
            print(f"Please verify the log directory path is correct.")
        
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
                horizon=horizon,
                device=device
            )
            
            # Run evaluation episodes
            rewards = []
            costs = []
            lengths = []
            
            for ep in range(num_episodes):
                env.reset(seed=ep)  # Different seed for each episode
                # Enable debug for first episode of first checkpoint
                debug = (ep == 0 and step == checkpoints[0][0])
                reward, cost, length = evaluate_episode(
                    env=env,
                    model=model,
                    device=device,
                    norm_fn=norm_fn,
                    diffusion_steps=diffusion_steps,
                    horizon=horizon,
                    act_dim=act_dim,
                    max_steps=max_episode_steps,
                    debug=debug
                )
                
                rewards.append(reward)
                costs.append(cost)
                lengths.append(length)
            
            # Compute statistics
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            mean_length = np.mean(lengths)
            
            result = {
                'step': step,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_cost': mean_cost,
                'std_cost': std_cost,
                'mean_length': mean_length,
                'all_rewards': rewards,
                'all_costs': costs,
                'all_lengths': lengths
            }
            results.append(result)
            
            print(f"  Step {step}: Reward={mean_reward:.2f}±{std_reward:.2f}, Cost={mean_cost:.2f}±{std_cost:.2f}, Length={mean_length:.1f}")
            
        except Exception as e:
            print(f"  Error evaluating checkpoint at step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    env.close()
    return results


def save_results(results, output_path):
    """Save evaluation results to JSON file"""
    # Convert numpy types to Python native types for JSON serialization
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
    """Print summary table of results"""
    if len(results) == 0:
        print("No results to display.")
        return
    
    print("\n" + "="*80)
    print("GUIDED FLOW EVALUATION SUMMARY")
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
    
    parser = argparse.ArgumentParser(description="Evaluate saved Guided Flow Matching models")
    
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
                       default="IPLV5/twinq",
                       help="Experiment subdirectory path")
    
    parser.add_argument("--seed_dir", type=str, default=None,
                       help="Specific seed directory to evaluate (optional, overrides auto-detection)")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to evaluate per checkpoint")
    
    parser.add_argument("--diffusion_steps", type=int, default=15,
                       help="Number of ODE integration steps")
    
    parser.add_argument("--horizon", type=int, default=5,
                       help="Planning horizon")
    
    parser.add_argument("--step_interval", type=int, default=20000,
                       help="Evaluate models at this step interval")
    
    parser.add_argument("--max_episode_steps", type=int, default=1000,
                       help="Maximum steps per episode")
    
    # Device and normalization
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run evaluation on (cpu or cuda)")
    
    parser.add_argument("--normalize_observation", action="store_true",
                       help="Normalize observations (requires norm_stats.json)")
    
    # Output
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file for results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Expand home directory
    base_log_dir = osp.expanduser(args.base_log_dir)
    
    # Check if specific seed directory is provided
    if args.seed_dir:
        # Use the provided seed directory directly
        run_dir = osp.expanduser(args.seed_dir)
        print(f"Using specified seed directory: {run_dir}")
    else:
        # Auto-detect: Construct full log directory path
        log_dir = osp.join(
            base_log_dir,
            args.task,
            args.experiment_subdir,
            args.task,
            "ipl_flow_twinq_fm"
        )
        
        # Find the specific run directory (should contain torch_save subdirectory with flow_model_*.pt files)
        # Look for subdirectories matching the pattern seed-XXX-*
        run_dirs = glob.glob(osp.join(log_dir, "seed-*"))
        
        if len(run_dirs) == 0:
            print(f"Error: No run directories found in {log_dir}")
            return
        
        # Use the most recent run directory
        run_dirs.sort()
        run_dir = run_dirs[-1]
        
        print(f"Using run directory: {run_dir}")
    
    # Check for torch_save subdirectory
    torch_save_dir = osp.join(run_dir, "torch_save")
    if osp.exists(torch_save_dir):
        log_dir = torch_save_dir
        print(f"Found torch_save directory: {torch_save_dir}")
    else:
        log_dir = run_dir
        print(f"Warning: No torch_save directory found, using: {run_dir}")
    
    # Check if directory exists
    if not osp.exists(log_dir):
        print(f"Error: Log directory does not exist: {log_dir}")
        return
    
    # Determine output file
    if args.output_file is None:
        # Save in the run directory (parent of torch_save)
        output_dir = run_dir
        output_file = osp.join(output_dir, f"eval_guided_results_step{args.step_interval}.json")
    else:
        output_file = args.output_file
    
    # Run evaluation
    results = evaluate_all_checkpoints(
        task_name=args.task,
        log_dir=log_dir,
        num_episodes=args.num_episodes,
        diffusion_steps=args.diffusion_steps,
        horizon=args.horizon,
        step_interval=args.step_interval,
        device=args.device,
        normalize_obs=args.normalize_observation,
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
