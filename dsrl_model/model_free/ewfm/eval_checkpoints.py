#!/usr/bin/env python3
"""
Evaluation script for dual_flow_contrastive checkpoints.

Evaluates all checkpoints saved in a directory across N episodes,
saves results to CSV with mean ± std statistics.

Usage:
    python eval_checkpoints.py --log_dir /path/to/checkpoints --task OfflinePointGoal1Gymnasium-v0 --num_episodes 10
"""

import os
import sys
import pickle
import glob
import csv
import argparse
import time
from typing import Tuple, Optional, List, Dict

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm

import gymnasium as gym

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import dsrl
import dsrl.offline_safety_gymnasium

from diffusion_SDE.model_jax import FlaxScoreNet
from utils.train_state import FlowTrainState, create_flow_train_state


# =============================================================================
# Core evaluation functions (copied from dual_flow_constrastive.py for speed)
# =============================================================================


@partial(jax.jit, static_argnames=("num_steps", "traj_dim"))
def euler_sample_flow(
    flow_state: FlowTrainState,
    obs: jnp.ndarray,        # [B, obs_dim] or [obs_dim]
    rng: jax.random.PRNGKey,
    num_steps: int = 20,
    traj_dim: int = 60,
) -> jnp.ndarray:
    """
    Sample trajectory from flow model using Euler integration.
    Integrates from t=1 (noise) to t=0 (data).
    """
    single_obs = obs.ndim == 1
    if single_obs:
        obs = obs[None, :]  # [1, obs_dim]
    
    B = obs.shape[0]
    x_t = jax.random.normal(rng, (B, traj_dim))
    dt = 1.0 / num_steps
    
    def euler_step(carry, t_val):
        x_t = carry
        t = jnp.full((B,), t_val)
        v = flow_state.apply_fn(
            flow_state.params, x_t, t, condition=obs, train=False
        )
        x_next = x_t - dt * v
        return x_next, None
    
    t_vals = jnp.linspace(1.0, dt, num_steps)
    x_final, _ = jax.lax.scan(euler_step, x_t, t_vals)
    
    if single_obs:
        x_final = x_final[0]
    
    return x_final


class ObsHistoryBuffer:
    """Maintains observation history for evaluation."""
    
    def __init__(self, hist_len: int, obs_dim: int):
        self.hist_len = hist_len
        self.obs_dim = obs_dim
        self.buffer = None
    
    def reset(self, first_obs: np.ndarray):
        """Reset buffer with first observation (repeated hist_len times)."""
        self.buffer = np.tile(first_obs, (self.hist_len, 1))
    
    def add(self, obs: np.ndarray):
        """Add new observation and shift buffer."""
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs
    
    def get(self) -> np.ndarray:
        """Get flattened observation history."""
        return self.buffer.flatten()


def make_action_fn(
    flow_state: FlowTrainState,
    mu_obs: Optional[jnp.ndarray],
    std_obs: Optional[jnp.ndarray],
    traj_dim: int,
    act_dim: int,
    obs_hist_len: int = 1,
    num_steps: int = 20,
):
    """Create a JIT-compiled action function from flow state."""
    # Tile normalization stats to match observation history dimension
    if mu_obs is not None and obs_hist_len > 1:
        mu_obs_tiled = jnp.tile(mu_obs, obs_hist_len)
        std_obs_tiled = jnp.tile(std_obs, obs_hist_len)
    else:
        mu_obs_tiled = mu_obs
        std_obs_tiled = std_obs
    
    @jax.jit
    def action_fn(obs_flat: jnp.ndarray, rng: jax.random.PRNGKey):
        # Normalize observation if needed
        if mu_obs_tiled is not None:
            obs_norm = (obs_flat - mu_obs_tiled) / (std_obs_tiled + 1e-6)
        else:
            obs_norm = obs_flat
        
        rng, sample_rng = jax.random.split(rng)
        
        # Sample trajectory from flow
        traj = euler_sample_flow(
            flow_state,
            obs_norm,
            sample_rng,
            num_steps=num_steps,
            traj_dim=traj_dim,
        )
        
        # Return first action
        action = traj[:act_dim]
        action = jnp.clip(action, -1.0, 1.0)
        
        return action, rng
    
    return action_fn


def evaluate_single_episode(
    eval_env,
    action_fn,
    rng: jax.random.PRNGKey,
    obs_hist_len: int,
    base_obs_dim: int,
    seed: int,
) -> Tuple[float, float, int, jax.random.PRNGKey]:
    """Evaluate a single episode."""
    obs, _ = eval_env.reset(seed=seed)
    obs = np.asarray(obs)
    
    obs_buffer = ObsHistoryBuffer(obs_hist_len, base_obs_dim)
    obs_buffer.reset(obs)
    
    total_reward, total_cost, total_len = 0.0, 0.0, 0
    done = False
    
    while not done:
        obs_flat = obs_buffer.get()
        obs_jax = jnp.asarray(obs_flat)
        
        action, rng = action_fn(obs_jax, rng)
        action_np = np.asarray(action)
        
        next_obs, reward, terminated, truncated, info = eval_env.step(action_np)
        obs_buffer.add(np.asarray(next_obs))
        
        total_reward += reward
        total_cost += info.get("cost", 0.0)
        total_len += 1
        done = terminated or truncated
    
    return total_reward, total_cost, total_len, rng


def evaluate_checkpoint(
    ckpt_path: str,
    env,
    flow_model: FlaxScoreNet,
    obs_dim: int,
    traj_dim: int,
    act_dim: int,
    base_obs_dim: int,
    obs_hist_len: int,
    diffusion_steps: int,
    num_episodes: int,
    base_seed: int,
    rng: jax.random.PRNGKey,
    eval_neg: bool = False,
) -> Tuple[Dict, jax.random.PRNGKey]:
    """
    Evaluate a single checkpoint across multiple episodes.
    
    Returns dict with results and updated rng.
    """
    # Load checkpoint
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    
    step = ckpt["step"]
    mu_obs = ckpt["mu_obs"]
    std_obs = ckpt["std_obs"]
    
    results = {"step": step}
    
    # Evaluate positive flow
    rng, init_rng = jax.random.split(rng)
    flow_state = create_flow_train_state(
        init_rng, flow_model, obs_dim, traj_dim,
        learning_rate=1e-4, max_grad_norm=1.0
    )
    flow_state = flow_state.replace(params=ckpt["flow_pos"])
    
    action_fn = make_action_fn(
        flow_state, mu_obs, std_obs, traj_dim, act_dim,
        obs_hist_len, diffusion_steps
    )
    
    # Warm up JIT
    dummy_obs = jnp.zeros(obs_dim)
    rng, warmup_rng = jax.random.split(rng)
    _, _ = action_fn(dummy_obs, warmup_rng)
    
    # Run episodes
    rewards_pos, costs_pos, lengths_pos = [], [], []
    for ep in range(num_episodes):
        r, c, l, rng = evaluate_single_episode(
            env, action_fn, rng, obs_hist_len, base_obs_dim, base_seed + ep
        )
        rewards_pos.append(r)
        costs_pos.append(c)
        lengths_pos.append(l)
    
    results["pos_mean_reward"] = np.mean(rewards_pos)
    results["pos_std_reward"] = np.std(rewards_pos)
    results["pos_mean_cost"] = np.mean(costs_pos)
    results["pos_std_cost"] = np.std(costs_pos)
    results["pos_mean_length"] = np.mean(lengths_pos)
    results["pos_std_length"] = np.std(lengths_pos)
    
    # Optionally evaluate negative flow
    if eval_neg:
        rng, init_rng = jax.random.split(rng)
        flow_state_neg = create_flow_train_state(
            init_rng, flow_model, obs_dim, traj_dim,
            learning_rate=1e-4, max_grad_norm=1.0
        )
        flow_state_neg = flow_state_neg.replace(params=ckpt["flow_neg"])
        
        action_fn_neg = make_action_fn(
            flow_state_neg, mu_obs, std_obs, traj_dim, act_dim,
            obs_hist_len, diffusion_steps
        )
        
        # Warm up JIT
        rng, warmup_rng = jax.random.split(rng)
        _, _ = action_fn_neg(dummy_obs, warmup_rng)
        
        rewards_neg, costs_neg, lengths_neg = [], [], []
        for ep in range(num_episodes):
            r, c, l, rng = evaluate_single_episode(
                env, action_fn_neg, rng, obs_hist_len, base_obs_dim, base_seed + ep
            )
            rewards_neg.append(r)
            costs_neg.append(c)
            lengths_neg.append(l)
        
        results["neg_mean_reward"] = np.mean(rewards_neg)
        results["neg_std_reward"] = np.std(rewards_neg)
        results["neg_mean_cost"] = np.mean(costs_neg)
        results["neg_std_cost"] = np.std(costs_neg)
        results["neg_mean_length"] = np.mean(lengths_neg)
        results["neg_std_length"] = np.std(lengths_neg)
    
    return results, rng


def find_checkpoints(log_dir: str) -> List[str]:
    """Find all checkpoint files sorted by step number."""
    pattern = os.path.join(log_dir, "ckpt_step_*.pkl")
    ckpt_files = glob.glob(pattern)
    
    if not ckpt_files:
        raise ValueError(f"No checkpoints found matching {pattern}")
    
    # Sort by step number
    def get_step(f):
        basename = os.path.basename(f)
        return int(basename.replace("ckpt_step_", "").replace(".pkl", ""))
    
    ckpt_files = sorted(ckpt_files, key=get_step)
    return ckpt_files


def save_results_csv(results: List[Dict], output_path: str, eval_neg: bool):
    """Save evaluation results to CSV."""
    if not results:
        return
    
    # Define columns
    if eval_neg:
        fieldnames = [
            "step",
            "pos_mean_reward", "pos_std_reward",
            "pos_mean_cost", "pos_std_cost",
            "pos_mean_length", "pos_std_length",
            "neg_mean_reward", "neg_std_reward",
            "neg_mean_cost", "neg_std_cost",
            "neg_mean_length", "neg_std_length",
        ]
    else:
        fieldnames = [
            "step",
            "pos_mean_reward", "pos_std_reward",
            "pos_mean_cost", "pos_std_cost",
            "pos_mean_length", "pos_std_length",
        ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[Dict], eval_neg: bool):
    """Print summary table to console."""
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    
    if eval_neg:
        header = f"{'Step':>10} | {'Pos Reward':>18} | {'Pos Cost':>18} | {'Neg Reward':>18} | {'Neg Cost':>18}"
        print(header)
        print("-" * 100)
        
        for r in results:
            pos_rew = f"{r['pos_mean_reward']:.2f} ± {r['pos_std_reward']:.2f}"
            pos_cost = f"{r['pos_mean_cost']:.2f} ± {r['pos_std_cost']:.2f}"
            neg_rew = f"{r['neg_mean_reward']:.2f} ± {r['neg_std_reward']:.2f}"
            neg_cost = f"{r['neg_mean_cost']:.2f} ± {r['neg_std_cost']:.2f}"
            print(f"{r['step']:>10} | {pos_rew:>18} | {pos_cost:>18} | {neg_rew:>18} | {neg_cost:>18}")
    else:
        header = f"{'Step':>10} | {'Reward':>20} | {'Cost':>20} | {'Length':>15}"
        print(header)
        print("-" * 80)
        
        for r in results:
            rew = f"{r['pos_mean_reward']:.2f} ± {r['pos_std_reward']:.2f}"
            cost = f"{r['pos_mean_cost']:.2f} ± {r['pos_std_cost']:.2f}"
            length = f"{r['pos_mean_length']:.1f} ± {r['pos_std_length']:.1f}"
            print(f"{r['step']:>10} | {rew:>20} | {cost:>20} | {length:>15}")
    
    print("=" * 100)


def main(args):
    """Main evaluation loop."""
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    
    # Find checkpoints
    ckpt_files = find_checkpoints(args.log_dir)
    print(f"\nFound {len(ckpt_files)} checkpoints in {args.log_dir}")
    
    # Create environment
    print(f"Creating environment: {args.task}")
    env = gym.make(args.task)
    env.reset(seed=args.seed)
    
    # Get dimensions
    base_obs_dim = env.observation_space.shape[0]
    obs_dim = base_obs_dim * args.obs_hist_len
    act_dim = env.action_space.shape[0]
    traj_dim = args.train_horizon * act_dim
    
    print(f"\nDimensions: obs_dim={obs_dim}, act_dim={act_dim}, traj_dim={traj_dim}")
    print(f"Observation history length: {args.obs_hist_len}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Episodes per checkpoint: {args.num_episodes}")
    print(f"Evaluate negative flow: {args.eval_neg}")
    
    # Initialize model
    flow_model = FlaxScoreNet(output_dim=traj_dim)
    rng = jax.random.PRNGKey(args.seed)
    
    # Evaluate all checkpoints
    results = []
    start_time = time.time()
    
    for ckpt_path in tqdm(ckpt_files, desc="Evaluating checkpoints"):
        result, rng = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            env=env,
            flow_model=flow_model,
            obs_dim=obs_dim,
            traj_dim=traj_dim,
            act_dim=act_dim,
            base_obs_dim=base_obs_dim,
            obs_hist_len=args.obs_hist_len,
            diffusion_steps=args.diffusion_steps,
            num_episodes=args.num_episodes,
            base_seed=args.seed,
            rng=rng,
            eval_neg=args.eval_neg,
        )
        results.append(result)
        
        # Print progress
        step = result["step"]
        pos_rew = result["pos_mean_reward"]
        pos_cost = result["pos_mean_cost"]
        tqdm.write(f"  Step {step}: Reward={pos_rew:.2f}, Cost={pos_cost:.2f}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal evaluation time: {elapsed:.1f}s ({elapsed/len(ckpt_files):.1f}s per checkpoint)")
    
    # Save and print results
    output_path = os.path.join(args.log_dir, "eval_results.csv")
    save_results_csv(results, output_path, args.eval_neg)
    print_summary(results, args.eval_neg)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate dual flow checkpoints across multiple episodes"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing ckpt_step_*.pkl checkpoint files"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="OfflineSwimmerVelocityGymnasium-v1",
        help="DSRL task name"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per checkpoint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for evaluation"
    )
    parser.add_argument(
        "--train_horizon",
        type=int,
        default=15,
        help="Training horizon (must match checkpoint)"
    )
    parser.add_argument(
        "--obs_hist_len",
        type=int,
        default=4,
        help="Observation history length (must match checkpoint)"
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=15,
        help="Number of Euler integration steps for sampling"
    )
    parser.add_argument(
        "--eval_neg",
        action="store_true",
        default=False,
        help="Also evaluate the negative flow model"
    )
    
    args = parser.parse_args()
    main(args)
