"""
Energy-Weighted Flow Matching for DSRL Safe Imitation Learning
"""
import os
import os.path as osp
import random
import sys
import time
import functools
from collections import deque
from copy import deepcopy

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

# DSRL imports
import gymnasium as gym
import dsrl
import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # type: ignore - Registers DSRL environments with gym

# Diffusion imports
from diffusion_SDE.loss import loss_fn as diffusion_loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, QGPO_Critic, update_target

# Local imports
from utils import get_args
from dsrl_adapter import DSRLSafetyDataset
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.utils import get_params_norm

EP = 1e-6

# Default configuration (following Energy-Weighted Flow Matching Algorithm)
default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 10,
    "hidden_sizes": [256, 256],
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "action_repeat": 1,
    "update_tau": 0.005,
    "train_horizon": 5,
    "weight_decay": 0.01,
    "energy_pretrain_iterations": int(5e5),  # 500K gradient updates for energy
    "flow_train_iterations": int(1e6),       # 1M gradient updates for flow
    # Algorithm stages (using gradient updates like SafeTD3):
    # 1. Pretrain energy function (discriminate D^N vs D^U): 500K gradient updates
    # 2. Energy-weighted flow matching: 1M gradient updates
    "evaluation_interval": 5,
}


def normalize_observation(mu_obs, std_obs, obs):
    """Normalize observation using mean and std"""
    if mu_obs is None:
        return obs
    return (obs - mu_obs) / (std_obs + EP)


@torch.no_grad()
def evaluate_policy(eval_env, score_model, device, norm_fn, diffusion_steps=15):
    """Evaluate diffusion policy on safety environment"""
    eval_done = False
    eval_obs, _ = eval_env.reset()
    eval_obs = torch.as_tensor(
        norm_fn(eval_obs), dtype=torch.float32, device=device
    ).unsqueeze(0)
    eval_reward, eval_cost, eval_len = 0.0, 0.0, 0
    
    while not eval_done:
        # Sample action from diffusion model
        act = score_model.select_actions(eval_obs, diffusion_steps=diffusion_steps)
        # select_actions can return: numpy array (single), list of arrays (batch), or tensor
        if isinstance(act, list):
            act_np = act[0]  # Take first element from list
        elif isinstance(act, np.ndarray):
            act_np = act
        else:
            act_np = act.squeeze().cpu().numpy()
        next_obs, reward, terminated, truncated, info = eval_env.step(act_np)
        cost = info.get("cost", 0)
        next_obs = torch.as_tensor(
            norm_fn(next_obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_len += 1
        eval_done = terminated or truncated
        
    return eval_reward, eval_cost, eval_len


def prepare_fake_actions_for_q(score_model, union_obs, args):
    with torch.no_grad():
        score_model.eval()
        batch_size = union_obs.shape[0]
        horizon = union_obs.shape[1]
        
        # Reshape: [num_traj, horizon, obs_dim] -> [num_traj * horizon, obs_dim]
        flat_obs = union_obs.reshape(-1, union_obs.shape[-1])
        
        # Sample M actions per state
        fake_actions = []
        sample_batch_size = 1024
        for i in range(0, len(flat_obs), sample_batch_size):
            batch_obs = flat_obs[i:i+sample_batch_size].to(args.device)
            batch_acts = score_model.sample(
                batch_obs, 
                sample_per_state=args.M, 
                diffusion_steps=args.diffusion_steps,
                is_numpy=False
            )
            fake_actions.append(batch_acts.cpu())
        
        fake_actions = torch.cat(fake_actions, dim=0)
        # Reshape back: [num_traj * horizon, M, act_dim] -> [num_traj, horizon, M, act_dim]
        fake_actions = fake_actions.reshape(batch_size, horizon, args.M, -1)
        
        score_model.train()
        return fake_actions


def pretrain_energy_function(energy_model, optimizer, neg_obs, neg_acts, union_obs, union_acts, args, config):
   
    energy_model.train()
    
    # Flatten trajectories
    horizon, batch_size = neg_obs.shape[0], neg_obs.shape[1]
    neg_flat_obs = neg_obs.reshape(horizon * batch_size, -1)
    neg_flat_acts = neg_acts.reshape(horizon * batch_size, -1)
    union_flat_obs = union_obs.reshape(horizon * batch_size, -1)
    union_flat_acts = union_acts.reshape(horizon * batch_size, -1)
    
    # Compute energy for negative samples (should be high)
    neg_input = torch.cat([neg_flat_obs, neg_flat_acts], dim=-1)
    neg_energy = energy_model(neg_input)  # Raw logits
    
    # Compute energy for union samples (should be low)
    union_input = torch.cat([union_flat_obs, union_flat_acts], dim=-1)
    union_energy = energy_model(union_input)  # Raw logits
    
    # Binary cross-entropy loss:
    # E_η(s,a) outputs probability that sample is from D^N
    # Loss = -[E_{D^N}[log E_η] + E_{D^U}[log(1 - E_η)]]
    neg_loss = -torch.mean(torch.log(torch.sigmoid(neg_energy) + 1e-8))
    union_loss = -torch.mean(torch.log(1 - torch.sigmoid(union_energy) + 1e-8))
    loss = neg_loss + union_loss
    
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(energy_model.parameters(), config["max_grad_norm"])
    optimizer.step()
    
    return loss.item(), neg_loss.item(), union_loss.item()


def compute_energy_weights(energy_model, union_obs, union_acts):
    with torch.no_grad():
        energy_model.eval()
        horizon, batch_size = union_obs.shape[0], union_obs.shape[1]
        flat_obs = union_obs.reshape(horizon * batch_size, -1)
        flat_acts = union_acts.reshape(horizon * batch_size, -1)
        
        # Compute energy (raw logits)
        union_input = torch.cat([flat_obs, flat_acts], dim=-1)
        energies = energy_model(union_input)
        
        # Compute normalized weights using softmax
        # w*(s,a) = exp(E_η*(s,a)) / Σ exp(E_η*(s'))
        weights = torch.softmax(energies.squeeze(), dim=0)
        
        # Reshape back to [horizon, batch]
        weights = weights.reshape(horizon, batch_size)
        
        return weights


def train_energy_weighted_flow(score_model, optimizer, union_obs, union_acts, weights, args, config):
    score_model.train()
    horizon, batch_size = union_obs.shape[0], union_obs.shape[1]
    flat_obs = union_obs.reshape(horizon * batch_size, -1)
    flat_acts = union_acts.reshape(horizon * batch_size, -1)
    flat_weights = weights.reshape(horizon * batch_size)
    
    # Set condition for diffusion model
    score_model.condition = flat_obs
    
    # Compute per-sample diffusion losses (flow matching loss)
    # The loss_fn internally computes: (ν_φ(t,x) - u_{t_0}(x,x_0))^2
    eps = 1e-3
    random_t = torch.rand(flat_acts.shape[0], device=flat_acts.device, dtype=flat_acts.dtype) * (1. - eps) + eps
    z = torch.randn_like(flat_acts)
    alpha_t, std = args.marginal_prob_std_fn(random_t)
    alpha_t, std = alpha_t[:, None], std[:, None]
    perturbed_x = flat_acts * alpha_t + z * std
    score = score_model(perturbed_x, random_t)
    
    # Per-sample losses: sum over action dimensions
    per_sample_losses = torch.sum((score * std + z)**2, dim=1)
    
    # Apply energy weights: w*(s,a) · loss (Algorithm Step 4)
    weighted_loss = torch.mean(flat_weights * per_sample_losses)
    
    optimizer.zero_grad()
    weighted_loss.backward()
    clip_grad_norm_(score_model.parameters(), config["max_grad_norm"])
    optimizer.step()
    
    score_model.condition = None
    return weighted_loss.item()


def train_weighted_policy(policy_model, optimizer, union_obs, union_acts, weights, args, config):
    policy_model.train()
    horizon, batch_size = union_obs.shape[0], union_obs.shape[1]
    flat_obs = union_obs.reshape(horizon * batch_size, -1)
    flat_acts = union_acts.reshape(horizon * batch_size, -1)
    flat_weights = weights.reshape(horizon * batch_size)
    
    # For diffusion models, the policy is the score model itself
    # The behavior cloning loss is the same as flow matching loss
    # But here we use it as policy update
    policy_model.condition = flat_obs
    
    # Weighted negative log-likelihood
    # For diffusion: -log π_ψ(a|s) ≈ diffusion_loss
    nll_loss = diffusion_loss_fn(
        policy_model,
        flat_acts,
        args.marginal_prob_std_fn,
        energy=None,
        alpha=1.0
    )
    
    # Apply energy weights
    weighted_loss = nll_loss  # TODO: Apply per-sample weights
    
    optimizer.zero_grad()
    weighted_loss.backward()
    clip_grad_norm_(policy_model.parameters(), config["max_grad_norm"])
    optimizer.step()
    
    policy_model.condition = None
    return weighted_loss.item()


# Removed old Q-function and energy-weighted diffusion functions
# Now using the algorithm from the paper directly


def sample_trajectory_batch(dataset_splits, batch_size, train_horizon, device):
    """
    Sample trajectory batches from negative and union sets
    Returns: (neg_obs, neg_acts, union_obs, union_acts, union_rewards)
    """
    neg_data = dataset_splits['negative']
    union_data = dataset_splits['union']
    
    # Sample trajectories
    neg_indices = torch.randint(0, len(neg_data['observations']), (batch_size,))
    union_indices = torch.randint(0, len(union_data['observations']), (batch_size,))
    
    # Sample starting positions within trajectories
    max_start = neg_data['observations'].shape[1] - train_horizon
    neg_starts = torch.randint(0, max(1, max_start), (batch_size,))
    union_starts = torch.randint(0, max(1, max_start), (batch_size,))
    
    # Extract chunks
    neg_obs_batch = []
    neg_acts_batch = []
    union_obs_batch = []
    union_acts_batch = []
    union_rew_batch = []
    
    for i in range(batch_size):
        neg_idx, neg_start = neg_indices[i], neg_starts[i]
        union_idx, union_start = union_indices[i], union_starts[i]
        
        neg_obs_batch.append(neg_data['observations'][neg_idx, neg_start:neg_start+train_horizon])
        neg_acts_batch.append(neg_data['actions'][neg_idx, neg_start:neg_start+train_horizon])
        
        union_obs_batch.append(union_data['observations'][union_idx, union_start:union_start+train_horizon])
        union_acts_batch.append(union_data['actions'][union_idx, union_start:union_start+train_horizon])
        union_rew_batch.append(union_data['rewards'][union_idx, union_start:union_start+train_horizon])
    
    return (
        torch.stack(neg_obs_batch).transpose(0, 1).to(device),  # [horizon, batch, dim]
        torch.stack(neg_acts_batch).transpose(0, 1).to(device),
        torch.stack(union_obs_batch).transpose(0, 1).to(device),
        torch.stack(union_acts_batch).transpose(0, 1).to(device),
        torch.stack(union_rew_batch).transpose(0, 1).to(device)
    )


# Energy function E_η (for discriminating D^N vs D^U)
# Simple MLP that outputs logits for binary classification
class EnergyFunction(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))  # Output single logit
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    
    # Setup device
    device_name = "cpu" if args.device == "cpu" else f"{args.device}:{args.device_id}"
    device = torch.device(device_name)
    args.device = device
    
    # Merge configs - convert epochs to steps
    # SafeTD3 uses steps (iterations), not epochs
    # Algorithm 1: Gradient Updates (like SafeTD3/SafeDICE)
    config = {**default_cfg}
    config["train_horizon"] = args.train_horizon
    config["log_freq"] = args.log_freq
    config["save_freq"] = args.save_freq
    config["normalize_observation"] = args.normalize_observation
    
    # Training iterations (gradient updates)
    energy_pretrain_iterations = config["energy_pretrain_iterations"]  # 500K
    flow_train_iterations = config["flow_train_iterations"]            # 1M
    total_iterations = energy_pretrain_iterations + flow_train_iterations  # 1.5M total
    
    print("=" * 60)
    print(args)
    print("=" * 60)
    print(f"  Batch size: {args.batch_size} trajectories × {config['train_horizon']} timesteps")
    print(f"  Transitions per batch: {args.batch_size * config['train_horizon']}")
    print("=" * 60)
    print(f"  Energy Pretraining: {energy_pretrain_iterations:,} gradient updates")
    print(f"  Flow Training: {flow_train_iterations:,} gradient updates")
    print(f"  Total: {total_iterations:,} gradient updates")
    print(f"  Evaluation: Every {args.eval_freq:,} gradient updates (ONLY during flow training)")
    print(f"  Logging: Every {args.log_freq:,} gradient updates")
    print(f"  Checkpoint Saving: Every {args.save_freq:,} gradient updates")
    print("=" * 60)
    
    # Create directories and setup logger (following SafeTD3 style)
    if not os.path.exists(os.path.join("./models_rl", str(args.expid))):
        os.makedirs(os.path.join("./models_rl", str(args.expid)))
    
    # Setup logger (like SafeTD3)
    dict_args = vars(args)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.save_config(dict_args)
    
    # Setup environment (DSRL environments registered via dsrl.offline_safety_gymnasium import)
    eval_env = gym.make(args.task)
    eval_env.reset(seed=args.seed)
    obs_space = eval_env.observation_space
    act_space = eval_env.action_space
    
    # Load and split DSRL dataset
    print("=" * 60)
    print(f"Loading DSRL dataset: {args.task}")
    dsrl_dataset = DSRLSafetyDataset(
        env_name=args.task,
        num_negative=args.num_non_preferred,
        horizon=100,  # Full episode length
        device=device
    )
    dataset_splits = dsrl_dataset.get_sets()
    
    # Normalization (optional)
    mu_obs, std_obs = None, None
    if args.normalize_observation:
        all_obs = torch.cat([
            dataset_splits['negative']['observations'].reshape(-1, dsrl_dataset.obs_dim),
            dataset_splits['union']['observations'].reshape(-1, dsrl_dataset.obs_dim)
        ])
        mu_obs = all_obs.mean(dim=0)
        std_obs = all_obs.std(dim=0)
        print(f"Observation normalization: mu={mu_obs.mean():.3f}, std={std_obs.mean():.3f}")
    
    norm_fn = functools.partial(normalize_observation, mu_obs, std_obs)
    
    # Setup models (following Algorithm 1)
    print("=" * 60)
    print("Initializing models...")
    
    # Marginal probability function for diffusion
    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, 
        schedule=args.schedule, 
        device=device
    )
    args.marginal_prob_std_fn = marginal_prob_std_fn
    
    # Create energy model
    energy_model = EnergyFunction(
        input_dim=obs_space.shape[0] + act_space.shape[0],
        hidden_sizes=config["hidden_sizes"]
    ).to(device)
    
    # Flow network ν_φ (diffusion score model)
    flow_model = ScoreNet(
        input_dim=obs_space.shape[0] + act_space.shape[0],
        output_dim=act_space.shape[0],
        marginal_prob_std=marginal_prob_std_fn,
        args=args
    ).to(device)
    
    # Policy network π_ψ (same architecture as flow model)
    policy_model = ScoreNet(
        input_dim=obs_space.shape[0] + act_space.shape[0],
        output_dim=act_space.shape[0],
        marginal_prob_std=marginal_prob_std_fn,
        args=args
    ).to(device)
    
    # Optimizers
    energy_optimizer = Adam(energy_model.parameters(), lr=args.lr, weight_decay=config["weight_decay"])
    flow_optimizer = Adam(flow_model.parameters(), lr=args.lr, weight_decay=config["weight_decay"])
    policy_optimizer = Adam(policy_model.parameters(), lr=args.lr, weight_decay=config["weight_decay"])
    
    # Training loop following Algorithm 1
    print("\n" + "=" * 60)
    print("Starting training (Algorithm 1: Energy-Weighted Flow Matching)")
    print("=" * 60)
    
    max_reward, max_cost = -float('inf'), float('inf')
    eval_rew_deque = deque(maxlen=config["eval_episode_freq"])
    eval_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_len_deque = deque(maxlen=config["eval_episode_freq"])
    
    # Training variables
    energy_loss = 0.0
    flow_loss = 0.0
    
    start_time = time.time()
    
    # ==================== PHASE 1: ENERGY PRETRAINING ====================
    print(f"\nEnergy Pretraining ({energy_pretrain_iterations:,} gradient updates)")
    print("=" * 60)
    
    pbar = tqdm(range(energy_pretrain_iterations), desc="Phase 1: Energy", unit="grad", dynamic_ncols=True)
    for grad_update in pbar:
        steps = grad_update + 1  # 1-indexed for logging
        
        # Sample batch
        neg_obs, neg_acts, union_obs, union_acts, union_rewards = sample_trajectory_batch(
            dataset_splits, args.batch_size, config['train_horizon'], device
        )
        
        # Energy pretraining: discriminate negative vs union
        energy_loss, neg_loss, union_loss = pretrain_energy_function(
            energy_model, energy_optimizer, 
            neg_obs, neg_acts, union_obs, union_acts, 
            args, config
        )
        
        # Update progress bar
        pbar.set_postfix({'E_loss': f'{energy_loss:.4f}', 'Neg': f'{neg_loss:.4f}', 'Union': f'{union_loss:.4f}'})
        
        # Logging
        if steps % args.log_freq == 0:
            logger.log_tabular("Train/GradUpdates", steps)
            logger.log_tabular("Loss/Energy", energy_loss)
            logger.log_tabular("Loss/Flow", 0.0)
            logger.log_tabular("Norm/energy", get_params_norm(energy_model.parameters(), grads=False))
            logger.log_tabular("Norm/flow", get_params_norm(flow_model.parameters(), grads=False))
            elapsed = time.time() - start_time
            logger.log_tabular("Time/GradUpdatesPerSec", steps / elapsed)
            logger.log_tabular("Time/TotalMinutes", elapsed / 60)
            logger.dump_tabular()
        
        # Checkpoint saving
        if steps % args.save_freq == 0:
            logger.torch_save(itr=steps, torch_saver_elements=energy_model, prefix="energy")
            logger.torch_save(itr=steps, torch_saver_elements=flow_model, prefix="flow")
    
    print(f"\nenergy func training done: {energy_pretrain_iterations:,} gradient updates | Loss: {energy_loss:.4f}")
    
    # ==================== PHASE 2: FLOW TRAINING ====================
    print(f"\nFlow Training ({flow_train_iterations:,} gradient updates)")
    print("=" * 60)
    
    pbar = tqdm(range(flow_train_iterations), desc="Phase 2: Flow", unit="grad", dynamic_ncols=True)
    for grad_update in pbar:
        steps = energy_pretrain_iterations + grad_update + 1  # Continue counting from Phase 1
        
        # Sample batch
        neg_obs, neg_acts, union_obs, union_acts, union_rewards = sample_trajectory_batch(
            dataset_splits, args.batch_size, config['train_horizon'], device
        )
        
        # Algorithm Step 2: Compute weights w*(s,a) for D^U using current energy model
        weights = compute_energy_weights(energy_model, union_obs, union_acts)
        
        # Algorithm Step 4: Update flow network ν_φ with energy-weighted flow matching
        flow_loss = train_energy_weighted_flow(
            flow_model, flow_optimizer,
            union_obs, union_acts, weights,  # Pass computed weights, not energy_model
            args, config
        )
        
        # Update progress bar
        pbar.set_postfix({'F_loss': f'{flow_loss:.4f}'})
        
        # Evaluation
        if args.use_eval and steps % args.eval_freq == 0:
            eval_start_time = time.time()
            for eval_id in range(config['eval_episode_freq']):
                eval_reward, eval_cost, eval_len = evaluate_policy(
                    eval_env, flow_model, device, norm_fn, args.diffusion_steps
                )
                eval_rew_deque.append(eval_reward)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
            
            mean_rew = np.mean(eval_rew_deque)
            mean_cost = np.mean(eval_cost_deque)
            print(f"\n[Eval @ {steps:,}] Reward: {mean_rew:.2f} | Cost: {mean_cost:.2f}")
            
            if mean_rew > max_reward:
                max_reward = mean_rew
                logger.torch_save(itr=steps, torch_saver_elements=flow_model, prefix="flow_best")
        
        # Logging
        if steps % args.log_freq == 0:
            logger.log_tabular("Train/GradUpdates", steps)
            logger.log_tabular("Loss/Energy", 0.0)
            logger.log_tabular("Loss/Flow", flow_loss)
            logger.log_tabular("Norm/energy", get_params_norm(energy_model.parameters(), grads=False))
            logger.log_tabular("Norm/flow", get_params_norm(flow_model.parameters(), grads=False))
            
            if args.use_eval and len(eval_rew_deque) > 0:
                logger.log_tabular("Metrics/EvalEpRet", np.mean(eval_rew_deque))
                logger.log_tabular("Metrics/EvalEpCost", np.mean(eval_cost_deque))
                logger.log_tabular("Metrics/EvalEpLen", np.mean(eval_len_deque))
            
            elapsed = time.time() - start_time
            logger.log_tabular("Time/GradUpdatesPerSec", steps / elapsed)
            logger.log_tabular("Time/TotalMinutes", elapsed / 60)
            logger.dump_tabular()
        
        # Checkpoint saving
        if steps % args.save_freq == 0:
            logger.torch_save(itr=steps, torch_saver_elements=energy_model, prefix="energy")
            logger.torch_save(itr=steps, torch_saver_elements=flow_model, prefix="flow")
    
    print(f"\nPhase 2 Complete: {flow_train_iterations:,} gradient updates | Loss: {flow_loss:.4f}")
    
    # Compute final totals
    total_grad_updates = energy_pretrain_iterations + flow_train_iterations
    

    
    # Final save (like SafeTD3)
    total_grad_updates = energy_pretrain_iterations + flow_train_iterations
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Total gradient updates: {total_grad_updates:,}")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")
    print("=" * 60)
    print("Saving final models...")
    
    logger.torch_save(itr=total_grad_updates, torch_saver_elements=energy_model, prefix="energy")
    logger.torch_save(itr=total_grad_updates, torch_saver_elements=flow_model, prefix="flow")
    # NOTE: Policy not trained yet, commented out
    # logger.torch_save(itr=env_steps, torch_saver_elements=policy_model, prefix="policy")
    
    # Save normalization stats if used
    if config["normalize_observation"] and mu_obs is not None:
        logger.save_state(
            state_dict={"mu_obs": mu_obs, "std_obs": std_obs}, dirname="norm"
        )
    
    print("All models saved successfully!")
    logger.close()


if __name__ == "__main__":
    args = get_args()
    
    # Setup logging directory (like SafeTD3)
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = "energy_flow_matching"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    # Convert to forward slashes for logger compatibility
    args.log_dir = args.log_dir.replace("\\", "/")
    
    # Redirect output to files (like SafeTD3)
    if not args.write_terminal:
        terminal_log_name = f"seed{args.seed}_terminal.log"
        error_log_name = f"seed{args.seed}_error.log"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(args.log_dir, terminal_log_name),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(args.log_dir, error_log_name),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args)
    else:
        main(args)
