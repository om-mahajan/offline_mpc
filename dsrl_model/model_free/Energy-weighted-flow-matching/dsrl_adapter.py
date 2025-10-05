import numpy as np
import torch
from torch.utils.data import Dataset
import gymnasium as gym
import dsrl.offline_safety_gymnasium  # Registers DSRL envs


class DSRLSafetyDataset(Dataset):
    """
    Dataset adapter for DSRL offline safety datasets.
    Filters trajectories based on reward and cost quantiles to focus on
    high-reward, low-cost demonstrations for safe imitation learning.
    """
    
    def __init__(self, env_name, num_negative=50, horizon=100, device='cpu'):
        """
        Args:
            env_name: DSRL environment name (e.g., 'OfflinePointGoal1Gymnasium-v0')
            num_negative: Number of non-preferred (negative) trajectories to select (by cost)
            horizon: Fixed length for each trajectory (truncate or pad)
            device: torch device for tensors
        """
        self.env_name = env_name
        self.device = device
        # Load DSRL dataset
        print(f"Loading DSRL dataset: {env_name}")
        # Create env and load dataset (following DSRL API)
        env = gym.make(env_name)
        dataset = env.get_dataset()

        # Extract trajectory information
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        costs = dataset['costs']
        terminals = dataset['terminals']
        timeouts = dataset['timeouts']

        # Split into trajectories
        trajectories = []
        start_idx = 0
        for i in range(len(terminals)):
            if terminals[i] or timeouts[i]:
                traj = {
                    'observations': observations[start_idx:i+1],
                    'actions': actions[start_idx:i+1],
                    'rewards': rewards[start_idx:i+1],
                    'costs': costs[start_idx:i+1],
                    'return': np.sum(rewards[start_idx:i+1]),
                    'cost_sum': np.sum(costs[start_idx:i+1])
                }
                trajectories.append(traj)
                start_idx = i + 1

        print(f"Total trajectories: {len(trajectories)}")

        # Sort trajectories by cost (descending: worst first)
        sorted_trajs = sorted(trajectories, key=lambda x: x['cost_sum'], reverse=True)
        negative_trajs = sorted_trajs[:num_negative]
        union_trajs = sorted_trajs[num_negative:]

        print(f"Negative (non-preferred) trajectories: {len(negative_trajs)}")
        print(f"Union (unlabeled/preferred) trajectories: {len(union_trajs)}")

        # Helper to pad/truncate trajectories to fixed horizon
        def fix_horizon(traj, horizon):
            obs = traj['observations']
            act = traj['actions']
            rew = traj['rewards']
            cost = traj['costs']
            length = len(obs)
            if length >= horizon:
                return {
                    'observations': obs[:horizon],
                    'actions': act[:horizon],
                    'rewards': rew[:horizon],
                    'costs': cost[:horizon]
                }
            else:
                pad_len = horizon - length
                obs_pad = np.pad(obs, ((0, pad_len), (0, 0)), mode='edge')
                act_pad = np.pad(act, ((0, pad_len), (0, 0)), mode='edge')
                rew_pad = np.pad(rew, (0, pad_len), mode='edge')
                cost_pad = np.pad(cost, (0, pad_len), mode='edge')
                return {
                    'observations': obs_pad,
                    'actions': act_pad,
                    'rewards': rew_pad,
                    'costs': cost_pad
                }

        # Process negative and union sets
        self.negative = [fix_horizon(traj, horizon) for traj in negative_trajs]
        self.union = [fix_horizon(traj, horizon) for traj in union_trajs]

        # Stack into arrays: shape [num_traj, horizon, dim]
        self.negative_obs = torch.FloatTensor(np.stack([traj['observations'] for traj in self.negative])).to(device)
        self.negative_act = torch.FloatTensor(np.stack([traj['actions'] for traj in self.negative])).to(device)
        self.negative_rew = torch.FloatTensor(np.stack([traj['rewards'] for traj in self.negative])).to(device)
        self.negative_cost = torch.FloatTensor(np.stack([traj['costs'] for traj in self.negative])).to(device)

        self.union_obs = torch.FloatTensor(np.stack([traj['observations'] for traj in self.union])).to(device)
        self.union_act = torch.FloatTensor(np.stack([traj['actions'] for traj in self.union])).to(device)
        self.union_rew = torch.FloatTensor(np.stack([traj['rewards'] for traj in self.union])).to(device)
        self.union_cost = torch.FloatTensor(np.stack([traj['costs'] for traj in self.union])).to(device)

        # Store dimensions
        self.obs_dim = self.negative_obs.shape[-1]
        self.action_dim = self.negative_act.shape[-1]
        self.horizon = horizon

        print(f"Negative set shape: {self.negative_obs.shape}")
        print(f"Union set shape: {self.union_obs.shape}")
        print(f"Horizon: {self.horizon}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")

    def get_sets(self):
        """Return negative and union sets as dicts of tensors"""
        return {
            'negative': {
                'observations': self.negative_obs,
                'actions': self.negative_act,
                'rewards': self.negative_rew,
                'costs': self.negative_cost
            },
            'union': {
                'observations': self.union_obs,
                'actions': self.union_act,
                'rewards': self.union_rew,
                'costs': self.union_cost
            }
        }
    
    def __len__(self):
        """Return total number of trajectories"""
        return len(self.negative) + len(self.union)
    
    def __getitem__(self, idx):
        """Get a single trajectory (negative or union)"""
        if idx < len(self.negative):
            return {
                'observations': self.negative_obs[idx],
                'actions': self.negative_act[idx],
                'rewards': self.negative_rew[idx],
                'costs': self.negative_cost[idx],
                'is_negative': True
            }
        else:
            union_idx = idx - len(self.negative)
            return {
                'observations': self.union_obs[union_idx],
                'actions': self.union_act[union_idx],
                'rewards': self.union_rew[union_idx],
                'costs': self.union_cost[union_idx],
                'is_negative': False
            }
