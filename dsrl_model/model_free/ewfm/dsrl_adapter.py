import numpy as np
import torch
from torch.utils.data import Dataset
import gymnasium as gym
import dsrl.offline_safety_gymnasium  # Registers DSRL envs


class DSRLSafetyDataset(Dataset):
    """
    Dataset adapter for DSRL offline safety datasets.
    Filters trajectories based on reward and cost:
    - Keep only high-reward trajectories (>= 50% max return)
    - Negative set: high-cost trajectories
    - Union set: remaining high-reward trajectories
    """
    
    def __init__(self, env_name, num_negative=50, horizon=100, device='cpu', high_reward_ratio=0.5, store_full_trajectories=True):
        self.env_name = env_name
        self.device = device
        self.high_reward_ratio = high_reward_ratio
        self.horizon = horizon
        self.store_full_trajectories = store_full_trajectories
        
        print(f"Loading DSRL dataset: {env_name}")
        print(f"Store full trajectories: {store_full_trajectories}")
        env = gym.make(env_name)
        dataset = env.get_dataset()

        # Extract trajectory info
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

        # Keep only high-reward trajectories
        max_return = max(traj['return'] for traj in trajectories)
        reward_threshold = self.high_reward_ratio * max_return
        high_reward_trajs = [traj for traj in trajectories if traj['return'] >= reward_threshold]

        print(f"High-reward trajectories (>= {self.high_reward_ratio*100:.0f}% max): {len(high_reward_trajs)}")

        # Sort by cost
        sorted_trajs = sorted(high_reward_trajs, key=lambda x: x['cost_sum'], reverse=True)
        negative_trajs = sorted_trajs[:num_negative]
        union_trajs = sorted_trajs[num_negative:]

        print(f"Negative (non-preferred, high-cost) trajectories: {len(negative_trajs)}")
        print(f"Union (high-reward) trajectories: {len(union_trajs)}")

        if store_full_trajectories:
            # Store full trajectories as lists (variable length)
            self.negative = negative_trajs
            self.union = union_trajs
            
            # Store dimensions from first trajectory
            self.obs_dim = negative_trajs[0]['observations'].shape[-1]
            self.action_dim = negative_trajs[0]['actions'].shape[-1]
            
            print(f"Negative set: {len(self.negative)} full trajectories")
            print(f"Union set: {len(self.union)} full trajectories")
            print(f"Trajectory lengths vary, will sample chunks during training")
            print(f"Observation dim: {self.obs_dim}")
            print(f"Action dim: {self.action_dim}")
        else:
            # Old behavior: pad/truncate to fixed horizon
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

            # Stack into tensors
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

            print(f"Negative set shape: {self.negative_obs.shape}")
            print(f"Union set shape: {self.union_obs.shape}")
            print(f"Horizon: {self.horizon}")
            print(f"Observation dim: {self.obs_dim}")
            print(f"Action dim: {self.action_dim}")

    def get_sets(self):
        """Return negative and union sets as dicts of tensors or lists"""
        if self.store_full_trajectories:
            # Return lists of trajectories (each with variable length)
            return {
                'negative': self.negative,
                'union': self.union,
                'device': self.device
            }
        else:
            # Return stacked tensors (fixed horizon)
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
        return len(self.negative) + len(self.union)
    
    def __getitem__(self, idx):
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
