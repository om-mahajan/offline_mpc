import numpy as np
import torch
from torch.utils.data import Dataset
import dsrl


class DSRLSafetyDataset(Dataset):
    """
    Dataset adapter for DSRL offline safety datasets.
    Filters trajectories based on reward and cost quantiles to focus on
    high-reward, low-cost demonstrations for safe imitation learning.
    """
    
    def __init__(self, env_name, reward_quantile=0.75, cost_quantile=0.25, device='cpu'):
        """
        Args:
            env_name: DSRL environment name (e.g., 'OfflinePointGoal1Gymnasium-v0')
            reward_quantile: Keep trajectories above this reward quantile (higher = better)
            cost_quantile: Keep trajectories below this cost quantile (lower = safer)
            device: torch device for tensors
        """
        self.env_name = env_name
        self.device = device
        
        # Load DSRL dataset
        print(f"Loading DSRL dataset: {env_name}")
        dataset = dsrl.offline_safety_gymnasium.load_dataset(env_name)
        
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
        
        # Filter by reward and cost quantiles
        returns = np.array([traj['return'] for traj in trajectories])
        cost_sums = np.array([traj['cost_sum'] for traj in trajectories])
        
        reward_threshold = np.quantile(returns, reward_quantile)
        cost_threshold = np.quantile(cost_sums, cost_quantile)
        
        filtered_trajectories = [
            traj for traj in trajectories
            if traj['return'] >= reward_threshold and traj['cost_sum'] <= cost_threshold
        ]
        
        print(f"Filtered to {len(filtered_trajectories)} trajectories")
        print(f"  Reward threshold (>= {reward_quantile} quantile): {reward_threshold:.2f}")
        print(f"  Cost threshold (<= {cost_quantile} quantile): {cost_threshold:.2f}")
        
        # Flatten filtered trajectories
        self.observations = []
        self.actions = []
        self.rewards = []
        self.costs = []
        
        for traj in filtered_trajectories:
            self.observations.append(traj['observations'])
            self.actions.append(traj['actions'])
            self.rewards.append(traj['rewards'])
            self.costs.append(traj['costs'])
        
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.costs = np.concatenate(self.costs, axis=0)
        
        # Convert to torch tensors
        self.observations = torch.FloatTensor(self.observations).to(device)
        self.actions = torch.FloatTensor(self.actions).to(device)
        self.rewards = torch.FloatTensor(self.rewards).to(device)
        self.costs = torch.FloatTensor(self.costs).to(device)
        
        # Store dimensions
        self.obs_dim = self.observations.shape[1]
        self.action_dim = self.actions.shape[1]
        
        print(f"Dataset size: {len(self)} transitions")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Mean reward: {self.rewards.mean().item():.2f}")
        print(f"  Mean cost: {self.costs.mean().item():.2f}")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observations': self.observations[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'costs': self.costs[idx]
        }
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = torch.randint(0, len(self), (batch_size,))
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'costs': self.costs[indices]
        }
