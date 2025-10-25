"""
Energy-based FilterNet for safety classification using NU (Negative-Unlabeled) learning.
Adapted from ewfm/pubc/models.py for use in SafeDICE framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyFilterNet(nn.Module):
    """
    Energy-based safety classifier that outputs raw logits (energies).
    
    Architecture:
    - Observation encoder: 512 → 512 → 256 → 128 → action_dim*2
    - Combined network: (action_dim*3) → 128 → 128 → 1 (energy)
    
    Output: Energy values (logits) where:
        - High energy → unsafe (negative class)
        - Low energy → safe (positive class)
    
    Used with BCEWithLogitsLoss for NU learning:
        - Label 0: Negative (unsafe) trajectories
        - Label 1: Unlabeled (mixed) trajectories
    """
    
    def __init__(self, obs_dim, action_dim, max_action, bias=True):
        super(EnergyFilterNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Feature extraction from observation
        self.net_obs = nn.Sequential(
            nn.Linear(obs_dim, 512, bias=bias),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=bias),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=bias),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=bias),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2, bias=bias),
        )
        
        # Combine obs embedding + action → produce energy
        self.net = nn.Sequential(
            nn.Linear(action_dim * 3, 128, bias=bias),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=bias),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1, bias=bias),  # Single energy output (logit)
        )
    
    def forward(self, obs, actions):
        """
        Forward pass with separated observation and action inputs.
        
        Args:
            obs: Observations [batch, obs_dim]
            actions: Actions [batch, action_dim]
        
        Returns:
            energy: Raw energy values (logits) [batch, 1]
        """
        # Encode observations
        obs_features = self.net_obs(obs)  # [batch, action_dim*2]
        
        # Concatenate obs features + actions
        x = torch.cat([obs_features, actions], dim=1)  # [batch, action_dim*3]
        
        # Compute energy
        energy = self.net(x)  # [batch, 1]
        
        return energy


class EnergyFilterNetWrapper(nn.Module):
    """
    Wrapper to make EnergyFilterNet compatible with SafeDiceCritic interface.
    Allows calling with concatenated [obs, action] input.
    """
    
    def __init__(self, obs_dim, action_dim, max_action, bias=True):
        super(EnergyFilterNetWrapper, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model = EnergyFilterNet(obs_dim, action_dim, max_action, bias)
    
    def forward(self, state_action):
        """
        Forward pass with concatenated state-action input.
        
        Args:
            state_action: Concatenated [obs, action] [batch, obs_dim + action_dim]
        
        Returns:
            energy: Raw energy values [batch, 1]
        """
        # Split into obs and actions
        obs = state_action[:, :self.obs_dim]
        actions = state_action[:, self.obs_dim:]
        
        # Call underlying model
        return self.model(obs, actions)
    
    def __call__(self, *args, **kwargs):
        """Support both interfaces: model(obs, act) and model(concat([obs, act]))"""
        if len(args) == 2:
            # Called as model(obs, act)
            return self.model(*args, **kwargs)
        elif len(args) == 1:
            # Called as model(concat([obs, act]))
            return self.forward(*args, **kwargs)
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(args)}")


def train_energy_filter_step(model, optimizer, neg_obs, neg_acts, union_obs, union_acts, config):
    """
    Single training step for energy-based filter using NU (Negative-Unlabeled) learning.
    
    Training strategy:
    - Negative samples (unsafe): Label = 0, minimize energy
    - Unlabeled samples (mixed): Label = 1, maximize energy
    
    Loss: BCEWithLogitsLoss (binary cross entropy with logits)
    
    Args:
        model: EnergyFilterNet model
        optimizer: PyTorch optimizer
        neg_obs: Negative observations [N_neg, obs_dim]
        neg_acts: Negative actions [N_neg, action_dim]
        union_obs: Union observations [N_union, obs_dim]
        union_acts: Union actions [N_union, action_dim]
        config: Configuration dict with 'max_grad_norm'
    
    Returns:
        loss: Scalar loss tensor (not .item())
    """
    model.train()
    
    # Forward pass for negative samples
    energy_neg = model(neg_obs, neg_acts)  # [N_neg, 1]
    
    # Forward pass for union samples
    energy_union = model(union_obs, union_acts)  # [N_union, 1]
    
    # Create labels: 0=negative (unsafe), 1=unlabeled (union)
    labels_neg = torch.zeros_like(energy_neg)
    labels_union = torch.ones_like(energy_union)
    
    # Combine energies and labels
    energies = torch.cat([energy_neg, energy_union], dim=0)
    labels = torch.cat([labels_neg, labels_union], dim=0)
    
    # Compute BCEWithLogitsLoss - squeeze to ensure matching shapes
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(energies.squeeze(), labels.squeeze())
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    if 'max_grad_norm' in config and config['max_grad_norm'] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
    
    optimizer.step()
    
    return loss  # Return tensor, not .item()
