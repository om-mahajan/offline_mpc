#!/usr/bin/env python3
"""
Quick test script to verify train_flow.py imports and basic functionality.
Run from the ewfm directory:
    python test_train_flow.py
"""

import sys
import torch

print("Testing imports...")

try:
    # Test basic imports
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    print("  ✓ Basic imports OK")
    
    # Test model imports
    from diffusion_SDE.model import ScoreNet
    print("  ✓ ScoreNet import OK")
    
    from agent.sac_models import DoubleQCritic, SingleV, Discriminator
    print("  ✓ SAC models import OK")
    
    # Test utils imports  
    from utils.utils import soft_update, hard_update
    print("  ✓ Utils import OK")
    
    # Test dataset imports
    from dataset.memory import Memory
    print("  ✓ Memory import OK")
    
    # Test flow matching functions
    from train_flow import psi_t_ot, u_t_ot
    print("  ✓ Flow matching functions import OK")
    
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

print("\nTesting OT flow matching functions...")

# Test psi_t_ot
x0 = torch.randn(32, 8)  # [B, D]
x1 = torch.randn(32, 8)  # [B, D]
t = torch.rand(32)       # [B]
sigma_min = 0.01

x_t = psi_t_ot(x0, x1, t, sigma_min)
assert x_t.shape == (32, 8), f"psi_t_ot output shape mismatch: {x_t.shape}"
print(f"  ✓ psi_t_ot: {x_t.shape}")

# Test u_t_ot
u_t = u_t_ot(x_t, x1, t, sigma_min)
assert u_t.shape == (32, 8), f"u_t_ot output shape mismatch: {u_t.shape}"
print(f"  ✓ u_t_ot: {u_t.shape}")

# Verify at t=1, x_t should be sigma_min * x0 + x1 (close to x1 for small sigma_min)
t_ones = torch.ones(32)
x_t_final = psi_t_ot(x0, x1, t_ones, sigma_min)
# At t=1: sigma_t = 1 - (1 - sigma_min) * 1 = sigma_min, so x_t = sigma_min * x0 + 1 * x1
expected = sigma_min * x0 + x1
assert torch.allclose(x_t_final, expected, atol=1e-5), "x_t at t=1 should equal sigma_min * x0 + x1"
print("  ✓ psi_t_ot(t=1) = sigma_min * x0 + x1")

print("\nTesting model instantiation...")

# Create dummy args
args = SimpleNamespace(device="cpu")

# Test ScoreNet
obs_dim = 16
action_dim = 4
flow_model = ScoreNet(
    input_dim=obs_dim + action_dim,
    output_dim=action_dim,
    marginal_prob_std=None,  # Flow matching mode
    embed_dim=32,
    args=args
)
print(f"  ✓ ScoreNet created: input={obs_dim + action_dim}, output={action_dim}")

# Test forward pass
batch_size = 8
x = torch.randn(batch_size, action_dim)
t = torch.rand(batch_size)
flow_model.condition = torch.randn(batch_size, obs_dim)
v_theta = flow_model(x, t)
assert v_theta.shape == (batch_size, action_dim), f"ScoreNet output shape mismatch: {v_theta.shape}"
flow_model.condition = None
print(f"  ✓ ScoreNet forward: {v_theta.shape}")

# Test critic
full_args = SimpleNamespace(
    device="cpu",
    gamma=0.99,
    method=SimpleNamespace(tanh=False, loss="v0")
)
critic = DoubleQCritic(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_dim=256,
    hidden_depth=2,
    args=full_args
)
print(f"  ✓ DoubleQCritic created")

obs = torch.randn(batch_size, obs_dim)
action = torch.randn(batch_size, action_dim)
q_val = critic(obs, action)
assert q_val.shape == (batch_size, 1), f"Critic output shape mismatch: {q_val.shape}"
print(f"  ✓ DoubleQCritic forward: {q_val.shape}")

# Test value network
value = SingleV(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_dim=256,
    hidden_depth=2,
    args=full_args
)
v_val = value(obs)
assert v_val.shape == (batch_size, 1), f"Value output shape mismatch: {v_val.shape}"
print(f"  ✓ SingleV forward: {v_val.shape}")

# Test discriminator
disc = Discriminator(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_dim=256,
    hidden_depth=2,
    reward_factor=1.0
)
d_val = disc(obs)
assert d_val.shape == (batch_size, 1), f"Discriminator output shape mismatch: {d_val.shape}"
assert (d_val >= 0.05).all() and (d_val <= 0.95).all(), "Discriminator values should be clipped"
print(f"  ✓ Discriminator forward: {d_val.shape}")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)
print("\nTo run full training:")
print("  python train_flow.py env.name=SafetyPointGoal1-v0")
