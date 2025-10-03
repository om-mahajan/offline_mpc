# DSRL Integration for Energy-Weighted Flow Matching

This directory contains the integration of DSRL offline safety datasets with energy-weighted flow matching for safe imitation learning.

## Changes Made

### 1. New Files Created

#### `dsrl_adapter.py`
- `DSRLSafetyDataset` class that loads DSRL offline safety datasets
- Filters trajectories by reward and cost quantiles to focus on high-reward, low-cost demonstrations
- Provides a dataset interface compatible with the training pipeline
- Tracks statistics: mean reward, mean cost, number of trajectories, etc.

#### `run_dsrl.ps1`
- PowerShell script with example commands for training on DSRL tasks
- Includes examples for OfflinePointGoal1, OfflineCarGoal1, and OfflineAntVelocity

### 2. Modified Files

#### `train_rl.py`
- Added imports for `dsrl`, `safety_gymnasium`, and `DSRLSafetyDataset`
- Created `eval_dsrl_policy()` function to evaluate policies on DSRL environments
  - Tracks both returns and costs during evaluation
  - Prints statistics for both reward and safety violations
- Modified `main()` function to support both D4RL and DSRL datasets
  - Detects DSRL usage via `--use_dsrl` flag or 'Gymnasium' in env name
  - Creates `DSRLDatasetWrapper` to match the D4RL dataset interface
  - Uses DSRL-specific evaluation function when appropriate

#### `utils.py`
- Added three new command-line arguments:
  - `--use_dsrl`: Flag to explicitly use DSRL datasets
  - `--reward_quantile`: Keep trajectories above this reward quantile (default: 0.75)
  - `--cost_quantile`: Keep trajectories below this cost quantile (default: 0.25)

## Usage

### Training on DSRL Tasks

```powershell
# Basic usage - OfflinePointGoal1
python -u train_rl.py \
    --expid dsrl_point_goal_1 \
    --env "OfflinePointGoal1Gymnasium-v0" \
    --seed 0 \
    --reward_quantile 0.75 \
    --cost_quantile 0.25 \
    --schedule OT \
    --device cuda

# Stricter safety filtering
python -u train_rl.py \
    --expid dsrl_car_goal_safe \
    --env "OfflineCarGoal1Gymnasium-v0" \
    --seed 0 \
    --reward_quantile 0.8 \
    --cost_quantile 0.1 \
    --schedule OT \
    --device cuda

# Using PowerShell script
.\run_dsrl.ps1
```

### Available DSRL Tasks

The following DSRL tasks are supported (append `Gymnasium-v0` to the task name):

**Goal-based tasks:**
- `OfflinePointGoal1`, `OfflinePointGoal2`
- `OfflineCarGoal1`, `OfflineCarGoal2`
- `OfflineAntGoal1`, `OfflineAntGoal2`

**Button-based tasks:**
- `OfflinePointButton1`, `OfflinePointButton2`
- `OfflineCarButton1`, `OfflineCarButton2`
- `OfflineAntButton1`, `OfflineAntButton2`

**Push-based tasks:**
- `OfflinePointPush1`, `OfflinePointPush2`
- `OfflineCarPush1`, `OfflineCarPush2`
- `OfflineAntPush1`, `OfflineAntPush2`

**Velocity-based tasks:**
- `OfflineAntVelocity`, `OfflineSwimmerVelocity`
- `OfflineWalkerVelocity`, `OfflineHalfCheetahVelocity`
- `OfflineHopperVelocity`, `OfflineHumanoidVelocity`

### Key Parameters

- **`reward_quantile`** (0-1): Higher values = keep only better performing trajectories
  - 0.75 = keep top 25% by reward
  - 0.9 = keep top 10% by reward
  
- **`cost_quantile`** (0-1): Lower values = keep only safer trajectories
  - 0.25 = keep bottom 25% by cost (safest)
  - 0.5 = keep bottom 50% by cost

- **`schedule`**: Diffusion schedule
  - `OT` = Optimal Transport (recommended for DSRL)
  - `linear` = Linear schedule
  - `cosine` = Cosine schedule

- **`alpha`**: Energy weighting parameter (beta in the paper)
  - Higher values = stronger energy-based filtering
  - Default: 3.0

## Training Stages

The training follows a three-stage process:

1. **Stage 1 (Epochs 0-600)**: Behavior cloning with diffusion model
   - Trains a diffusion policy on filtered safe demonstrations
   - Checkpoint saved: `behavior_ckpt600.pth`

2. **Stage 2 (Epochs 601-1100)**: Q-function training
   - Trains a Q-critic to estimate state-action values
   - Uses sampled actions from diffusion model
   - Checkpoint saved: `critic_ckpt500.pth`

3. **Stage 3 (Epochs 1101-1200)**: Energy-weighted retraining
   - Retrains diffusion model with Q-value based energy weighting
   - Higher Q-values get higher weights in the loss
   - Best model saved: `behavior_best_ckpt.pth`

## Evaluation

During training, the policy is evaluated every 5 epochs during Stage 3:
- Runs 100 episodes in the safety environment
- Tracks both cumulative reward and cumulative cost
- Saves the best model based on reward performance

## Output

Checkpoints are saved to: `./models_rl/<expid>/`
- `behavior_ckpt600.pth` - After BC stage
- `critic_ckpt500.pth` - After Q-training stage
- `behavior_best_ckpt.pth` - Best model during energy-weighted stage

TensorBoard logs are saved to: `./logs/<expid>/`
- Training loss curves
- Evaluation rewards and costs

## Advanced: Using Cost Models as Energy

To integrate SafeTD3-style cost models as energy functions instead of Q-values, you can modify the energy computation in Stage 3 of `train_rl.py`:

```python
# Load cost model from SafeTD3
from dsrl_model.model_free.safetd3 import ExpCostModel

cost_model = ExpCostModel(...).to(args.device)
cost_model.load_state_dict(torch.load('path/to/cost_model.pth'))
cost_model.eval()

# In the training loop, replace Q-based energy with cost-based energy:
with torch.no_grad():
    cost = cost_model(a, s).detach().squeeze()
    e = -cost  # Negative cost = higher weight for safer actions
```

This allows you to combine diffusion models with explicit cost models for enhanced safety.

## Troubleshooting

**Issue: DSRL dataset not found**
- Make sure `dsrl` package is installed: `pip install dsrl`
- Download datasets using: `python -c "import dsrl; dsrl.offline_safety_gymnasium.load_dataset('OfflinePointGoal1Gymnasium-v0')"`

**Issue: Out of memory**
- Reduce batch size in `train()` function (default: 4096)
- Reduce `M` parameter (number of sampled actions, default: 64)
- Use CPU for dataset: set `device='cpu'` in DSRLSafetyDataset

**Issue: No trajectories after filtering**
- Lower `reward_quantile` or increase `cost_quantile`
- Check dataset statistics in the output logs

## References

- Original Energy-Weighted Flow Matching paper
- DSRL: https://github.com/liuzuxin/DSRL
- Safety Gymnasium: https://github.com/PKU-Alignment/safety-gymnasium
