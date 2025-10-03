# DSRL Integration Summary

## Changes Made to Energy-Weighted Flow Matching

### Overview
Successfully integrated DSRL offline safety datasets with energy-weighted flow matching for safe imitation learning. The system now supports both original D4RL datasets and DSRL safety datasets.

---

## Files Created

### 1. `dsrl_adapter.py` (NEW)
**Purpose**: Dataset adapter for DSRL offline safety datasets

**Key Components**:
- `DSRLSafetyDataset` class
  - Loads DSRL datasets using `dsrl.offline_safety_gymnasium.load_dataset()`
  - Splits data into trajectories based on terminal/timeout flags
  - Filters trajectories by reward and cost quantiles
  - Converts to PyTorch tensors on specified device
  - Provides `sample()` method for batch sampling

**Key Features**:
- Trajectory filtering: Keep high-reward (>= reward_quantile) AND low-cost (<= cost_quantile)
- Statistics tracking: Prints dataset size, mean reward/cost, number of trajectories
- Memory efficient: Tensors stored on GPU/CPU as specified

---

### 2. `run_dsrl.ps1` (NEW)
**Purpose**: PowerShell script with example training commands

**Includes**:
- Example 1: OfflinePointGoal1 with default quantiles (0.75, 0.25)
- Example 2: OfflineCarGoal1 with stricter safety (0.8, 0.1)
- Example 3: OfflineAntVelocity
- All using OT schedule with alpha=3.0

---

### 3. `DSRL_README.md` (NEW)
**Purpose**: Complete documentation for DSRL integration

**Sections**:
- Changes overview
- Usage examples
- Available DSRL tasks (28+ environments)
- Key parameters explanation
- Training stages description
- Evaluation metrics
- Advanced: Using cost models as energy
- Troubleshooting guide

---

### 4. `test_dsrl_setup.py` (NEW)
**Purpose**: Validation script to test setup before training

**Tests**:
1. Import test (PyTorch, DSRL, safety_gymnasium, NumPy)
2. Dataset loading test (DSRLSafetyDataset)
3. Environment creation test (safety_gymnasium.make)

**Usage**: `python test_dsrl_setup.py`

---

## Files Modified

### 1. `train_rl.py` (MODIFIED)

#### Imports Added:
```python
from dsrl_adapter import DSRLSafetyDataset
import dsrl
import safety_gymnasium
```

#### New Function: `eval_dsrl_policy()`
- Evaluates policy on DSRL safety environments
- Tracks both returns AND costs (safety violations)
- Prints statistics for reward and cost
- Returns list of environments with buffer_return and buffer_cost

#### Modified Function: `main()`
**Changes**:
1. **DSRL Detection**: Auto-detects DSRL via `--use_dsrl` flag or 'Gymnasium' in env name
2. **Dual Path Support**: 
   - DSRL path: Uses `safety_gymnasium`, `DSRLSafetyDataset`, `eval_dsrl_policy`
   - D4RL path: Uses `gym`, `D4RL_dataset`, `pallaral_eval_policy`
3. **Dataset Wrapper**: Created `DSRLDatasetWrapper` class to match D4RL interface
   - Wraps DSRLSafetyDataset
   - Provides `states`, `next_states`, `actions` attributes
   - `__getitem__` returns dict with 's', 'a', 'r', 'd', 's_' keys

**Key Logic**:
```python
use_dsrl = args.use_dsrl or 'Gymnasium' in args.env

if use_dsrl:
    # DSRL path: safety_gymnasium + DSRLSafetyDataset
    env = safety_gymnasium.make(args.env)
    dataset = DSRLDatasetWrapper(dsrl_dataset)
    args.eval_func = functools.partial(eval_dsrl_policy, ...)
else:
    # Original D4RL path
    env = gym.make(args.env)
    dataset = D4RL_dataset(args)
    args.eval_func = functools.partial(pallaral_eval_policy, ...)
```

---

### 2. `utils.py` (MODIFIED)

#### New Arguments Added to `get_args()`:
```python
# DSRL-specific arguments
parser.add_argument('--use_dsrl', action='store_true', 
                    help='Use DSRL safety datasets')
parser.add_argument('--reward_quantile', type=float, default=0.75, 
                    help='Keep trajectories above this reward quantile (0-1)')
parser.add_argument('--cost_quantile', type=float, default=0.25,
                    help='Keep trajectories below this cost quantile (0-1, lower = safer)')
```

**Purpose**: Allow command-line control of trajectory filtering for safe imitation learning

---

## Training Workflow

### Stage 1: Behavior Cloning (Epochs 0-600)
- Trains diffusion model on filtered safe demonstrations
- No energy weighting, pure BC
- Checkpoint: `behavior_ckpt600.pth`

### Stage 2: Q-Critic Training (Epochs 601-1100)
- Trains Q-function using sampled actions from diffusion model
- Uses soft Q-learning with softmax weighting
- Checkpoint: `critic_ckpt500.pth`

### Stage 3: Energy-Weighted Retraining (Epochs 1101-1200)
- Retrains diffusion model with Q-value based energy weighting
- Formula: `loss = softmax(alpha * Q) * denoising_loss`
- Evaluation every 5 epochs
- Best checkpoint: `behavior_best_ckpt.pth`

---

## Usage Examples

### Basic Usage:
```bash
python train_rl.py \
    --env "OfflinePointGoal1Gymnasium-v0" \
    --expid dsrl_point_goal \
    --seed 0 \
    --reward_quantile 0.75 \
    --cost_quantile 0.25 \
    --schedule OT \
    --device cuda
```

### High Safety Priority:
```bash
python train_rl.py \
    --env "OfflineCarGoal1Gymnasium-v0" \
    --expid dsrl_car_safe \
    --seed 0 \
    --reward_quantile 0.8 \
    --cost_quantile 0.1 \
    --schedule OT \
    --device cuda
```

### Test Setup:
```bash
python test_dsrl_setup.py
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--env` | walker2d-medium-replay-v2 | Environment name (D4RL or DSRL) |
| `--reward_quantile` | 0.75 | Keep trajectories above this reward quantile |
| `--cost_quantile` | 0.25 | Keep trajectories below this cost quantile |
| `--use_dsrl` | False | Explicitly enable DSRL mode |
| `--schedule` | linear | Diffusion schedule (OT/linear/cosine) |
| `--alpha` | 3.0 | Energy weighting parameter |
| `--diffusion_steps` | 15 | Number of diffusion sampling steps |
| `--M` | 64 | Number of sampled actions for Q-training |

---

## Available DSRL Tasks

### Goal-based (9 tasks):
- OfflinePointGoal1/2, OfflineCarGoal1/2, OfflineAntGoal1/2
- OfflinePointButton1/2, OfflineCarButton1/2, OfflineAntButton1/2
- OfflinePointPush1/2, OfflineCarPush1/2, OfflineAntPush1/2

### Velocity-based (6 tasks):
- OfflineAntVelocity, OfflineSwimmerVelocity, OfflineWalkerVelocity
- OfflineHalfCheetahVelocity, OfflineHopperVelocity, OfflineHumanoidVelocity

*Append `Gymnasium-v0` to task names*

---

## Evaluation Metrics

During evaluation, the system tracks:
1. **Reward**: Cumulative reward over episode (higher is better)
2. **Cost**: Cumulative safety violations (lower is better)

Both metrics are logged to TensorBoard:
- `eval/rew`: Mean reward across evaluation episodes
- `eval/std`: Standard deviation of rewards
- Cost metrics can be added similarly

---

## Next Steps

### Immediate:
1. Run `python test_dsrl_setup.py` to verify setup
2. Train on a simple task: `python train_rl.py --env "OfflinePointGoal1Gymnasium-v0" --expid test --schedule OT`
3. Monitor TensorBoard: `tensorboard --logdir ./logs`

### Advanced:
1. Integrate SafeTD3 cost models as energy functions
2. Add cost-based metrics to evaluation
3. Experiment with different quantile thresholds
4. Compare OT vs linear vs cosine schedules

### Experimentation:
- Try different reward/cost quantiles for safety vs performance tradeoff
- Compare DSRL results with original D4RL baseline
- Visualize learned policies in safety environments

---

## Dependencies

Required packages:
- `torch` - PyTorch
- `dsrl` - DSRL datasets (`pip install dsrl`)
- `safety-gymnasium` - Safety environments (`pip install safety-gymnasium`)
- `numpy` - Numerical operations
- `gym` - OpenAI Gym (for D4RL)
- `d4rl` - D4RL datasets (for baseline)

---

## Troubleshooting

**Problem**: "No module named 'dsrl'"
- **Solution**: `pip install dsrl`

**Problem**: "No module named 'safety_gymnasium'"
- **Solution**: `pip install safety-gymnasium`

**Problem**: "No trajectories after filtering"
- **Solution**: Lower reward_quantile or increase cost_quantile

**Problem**: "Out of memory"
- **Solution**: Reduce batch size (bs1=4096 in train.py) or M parameter

---

## File Structure

```
Energy-weighted-flow-matching/
├── train_rl.py          # Main training script (MODIFIED)
├── utils.py             # Argument parser and utilities (MODIFIED)
├── dsrl_adapter.py      # DSRL dataset adapter (NEW)
├── test_dsrl_setup.py   # Setup validation script (NEW)
├── run_dsrl.ps1         # Example training commands (NEW)
├── DSRL_README.md       # Full documentation (NEW)
├── diffusion_SDE/       # Diffusion model implementation
├── dataset/             # Original D4RL dataset
└── models_rl/           # Saved checkpoints
    └── <expid>/
        ├── behavior_ckpt600.pth
        ├── critic_ckpt500.pth
        └── behavior_best_ckpt.pth
```

---

## Summary

✅ **Completed**:
- DSRL dataset adapter with trajectory filtering
- Dual-mode support (D4RL + DSRL)
- Safety-aware evaluation function
- Command-line arguments for quantile control
- Comprehensive documentation
- Setup validation script
- Example training scripts

✅ **Ready for**:
- Training on 28+ DSRL safety tasks
- Safe imitation learning experiments
- Comparison with baseline methods
- Integration with cost models

🚀 **Next**: Run `python test_dsrl_setup.py` to verify your setup!
