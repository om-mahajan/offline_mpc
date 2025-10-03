# Quick Start Guide - DSRL Integration

## 🚀 Getting Started in 3 Steps

### Step 1: Verify Setup
Run the test script to ensure all dependencies are installed:

```powershell
python test_dsrl_setup.py
```

If you see any errors, install missing packages:
```powershell
pip install dsrl safety-gymnasium
```

---

### Step 2: Run Your First Training

Choose one of these commands based on your hardware:

#### **Option A: GPU Training (Recommended)**
```powershell
python train_rl.py --env "OfflinePointGoal1Gymnasium-v0" --expid my_first_dsrl --seed 0 --reward_quantile 0.75 --cost_quantile 0.25 --schedule OT --device cuda
```

#### **Option B: CPU Training (Slower)**
```powershell
python train_rl.py --env "OfflinePointGoal1Gymnasium-v0" --expid my_first_dsrl --seed 0 --reward_quantile 0.75 --cost_quantile 0.25 --schedule OT --device cpu
```

---

### Step 3: Monitor Progress

Open a new terminal and run TensorBoard:
```powershell
tensorboard --logdir ./logs
```

Then open your browser to: http://localhost:6006

---

## 📊 What to Expect

### Training Time
- **Stage 1** (BC): ~2-4 hours (600 epochs)
- **Stage 2** (Q-critic): ~1-2 hours (500 epochs)  
- **Stage 3** (Energy-weighted): ~20-40 minutes (100 epochs)
- **Total**: ~4-6 hours on modern GPU

### Output Files
```
./models_rl/my_first_dsrl/
├── behavior_ckpt600.pth      # After behavior cloning
├── critic_ckpt500.pth         # After Q-training
└── behavior_best_ckpt.pth     # Best model (use this for evaluation)
```

### Logs
```
./logs/my_first_dsrl/
└── (TensorBoard event files)
```

---

## 🎯 Key Parameters Explained

| Parameter | What it does | Recommended values |
|-----------|--------------|-------------------|
| `--reward_quantile` | Keep only top X% of trajectories by reward | 0.75 (top 25%) or 0.8 (top 20%) |
| `--cost_quantile` | Keep only bottom X% by cost (safest) | 0.25 (safest 25%) or 0.1 (safest 10%) |
| `--schedule` | Diffusion noise schedule | OT (best), linear, cosine |
| `--alpha` | Energy weighting strength | 3.0 (default), try 1-5 |

---

## 🔧 Common Issues & Solutions

### Issue: "No module named 'dsrl'"
```powershell
pip install dsrl
```

### Issue: "CUDA out of memory"
1. Reduce batch size in `train_rl.py` (line ~52: change `bs1=4096` to `bs1=2048`)
2. Or use CPU: `--device cpu`

### Issue: "No trajectories after filtering"
Lower your standards:
```powershell
python train_rl.py --env "OfflinePointGoal1Gymnasium-v0" --reward_quantile 0.5 --cost_quantile 0.5 ...
```

---

## 📝 Available Tasks

Quick reference for `--env` parameter:

### Easy Tasks (Good for testing)
- `OfflinePointGoal1Gymnasium-v0` ⭐ Start here
- `OfflinePointButton1Gymnasium-v0`
- `OfflineCarGoal1Gymnasium-v0`

### Medium Tasks
- `OfflineAntGoal1Gymnasium-v0`
- `OfflineAntVelocityGymnasium-v0`
- `OfflineWalkerVelocityGymnasium-v0`

### Hard Tasks
- `OfflineAntGoal2Gymnasium-v0` (more constraints)
- `OfflineHumanoidVelocityGymnasium-v0`

---

## 🎓 Understanding the Training

### What happens during training?

1. **Stage 1 (Epochs 0-600)**: Learn to imitate safe demonstrations
   - Model learns: "What actions do experts take?"
   - No fancy stuff yet, just copying behavior

2. **Stage 2 (Epochs 601-1100)**: Learn which actions are good
   - Model learns: "Which actions lead to high rewards?"
   - Builds a Q-function (value estimator)

3. **Stage 3 (Epochs 1101-1200)**: Combine imitation + value
   - Model learns: "Imitate, but prefer high-value actions"
   - This is where energy-weighting happens!

### Evaluation (every 5 epochs in Stage 3)
- Runs 100 test episodes
- Tracks reward (higher = better)
- Tracks cost (lower = safer)
- Saves best model by reward

---

## 💡 Pro Tips

### Tip 1: Start Simple
Use PointGoal1 first. It's fast and you'll see results quickly.

### Tip 2: Adjust Safety vs Performance
- More safety? → Lower `--cost_quantile` (e.g., 0.1)
- More performance? → Higher `--reward_quantile` (e.g., 0.9)

### Tip 3: Use OT Schedule
The OT (Optimal Transport) schedule works best for DSRL tasks.

### Tip 4: Monitor Both Metrics
Watch both reward AND cost in evaluation. High reward + low cost = success!

### Tip 5: Save Your Results
Keep notes of which hyperparameters work best for each task.

---

## 📚 More Information

- **Full Documentation**: See `DSRL_README.md`
- **Changes Summary**: See `CHANGES_SUMMARY.md`
- **Original README**: See `Readme.md`

---

## 🆘 Need Help?

1. Run the test script: `python test_dsrl_setup.py`
2. Check the error messages carefully
3. Make sure you're in the right directory
4. Verify your Python environment is activated

---

## ✅ Success Checklist

Before you start training, make sure:

- [ ] Test script passes: `python test_dsrl_setup.py`
- [ ] CUDA available (optional): `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Enough disk space: ~10 GB free
- [ ] Environment activated: `.venv` or conda env

**Ready?** Run the command from Step 2 above! 🚀

---

## 📈 Expected Results

After training on OfflinePointGoal1Gymnasium-v0:

- **Reward**: Should reach 15-25 (higher = better)
- **Cost**: Should be < 5 (lower = safer)
- **Training Loss**: Should decrease steadily

If you see these results, congratulations! You've successfully trained a safe diffusion policy! 🎉
