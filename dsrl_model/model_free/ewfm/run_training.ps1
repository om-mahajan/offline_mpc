Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Energy-Weighted Flow Matching Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to script directory
Set-Location $PSScriptRoot

# Set experiment parameters
$TASK = "OfflinePointGoal1Gymnasium-v0"
$NUM_NON_PREFERRED = 100
$SEED = 0
$EXPID = "exp001"
$BATCH_SIZE = 256
$DIFFUSION_STEPS = 15
$LR = 0.0003
$LOG_FREQ = 2500
$EVAL_FREQ = 5000
$SAVE_FREQ = 20000

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Task: $TASK"
Write-Host "  Non-preferred trajectories: $NUM_NON_PREFERRED"
Write-Host "  Seed: $SEED"
Write-Host "  Experiment ID: $EXPID"
Write-Host "  Batch size: $BATCH_SIZE"
Write-Host "  Diffusion steps: $DIFFUSION_STEPS"
Write-Host "  Learning rate: $LR"
Write-Host "  Log frequency: $LOG_FREQ"
Write-Host "  Eval frequency: $EVAL_FREQ"
Write-Host "  Save frequency: $SAVE_FREQ"
Write-Host ""

# Run training with evaluation enabled
python train_rl_dsrl.py `
  --task $TASK `
  --num_non_preferred $NUM_NON_PREFERRED `
  --seed $SEED `
  --expid $EXPID `
  --batch_size $BATCH_SIZE `
  --diffusion_steps $DIFFUSION_STEPS `
  --lr $LR `
  --log_freq $LOG_FREQ `
  --eval_freq $EVAL_FREQ `
  --save_freq $SAVE_FREQ `
  --use_eval `
  --write_terminal

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved in:" -ForegroundColor Yellow
Write-Host "  Models: ./models_rl/$EXPID/"
Write-Host "  Logs: ./logs/energy_flow_dsrl/$TASK/energy_flow_matching/"
Write-Host ""

Read-Host "Press Enter to exit"Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Energy-Weighted Flow Matching Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python train_rl_dsrl.py `
  --task OfflinePointGoal1Gymnasium-v0 `
  --num_non_preferred 100 `
  --seed 0 `
  --expid exp001 `
  --batch_size 256 `
  --diffusion_steps 15 `
  --lr 0.0003 `
  --log_freq 2500 `
  --eval_freq 5000 `
  --save_freq 20000 `
  --use_eval `
  --write_terminal