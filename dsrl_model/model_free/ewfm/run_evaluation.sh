#!/bin/bash
# run_evaluations.sh
# Script to run evaluations for both tasks in separate tmux windows

set -e

# Configuration
CONDA_ENV="myenv"
BASE_LOG_DIR="$HOME/safe_diff/offline_mpc/logs/merged"
EXPERIMENT_SUBDIR="Pure_Flow/twinq"
SCRIPT_DIR="$HOME/safe_diff/offline_mpc/dsrl_model/model_free/ewfm"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_flow_models.py"

# Evaluation parameters
NUM_EPISODES=10
DIFFUSION_STEPS=15
HORIZON=5
STEP_INTERVAL=20000
MAX_EPISODE_STEPS=1000
DEVICE="cpu"

# Tasks to evaluate
TASK1="OfflineSwimmerVelocityGymnasium-v1"
TASK2="OfflinePointGoal1Gymnasium-v0"

# Tmux session name
SESSION_NAME="flow_eval"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Flow Matching Model Evaluation Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if evaluation script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo -e "${RED}Error: Evaluation script not found at $EVAL_SCRIPT${NC}"
    echo -e "${YELLOW}Please make sure evaluate_flow_models.py is in the correct location.${NC}"
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed. Please install tmux first.${NC}"
    echo "Install with: sudo apt-get install tmux"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not available. Please install conda/miniconda first.${NC}"
    exit 1
fi

# Get conda initialization script path
CONDA_BASE=$(conda info --base)
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_SH" ]; then
    echo -e "${RED}Error: Cannot find conda.sh at $CONDA_SH${NC}"
    exit 1
fi

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Killing existing tmux session: $SESSION_NAME${NC}"
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

# Create new tmux session
echo -e "${GREEN}Creating new tmux session: $SESSION_NAME${NC}"
tmux new-session -d -s "$SESSION_NAME" -n "eval_swimmer"

# Setup first window for SwimmerVelocity
echo -e "${GREEN}Setting up evaluation for $TASK1${NC}"
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "source $CONDA_SH" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "conda activate $CONDA_ENV" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "cd $SCRIPT_DIR" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "echo '==========================================='" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "echo 'Starting evaluation for $TASK1...'" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "echo 'Conda env: $CONDA_ENV'" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "echo 'Device: $DEVICE'" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "echo '==========================================='" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "python $EVAL_SCRIPT \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --task $TASK1 \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --base_log_dir $BASE_LOG_DIR \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --experiment_subdir '$EXPERIMENT_SUBDIR' \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --num_episodes $NUM_EPISODES \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --diffusion_steps $DIFFUSION_STEPS \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --horizon $HORIZON \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --step_interval $STEP_INTERVAL \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --max_episode_steps $MAX_EPISODE_STEPS \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "  --device $DEVICE" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "echo 'Evaluation for $TASK1 completed!'" C-m
tmux send-keys -t "$SESSION_NAME:eval_swimmer" "echo 'Press any key to exit or Ctrl+b d to detach'" C-m

# Create second window for PointGoal1
echo -e "${GREEN}Setting up evaluation for $TASK2${NC}"
tmux new-window -t "$SESSION_NAME" -n "eval_pointgoal"
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "source $CONDA_SH" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "conda activate $CONDA_ENV" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "cd $SCRIPT_DIR" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "echo '==========================================='" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "echo 'Starting evaluation for $TASK2...'" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "echo 'Conda env: $CONDA_ENV'" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "echo 'Device: $DEVICE'" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "echo '==========================================='" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "python $EVAL_SCRIPT \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --task $TASK2 \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --base_log_dir $BASE_LOG_DIR \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --experiment_subdir '$EXPERIMENT_SUBDIR' \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --num_episodes $NUM_EPISODES \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --diffusion_steps $DIFFUSION_STEPS \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --horizon $HORIZON \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --step_interval $STEP_INTERVAL \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --max_episode_steps $MAX_EPISODE_STEPS \\" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "  --device $DEVICE" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "echo 'Evaluation for $TASK2 completed!'" C-m
tmux send-keys -t "$SESSION_NAME:eval_pointgoal" "echo 'Press any key to exit or Ctrl+b d to detach'" C-m

# Switch back to first window
tmux select-window -t "$SESSION_NAME:eval_swimmer"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Tmux session created successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Session name:${NC} $SESSION_NAME"
echo -e "${BLUE}Conda environment:${NC} $CONDA_ENV"
echo -e "${BLUE}Device:${NC} $DEVICE"
echo ""
echo -e "${YELLOW}Windows:${NC}"
echo "  1. eval_swimmer   - Evaluating $TASK1"
echo "  2. eval_pointgoal - Evaluating $TASK2"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo ""
echo -e "${BLUE}To attach to the session:${NC}"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo -e "${BLUE}To switch between windows (inside tmux):${NC}"
echo "  Ctrl+b then 1 (for swimmer)"
echo "  Ctrl+b then 2 (for pointgoal)"
echo ""
echo -e "${BLUE}To detach from session (leave running in background):${NC}"
echo "  Ctrl+b then d"
echo ""
echo -e "${BLUE}To view session in real-time:${NC}"
echo "  watch -n 2 'tmux capture-pane -pt $SESSION_NAME:eval_swimmer && tmux show-buffer'"
echo ""
echo -e "${BLUE}To list all sessions:${NC}"
echo "  tmux ls"
echo ""
echo -e "${BLUE}To kill the session when done:${NC}"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo -e "${GREEN}✓ Evaluations running in background!${NC}"
echo -e "${GREEN}✓ Both tasks running in parallel${NC}"
echo ""
echo -e "${YELLOW}Tip: Attach to the session to monitor progress${NC}"
echo ""
