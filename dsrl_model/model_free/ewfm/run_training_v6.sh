#!/bin/bash
# run_training_v6.sh
# Script to run V6 training for multiple tasks and seeds in separate tmux windows

set -e

# Configuration
CONDA_ENV="myenv"
BASE_LOG_DIR="$HOME/safe_diff/offline_mpc/logs/merged"
EXPERIMENT_SUBDIR="IPLV6"
SCRIPT_DIR="$HOME/safe_diff/offline_mpc/dsrl_model/model_free/ewfm"
TRAIN_SCRIPT="$SCRIPT_DIR/ipltwin_V6.py"
BASHRC="$HOME/.bashrc"

# Training parameters
PREFERENCE_ITERATIONS=100000
FLOW_TRAIN_ITERATIONS=300000
LOG_FREQ=20000
SAVE_FREQ=20000
SEGMENT_LENGTH=15
BATCH_SIZE=64
NUM_NEGATIVE_TRAJECTORIES=50

# Tasks to train
TASK1="OfflineSwimmerVelocityGymnasium-v1"
TASK2="OfflinePointGoal1Gymnasium-v0"

# Seeds
SEEDS=(1 2 3)

# Tmux session name
SESSION_NAME="s2_v6"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}IPL V6 Model Training Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo -e "${RED}Error: Training script not found at $TRAIN_SCRIPT${NC}"
    echo -e "${YELLOW}Please make sure ipltwin_V6.py is in the correct location.${NC}"
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed. Please install tmux first.${NC}"
    echo "Install with: sudo apt-get install tmux"
    exit 1
fi

# Check if .bashrc exists
if [ ! -f "$BASHRC" ]; then
    echo -e "${RED}Error: .bashrc not found at $BASHRC${NC}"
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

# Window counter
WINDOW_NUM=0

# Function to create a training window
create_training_window() {
    local task=$1
    local seed=$2
    local window_name=$3
    local is_first=$4
    
    local log_dir="$BASE_LOG_DIR/$task/$EXPERIMENT_SUBDIR"
    
    if [ "$is_first" = true ]; then
        # Create first window with the session
        tmux new-session -d -s "$SESSION_NAME" -n "$window_name"
    else
        # Create subsequent windows
        tmux new-window -t "$SESSION_NAME" -n "$window_name"
    fi
    
    echo -e "${GREEN}Setting up V6 training for $task (seed $seed)${NC}"
    
    # Send commands to the window
    tmux send-keys -t "$SESSION_NAME:$window_name" "source $BASHRC" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $SCRIPT_DIR" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "conda activate $CONDA_ENV" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '==========================================='" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Starting V6 training for $task (seed $seed)...'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Conda env: $CONDA_ENV'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Log dir: $log_dir'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '==========================================='" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "python $TRAIN_SCRIPT \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --task $task \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --log_dir $log_dir \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --seed $seed \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --preference_iterations $PREFERENCE_ITERATIONS \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --flow_train_iterations $FLOW_TRAIN_ITERATIONS \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --batch_size $BATCH_SIZE \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --log_freq $LOG_FREQ \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --save_freq $SAVE_FREQ" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Training for $task (seed $seed) completed!'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Press any key to exit or Ctrl+b d to detach'" C-m
}

# Create windows for SwimmerVelocity with different seeds
for seed in "${SEEDS[@]}"; do
    window_name="swimmer_s${seed}"
    if [ $WINDOW_NUM -eq 0 ]; then
        create_training_window "$TASK1" "$seed" "$window_name" true
    else
        create_training_window "$TASK1" "$seed" "$window_name" false
    fi
    WINDOW_NUM=$((WINDOW_NUM + 1))
done

# Create windows for PointGoal1 with different seeds
for seed in "${SEEDS[@]}"; do
    window_name="pointgoal_s${seed}"
    create_training_window "$TASK2" "$seed" "$window_name" false
    WINDOW_NUM=$((WINDOW_NUM + 1))
done

# Switch back to first window
tmux select-window -t "$SESSION_NAME:0"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Tmux session created successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Session name:${NC} $SESSION_NAME"
echo -e "${BLUE}Conda environment:${NC} $CONDA_ENV"
echo -e "${BLUE}Total windows:${NC} 6"
echo ""
echo -e "${YELLOW}Training Parameters:${NC}"
echo "  Preference Iterations: $PREFERENCE_ITERATIONS"
echo "  Flow Train Iterations: $FLOW_TRAIN_ITERATIONS"
echo "  Segment Length: $SEGMENT_LENGTH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Log Freq: $LOG_FREQ"
echo "  Save Freq: $SAVE_FREQ"
echo ""
echo -e "${YELLOW}Windows:${NC}"
echo "  0. swimmer_s1     - Training $TASK1 (seed 1)"
echo "  1. swimmer_s2     - Training $TASK1 (seed 2)"
echo "  2. swimmer_s3     - Training $TASK1 (seed 3)"
echo "  3. pointgoal_s1   - Training $TASK2 (seed 1)"
echo "  4. pointgoal_s2   - Training $TASK2 (seed 2)"
echo "  5. pointgoal_s3   - Training $TASK2 (seed 3)"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo ""
echo -e "${BLUE}To attach to the session:${NC}"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo -e "${BLUE}To switch between windows (inside tmux):${NC}"
echo "  Ctrl+b then 0-5 (window number)"
echo "  Ctrl+b then n (next window)"
echo "  Ctrl+b then p (previous window)"
echo ""
echo -e "${BLUE}To detach from session (leave running in background):${NC}"
echo "  Ctrl+b then d"
echo ""
echo -e "${BLUE}To list all sessions:${NC}"
echo "  tmux ls"
echo ""
echo -e "${BLUE}To kill the session when done:${NC}"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo -e "${GREEN}✓ V6 training jobs running in background!${NC}"
echo -e "${GREEN}✓ All tasks and seeds running in parallel${NC}"
echo ""
echo -e "${YELLOW}Tip: Attach to the session to monitor progress${NC}"
echo ""