#!/bin/bash
# run_eval_V5_hist.sh
# Script to evaluate V5_hist flow model checkpoints in tmux windows

set -e

# Configuration
CONDA_ENV="myenv"
BASE_LOG_DIR="$HOME/safe_diff/offline_mpc/logs/merged"
EXPERIMENT_SUBDIR="IPLV5_hist/twinq_fixed_hist"
ALGO_NAME="ipl_flow_twinq_fm_v5_fixed"
SCRIPT_DIR="$HOME/safe_diff/offline_mpc/dsrl_model/model_free/ewfm"
EVAL_SCRIPT="$SCRIPT_DIR/eval_V5_hist.py"

# Evaluation parameters
NUM_EPISODES=10
DIFFUSION_STEPS=15
STEP_INTERVAL=2000
MAX_EPISODE_STEPS=1000
DEVICE="cpu"
MIN_CHECKPOINT=100000  # Minimum checkpoint step to consider complete

# Tasks to evaluate
TASK1="OfflineSwimmerVelocityGymnasium-v1"
TASK2="OfflinePointGoal1Gymnasium-v0"

# Tmux session name
SESSION_NAME="eval_V5_hist"

echo "========================================="
echo "V5_hist Flow Model Evaluation"
echo "========================================="
echo "Experiment: $EXPERIMENT_SUBDIR"
echo "Algorithm: $ALGO_NAME"
echo "Min checkpoint: $MIN_CHECKPOINT"
echo ""

# Check if evaluation script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: Evaluation script not found at $EVAL_SCRIPT"
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed"
    exit 1
fi

# Get conda initialization
CONDA_BASE=$(conda info --base)
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: Cannot find conda.sh at $CONDA_SH"
    exit 1
fi

# Function to check if seed folder has sufficient checkpoints
check_seed_completeness() {
    local seed_dir=$1
    local torch_save_dir="${seed_dir}/torch_save"
    
    if [ ! -d "$torch_save_dir" ]; then
        echo "incomplete"
        return
    fi
    
    # Look for flow model checkpoints (not best/final)
    local max_checkpoint=$(find "$torch_save_dir" -name "flow_model_*.pt" 2>/dev/null | \
        grep -v "best" | grep -v "final" | \
        sed 's/.*flow_model_\([0-9]*\)\.pt/\1/' | \
        sort -n | tail -1)
    
    if [ -z "$max_checkpoint" ]; then
        # Try alternative naming: flow_*.pt
        max_checkpoint=$(find "$torch_save_dir" -name "flow_*.pt" 2>/dev/null | \
            grep -v "best" | grep -v "final" | \
            sed 's/.*flow_\([0-9]*\)\.pt/\1/' | \
            sort -n | tail -1)
    fi
    
    if [ -z "$max_checkpoint" ]; then
        echo "incomplete"
        return
    fi
    
    if [ "$max_checkpoint" -ge "$MIN_CHECKPOINT" ]; then
        echo "complete"
    else
        echo "incomplete"
    fi
}

# Function to find valid seeds for a task
find_valid_seeds() {
    local task=$1
    
    # Try multiple path patterns
    local log_dirs=(
        "${BASE_LOG_DIR}/${task}/${EXPERIMENT_SUBDIR}/${task}/${ALGO_NAME}"
        "${BASE_LOG_DIR}/${task}/${EXPERIMENT_SUBDIR}/${ALGO_NAME}"
        "${BASE_LOG_DIR}/${EXPERIMENT_SUBDIR}/${task}/${ALGO_NAME}"
    )
    
    for log_dir in "${log_dirs[@]}"; do
        if [ -d "$log_dir" ]; then
            echo "Checking path: $log_dir" >&2
            local seed_dirs=($(find "$log_dir" -mindepth 1 -maxdepth 1 -type d -name "seed-*" 2>/dev/null | sort))
            
            if [ ${#seed_dirs[@]} -gt 0 ]; then
                echo "Found ${#seed_dirs[@]} seed directories" >&2
                
                for seed_dir in "${seed_dirs[@]}"; do
                    local status=$(check_seed_completeness "$seed_dir")
                    if [ "$status" = "complete" ]; then
                        echo "$seed_dir"
                    fi
                done
                return
            fi
        fi
    done
    
    echo "No directories found for $task" >&2
}

# Find valid seeds
echo "Scanning Task 1: $TASK1"
TASK1_SEEDS=($(find_valid_seeds "$TASK1"))
echo "  Valid seeds: ${#TASK1_SEEDS[@]}"

echo "Scanning Task 2: $TASK2"
TASK2_SEEDS=($(find_valid_seeds "$TASK2"))
echo "  Valid seeds: ${#TASK2_SEEDS[@]}"
echo ""

TOTAL_SEEDS=$((${#TASK1_SEEDS[@]} + ${#TASK2_SEEDS[@]}))

if [ $TOTAL_SEEDS -eq 0 ]; then
    echo "ERROR: No valid seeds found with checkpoints >= $MIN_CHECKPOINT"
    echo ""
    echo "Debugging: Listing available directories..."
    echo ""
    for task in "$TASK1" "$TASK2"; do
        echo "Task: $task"
        ls -la "${BASE_LOG_DIR}/${task}/${EXPERIMENT_SUBDIR}/" 2>/dev/null || echo "  Path not found"
        echo ""
    done
    exit 1
fi

echo "Total valid seeds: $TOTAL_SEEDS"
echo ""

# Kill existing tmux session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

echo "Creating tmux session with $TOTAL_SEEDS windows..."
echo ""

# Create evaluation windows
WINDOW_IDX=0
FIRST_WINDOW=true

create_eval_window() {
    local task=$1
    local seed_dir=$2
    local seed_name=$(basename "$seed_dir")
    local window_name="eval_${task:7:10}_${seed_name:5:8}"
    
    if $FIRST_WINDOW; then
        tmux new-session -d -s "$SESSION_NAME" -n "$window_name"
        FIRST_WINDOW=false
    else
        tmux new-window -t "$SESSION_NAME" -n "$window_name"
    fi
    
    echo "  [Window $((WINDOW_IDX+1))/$TOTAL_SEEDS] $window_name"
    
    tmux send-keys -t "$SESSION_NAME:$window_name" "source $CONDA_SH" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "conda activate $CONDA_ENV" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $SCRIPT_DIR" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '==================================='" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'V5_hist Evaluation'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Task: ${task}'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Seed: ${seed_name}'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '==================================='" C-m
    
    tmux send-keys -t "$SESSION_NAME:$window_name" "python $EVAL_SCRIPT \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --task $task \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --seed_dir '$seed_dir' \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --num_episodes $NUM_EPISODES \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --diffusion_steps $DIFFUSION_STEPS \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --step_interval $STEP_INTERVAL \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --max_episode_steps $MAX_EPISODE_STEPS \\" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "  --device $DEVICE" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Evaluation completed!'" C-m
    
    WINDOW_IDX=$((WINDOW_IDX + 1))
}

# Create windows for all valid seeds
for seed_dir in "${TASK1_SEEDS[@]}"; do
    create_eval_window "$TASK1" "$seed_dir"
done

for seed_dir in "${TASK2_SEEDS[@]}"; do
    create_eval_window "$TASK2" "$seed_dir"
done

tmux select-window -t "$SESSION_NAME:0"

echo ""
echo "========================================="
echo "Tmux session created successfully!"
echo "========================================="
echo "Session name: $SESSION_NAME"
echo "Total windows: $TOTAL_SEEDS"
echo ""
echo "Evaluation parameters:"
echo "  Num episodes: $NUM_EPISODES"
echo "  Diffusion steps: $DIFFUSION_STEPS"
echo "  Step interval: $STEP_INTERVAL"
echo "  Max episode steps: $MAX_EPISODE_STEPS"
echo "  Device: $DEVICE"
echo ""
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Ctrl+b d"
echo "To kill: tmux kill-session -t $SESSION_NAME"
echo ""
