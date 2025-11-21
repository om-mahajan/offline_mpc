#!/bin/bash
#
# Batch evaluation script for IPL+Flow models
# Evaluates all seeds across multiple tasks in parallel using tmux
#

# Configuration
BASE_LOG_DIR="$HOME/safe_diff/offline_mpc/logs/merged"
TASKS=("OfflinePointGoal2Gymnasium-v0" "OfflinePointButton1Gymnasium-v0")
SEEDS=(0 1 2)
NUM_EVALS=50
DEVICE="cpu"  # Use "cuda" if GPU available on HPC
ALGO_NAME="ipl_flow_matching_v2"

# Evaluation script parameters
TRAIN_HORIZON=5
DIFFUSION_STEPS=15
MAX_STEPS=1000

# Tmux session name
SESSION_NAME="flow_eval"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}IPL+Flow Model Batch Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Function to find model directory for a given task and seed
find_model_dir() {
    local task=$1
    local seed=$2
    
    # Construct the full path: BASE_LOG_DIR/TASK/IPLV4/merged/TASK/ALGO/seed-XXX-*
    local task_base_path="${BASE_LOG_DIR}/${task}/IPLV4/merged/${task}/${ALGO_NAME}"
    
    # Check if the directory exists
    if [ ! -d "$task_base_path" ]; then
        echo ""
        return 1
    fi
    
    # Find the seed directory
    local seed_pattern="seed-$(printf '%03d' $seed)-*"
    local seed_dirs=($task_base_path/$seed_pattern)
    
    if [ ${#seed_dirs[@]} -eq 0 ] || [ ! -d "${seed_dirs[0]}" ]; then
        echo ""
        return 1
    fi
    
    # Return the first matching directory
    echo "${seed_dirs[0]}"
    return 0
}

# Create tmux session
echo -e "${YELLOW}Creating tmux session: $SESSION_NAME${NC}"
tmux new-session -d -s $SESSION_NAME

# Counter for window numbering
window_num=0

# Loop through tasks and seeds
for task in "${TASKS[@]}"; do
    echo -e "\n${GREEN}Processing task: $task${NC}"
    
    for seed in "${SEEDS[@]}"; do
        # Find model directory
        model_dir=$(find_model_dir "$task" "$seed")
        
        if [ -z "$model_dir" ]; then
            echo -e "${RED}  ✗ Seed $seed: Model directory not found${NC}"
            continue
        fi
        
        if [ ! -d "$model_dir/torch_save" ]; then
            echo -e "${RED}  ✗ Seed $seed: torch_save directory not found in $model_dir${NC}"
            continue
        fi
        
        echo -e "${GREEN}  ✓ Seed $seed: Found model at $model_dir${NC}"
        
        # Create new window in tmux for this evaluation
        window_name="${task##*-}_s${seed}"
        
        if [ $window_num -eq 0 ]; then
            # Rename first window
            tmux rename-window -t $SESSION_NAME:0 "$window_name"
        else
            # Create new window
            tmux new-window -t $SESSION_NAME -n "$window_name"
        fi
	tmux send-keys -t $SESSION_NAME:$window_num "source ~/.bashrc" C-m
	tmux send-keys -t $SESSION_NAME:$window_num "source ~/anaconda3/etc/profile.d/conda.sh" C-m
	tmux send-keys -t $SESSION_NAME:$window_num "conda activate myenv" C-m	
        tmux send-keys -t $SESSION_NAME:$window_num "export CUDA_VISIBLE_DEVICES=3" C-m
        # Build the evaluation command
        eval_cmd="python /home/me22b018/safe_diff/offline_mpc/dsrl_model/model_free/ewfm/inference.py \
            --model-path '$model_dir' \
            --task '$task' \
            --seed $seed \
            --num-evals $NUM_EVALS \
            --train-horizon $TRAIN_HORIZON \
            --diffusion-steps $DIFFUSION_STEPS \
            --max-steps $MAX_STEPS \
            --device $DEVICE"
        
        # Send command to tmux window
        tmux send-keys -t $SESSION_NAME:$window_num "$eval_cmd" C-m
        
        window_num=$((window_num + 1))
    done
done

