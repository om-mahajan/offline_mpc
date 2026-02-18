#!/bin/bash


ENVS=(
    "OfflineSwimmerVelocityGymnasium-v1"
)

SEEDS=(0 1 2 3 4)

SAVE_FREQ=20000
EXPERIMENT="IPLFLOWSAC_unionpos20"

GPU_ID=0
NUM_GPUS=4
PIDS=()


mkdir -p logs/nohup_logs_unionpos20

cd ./dsrl_model/model_free/ewfm
for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        GPU=$((GPU_ID % NUM_GPUS))
        
        env_clean=$(echo "$env" | sed 's/[^a-zA-Z0-9]/_/g')
        log_file="../../../logs/nohup_logs_unionpos20/${env_clean}_seed${seed}_gpu${GPU}.out"
        
        echo "Launching: ENV=$env SEED=$seed GPU=$GPU"
        
        CUDA_VISIBLE_DEVICES=$GPU nohup python ipltwin_V6_softmaxsac_without_alphafm.py \
            --task "$env" \
            --seed "$seed" \
            --device cuda \
            --experiment "$EXPERIMENT" \
            --batch_size 128 \
            >> "$log_file" 2>&1 &
        
        PIDS+=($!)
        

        GPU_ID=$((GPU_ID + 1))
        
        sleep 2
    done
done


echo "========================================="
echo "All training jobs running in background. PIDs: ${PIDS[*]}"


