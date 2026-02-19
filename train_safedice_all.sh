#!/bin/bash


ENVS=(
    "OfflineSwimmerVelocityGymnasium-v1"
)

SEEDS=(0 1 2 3 4)

SAVE_FREQ=20000
EXPERIMENT="safedice_unionpos100"

GPUS=(1 2 3)
GPU_IDX=0
PIDS=()


mkdir -p logs/safedice_unionpos100

cd ./dsrl_model/model_free
for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        GPU=${GPUS[$((GPU_IDX % ${#GPUS[@]}))]}
        
        env_clean=$(echo "$env" | sed 's/[^a-zA-Z0-9]/_/g')
        log_file="../../logs/safedice_unionpos100/${env_clean}_seed${seed}_gpu${GPU}.out"
        
        echo "Launching: ENV=$env SEED=$seed GPU=$GPU"
        
        CUDA_VISIBLE_DEVICES=$GPU nohup python safedice.py \
            --task "$env" \
            --seed "$seed" \
            --device cuda \
            --experiment "$EXPERIMENT" \
            --batch-size 128 \
            >> "$log_file" 2>&1 &
        
        PIDS+=($!)
        

        GPU_IDX=$((GPU_IDX + 1))
        
        sleep 2
    done
done


echo "========================================="
echo "All training jobs running in background. PIDs: ${PIDS[*]}"


