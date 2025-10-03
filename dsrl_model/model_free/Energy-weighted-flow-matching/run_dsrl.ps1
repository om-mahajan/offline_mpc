# PowerShell script to run energy-weighted flow matching on DSRL safety datasets

# Example 1: OfflinePointGoal1 with default quantiles
python -u train_rl.py `
    --expid dsrl_point_goal_1 `
    --env "OfflinePointGoal1Gymnasium-v0" `
    --seed 0 `
    --reward_quantile 0.75 `
    --cost_quantile 0.25 `
    --schedule OT `
    --device cuda `
    --alpha 3.0 `
    --diffusion_steps 15

# Example 2: OfflineCarGoal1 with stricter safety
# python -u train_rl.py `
#     --expid dsrl_car_goal_1 `
#     --env "OfflineCarGoal1Gymnasium-v0" `
#     --seed 0 `
#     --reward_quantile 0.8 `
#     --cost_quantile 0.1 `
#     --schedule OT `
#     --device cuda `
#     --alpha 3.0 `
#     --diffusion_steps 15

# Example 3: OfflineAntVelocity
# python -u train_rl.py `
#     --expid dsrl_ant_velocity `
#     --env "OfflineAntVelocityGymnasium-v0" `
#     --seed 0 `
#     --reward_quantile 0.75 `
#     --cost_quantile 0.25 `
#     --schedule OT `
#     --device cuda `
#     --alpha 3.0 `
#     --diffusion_steps 15

Write-Host "Training completed!"
