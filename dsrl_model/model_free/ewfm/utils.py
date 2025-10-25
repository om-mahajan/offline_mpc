import argparse
import gym
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

def get_args():
    """
    Argument parser for energy-based flow matching (EWFM) methods.
    Compatible with both pubc (3-stage training) and difdice (2-phase training) scripts.
    """
    parser = argparse.ArgumentParser()
    # Environment and experiment
    parser.add_argument("--task", "--env", dest="task", default="OfflinePointGoal1Gymnasium-v0", 
                        help="DSRL task name")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--expid", default="debug", type=str, help="Experiment ID")
    parser.add_argument("--experiment", default="energy_flow_dsrl", type=str, help="Experiment name")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device ID")
    
    # Training configuration
    parser.add_argument('--alpha', type=float, default=3.0, help="Energy weighting parameter (beta in paper)")
    parser.add_argument('--diffusion_steps', type=int, default=15, help="Number of diffusion sampling steps")
    parser.add_argument('--M', type=int, default=64, help="Number of sampled actions for Q-training")
    parser.add_argument('--schedule', type=str, default="OT", choices=["OT", "linear", "cosine"],
                        help="Diffusion schedule")
    parser.add_argument('--K_renew', type=int, default=10, help="Frequency to renew fake actions")
    
    # Dataset configuration
    parser.add_argument('--num_non_preferred', '--num_negative', dest='num_non_preferred', 
                        type=int, default=50, help="Number of non-preferred (high-cost) trajectories")
    parser.add_argument('--num_union', type=int, default=-1, help="Number of union trajectories (-1 = all remaining)")
    parser.add_argument('--train_horizon', type=int, default=60, help="Training horizon for trajectories")
    parser.add_argument('--non_pref_noise', type=float, default=0.0, help="Noise for non-preferred selection")
    parser.add_argument('--normalize_observation', action='store_true', help="Normalize observations")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--total_iteration', type=int, default=int(1e6), 
                        help="Total training steps (iterations). Default 1M for pubc 3-stage training")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")
    
    # DifdICE-specific arguments (2-phase training)
    parser.add_argument('--cost_pretrain_iterations', type=int, default=50000,
                        help="Phase 1: Cost model pretraining iterations (difdice only)")
    parser.add_argument('--flow_train_iterations', type=int, default=10000,
                        help="Phase 2: Flow and critic training iterations (difdice only)")
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help="Discount factor for critic (difdice only)")
    parser.add_argument('--grad_reg_coeffs_nu', type=float, default=1e-6,
                        help="Gradient penalty coefficient for critic (difdice only)")
    parser.add_argument('--cost_weight_temp', type=float, default=1.0,
                        help="Temperature for advantage-based weighting (difdice only)")
    
    # Note: pubc uses 3-stage training automatically split as 60%/30%/10% of total_iteration
    # Stage 1 (BC): 600k steps (60% of 1M)
    # Stage 2 (Q): 300k steps (30% of 1M)  
    # Stage 3 (Energy): 100k steps (10% of 1M)
    
    # Evaluation
    parser.add_argument('--seed_per_evaluation', '--eval_episodes', dest='seed_per_evaluation', 
                        type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument('--eval_freq', type=int, default=10000, help="Evaluation frequency (steps)")
    parser.add_argument('--use_eval', action='store_true', default=True, help="Enable evaluation")
    parser.add_argument('--eval_episode_freq', type=int, default=3,
                        help="Number of episodes per evaluation (difdice)")
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default="./logs", help="Log directory")
    parser.add_argument('--log_freq', type=int, default=int(1e4), help="Logging frequency (steps)")
    parser.add_argument('--save_freq', type=int, default=int(2e4), help="Model saving frequency (steps)")
    parser.add_argument('--write_terminal', action='store_true', help="Write logs to terminal")
    
    # Legacy/optional
    parser.add_argument('--debug', action='store_true', help="Debug mode (reduces to 1000 steps)")
    parser.add_argument('--actor_load_path', type=str, default=None, help="Path to load pre-trained actor")
    parser.add_argument('--q_alpha', type=float, default=None, help="Q-alpha value (defaults to alpha)")
    parser.add_argument('--action_repeat', type=int, default=1, help="Action repeat")
    
    print("=" * 60)
    args = parser.parse_known_args()[0]
    
    if args.debug:
        # Debug mode: reduce to 1000 steps
        args.total_iteration = 1000
        args.cost_pretrain_iterations = 100
        args.flow_train_iterations = 100
        print("DEBUG MODE: Reduced iterations to 1000/100/100")
        
    if args.q_alpha is None:
        args.q_alpha = args.alpha
        
    print(args)
    print("=" * 60)
    return args

def bandit_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="8gaussians") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str)    # 
    parser.add_argument("--device", default="cuda", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=3.0)        # beta parameter in the paper, use alpha because of legacy
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--method', type=str, default="CEP")
    parser.add_argument('--schedule', type=str, default="linear")  
    print("**************************")
    args = parser.parse_known_args()[0]
    print(args)
    return args

def pallaral_eval_policy(policy_fn, env_name, seed, eval_episodes=20, diffusion_steps=15):
    eval_envs = []
    for i in range(eval_episodes):
        env = gym.make(env_name)
        eval_envs.append(env)
        env.seed(seed + 1001 + i)
        env.buffer_state = env.reset()
        env.buffer_return = 0.0
    ori_eval_envs = [env for env in eval_envs]
    import time
    
    while len(eval_envs) > 0:
        new_eval_envs = []
        states = np.stack([env.buffer_state for env in eval_envs])
        actions = policy_fn(states, diffusion_steps=diffusion_steps)
        for i, env in enumerate(eval_envs):
            state, reward, done, info = env.step(actions[i])
            env.buffer_return += reward
            env.buffer_state = state
            if not done:
                new_eval_envs.append(env)
        eval_envs = new_eval_envs
    print([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
    mean = np.mean([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
    std = np.std([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
    print("reward {} +- {}".format(mean,std))
    return ori_eval_envs

def simple_eval_policy(policy_fn, env_name, seed, eval_episodes=20):
    env = gym.make(env_name)
    env.seed(seed+561)
    all_rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()
        total_reward = 0.
        done = False
        while not done:
            with torch.no_grad():
                action = policy_fn(torch.Tensor(obs).unsqueeze(0).to("cuda")).cpu().numpy().squeeze()
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
            else:
                obs = next_obs
        all_rewards.append(total_reward)
    return np.mean(all_rewards), np.std(all_rewards)
