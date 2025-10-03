import argparse
import gym
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="walker2d-medium-replay-v2") # OpenAI gym environment name or DSRL task
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="debug", type=str)    # 
    parser.add_argument("--device", default="cuda", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=3.0)        # beta parameter in the paper, use alpha because of legacy
    parser.add_argument('--n_behavior_epochs', type=int, default=600)
    parser.add_argument('--actor_load_path', type=str, default=None)
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--M', type=int, default=64)               # support action number
    parser.add_argument('--seed_per_evaluation', type=int, default=100)
    parser.add_argument('--s', type=float, nargs="*", default=None)# guidance scale
    parser.add_argument('--q_alpha', type=float, default=None)  
    parser.add_argument('--schedule', type=str, default="linear")  
    parser.add_argument('--K_renew', type=int, default=10)
    
    # DSRL-specific arguments
    parser.add_argument('--use_dsrl', action='store_true', help='Use DSRL safety datasets')
    parser.add_argument('--reward_quantile', type=float, default=0.75, 
                        help='Keep trajectories above this reward quantile (0-1)')
    parser.add_argument('--cost_quantile', type=float, default=0.25,
                        help='Keep trajectories below this cost quantile (0-1, lower = safer)')
    
    print("**************************")
    args = parser.parse_known_args()[0]
    if args.debug:
        args.actor_epoch =1
        args.critic_epoch =1
        args.env = "antmaze-medium-play-v2"
    if args.q_alpha is None:
        args.q_alpha = args.alpha
    print(args)
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
