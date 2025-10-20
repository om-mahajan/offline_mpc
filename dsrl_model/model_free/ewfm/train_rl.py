import os
import gym
import d4rl
import scipy
import tqdm
import functools
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusion_SDE.loss import loss_fn as loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, QGPO_Critic, update_target
from utils import get_args, pallaral_eval_policy
from dataset.dataset import D4RL_dataset

def prepare_fake_actions(score_model, dataset, args, sample_batch_size=32768, bs2=256, nw=6):
    with torch.no_grad():
        score_model.eval()
        allstates = dataset.states[:]
        actions = []
        for _states in tqdm.tqdm(torch.split(allstates, sample_batch_size)):
            actions.append(score_model.sample(_states.to(args.device), sample_per_state=args.M, diffusion_steps=args.diffusion_steps, is_numpy=False).cpu())
        dataset.fake_actions = torch.cat(actions)
        allstates = dataset.next_states[:]
        actions = []
        for _states in tqdm.tqdm(torch.split(allstates, sample_batch_size)):
            actions.append(score_model.sample(_states.to(args.device), sample_per_state=args.M, diffusion_steps=args.diffusion_steps, is_numpy=False).cpu())
        dataset.fake_next_actions = torch.cat(actions)
        data_loader = DataLoader(dataset, batch_size=bs2, shuffle=True, pin_memory=True, num_workers=nw)
        return data_loader

def train(args, score_model, q_model, dataset, start_epoch=0, score_model_target=None):
    def datas_(dataloader):
        while True:
            yield from dataloader
    start_epoch = 0

    if "antmaze" not in args.env:
        q_alpha = 1
    else:
        q_alpha = 20


    train_score_model = 600
    train_q_model = 500
    retrain_score_model = 100
    evaluation_interval = 5
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-4)
    q_optimizer = torch.optim.Adam(q_model.q0.parameters(), lr=3e-4)
    tqdm_step = tqdm.trange(start_epoch, train_score_model + train_q_model + retrain_score_model, 1)
    max_reward, max_std = -1, -1
    bs1, nw = 4096, 6
    # prepare the first dataloader for training the score model
    data_loader = DataLoader(dataset, batch_size=bs1, shuffle=True, pin_memory=True, num_workers=nw)
    datas = datas_(data_loader)
    for epoch in tqdm_step:
        avg_loss, num_items = 0.0, 0
        if epoch <= train_score_model:
            score_model.train()
            # print("initialize baseline model")
            for _ in range(1000 if epoch < train_score_model else 100):
                data = next(datas)
                data = {k: d.to(args.device) for k, d in data.items()}
                s, a = data['s'], data['a']
                score_model.condition = s
                loss = loss_fn(score_model, a, args.marginal_prob_std_fn, energy=None, alpha=args.alpha)
                if epoch < train_score_model:
                    optimizer.zero_grad()
                    loss.backward()  
                    optimizer.step()
                score_model.condition = None
                avg_loss += loss.detach().item()
                num_items += 1
            if epoch == train_score_model and not os.path.exists(os.path.join("./models_rl", str(args.expid), f"behavior_ckpt600.pth")):
                torch.save(score_model.state_dict(), os.path.join("./models_rl", str(args.expid), f"behavior_ckpt600.pth"))

        elif epoch <= train_score_model + train_q_model + 1:
            if epoch == train_score_model + 1 or epoch % 100 == 1:
                print('prepare fake actions')
                data_loader = prepare_fake_actions(score_model, dataset, args)
                datas = datas_(data_loader)
                print('train q model')
                q_model.train()

            for _ in range(1000 if epoch < train_score_model + train_q_model + 1 else 100):
                data = next(datas)
                data = {k: d.to(args.device) for k, d in data.items()}
                s, a, r, s_, d, fake_a_ = data["s"], data["a"], data["r"], data["s_"], data["d"], data["fake_a_"]
                with torch.no_grad():
                    softmax = nn.Softmax(dim=1)
                    next_energy = q_model.q0_target(fake_a_ , torch.stack([s_]*fake_a_.shape[1] ,axis=1)).detach().squeeze() # <bz, 16>            
                    next_v = torch.sum(softmax(q_alpha * next_energy) * next_energy, dim=-1, keepdim=True) # the Q-alpha is set to 1 in the script

                # Update Q function
                targets = r + (1. - d.float()) * q_model.discount * next_v.detach()
                qs = q_model.q0.both(a, s)
                q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
                if epoch < train_score_model + train_q_model + 1:
                    q_optimizer.zero_grad(set_to_none=True)
                    q_loss.backward()
                    q_optimizer.step()
                    update_target(q_model.q0, q_model.q0_target, 0.005)
                avg_loss += q_loss.detach().item()
                num_items += 1

            if epoch == train_score_model + train_q_model + 1: # we only need to prepare this once
                # freeze Q functions
                for p in q_model.q0.parameters():
                    p.requires_grad_(False)
                q_model.eval(), score_model.train()
                score_model_target.load_state_dict(score_model.state_dict())
                if not os.path.exists(os.path.join("./models_rl", str(args.expid), f"critic_ckpt500.pth")):
                    torch.save(q_model.state_dict(), os.path.join("./models_rl", str(args.expid), f"critic_ckpt500.pth"))
        else:
            if epoch % args.K_renew == 0:
                print('update fake actions')
                data_loader = prepare_fake_actions(score_model_target, dataset, args)
                datas = datas_(data_loader)

            for _ in range(1000): # perfer report and test more frequently
                data = next(datas)
                s, a= data['S'].to(args.device), data['fake_a'].to(args.device)
                with torch.no_grad():
                    e = q_model.q0(a, s).detach().squeeze()
                score_model.condition = s
                loss = loss_fn(score_model, a, args.marginal_prob_std_fn, energy=e, alpha=args.alpha)
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step()
                update_target(score_model, score_model_target, 0.005)
                score_model.condition = None
                avg_loss += loss.detach().item()
                num_items += 1
        tqdm_step.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        args.writer.add_scalar("loss", avg_loss / num_items, global_step=epoch)
        if (epoch % evaluation_interval == (evaluation_interval - 1)) and epoch >= train_score_model + train_q_model:
            envs = args.eval_func(score_model.select_actions)
            mean = np.mean([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
            std = np.std([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
            args.writer.add_scalar("eval/rew", mean, global_step=epoch)
            args.writer.add_scalar("eval/std", std, global_step=epoch)
            if mean > max_reward:
                max_reward = mean
                max_std=std
                torch.save(score_model.state_dict(), os.path.join("./models_rl", str(args.expid), f"behavior_best_ckpt.pth"))
    print("best rewards:",max_reward,"+-",max_std)


def main(args):
    # The diffusion behavior training pipeline is copied directly from https://github.com/ChenDRAG/SfBC/blob/master/train_behavior.py
    for dir in ["./models_rl", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models_rl", str(args.expid))):
        os.makedirs(os.path.join("./models_rl", str(args.expid)))
    if not os.path.exists(os.path.join("./models_rl", str(args.env))):
        os.makedirs(os.path.join("./models_rl", str(args.env)))
    if not os.path.exists(os.path.join("./logs", "OT")):
        os.makedirs(os.path.join("./logs", "OT"))
    writer = SummaryWriter("./logs/" + str(args.expid))
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, seed=args.seed, eval_episodes=args.seed_per_evaluation, diffusion_steps=args.diffusion_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, schedule = args.schedule, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model_target= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model_target.eval()
    for p in score_model_target.parameters():
        p.requires_grad_(False)
    q_model = QGPO_Critic(adim=action_dim, sdim=state_dim, args=args).to(args.device)
    dataset = D4RL_dataset(args)
    # since these S are frequently used, we store them in the dataset
    dataset.S = torch.stack([dataset.states[:]] * args.M, dim=1)
    print("training behavior")
    train(args, score_model, q_model, dataset, start_epoch=0, score_model_target=score_model_target)
    print("finished")

if __name__ == "__main__":
    
    args = get_args()
    import datetime 
    now = datetime.datetime.now()
    print(now)
    print(args.task,args.alpha)
    args.seed_per_evaluation = 100
    
    
    main(args)
