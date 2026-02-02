"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which.
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types
import safety_gymnasium

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from dataset.memory import Memory
from agent import make_agent
from utils.utils import eval_mode, average_dicts, get_concat_samples, eval_parallel, soft_update, hard_update,\
    load_dataset, merge_dataset

torch.set_num_threads(2)

def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'env = {args.env.name} with num_envs = {args.eval.n_envs}')
    eval_env = safety_gymnasium.vector.make(env_id=args.env.name, num_envs=args.eval.n_envs)
    eval_env.reset(seed=[i for i in range(args.eval.n_envs)])

    agent = make_agent(eval_env, args)

    c_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_C/full_data.hdf5')
    c_dataset = load_dataset(c_data_path,num_trajectories=args.expert.n_mix_good,seed=0)
    u_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_U/full_data.hdf5')
    u_dataset = load_dataset(u_data_path,num_trajectories=args.expert.n_mix_bad,seed=1)
    dataset = merge_dataset(c_dataset=c_dataset, u_dataset=u_dataset)
    del c_dataset
    del u_dataset
    
    mix_memory_replay = Memory(1, args.seed)
    mix_memory_replay.load_from_data(dataset)
    print(f'--> mix memory size: {mix_memory_replay.size()}')

    vio_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_U/undesirable_data.hdf5')
    vio_dataset = load_dataset(vio_data_path,num_trajectories=args.expert.n_bad,seed=2)
    vio_dataset['is_constrained'] = np.zeros_like(vio_dataset['costs'],dtype=bool)
            
    bad_memory_replay = Memory(1, args.seed)
    bad_memory_replay.load_from_data(vio_dataset)

    agent.update_disc = types.MethodType(update_disc, agent)
    agent.pretrain_disc = types.MethodType(pretrain_disc, agent)

    agent.update = types.MethodType(iq_update, agent)
    agent.update_critic = types.MethodType(update_critic, agent)
    if (args.train.update_actor_Q):
        print('update actor with Q inference')
        agent.actor_update = types.MethodType(update_actor_Q, agent)
    else:
        print('update actor with weighted-BC')
        agent.actor_update = types.MethodType(update_actor_BC, agent)

    disc_path = f'{hydra.utils.to_absolute_path("experts")}/{args.env.name}/Disc'
    agent.pretrain_disc(mix_memory_replay,bad_memory_replay,disc_path,
            f'state_disc_C({args.expert.n_mix_good})_U({args.expert.n_mix_bad})_B({args.expert.n_bad})')
    agent.disc_mix.eval()
    agent.disc_bad.eval()

    learn_steps = 0
    
    LEARN_STEPS = int(args.env.learn_steps)
    
    for learn_steps in range(LEARN_STEPS+1):
        info = {}
        if learn_steps % args.env.eval_interval == 0:
            eval_returns,eval_costs,cost_rate = \
                eval_parallel(agent, eval_env, num_episodes=args.eval.eps,args = args)
    
            returns = np.mean(eval_returns)
            cvar_cost = np.mean(top_10_percent_numpy(eval_costs))
            costs = np.mean(eval_costs)
            print(f'[Eval {learn_steps/LEARN_STEPS*100:.1f}%] Returns: {returns:.2f}, Costs: {costs:.2f}, CVaR: {cvar_cost:.2f}, Cost rate: {cost_rate:.3f}')
        
        if (learn_steps % 1000 == 0):
            print(f'[{learn_steps}/{LEARN_STEPS}]')
        info.update(agent.update(mix_memory_replay,
                                 bad_memory_replay, learn_steps))


def top_10_percent_numpy(lst):
    np_array = np.array(lst)
    sorted_array = np.sort(np_array)[::-1]
    top_10_count = int(len(sorted_array) * 0.1)
    return sorted_array[:top_10_count]

def pretrain_disc(self, mix_buffer, bad_buffer, disc_path,file_name,total_step = 50000):
    os.makedirs(disc_path, exist_ok=True)
    print(disc_path)
    print(file_name)
    
    mix_path = f'{disc_path}/mix_{file_name}'
    bad_path = f'{disc_path}/bad_{file_name}'
    
    if (os.path.isfile(mix_path)):
        print('load mix disc from ',mix_path)
        self.disc_mix.load_state_dict(torch.load(mix_path, map_location=self.device))
        print('[mix disc]',self.update_disc(mix_buffer,bad_buffer,mix_path, 0))
        print('load bad disc from ',bad_path)
        self.disc_bad.load_state_dict(torch.load(bad_path, map_location=self.device))
        print('[bad disc]',self.update_disc(mix_buffer,bad_buffer,bad_path, 0))
        
    else:
        print('Pretrain disc mix')
        print(mix_path)
        for itr in range(total_step+1):
            info = self.update_disc(mix_buffer,bad_buffer,mix_path, itr)
            if (itr%1000 == 0):
                print(f'{itr}/{total_step} dif = {abs(info["disc/bad"] - info["disc/mix"]):.3f}',info)
                if (abs(info['disc/bad'] - info['disc/mix']) >0.2):
                    break
        torch.save(self.disc_mix.state_dict(), mix_path)
        print('save disc mix to ',mix_path)      
        print('-'*20)       
   
        print('Pretrain disc bad')
        print(bad_path)
        for itr in range(total_step+1):
            info = self.update_disc(mix_buffer,bad_buffer,bad_path, itr)
            if (itr%1000 == 0):
                print(f'{itr}/{total_step} dif = {abs(info["disc/bad"] - info["disc/mix"]):.3f}',info)
                if (abs(info['disc/bad'] - info['disc/mix']) >0.2):
                    break
                
        torch.save(self.disc_bad.state_dict(), bad_path)
        print('save disc bad to ',bad_path)      
        print('-'*20)          

def update_disc(self,mix_buffer,bad_buffer,file_path,step):
    if ('mix' in file_path.split('/')[-1]):
        self.disc = self.disc_mix
        self.disc_optimizer = self.disc_mix_optimizer
    else:
        self.disc = self.disc_bad
        self.disc_optimizer = self.disc_bad_optimizer
        
    mix_batch       = mix_buffer.get_samples(1024, self.device, noise=0.0)
    bad_batch       = bad_buffer.get_samples(1024, self.device, noise=0.0)
    mix_obs, mix_next_obs, mix_action, _,_, _,mix_is_constrained = mix_batch
    bad_obs, bad_next_obs, bad_action, _,_, _,bad_is_constrained = bad_batch

    mix_d = self.disc(mix_obs)
    bad_d = self.disc(bad_obs)
    
    if ('bad' in file_path.split('/')[-1]):
        loss_bad = - torch.log(bad_d).mean()
        loss_mix = - torch.log(1-mix_d).mean()

    else:
        loss_bad = - torch.log(1-bad_d).mean()
        loss_mix = - torch.log(mix_d).mean()

    loss = loss_mix + loss_bad
    
    
    self.disc_optimizer.zero_grad()
    loss.backward()
    self.disc_optimizer.step()
    info = {}
    
    if (step%1000 == 0):
        with torch.no_grad():
            mix_batch       = mix_buffer.get_samples(5000, self.device)
            bad_batch       = bad_buffer.get_samples(5000, self.device)
            self.disc.eval()
            
            mix_obs, mix_next_obs, mix_action, _,_, _,mix_is_constrained = mix_batch
            bad_obs, bad_next_obs, bad_action, _,_, _,bad_is_constrained = bad_batch
            mix_d = self.disc(mix_obs)
            bad_d = self.disc(bad_obs)

            self.disc.train()
        
            info = {
                'disc/mix': round(mix_d.mean().item(),3),
                'disc/bad': round(bad_d.mean().item(),3),
            }
    return info

def update_actor_Q(self, obs, exp_action,next_obs,
                    done,env_cost,is_constrained,step):
    action, log_prob, _ = self.actor.sample(obs)
    actor_Q = self.critic(obs, action)

    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

    # optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    info = {}
    if (step%self.args.env.eval_interval == 0):
        info = {
            'actor_loss/target_entropy': round(self.target_entropy,1),
            'actor_loss/entropy': round(-log_prob.mean().item(),3)}

    return info
  
def update_actor_BC(self, obs, exp_action,next_obs,
                    done,env_cost,is_constrained,step):
    with torch.no_grad():
        Qs = self.critic_target(obs,exp_action).clip(min=-self.max_v,max=self.max_v)
        if (self.args.train.sep_V):
            current_V = self.value(obs).clip(min=-self.max_v,max=self.max_v)
        else:
            current_V = self.get_targetV(obs).clip(min=-self.max_v,max=self.max_v)
        
        cur_weight = self.get_weight(obs) + self.gamma * self.get_weight(next_obs)
        disc_weight = (1/(cur_weight**2)).clip(min=0,max=100)

        Q_adv = (Qs - current_V).clip(max=10)
        adv = Q_adv
        
        adv_weight = torch.exp(adv - adv.max())
        adv_weight = adv_weight / adv_weight.mean()
        
        weight = (adv_weight*disc_weight).clip(max=100)
        
    logp = self.actor.get_logp(obs,exp_action).clip(min=-10,max=5)
    
    actor_loss = - (weight.detach() * logp).mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    
    loss_dict = {}
    if (step%self.args.env.eval_interval == 0):
        loss_dict = {
            'actor_BC_loss/C_weight': round(weight[is_constrained.squeeze(-1)].mean().item(),3),
            'actor_BC_loss/U_weight': round(weight[~is_constrained.squeeze(-1)].mean().item(),3),
            
            'actor_BC_loss/C_logp': logp[is_constrained].mean().item(),
            'actor_BC_loss/U_logp': logp[~is_constrained].mean().item(),
            
            'actor_BC_loss/actor_loss': actor_loss.item(),
        }
    return loss_dict

def update_value(self, obs, action, next_obs, step):
    def iql_loss(pred, target, expectile=0.7):
        err = target - pred
        weight = torch.abs(expectile - (err < 0).float())
        return (weight * torch.square(err)).mean()
    
    args = self.args
    cur_V = self.value(obs)
    with torch.no_grad():
        cur_Q = self.critic_target(obs, action)
        
    loss = iql_loss(cur_V,cur_Q,0.7)
    
    self.value_optimizer.zero_grad()
    loss.backward()
    self.value_optimizer.step()
    
    infos = {}
    if (step%args.env.eval_interval == 0):
        infos = {
            'value/value_loss': round(loss.item(),3),
            'value/V': round(cur_V.mean().item(),3),
            'value/Q': round(cur_Q.mean().item(),3),
        }

    return infos

def update_critic(self, mix_batch, bad_batch, step):
    args = self.args
    batch = get_concat_samples(mix_batch,bad_batch, args)
    obs,next_obs,action,reward,cost,done,is_constrained, is_bad = batch
    agent = self
    infos = {}
    # update V
    if (args.train.sep_V):
        infos = update_value(self, obs, action, next_obs, step)
    
    # update Q    
    with torch.no_grad():
        if (args.train.sep_V):
            next_V = self.value(next_obs).clip(min=-self.max_v,max=self.max_v)
        else:
            next_V = self.get_targetV(next_obs).clip(min=-self.max_v,max=self.max_v)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = critic_loss(agent,agent.critic.Q1, current_Q1, next_V, batch,step)
        q2_loss, loss_dict2 = critic_loss(agent,agent.critic.Q2, current_Q2, next_V, batch,step)
        c_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        raise NotImplementedError
        current_Q = self.critic(obs, action)
        c_loss, loss_dict = c_loss(agent, current_Q, current_V, next_V, batch,step)
    infos.update(loss_dict)
    
    self.critic_optimizer.zero_grad()
    c_loss.backward()
    self.critic_optimizer.step()
    
    return infos

def iq_update(self, mix_buffer,bad_buffer, step):
    mix_batch       = mix_buffer.get_samples(self.batch_size, self.device)
    bad_batch       = bad_buffer.get_samples(self.batch_size, self.device)

    info = self.update_critic(mix_batch,bad_batch, step)
    mix_batch       = mix_buffer.get_samples(self.batch_size, self.device)
    obs, next_obs, action, env_reward,env_cost, done,is_constrained = mix_batch
    info.update(self.actor_update(obs,action,next_obs,done,env_cost,is_constrained, step))

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    if (self.first_log):
        self.first_log = False
    return info

def critic_loss(agent,critic_Q, current_Q, next_v, batch,step):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward,env_cost, done,is_constrained, is_bad = batch

    loss_dict = {}

    y = (1 - done) * gamma * next_v
    with torch.no_grad():
        cur_weight = agent.get_weight(obs,is_bad)
        next_weight = agent.get_weight(next_obs,is_bad)
        weight = (cur_weight + gamma * next_weight).clip(min=-agent.reward_factor,max=agent.reward_factor)
        
    reward = (current_Q - y)
    if (not args.agent.pen_bad):
        reward = reward[~is_bad]
        weight = weight[~is_bad]
    else:
        if (agent.first_log):
            print('[critic] penalize bad samples')
    reward_loss = -(weight * reward).mean()

    y = (1 - done) * gamma * next_v
    all_reward = (current_Q - y)
    chi2_loss = 0.5 * (all_reward**2).mean()
    
    loss = reward_loss + chi2_loss
    
    if args.method.loss == "v0":
        if (agent.first_log):
            print('[critic] v0 value loss')
        if (args.train.sep_V):
            v0 = agent.value(obs).mean()
        else:
            v0 = agent.getV(obs).mean()
            
        v0_loss = (1 - gamma) * v0
        loss += v0_loss
    elif args.method.loss == "no":
        if (agent.first_log):
            print('[critic] no value loss')
    else:
        raise NotImplementedError

    return loss, loss_dict


if __name__ == "__main__":
    main()
