import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from diffusion_SDE import dpm_solver_pytorch
from diffusion_SDE import schedule
from scipy.special import softmax

def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)


def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class Residual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim=128, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim),SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim),SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    def forward(self, x, t):
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)

class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [state_dim + action_dim, 256, 256, 256, 1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))

class QGPO_Critic(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        # is sdim is 0  means unconditional guidance
        assert sdim > 0
        # only apply to conditional sampling here
        self.q0 = TwinQ(adim, sdim).to(args.device)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False).to(args.device)
        self.discount = 0.99
        
        self.args = args
        self.alpha = args.alpha
class ScoreBase(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.device=args.device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule=args.schedule)
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule, predict_x0=True)
        # self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.args = args

    def forward_dmp_wrapper_fn(self, x, t):
        return -self(x, t) * self.marginal_prob_std(t)[1][..., None]
    
    def dpm_wrapper_sample(self, dim, batch_size, is_numpy=True, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            sample = self.dpm_solver.sample(init_x, **kwargs)
            return sample.cpu().numpy() if is_numpy else sample
    
    def forward(self, x, t, condition=None):
        raise NotImplementedError

    def select_actions(self, states, diffusion_steps=15):
        multiple_input=True
        with torch.no_grad():
            # Handle both numpy arrays and tensors
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            else:
                states = states.to(self.device)
            
            if states.dim == 1:
                states = states.unsqueeze(0)
                multiple_input=False
            num_states = states.shape[0]
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=2)
            actions = results.reshape(num_states, self.output_dim).copy() # <bz, A>
            self.condition = None
        out_actions = [actions[i] for i in range(actions.shape[0])] if multiple_input else actions[0]
        return out_actions

    def sample(self, states, sample_per_state=16, diffusion_steps=15, is_numpy=True):
        num_states = states.shape[0]
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device) if is_numpy else states
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=2, is_numpy=is_numpy)
            actions = results[:, :].reshape(num_states, sample_per_state, self.output_dim)
            if is_numpy:
                actions = actions.copy()
            self.condition = None
        return actions


class ScoreNet(ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function (removed unused lambda to enable pickling)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim-output_dim, 32), SiLU())
        self.sort_t = nn.Sequential(
                        nn.Linear(64, 128),                        
                        SiLU(),
                        nn.Linear(128, 128),
                    )
        self.down_block1 = Residual_Block(output_dim, 512)
        self.down_block2 = Residual_Block(512, 256)
        self.down_block3 = Residual_Block(256, 128)
        self.middle1 = Residual_Block(128, 128)
        self.up_block3 = Residual_Block(256, 256)
        self.up_block2 = Residual_Block(512, 512)
        self.last = nn.Linear(1024, output_dim)
        
    def forward(self, x, t, condition=None):
        embed = self.embed(t)
        
        if condition is not None:
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        else:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition]*x.shape[0])
            else:
                assert False
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        embed = self.sort_t(embed)
        d1 = self.down_block1(x, embed)
        d2 = self.down_block2(d1, embed)
        d3 = self.down_block3(d2, embed)
        u3 = self.middle1(d3, embed)
        u2 = self.up_block3(torch.cat([d3, u3], dim=-1), embed)
        u1 = self.up_block2(torch.cat([d2, u2], dim=-1), embed)
        u0 = torch.cat([d1, u1], dim=-1)
        h = self.last(u0)
        self.h = h
        # Normalize output
        return h / self.marginal_prob_std(t)[1][..., None]
