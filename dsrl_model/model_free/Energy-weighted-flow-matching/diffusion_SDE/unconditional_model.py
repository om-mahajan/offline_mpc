import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusion_SDE import dpm_solver_pytorch
from diffusion_SDE import schedule
from diffusion_SDE.model import GaussianFourierProjection, Dense, SiLU, mlp

class Bandit_ScoreBase(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, args=None):
        super().__init__()
        assert input_dim == output_dim
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.device=args.device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule=args.schedule)
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.args = args

    def forward_dmp_wrapper_fn(self, x, t):
        return -self(x, t) * self.marginal_prob_std(t)[1][..., None]
    
    def dpm_wrapper_sample(self, dim, batch_size, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()
    
    def forward(self, x, t, condition=None):
        raise NotImplementedError


    def sample(self, states=None, sample_per_state=16, diffusion_steps=15):
        self.eval()
        with torch.no_grad():
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=sample_per_state, steps=diffusion_steps, order=2, method='multistep')
            actions = results[:, :]
        self.train()
        return actions

class Bandit_MlpScoreNet(Bandit_ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.dense1 = Dense(embed_dim, 32)
        self.dense2 = Dense(output_dim, 256 - 32)
        self.block1 = nn.Sequential(
            nn.Linear(256, 512),
            SiLU(),
            nn.Linear(512, 512),
            SiLU(),
            nn.Linear(512, 512),
            SiLU(),
            nn.Linear(512, 512),
            SiLU(),
            nn.Linear(512, 256),
        )
        self.decoder = Dense(256, output_dim)
    def forward(self, x, t, condition=None):
        x=x
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))
        # Encoding path
        h = torch.cat((self.dense2(x), self.dense1(embed)),dim=-1)
        
        h = self.block1(h)
        h = self.decoder(self.act(h))
        # Normalize output
        h = h / self.marginal_prob_std(t)[1][..., None]
        return h