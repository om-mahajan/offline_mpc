import os
import gym
import d4rl
import scipy
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.unconditional_model import Bandit_MlpScoreNet
from utils import bandit_get_args
from dataset.dataset import Toy_dataset


def train(args, score_model, score_model_positive, data_loader, start_epoch=0):
    n_epochs = 359
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    optimizer = Adam(list(score_model.parameters()) + list(score_model_positive.parameters()), lr=1e-4)
    
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        # training behavior
        for data in data_loader:
            data = {k: d.to(args.device) for k, d in data.items()}
            x = data["a"]
            B = x.shape[0]
            e = data["e"]
            binary_label = torch.rand(B, device=x.device) < e.reshape(-1).exp()
            loss_regular = loss_fn(score_model, x, args.marginal_prob_std_fn, energy=None, alpha=args.alpha)
            loss_positive = loss_fn(score_model_positive, x[binary_label == 1, :], args.marginal_prob_std_fn, energy=None, alpha=args.alpha)
            loss = loss_regular + loss_positive
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        if epoch % 50 == 49 and args.save_model:
            torch.save(score_model.state_dict(), os.path.join("./models", str(args.expid), "ckpt{}reg.pth".format(epoch+1)))
            torch.save(score_model_positive.state_dict(), os.path.join("./models", str(args.expid), "ckpt{}pos.pth".format(epoch+1)))
        args.writer.add_scalar("actor/loss", avg_loss / num_items, global_step=epoch)

def main(args):
    for dir in ["./models", "./toylogs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", str(args.expid))):
        os.makedirs(os.path.join("./models", str(args.expid)))
    writer = SummaryWriter("./toylogs/" + str(args.expid))
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn

    dataset = Toy_dataset(args.env)
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    score_model = Bandit_MlpScoreNet(input_dim=0+dataset.datadim, output_dim=dataset.datadim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model_positive = Bandit_MlpScoreNet(input_dim=0+dataset.datadim, output_dim=dataset.datadim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)


    print("training")
    train(args, score_model, score_model_positive, data_loader, start_epoch=0)
    print("finished")

if __name__ == "__main__":
    args = bandit_get_args()
    main(args)