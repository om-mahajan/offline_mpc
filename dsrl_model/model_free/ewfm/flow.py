import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import functools
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.loss import loss_fn
from diffusion_SDE.model import ScoreNet
from utils import get_args
from dataset.dataset import D4RL_dataset


# -----------------------
# Conditional Flow Matching Trainer
# -----------------------
def train_conditional_flow(args, score_model, dataset, start_epoch=0):
    """
    Train a conditional flow matching model that learns to generate safe trajectories.
    """
    writer = args.writer
    device = args.device
    score_model.train()

    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-4)
    num_epochs = args.num_epochs
    bs = args.batch_size
    nw = args.num_workers
    unconditional_prob = 0.2  # classifier-free guidance dropout prob

    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=nw)
    tqdm_step = tqdm.trange(start_epoch, num_epochs, 1)
    avg_loss = 0.0

    for epoch in tqdm_step:
        score_model.train()
        running_loss, num_items = 0.0, 0

        for batch in data_loader:
            data = {k: v.to(device) for k, v in batch.items()}
            s, a, y = data["s"], data["a"], data["label"]  # label = 1 (safe), 0 (unsafe), -1 (unlabelled)

            # Replace unlabeled with safe for now (optional: semi-supervised weighting)
            y = torch.where(y == -1, torch.ones_like(y), y)

            # classifier-free guidance dropout
            mask = (torch.rand(y.shape, device=device) < unconditional_prob)
            # Classifier-free guidance dropout: remove condition sometimes
            y_cond = torch.where(mask, torch.full_like(y, -1), y)  # -1 = unconditional

            # Concatenate state and action as input (as in original diffusion code)
            score_model.condition = torch.cat([s, y_cond.unsqueeze(-1)], dim=-1)

            # Compute flow-matching (diffusion) loss
            loss = loss_fn(
                score_model,
                a,
                args.marginal_prob_std_fn,
                energy=None,   # we are not using QGPO or energy-based weighting
                alpha=args.alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            score_model.condition = None
            running_loss += loss.item()
            num_items += 1

        avg_loss = running_loss / max(num_items, 1)
        tqdm_step.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")
        writer.add_scalar("loss/train", avg_loss, global_step=epoch)

        # Optional: save model checkpoint every few epochs
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join("./models_rl", str(args.expid), f"cfm_ckpt_{epoch+1}.pth")
            torch.save(score_model.state_dict(), ckpt_path)

    print("✅ Training complete. Final avg loss:", avg_loss)


# -----------------------
# Inference with Guidance
# -----------------------
def guided_sampling(score_model, states, args, guidance_scale=2.0):
    """
    Generate actions from the conditional flow model using classifier-free guidance.
    """
    with torch.no_grad():
        score_model.eval()
        s = torch.tensor(states, device=args.device).float()

        # Two passes: unconditional and safe-conditional
        safe_cond = torch.ones((s.shape[0], 1), device=args.device)
        uncond = torch.full_like(safe_cond, -1)

        score_model.condition = torch.cat([s, safe_cond], dim=-1)
        a_safe = score_model.sample(s, sample_per_state=1, diffusion_steps=args.diffusion_steps, is_numpy=False)

        score_model.condition = torch.cat([s, uncond], dim=-1)
        a_uncond = score_model.sample(s, sample_per_state=1, diffusion_steps=args.diffusion_steps, is_numpy=False)

        # Classifier-free blending
        a_guided = a_uncond + guidance_scale * (a_safe - a_uncond)
        score_model.condition = None

        return a_guided.cpu()


# -----------------------
# Main Training Pipeline
# -----------------------
def main(args):
    # Prepare directories
    os.makedirs("./models_rl", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs(os.path.join("./models_rl", str(args.expid)), exist_ok=True)

    writer = SummaryWriter(os.path.join("./logs", str(args.expid)))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    # Setup environment
    env = gym.make(args.env)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # Setup scheduler
    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, schedule=args.schedule, device=args.device
    )
    args.marginal_prob_std_fn = marginal_prob_std_fn
    args.writer = writer

    # Model and dataset
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    score_model = ScoreNet(
        input_dim=state_dim + action_dim + 1,  # +1 for safety label conditioning
        output_dim=action_dim,
        marginal_prob_std=marginal_prob_std_fn,
        args=args
    ).to(args.device)

  

    # Train conditional flow matching model
    train_conditional_flow(args, score_model, dataset)

    # Save final model
    final_ckpt = os.path.join("./models_rl", str(args.expid), "cfm_final.pth")
    torch.save(score_model.state_dict(), final_ckpt)
    print(f"Model saved at: {final_ckpt}")

    # Optional: test sampling
    example_states = dataset.states[:8]
    guided_actions = guided_sampling(score_model, example_states, args)
    print("Generated guided actions:", guided_actions.shape)

    writer.close()


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    args = get_args()

    # Add or override key hyperparameters for this version
    args.num_epochs = getattr(args, "num_epochs", 500)
    args.batch_size = getattr(args, "batch_size", 4096)
    args.num_workers = getattr(args, "num_workers", 6)
    args.save_interval = getattr(args, "save_interval", 50)
    args.alpha = getattr(args, "alpha", 1.0)
    args.seed_per_evaluation = getattr(args, "seed_per_evaluation", 0)

    print(f"Training Conditional Flow Model for {args.env} (expid={args.expid})")
    print(f"Alpha: {args.alpha}, Epochs: {args.num_epochs}")

    main(args)
