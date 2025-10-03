import torch
def marginal_prob_std(t, schedule,device="cuda"):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """    
    t = torch.tensor(t, device=device)
    if schedule == "linear":
        beta_1 = 20.0
        beta_0 = 0.1
        log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
        alpha_t = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    elif schedule == "OT":
        alpha_t = (1 - t)
        std = t
    else:
        raise EOFError
    return alpha_t, std