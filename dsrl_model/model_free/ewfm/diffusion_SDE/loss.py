import torch
def loss_fn(model, x, marginal_prob_std, energy,alpha, eps=1e-3):
    if energy is None:
        random_t = torch.rand(x.shape[0], device=x.device,dtype = x.dtype) * (1. - eps) + eps  
        z = torch.randn_like(x)
        alpha_t, std = marginal_prob_std(random_t)
        alpha_t, std = alpha_t[:,None],std[:,None]
        perturbed_x = x * alpha_t+ z * std
        score = model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std + z)**2, dim=(1,)))
    else:
        random_t = torch.rand((x.shape[0],x.shape[1]), device=x.device,dtype = x.dtype) * (1. - eps) + eps  
        z = torch.randn_like(x)
        alpha_t, std = marginal_prob_std(random_t)
        alpha_t, std = alpha_t.unsqueeze(-1), std.unsqueeze(-1)
        perturbed_x = x * alpha_t+ z * std
        score = model(perturbed_x, random_t)
        guidance = (alpha * energy).softmax(dim=1).detach()
        individual_loss = torch.sum((score * std + z)**2, dim=(2,))
        loss = torch.mean(torch.sum(individual_loss * guidance,dim=1),dim=0)
    return loss

def loss_fn_bandit(model, x, marginal_prob_std, energy,alpha, eps=1e-3):
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    alpha_t, std = marginal_prob_std(random_t)
    perturbed_x = x * alpha_t[:, None] + z * std[:, None]
    score = model(perturbed_x, random_t)
    if energy is None:
        loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1,)))
    else:
        guidance = energy.mul(alpha).softmax(dim=0).squeeze()
        individual_loss = torch.sum((score * std[:, None] + z)**2, dim=(1,))
        loss = individual_loss.dot(guidance)
    return loss


