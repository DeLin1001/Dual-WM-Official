import math
import torch
import torch.nn.functional as F

def diagonal_gaussian_kl(mean: torch.Tensor, logvar: torch.Tensor, free_bits: float=0.0) -> tuple[torch.Tensor, torch.Tensor]:
    if mean.shape != logvar.shape:
        raise ValueError(f'mean and logvar must have identical shapes, got {tuple(mean.shape)} and {tuple(logvar.shape)}')
    if mean.ndim < 2:
        raise ValueError('mean and logvar must include sample and latent dims')
    if free_bits < 0:
        raise ValueError(f'free_bits must be non-negative, got {free_bits}')
    per_dim = 0.5 * (mean.square() + logvar.exp() - 1 - logvar)
    raw_kl = per_dim.sum(dim=-1).mean()
    reduce_dims = tuple(range(per_dim.ndim - 1))
    per_dim_mean = per_dim.mean(dim=reduce_dims)
    objective = per_dim_mean.clamp_min(free_bits).sum()
    return (raw_kl, objective)
