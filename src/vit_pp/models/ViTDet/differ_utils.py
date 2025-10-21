import torch
import torch.nn as nn
from typing import Tuple, Union, Dict
import torch.nn.functional as F

def diff_convolution(
    module_conv: nn.Conv2d,
    dx: torch.Tensor
):
    dx = F.conv2d(dx, module_conv.proj.weight, None, stride=module_conv.proj.stride, padding=module_conv.proj.padding)
    return dx

def diff_layernorm(ln_layer: nn.LayerNorm, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """
    Computes the first-order Taylor approximation for LayerNorm.
    Calculates dy for LN(x + dx) ≈ LN(x) + dy.
    """
    gamma = ln_layer.weight
    eps = ln_layer.eps
    normalized_shape = ln_layer.normalized_shape
    
    axis = tuple(range(x.dim() - len(normalized_shape), x.dim()))

    mu = x.mean(dim=axis, keepdim=True)
    var = x.var(dim=axis, unbiased=False, keepdim=True)
    sigma_prime = torch.sqrt(var + eps)
    
    x_hat = (x - mu) / sigma_prime

    mean_dx = dx.mean(dim=axis, keepdim=True)
    
    proj_dx = (x_hat * dx).mean(dim=axis, keepdim=True)

    # dy = (gamma / sigma') * (dx - mean(dx) - x_hat * mean(x_hat * dx))
    dx_norm = (gamma / sigma_prime) * (dx - mean_dx - x_hat * proj_dx)
    
    return dx_norm
