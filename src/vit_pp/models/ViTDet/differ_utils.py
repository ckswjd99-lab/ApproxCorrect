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

def diff_mlp_gelu(
    mlp: nn.Module,
    x: torch.Tensor,
    dx: torch.Tensor
) -> torch.Tensor:
    """
    Computes the first-order Taylor approximation for MLP with GELU activation.
    Calculates dy for MLP(x + dx) ≈ MLP(x) + dy.
    """

    # First linear layer
    w1, b1 = mlp.fc1.weight, mlp.fc1.bias
    z = F.linear(x, w1, b1)
    dz = F.linear(dx, w1, None)
    
    # GELU'(z) = 0.5 * (1 + erf(z/sqrt(2))) + (z / sqrt(2*pi)) * exp(-0.5*z^2)
    gelu_derivative_theory = 0.5 * (1.0 + torch.erf(z / (2**0.5))) + \
                      (z / ((2 * torch.pi)**0.5)) * torch.exp(-0.5 * z**2)

    # Prune
    prune_rate = 0.0
    threshold = torch.quantile(gelu_derivative_theory.abs(), prune_rate, dim=-1)[:, :, :, None]
    alive_mask = (gelu_derivative_theory.abs() >= threshold).float()

    # Simple: GELU(z + dz) - GELU(z)
    gelu_derivative = F.gelu(z + dz) - F.gelu(z)
    gelu_derivative = gelu_derivative * alive_mask
    
    # d(GELU(z)) = GELU'(z) * dz
    # dgelu_z = gelu_derivative * dz
    dgelu_z = gelu_derivative
    
    # Second linear layer
    w2, b2 = mlp.fc2.weight, mlp.fc2.bias
    dy = F.linear(dgelu_z, w2, None)
    
    return dy