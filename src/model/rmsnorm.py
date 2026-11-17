"""Root Mean Square Normalization."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize each token across its final hidden dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self.eps)
        x_norm = (x_float * inv_rms).to(original_dtype)
        return x_norm * self.weight
