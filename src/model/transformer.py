"""Transformer blocks used by the readable attention baseline."""

import torch
from torch import nn

from .attention import CausalSelfAttention
from .rmsnorm import RMSNorm


class AttentionBlock(nn.Module):
    """Pre-normalized residual block containing causal self-attention."""

    def __init__(self, dim: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attention = CausalSelfAttention(dim, n_heads, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attention(self.attn_norm(x))
