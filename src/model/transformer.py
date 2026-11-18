"""Transformer blocks used by the readable attention baseline."""

import torch
from torch import nn

from .attention import CausalSelfAttention
from .ffn import SwiGLUFFN
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-normalized attention and SwiGLU residual block."""

    def __init__(self, config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=1e-6)
        self.attention = CausalSelfAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )
        self.ffn_norm = RMSNorm(config.dim, eps=1e-6)
        self.ffn = SwiGLUFFN(
            dim=config.dim,
            hidden_dim=config.inter_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class AttentionBlock(nn.Module):
    """Compatibility wrapper for the earlier attention-only block."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=1e-6)
        self.attention = CausalSelfAttention(
            dim=dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attention(self.attn_norm(x))
