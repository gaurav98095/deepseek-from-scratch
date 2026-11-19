"""Transformer blocks used by the readable attention baseline."""

import torch
from torch import nn

from .attention import CausalSelfAttention
from .ffn import SwiGLUFFN
from .mla_cache import CachedMLA
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


class DeepSeekBlock(nn.Module):
    """Dense Transformer block with trainable and cached MLA paths."""

    def __init__(self, config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=1e-6)
        self.attention = CachedMLA(config)
        self.ffn_norm = RMSNorm(config.dim, eps=1e-6)
        self.ffn = SwiGLUFFN(
            dim=config.dim,
            hidden_dim=config.inter_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

    def reset_cache(self) -> None:
        self.attention.reset_cache()

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block over a prompt and populate its attention cache."""
        hidden_states = x + self.attention.prefill(self.attn_norm(x))
        return hidden_states + self.ffn(self.ffn_norm(hidden_states))

    def decode_step(self, x: torch.Tensor) -> torch.Tensor:
        """Run one hidden-state token through the cached block."""
        hidden_states = x + self.attention.decode_step(self.attn_norm(x))
        return hidden_states + self.ffn(self.ffn_norm(hidden_states))
