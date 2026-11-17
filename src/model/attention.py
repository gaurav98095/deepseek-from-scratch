"""Readable multi-head causal self-attention."""

import math

import torch
from torch import nn

from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention that cannot read future positions."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or n_heads <= 0 or max_seq_len <= 0:
            raise ValueError("dim, n_heads, and max_seq_len must be positive")
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

        future = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("future_mask", future, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        x = x.view(batch, time, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch, sequence, dim]")

        batch, time, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected hidden size {self.dim}, got {channels}")
        if time > self.future_mask.size(0):
            raise ValueError("sequence is longer than max_seq_len")

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        q = self.rope(q, start_pos=start_pos)
        k = self.rope(k, start_pos=start_pos)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.future_mask[:time, :time]
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch, time, self.dim)
        return self.out_proj(attended)
