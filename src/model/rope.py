"""Rotary position embeddings for query and key tensors."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Apply interleaved-pair rotary position embeddings."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("RoPE dimension must be positive and even")
        if max_seq_len <= 0 or theta <= 0:
            raise ValueError("max_seq_len and theta must be positive")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        pair_indices = torch.arange(0, dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (pair_indices / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer(
            "cos_cached",
            angles.cos()[None, None, :, :],
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            angles.sin()[None, None, :, :],
            persistent=False,
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Rotate x shaped [batch, heads, sequence, dim]."""
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError("Expected x with shape [batch, heads, sequence, dim]")
        if start_pos < 0 or start_pos + x.shape[-2] > self.max_seq_len:
            raise ValueError("requested positions exceed max_seq_len")

        end_pos = start_pos + x.shape[-2]
        cos = self.cos_cached[:, :, start_pos:end_pos, :].to(
            device=x.device,
            dtype=x.dtype,
        )
        sin = self.sin_cached[:, :, start_pos:end_pos, :].to(
            device=x.device,
            dtype=x.dtype,
        )

        pairs = x.reshape(*x.shape[:-1], self.dim // 2, 2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rotated.flatten(-2)
