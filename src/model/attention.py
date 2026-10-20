"""Standard self-attention building blocks.

This stage intentionally implements ordinary multi-head attention. It does not
include KV caching, multi-query attention, or latent attention variants.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _validate_hidden_states(hidden_states: Tensor, embed_dim: int) -> None:
    # Attention expects a batch of token sequences, not a single vector.
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (batch, sequence, embed_dim)")
    if hidden_states.shape[-1] != embed_dim:
        raise ValueError(f"expected embedding size {embed_dim}, got {hidden_states.shape[-1]}")
    if hidden_states.shape[1] == 0:
        raise ValueError("sequence length must be greater than zero")


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    Inputs use the shape ``(batch, sequence, embed_dim)``. A boolean
    ``attention_mask`` may be broadcast to ``(batch, heads, query, key)``;
    ``True`` means that a position is visible. When ``causal=True``, future
    key positions are hidden automatically.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, causal: bool = False) -> None:
        super().__init__()
        # These checks catch shape errors before projection parameters are used.
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        # Separate projections let each head learn query, key, and value features.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, states: Tensor) -> Tensor:
        batch, sequence, _ = states.shape
        # Move heads before sequence so matrix multiplication runs per head.
        return states.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        _validate_hidden_states(hidden_states, self.embed_dim)
        batch, sequence, _ = hidden_states.shape

        # Self-attention uses the same hidden states to produce Q, K, and V.
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        # Scaling keeps dot products numerically stable as head dimensions grow.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            # A lower-triangular mask prevents a token from seeing future tokens.
            causal_mask = torch.ones(sequence, sequence, dtype=torch.bool, device=hidden_states.device).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            # Caller masks can hide padding or other unavailable positions.
            if attention_mask.dtype is not torch.bool:
                raise TypeError("attention_mask must have boolean dtype")
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        # Softmax converts scores into probabilities over key positions.
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        # Weighted values are the information selected by each query token.
        attended = torch.matmul(weights, value)
        # Recombine heads back into the model's hidden dimension.
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, self.embed_dim)
        output = self.out_proj(attended)
        if return_attention:
            return output, weights
        return output


class SelfAttention(MultiHeadAttention):
    """Ordinary single-head self-attention without a causal restriction."""

    def __init__(self, embed_dim: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads=1, dropout=dropout, bias=bias, causal=False)


class CausalSelfAttention(MultiHeadAttention):
    """Multi-head self-attention that cannot read future tokens."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, causal=True)


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings for a fixed context length."""

    def __init__(self, max_sequence_length: int, embed_dim: int) -> None:
        super().__init__()
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(max_sequence_length, embed_dim)

    def forward(self, sequence_length: int, *, device: torch.device | None = None) -> Tensor:
        # Position IDs are generated at runtime so the same table works for any shorter sequence.
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if sequence_length > self.max_sequence_length:
            raise ValueError("sequence_length exceeds max_sequence_length")
        positions = torch.arange(sequence_length, device=device)
        # The leading dimension broadcasts position vectors across the batch.
        return self.embedding(positions).unsqueeze(0)
