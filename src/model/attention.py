"""Simple self-attention components used by the educational decoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention for tensors shaped ``[B, T, D]``."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()

        # Every head receives an equal slice of the embedding dimension, so all
        # dimensions must be positive and `embed_dim` must divide evenly.
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        # D = H * Dh. For the reference config: 7168 = 128 * 56.
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # `causal` controls whether token t may read tokens after position t.
        self.causal = causal

        # These four matrices are learned independently. Q and K decide which
        # tokens interact, V carries content, and WO mixes the merged heads.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout applies to normalized attention probabilities during training.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Project, attend, merge heads, and project back to ``embed_dim``."""

        # Attention expects one vector per token: batch B, sequence T, width D.
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, embed_dim)")
        if hidden_states.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected embedding size {self.embed_dim}, got {hidden_states.shape[-1]}"
            )
        if hidden_states.shape[1] == 0:
            raise ValueError("sequence length must be greater than zero")

        # Only B and T vary at runtime; D was fixed when this module was built.
        batch, sequence, _ = hidden_states.shape

        # Each projection starts as [B, T, D]. `view` exposes H independent
        # heads, and `transpose` puts H before T for batched matrix products:
        # [B, T, D] -> [B, T, H, Dh] -> [B, H, T, Dh].
        query = self.q_proj(hidden_states).view(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Every query is compared with every key in the same head. Dividing by
        # sqrt(Dh) prevents large dot products from saturating the softmax:
        # [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T].
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            # The lower triangle contains positions at or before each query.
            # Its [T, T] shape broadcasts over the batch and head dimensions.
            visible = torch.ones(
                sequence,
                sequence,
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril()

            # Hidden future positions receive the smallest representable value,
            # which makes their probability effectively zero after softmax.
            scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            # In this API True means visible and False means blocked. The mask
            # must be broadcastable to [B, H, T, T].
            if attention_mask.dtype is not torch.bool:
                raise TypeError("attention_mask must have boolean dtype")
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        # Softmax converts each query row into probabilities over key positions.
        # The probabilities sum to one before dropout is applied.
        weights = self.dropout(torch.softmax(scores, dim=-1))

        # Use those probabilities to take a weighted sum of value vectors:
        # [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh].
        attended = torch.matmul(weights, value)

        # Return from separate heads to one vector per token. `contiguous` makes
        # the transposed memory layout safe to flatten: [B, H, T, Dh] -> [B, T, D].
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, self.embed_dim)

        # WO is not a reshape: it is a learned D x D transformation that mixes
        # information from all heads before the decoder's residual addition.
        output = self.out_proj(attended)

        # Attention probabilities are optional because training normally needs
        # only the transformed hidden states; tests and visualizations may need both.
        return (output, weights) if return_attention else output


class SelfAttention(MultiHeadAttention):
    """Non-causal single-head attention kept as a simple specialization."""

    def __init__(self, embed_dim: int, dropout: float = 0.0, bias: bool = True) -> None:
        # One head and no causal mask provide the simplest self-attention case.
        super().__init__(embed_dim, 1, dropout=dropout, bias=bias, causal=False)


class CausalSelfAttention(MultiHeadAttention):
    """Causal multi-head attention used inside every decoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        # Decoder language modeling needs multiple heads plus future-token masking.
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, causal=True)


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute position vectors for a fixed maximum sequence length."""

    def __init__(self, max_sequence_length: int, embed_dim: int) -> None:
        super().__init__()

        # The embedding table needs one learned row for every allowed position.
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(max_sequence_length, embed_dim)

    def forward(self, sequence_length: int, *, device: torch.device | None = None) -> Tensor:
        """Return position vectors with shape ``[1, sequence_length, embed_dim]``."""

        # Reject invalid lengths before indexing the finite embedding table.
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if sequence_length > self.max_sequence_length:
            raise ValueError("sequence length exceeds max_sequence_length")

        # The leading size-one batch axis lets this tensor broadcast over every
        # example when it is added to token embeddings shaped [B, T, D].
        positions = torch.arange(sequence_length, device=device)
        return self.embedding(positions).unsqueeze(0)
