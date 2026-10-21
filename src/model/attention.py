"""Simple self-attention components used by the educational decoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# One decoder layer caches its projected keys and values as [B, H, T, Dh].
KVCache = tuple[Tensor, Tensor]


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
        past_key_value: KVCache | None = None,
        *,
        return_attention: bool = False,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, KVCache]:
        """Project, attend, merge heads, and project back to ``embed_dim``."""

        # Cache output and attention-weight output are separate teaching/debugging
        # modes; keeping them exclusive makes each return shape unambiguous.
        if return_attention and use_cache:
            raise ValueError("return_attention and use_cache cannot both be enabled")
        if use_cache and not self.causal:
            raise ValueError("KV caching is supported only for causal attention")

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

        # During incremental inference, K and V for earlier tokens have already
        # been projected by this layer. Append only the new token projections.
        past_length = 0
        if past_key_value is not None:
            if not self.causal:
                raise ValueError("past_key_value is supported only for causal attention")
            past_key, past_value = past_key_value
            expected_prefix = (batch, self.num_heads)
            if past_key.ndim != 4 or past_value.ndim != 4:
                raise ValueError("cached keys and values must have shape [B, H, T, Dh]")
            if past_key.shape != past_value.shape:
                raise ValueError("cached keys and values must have identical shapes")
            if past_key.shape[:2] != expected_prefix or past_key.shape[3] != self.head_dim:
                raise ValueError("cached keys and values do not match this attention layer")
            if (
                past_key.device != key.device
                or past_value.device != value.device
                or past_key.dtype != key.dtype
                or past_value.dtype != value.dtype
            ):
                raise ValueError("cached keys and values must match the current device and dtype")

            past_length = past_key.shape[2]
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        # `present_key_value` contains every position available after this call
        # and becomes this layer's input cache on the next generation step.
        present_key_value = (key, value)
        key_length = key.shape[2]

        # Every query is compared with every key in the same head. Dividing by
        # sqrt(Dh) prevents large dot products from saturating the softmax:
        # [B, H, Q, Dh] @ [B, H, Dh, K] -> [B, H, Q, K].
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            # A cached query starts at absolute position `past_length`, so the
            # diagonal offset lets query i see keys through past_length + i.
            # For a one-token cached step, [1, K] is entirely visible.
            visible = torch.ones(
                sequence,
                key_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril(diagonal=past_length)

            # Hidden future positions receive the smallest representable value,
            # which makes their probability effectively zero after softmax.
            scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            # In this API True means visible and False means blocked. The mask
            # must be broadcastable to [B, H, query_length, key_length].
            if attention_mask.dtype is not torch.bool:
                raise TypeError("attention_mask must have boolean dtype")
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        # Softmax converts each query row into probabilities over key positions.
        # The probabilities sum to one before dropout is applied.
        weights = self.dropout(torch.softmax(scores, dim=-1))

        # Use those probabilities to take a weighted sum of value vectors:
        # [B, H, Q, K] @ [B, H, K, Dh] -> [B, H, Q, Dh].
        attended = torch.matmul(weights, value)

        # Return from separate heads to one vector per token. `contiguous` makes
        # the transposed memory layout safe to flatten: [B, H, T, Dh] -> [B, T, D].
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, self.embed_dim)

        # WO is not a reshape: it is a learned D x D transformation that mixes
        # information from all heads before the decoder's residual addition.
        output = self.out_proj(attended)

        # Inference requests the updated cache, while tests and visualizations may
        # request attention probabilities. Ordinary training returns only output.
        if use_cache:
            return output, present_key_value
        if return_attention:
            return output, weights
        return output


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

    def forward(
        self,
        sequence_length: int,
        *,
        start_position: int = 0,
        device: torch.device | None = None,
    ) -> Tensor:
        """Return position vectors with shape ``[1, sequence_length, embed_dim]``."""

        # Reject invalid lengths before indexing the finite embedding table.
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if start_position < 0:
            raise ValueError("start_position must be non-negative")
        if start_position + sequence_length > self.max_sequence_length:
            raise ValueError("position range exceeds max_sequence_length")

        # The leading size-one batch axis lets this tensor broadcast over every
        # example when it is added to token embeddings shaped [B, T, D].
        positions = torch.arange(
            start_position,
            start_position + sequence_length,
            device=device,
        )
        return self.embedding(positions).unsqueeze(0)
