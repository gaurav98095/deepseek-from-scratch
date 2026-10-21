"""Attention components used by the educational decoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# Multi-head latent attention caches its compressed KV representation as [B, T, R].
LatentKVCache = Tensor


class CausalMultiHeadLatentAttention(nn.Module):
    """Causal multi-head attention with a compressed per-token KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if embed_dim <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("embedding, head count, and head dimension must be positive")
        if q_lora_rank <= 0 or kv_lora_rank <= 0:
            raise ValueError("query and KV latent ranks must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.q_a_proj = nn.Linear(embed_dim, q_lora_rank, bias=False)
        self.q_layernorm = nn.LayerNorm(q_lora_rank, eps=norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * head_dim, bias=bias)
        self.kv_a_proj = nn.Linear(embed_dim, kv_lora_rank, bias=False)
        self.kv_layernorm = nn.LayerNorm(kv_lora_rank, eps=norm_eps)
        self.kv_b_proj = nn.Linear(kv_lora_rank, 2 * num_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: LatentKVCache | None = None,
        *,
        return_attention: bool = False,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, LatentKVCache]:
        """Attend with expanded K/V vectors while retaining only their latent."""

        if return_attention and use_cache:
            raise ValueError("return_attention and use_cache cannot both be enabled")
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, embed_dim)")
        if hidden_states.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected embedding size {self.embed_dim}, got {hidden_states.shape[-1]}"
            )
        if hidden_states.shape[1] == 0:
            raise ValueError("sequence length must be greater than zero")

        batch, sequence, _ = hidden_states.shape
        query = self.q_b_proj(self.q_layernorm(self.q_a_proj(hidden_states)))
        query = query.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

        kv_latent = self.kv_layernorm(self.kv_a_proj(hidden_states))
        past_length = 0
        if past_key_value is not None:
            if past_key_value.ndim != 3:
                raise ValueError("cached KV latent must have shape [B, T, kv_lora_rank]")
            if (
                past_key_value.shape[0] != batch
                or past_key_value.shape[2] != self.kv_lora_rank
            ):
                raise ValueError("cached KV latent does not match this attention layer")
            if (
                past_key_value.device != kv_latent.device
                or past_key_value.dtype != kv_latent.dtype
            ):
                raise ValueError("cached KV latent must match the current device and dtype")
            past_length = past_key_value.shape[1]
            kv_latent = torch.cat((past_key_value, kv_latent), dim=1)

        expanded_kv = self.kv_b_proj(kv_latent).view(
            batch,
            past_length + sequence,
            2,
            self.num_heads,
            self.head_dim,
        )
        key = expanded_kv[:, :, 0].transpose(1, 2)
        value = expanded_kv[:, :, 1].transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_length = key.shape[2]
        visible = torch.ones(
            sequence,
            key_length,
            dtype=torch.bool,
            device=hidden_states.device,
        ).tril(diagonal=past_length)
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.dtype is not torch.bool:
                raise TypeError("attention_mask must have boolean dtype")
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch,
            sequence,
            self.num_heads * self.head_dim,
        )
        output = self.out_proj(attended)

        if use_cache:
            return output, kv_latent
        if return_attention:
            return output, weights
        return output


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
