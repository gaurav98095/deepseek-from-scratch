"""Stateful compressed-cache inference for the educational MLA layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .mla import MLA


class CachedMLA(MLA):
    """MLA with a dynamically growing compressed KV cache.

    ``forward`` remains the full-sequence MLA reference implementation.
    ``prefill`` initializes the cache for a prompt, while ``decode_step``
    computes one new token using matrix absorption.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.max_seq_len = config.max_seq_len
        self.kv_cache: torch.Tensor | None = None
        self.pe_cache: torch.Tensor | None = None

    def reset_cache(self) -> None:
        """Start a new inference request without changing model weights."""
        self.kv_cache = None
        self.pe_cache = None

    @property
    def cache_length(self) -> int:
        """Return the number of positions currently stored."""
        if self.kv_cache is None:
            return 0
        return self.kv_cache.shape[1]

    def _append_cache(
        self,
        kv_latent: torch.Tensor,
        positional_key: torch.Tensor,
    ) -> None:
        """Append normalized latents and rotated positional keys."""
        if kv_latent.ndim != 3 or positional_key.ndim != 3:
            raise ValueError("cache tensors must have shape [batch, sequence, dim]")
        if kv_latent.shape[:2] != positional_key.shape[:2]:
            raise ValueError("cache tensors must have matching batch and sequence")
        if kv_latent.shape[-1] != self.kv_branch.kv_rank:
            raise ValueError("invalid KV latent dimension")
        if positional_key.shape[-1] != self.d_rope:
            raise ValueError("invalid positional-key dimension")
        if self.cache_length + kv_latent.shape[1] > self.max_seq_len:
            raise ValueError("KV cache is full")

        if self.kv_cache is None:
            self.kv_cache = kv_latent
            self.pe_cache = positional_key
            return

        if self.kv_cache.shape[0] != kv_latent.shape[0]:
            raise ValueError("Batch size changed")
        self.kv_cache = torch.cat([self.kv_cache, kv_latent], dim=1)
        self.pe_cache = torch.cat([self.pe_cache, positional_key], dim=1)

    @torch.no_grad()
    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Run full MLA on a prompt and populate its compressed cache."""
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch, sequence, dim]")
        if x.shape[1] == 0:
            raise ValueError("Prompt must contain at least one token")
        if x.shape[1] > self.max_seq_len:
            raise ValueError("Prompt exceeds maximum sequence length")
        if self.cache_length != 0:
            raise ValueError("Call reset_cache before prefill")

        output = super().forward(x)
        kv_latent, k_position, _, _ = self.kv_branch(x)
        k_position = self.rope(
            k_position.unsqueeze(1),
            start_pos=0,
        )[:, 0]
        self._append_cache(kv_latent, k_position)
        return output

    @torch.no_grad()
    def decode_step(self, x: torch.Tensor) -> torch.Tensor:
        """Decode exactly one hidden-state token using absorbed MLA."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("decode_step expects x with shape [batch, 1, dim]")
        position = self.cache_length
        if position >= self.max_seq_len:
            raise ValueError("KV cache is full")
        if self.kv_cache is not None and self.kv_cache.shape[0] != x.shape[0]:
            raise ValueError("Batch size changed")

        q_content, q_position = self.q_branch(x)
        kv_latent, k_position, _, _ = self.kv_branch(x)

        q_content = q_content.transpose(1, 2)
        q_position = q_position.transpose(1, 2)
        q_position = self.rope(q_position, start_pos=position)
        k_position = self.rope(
            k_position.unsqueeze(1),
            start_pos=position,
        )[:, 0]
        self._append_cache(kv_latent, k_position)

        up_weight = self.kv_branch.kv_up.weight.reshape(
            self.n_heads,
            self.d_content + self.d_value,
            self.kv_branch.kv_rank,
        )
        key_weight = up_weight[:, : self.d_content, :]
        value_weight = up_weight[:, self.d_content :, :]

        absorbed_query = torch.einsum(
            "bhsd,hdr->bhsr",
            q_content,
            key_weight,
        )
        content_scores = torch.einsum(
            "bhsr,btr->bhst",
            absorbed_query,
            self.kv_cache,
        )
        position_scores = torch.einsum(
            "bhsd,btd->bhst",
            q_position,
            self.pe_cache,
        )
        scores = (content_scores + position_scores) * self.scale
        weights = F.softmax(scores.float(), dim=-1).to(self.kv_cache.dtype)

        latent_context = torch.einsum(
            "bhst,btr->bhsr",
            weights,
            self.kv_cache,
        )
        attended = torch.einsum(
            "bhsr,hdr->bhsd",
            latent_context,
            value_weight,
        )
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(
            x.shape[0],
            1,
            self.n_heads * self.d_value,
        )
        return self.out_proj(attended)
