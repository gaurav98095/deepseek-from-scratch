"""Consolidated DeepSeek-style model for training and cached inference."""

from __future__ import annotations

import torch
from torch import nn

from .config import DeepSeekConfig
from .embeddings import TokenEmbedding
from .ffn import SwiGLUFFN
from .mla_cache import CachedMLA
from .moe import DeepSeekMoE
from .rmsnorm import RMSNorm


class FinalDeepSeekBlock(nn.Module):
    """One pre-RMSNorm block with shared training and inference weights."""

    def __init__(self, config: DeepSeekConfig, layer_id: int) -> None:
        super().__init__()
        if not 0 <= layer_id < config.n_layers:
            raise ValueError("layer_id must be within the configured layer range")

        self.layer_id = layer_id
        self.is_moe = layer_id >= config.n_dense_layers
        self.attn_norm = RMSNorm(config.dim, eps=1e-6)
        self.attention = CachedMLA(config)
        self.ffn_norm = RMSNorm(config.dim, eps=1e-6)
        if self.is_moe:
            self.ffn: nn.Module = DeepSeekMoE(config)
        else:
            self.ffn = SwiGLUFFN(config.dim, config.inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

    @torch.no_grad()
    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Run this block on a prompt and populate its MLA cache."""
        x = x + self.attention.prefill(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

    @torch.no_grad()
    def decode_step(self, x: torch.Tensor) -> torch.Tensor:
        """Run this block for one token using its compressed MLA cache."""
        x = x + self.attention.decode_step(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

    def reset_cache(self) -> None:
        """Clear this block's dynamic inference state."""
        self.attention.reset_cache()


class TinyDeepSeek(nn.Module):
    """Final small model with full-sequence and cached execution paths."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.embed_tokens = TokenEmbedding(config)
        self.layers = nn.ModuleList(
            [
                FinalDeepSeekBlock(config, layer_id)
                for layer_id in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, eps=1e-6)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    @staticmethod
    def _validate_config(config: DeepSeekConfig) -> None:
        if not 0 <= config.n_dense_layers <= config.n_layers:
            raise ValueError("Invalid number of dense layers")
        if not 1 <= config.n_activated_experts <= config.n_routed_experts:
            raise ValueError("Invalid number of activated experts")

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("Expected input_ids with shape [batch, sequence]")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids cannot be empty")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds maximum length")

    def _project(
        self,
        hidden_states: torch.Tensor,
        *,
        return_hidden: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if return_hidden:
            return logits, hidden_states
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run full causal attention, optionally returning final hidden states."""
        self._validate_input_ids(input_ids)
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self._project(hidden_states, return_hidden=return_hidden)

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Process a prompt, reset all caches, and return prompt logits."""
        self._validate_input_ids(input_ids)
        self.reset_cache()
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer.prefill(hidden_states)
        return self._project(hidden_states, return_hidden=return_hidden)

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Process exactly one token using each block's compressed KV cache."""
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step expects input_ids with shape [batch, 1]")
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer.decode_step(hidden_states)
        return self._project(hidden_states, return_hidden=return_hidden)

    def reset_cache(self) -> None:
        """Clear dynamic KV caches without changing learned parameters."""
        for layer in self.layers:
            layer.reset_cache()

    @property
    def cache_length(self) -> int:
        """Return the cached sequence length in the first attention block."""
        return self.layers[0].attention.cache_length
