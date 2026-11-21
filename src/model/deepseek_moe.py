"""Complete small DeepSeek-style Transformer with MLA and MoE FFNs."""

from __future__ import annotations

import torch
from torch import nn

from .config import DeepSeekConfig
from .embeddings import TokenEmbedding
from .ffn import SwiGLUFFN
from .mla import MLA
from .moe import DeepSeekMoE
from .rmsnorm import RMSNorm


class DeepSeekMoEBlock(nn.Module):
    """MLA block with a dense or sparse feed-forward sublayer."""

    def __init__(self, config: DeepSeekConfig, layer_id: int) -> None:
        super().__init__()
        if not 0 <= layer_id < config.n_layers:
            raise ValueError("layer_id must be within the configured layer range")

        self.layer_id = layer_id
        self.is_moe = layer_id >= config.n_dense_layers
        self.attn_norm = RMSNorm(config.dim, eps=1e-6)
        self.attention = MLA(config)
        self.ffn_norm = RMSNorm(config.dim, eps=1e-6)
        self.ffn: nn.Module
        if self.is_moe:
            self.ffn = DeepSeekMoE(config)
        else:
            self.ffn = SwiGLUFFN(
                dim=config.dim,
                hidden_dim=config.inter_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TinyDeepSeek(nn.Module):
    """Trainable MLA language model with dense early and MoE later blocks."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DeepSeekMoEBlock(config, layer_id=layer_id)
                for layer_id in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, eps=1e-6)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Expected input_ids with shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds maximum length")

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
