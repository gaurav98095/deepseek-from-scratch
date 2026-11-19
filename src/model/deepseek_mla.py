"""Dense language model using the naive MLA attention layer."""

import torch
from torch import nn

from .config import DeepSeekConfig
from .embeddings import TokenEmbedding
from .rmsnorm import RMSNorm
from .transformer import DeepSeekBlock


class DeepSeekMLA(nn.Module):
    """Dense Transformer with MLA attention and SwiGLU FFNs."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(config)
        self.blocks = nn.ModuleList(
            [DeepSeekBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=1e-6)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Expected input_ids with shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds maximum length")

        hidden_states = self.embed_tokens(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def reset_cache(self) -> None:
        """Clear the attention cache in every Transformer block."""
        for block in self.blocks:
            block.reset_cache()

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Process a prompt and populate every block's compressed cache."""
        if input_ids.ndim != 2:
            raise ValueError("Expected input_ids with shape [batch, sequence]")
        if input_ids.shape[1] == 0:
            raise ValueError("Prompt must contain at least one token")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds maximum length")

        hidden_states = self.embed_tokens(input_ids)
        for block in self.blocks:
            hidden_states = block.prefill(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def decode_step(self, new_token_ids: torch.Tensor) -> torch.Tensor:
        """Process one token using the compressed cache in every block."""
        if new_token_ids.ndim != 2 or new_token_ids.shape[1] != 1:
            raise ValueError("decode_step expects input_ids with shape [batch, 1]")

        hidden_states = self.embed_tokens(new_token_ids)
        for block in self.blocks:
            hidden_states = block.decode_step(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
