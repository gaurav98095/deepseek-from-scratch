"""Pre-MLA DeepSeek model: token embeddings, causal MHA, and transformer blocks."""

import torch
from torch import nn

from .config import ModelConfig


class CausalMultiHeadAttention(nn.Module):
    """Standard causal multi-head self-attention used before MLA."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = hidden_states.shape
        if sequence_length > self.causal_mask.size(-1):
            raise ValueError("sequence length exceeds context_length")
        qkv = self.qkv(hidden_states).view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        scores = scores.masked_fill(~self.causal_mask[:, :, :sequence_length, :sequence_length], torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = (weights @ value).transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        return self.resid_dropout(self.out_proj(attended))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalMultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.norm1(hidden_states))
        return hidden_states + self.feed_forward(self.norm2(hidden_states))


class DeepSeekForCausalLM(nn.Module):
    """Decoder-only language model up to and including standard MHA."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("sequence length exceeds context_length")
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
