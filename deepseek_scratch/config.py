"""Configuration for the pre-MLA portion of the DeepSeek architecture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Model dimensions used before Multi-Head Latent Attention is introduced."""

    vocab_size: int
    context_length: int = 512
    embedding_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.context_length < 1 or self.embedding_dim < 1:
            raise ValueError("context_length and embedding_dim must be positive")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("num_layers and num_heads must be positive")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads
