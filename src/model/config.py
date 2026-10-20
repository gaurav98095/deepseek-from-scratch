"""Configuration for the educational DeepSeek model implementation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """All dimensions and architectural defaults used by model components."""

    # Metadata describes which educational model profile the values belong to.
    model_name: str = "DeepSeek-V3"
    model_type: str = "decoder_only"
    implementation_profile: str = "standard_attention_educational"
    vocab_size: int = 129280
    max_position_embeddings: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    head_dim: int = 56
    intermediate_size: int = 18432
    hidden_activation: str = "silu"
    attention_dropout: float = 0.0
    attention_bias: bool = False
    position_embedding_type: str = "learned_absolute"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    torch_dtype: str = "bfloat16"
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        # Standard attention must exactly partition hidden features across heads.
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("standard attention requires num_key_value_heads == num_attention_heads")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive")
        if self.max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be positive")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """Load a model configuration from a JSON file."""

        # JSON keeps architecture changes reviewable without editing Python code.
        with Path(path).open(encoding="utf-8") as config_file:
            values: dict[str, Any] = json.load(config_file)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this configuration."""

        # asdict makes checkpoints and experiment logs easy to serialize later.
        return asdict(self)


def default_config_path() -> Path:
    """Return the checked-in DeepSeek-V3 configuration path."""

    # Resolve from this file so the loader works from any current working directory.
    return Path(__file__).resolve().parents[2] / "configs" / "deepseek_v3.json"


def load_default_config() -> ModelConfig:
    """Load the repository's default model configuration."""

    # Keep the default model construction consistent across training and notebooks.
    return ModelConfig.from_json(default_config_path())
