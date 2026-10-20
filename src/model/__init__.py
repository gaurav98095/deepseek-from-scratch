"""Public model API for the decoder-only model."""

# Re-export building blocks so callers do not need to know internal file layout.
from .attention import (
    CausalSelfAttention,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    SelfAttention,
)
from .config import ModelConfig, default_config_path, load_default_config
from .deepseek import DecoderBlock, DeepSeek, DeepSeekOutput, DeepSeekTrainer, TrainingConfig

__all__ = [
    "CausalSelfAttention",
    "LearnedPositionalEmbedding",
    "MultiHeadAttention",
    "SelfAttention",
    "ModelConfig",
    "default_config_path",
    "load_default_config",
    "DecoderBlock",
    "DeepSeek",
    "DeepSeekOutput",
    "DeepSeekTrainer",
    "TrainingConfig",
]
