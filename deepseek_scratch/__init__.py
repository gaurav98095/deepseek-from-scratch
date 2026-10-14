"""A small, educational DeepSeek-style language model implementation."""

from .config import ModelConfig
from .model import DeepSeekForCausalLM
from .tokenizer import CharacterTokenizer

__all__ = ["CharacterTokenizer", "DeepSeekForCausalLM", "ModelConfig"]
