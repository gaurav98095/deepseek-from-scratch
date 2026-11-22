"""Final educational DeepSeek-style model components."""

from .config import DeepSeekConfig
from .embeddings import TokenEmbedding
from .experts import Expert, ExpertBank
from .ffn import SwiGLUFFN
from .final_deepseek import (
    FinalDeepSeekBlock,
    TinyDeepSeek,
)
from .mla import KVCompression, MLA, QueryCompression
from .mla_cache import CachedMLA
from .moe import DeepSeekMoE
from .mtp import MTPModule
from .deepseek_with_mtp import DeepSeekWithMTP
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding
from .router import TopKRouter

DeepSeekMoEBlock = FinalDeepSeekBlock
FinalTinyDeepSeek = TinyDeepSeek

__all__ = [
    "CachedMLA",
    "DeepSeekConfig",
    "FinalDeepSeekBlock",
    "FinalTinyDeepSeek",
    "DeepSeekMoEBlock",
    "DeepSeekMoE",
    "DeepSeekWithMTP",
    "Expert",
    "ExpertBank",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUFFN",
    "KVCompression",
    "MLA",
    "MTPModule",
    "QueryCompression",
    "TokenEmbedding",
    "TinyDeepSeek",
    "TopKRouter",
]
