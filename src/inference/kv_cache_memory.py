"""Logical KV-cache memory estimates for attention variants."""

from typing import Literal


AttentionType = Literal["mha", "gqa", "mqa", "mla"]


def cached_elements_per_token_layer(
    attention_type: AttentionType,
    *,
    query_heads: int,
    head_dim: int,
    gqa_kv_heads: int,
    kv_latent_dim: int,
    rope_dim: int,
) -> int:
    """Return cached elements for one token in one Transformer layer."""
    if query_heads <= 0 or head_dim <= 0:
        raise ValueError("query_heads and head_dim must be positive")
    if gqa_kv_heads <= 0 or gqa_kv_heads > query_heads:
        raise ValueError("gqa_kv_heads must be between 1 and query_heads")
    if query_heads % gqa_kv_heads != 0:
        raise ValueError("query_heads must divide evenly into KV heads")
    if kv_latent_dim <= 0 or rope_dim <= 0:
        raise ValueError("latent and positional dimensions must be positive")

    if attention_type == "mha":
        kv_heads = query_heads
        return 2 * kv_heads * head_dim
    if attention_type == "gqa":
        return 2 * gqa_kv_heads * head_dim
    if attention_type == "mqa":
        return 2 * head_dim
    if attention_type == "mla":
        return kv_latent_dim + rope_dim
    raise ValueError(f"unknown attention type: {attention_type}")


def estimate_cache_mib(
    attention_type: AttentionType,
    *,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    query_heads: int,
    head_dim: int,
    gqa_kv_heads: int,
    kv_latent_dim: int,
    rope_dim: int,
    bytes_per_element: int = 2,
) -> float:
    """Estimate logical cache storage in MiB, excluding allocator overhead."""
    if batch_size <= 0 or seq_len <= 0 or n_layers <= 0:
        raise ValueError("batch_size, seq_len, and n_layers must be positive")
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")

    elements = cached_elements_per_token_layer(
        attention_type,
        query_heads=query_heads,
        head_dim=head_dim,
        gqa_kv_heads=gqa_kv_heads,
        kv_latent_dim=kv_latent_dim,
        rope_dim=rope_dim,
    )
    total_bytes = (
        batch_size
        * seq_len
        * n_layers
        * elements
        * bytes_per_element
    )
    return total_bytes / (1024 * 1024)
