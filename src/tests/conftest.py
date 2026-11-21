import pytest

from src.model import DeepSeekConfig


@pytest.fixture
def tiny_config() -> DeepSeekConfig:
    """Small configuration that exercises every final-model mechanism."""
    return DeepSeekConfig(
        vocab_size=128,
        max_seq_len=32,
        dim=64,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        inter_dim=128,
        moe_inter_dim=32,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        route_scale=1.0,
    )
