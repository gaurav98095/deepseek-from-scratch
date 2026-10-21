import pytest
import torch

from src.model.attention import (
    CausalMultiHeadLatentAttention,
    LearnedPositionalEmbedding,
)
from src.model.config import ModelConfig
from src.model.deepseek import DeepSeek


def test_learned_positional_embedding_shape_and_limit() -> None:
    # Position vectors broadcast across the batch and enforce the configured limit.
    positions = LearnedPositionalEmbedding(max_sequence_length=16, embed_dim=8)

    assert positions(5).shape == (1, 5, 8)
    with pytest.raises(ValueError, match="exceeds"):
        positions(17)


def test_latent_attention_cache_matches_full_attention_and_is_compressed() -> None:
    torch.manual_seed(3)
    attention = CausalMultiHeadLatentAttention(
        embed_dim=12,
        num_heads=3,
        head_dim=4,
        q_lora_rank=6,
        kv_lora_rank=2,
    ).eval()
    hidden_states = torch.randn(2, 5, 12)

    full_output = attention(hidden_states)
    first_output, cache = attention(hidden_states[:, :3], use_cache=True)
    fourth_output, cache = attention(
        hidden_states[:, 3:4],
        past_key_value=cache,
        use_cache=True,
    )
    fifth_output, cache = attention(
        hidden_states[:, 4:5],
        past_key_value=cache,
        use_cache=True,
    )
    cached_output = torch.cat((first_output, fourth_output, fifth_output), dim=1)

    assert torch.allclose(cached_output, full_output, atol=1e-6)
    assert cache.shape == (2, 5, 2)
    assert cache.numel() < 2 * 3 * 5 * 4 * 2


def test_latent_attention_is_causal() -> None:
    torch.manual_seed(4)
    attention = CausalMultiHeadLatentAttention(
        embed_dim=8,
        num_heads=2,
        head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=2,
    ).eval()
    original = torch.randn(1, 4, 8)
    changed_future = original.clone()
    changed_future[:, 2:] += 1000

    first_two_original = attention(original)[:, :2]
    first_two_changed = attention(changed_future)[:, :2]

    assert torch.allclose(first_two_original, first_two_changed, atol=1e-5)


def test_latent_attention_head_width_can_differ_from_hidden_size() -> None:
    attention = CausalMultiHeadLatentAttention(
        embed_dim=10,
        num_heads=2,
        head_dim=3,
        q_lora_rank=4,
        kv_lora_rank=2,
    )

    output, weights = attention(torch.randn(1, 4, 10), return_attention=True)

    assert output.shape == (1, 4, 10)
    assert weights.shape == (1, 2, 4, 4)


def test_cached_model_matches_full_sequence_logits() -> None:
    # This end-to-end check also proves cached tokens use continued positions.
    torch.manual_seed(2)
    config = ModelConfig(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=3,
        head_dim=4,
        q_lora_rank=6,
        kv_lora_rank=2,
        intermediate_size=24,
        torch_dtype="float32",
    )
    model = DeepSeek(config).eval()
    input_ids = torch.tensor([[1, 5, 9, 4, 2]])

    full_logits = model(input_ids).logits
    past_key_values = None
    cached_logits = []
    for token_index in range(input_ids.shape[1]):
        output = model(
            input_ids[:, token_index : token_index + 1],
            past_key_values=past_key_values,
            use_cache=True,
        )
        assert output.past_key_values is not None
        past_key_values = output.past_key_values
        cached_logits.append(output.logits)

    assert torch.allclose(torch.cat(cached_logits, dim=1), full_logits, atol=1e-6)
    assert past_key_values is not None
    assert len(past_key_values) == config.num_hidden_layers
    assert past_key_values[0].shape == (1, 5, 2)
