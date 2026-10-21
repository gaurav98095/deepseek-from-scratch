import pytest
import torch

from src.model.attention import CausalSelfAttention, LearnedPositionalEmbedding, MultiHeadAttention, SelfAttention
from src.model.config import ModelConfig
from src.model.deepseek import DeepSeek


def test_multi_head_attention_preserves_shape() -> None:
    # A 12-wide model split across 3 heads should return the original width.
    attention = MultiHeadAttention(embed_dim=12, num_heads=3)

    output, weights = attention(torch.randn(2, 5, 12), return_attention=True)

    assert output.shape == (2, 5, 12)
    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)


def test_self_attention_is_one_head() -> None:
    # SelfAttention is the single-head specialization of the general module.
    attention = SelfAttention(embed_dim=8)

    _, weights = attention(torch.randn(2, 4, 8), return_attention=True)

    assert weights.shape == (2, 1, 4, 4)


def test_causal_attention_cannot_use_future_tokens() -> None:
    # Changing future tokens must not change outputs for earlier positions.
    torch.manual_seed(0)
    attention = CausalSelfAttention(embed_dim=8, num_heads=2).eval()
    original = torch.randn(1, 4, 8)
    changed_future = original.clone()
    changed_future[:, 2:] += 1000

    first_two_original = attention(original)[:, :2]
    first_two_changed = attention(changed_future)[:, :2]

    assert torch.allclose(first_two_original, first_two_changed, atol=1e-5)


def test_learned_positional_embedding_shape_and_limit() -> None:
    # Position vectors broadcast across the batch and enforce the configured limit.
    positions = LearnedPositionalEmbedding(max_sequence_length=16, embed_dim=8)

    assert positions(5).shape == (1, 5, 8)
    with pytest.raises(ValueError, match="exceeds"):
        positions(17)


def test_attention_rejects_invalid_head_configuration() -> None:
    # The hidden width must divide evenly into the requested number of heads.
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(embed_dim=10, num_heads=3)


def test_cached_attention_matches_full_causal_attention() -> None:
    # Incremental attention must reproduce a single full-sequence forward pass.
    torch.manual_seed(1)
    attention = CausalSelfAttention(embed_dim=12, num_heads=3).eval()
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
    assert cache[0].shape == (2, 3, 5, 4)
    assert cache[1].shape == (2, 3, 5, 4)


def test_cached_model_matches_full_sequence_logits() -> None:
    # This end-to-end check also proves cached tokens use continued positions.
    torch.manual_seed(2)
    config = ModelConfig(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=3,
        head_dim=4,
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
    assert past_key_values[0][0].shape == (1, 3, 5, 4)
