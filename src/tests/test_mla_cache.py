import pytest
import torch

from src.model import CachedMLA, DeepSeekConfig, DeepSeekMLA


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=128,
        dim=64,
        n_layers=2,
        n_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        inter_dim=128,
        max_seq_len=32,
    )


def test_prefill_populates_compressed_cache() -> None:
    torch.manual_seed(0)
    config = small_config()
    model = CachedMLA(config).eval()
    hidden_states = torch.randn(2, 6, config.dim)

    with torch.no_grad():
        output = model.prefill(hidden_states)

    assert output.shape == (2, 6, config.dim)
    assert model.kv_cache is not None
    assert model.pe_cache is not None
    assert model.kv_cache.shape == (2, 6, config.kv_lora_rank)
    assert model.pe_cache.shape == (2, 6, config.qk_rope_head_dim)
    assert model.cache_length == 6


def test_absorbed_last_token_matches_naive_forward() -> None:
    torch.manual_seed(1)
    config = small_config()
    model = CachedMLA(config).eval()
    hidden_states = torch.randn(2, 6, config.dim)

    with torch.no_grad():
        naive_output = model(hidden_states)
        model.reset_cache()
        model.prefill(hidden_states[:, :-1])
        absorbed_output = model.decode_step(hidden_states[:, -1:])

    torch.testing.assert_close(
        absorbed_output,
        naive_output[:, -1:],
        rtol=2e-4,
        atol=2e-4,
    )


def test_incremental_decoding_matches_full_forward() -> None:
    torch.manual_seed(2)
    config = small_config()
    model = CachedMLA(config).eval()
    hidden_states = torch.randn(1, 10, config.dim)

    with torch.no_grad():
        expected = model(hidden_states)
        model.reset_cache()
        outputs = [
            model.decode_step(hidden_states[:, position : position + 1])
            for position in range(hidden_states.shape[1])
        ]
    actual = torch.cat(outputs, dim=1)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert model.cache_length == hidden_states.shape[1]


def test_prefill_requires_a_fresh_cache() -> None:
    config = small_config()
    model = CachedMLA(config)
    hidden_states = torch.randn(1, 3, config.dim)

    model.prefill(hidden_states)
    with pytest.raises(ValueError, match="reset_cache"):
        model.prefill(hidden_states)


def test_cache_rejects_batch_size_changes() -> None:
    config = small_config()
    model = CachedMLA(config)
    model.decode_step(torch.randn(1, 1, config.dim))

    with pytest.raises(ValueError, match="Batch size changed"):
        model.decode_step(torch.randn(2, 1, config.dim))


def test_full_model_prefill_and_decode_match_forward() -> None:
    torch.manual_seed(3)
    config = small_config()
    model = DeepSeekMLA(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 9))

    with torch.no_grad():
        expected = model(tokens)
        model.reset_cache()
        model.prefill(tokens[:, :-1])
        actual = model.decode_step(tokens[:, -1:])

    torch.testing.assert_close(
        actual,
        expected[:, -1:],
        rtol=2e-4,
        atol=2e-4,
    )


def test_full_model_incremental_decoding_matches_forward() -> None:
    torch.manual_seed(4)
    config = small_config()
    model = DeepSeekMLA(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 7))

    with torch.no_grad():
        expected = model(tokens)
        model.reset_cache()
        outputs = [
            model.decode_step(tokens[:, position : position + 1])
            for position in range(tokens.shape[1])
        ]
    actual = torch.cat(outputs, dim=1)

    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
