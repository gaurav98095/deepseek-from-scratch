import pytest

from src.inference.kv_cache_memory import (
    cached_elements_per_token_layer,
    estimate_cache_mib,
)


def settings() -> dict[str, int]:
    return {
        "query_heads": 8,
        "head_dim": 32,
        "gqa_kv_heads": 2,
        "kv_latent_dim": 64,
        "rope_dim": 16,
    }


def test_cached_elements_per_token_layer() -> None:
    values = settings()

    assert cached_elements_per_token_layer("mha", **values) == 512
    assert cached_elements_per_token_layer("gqa", **values) == 128
    assert cached_elements_per_token_layer("mqa", **values) == 64
    assert cached_elements_per_token_layer("mla", **values) == 80


def test_cache_scales_with_layers_and_sequence_length() -> None:
    values = settings()
    one = estimate_cache_mib(
        "mha",
        batch_size=1,
        seq_len=1024,
        n_layers=1,
        bytes_per_element=2,
        **values,
    )
    four = estimate_cache_mib(
        "mha",
        batch_size=1,
        seq_len=2048,
        n_layers=2,
        bytes_per_element=2,
        **values,
    )

    assert four == pytest.approx(one * 4)


def test_invalid_attention_configuration() -> None:
    values = settings()
    with pytest.raises(ValueError):
        cached_elements_per_token_layer("gqa", **{**values, "gqa_kv_heads": 3})
    with pytest.raises(ValueError):
        estimate_cache_mib(
            "unknown",  # type: ignore[arg-type]
            batch_size=1,
            seq_len=1,
            n_layers=1,
            **values,
        )
