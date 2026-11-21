from src.experiments.inspect_architecture import (
    count_parameters,
    inspect_architecture,
)
from src.model import DeepSeekConfig, TinyDeepSeek


def test_default_architecture_has_dense_prefix_and_moe_suffix() -> None:
    model = TinyDeepSeek(DeepSeekConfig())

    records = inspect_architecture(model)

    assert len(records) == 6
    assert [record["ffn"] for record in records] == [
        "Dense SwiGLU",
        "MoE",
        "MoE",
        "MoE",
        "MoE",
        "MoE",
    ]
    assert all(record["attention"] == "MLA" for record in records)


def test_default_parameter_accounting_matches_the_architecture() -> None:
    model = TinyDeepSeek(DeepSeekConfig())
    records = inspect_architecture(model)

    assert records[0]["parameters"] == 791_232
    assert all(record["parameters"] == 1_972_928 for record in records[1:])
    assert count_parameters(model) == 14_850_432


def test_custom_dense_prefix_is_reflected_in_the_report() -> None:
    config = DeepSeekConfig(
        vocab_size=128,
        n_layers=4,
        n_dense_layers=2,
        dim=64,
        n_heads=4,
        inter_dim=128,
        moe_inter_dim=32,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )
    records = inspect_architecture(TinyDeepSeek(config))

    assert [record["ffn"] for record in records] == [
        "Dense SwiGLU",
        "Dense SwiGLU",
        "MoE",
        "MoE",
    ]
