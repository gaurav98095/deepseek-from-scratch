import torch

from src.experiments.analyze_routing import (
    inspect_model_routing,
    moe_parameter_report,
    routing_metrics,
)
from src.experiments.benchmark_ffn import benchmark
from src.model import DeepSeekConfig, DeepSeekMoE, TinyDeepSeek


def small_config(**overrides) -> DeepSeekConfig:
    values = dict(
        vocab_size=128,
        dim=32,
        n_layers=2,
        n_dense_layers=0,
        n_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        inter_dim=96,
        moe_inter_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        max_seq_len=8,
    )
    values.update(overrides)
    return DeepSeekConfig(**values)


def test_routing_metrics_reports_balanced_assignments() -> None:
    indices = torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]])
    metrics = routing_metrics(indices, n_experts=4)

    assert metrics["counts"] == [2.0, 2.0, 2.0, 2.0]
    assert metrics["cv"] == 0.0
    assert metrics["max_load_ratio"] == 1.0
    assert abs(metrics["normalized_entropy"] - 1.0) < 1e-6


def test_inspect_model_routing_restores_training_state() -> None:
    model = TinyDeepSeek(small_config())
    model.train()
    tokens = torch.randint(0, model.config.vocab_size, (2, 6))

    records = inspect_model_routing(model, tokens)

    assert model.training
    assert len(records) == 2
    assert all("cv" in values for values in records.values())


def test_zero_shared_experts_is_a_valid_ablation() -> None:
    config = small_config(n_shared_experts=0)
    layer = DeepSeekMoE(config)
    x = torch.randn(2, 4, config.dim)

    output = layer(x)
    assert layer.experts.shared_expert is None
    torch.testing.assert_close(layer.experts.forward_shared(x), torch.zeros_like(x))
    assert output.shape == x.shape


def test_parameter_report_matches_moe_layer() -> None:
    config = small_config()
    report = moe_parameter_report(config)
    layer = DeepSeekMoE(config)

    assert report["total_moe_parameters"] == sum(
        parameter.numel() for parameter in layer.parameters()
    )
    assert report["active_moe_parameters"] > 0


def test_cpu_benchmark_returns_positive_timings() -> None:
    result = benchmark(torch.nn.Identity(), torch.randn(2, 3, 4), warmup=1, repeats=2)

    assert result["median_ms"] > 0
    assert result["mean_ms"] > 0
