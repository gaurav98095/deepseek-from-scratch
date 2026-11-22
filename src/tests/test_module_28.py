import numpy as np
import torch

from src.model import DeepSeekConfig, TinyDeepSeek
from src.training.fixed_evaluate import FixedTokenWindows, evaluate_fixed_windows
from src.training.routing_diagnostics import collect_routing_statistics


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=64,
        max_seq_len=8,
        dim=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        inter_dim=64,
        moe_inter_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )


def make_tokens(tmp_path):
    path = tmp_path / "val.bin"
    np.arange(33, dtype=np.uint16).tofile(path)
    return path


def test_fixed_windows_are_non_overlapping(tmp_path) -> None:
    windows = FixedTokenWindows(make_tokens(tmp_path), seq_len=8, dtype="<u2")

    assert len(windows) == 4
    torch.testing.assert_close(
        windows.window(0),
        torch.arange(9, dtype=torch.int64),
    )
    torch.testing.assert_close(
        windows.window(1),
        torch.arange(8, 17, dtype=torch.int64),
    )


def test_fixed_evaluation_is_token_weighted_and_restores_mode(tmp_path) -> None:
    windows = FixedTokenWindows(make_tokens(tmp_path), seq_len=8, dtype="<u2")
    model = TinyDeepSeek(small_config())
    model.train()

    metrics = evaluate_fixed_windows(
        model,
        windows,
        batch_size=2,
        max_windows=3,
    )

    assert model.training
    assert metrics["windows"] == 3
    assert metrics["target_tokens"] == 3 * 8
    assert metrics["perplexity"] > 1


def test_routing_statistics_report_each_moe_router(tmp_path) -> None:
    windows = FixedTokenWindows(make_tokens(tmp_path), seq_len=8, dtype="<u2")
    model = TinyDeepSeek(small_config())
    batches = (batch[:, :-1] for batch in windows.batches(2, max_windows=2))

    report = collect_routing_statistics(model, batches)

    assert len(report) == 1
    stats = next(iter(report.values()))
    assert stats["assignments"] == 2 * 8 * 2
    assert len(stats["counts"]) == 4
    assert stats["max_load_ratio"] >= 1
