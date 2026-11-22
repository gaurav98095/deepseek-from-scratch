import copy

import pytest
import torch

from src.model import DeepSeekConfig, DeepSeekWithMTP, FinalTinyDeepSeek
from src.training import (
    MoELoadBalancer,
    create_optimizer,
    efficient_train_step,
    get_learning_rate,
    profile_train_step,
)


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=128,
        max_seq_len=32,
        dim=32,
        n_layers=2,
        n_dense_layers=1,
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
    )


def test_accumulation_updates_once_and_aggregates_router_records() -> None:
    torch.manual_seed(26)
    config = small_config()
    model = DeepSeekWithMTP(config)
    optimizer = create_optimizer(model)
    batches = [
        torch.randint(0, config.vocab_size, (2, 9)),
        torch.randint(0, config.vocab_size, (2, 9)),
    ]

    with MoELoadBalancer(model, gamma=0.01) as balancer:
        metrics = efficient_train_step(
            model,
            optimizer,
            balancer,
            batches,
            step=1,
            total_steps=5,
            warmup_steps=1,
            use_bf16=False,
        )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert len(metrics["load_ratios"]) == 2
    assert balancer.records == []


def test_empty_microbatches_are_rejected() -> None:
    model = DeepSeekWithMTP(small_config())
    optimizer = create_optimizer(model)
    with MoELoadBalancer(model) as balancer:
        with pytest.raises(ValueError, match="cannot be empty"):
            efficient_train_step(
                model,
                optimizer,
                balancer,
                [],
                step=1,
                total_steps=5,
                warmup_steps=1,
            )


def test_checkpointing_preserves_attention_outputs_and_gradients() -> None:
    torch.manual_seed(27)
    config = small_config()
    ordinary = FinalTinyDeepSeek(config)
    checkpointed = copy.deepcopy(ordinary)
    checkpointed.set_activation_checkpointing(True)
    tokens = torch.randint(0, config.vocab_size, (2, 8))

    ordinary_loss = ordinary(tokens).square().mean()
    checkpointed_loss = checkpointed(tokens).square().mean()
    ordinary_loss.backward()
    checkpointed_loss.backward()

    torch.testing.assert_close(ordinary_loss, checkpointed_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        ordinary.embed_tokens.embedding.weight.grad,
        checkpointed.embed_tokens.embedding.weight.grad,
        atol=1e-5,
        rtol=1e-5,
    )


def test_profile_reports_step_time_and_throughput() -> None:
    result = profile_train_step(lambda: {"loss": 1.0}, tokens_per_step=16)

    assert result["step_seconds"] > 0
    assert result["tokens_per_second"] > 0
    assert result["loss"] == 1.0


def test_bf16_request_requires_supported_cuda() -> None:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is available; this validation targets unsupported devices")
    config = small_config()
    model = DeepSeekWithMTP(config)
    optimizer = create_optimizer(model)
    tokens = [torch.randint(0, config.vocab_size, (1, 9))]
    with MoELoadBalancer(model) as balancer:
        with pytest.raises(ValueError, match="BF16"):
            efficient_train_step(
                model,
                optimizer,
                balancer,
                tokens,
                step=1,
                total_steps=5,
                warmup_steps=1,
                use_bf16=True,
            )
