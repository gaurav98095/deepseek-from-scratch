import pytest
import torch

from src.model import DeepSeekConfig, DeepSeekWithMTP
from src.training import (
    MoELoadBalancer,
    create_optimizer,
    get_learning_rate,
    set_learning_rate,
    train_step,
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


def test_learning_rate_warmup_and_cosine_boundaries() -> None:
    first = get_learning_rate(1, 100, 10, 3e-4, 3e-5)
    peak = get_learning_rate(10, 100, 10, 3e-4, 3e-5)
    middle = get_learning_rate(55, 100, 10, 3e-4, 3e-5)
    final = get_learning_rate(100, 100, 10, 3e-4, 3e-5)

    assert first < peak
    assert middle < peak
    assert middle > final
    assert abs(peak - 3e-4) < 1e-10
    assert abs(final - 3e-5) < 1e-10


def test_learning_rate_validation() -> None:
    with pytest.raises(ValueError, match="smaller"):
        get_learning_rate(1, 10, 10, 3e-4, 3e-5)
    with pytest.raises(ValueError, match="bounds"):
        get_learning_rate(1, 10, 2, 3e-5, 3e-4)


def test_optimizer_groups_decay_only_matrix_parameters() -> None:
    model = DeepSeekWithMTP(small_config())
    optimizer = create_optimizer(model)
    decay_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}

    assert decay_ids.isdisjoint(no_decay_ids)
    assert all(parameter.ndim >= 2 for parameter in optimizer.param_groups[0]["params"])
    assert all(parameter.ndim < 2 for parameter in optimizer.param_groups[1]["params"])
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_adamw_updates_a_parameter() -> None:
    torch.manual_seed(25)
    model = torch.nn.Linear(4, 2)
    optimizer = create_optimizer(model, lr=1e-2)
    before = model.weight.detach().clone()
    loss = model(torch.ones(3, 4)).square().mean()
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, model.weight.detach())


def test_train_step_updates_model_and_router_biases() -> None:
    torch.manual_seed(26)
    config = small_config()
    model = DeepSeekWithMTP(config)
    optimizer = create_optimizer(model)
    tokens = torch.randint(0, config.vocab_size, (2, 9))
    before = model.main_model.embed_tokens.embedding.weight.detach().clone()

    with MoELoadBalancer(model, gamma=0.01) as balancer:
        metrics = train_step(
            model,
            optimizer,
            balancer,
            tokens,
            step=1,
            total_steps=5,
            warmup_steps=2,
        )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["lr"] > 0
    assert metrics["grad_norm"] >= 0
    assert len(metrics["load_ratios"]) == 2
    assert not torch.equal(before, model.main_model.embed_tokens.embedding.weight)
