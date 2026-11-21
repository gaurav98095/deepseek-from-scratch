import torch

from src.model import DeepSeekConfig, DeepSeekMoE, SwiGLUFFN, TinyDeepSeek
from src.training import MoELoadBalancer


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=128,
        dim=64,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        inter_dim=128,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        max_seq_len=16,
    )


def test_layer_selection_uses_dense_then_moe() -> None:
    model = TinyDeepSeek(small_config())

    assert isinstance(model.layers[0].ffn, SwiGLUFFN)
    assert all(isinstance(layer.ffn, DeepSeekMoE) for layer in model.layers[1:])
    assert [layer.is_moe for layer in model.layers] == [False, True, True]


def test_full_model_output_shape() -> None:
    config = small_config()
    model = TinyDeepSeek(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))

    logits = model(tokens)

    assert logits.shape == (2, 8, config.vocab_size)


def test_full_model_backward_reaches_mla_router_and_experts() -> None:
    torch.manual_seed(0)
    config = small_config()
    model = TinyDeepSeek(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))

    loss = model(tokens).square().mean()
    loss.backward()

    assert model.layers[0].attention.q_branch.q_down.weight.grad is not None
    moe_layer = model.layers[1].ffn
    assert moe_layer.router.gate.weight.grad is not None
    assert moe_layer.experts.shared_expert.gate_proj.weight.grad is not None


def test_full_model_is_causal() -> None:
    torch.manual_seed(1)
    config = small_config()
    model = TinyDeepSeek(config).eval()
    original = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    changed = torch.tensor([[10, 20, 30, 40, 50, 91, 92, 93]])

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    torch.testing.assert_close(
        original_logits[:, :5],
        changed_logits[:, :5],
    )


def test_balancer_collects_each_moe_layer_and_updates_biases() -> None:
    torch.manual_seed(2)
    model = TinyDeepSeek(small_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 6))

    with MoELoadBalancer(model, gamma=0.05) as balancer:
        model(tokens)
        assert len(balancer.records) == 2
        ratios = balancer.update()

    assert len(ratios) == 2
    assert all(ratio >= 0 for ratio in ratios)
    for layer in model.layers[1:]:
        assert layer.ffn.router.expert_bias.numel() == 4


def test_balancer_accumulates_multiple_forward_calls() -> None:
    model = TinyDeepSeek(small_config())
    first = torch.randint(0, model.config.vocab_size, (1, 4))
    second = torch.randint(0, model.config.vocab_size, (1, 4))

    with MoELoadBalancer(model, gamma=0.01) as balancer:
        model(first)
        model(second)
        assert len(balancer.records) == 4
        ratios = balancer.update()

    assert len(ratios) == 2
    assert balancer.records == []
