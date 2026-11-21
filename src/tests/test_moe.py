import torch

from src.model import DeepSeekConfig, DeepSeekMoE, TopKRouter


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        dim=64,
        n_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        inter_dim=128,
        moe_inter_dim=32,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        route_scale=1.0,
        max_seq_len=16,
    )


def test_router_shapes_and_normalized_weights() -> None:
    torch.manual_seed(0)
    config = small_config()
    router = TopKRouter(config)
    x = torch.randn(2, 5, config.dim)

    indices, weights = router(x)

    assert indices.shape == (2, 5, config.n_activated_experts)
    assert indices.dtype == torch.long
    assert weights.shape == indices.shape
    assert int(indices.min()) >= 0
    assert int(indices.max()) < config.n_routed_experts
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(2, 5),
    )


def test_router_selects_largest_sigmoid_scores() -> None:
    config = small_config()
    config.n_routed_experts = 4
    config.n_activated_experts = 2
    router = TopKRouter(config)
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0, 0] = 1.0
        router.gate.weight[1, 0] = 3.0
        router.gate.weight[2, 0] = 2.0
        router.gate.weight[3, 0] = -1.0

    x = torch.zeros(1, 1, config.dim)
    x[..., 0] = 1.0
    indices, weights = router(x)

    assert indices.tolist() == [[[1, 2]]]
    assert weights[0, 0, 0] > weights[0, 0, 1]
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))


def test_route_scale_scales_selected_weights() -> None:
    config = small_config()
    config.route_scale = 2.5
    router = TopKRouter(config)
    _, weights = router(torch.randn(2, 3, config.dim))

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.full((2, 3), 2.5),
    )


def test_moe_shape_and_parameter_count() -> None:
    config = small_config()
    moe = DeepSeekMoE(config)
    x = torch.randn(2, 5, config.dim)

    output = moe(x)

    assert output.shape == x.shape
    expert_params = 3 * config.dim * config.moe_inter_dim
    expected = (
        (config.n_routed_experts + config.n_shared_experts) * expert_params
        + config.dim * config.n_routed_experts
    )
    assert sum(parameter.numel() for parameter in moe.parameters()) == expected


def test_sparse_dispatch_matches_all_expert_reference() -> None:
    torch.manual_seed(1)
    config = small_config()
    moe = DeepSeekMoE(config).eval()
    x = torch.randn(2, 4, config.dim)
    flat_x = x.reshape(-1, config.dim)

    with torch.no_grad():
        sparse_output = moe(x).reshape(-1, config.dim)
        indices, weights = moe.router(flat_x)
        reference = moe.experts.forward_shared(flat_x)
        for expert_id in range(config.n_routed_experts):
            expert_output = moe.experts.forward_expert(expert_id, flat_x)
            selected = (indices == expert_id).to(weights.dtype)
            expert_weight = (weights * selected).sum(dim=-1)
            reference = reference + expert_output * expert_weight.unsqueeze(-1)

    torch.testing.assert_close(sparse_output, reference)


def test_router_and_selected_experts_receive_gradients() -> None:
    torch.manual_seed(2)
    config = small_config()
    moe = DeepSeekMoE(config)
    x = torch.randn(2, 5, config.dim, requires_grad=True)

    with torch.no_grad():
        indices, _ = moe.router(x.detach())
    selected_ids = set(indices.reshape(-1).tolist())

    moe(x).square().mean().backward()

    router_grad = moe.router.gate.weight.grad
    assert router_grad is not None
    assert torch.isfinite(router_grad).all()
    assert router_grad.norm().item() > 0
    assert moe.experts.shared_expert.gate_proj.weight.grad is not None

    for expert_id, expert in enumerate(moe.experts.routed_experts):
        has_gradient = expert.gate_proj.weight.grad is not None
        assert has_gradient is (expert_id in selected_ids)


def test_expert_usage_counts_equal_token_assignments() -> None:
    config = small_config()
    moe = DeepSeekMoE(config)
    x = torch.randn(3, 7, config.dim)
    indices, _ = moe.router(x)
    counts = torch.bincount(
        indices.reshape(-1),
        minlength=config.n_routed_experts,
    )

    assert int(counts.sum()) == 3 * 7 * config.n_activated_experts
