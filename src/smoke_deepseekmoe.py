"""Smoke test for the complete MLA + MoE language model."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    # Make direct execution use the real ``src`` package, so package-relative
    # imports inside src.training continue to work.
    repository_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repository_root))
    from src.model import DeepSeekConfig, DeepSeekMoE, TinyDeepSeek
    from src.training import MoELoadBalancer
else:
    from .model import DeepSeekConfig, DeepSeekMoE, TinyDeepSeek
    from .training import MoELoadBalancer


def main() -> None:
    torch.manual_seed(19)
    config = DeepSeekConfig(
        vocab_size=512,
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
        max_seq_len=32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDeepSeek(config).to(device)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
        dtype=torch.long,
        device=device,
    )
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    with MoELoadBalancer(model, gamma=0.01) as balancer:
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.float().reshape(-1, config.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        load_ratios = balancer.update()

    layer_types = [
        "MoE" if layer.is_moe else "Dense"
        for layer in model.layers
    ]
    moe_layers = [
        layer for layer in model.layers
        if isinstance(layer.ffn, DeepSeekMoE)
    ]

    print("Device:", device)
    print("Input:", inputs.shape)
    print("Logits:", logits.shape)
    print("Layers:", layer_types)
    print("Loss:", f"{loss.item():.4f}")
    print("MoE max-load ratios:", [f"{ratio:.3f}" for ratio in load_ratios])
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    assert logits.shape == (2, 7, config.vocab_size)
    assert len(moe_layers) == config.n_layers - config.n_dense_layers
    assert len(load_ratios) == len(moe_layers)
    assert all(
        layer.ffn.router.gate.weight.grad is not None
        for layer in moe_layers
    )
    assert all(
        torch.isfinite(layer.ffn.router.expert_bias).all()
        for layer in moe_layers
    )
    print("DeepSeekMoE smoke test passed")


if __name__ == "__main__":
    main()
