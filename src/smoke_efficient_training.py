"""Smoke test for BF16-aware accumulation and attention checkpointing."""

import torch

from src.model import DeepSeekConfig, DeepSeekWithMTP
from src.training import (
    MoELoadBalancer,
    create_optimizer,
    efficient_train_step,
    profile_train_step,
)


def main() -> None:
    torch.manual_seed(26)
    config = DeepSeekConfig(
        vocab_size=256,
        max_seq_len=32,
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
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSeekWithMTP(config, mtp_weight=0.3).to(device)
    model.set_activation_checkpointing(True)
    optimizer = create_optimizer(model)
    microbatches = [
        torch.randint(0, config.vocab_size, (2, 17), device=device)
        for _ in range(2)
    ]

    with MoELoadBalancer(model, gamma=0.01) as balancer:
        metrics = profile_train_step(
            lambda: efficient_train_step(
                model,
                optimizer,
                balancer,
                microbatches,
                step=1,
                total_steps=5,
                warmup_steps=1,
            ),
            tokens_per_step=2 * 2 * 16,
        )

    print("Device:", device)
    print("Metrics:", metrics)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert len(metrics["load_ratios"]) == 3
    print("efficient training smoke test passed")


if __name__ == "__main__":
    main()
