"""Overfit a fixed batch with the FP32 DeepSeek+MTP optimization step."""

import torch

from src.model import DeepSeekConfig, DeepSeekWithMTP
from src.training import MoELoadBalancer, create_optimizer, train_step


def main() -> None:
    torch.manual_seed(25)
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
    optimizer = create_optimizer(model)
    tokens = torch.randint(0, config.vocab_size, (2, 17), device=device)

    with MoELoadBalancer(model, gamma=0.01) as balancer:
        for step in range(1, 6):
            metrics = train_step(
                model,
                optimizer,
                balancer,
                tokens,
                step=step,
                total_steps=5,
                warmup_steps=2,
            )
            print(
                f"step {step} | loss {metrics['loss']:.4f} | "
                f"main {metrics['main_loss']:.4f} | "
                f"mtp {metrics['mtp_loss']:.4f} | "
                f"lr {metrics['lr']:.7f} | "
                f"grad {metrics['grad_norm']:.3f}"
            )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert len(metrics["load_ratios"]) == config.n_layers - config.n_dense_layers + 1
    print("optimizer smoke test passed")


if __name__ == "__main__":
    main()
