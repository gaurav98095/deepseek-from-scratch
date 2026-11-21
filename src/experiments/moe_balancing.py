"""Small simulation of auxiliary-loss-free expert-bias balancing."""

import torch

from src.model import DeepSeekConfig, TopKRouter


def load_metrics(counts: torch.Tensor) -> tuple[float, float]:
    """Return coefficient of variation and max-load ratio."""
    values = counts.float()
    mean = values.mean()
    if mean.item() == 0:
        return 0.0, 0.0
    coefficient_of_variation = (values.std(unbiased=False) / mean).item()
    max_load_ratio = (values.max() / mean).item()
    return coefficient_of_variation, max_load_ratio


def main() -> None:
    torch.manual_seed(42)
    config = DeepSeekConfig(
        dim=64,
        n_routed_experts=8,
        n_activated_experts=2,
    )
    router = TopKRouter(config).eval()
    tokens = torch.randn(128, config.dim) * 0.1
    tokens[:, 0] = 1.0

    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0, 0] = 2.0
        router.gate.weight[1, 0] = 1.5

    print("step | counts | CV | max-load ratio")
    for step in range(101):
        with torch.no_grad():
            indices, _ = router(tokens)
        counts = torch.bincount(
            indices.reshape(-1),
            minlength=config.n_routed_experts,
        )
        coefficient_of_variation, max_load_ratio = load_metrics(counts)
        if step % 20 == 0 or step == 100:
            print(
                f"{step:4d} | {counts.tolist()} | "
                f"{coefficient_of_variation:.3f} | {max_load_ratio:.3f}"
            )
        router.update_bias(indices, gamma=0.01)


if __name__ == "__main__":
    main()
