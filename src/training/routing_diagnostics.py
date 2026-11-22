"""Expert-utilization statistics for a final DeepSeek model."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from ..model.router import TopKRouter


@torch.no_grad()
def collect_routing_statistics(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
) -> dict[str, dict[str, float | int | list[int]]]:
    """Aggregate expert assignments over a fixed collection of input batches."""
    counts_by_name: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(
            router: TopKRouter,
            _inputs: tuple[torch.Tensor, ...],
            outputs: tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            indices, _weights = outputs
            counts = torch.bincount(
                indices.detach().reshape(-1).cpu(),
                minlength=router.n_experts,
            )
            counts_by_name[name] = counts_by_name.get(name, torch.zeros_like(counts)) + counts

        return hook

    for name, module in model.named_modules():
        if isinstance(module, TopKRouter):
            handles.append(module.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device
    was_training = model.training
    try:
        model.eval()
        for batch in batches:
            if batch.ndim != 2:
                raise ValueError("routing batches must have shape [batch, sequence]")
            model(batch.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    results: dict[str, dict[str, float | int | list[int]]] = {}
    for name, counts in counts_by_name.items():
        counts = counts.float()
        mean = counts.mean()
        if mean.item() == 0:
            max_load_ratio = 0.0
            coefficient_of_variation = 0.0
        else:
            max_load_ratio = float((counts.max() / mean).item())
            coefficient_of_variation = float(
                (counts.std(unbiased=False) / mean).item()
            )
        results[name] = {
            "counts": counts.to(torch.int64).tolist(),
            "assignments": int(counts.sum().item()),
            "max_load_ratio": max_load_ratio,
            "cv": coefficient_of_variation,
        }
    return results
