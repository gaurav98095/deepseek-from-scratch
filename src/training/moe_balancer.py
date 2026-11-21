"""Collect routing assignments and update MoE balancing biases."""

from __future__ import annotations

import torch

from ..model.router import TopKRouter


class MoELoadBalancer:
    """Update every router once from assignments observed during a step."""

    def __init__(self, model: torch.nn.Module, gamma: float = 0.01) -> None:
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.gamma = gamma
        self.records: list[tuple[TopKRouter, torch.Tensor]] = []
        self.handles = [
            module.register_forward_hook(self._capture)
            for module in model.modules()
            if isinstance(module, TopKRouter)
        ]

    def _capture(
        self,
        router: TopKRouter,
        _inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        indices, _weights = outputs
        self.records.append((router, indices.detach()))

    def clear(self) -> None:
        """Discard assignments collected for the current training step."""
        self.records.clear()

    @staticmethod
    def _max_load_ratio(counts: torch.Tensor) -> float:
        target = counts.sum() / counts.numel()
        if target.item() == 0:
            return 0.0
        return float((counts.max() / target).item())

    @torch.no_grad()
    def update(self) -> list[float]:
        """Update each observed router and return its max-load ratio."""
        grouped: dict[int, tuple[TopKRouter, list[torch.Tensor]]] = {}
        for router, indices in self.records:
            key = id(router)
            if key not in grouped:
                grouped[key] = (router, [])
            grouped[key][1].append(indices.reshape(-1, indices.shape[-1]))

        ratios: list[float] = []
        for router, batches in grouped.values():
            indices = torch.cat(batches, dim=0)
            counts = router.update_bias(indices, gamma=self.gamma)
            ratios.append(self._max_load_ratio(counts))
        self.clear()
        return ratios

    def close(self) -> None:
        """Remove all forward hooks owned by this balancer."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    def __enter__(self) -> "MoELoadBalancer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
