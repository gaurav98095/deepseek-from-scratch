"""Routing diagnostics and parameter reports for the educational MoE model."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from src.model.config import DeepSeekConfig
from src.model.router import TopKRouter


def routing_metrics(
    indices: torch.Tensor,
    n_experts: int,
) -> dict[str, object]:
    """Summarize expert assignments from any ``[..., top_k]`` index tensor."""
    if indices.ndim < 1:
        raise ValueError("indices must have at least one dimension")
    if indices.dtype != torch.long:
        raise ValueError("indices must have dtype torch.long")
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    if indices.numel() == 0:
        raise ValueError("No routing assignments")
    if int(indices.min()) < 0 or int(indices.max()) >= n_experts:
        raise ValueError("indices contain an invalid expert ID")

    counts = torch.bincount(indices.reshape(-1), minlength=n_experts).float()
    mean = counts.mean()
    if mean.item() == 0:
        raise ValueError("No routing assignments")
    fractions = counts / counts.sum()
    nonzero = fractions[fractions > 0]
    entropy = -(nonzero * nonzero.log()).sum()
    return {
        "counts": counts.cpu().tolist(),
        "assignments": int(indices.numel()),
        "cv": float((counts.std(unbiased=False) / mean).item()),
        "max_load_ratio": float((counts.max() / mean).item()),
        "entropy": float(entropy.item()),
        "normalized_entropy": float(
            (entropy / math.log(n_experts)).item()
            if n_experts > 1
            else 1.0
        ),
    }


def inspect_model_routing(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, dict[str, object]]:
    """Record routing decisions during one evaluation forward pass.

    Hooks observe router outputs without changing weights, router biases, or
    gradients. The model's original training/evaluation state is restored.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")

    records: dict[str, dict[str, object]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(module: TopKRouter, _inputs, outputs) -> None:
            indices, _weights = outputs
            records[name] = routing_metrics(
                indices.detach(),
                module.n_experts,
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, TopKRouter):
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return records


def parameter_count(module: nn.Module) -> int:
    """Return the number of trainable and non-trainable model parameters."""
    return sum(parameter.numel() for parameter in module.parameters())


def moe_parameter_report(config: DeepSeekConfig) -> Mapping[str, int]:
    """Calculate total and active parameters for one MoE FFN layer.

    The active path includes the selected routed experts, all shared experts,
    and the router. Bias state is intentionally not counted as a parameter.
    """
    expert_parameters = 3 * config.dim * config.moe_inter_dim
    router_parameters = config.dim * config.n_routed_experts
    routed_total = config.n_routed_experts * expert_parameters
    shared_total = config.n_shared_experts * expert_parameters
    active = (
        config.n_activated_experts * expert_parameters
        + shared_total
        + router_parameters
    )
    return {
        "routed_expert_parameters": routed_total,
        "shared_expert_parameters": shared_total,
        "router_parameters": router_parameters,
        "total_moe_parameters": routed_total + shared_total + router_parameters,
        "active_moe_parameters": active,
    }
