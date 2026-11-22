"""Checkpoint save/load helpers for the consolidated DeepSeek model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.model.config import DeepSeekConfig
from src.model.final_deepseek import TinyDeepSeek


def save_checkpoint(
    path: str | Path,
    model: TinyDeepSeek,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    **metadata: Any,
) -> None:
    """Save model configuration, weights, optional optimizer, and metadata."""
    if step < 0:
        raise ValueError("step must be nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "step": step,
        **metadata,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[TinyDeepSeek, dict[str, Any]]:
    """Restore a consolidated model and optionally its optimizer state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    try:
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
    except KeyError as error:
        raise ValueError("checkpoint is missing model configuration or weights") from error

    model = TinyDeepSeek(DeepSeekConfig(**config)).to(device)
    model.load_state_dict(state_dict)
    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()
    return model, checkpoint


def load_inference_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[TinyDeepSeek, dict[str, Any]]:
    """Load model weights from either model-only or full training checkpoints."""
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "config" not in state:
        raise ValueError("checkpoint is missing model configuration")

    if "model_state_dict" in state:
        model_state = state["model_state_dict"]
    elif "model" in state:
        model_state = {
            key.removeprefix("main_model."): value
            for key, value in state["model"].items()
            if key.startswith("main_model.")
        }
    else:
        raise ValueError("checkpoint is missing model weights")

    model = TinyDeepSeek(DeepSeekConfig(**state["config"])).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    model.reset_cache()
    return model, state
