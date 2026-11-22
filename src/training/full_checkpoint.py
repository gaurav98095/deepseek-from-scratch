"""Atomic, resumable checkpoints for the complete pretraining loop."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..data.token_batches import TokenBatcher
from ..model.deepseek_with_mtp import DeepSeekWithMTP


def _cuda_rng_state() -> list[torch.Tensor]:
    if not torch.cuda.is_available():
        return []
    return [state.cpu() for state in torch.cuda.get_rng_state_all()]


def save_training_checkpoint(
    path: str | Path,
    *,
    model: DeepSeekWithMTP,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_batcher: TokenBatcher,
    val_batcher: TokenBatcher,
    run_settings: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    best_val_loss: float | None = None,
) -> None:
    """Save model, optimizer, samplers, RNG state, and run metadata atomically."""
    if step < 0:
        raise ValueError("step must be nonnegative")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "version": 1,
        "config": asdict(model.config),
        "mtp_weight": model.mtp_weight,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "train_batcher": train_batcher.state_dict(),
        "val_batcher": val_batcher.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": _cuda_rng_state(),
        "run_settings": dict(run_settings),
        "metrics": dict(metrics or {}),
        "best_val_loss": best_val_loss,
    }
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def _check_mapping(
    saved: object,
    expected: dict[str, Any] | None,
    label: str,
) -> None:
    if expected is None:
        return
    if not isinstance(saved, dict):
        raise ValueError(f"checkpoint is missing {label}")
    mismatches = {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint {label} does not match: {mismatches}")


def load_training_checkpoint(
    path: str | Path,
    *,
    model: DeepSeekWithMTP,
    optimizer: torch.optim.Optimizer,
    train_batcher: TokenBatcher,
    val_batcher: TokenBatcher,
    expected_run_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint into already-constructed training objects."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a mapping")
    if state.get("config") != asdict(model.config):
        raise ValueError("checkpoint model configuration does not match")
    if float(state.get("mtp_weight", -1.0)) != model.mtp_weight:
        raise ValueError("checkpoint MTP weight does not match")
    _check_mapping(state.get("run_settings"), expected_run_settings, "run settings")

    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    train_batcher.load_state_dict(state["train_batcher"])
    val_batcher.load_state_dict(state["val_batcher"])
    torch.set_rng_state(state["torch_rng"])
    cuda_rng = state.get("cuda_rng", [])
    if torch.cuda.is_available() and cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    return state
