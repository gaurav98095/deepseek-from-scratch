"""Score a full pretraining checkpoint on fixed validation windows."""

from __future__ import annotations

import argparse
import json

import torch

from .model import DeepSeekConfig, DeepSeekWithMTP
from .training.fixed_evaluate import FixedTokenWindows, evaluate_fixed_windows
from .training.routing_diagnostics import collect_routing_statistics


def load_model(path: str, device: torch.device) -> DeepSeekWithMTP:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "config" not in state or "model" not in state:
        raise ValueError("checkpoint is not a full pretraining checkpoint")
    model = DeepSeekWithMTP(
        DeepSeekConfig(**state["config"]),
        mtp_weight=float(state.get("mtp_weight", 0.3)),
    )
    model.load_state_dict(state["model"], strict=True)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--val-file", default="data/processed/val.bin")
    parser.add_argument("--dtype", default="<u2")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--routing", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(args.checkpoint, device)
    windows = FixedTokenWindows(
        args.val_file,
        seq_len=args.seq_len,
        dtype=args.dtype,
    )
    metrics = evaluate_fixed_windows(
        model.main_model,
        windows,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
    )
    output: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "step": torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        ).get("step"),
        "validation": metrics,
    }
    if args.routing:
        inputs = (
            batch[:, :-1]
            for batch in windows.batches(
                args.batch_size,
                max_windows=args.max_windows,
            )
        )
        output["routing"] = collect_routing_statistics(model.main_model, inputs)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
