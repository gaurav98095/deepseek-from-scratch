"""Plot training and validation metrics from the Module 27 CSV log."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_training_log(
    log_path: str | Path,
    output_path: str | Path = "training_curves.png",
) -> None:
    """Write a loss, perplexity, and generalization-gap plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib; install requirements.txt first"
        ) from error

    rows = read_rows(log_path)
    if not rows:
        raise ValueError("training log is empty")
    steps = [int(row["step"]) for row in rows]
    main_losses = [float(row["main_loss"]) for row in rows]
    validation = [
        (int(row["step"]), float(row["val_loss"]))
        for row in rows
        if row.get("val_loss")
    ]
    val_steps = [step for step, _loss in validation]
    val_losses = [loss for _step, loss in validation]

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(steps, main_losses, label="Training main loss", alpha=0.75)
    if validation:
        axes[0].plot(val_steps, val_losses, "o-", label="Validation loss")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if validation:
        train_by_step = {step: loss for step, loss in zip(steps, main_losses)}
        gap_steps = [step for step in val_steps if step in train_by_step]
        gaps = [
            next(loss for step, loss in validation if step == current)
            - train_by_step[current]
            for current in gap_steps
        ]
        axes[1].plot(gap_steps, gaps, "o-", label="Validation - training main loss")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Generalization gap")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="training_log.csv")
    parser.add_argument("--output", default="training_curves.png")
    args = parser.parse_args()
    plot_training_log(args.log, args.output)
    print(f"saved plot: {args.output}")


if __name__ == "__main__":
    main()
