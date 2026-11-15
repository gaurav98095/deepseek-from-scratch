"""Command-line entry point for training experiments."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DeepSeek-style training experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiny.yaml"),
        help="Path to the experiment configuration.",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        parser.error(f"Configuration file not found: {args.config}")

    print(f"Selected configuration: {args.config}")
    print("Training loop is ready to be implemented.")


if __name__ == "__main__":
    main()
