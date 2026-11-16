"""Inspect serialized token data and its metadata."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .tokenizer import BPETokenizer
except ImportError:
    # Supports running this file directly from a debugger or from src/data/.
    from tokenizer import BPETokenizer


def inspect_dataset(output_dir: str | Path = "data/processed") -> dict:
    output_dir = Path(output_dir)
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    tokenizer = BPETokenizer.load(output_dir / "tokenizer.json")
    dtype = np.dtype(metadata["dtype"])
    train = np.memmap(output_dir / "train.bin", dtype=dtype, mode="r")
    validation = np.memmap(output_dir / "val.bin", dtype=dtype, mode="r")

    if len(train) == 0:
        raise ValueError("Training split is empty")

    if train.max() >= tokenizer.vocab_size:
        raise ValueError("Training token ID exceeds tokenizer vocabulary")
    if len(train) != metadata["tokens"]["train"]:
        raise ValueError("Training token count does not match metadata")
    if len(validation) != metadata["tokens"]["val"]:
        raise ValueError("Validation token count does not match metadata")

    summary = {
        "training_tokens": len(train),
        "validation_tokens": len(validation),
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype.str,
        "first_training_ids": train[:20].tolist(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prepared token files.")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
