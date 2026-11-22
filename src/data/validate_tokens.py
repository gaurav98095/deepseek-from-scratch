"""Preflight validation for serialized token datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def validate_dataset(data_dir: str | Path) -> dict[str, Any]:
    """Validate metadata, binary layout, dtype capacity, and token ranges."""
    root = Path(data_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vocab_size = int(
        metadata.get("tokenizer_vocab_size", metadata.get("vocab_size", 0))
    )
    dtype_name = metadata.get("dtype", metadata.get("token_dtype"))
    if vocab_size < 2:
        raise ValueError("metadata contains an invalid vocabulary size")
    if dtype_name is None:
        raise ValueError("metadata must contain dtype")

    dtype = np.dtype(dtype_name)
    if dtype.kind != "u":
        raise ValueError("token dtype must be an unsigned integer")
    if vocab_size - 1 > np.iinfo(dtype).max:
        raise ValueError("token dtype cannot represent the vocabulary")

    report: dict[str, Any] = {
        "data_dir": str(root),
        "vocab_size": vocab_size,
        "dtype": dtype.str,
        "splits": {},
    }
    for split in ("train", "val"):
        path = root / f"{split}.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size % dtype.itemsize:
            raise ValueError(f"{path} has a partial token at the end")

        tokens = np.memmap(path, dtype=dtype, mode="r")
        if len(tokens) == 0:
            raise ValueError(f"{path} is empty")
        minimum = int(tokens.min())
        maximum = int(tokens.max())
        if minimum < 0 or maximum >= vocab_size:
            raise ValueError(
                f"{path} contains token IDs outside [0, {vocab_size})"
            )
        report["splits"][split] = {
            "path": str(path),
            "tokens": len(tokens),
            "min_id": minimum,
            "max_id": maximum,
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
