"""Stable identities for tokenizer artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file as a hexadecimal string."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: str | Path, expected: str) -> None:
    """Raise when a file does not match the recorded SHA-256 digest."""
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {Path(path)}: expected {expected}, got {actual}"
        )
