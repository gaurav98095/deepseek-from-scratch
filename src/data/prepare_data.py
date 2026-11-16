"""Stream, clean, tokenize, split, and serialize pretraining data."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

try:
    from .tokenizer import BPETokenizer
except ImportError:
    # Supports running this file directly from a debugger or from src/data/.
    from tokenizer import BPETokenizer


def clean_document(
    text: object,
    *,
    min_chars: int = 200,
    min_words: int = 40,
) -> str | None:
    """Normalize a document and reject very short records."""

    if not isinstance(text, str):
        return None

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "").strip()

    if len(text) < min_chars or len(text.split()) < min_words:
        return None
    return text


def document_split(text: str, val_fraction: float = 0.02) -> str:
    """Assign a document deterministically to ``train`` or ``val``."""

    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], "big")
    fraction = number / (2**64)
    return "val" if fraction < val_fraction else "train"


def _next_clean_document(
    iterator: Iterator[dict],
) -> str | None:
    """Read until the next usable document or the stream ends."""

    while True:
        try:
            record = next(iterator)
        except StopIteration:
            return None

        text = clean_document(record.get("text"))
        if text is not None:
            return text


def _collect_tokenizer_texts(
    iterator: Iterator[dict],
    limit: int,
) -> list[str]:
    texts = []
    while len(texts) < limit:
        text = _next_clean_document(iterator)
        if text is None:
            break
        texts.append(text)
    return texts


def _stream_dataset(
    dataset_name: str,
    dataset_config: str,
) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install the dataset dependencies with "
            "`python -m pip install -r requirements.txt`"
        ) from error

    return load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
    )


def prepare_dataset(
    *,
    output_dir: str | Path = "data/processed",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    tokenizer_documents: int = 1_000,
    tokenizer_vocab_size: int = 350,
    max_documents: int = 5_000,
    val_fraction: float = 0.02,
    dtype: str = "<u2",
) -> dict:
    """Prepare a bounded streaming sample and return its metadata."""

    if tokenizer_documents <= 0 or max_documents <= 0:
        raise ValueError("document limits must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_dtype = np.dtype(dtype)

    iterator = iter(_stream_dataset(dataset_name, dataset_config))
    tokenizer_texts = _collect_tokenizer_texts(iterator, tokenizer_documents)
    if not tokenizer_texts:
        raise RuntimeError("No usable documents found for tokenizer training")

    tokenizer = BPETokenizer()
    tokenizer.train(tokenizer_texts, vocab_size=tokenizer_vocab_size)
    tokenizer.save(output_dir / "tokenizer.json")

    token_counts = {"train": 0, "val": 0}
    document_counts = {"train": 0, "val": 0}
    paths = {
        split: output_dir / f"{split}.bin"
        for split in ("train", "val")
    }

    with paths["train"].open("wb") as train_file, paths["val"].open(
        "wb"
    ) as val_file:
        writers = {"train": train_file, "val": val_file}
        accepted = 0

        while accepted < max_documents:
            text = _next_clean_document(iterator)
            if text is None:
                break

            split = document_split(text, val_fraction)
            ids = tokenizer.encode(text, add_eos=True)
            if not ids:
                continue
            if max(ids) > np.iinfo(storage_dtype).max:
                raise ValueError("Token ID exceeds dtype capacity")

            np.asarray(ids, dtype=storage_dtype).tofile(writers[split])
            token_counts[split] += len(ids)
            document_counts[split] += 1
            accepted += 1

    metadata = {
        "source": dataset_name,
        "config": dataset_config,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "dtype": storage_dtype.str,
        "tokens": token_counts,
        "documents": document_counts,
        "tokenizer_training_documents": len(tokenizer_texts),
        "val_fraction": val_fraction,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a bounded FineWeb-Edu token stream."
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-documents", type=int, default=5_000)
    parser.add_argument("--tokenizer-documents", type=int, default=1_000)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=350)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--dtype", default="<u2")
    args = parser.parse_args()

    metadata = prepare_dataset(
        output_dir=args.output_dir,
        max_documents=args.max_documents,
        tokenizer_documents=args.tokenizer_documents,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        val_fraction=args.val_fraction,
        dtype=args.dtype,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
