"""Download and normalize TinyStories for later tokenizer/training steps.

The default mode streams the Hugging Face dataset and writes one JSONL record
per story. This keeps the preparation process bounded by the output buffer
instead of loading the complete dataset into memory.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
DATASET_NAME = "roneneldan/TinyStories"


@dataclass(frozen=True)
class PreparationConfig:
    """Inputs that define a reproducible TinyStories preparation run."""

    split: str = "train"
    output: Path = Path("data/processed/tinystories-train.jsonl")
    cache_dir: Path | None = None
    max_stories: int | None = None
    min_characters: int = 1


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Load through Hugging Face lazily, keeping offline helpers importable."""

    from datasets import load_dataset as huggingface_load_dataset

    return huggingface_load_dataset(*args, **kwargs)


def _progress(iterable: Iterable[dict[str, Any]], *, desc: str) -> Iterable[dict[str, Any]]:
    """Use tqdm when installed; keep the transformation easy to unit test."""

    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, unit="story")


def _story_records(dataset: Iterable[dict[str, Any]], config: PreparationConfig) -> Iterator[dict[str, Any]]:
    """Yield cleaned records while preserving source order."""

    emitted = 0
    for source_index, row in enumerate(dataset):
        story = row.get("text")
        if not isinstance(story, str):
            LOGGER.warning("Skipping row %d: expected a string in the 'text' field", source_index)
            continue

        story = story.replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(story) < config.min_characters:
            continue

        yield {"id": emitted, "text": story}
        emitted += 1
        if config.max_stories is not None and emitted >= config.max_stories:
            return


def prepare_dataset(config: PreparationConfig) -> int:
    """Download/stream TinyStories and write normalized JSONL records.

    Returns the number of records written. The Hugging Face cache is separate
    from the output file, so the output can be copied to another machine.
    """

    if config.max_stories is not None and config.max_stories < 1:
        raise ValueError("max_stories must be positive when provided")
    if config.min_characters < 0:
        raise ValueError("min_characters cannot be negative")

    LOGGER.info("Loading %s (%s) in streaming mode", DATASET_NAME, config.split)
    dataset = load_dataset(
        DATASET_NAME,
        split=config.split,
        streaming=True,
        cache_dir=str(config.cache_dir) if config.cache_dir else None,
    )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    records = _story_records(dataset, config)
    written = 0
    with config.output.open("w", encoding="utf-8") as output_file:
        for record in _progress(records, desc="Preparing stories"):
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    LOGGER.info("Wrote %d stories to %s", written, config.output)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="Dataset split to prepare (default: train)")
    parser.add_argument("--output", type=Path, default=PreparationConfig.output)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-stories", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--min-characters", type=int, default=1)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    prepare_dataset(
        PreparationConfig(
            split=args.split,
            output=args.output,
            cache_dir=args.cache_dir,
            max_stories=args.max_stories,
            min_characters=args.min_characters,
        )
    )


if __name__ == "__main__":
    main()
