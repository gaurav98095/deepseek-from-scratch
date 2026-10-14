"""Stream a reproducible-sized text corpus from Hugging Face and keep it locally."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--split", default="train")
    parser.add_argument("--field", default="text", help="Dataset column containing text")
    parser.add_argument("--output", type=Path, default=Path("data/tinystories-train.txt"))
    parser.add_argument("--min-bytes", type=int, default=100 * 1024 * 1024, help="Existing file size accepted as already downloaded")
    parser.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024, help="Maximum local corpus size")
    parser.add_argument("--force", action="store_true", help="Replace an existing corpus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_bytes < 1 or args.max_bytes < args.min_bytes:
        raise ValueError("max-bytes must be at least min-bytes, and both must be positive")
    if args.output.exists() and not args.force and args.output.stat().st_size >= args.min_bytes:
        print(f"Using existing dataset: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")
        return

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install dependencies first with: python -m pip install -r requirements.txt") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.partial")
    if temporary_output.exists():
        temporary_output.unlink()
    print(f"Streaming {args.dataset} ({args.split}) to {args.output} ...")
    bytes_written = 0
    examples_written = 0
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    try:
        with temporary_output.open("w", encoding="utf-8") as output_file:
            for example in dataset:
                text = str(example[args.field]).strip()
                if not text:
                    continue
                record = text + "\n\n"
                record_bytes = len(record.encode("utf-8"))
                if bytes_written + record_bytes > args.max_bytes:
                    break
                output_file.write(record)
                bytes_written += record_bytes
                examples_written += 1
                if examples_written % 10_000 == 0:
                    print(f"  examples={examples_written:,} size={bytes_written / 1024**2:.1f} MiB")
            output_file.flush()
            os.fsync(output_file.fileno())
        if bytes_written < args.min_bytes:
            raise RuntimeError(f"Only downloaded {bytes_written:,} bytes; increase the available dataset or lower --min-bytes")
        temporary_output.replace(args.output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise
    print(f"Saved {examples_written:,} examples / {bytes_written / 1024**2:.1f} MiB to {args.output}")


if __name__ == "__main__":
    main()
