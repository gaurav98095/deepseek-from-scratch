# Dataset preparation: TinyStories

This is the first data step in the DeepSeek-from-scratch project. It downloads
the public `roneneldan/TinyStories` dataset through Hugging Face Datasets and
writes a normalized JSON Lines file. Tokenization and binary packing are
deliberately separate future steps so each stage can be inspected and rerun.

## Reproduce this step

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.prepare_tinystories \
  --split train \
  --output data/processed/tinystories-train.jsonl \
  --cache-dir data/cache
```

The full `train` split is the intended substantial dataset run. It is streamed
from the dataset provider, so the script does not build a second full in-memory
copy. The Hugging Face cache and generated output are ignored by Git because
they are local data artifacts, not source code.

For a fast smoke test before committing a change:

```bash
python -m src.data.prepare_tinystories \
  --max-stories 100 \
  --output data/processed/tinystories-smoke.jsonl \
  --cache-dir data/cache
```

Each output line has this stable shape:

```json
{"id": 0, "text": "Once upon a time..."}
```

The `id` is assigned after filtering and follows the source iteration order.
Line endings are normalized and surrounding whitespace is removed. Rows without
a string `text` field are skipped with a warning.

## Files added in this step

- `src/data/prepare_tinystories.py`: CLI and reusable preparation function.
- `tests/test_prepare_tinystories.py`: offline tests using a fake dataset.
- `requirements.txt`: runtime and test dependencies.
- `.gitignore`: excludes downloaded/cache data and Python artifacts.

## Commit-by-commit record

Suggested first commit:

```text
feat(data): add streaming TinyStories preparation
```

That commit should contain exactly the files listed above. The next planned
data commit is tokenizer training/selection and conversion from JSONL to a
training-efficient token format; it should not be mixed into this dataset
download step.

