# DeepSeek from scratch

An educational, reproducible implementation of a small decoder-only language
model, developed step by step. The repository keeps runnable source code in
`src/` and records the exact workflow in `docs/` so each stage can be rebuilt
from a clean checkout.

## Current stage

Dataset setup is implemented for TinyStories. See
[`docs/data-preparation.md`](docs/data-preparation.md) for installation,
download, smoke-test, output-format, and commit instructions.

Run the offline tests with:

```bash
pytest
```

Large downloads and generated datasets are intentionally not committed.

