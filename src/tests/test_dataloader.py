import json

import numpy as np

from src.data.dataloader import create_dataloader


def test_dataloader_batches_shifted_windows(tmp_path) -> None:
    tokens = np.arange(30, dtype="<u2")
    tokens.tofile(tmp_path / "train.bin")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"dtype": "<u2"}),
        encoding="utf-8",
    )

    loader = create_dataloader(
        data_dir=str(tmp_path),
        split="train",
        batch_size=2,
        seq_len=4,
        stride=4,
        num_workers=0,
    )
    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    assert (inputs[:, 1:] == targets[:, :-1]).all()
