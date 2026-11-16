import json

import numpy as np
import pytest

from src.data.dataset import TokenDataset


def write_token_split(tmp_path, name: str = "train"):
    tokens = np.array(
        [10, 20, 30, 40, 50, 60, 70, 80],
        dtype="<u2",
    )
    tokens.tofile(tmp_path / f"{name}.bin")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"dtype": "<u2"}),
        encoding="utf-8",
    )


def test_window_and_target_shift(tmp_path) -> None:
    write_token_split(tmp_path)
    dataset = TokenDataset(
        data_dir=tmp_path,
        split="train",
        seq_len=4,
        stride=1,
    )

    assert len(dataset) == 4
    x, y = dataset[0]
    assert x.tolist() == [10, 20, 30, 40]
    assert y.tolist() == [20, 30, 40, 50]

    x2, y2 = dataset[2]
    assert x2.tolist() == [30, 40, 50, 60]
    assert y2.tolist() == [40, 50, 60, 70]


def test_invalid_split_and_index(tmp_path) -> None:
    write_token_split(tmp_path)

    with pytest.raises(ValueError):
        TokenDataset(tmp_path, split="test")

    dataset = TokenDataset(tmp_path, seq_len=4, stride=1)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_sequence_requires_t_plus_one_tokens(tmp_path) -> None:
    write_token_split(tmp_path)

    with pytest.raises(ValueError):
        TokenDataset(tmp_path, seq_len=8)
