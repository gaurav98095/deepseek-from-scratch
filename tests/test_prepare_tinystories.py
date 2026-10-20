import json

import pytest

from src.data.prepare_tinystories import PreparationConfig, _story_records, prepare_dataset


def test_story_records_normalizes_text_and_skips_invalid_rows() -> None:
    # Preparation should produce clean text and ignore malformed dataset rows.
    rows = [{"text": "  Once\r\nupon a time.  "}, {"text": None}, {"other": "ignored"}]

    assert list(_story_records(rows, PreparationConfig())) == [{"id": 0, "text": "Once\nupon a time."}]


def test_story_records_respects_max_stories() -> None:
    # The smoke-test limit should stop iteration without consuming extra stories.
    rows = ({"text": f"story {index}"} for index in range(3))

    assert list(_story_records(rows, PreparationConfig(max_stories=2))) == [
        {"id": 0, "text": "story 0"},
        {"id": 1, "text": "story 1"},
    ]


def test_prepare_dataset_writes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Replace the network dataset with deterministic rows for an offline test.
    monkeypatch.setattr(
        "src.data.prepare_tinystories.load_dataset",
        lambda *args, **kwargs: [{"text": "A story."}, {"text": "Another story."}],
    )
    output = tmp_path / "nested" / "stories.jsonl"

    assert prepare_dataset(PreparationConfig(output=output)) == 2
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [
        {"id": 0, "text": "A story."},
        {"id": 1, "text": "Another story."},
    ]


def test_invalid_max_stories_is_rejected(tmp_path) -> None:
    # Invalid limits should fail before any output is created.
    with pytest.raises(ValueError, match="max_stories"):
        prepare_dataset(PreparationConfig(output=tmp_path / "stories.jsonl", max_stories=0))
