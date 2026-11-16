from src.data.prepare_data import clean_document, document_split


def test_clean_document_normalizes_and_keeps_long_text() -> None:
    text = "word " * 50 + "\r\nextra"

    cleaned = clean_document(text)

    assert cleaned is not None
    assert "\r" not in cleaned
    assert cleaned.startswith("word")


def test_clean_document_rejects_short_text() -> None:
    assert clean_document("too short") is None
    assert clean_document(None) is None


def test_document_split_is_deterministic() -> None:
    text = "A stable document with a deterministic assignment."

    assert document_split(text) == document_split(text)


def test_document_split_extreme_fractions() -> None:
    text = "Any document"

    assert document_split(text, val_fraction=0.0) == "train"
    assert document_split(text, val_fraction=1.0) == "val"
