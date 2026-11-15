import pytest

from src.data.tokenizer import BPETokenizer, merge_pair


@pytest.fixture
def tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer()
    corpus = [
        "hello world",
        "hello DeepSeek",
        "world of language models",
    ] * 100
    tokenizer.train(corpus, vocab_size=300)
    return tokenizer


def test_merge_pair() -> None:
    assert merge_pair([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]


def test_roundtrip(tokenizer: BPETokenizer) -> None:
    text = "hello world"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unicode_roundtrip(tokenizer: BPETokenizer) -> None:
    text = "hello 🚀"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_eos(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("hello", add_eos=True)

    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hello"


def test_vocab_size_includes_eos(tokenizer: BPETokenizer) -> None:
    assert tokenizer.vocab_size <= 300
    assert tokenizer.eos_id is not None


def test_invalid_vocab_size() -> None:
    with pytest.raises(ValueError):
        BPETokenizer().train(["hello"], vocab_size=256)
