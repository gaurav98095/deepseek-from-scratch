"""Train and inspect a tokenizer on a small demonstration corpus."""

from src.data.tokenizer import BPETokenizer


def main() -> None:
    corpus = [
        "DeepSeek is a language model.",
        "DeepSeek uses attention.",
        "DeepSeek uses mixture of experts.",
        "Language models learn from text.",
        "Attention helps language models.",
        "We are building DeepSeek from scratch.",
    ] * 100

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=350)

    text = "DeepSeek uses attention."
    ids = tokenizer.encode(text)

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Number of merges:", len(tokenizer.merges))
    print("Original text:", text)
    print("Encoded IDs:", ids)
    print("Decoded text:", tokenizer.decode(ids))

    print("First 10 merges:")
    for pair, new_id in list(tokenizer.merges.items())[:10]:
        print(new_id, repr(tokenizer.vocab[new_id]))


if __name__ == "__main__":
    main()
