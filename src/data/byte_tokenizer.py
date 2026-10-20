"""A dependency-free byte tokenizer for the first runnable training pipeline."""

from __future__ import annotations


class ByteTokenizer:
    """Encode UTF-8 bytes plus BOS/EOS tokens for a small demo vocabulary."""

    # IDs 0-255 are reserved for raw UTF-8 byte values.
    bos_token_id = 256
    eos_token_id = 257
    # Two special tokens are added to the 256 possible byte values.
    vocab_size = 258

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        # UTF-8 makes this tokenizer work for stories containing non-ASCII text.
        tokens = list(text.encode("utf-8"))
        if add_bos:
            # BOS tells the model where a story starts.
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            # EOS tells the model where a story ends.
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        # Special tokens are ignored because only byte IDs can form text.
        byte_tokens = [token for token in tokens if 0 <= token < 256]
        # Replacement decoding keeps generation readable if bytes are incomplete.
        return bytes(byte_tokens).decode("utf-8", errors="replace")
