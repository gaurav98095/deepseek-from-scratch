"""A small byte-level Byte Pair Encoding tokenizer."""

from collections import Counter
import json
from pathlib import Path


def merge_pair(
    ids: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``."""

    merged: list[int] = []
    index = 0

    while index < len(ids):
        if index < len(ids) - 1 and (ids[index], ids[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1

    return merged


class BPETokenizer:
    """Train and use a compact byte-level BPE vocabulary.

    ``vocab_size`` includes the reserved EOS token. The first 256 IDs are
    always the individual byte values 0 through 255.
    """

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.eos_id: int | None = None
        self._reset()

    def _reset(self) -> None:
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        self.merges = {}
        self.eos_id = None

    def train(self, texts: list[str], vocab_size: int) -> None:
        """Learn frequent byte-pair merges from ``texts``.

        The requested vocabulary includes 256 base byte tokens and one EOS
        token. Training stops early when no pair occurs at least twice.
        """

        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257")

        self._reset()
        corpus = [list(text.encode("utf-8")) for text in texts]
        next_id = 256

        while next_id < vocab_size - 1:
            counts: Counter[tuple[int, int]] = Counter()
            for ids in corpus:
                counts.update(zip(ids, ids[1:]))

            if not counts:
                break

            pair, frequency = min(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if frequency < 2:
                break

            self.merges[pair] = next_id
            self.vocab[next_id] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )
            corpus = [merge_pair(ids, pair, next_id) for ids in corpus]
            next_id += 1

        self.eos_id = next_id

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode UTF-8 text using the learned merge order."""

        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            ids = merge_pair(ids, pair, new_id)

        if add_eos:
            if self.eos_id is None:
                raise RuntimeError("Train the tokenizer before adding EOS")
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs as strict UTF-8 text."""

        pieces: list[bytes] = []
        for token_id in ids:
            if token_id == self.eos_id:
                continue
            try:
                pieces.append(self.vocab[token_id])
            except KeyError as error:
                raise ValueError(f"Unknown token ID: {token_id}") from error

        try:
            return b"".join(pieces).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Token IDs do not form valid UTF-8") from error

    def save(self, path: str | Path) -> None:
        """Save vocabulary bytes, merge order, and the EOS ID as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vocab": {
                str(token_id): piece.hex()
                for token_id, piece in self.vocab.items()
            },
            "merges": [
                [left, right, new_id]
                for (left, right), new_id in self.merges.items()
            ],
            "eos_id": self.eos_id,
        }
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer previously written by :meth:`save`."""

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.vocab = {
            int(token_id): bytes.fromhex(piece)
            for token_id, piece in state["vocab"].items()
        }
        tokenizer.merges = {
            (left, right): new_id
            for left, right, new_id in state["merges"]
        }
        tokenizer.eos_id = state["eos_id"]
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Return the current vocabulary size, including EOS after training."""

        return len(self.vocab) + (self.eos_id is not None)
