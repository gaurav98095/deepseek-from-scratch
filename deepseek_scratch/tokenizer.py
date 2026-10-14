"""The character tokenizer used for the first from-scratch experiments."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CharacterTokenizer:
    """Deterministic character-level tokenizer with an explicit unknown token."""

    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "<unk>"

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("text must contain at least one character")
        chars = sorted(set(text))
        stoi = {"<unk>": 0}
        stoi.update({char: index + 1 for index, char in enumerate(chars)})
        return cls(stoi=stoi, itos={index: token for token, index in stoi.items()})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(char, self.stoi[self.unk_token]) for char in text]

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().flatten().tolist()
        return "".join(self.itos.get(int(token_id), self.unk_token) for token_id in token_ids)

    def encode_tensor(self, text: str, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long, device=device)
