"""Training wrapper that combines the main loss with one MTP loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import DeepSeekConfig
from .final_deepseek import TinyDeepSeek
from .mtp import MTPModule


class DeepSeekWithMTP(nn.Module):
    """Final model plus a one-stage sequential MTP training objective."""

    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        mtp_weight: float = 0.3,
    ) -> None:
        super().__init__()
        if mtp_weight < 0:
            raise ValueError("mtp_weight must be nonnegative")
        self.config = config
        self.mtp_weight = mtp_weight
        self.main_model = TinyDeepSeek(config)
        self.mtp_module = MTPModule(config)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute main next-token and two-ahead MTP losses.

        For a sequence of length ``L``, the main path uses ``L-1`` positions
        and the one-stage MTP path uses ``L-2`` aligned positions.
        """
        if tokens.ndim != 2:
            raise ValueError("Expected tokens with shape [batch, sequence]")
        if tokens.shape[1] < 3:
            raise ValueError("MTP requires sequences with at least 3 tokens")
        if tokens.shape[1] - 1 > self.config.max_seq_len:
            raise ValueError("Main input sequence exceeds maximum length")

        main_inputs = tokens[:, :-1]
        main_targets = tokens[:, 1:]
        main_logits, main_hidden = self.main_model(
            main_inputs,
            return_hidden=True,
        )

        mtp_hidden = main_hidden[:, :-1]
        mtp_next_tokens = tokens[:, 1:-1]
        mtp_targets = tokens[:, 2:]
        mtp_logits = self.mtp_module(
            main_hidden=mtp_hidden,
            next_token_ids=mtp_next_tokens,
            shared_embedding=self.main_model.embed_tokens,
            shared_lm_head=self.main_model.lm_head,
        )

        main_loss = F.cross_entropy(
            main_logits.float().reshape(-1, self.config.vocab_size),
            main_targets.reshape(-1),
        )
        mtp_loss = F.cross_entropy(
            mtp_logits.float().reshape(-1, self.config.vocab_size),
            mtp_targets.reshape(-1),
        )
        loss = main_loss + self.mtp_weight * mtp_loss
        return {
            "loss": loss,
            "main_loss": main_loss,
            "mtp_loss": mtp_loss,
            "main_logits": main_logits,
            "mtp_logits": mtp_logits,
            "main_hidden": main_hidden,
        }

    def prefill(self, input_ids: torch.Tensor, **kwargs):
        """Delegate ordinary inference to the main model."""
        return self.main_model.prefill(input_ids, **kwargs)

    def decode_step(self, input_ids: torch.Tensor, **kwargs):
        """Delegate one-token cached decoding to the main model."""
        return self.main_model.decode_step(input_ids, **kwargs)

    def reset_cache(self) -> None:
        """Clear the main model's inference caches."""
        self.main_model.reset_cache()
