"""End-to-end software integration checks for the final model path."""

from __future__ import annotations

import json

import numpy as np
import torch

from src.data.token_batches import TokenBatcher
from src.data.validate_tokens import validate_dataset
from src.generate import generate
from src.model import DeepSeekConfig, DeepSeekWithMTP
from src.training import MoELoadBalancer, create_optimizer, efficient_train_step
from src.training.full_checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def _write_dataset(tmp_path) -> tuple[object, object]:
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    values = (np.arange(256, dtype=np.uint16) % 64).astype(np.uint16)
    for split in ("train", "val"):
        values.tofile(data_dir / f"{split}.bin")
    (data_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tokenizer_vocab_size": 64,
                "dtype": "<u2",
            }
        ),
        encoding="utf-8",
    )
    return data_dir, values


def _config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=64,
        max_seq_len=16,
        dim=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        inter_dim=64,
        moe_inter_dim=16,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )


def test_dataset_preflight_accepts_valid_token_files(tmp_path) -> None:
    data_dir, _ = _write_dataset(tmp_path)
    report = validate_dataset(data_dir)
    assert report["vocab_size"] == 64
    assert report["splits"]["train"]["tokens"] == 256


def test_complete_training_checkpoint_and_generation_lifecycle(tmp_path) -> None:
    torch.manual_seed(42)
    data_dir, _ = _write_dataset(tmp_path)
    train_batcher = TokenBatcher(data_dir / "train.bin", seq_len=8, seed=1)
    val_batcher = TokenBatcher(data_dir / "val.bin", seq_len=8, seed=2)
    model = DeepSeekWithMTP(_config())
    optimizer = create_optimizer(model)
    tokens = train_batcher.next_batch(2)

    with MoELoadBalancer(model) as balancer:
        metrics = efficient_train_step(
            model,
            optimizer,
            balancer,
            [tokens],
            step=1,
            total_steps=2,
            warmup_steps=1,
            use_bf16=False,
        )
    assert np.isfinite(metrics["loss"])

    checkpoint_path = tmp_path / "latest.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=1,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        run_settings={"steps": 2, "seq_len": 8},
    )

    restored = DeepSeekWithMTP(_config())
    restored_optimizer = create_optimizer(restored)
    restored_train = TokenBatcher(data_dir / "train.bin", seq_len=8, seed=99)
    restored_val = TokenBatcher(data_dir / "val.bin", seq_len=8, seed=99)
    load_training_checkpoint(
        checkpoint_path,
        model=restored,
        optimizer=restored_optimizer,
        train_batcher=restored_train,
        val_batcher=restored_val,
        expected_run_settings={"steps": 2, "seq_len": 8},
    )

    prompt = tokens[:, :5]
    with torch.no_grad():
        full_logits = restored.main_model(prompt)
        restored.main_model.eval()
        restored.main_model.prefill(prompt[:, :-1])
        cached_logits = restored.main_model.decode_step(prompt[:, -1:])
    torch.testing.assert_close(
        full_logits[:, -1:, :],
        cached_logits,
        rtol=1e-4,
        atol=1e-4,
    )

    generated = generate(
        restored.main_model,
        prompt[:, :3],
        max_new_tokens=2,
        greedy=True,
    )
    assert generated.shape == (2, 5)
    assert restored.main_model.cache_length == 0
