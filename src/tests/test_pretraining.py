import json

import numpy as np
import pytest
import torch

from src.data.token_batches import TokenBatcher
from src.model import DeepSeekConfig, DeepSeekWithMTP
from src.training import (
    MoELoadBalancer,
    create_optimizer,
    evaluate_main_model,
    load_training_checkpoint,
    save_training_checkpoint,
)


def small_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=64,
        max_seq_len=8,
        dim=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        inter_dim=64,
        moe_inter_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )


def make_data(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    values = np.arange(256, dtype=np.uint16) % 64
    for split in ("train", "val"):
        values.tofile(data_dir / f"{split}.bin")
    (data_dir / "metadata.json").write_text(
        json.dumps({"tokenizer_vocab_size": 64, "dtype": "<u2"}),
        encoding="utf-8",
    )
    return data_dir


def test_token_batcher_state_reproduces_next_batch(tmp_path) -> None:
    data_dir = make_data(tmp_path)
    first = TokenBatcher(data_dir / "train.bin", seq_len=8, dtype="<u2", seed=27)
    first.next_batch(2)
    state = first.state_dict()
    expected = first.next_batch(2)

    second = TokenBatcher(data_dir / "train.bin", seq_len=8, dtype="<u2", seed=99)
    second.load_state_dict(state)
    torch.testing.assert_close(expected, second.next_batch(2))


def test_evaluate_reports_main_loss_and_restores_mode(tmp_path) -> None:
    data_dir = make_data(tmp_path)
    batcher = TokenBatcher(data_dir / "val.bin", seq_len=8, dtype="<u2", seed=1)
    model = DeepSeekWithMTP(small_config())
    model.train()
    result = evaluate_main_model(
        model.main_model,
        batcher,
        batch_size=2,
        num_batches=2,
    )
    assert model.main_model.training
    assert result["evaluated_tokens"] == 2 * 2 * 8
    assert result["val_loss"] > 0
    assert result["perplexity"] > 1


def test_full_checkpoint_restores_training_state(tmp_path) -> None:
    data_dir = make_data(tmp_path)
    train_batcher = TokenBatcher(
        data_dir / "train.bin", seq_len=8, dtype="<u2", seed=1
    )
    val_batcher = TokenBatcher(data_dir / "val.bin", seq_len=8, dtype="<u2", seed=2)
    model = DeepSeekWithMTP(small_config())
    optimizer = create_optimizer(model)
    settings = {"steps": 2, "seq_len": 8}
    train_batcher.next_batch(2)
    val_batcher.next_batch(1)
    path = tmp_path / "latest.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=3,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        run_settings=settings,
        best_val_loss=2.5,
    )

    restored_model = DeepSeekWithMTP(small_config())
    restored_optimizer = create_optimizer(restored_model)
    restored_train = TokenBatcher(
        data_dir / "train.bin", seq_len=8, dtype="<u2", seed=99
    )
    restored_val = TokenBatcher(data_dir / "val.bin", seq_len=8, dtype="<u2", seed=99)
    state = load_training_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        train_batcher=restored_train,
        val_batcher=restored_val,
        expected_run_settings=settings,
    )

    assert state["step"] == 3
    assert state["best_val_loss"] == 2.5
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored_model.state_dict()[key])

    with pytest.raises(ValueError, match="run settings"):
        load_training_checkpoint(
            path,
            model=restored_model,
            optimizer=restored_optimizer,
            train_batcher=restored_train,
            val_batcher=restored_val,
            expected_run_settings={"steps": 4},
        )
