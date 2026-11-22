"""Self-contained smoke test for the resumable pretraining pipeline."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .pretrain import run_pretraining


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="deepseek-pretrain-") as directory:
        root = Path(directory)
        data_dir = root / "processed"
        checkpoint = root / "checkpoints" / "latest.pt"
        data_dir.mkdir(parents=True)
        rng = np.random.default_rng(27)
        for split in ("train", "val"):
            rng.integers(0, 64, size=2_000, dtype=np.uint16).tofile(
                data_dir / f"{split}.bin"
            )
        (data_dir / "metadata.json").write_text(
            json.dumps({"tokenizer_vocab_size": 64, "dtype": "<u2"}),
            encoding="utf-8",
        )

        args = argparse.Namespace(
            data_dir=str(data_dir),
            checkpoint=str(checkpoint),
            best_checkpoint=str(root / "checkpoints" / "best.pt"),
            resume=None,
            log_path=str(root / "training_log.csv"),
            steps=2,
            batch_size=2,
            accum_steps=2,
            seq_len=8,
            warmup_steps=1,
            eval_interval=1,
            eval_batches=1,
            save_interval=1,
            max_lr=3e-4,
            min_lr=3e-5,
            weight_decay=0.1,
            max_grad_norm=1.0,
            mtp_weight=0.3,
            gamma=0.01,
            seed=27,
            device="cpu",
            dtype=None,
            bf16=False,
            activation_checkpointing=False,
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
        result = run_pretraining(args)
        if result["step"] != 2 or not checkpoint.is_file():
            raise AssertionError("pretraining smoke test did not finish cleanly")
        print("pretraining smoke test passed")


if __name__ == "__main__":
    main()
