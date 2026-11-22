PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

DATA_DIR ?= data/processed
CHECKPOINT ?= checkpoints/latest.pt
LOG_PATH ?= training_log.csv
STEPS ?= 2000
BATCH_SIZE ?= 4
ACCUM_STEPS ?= 4
SEQ_LEN ?= 128
WARMUP_STEPS ?= 100
EVAL_INTERVAL ?= 100
SAVE_INTERVAL ?= 200

.PHONY: help prepare-data test test-module-27 smoke-pretrain run-pretraining resume-pretraining

help:
	@printf '%s\n' \
		'make prepare-data       Build data/processed/train.bin and val.bin' \
		'make test               Run the complete test suite' \
		'make test-module-27     Run Module 27 focused tests' \
		'make smoke-pretrain     Run a tiny end-to-end pretraining smoke test' \
		'make run-pretraining    Start the default 2,000-step training run' \
		'make resume-pretraining Resume from CHECKPOINT'

prepare-data:
	$(PYTHON) -m src.data.prepare_data --output-dir $(DATA_DIR)

test:
	$(PYTEST) src/tests -v

test-module-27:
	$(PYTEST) src/tests/test_pretraining.py src/tests/test_efficient_training.py -v

smoke-pretrain:
	$(PYTHON) -m src.smoke_pretrain

run-pretraining:
	$(PYTHON) -m src.pretrain \
		--data-dir $(DATA_DIR) \
		--checkpoint $(CHECKPOINT) \
		--log-path $(LOG_PATH) \
		--steps $(STEPS) \
		--batch-size $(BATCH_SIZE) \
		--accum-steps $(ACCUM_STEPS) \
		--seq-len $(SEQ_LEN) \
		--warmup-steps $(WARMUP_STEPS) \
		--eval-interval $(EVAL_INTERVAL) \
		--save-interval $(SAVE_INTERVAL)

resume-pretraining:
	$(PYTHON) -m src.pretrain \
		--data-dir $(DATA_DIR) \
		--checkpoint $(CHECKPOINT) \
		--resume $(CHECKPOINT) \
		--log-path $(LOG_PATH) \
		--steps $(STEPS) \
		--batch-size $(BATCH_SIZE) \
		--accum-steps $(ACCUM_STEPS) \
		--seq-len $(SEQ_LEN) \
		--warmup-steps $(WARMUP_STEPS) \
		--eval-interval $(EVAL_INTERVAL) \
		--save-interval $(SAVE_INTERVAL)
