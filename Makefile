PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

DATA_DIR ?= data/processed
CHECKPOINT ?= checkpoints/latest.pt
BEST_CHECKPOINT ?= checkpoints/best.pt
LOG_PATH ?= training_log.csv
VAL_FILE ?= data/processed/val.bin
EVAL_WINDOWS ?= 128
PLOT_OUTPUT ?= training_curves.png
STEPS ?= 2000
BATCH_SIZE ?= 4
ACCUM_STEPS ?= 4
SEQ_LEN ?= 128
WARMUP_STEPS ?= 100
EVAL_INTERVAL ?= 100
SAVE_INTERVAL ?= 200

.PHONY: help prepare-data test test-module-27 test-module-28 test-generation smoke-pretrain run-pretraining resume-pretraining evaluate-checkpoint routing-report generate plot-training

help:
	@printf '%s\n' \
		'make prepare-data       Build data/processed/train.bin and val.bin' \
		'make test               Run the complete test suite' \
		'make test-module-27     Run Module 27 focused tests' \
		'make test-module-28     Run Module 28 focused tests' \
		'make test-generation    Run generation and sampling tests' \
		'make smoke-pretrain     Run a tiny end-to-end pretraining smoke test' \
		'make run-pretraining    Start the default 2,000-step training run' \
		'make resume-pretraining Resume from CHECKPOINT' \
		'make evaluate-checkpoint Score a checkpoint on fixed validation windows' \
		'make routing-report     Report expert utilization for a checkpoint' \
		'make generate           Generate text from a checkpoint' \
		'make plot-training      Plot the training CSV'

prepare-data:
	$(PYTHON) -m src.data.prepare_data --output-dir $(DATA_DIR)

test:
	$(PYTEST) src/tests -v

test-module-27:
	$(PYTEST) src/tests/test_pretraining.py src/tests/test_efficient_training.py -v

test-module-28:
	$(PYTEST) src/tests/test_module_28.py -v

test-generation:
	$(PYTEST) src/tests/test_generation.py -v

smoke-pretrain:
	$(PYTHON) -m src.smoke_pretrain

run-pretraining:
	$(PYTHON) -m src.pretrain \
		--data-dir $(DATA_DIR) \
		--checkpoint $(CHECKPOINT) \
		--best-checkpoint $(BEST_CHECKPOINT) \
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
		--best-checkpoint $(BEST_CHECKPOINT) \
		--resume $(CHECKPOINT) \
		--log-path $(LOG_PATH) \
		--steps $(STEPS) \
		--batch-size $(BATCH_SIZE) \
		--accum-steps $(ACCUM_STEPS) \
		--seq-len $(SEQ_LEN) \
		--warmup-steps $(WARMUP_STEPS) \
		--eval-interval $(EVAL_INTERVAL) \
		--save-interval $(SAVE_INTERVAL)

evaluate-checkpoint:
	$(PYTHON) -m src.evaluate_checkpoint \
		--checkpoint $(CHECKPOINT) \
		--val-file $(VAL_FILE) \
		--seq-len $(SEQ_LEN) \
		--batch-size $(BATCH_SIZE) \
		--max-windows $(EVAL_WINDOWS)

routing-report:
	$(PYTHON) -m src.evaluate_checkpoint \
		--checkpoint $(CHECKPOINT) \
		--val-file $(VAL_FILE) \
		--seq-len $(SEQ_LEN) \
		--batch-size $(BATCH_SIZE) \
		--max-windows $(EVAL_WINDOWS) \
		--routing

generate:
	$(PYTHON) -m src.generate \
		--checkpoint $(CHECKPOINT) \
		--max-new-tokens 50 \
		--temperature 0.8 \
		--top-p 0.9

plot-training:
	$(PYTHON) -m src.plot_training --log $(LOG_PATH) --output $(PLOT_OUTPUT)
