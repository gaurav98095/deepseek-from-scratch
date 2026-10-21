# Use the repository virtual environment for every Python operation.
.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip
TENSORBOARD := .venv/bin/tensorboard
DEPENDENCY_STAMP := .venv/.requirements-installed

# Keep paths configurable so CI, local machines, and future servers can reuse this file.
DEV_DATA := data/processed/tinystories-smoke.jsonl
PROD_DATA := data/processed/tinystories-train.jsonl
DEV_CONFIG := configs/tiny_train.json
PROD_CONFIG := configs/deepseek_v3.json
DEV_CHECKPOINT := checkpoints/deepseek-tiny.pt
PROD_CHECKPOINT := checkpoints/deepseek-prod.pt
DEV_LOGDIR := runs/tensorboard/dev
PROD_LOGDIR := runs/tensorboard/prod
INFERENCE_CONFIG ?= $(DEV_CONFIG)
INFERENCE_PROMPT ?= Once upon a time
INFERENCE_TOKENS ?= 50
INFERENCE_TEMPERATURE ?= 0.8
INFERENCE_TOP_K ?= 50

# Development defaults are intentionally small enough for iteration on a laptop.
DEV_STORIES ?= 1000
DEV_STEPS ?= 1000
DEV_BATCH_SIZE ?= 10
DEV_BLOCK_SIZE ?= 128

# Production defaults describe a serious long-running pre-training job. They
# remain overridable because the correct values depend on available hardware.
PROD_STEPS ?= 100000
PROD_WARMUP_STEPS ?= 2000
PROD_BATCH_SIZE ?= 1
PROD_BLOCK_SIZE ?= 2048
PROD_LEARNING_RATE ?= 3e-4
PROD_DEVICE ?= cuda

.PHONY: help setup prepare-data \
	train-model-dev train-model-prod inference tensorboard test clean

help:
	@echo "DeepSeek from scratch commands"
	@echo ""
	@echo "Workflow:"
	@echo "  make prepare-data                  Prepare the full TinyStories training dataset"
	@echo "  make train-model-dev               Train the local development model"
	@echo "  make train-model-prod              Train the production model on a GPU"
	@echo "  make inference                     Generate text from a saved checkpoint"
	@echo ""
	@echo "Utilities:"
	@echo "  make setup                         Create the virtual environment and install dependencies"
	@echo "  make test                          Run the test suite"
	@echo "  make tensorboard                   Start TensorBoard for all runs"
	@echo "  make clean                         Remove generated checkpoints, runs, and prepared data"

# Install dependencies only when the environment is missing or requirements.txt
# has changed. This prevents repeated `make` calls from reinstalling packages.
$(DEPENDENCY_STAMP): requirements.txt
	@test -x "$(PYTHON)" || python3 -m venv .venv
	$(PIP) install -r requirements.txt
	@touch "$(DEPENDENCY_STAMP)"

# All operational targets depend on this inexpensive, incremental setup check.
setup: $(DEPENDENCY_STAMP)

# Prepare a small dataset automatically for local development training.
$(DEV_DATA): | setup
	$(PYTHON) -m src.data.prepare_tinystories \
		--max-stories $(DEV_STORIES) \
		--output $(DEV_DATA) \
		--cache-dir data/cache

# Prepare the full training split separately or as a production-training prerequisite.
prepare-data: $(PROD_DATA)

# Keep production data generation incremental so a pre-prepared/copied dataset
# can be used directly by setting PROD_DATA to its path.
$(PROD_DATA): | setup
	$(PYTHON) -m src.data.prepare_tinystories \
		--split train \
		--output $(PROD_DATA) \
		--cache-dir data/cache

# Train the small model locally using the development dataset.
train-model-dev: $(DEV_DATA)
	$(PYTHON) -m src.train \
		--data $(DEV_DATA) \
		--config $(DEV_CONFIG) \
		--steps $(DEV_STEPS) \
		--warmup-steps 0 \
		--batch-size $(DEV_BATCH_SIZE) \
		--block-size $(DEV_BLOCK_SIZE) \
		--device auto \
		--tensorboard-logdir $(DEV_LOGDIR) \
		--checkpoint $(DEV_CHECKPOINT)

# Train with the full dataset and reference model configuration. This requires
# suitable GPU hardware and memory for the selected model dimensions.
train-model-prod: $(PROD_DATA)
	$(PYTHON) -m src.train \
		--data $(PROD_DATA) \
		--config $(PROD_CONFIG) \
		--steps $(PROD_STEPS) \
		--warmup-steps $(PROD_WARMUP_STEPS) \
		--batch-size $(PROD_BATCH_SIZE) \
		--block-size $(PROD_BLOCK_SIZE) \
		--learning-rate $(PROD_LEARNING_RATE) \
		--log-interval 10 \
		--device $(PROD_DEVICE) \
		--tensorboard-logdir $(PROD_LOGDIR) \
		--checkpoint $(PROD_CHECKPOINT)

# Infer from the newest checkpoint unless CHECKPOINT is supplied explicitly.
inference: setup
	@test -n "$(CHECKPOINT)" || find checkpoints -maxdepth 1 -type f -name '*.pt' | grep -q . || (echo "No checkpoint found. Run make train-model-dev first." && exit 1)
	$(PYTHON) inference.py \
		--config $(INFERENCE_CONFIG) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		--prompt "$(INFERENCE_PROMPT)" \
		--max-new-tokens $(INFERENCE_TOKENS) \
		--temperature $(INFERENCE_TEMPERATURE) \
		--top-k $(INFERENCE_TOP_K) \
		--device auto

# TensorBoard reads all event files so dev and prod runs can be compared.
tensorboard: setup
	$(TENSORBOARD) --logdir runs/tensorboard

# Run all repository tests through the configured environment.
test: setup
	$(PYTHON) -m pytest

# Remove generated local outputs while leaving source, docs, and configuration intact.
clean:
	rm -rf checkpoints runs data/processed
