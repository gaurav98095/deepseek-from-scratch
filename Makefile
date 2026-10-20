# Use the repository virtual environment for every Python operation.
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

.PHONY: help install setup step \
	step-1-dev step-1-prod step-2-dev step-2-prod \
	dev prod train-dev train-prod infer tensorboard web test clean

help:
	@echo "DeepSeek from scratch commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup                         Create the virtual environment and install dependencies"
	@echo ""
	@echo "Ordered steps:"
	@echo "  make step STEP=1 ENV=dev           Prepare a small smoke-test dataset"
	@echo "  make step STEP=1 ENV=prod          Prepare the full TinyStories dataset"
	@echo "  make step STEP=2 ENV=dev           Run the local model/training pipeline"
	@echo "  make step STEP=2 ENV=prod          Run the full long-training pipeline"
	@echo "  make step-1-dev                    Alias for step 1 development"
	@echo "  make step-1-prod                   Alias for step 1 production"
	@echo "  make step-2-dev                    Alias for step 2 development"
	@echo "  make step-2-prod                   Alias for step 2 production"
	@echo ""
	@echo "Operations:"
	@echo "  make dev                           Prepare and train the local configuration"
	@echo "  make prod                          Prepare and train the full configuration"
	@echo "  make infer PROMPT='...'            Generate from the newest checkpoint"
	@echo "  make tensorboard                   Start TensorBoard for all runs"
	@echo "  make web                           Start the TensorBoard web interface"
	@echo "  make test                          Run the test suite"

# Install dependencies only when the environment is missing or requirements.txt
# has changed. This prevents repeated `make` calls from reinstalling packages.
$(DEPENDENCY_STAMP): requirements.txt
	@test -x "$(PYTHON)" || python3 -m venv .venv
	$(PIP) install -r requirements.txt
	@touch "$(DEPENDENCY_STAMP)"

# All operational targets depend on this inexpensive, incremental setup check.
setup: $(DEPENDENCY_STAMP)

# Explicit `make install` remains available when a user wants to reinstall.
install: setup
	$(PIP) install -r requirements.txt
	@touch "$(DEPENDENCY_STAMP)"

# Dispatch a numbered step to the requested dev/prod implementation.
# Usage: make step STEP=1 ENV=dev (GNU Make does not support `make step 1`).
step:
	@test -n "$(STEP)" || (echo "Usage: make step STEP=1 ENV=dev" && exit 1)
	@test -n "$(ENV)" || (echo "Usage: make step STEP=1 ENV=dev" && exit 1)
	@test "$(STEP)" = "1" || test "$(STEP)" = "2" || (echo "STEP must be 1 or 2" && exit 1)
	@test "$(ENV)" = "dev" || test "$(ENV)" = "prod" || (echo "ENV must be dev or prod" && exit 1)
	$(MAKE) step-$(STEP)-$(ENV)

# Step 1 development uses a small story limit so it is safe to repeat locally.
# The output file is a real Make dependency, so an existing dataset is reused.
step-1-dev: $(DEV_DATA)

$(DEV_DATA): | setup
	$(PYTHON) -m src.data.prepare_tinystories \
		--max-stories $(DEV_STORIES) \
		--output $(DEV_DATA) \
		--cache-dir data/cache

# Step 1 production streams the complete training split to the prepared JSONL file.
step-1-prod: $(PROD_DATA)

$(PROD_DATA): | setup
	$(PYTHON) -m src.data.prepare_tinystories \
		--split train \
		--output $(PROD_DATA) \
		--cache-dir data/cache

# Step 2 development runs the small model, short schedule, and local TensorBoard run.
step-2-dev: step-1-dev
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

# Step 2 production uses the full data path, reference model configuration, a
# long schedule, warmup, and a context length suitable for serious training.
# It requires hardware and memory appropriate for the selected configuration.
step-2-prod: step-1-prod
	$(PYTHON) -m src.train \
		--data $(PROD_DATA) \
		--config $(PROD_CONFIG) \
		--steps $(PROD_STEPS) \
		--warmup-steps $(PROD_WARMUP_STEPS) \
		--batch-size $(PROD_BATCH_SIZE) \
		--block-size $(PROD_BLOCK_SIZE) \
		--learning-rate $(PROD_LEARNING_RATE) \
		--log-interval 10 \
		--device auto \
		--tensorboard-logdir $(PROD_LOGDIR) \
		--checkpoint $(PROD_CHECKPOINT)

# Friendly aliases for completing the current development or production path.
dev: step-2-dev
prod: step-2-prod
train-dev: step-2-dev
train-prod: step-2-prod

# Infer from the newest checkpoint unless CHECKPOINT is supplied explicitly.
infer: setup
	@test -n "$(CHECKPOINT)" || find checkpoints -maxdepth 1 -type f -name '*.pt' | grep -q . || (echo "No checkpoint found. Run make step-2-dev first." && exit 1)
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

# There is no separate web application yet; TensorBoard is the current web UI.
web: tensorboard

# Run all repository tests through the configured environment.
test: setup
	$(PYTHON) -m pytest

# Remove generated local outputs while leaving source, docs, and configuration intact.
clean:
	rm -rf checkpoints runs data/processed
