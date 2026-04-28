PYTHON   := $(shell conda run -n base which python 2>/dev/null || echo python3)
SCRIPT   := scripts/run_signal.py
LOG_DIR  := logs

# ── Version control ───────────────────────────────────────────────────────────
# Override on the command line: make start-bg VERSION=v1.0.1
VERSION  ?= v1
LOG_FILE  = $(LOG_DIR)/signal_$(VERSION).log
PID_FILE  = $(LOG_DIR)/signal_$(VERSION).pid
DATA_DIR  = data/processed/$(VERSION)
MODEL_DIR = models/$(VERSION)

.PHONY: init start start-bg stop status logs restart \
        build-features train backtest feature-importance \
        _logs_dir

## init    — verify conda environment and install/update dependencies
init: | _logs_dir
	@if [ -z "$$CONDA_DEFAULT_ENV" ]; then \
		echo "❌ Error: Not in a conda environment"; \
		echo ""; \
		echo "This project requires conda. Please:"; \
		echo "  1. Install conda: https://docs.conda.io/en/latest/miniconda.html"; \
		echo "  2. Create environment: conda env create -f environment.yml"; \
		echo "  3. Activate environment: conda activate quantx"; \
		exit 1; \
	else \
		echo "✓ Conda environment: $$CONDA_DEFAULT_ENV"; \
		$(PYTHON) -c "import sys; v=sys.version_info; \
		    print(f'✓ Python {v.major}.{v.minor}.{v.micro}')"; \
		echo ""; \
		echo "Installing/updating dependencies..."; \
		pip install -r requirements.txt; \
		echo ""; \
		echo "✓ Environment ready"; \
	fi

## start   — run in foreground (Ctrl-C to stop)
start:
	$(PYTHON) $(SCRIPT) --long-model-dir $(MODEL_DIR)/long --short-model-dir $(MODEL_DIR)/short

## start-bg — run in background, log to logs/signal_$(VERSION).log
start-bg: | _logs_dir
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "⚠️  Already running VERSION=$(VERSION) (PID $$(cat $(PID_FILE)))"; \
		echo "   Log: $(LOG_FILE)"; \
		echo "   Use 'make stop VERSION=$(VERSION)' to stop it first."; \
	else \
		nohup $(PYTHON) -u $(SCRIPT) \
			--long-model-dir $(MODEL_DIR)/long \
			--short-model-dir $(MODEL_DIR)/short \
			>> $(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		echo "Started VERSION=$(VERSION) (PID $$(cat $(PID_FILE))) — log: $(LOG_FILE)"; \
	fi

## stop    — send SIGTERM, wait for clean shutdown
stop:
	@if [ ! -f $(PID_FILE) ]; then \
		echo "No PID file found for VERSION=$(VERSION) ($(PID_FILE))"; \
	else \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID; \
			echo "Sent SIGTERM to PID $$PID — waiting …"; \
			for i in 1 2 3 4 5; do \
				sleep 1; \
				kill -0 $$PID 2>/dev/null || break; \
			done; \
			kill -0 $$PID 2>/dev/null && echo "Still running after 5s" || echo "Stopped"; \
		else \
			echo "Process $$PID not running — cleaning up"; \
		fi; \
		rm -f $(PID_FILE); \
	fi

## status  — show whether process is running + last 5 log lines
status:
	@echo "VERSION=$(VERSION)"
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Running (PID $$PID)"; \
		else \
			echo "Not running (stale PID $$PID)"; \
		fi; \
	else \
		echo "Not running"; \
	fi
	@echo ""
	@if [ -f $(LOG_FILE) ]; then \
		echo "Last 5 log lines:"; \
		tail -5 $(LOG_FILE); \
	fi

## logs    — tail -f the log file
logs:
	@echo "[VERSION=$(VERSION)] $(LOG_FILE)"
	@tail -f $(LOG_FILE)

## restart — stop then start in background
restart: stop start-bg

## build-features — build feature parquets for both directions (VERSION controls output path)
##   make build-features VERSION=v1.0.1
build-features: | _logs_dir
	@echo "Building features → $(DATA_DIR)/ (long)"
	$(PYTHON) scripts/build_features.py --output $(DATA_DIR)/features_thr005_h30_long.parquet --direction long
	@echo "Building features → $(DATA_DIR)/ (short)"
	$(PYTHON) scripts/build_features.py --output $(DATA_DIR)/features_thr005_h30_short.parquet --direction short

## train — train long + short models (VERSION controls input/output paths)
##   make train VERSION=v1.0.1
train:
	@echo "Training LONG model → $(MODEL_DIR)/long"
	$(PYTHON) scripts/train_model.py --direction long \
		--input $(DATA_DIR)/features_thr005_h30_long.parquet \
		--output $(MODEL_DIR)/long
	@echo "Training SHORT model → $(MODEL_DIR)/short"
	$(PYTHON) scripts/train_model.py --direction short \
		--input $(DATA_DIR)/features_thr005_h30_short.parquet \
		--output $(MODEL_DIR)/short

## backtest — run backtest for both directions
##   make backtest VERSION=v1.0.1 [START=2024-04-01]
backtest:
	@echo "Backtesting LONG — $(MODEL_DIR)/long"
	$(PYTHON) scripts/run_backtest.py --direction long \
		--input $(DATA_DIR)/features_thr005_h30_long.parquet \
		--models $(MODEL_DIR)/long \
		$(if $(START),--start $(START),)
	@echo "Backtesting SHORT — $(MODEL_DIR)/short"
	$(PYTHON) scripts/run_backtest.py --direction short \
		--input $(DATA_DIR)/features_thr005_h30_short.parquet \
		--models $(MODEL_DIR)/short \
		$(if $(START),--start $(START),)

## feature-importance — analyze feature importance for trained models
##   make feature-importance VERSION=v1.0.1
feature-importance:
	@echo "Analyzing feature importance — $(MODEL_DIR)"
	$(PYTHON) scripts/analyze_importance.py --models-dir $(MODEL_DIR)

_logs_dir:
	mkdir -p $(LOG_DIR)
