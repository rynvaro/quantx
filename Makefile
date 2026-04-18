PYTHON   := python3
VENV     := .venv
SCRIPT   := scripts/run_signal.py
LOG_DIR  := logs
LOG_FILE := $(LOG_DIR)/signal.log
PID_FILE := $(LOG_DIR)/signal.pid

.PHONY: init start start-bg stop status logs restart _logs_dir

## init    — create .venv and install dependencies
init: | _logs_dir
	@$(PYTHON) -c "import sys; v=sys.version_info; ok=v>=(3,12); \
	    print(f'Python {v.major}.{v.minor}.{v.micro}  ✓' if ok else \
	          f'[WARN] Python {v.major}.{v.minor} detected — 3.12+ recommended')"
	@if [ ! -d $(VENV) ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "Created $(VENV)"; \
	fi
	$(VENV)/bin/pip install --upgrade pip -q
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install xgboost==3.2.0 --no-deps
	@echo ""
	@echo "Done. Activate with: source $(VENV)/bin/activate"

## start   — run in foreground (Ctrl-C to stop)
start:
	$(PYTHON) $(SCRIPT)

## start-bg — run in background, log to logs/signal.log
start-bg: | _logs_dir
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Already running (PID $$(cat $(PID_FILE)))"; \
	else \
		nohup $(PYTHON) -u $(SCRIPT) >> $(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		echo "Started (PID $$(cat $(PID_FILE))) — log: $(LOG_FILE)"; \
	fi

## stop    — send SIGTERM, wait for clean shutdown
stop:
	@if [ ! -f $(PID_FILE) ]; then \
		echo "No PID file found ($(PID_FILE))"; \
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
	tail -f $(LOG_FILE)

## restart — stop then start in background
restart: stop start-bg

_logs_dir:
	mkdir -p $(LOG_DIR)
