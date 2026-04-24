# QuantX (v1)

Perpetual futures signal system based on XGBoost walk-forward training.  
Supports both long and short directions independently, with an event-driven backtest engine and a live 5-minute bar inference daemon.

**Note:** This is the v1 branch with baseline features (131 indicators). See other branches for experimental versions.

---

## Architecture

```
Raw klines (5m)
      │
      ▼
build_features.py          → data/processed/v1/features_thr005_h30_{long|short}.parquet
      │
      ▼
train_model.py             → models/v1/{long|short}/
  Walk-forward 6 folds
  3-head XGBoost (classifier + TP/SL regressors)
  Platt calibration (LogisticRegression on OOF scores)
      │
      ├──▶ run_backtest.py  → OOS equity curve, win rate, PF, max DD
      │
      ├──▶ analyze_importance.py → feature importance analysis
      │
      └──▶ run_signal.py    → live 5m bar inference (dual long+short)
                              with pattern matching (6 known success patterns)
```

---

## Model Design

| Item | Detail |
|---|---|
| Algorithm | XGBoost 3-head: `XGBClassifier` (y_class) + `XGBRegressor` × 2 (y_max, y_min) |
| Calibration | Platt scaling — `LogisticRegression` fit on OOF classifier scores |
| Walk-forward | 6 folds, 4-month validation window, 5-day gap |
| Features | 131 technical indicators; OHLC excluded (non-stationary + not available at inference) |
| Horizon | 30 bars (150 min / 2.5 hours) |
| Threshold | 0.60 calibrated probability |
| Training | y_max/y_min trained on all samples (fixed OOD issue) |

---

## Backtest Results (2025 full year, retrained model)

| Direction | TP | SL | Win rate | Profit factor | Max DD | Avg PnL/trade | Total return |
|---|---|---|---|---|---|---|---|
| Long | 0.8% | 0.8% | 79.1% | 3.72 | -6.3% | 18.08 U | +19813% |
| Short | 0.8% | 1.0% | 78.5% | 4.06 | -11.9% | 18.93 U | +20397% |

Capital assumptions: 312 U total, 78 U margin/trade (25%), 75× leverage, 0.04% taker fee, 0.03% slippage.

**Note:** SHORT model performs poorly in strong bull markets (2021 early). Use with caution or disable in trending up markets.

---

## Quick Start

### 1. Download raw klines

```bash
python scripts/download_data.py
```

### 2. Build features

```bash
python scripts/build_features.py --threshold 0.005 --horizon 30
```

### 3. Train models

```bash
python scripts/train_model.py --direction long
python scripts/train_model.py --direction short
```

### 4. Backtest

```bash
python scripts/run_backtest.py --direction long  --start 2024-04-01
python scripts/run_backtest.py --direction short --start 2024-04-01
```

### 5. Live signal

```bash
# Foreground (with countdown)
make start

# Background daemon (log → logs/signal.log)
make start-bg
make stop
make status
make logs
```

---

## Project Structure

```
scripts/
  download_data.py       download raw klines
  build_features.py      feature engineering → parquet
  train_model.py         walk-forward training → models/v1/
  run_backtest.py        OOS backtest with TP/SL simulation
  run_signal.py          live signal monitor (5m bar loop)
  analyze_importance.py  feature importance analysis

src/
  features/              indicators, labels, loader, timeframe merge
  train/                 fold splitting, model training, calibration, evaluation
  backtest/              event-driven backtest engine
  live/                  kline fetcher, feature compute, model inference
    pattern_matcher.py   pattern recognition (6 success patterns: S1-S4, L1-L2)

models/v1/
  long/                  trained long model artifacts
  short/                 trained short model artifacts

data/processed/v1/       feature parquet files (gitignored)

docs/
  pipeline_guide.md      full step-by-step pipeline guide
  backtest_params.md     selected TP/SL params with grid results
  training_design.md     model design decisions
  pattern_codes.md       pattern matching reference
  v1_live_analysis_*.md  live signal analysis reports

logs/                    signal daemon output (gitignored)
Makefile                 start / stop / status / logs / restart
```

---

## Makefile Reference

| Target | Action |
|---|---|
| `make init` | Create `.venv` and `pip install -r requirements.txt` |
| `make start` | Run `run_signal.py` in foreground with countdown |
| `make start-bg` | Start as background daemon; PID → `logs/signal-v1.pid` |
| `make stop` | Send SIGTERM, wait up to 5 s for clean exit |
| `make status` | Print PID status + last 5 log lines |
| `make logs` | `tail -f logs/signal-v1.log` |
| `make restart` | `stop` + `start-bg` |

---

## Setup (new machine)

```bash
git clone <repo-url> quantx && cd quantx
make init                              # creates .venv + pip install -r requirements.txt
source .venv/bin/activate
```

Then proceed from **Quick Start Step 1** above.

## Requirements

- Python 3.12
- See [requirements.txt](requirements.txt) for pinned versions
