# QuantX

Perpetual futures signal system based on XGBoost walk-forward training.  
Supports both long and short directions independently, with an event-driven backtest engine and a live 5-minute bar inference daemon.

---

## Architecture

```
Raw klines (5m)
      │
      ▼
build_features.py          → features_thr005_h30_{long|short}.parquet
      │
      ▼
train_model.py             → models/{long|short}/
  Walk-forward 6 folds
  3-head XGBoost (classifier + TP/SL regressors)
  Platt calibration (LogisticRegression on OOF scores)
      │
      ├──▶ run_backtest.py  → OOS equity curve, win rate, PF, max DD
      │
      └──▶ run_signal.py    → live 5m bar inference (dual long+short)
```

---

## Model Design

| Item | Detail |
|---|---|
| Algorithm | XGBoost 3-head: `XGBClassifier` (y_class) + `XGBRegressor` × 2 (y_max, y_min) |
| Calibration | Platt scaling — `LogisticRegression` fit on OOF classifier scores |
| Walk-forward | 6 folds, 4-month validation window, 5-day gap |
| Features | ~131 technical indicators; OHLC excluded (non-stationary + not available at inference) |
| Horizon | 6 bars (30 min) |
| Threshold | 0.60 calibrated probability |

---

## Backtest Results (OOS 2024-04-01 → present)

| Direction | TP | SL | Win rate | Profit factor | Max DD | Avg PnL/trade |
|---|---|---|---|---|---|---|
| Long | 0.8% | 0.8% | 78.4% | 3.69 | -20.5% | 17.09 U |
| Short | 0.8% | 1.0% | 79.7% | 4.59 | -7.4% | 19.19 U |

Capital assumptions: 312 U total, 78 U margin/trade (25%), 75× leverage, 0.04% taker fee, 0.03% slippage.

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
  train_model.py         walk-forward training → models/
  run_backtest.py        OOS backtest with TP/SL simulation
  run_signal.py          live signal monitor (5m bar loop)

src/
  features/              indicators, labels, loader, timeframe merge
  train/                 fold splitting, model training, calibration, evaluation
  backtest/              event-driven backtest engine
  live/                  kline fetcher, feature compute, model inference

models/
  long/                  trained long model artifacts
  short/                 trained short model artifacts

docs/
  pipeline_guide.md      full step-by-step pipeline guide
  backtest_params.md     selected TP/SL params with grid results
  training_design.md     model design decisions

logs/                    signal daemon output (gitignored)
Makefile                 start / stop / status / logs / restart
```

---

## Makefile Reference

| Target | Action |
|---|---|
| `make start` | Run `run_signal.py` in foreground with countdown |
| `make start-bg` | Start as background daemon; PID → `logs/signal.pid` |
| `make stop` | Send SIGTERM, wait up to 5 s for clean exit |
| `make status` | Print PID status + last 5 log lines |
| `make logs` | `tail -f logs/signal.log` |
| `make restart` | `stop` + `start-bg` |

---

## Setup (new machine)

```bash
# 1. clone
git clone <repo-url> quantx && cd quantx

# 2. create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt
```

Then proceed from **Quick Start Step 1** above.

## Requirements

- Python 3.12
- See [requirements.txt](requirements.txt) for pinned versions
