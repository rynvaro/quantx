#!/usr/bin/env python
"""
CLI entry point for the backtest engine.

Usage
-----
    # Backtest on the full val zone (2024-04 to 2026-03, walk-forward OOS data)
    python scripts/run_backtest.py \\
        --input  data/processed/features_thr005_h30_long.parquet \\
        --models models/ \\
        --start  2024-04-01 \\
        --threshold 0.60

    # Try different TP/SL settings
    python scripts/run_backtest.py \\
        --input data/processed/features_thr005_h30_long.parquet \\
        --models models/ \\
        --start 2024-04-01 \\
        --threshold 0.60 \\
        --tp 0.010 --sl 0.007
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine, BacktestConfig, print_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backtest on features parquet using trained models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",     type=Path, default=Path("data/processed/features_thr005_h30_long.parquet"))
    parser.add_argument("--models",    type=Path, default=Path("models/"))
    parser.add_argument("--start",     type=str,  default=None,  metavar="YYYY-MM-DD")
    parser.add_argument("--end",       type=str,  default=None,  metavar="YYYY-MM-DD")
    parser.add_argument("--threshold", type=float, default=0.60, metavar="FLOAT",
                        help="Signal probability threshold for entry.")
    parser.add_argument("--tp",        type=float, default=0.008, metavar="FLOAT",
                        help="Take-profit as ETH price move fraction (e.g. 0.008 = 0.8%%).")
    parser.add_argument("--sl",        type=float, default=0.008, metavar="FLOAT",
                        help="Stop-loss as ETH price move fraction (e.g. 0.008 = 0.8%%).")
    parser.add_argument("--equity",    type=float, default=312.0, metavar="FLOAT",
                        help="Initial equity in USDT.")
    parser.add_argument("--margin-pct", type=float, default=0.25, metavar="FLOAT",
                        help="Fraction of equity used as margin per trade.")
    parser.add_argument("--leverage",  type=float, default=75.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.models.exists():
        print(f"ERROR: models directory not found: {args.models}", file=sys.stderr)
        sys.exit(1)

    cfg = BacktestConfig(
        signal_threshold=args.threshold,
        initial_equity=args.equity,
        margin_pct=args.margin_pct,
        leverage=args.leverage,
        take_profit=args.tp,
        stop_loss=args.sl,
    )

    print(f"Loading features …")
    df = pd.read_parquet(args.input)

    engine = BacktestEngine(cfg)
    trades, equity_curve = engine.run(
        df=df,
        model_dir=args.models,
        start_date=args.start,
        end_date=args.end,
    )

    print_stats(trades, equity_curve, cfg.initial_equity)

    # Save equity curve
    out_path = args.models / "equity_curve.csv"
    equity_curve.to_csv(out_path)
    print(f"Equity curve saved to {out_path}")


if __name__ == "__main__":
    main()
