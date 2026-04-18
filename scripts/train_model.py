#!/usr/bin/env python
"""
CLI entry point for the training pipeline.

Usage
-----
    python scripts/train_model.py \\
        --input data/processed/features_thr005_h30_long.parquet \\
        --output models/long/ \\
        --n-folds 6 \\
        --gap-days 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.pipeline import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3-head XGBoost models on preprocessed features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features_thr005_h30_long.parquet"),
        metavar="PATH",
        help="Path to input features parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/long"),
        metavar="DIR",
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=6,
        metavar="INT",
        help="Number of walk-forward folds.",
    )
    parser.add_argument(
        "--val-months",
        type=int,
        default=4,
        metavar="INT",
        help="Validation window size in months per fold.",
    )
    parser.add_argument(
        "--gap-days",
        type=int,
        default=5,
        metavar="INT",
        help="Gap days between train end and val start (leakage buffer).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(
        parquet_path=args.input,
        output_dir=args.output,
        n_folds=args.n_folds,
        val_months=args.val_months,
        gap_days=args.gap_days,
    )


if __name__ == "__main__":
    main()
