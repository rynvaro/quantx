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

    # Train short model (auto-infers input/output from --direction)
    python scripts/train_model.py --direction short
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
        "--direction",
        choices=["long", "short"],
        default="long",
        help="Trade direction. Controls default --input and --output paths.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to input features parquet (default: auto from --direction).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to save model artifacts (default: auto from --direction).",
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

    # Auto-infer input/output from direction if not explicitly provided
    _DATA_DIR = Path("data/processed/v1")
    _MODELS_DIR = Path("models/v1")
    input_path = args.input or _DATA_DIR / f"features_thr005_h30_{args.direction}.parquet"
    output_dir = args.output or _MODELS_DIR / args.direction

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    run(
        parquet_path=input_path,
        output_dir=output_dir,
        n_folds=args.n_folds,
        val_months=args.val_months,
        gap_days=args.gap_days,
    )


if __name__ == "__main__":
    main()
