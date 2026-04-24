"""
CLI entry point for feature engineering.

Usage examples:
  # Full run (2021-01 to present)
  python scripts/build_features.py

  # Custom date range
  python scripts/build_features.py --start 2022-01-01 --end 2024-12-31

  # Different threshold scan
  python scripts/build_features.py --threshold 0.012

  # Include slow P2 features (Hurst, sample entropy)
  python scripts/build_features.py --include-p2

  # Custom output path
  python scripts/build_features.py --output data/processed/features_v2.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.pipeline import run, DATA_ROOT, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ETHUSDT feature matrix from raw Binance kline data."
    )
    parser.add_argument(
        "--data-root",
        default=str(DATA_ROOT),
        help=f"Path to raw data root (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output Parquet path (default: {OUTPUT_DIR}/features.parquet)",
    )
    parser.add_argument(
        "--start",
        metavar="YYYY-MM-DD",
        default=None,
        help="Start date filter (inclusive)",
    )
    parser.add_argument(
        "--end",
        metavar="YYYY-MM-DD",
        default=None,
        help="End date filter (inclusive)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.010,
        metavar="FLOAT",
        help="y_class entry threshold, e.g. 0.010 = 1%% (default: 0.010)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        metavar="INT",
        help="Number of future 1m bars for label window (default: 30 = 30 min)",
    )
    parser.add_argument(
        "--scan-threshold",
        action="store_true",
        help=(
            "Scan multiple thresholds [0.005, 0.008, 0.010, 0.012, 0.015] "
            "and print positive-class rates before the full run."
        ),
    )
    parser.add_argument(
        "--include-p2",
        action="store_true",
        help="Include slow P2 features (Hurst exponent, sample entropy). Much slower.",
    )
    parser.add_argument(
        "--direction",
        choices=["long", "short"],
        default="long",
        help="Trade direction: 'long' (default) or 'short'. Controls directional features and labels.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.output:
        output_path = Path(args.output)
    else:
        thr_str = f"{int(round(args.threshold * 1000)):03d}"
        output_path = OUTPUT_DIR / "v1" / f"features_thr{thr_str}_h{args.horizon}_{args.direction}.parquet"

    if args.scan_threshold:
        _scan_thresholds(data_root, args.start, args.end, args.horizon)

    run(
        data_root=data_root,
        output_path=output_path,
        entry_threshold=args.threshold,
        horizon=args.horizon,
        start_date=args.start,
        end_date=args.end,
        include_p2=args.include_p2,
        direction=args.direction,
    )


def _scan_thresholds(data_root: Path, start_date, end_date, horizon: int = 30) -> None:
    """Quick scan to find which threshold gives 20-30% positive-class rate."""
    import logging
    from src.features.loader import load_klines
    from src.features.labels import compute_labels

    log = logging.getLogger(__name__)
    log.info("── Threshold scan ──────────────────────────────────────")

    df_5m = load_klines("5m", data_root)
    df_1m = load_klines("1m", data_root)

    if start_date:
        df_5m = df_5m[df_5m.index >= __import__("pandas").Timestamp(start_date, tz="UTC")]
    if end_date:
        df_5m = df_5m[df_5m.index <= __import__("pandas").Timestamp(end_date, tz="UTC")]

    # Weekend filter for the scan
    df_5m_wd = df_5m[df_5m.index.dayofweek < 5]

    for thr in [0.005, 0.008, 0.010, 0.012, 0.015]:
        labels = compute_labels(df_5m_wd, df_1m, entry_threshold=thr, horizon=horizon)
        rate = labels["y_class"].mean()
        log.info("  threshold=%.3f  →  positive-class rate = %.1f%%", thr, 100 * rate)

    log.info("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
