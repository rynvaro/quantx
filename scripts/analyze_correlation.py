"""
Feature correlation analysis.

Reads a features parquet and optionally a feature_importance CSV,
then prints all pairs with |Pearson r| > threshold, sorted by correlation,
and recommends which to drop (keep the one with higher importance).

Usage
-----
    python scripts/analyze_correlation.py \\
        --parquet data/processed/v1.0.3/features_thr005_h30_long.parquet \\
        --importance models/v1.0.3/long/feature_importance.csv \\
        --threshold 0.85

    # Without importance (recommendation: drop feat_b alphabetically)
    python scripts/analyze_correlation.py \\
        --parquet data/processed/v1.0.3/features_thr005_h30_long.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Label columns to exclude from correlation analysis
_LABEL_COLS = {"y_class", "y_max", "y_min", "price_open", "price_high",
               "price_low", "price_close", "open_time"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find highly-correlated feature pairs in a parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        required=True,
        metavar="PATH",
        help="Path to features parquet file.",
    )
    parser.add_argument(
        "--importance",
        default=None,
        metavar="PATH",
        help="Path to feature_importance.csv (gain-based). Used for drop recommendations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        metavar="FLOAT",
        help="Absolute Pearson correlation threshold (default: 0.85).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Optional CSV path to save results.",
    )
    return parser.parse_args()


def run(parquet_path: str, importance_path: str | None,
        threshold: float, output_path: str | None) -> pd.DataFrame:
    print(f"Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)

    # Keep only numeric feature columns
    feature_cols = [
        c for c in df.columns
        if c not in _LABEL_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"Feature columns: {len(feature_cols)}")

    feat_df = df[feature_cols].astype(float)

    # Load importance if provided
    importance: dict[str, float] = {}
    if importance_path:
        imp_df = pd.read_csv(importance_path)
        # Handle either column layout: (feature, importance) or indexed
        if "feature" in imp_df.columns and "importance" in imp_df.columns:
            importance = dict(zip(imp_df["feature"], imp_df["importance"]))
        elif imp_df.shape[1] == 2:
            importance = dict(zip(imp_df.iloc[:, 0], imp_df.iloc[:, 1]))
        print(f"Importance entries loaded: {len(importance)}")

    print(f"Computing correlation matrix (n={len(feature_cols)}) …")
    corr = feat_df.corr(method="pearson")

    # Extract upper triangle pairs above threshold
    rows = []
    cols_list = corr.columns.tolist()
    n = len(cols_list)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                fa, fb = cols_list[i], cols_list[j]
                imp_a = importance.get(fa, 0.0)
                imp_b = importance.get(fb, 0.0)
                if imp_a >= imp_b:
                    keep, drop = fa, fb
                else:
                    keep, drop = fb, fa
                rows.append({
                    "feat_a": fa,
                    "feat_b": fb,
                    "corr": round(r, 4),
                    "importance_a": round(imp_a, 1),
                    "importance_b": round(imp_b, 1),
                    "recommendation": f"drop {drop} (keep {keep})",
                })

    result = pd.DataFrame(rows).sort_values("corr", key=abs, ascending=False)

    print(f"\nHigh-correlation pairs (|r| ≥ {threshold}): {len(result)}")
    if not result.empty:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", 60)
        print(result.to_string(index=False))

        if output_path:
            result.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")

    return result


def main() -> None:
    args = _parse_args()
    run(
        parquet_path=args.parquet,
        importance_path=args.importance,
        threshold=args.threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
