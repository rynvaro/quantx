"""
Multi-horizon feature engineering pipeline.

Builds features for multiple prediction horizons (M15/M30/H1/H2) with
time-scale matched features according to design principles:

- M15: 60% from 1m/3m, 30% from 5m, 10% from 15m
- M30: 60% from 3m/5m, 30% from 15m, 10% from 1h
- H1: 60% from 5m/15m, 30% from 1h, 10% from 4h
- H2: 60% from 15m/1h, 30% from 4h, 10% from 1h

Main entry point: build_multi_horizon_features()
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .loader import load_klines, load_funding_rate
from .timeframe import compute_features
from .advanced import compute_advanced_features
from .multi_labels import compute_default_labels

log = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


# ── Timeframe configurations ───────────────────────────────────────────────

# Which timeframes to load and compute features for
ALL_TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "4h"]

# Feature selection for each prediction horizon
HORIZON_FEATURE_CONFIG = {
    "M15": {
        "core": ["1m", "3m"],      # 60%
        "auxiliary": ["5m"],        # 30%
        "background": ["15m"],      # 10%
    },
    "M30": {
        "core": ["3m", "5m"],
        "auxiliary": ["15m"],
        "background": ["1h"],
    },
    "H1": {
        "core": ["5m", "15m"],
        "auxiliary": ["1h"],
        "background": ["4h"],
    },
    "H2": {
        "core": ["15m", "1h"],
        "auxiliary": ["4h"],
        "background": ["1h"],  # Note: design doc has 1h in background for H2
    },
}


# ── Alignment helpers ──────────────────────────────────────────────────────

def align_to_1m(feat_df: pd.DataFrame, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align features from any timeframe to 1m base index using forward-fill.

    For sub-1m timeframes: not supported (we don't have sub-1m data).
    For ≥1m timeframes: standard forward-fill.
    """
    return feat_df.reindex(base_index, method="ffill")


# ── Feature loading and computation ────────────────────────────────────────

def load_and_compute_features(
    timeframe: str,
    data_root: Path,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    Load raw kline data and compute all features for a single timeframe.

    Parameters
    ----------
    timeframe : '1m', '3m', '5m', etc.
    data_root : root directory of raw data
    include_advanced : whether to compute advanced features (derivatives, etc.)

    Returns
    -------
    DataFrame with all features for this timeframe (columns suffixed with _<tf>)
    """
    log.info(f"Loading {timeframe} klines...")
    df = load_klines(timeframe, data_root)

    log.info(f"Computing base features for {timeframe}...")
    df_feat = compute_features(df, timeframe, include_p2=False)

    if include_advanced:
        log.info(f"Computing advanced features for {timeframe}...")
        df_feat = compute_advanced_features(
            df_feat,
            timeframe,
            include_derivatives=True,
            include_volume=True,
            include_volatility=True,
        )

    return df_feat


# ── Feature selection by horizon ───────────────────────────────────────────

def select_features_for_horizon(
    all_features: Dict[str, pd.DataFrame],
    horizon: str,
) -> pd.DataFrame:
    """
    Select and combine features for a specific prediction horizon.

    Parameters
    ----------
    all_features : dict mapping timeframe -> feature DataFrame
    horizon : "M15", "M30", "H1", or "H2"

    Returns
    -------
    Combined DataFrame with selected features aligned to 1m base
    """
    config = HORIZON_FEATURE_CONFIG[horizon]
    base_index = all_features["1m"].index

    selected_dfs = []

    # Add core features (60%)
    for tf in config["core"]:
        if tf in all_features:
            df_aligned = align_to_1m(all_features[tf], base_index)
            selected_dfs.append(df_aligned)

    # Add auxiliary features (30%)
    for tf in config["auxiliary"]:
        if tf in all_features:
            df_aligned = align_to_1m(all_features[tf], base_index)
            selected_dfs.append(df_aligned)

    # Add background features (10%)
    for tf in config["background"]:
        if tf in all_features:
            df_aligned = align_to_1m(all_features[tf], base_index)
            selected_dfs.append(df_aligned)

    # Concatenate all selected features
    result = pd.concat(selected_dfs, axis=1)

    # Remove duplicate columns (keep first occurrence)
    result = result.loc[:, ~result.columns.duplicated()]

    return result


# ── Main pipeline ──────────────────────────────────────────────────────────

def build_multi_horizon_features(
    data_root: Path = DATA_ROOT,
    output_dir: Path = OUTPUT_DIR,
    direction: str = "long",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build features and labels for all prediction horizons.

    This is the main entry point for multi-horizon feature engineering.

    Parameters
    ----------
    data_root : root directory of raw data
    output_dir : where to save processed features (one file per horizon)
    direction : "long" or "short"
    start_date : optional ISO date string to slice data
    end_date : optional ISO date string to slice data

    Returns
    -------
    Dict mapping horizon name -> feature DataFrame with labels

    Output files
    ------------
    Saves one Parquet file per horizon:
        - data/processed/v2/features_M15_{direction}.parquet
        - data/processed/v2/features_M30_{direction}.parquet
        - data/processed/v2/features_H1_{direction}.parquet
        - data/processed/v2/features_H2_{direction}.parquet

    Each file contains:
        - All selected features for that horizon
        - Labels for all 3 thresholds (y_class_<horizon>_<thr>)
        - Regression targets (y_max_<horizon>, y_min_<horizon>)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and compute features for all timeframes ───────────────────
    log.info("=" * 70)
    log.info("STEP 1: Loading and computing features for all timeframes")
    log.info("=" * 70)

    all_features: Dict[str, pd.DataFrame] = {}
    for tf in ALL_TIMEFRAMES:
        all_features[tf] = load_and_compute_features(tf, data_root)

    # ── 2. Load 1m OHLC for label computation ─────────────────────────────
    log.info("=" * 70)
    log.info("STEP 2: Loading 1m OHLC for label computation")
    log.info("=" * 70)

    df_1m_ohlc = load_klines("1m", data_root)[["open", "high", "low", "close"]]

    # Optional date slicing
    if start_date:
        df_1m_ohlc = df_1m_ohlc[df_1m_ohlc.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df_1m_ohlc = df_1m_ohlc[df_1m_ohlc.index <= pd.Timestamp(end_date, tz="UTC")]

    # ── 3. Compute labels for all horizons ────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 3: Computing labels for all horizons")
    log.info("=" * 70)

    all_labels = compute_default_labels(df_1m_ohlc, direction=direction)

    # ── 4. Build feature sets for each horizon ────────────────────────────
    log.info("=" * 70)
    log.info("STEP 4: Building feature sets for each horizon")
    log.info("=" * 70)

    results = {}

    for horizon in ["M15", "M30", "H1", "H2"]:
        log.info(f"\n--- Building features for {horizon} ---")

        # Select features for this horizon
        features = select_features_for_horizon(all_features, horizon)

        # Align to 1m OHLC index (in case of date slicing)
        features = features.reindex(df_1m_ohlc.index)

        # Extract labels for this horizon
        label_cols = [col for col in all_labels.columns if col.startswith(f"y_") and horizon in col]
        labels = all_labels[label_cols].reindex(df_1m_ohlc.index)

        # Combine features and labels
        combined = pd.concat([features, labels], axis=1)

        # Drop rows with NaN labels (last <horizon> minutes)
        combined = combined.dropna(subset=label_cols)

        log.info(f"  Features: {features.shape[1]} columns")
        log.info(f"  Labels: {len(label_cols)} columns")
        log.info(f"  Total samples: {len(combined):,}")

        # Save to file
        output_path = output_dir / f"features_{horizon}_{direction}.parquet"
        combined.to_parquet(output_path)
        log.info(f"  Saved to {output_path}")

        results[horizon] = combined

    # ── 5. Summary ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    for horizon, df in results.items():
        log.info(f"{horizon}: {len(df):,} samples, {df.shape[1]} features")

    return results
