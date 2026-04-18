"""
Feature engineering pipeline.

Orchestrates:
  1. Load raw kline + funding-rate data
  2. Compute per-timeframe features
  3. Align all timeframes to 5m sample base
  4. Add cross-timeframe features (cpai, CTI)
  5. Compute labels (y_class, y_max, y_min)
  6. Add filter flags (is_weekend, is_black_swan)
  7. Save to Parquet

Main entry point: run()
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_klines, load_funding_rate
from .timeframe import compute_features
from .labels import compute_labels

log = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

KLINE_INTERVALS = ["1m", "5m", "15m", "1h", "4h"]


# ─── Alignment helpers ────────────────────────────────────────────────────────


def _align_to_5m(feat_df: pd.DataFrame, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Forward-fill a feature DataFrame onto the 5m sample index.

    For sub-5m timeframes (1m): use the bar closing just before each 5m bar
    closes, i.e., the 1m bar at base_index + 4min.
    For ≥5m timeframes: standard backward-fill (last bar before or at T5).
    """
    if feat_df.index.freq is not None and feat_df.index.freq <= pd.tseries.frequencies.to_offset("1min"):
        # 1m: look up bar at T5 + 4min
        lookup = base_index + pd.Timedelta("4min")
        aligned = feat_df.reindex(lookup, method="ffill")
        aligned.index = base_index
    else:
        aligned = feat_df.reindex(base_index, method="ffill")
    return aligned


def _align_coarse(feat_df: pd.DataFrame, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align coarser-than-5m features onto 5m base with forward-fill."""
    return feat_df.reindex(base_index, method="ffill")


# ─── Cross-timeframe features ─────────────────────────────────────────────────


def _add_cross_tf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that require columns from multiple timeframes.

    cpai  — cross-period acceleration index (1m / 5m / 15m agreement)
    CTI   — Calm-Trigger Index (calm depth × trigger × trend gate)
    """
    # ── cpai ─────────────────────────────────────────────────────────────────
    # Requires roc5 from three timeframes
    r1 = df.get("roc5_1m")
    r5 = df.get("roc5_5m")
    r15 = df.get("roc5_15m")
    if r1 is not None and r5 is not None and r15 is not None:
        sign_mean = (np.sign(r1) + np.sign(r5) + np.sign(r15)) / 3.0
        mag_mean = (r1.abs() + r5.abs() + r15.abs()) / 3.0
        df["cpai"] = sign_mean * mag_mean

    # ── CTI ───────────────────────────────────────────────────────────────────
    # CTI = vcdi_5m × max(trigger_score_5m, 0) × int(trend_ok)
    # trend_ok gate uses 4h and 1h features
    vcdi = df.get("vcdi_5m")
    ts = df.get("trigger_score_5m")
    trend_score = df.get("trend_score_4h")
    rsi_regime = df.get("rsi_regime_4h")
    macd_zone = df.get("macd_zone_1h")
    adx_val = df.get("adx_4h")

    if all(x is not None for x in [vcdi, ts, trend_score, rsi_regime, macd_zone, adx_val]):
        trend_ok = (
            (trend_score >= 0.0)
            & (rsi_regime >= 0.45)
            & (macd_zone > 0.0)
            & (adx_val > 15.0)
        ).astype(float)
        df["cti"] = vcdi * ts.clip(lower=0.0) * trend_ok

    return df


# ─── Funding rate alignment ───────────────────────────────────────────────────


def _align_funding(df_funding: pd.DataFrame, base_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill the 8h funding rate onto the 5m base index."""
    return df_funding["funding_rate"].reindex(base_index, method="ffill")


# ─── Time features ────────────────────────────────────────────────────────────


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek  # 0=Mon … 6=Sun
    return df


# ─── Filter flags ─────────────────────────────────────────────────────────────


def _add_filter_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_weekend    : True for Sat/Sun samples (NOT removed, just flagged)
    is_black_swan : True when 1m ATR z-score >= 4.0  (NOT removed, just flagged)
    """
    df["is_weekend"] = (df.index.dayofweek >= 5)

    atr_z = df.get("atr_zscore_1m")
    if atr_z is not None:
        df["is_black_swan"] = atr_z >= 4.0
    else:
        df["is_black_swan"] = False

    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────


def run(
    data_root: Path = DATA_ROOT,
    output_path: Path | None = None,
    entry_threshold: float = 0.010,
    horizon: int = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    include_p2: bool = False,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    data_root       : root of raw data (contains klines/ and funding_rate/)
    output_path     : where to write the Parquet file (default: data/processed/features.parquet)
    entry_threshold : y_class threshold (default 1.0 %)
    horizon         : number of future 1m bars for label computation (default 30 = 30 min)
    start_date      : optional ISO date string to slice data, e.g. "2021-01-01"
    end_date        : optional ISO date string to slice data, e.g. "2025-12-31"
    include_p2      : whether to compute slow P2 features (Hurst, sample entropy)

    Returns
    -------
    The assembled feature DataFrame (also written to Parquet).
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "features.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    log.info("Loading raw kline data …")
    raw: dict[str, pd.DataFrame] = {}
    for tf in KLINE_INTERVALS:
        log.info("  %s", tf)
        raw[tf] = load_klines(tf, data_root)

    log.info("Loading funding rate …")
    df_funding = load_funding_rate(data_root)

    # Optional date slicing (apply to 5m base first, then slice others)
    df_5m = raw["5m"]
    if start_date:
        df_5m = df_5m[df_5m.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df_5m = df_5m[df_5m.index <= pd.Timestamp(end_date, tz="UTC")]

    base_index: pd.DatetimeIndex = df_5m.index
    log.info("5m sample base: %d rows (%s → %s)", len(base_index), base_index[0], base_index[-1])

    # ── 2. Compute per-timeframe features ─────────────────────────────────────
    log.info("Computing per-timeframe features …")
    feat_parts: list[pd.DataFrame] = []

    for tf in KLINE_INTERVALS:
        log.info("  computing %s features …", tf)
        df_raw = raw[tf]
        # Slice raw data to a safe window before the 5m end date to avoid computing on unnecessary data
        feat = compute_features(df_raw, tf, include_p2=include_p2)

        # Detect effective bar frequency for alignment
        if tf == "1m":
            aligned = _align_1m(feat, base_index)
        else:
            aligned = _align_coarse(feat, base_index)

        feat_parts.append(aligned)
        log.info("    → %d columns", aligned.shape[1])

    # ── 3. Merge all timeframes ───────────────────────────────────────────────
    log.info("Merging timeframes …")
    merged = pd.concat(feat_parts, axis=1)

    # ── 4. Funding rate ───────────────────────────────────────────────────────
    merged["funding_rate"] = _align_funding(df_funding, base_index)

    # ── 5. Time features ─────────────────────────────────────────────────────
    merged = _add_time_features(merged)

    # ── 6. Cross-timeframe features ───────────────────────────────────────────
    log.info("Computing cross-timeframe features (cpai, CTI) …")
    merged = _add_cross_tf_features(merged)

    # ── 7. Labels ─────────────────────────────────────────────────────────────
    log.info("Computing labels (horizon=%d × 1m, threshold=%.3f) …", horizon, entry_threshold)
    labels = compute_labels(df_5m, raw["1m"], entry_threshold=entry_threshold, horizon=horizon)
    merged = merged.join(labels)

    # ── 8. Filter flags ───────────────────────────────────────────────────────
    merged = _add_filter_flags(merged)

    # ── 9. Attach raw 5m OHLC (for backtest TP/SL simulation) ────────────────
    # Prefixed price_ so they are never picked up as model features.
    for col in ["open", "high", "low", "close"]:
        merged[f"price_{col}"] = df_5m[col].reindex(base_index)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    n_total = len(merged)
    n_weekend = merged["is_weekend"].sum()
    n_bs = merged["is_black_swan"].sum()
    n_valid = (~merged["is_weekend"] & ~merged["is_black_swan"]).sum()
    pos_rate = merged.loc[~merged["is_weekend"] & ~merged["is_black_swan"], "y_class"].mean()

    log.info("── Feature matrix summary ──────────────────────────────")
    log.info("  Total samples   : %d", n_total)
    log.info("  Weekend samples : %d  (%.1f%%)", n_weekend, 100 * n_weekend / n_total)
    log.info("  Black-swan rows : %d  (%.1f%%)", n_bs, 100 * n_bs / n_total)
    log.info("  Training-eligible : %d", n_valid)
    log.info("  Positive-class rate (train eligible): %.1f%%", 100 * pos_rate)
    log.info("  Feature columns : %d", merged.shape[1])

    # ── 10. Save ──────────────────────────────────────────────────────────────
    log.info("Writing Parquet → %s  (%d columns)", output_path, merged.shape[1])
    merged.to_parquet(output_path, engine="pyarrow", compression="snappy")
    log.info("Done.  File size: %.1f MB", output_path.stat().st_size / 1e6)

    return merged


# ─── 1m alignment (special case: look up T5+4min) ────────────────────────────


def _align_1m(feat_1m: pd.DataFrame, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For 1m features, align using the 1m bar that closes at the same time
    the 5m bar closes (i.e., open_time = T5 + 4min).
    """
    lookup = base_index + pd.Timedelta("4min")
    aligned = feat_1m.reindex(lookup, method="ffill")
    aligned.index = base_index
    return aligned
