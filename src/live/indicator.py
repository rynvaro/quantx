"""
Live feature computation.

Reuses src.features.timeframe.compute_features() and
src.features.pipeline cross-timeframe / time feature logic verbatim,
ensuring the live feature values match the offline parquet exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.timeframe import compute_features
from src.features.pipeline import (
    _align_1m,
    _align_coarse,
    _align_funding,
    _add_cross_tf_features,
    _add_time_features,
)

_INTERVALS = ["1m", "5m", "15m", "1h", "4h"]


def compute_live_features(
    klines: dict[str, pd.DataFrame],
    funding_rate: float,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Compute the feature row for the latest completed 5m bar.

    Parameters
    ----------
    klines       : dict mapping interval → DataFrame (from fetcher.fetch_klines)
    funding_rate : most recent settled funding rate (float)
    direction    : "long" or "short" — must match the model being evaluated.
                   Controls the CTI cross-timeframe feature computation so that
                   live features match the offline training parquet exactly.

    Returns
    -------
    Single-row DataFrame with all feature columns, indexed by the latest
    completed 5m bar open_time. Returns an empty DataFrame on failure.
    """
    df_5m = klines["5m"]
    if df_5m.empty:
        return pd.DataFrame()

    # Use the latest completed 5m bar as the single base index
    base_index: pd.DatetimeIndex = df_5m.index[-1:]

    # ── Per-timeframe features ────────────────────────────────────────────────
    feat_parts: list[pd.DataFrame] = []
    for tf in _INTERVALS:
        df_raw = klines[tf]
        if df_raw.empty:
            continue
        feat = compute_features(df_raw, tf, include_p2=False, direction=direction)

        if tf == "1m":
            aligned = _align_1m(feat, base_index)
        else:
            aligned = _align_coarse(feat, base_index)

        feat_parts.append(aligned)

    if not feat_parts:
        return pd.DataFrame()

    merged = pd.concat(feat_parts, axis=1)

    # ── Funding rate ──────────────────────────────────────────────────────────
    # Build a minimal Series that _align_funding can forward-fill
    fr_series = pd.DataFrame(
        {"funding_rate": [funding_rate]},
        index=pd.DatetimeIndex([base_index[0]], tz="UTC"),
    )
    merged["funding_rate"] = _align_funding(fr_series, base_index)

    # ── Time features ─────────────────────────────────────────────────────────
    merged = _add_time_features(merged)

    # ── Cross-timeframe features (cpai, CTI) ──────────────────────────────────
    # direction must be passed so CTI uses the same trend_ok gate as training
    merged = _add_cross_tf_features(merged, direction=direction)

    return merged
