"""
Multi-horizon label generation.

Generates labels for multiple prediction horizons (M15/M30/H1/H2) and thresholds.

Label definition (Definition 2 from design doc):
    y_class = 1 if any bar in future window reaches threshold, else 0
    y_max = max price in future window
    y_min = min price in future window

This captures "will price move ≥X% within next N minutes" for each horizon.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ── Label computation ──────────────────────────────────────────────────────

def compute_multi_horizon_labels(
    df: pd.DataFrame,
    horizons: List[Tuple[str, int]],
    thresholds: List[float],
    direction: str = "long",
) -> pd.DataFrame:
    """
    Compute labels for multiple horizons and thresholds.

    Parameters
    ----------
    df : DataFrame with 1m OHLC data (must have 'high', 'low', 'close' columns)
    horizons : List of (name, minutes) tuples, e.g., [("M15", 15), ("M30", 30)]
    thresholds : List of threshold percentages, e.g., [0.005, 0.008, 0.010]
    direction : "long" or "short"

    Returns
    -------
    DataFrame with label columns:
        - y_class_{horizon}_{thr}: binary label (0/1)
        - y_max_{horizon}: max price in window (for LONG)
        - y_min_{horizon}: min price in window (for SHORT)

    Example column names:
        y_class_M15_005, y_class_M15_008, y_class_M15_010
        y_max_M15, y_min_M15
    """
    result = df[["close"]].copy()

    for horizon_name, horizon_minutes in horizons:
        # Compute y_max and y_min for this horizon
        high_future = df["high"].shift(-1).rolling(horizon_minutes).max()
        low_future = df["low"].shift(-1).rolling(horizon_minutes).min()

        result[f"y_max_{horizon_name}"] = high_future
        result[f"y_min_{horizon_name}"] = low_future

        # Compute y_class for each threshold
        for thr in thresholds:
            thr_str = f"{int(thr * 1000):03d}"  # 0.005 -> "005", 0.010 -> "010"

            if direction == "long":
                # LONG: check if high reaches entry + threshold
                target = result["close"] * (1.0 + thr)
                hit = high_future >= target
            else:
                # SHORT: check if low reaches entry - threshold
                target = result["close"] * (1.0 - thr)
                hit = low_future <= target

            result[f"y_class_{horizon_name}_{thr_str}"] = hit.astype(int)

    # Drop the temporary close column
    result = result.drop(columns=["close"])

    return result


# ── Label statistics ───────────────────────────────────────────────────────

def compute_label_stats(
    df: pd.DataFrame,
    horizons: List[Tuple[str, int]],
    thresholds: List[float],
) -> pd.DataFrame:
    """
    Compute statistics for each horizon-threshold combination.

    Returns DataFrame with columns:
        - horizon: horizon name (M15, M30, etc.)
        - threshold: threshold percentage
        - positive_rate: fraction of samples with y_class=1
        - avg_time_to_target: average bars to reach target (for positive samples)
    """
    stats = []

    for horizon_name, horizon_minutes in horizons:
        for thr in thresholds:
            thr_str = f"{int(thr * 1000):03d}"
            col = f"y_class_{horizon_name}_{thr_str}"

            if col not in df.columns:
                continue

            y = df[col]
            pos_rate = y.mean()

            # Compute average time to target (for positive samples)
            # This requires re-computing when target is first hit
            # For now, use horizon_minutes / 2 as approximation
            avg_time = horizon_minutes / 2.0  # TODO: compute actual time

            stats.append({
                "horizon": horizon_name,
                "threshold": thr,
                "positive_rate": pos_rate,
                "avg_time_to_target": avg_time,
            })

    return pd.DataFrame(stats)


# ── Helper: compute actual time to target ──────────────────────────────────

def compute_time_to_target(
    df: pd.DataFrame,
    horizon_minutes: int,
    threshold: float,
    direction: str = "long",
) -> pd.Series:
    """
    For each sample, compute how many minutes until target is first reached.

    Returns Series with time in minutes (NaN if target not reached).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    time_to_target = pd.Series(np.nan, index=df.index)

    for i in range(len(df) - horizon_minutes):
        entry_price = close.iloc[i]

        if direction == "long":
            target = entry_price * (1.0 + threshold)
            future_high = high.iloc[i+1:i+1+horizon_minutes]
            hit_idx = (future_high >= target).idxmax()

            if future_high.loc[hit_idx] >= target:
                # Find position of first hit
                hit_pos = future_high.index.get_loc(hit_idx)
                time_to_target.iloc[i] = hit_pos + 1  # +1 because we start from i+1
        else:
            target = entry_price * (1.0 - threshold)
            future_low = low.iloc[i+1:i+1+horizon_minutes]
            hit_idx = (future_low <= target).idxmax()

            if future_low.loc[hit_idx] <= target:
                hit_pos = future_low.index.get_loc(hit_idx)
                time_to_target.iloc[i] = hit_pos + 1

    return time_to_target


# ── Preset configurations ──────────────────────────────────────────────────

# Default horizons and thresholds from design doc
DEFAULT_HORIZONS = [
    ("M15", 15),
    ("M30", 30),
    ("H1", 60),
    ("H2", 120),
]

DEFAULT_THRESHOLDS = {
    "M15": [0.003, 0.005, 0.008],  # 0.3%, 0.5%, 0.8%
    "M30": [0.005, 0.008, 0.010],  # 0.5%, 0.8%, 1.0%
    "H1": [0.008, 0.010, 0.012],   # 0.8%, 1.0%, 1.2%
    "H2": [0.010, 0.015, 0.020],   # 1.0%, 1.5%, 2.0%
}


def compute_default_labels(df: pd.DataFrame, direction: str = "long") -> pd.DataFrame:
    """
    Compute labels using default horizon-threshold combinations.

    This generates 12 models total:
        - M15: 3 thresholds (0.3%, 0.5%, 0.8%)
        - M30: 3 thresholds (0.5%, 0.8%, 1.0%)
        - H1: 3 thresholds (0.8%, 1.0%, 1.2%)
        - H2: 3 thresholds (1.0%, 1.5%, 2.0%)
    """
    all_labels = []

    for horizon_name, horizon_minutes in DEFAULT_HORIZONS:
        thresholds = DEFAULT_THRESHOLDS[horizon_name]
        labels = compute_multi_horizon_labels(
            df,
            horizons=[(horizon_name, horizon_minutes)],
            thresholds=thresholds,
            direction=direction,
        )
        all_labels.append(labels)

    # Concatenate all label columns
    return pd.concat(all_labels, axis=1)
