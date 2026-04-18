"""
Compute training labels for each 5m sample.

For a 5m bar with open_time T5:
  - Entry reference price = close of the 5m bar = close of 1m bar at T5+4min
  - Future window         = next 15 × 1m closes (open_times T5+5m … T5+19m)

Labels:
  y_max   = max(future_closes) / ref_close - 1   (positive float)
  y_min   = min(future_closes) / ref_close - 1   (negative float)
  y_class = 1 if y_max >= entry_threshold else 0
"""

import numpy as np
import pandas as pd

# Number of 1m bars to look forward
HORIZON = 15


def compute_labels(
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame,
    entry_threshold: float = 0.010,
    horizon: int = HORIZON,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Vectorised label computation.

    Parameters
    ----------
    df_5m : 5m OHLCV DataFrame indexed by UTC open_time.
    df_1m : 1m OHLCV DataFrame indexed by UTC open_time.
    entry_threshold : positive rate-of-return threshold (e.g. 0.010 = 1 %).
    horizon : number of 1m bars to look forward (default 15).
    direction : "long" (default) or "short". For short, y_class fires when
                price drops by entry_threshold; y_max/y_min are flipped so
                y_max = max favorable drop and y_min = max adverse rise.

    Returns
    -------
    DataFrame with columns [y_class, y_max, y_min] indexed by 5m open_time.
    Rows where fewer than 10 future bars exist are set to NaN.
    """
    delta5 = pd.Timedelta("5min")
    delta1 = pd.Timedelta("1min")

    close_1m = df_1m["close"]
    ref_close = df_5m["close"].values  # entry price = close of the 5m bar

    # Build (n_samples × horizon) matrix of future 1m closes
    # The i-th future bar (0-indexed) opens at T5 + 5min + i*1min
    future_ts_list = [
        df_5m.index + delta5 + i * delta1
        for i in range(horizon)
    ]

    # Each column: close_1m reindexed to the future timestamps
    # Rows missing in 1m (end of data) become NaN
    future_matrix = np.column_stack([
        close_1m.reindex(ts).values
        for ts in future_ts_list
    ])  # shape: (n_5m_samples, horizon)

    # Count valid (non-NaN) future bars per sample
    valid_count = np.sum(~np.isnan(future_matrix), axis=1)

    with np.errstate(invalid="ignore"):
        future_ret = future_matrix / ref_close[:, None] - 1.0

    y_max = np.nanmax(future_ret, axis=1)
    y_min = np.nanmin(future_ret, axis=1)

    # Mask samples with insufficient future data
    insufficient = valid_count < 10
    y_max[insufficient] = np.nan
    y_min[insufficient] = np.nan

    y_class = np.where(np.isnan(y_max), np.nan, (y_max >= entry_threshold).astype(float))

    if direction == "long":
        return pd.DataFrame(
            {"y_class": y_class, "y_max": y_max, "y_min": y_min},
            index=df_5m.index,
        )

    # short: win when price drops by entry_threshold
    # y_max_s = max favorable drop (positive) = -y_min
    # y_min_s = max adverse rise  (negative)  = -y_max
    y_max_s = -y_min
    y_min_s = -y_max
    y_class_s = np.where(
        np.isnan(y_max_s), np.nan, (y_max_s >= entry_threshold).astype(float)
    )
    return pd.DataFrame(
        {"y_class": y_class_s, "y_max": y_max_s, "y_min": y_min_s},
        index=df_5m.index,
    )
