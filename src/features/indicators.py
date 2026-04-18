"""
Primitive indicator functions.

All functions are pure: they take pandas Series / DataFrames and return
new Series.  No global state, no side effects.
"""

import numpy as np
import pandas as pd

# ── Smoothing helpers ─────────────────────────────────────────────────────────


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard exponential moving average (pandas span convention)."""
    return series.ewm(span=period, adjust=False).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: equivalent to EWM with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


# ── Core price indicators ─────────────────────────────────────────────────────


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    tr = true_range(high, low, close)
    return wilder_smooth(tr, period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def macd_histogram(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram (MACD line minus signal line)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0,
) -> dict:
    """
    Bollinger Bands.

    Returns dict with keys: upper, lower, mid, width, position.
    - width    = (upper - lower) / mid  (normalized spread)
    - position = (close - lower) / (upper - lower)  [0, 1]
    """
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    band = upper - lower
    return {
        "upper": upper,
        "lower": lower,
        "mid": mid,
        "width": band / mid.replace(0.0, np.nan),
        "position": (close - lower) / band.replace(0.0, np.nan),
    }


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    tr = true_range(high, low, close)
    plus_dm = (high - high.shift(1)).clip(lower=0.0)
    minus_dm = (low.shift(1) - low).clip(lower=0.0)
    # Keep only the dominant direction per bar
    mask = plus_dm >= minus_dm
    plus_dm = plus_dm.where(mask, 0.0)
    minus_dm = minus_dm.where(~mask, 0.0)

    atr14 = wilder_smooth(tr, period)
    plus_di = 100.0 * wilder_smooth(plus_dm, period) / atr14.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_smooth(minus_dm, period) / atr14.replace(0.0, np.nan)
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return wilder_smooth(dx, period)


def vwap_daily(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    VWAP reset at UTC midnight each day.

    typical_price = (H + L + C) / 3
    """
    tp = (high + low + close) / 3.0
    tp_vol = tp * volume
    day_key = close.index.floor("D")
    cum_tp_vol = tp_vol.groupby(day_key).cumsum()
    cum_vol = volume.groupby(day_key).cumsum()
    return cum_tp_vol / cum_vol.replace(0.0, np.nan)


# ── Rolling utilities ─────────────────────────────────────────────────────────


def rolling_pct_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank of the current value within the past `window` bars.
    Returns values in [0, 1].  NaN until enough history.
    """
    def _rank(arr: np.ndarray) -> float:
        cur = arr[-1]
        hist = arr[:-1]
        if len(hist) == 0:
            return np.nan
        return float((hist < cur).sum()) / len(hist)

    return series.rolling(window + 1, min_periods=window + 1).apply(_rank, raw=True)


def consecutive_true(bool_series: pd.Series) -> pd.Series:
    """
    For each position, count how many consecutive True values end here
    (resets to 0 on False).

    Example: [F, T, T, T, F, T] → [0, 1, 2, 3, 0, 1]
    """
    s = bool_series.astype(int)
    groups = (~bool_series).cumsum()
    count = s.groupby(groups).cumsum()
    return count.where(bool_series, 0)


# ── Rate of change & acceleration ─────────────────────────────────────────────


def roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of change over `period` bars: (close[-1] / close[-1-period]) - 1."""
    return close / close.shift(period) - 1.0


def roc_accel(close: pd.Series, period: int) -> pd.Series:
    """
    Second derivative of price: difference of consecutive ROC values.
    period=5 → acceleration over 5-bar momentum window.
    """
    r = roc(close, period)
    return r - r.shift(1)


# ── Statistical features ──────────────────────────────────────────────────────


def autocorr_rolling(series: pd.Series, window: int = 50, lag: int = 5) -> pd.Series:
    """Rolling autocorrelation at given lag."""
    def _ac(arr: np.ndarray) -> float:
        n = len(arr)
        if n <= lag:
            return np.nan
        x = arr - arr.mean()
        num = np.dot(x[: n - lag], x[lag:])
        denom = np.sqrt(np.dot(x, x) * np.dot(x, x))  # not perfect but fast
        if denom < 1e-12:
            return np.nan
        return float(num / (np.std(x) * (n - lag) + 1e-12))

    return series.rolling(window, min_periods=window).apply(_ac, raw=True)


def hurst_rolling(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Rolling Hurst exponent via R/S analysis.
    H > 0.5: trending, H < 0.5: mean-reverting, H ≈ 0.5: random walk.
    Slow (rolling apply).  P2 feature — disabled by default.
    """
    def _hurst(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 20:
            return np.nan
        log_ret = np.diff(np.log(np.abs(arr) + 1e-10))
        if len(log_ret) < 4:
            return np.nan
        mean_r = log_ret.mean()
        centered = log_ret - mean_r
        cum = np.cumsum(centered)
        R = cum.max() - cum.min()
        S = log_ret.std(ddof=1)
        if S < 1e-12:
            return 0.5
        return float(np.log(R / S + 1e-10) / np.log(n / 2.0 + 1))

    return series.rolling(window, min_periods=window).apply(_hurst, raw=True)


def sample_entropy_rolling(series: pd.Series, window: int = 50) -> pd.Series:
    """
    Rolling approximate sample entropy (fast vectorized approximation).
    Lower = more predictable / quieter.
    P2 feature — disabled by default.
    """
    def _sampent(arr: np.ndarray, m: int = 2, r_scale: float = 0.2) -> float:
        n = len(arr)
        if n < 2 * (m + 1):
            return np.nan
        r = r_scale * arr.std()
        # Build template matrices
        templates_m = np.lib.stride_tricks.sliding_window_view(arr, m)
        templates_m1 = np.lib.stride_tricks.sliding_window_view(arr, m + 1)
        # Count matching pairs (Chebyshev distance < r)
        def _count(tmpl):
            nT = len(tmpl)
            count = 0
            for i in range(nT):
                diffs = np.max(np.abs(tmpl - tmpl[i]), axis=1)
                count += (diffs <= r).sum() - 1  # exclude self
            return count

        A = _count(templates_m1)
        B = _count(templates_m)
        if B == 0:
            return 0.0
        return float(-np.log((A + 1e-10) / (B + 1e-10)))

    return series.rolling(window, min_periods=window).apply(_sampent, raw=True)
