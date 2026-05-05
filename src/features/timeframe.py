"""
Compute all features for a single timeframe DataFrame.

Usage:
    feat_1m  = compute_features(df_1m,  "1m",  include_p2=False)
    feat_4h  = compute_features(df_4h,  "4h",  include_p2=False)

Each output column is suffixed with the timeframe name (e.g. "atr14_1m").
Only the timeframe-appropriate features are computed for each interval.
"""

import numpy as np
import pandas as pd

from .indicators import (
    atr, rsi, macd_histogram, bollinger_bands, adx, ema,
    vwap_daily, rolling_pct_rank, consecutive_true,
    roc, roc_accel,
    autocorr_rolling, hurst_rolling, sample_entropy_rolling,
)

# ─── Timeframe tiers ──────────────────────────────────────────────────────────
# "micro"  : 1m, 5m, 15m  →  full calm + trigger + momentum + TA
# "medium" : 1h            →  TA + trend context (1h-specific)
# "macro"  : 4h            →  TA + trend context (4h-specific) + ADX

_MICRO_TF = {"1m", "5m", "15m"}
_MEDIUM_TF = {"1h"}
_MACRO_TF = {"4h"}
_VWAP_TF = {"1m", "5m"}   # VWAP only for sub-hour frames


def compute_features(
    df: pd.DataFrame,
    tf: str,
    include_p2: bool = False,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Compute all features for a single-timeframe OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with columns [open, high, low, close, volume,
                                  taker_buy_volume, (quote_volume, count)]
         indexed by UTC open_time, sorted ascending.
    tf : timeframe label, e.g. "1m", "5m", "15m", "1h", "4h"
    include_p2 : whether to compute slow P2 features (Hurst, sample entropy)
    direction : "long" (default) or "short" — controls directional features

    Returns
    -------
    DataFrame with feature columns only (no raw OHLCV), indexed by open_time.
    All column names end in f"_{tf}".
    """
    out: dict[str, pd.Series] = {}
    suffix = f"_{tf}"

    hi, lo, cl, op = df["high"], df["low"], df["close"], df["open"]
    vol = df["volume"]

    # ── ATR (needed by multiple features) ────────────────────────────────────
    atr14 = atr(hi, lo, cl, 14)
    out[f"atr14{suffix}"] = atr14

    # ── Bollinger Bands (needed by multiple features) ─────────────────────────
    bb = bollinger_bands(cl, 20, 2.0)
    out[f"bb_width{suffix}"] = bb["width"]
    out[f"bb_position{suffix}"] = bb["position"]

    # ── RSI ───────────────────────────────────────────────────────────────────
    out[f"rsi14{suffix}"] = rsi(cl, 14)

    # ── MACD histogram ────────────────────────────────────────────────────────
    out[f"macd_hist{suffix}"] = macd_histogram(cl)

    # ── EMA cross ─────────────────────────────────────────────────────────────
    ema9 = ema(cl, 9)
    ema21 = ema(cl, 21)
    ema55 = ema(cl, 55)
    out[f"ema9{suffix}"] = ema9
    out[f"ema21{suffix}"] = ema21
    out[f"ema55{suffix}"] = ema55
    out[f"ema_cross{suffix}"] = ema9 / ema21.replace(0.0, np.nan) - 1.0

    if tf in _MICRO_TF:
        _add_micro_features(out, df, atr14, bb, suffix, include_p2, direction)

    if tf in _VWAP_TF:
        vwap = vwap_daily(hi, lo, cl, vol)
        out[f"vwap_dev{suffix}"] = cl / vwap.replace(0.0, np.nan) - 1.0

    if tf in _MEDIUM_TF:
        _add_medium_features(out, df, suffix)

    if tf in _MACRO_TF:
        _add_macro_features(out, df, ema9, ema21, ema55, atr14, suffix)

    return pd.DataFrame(out, index=df.index)


# ─── Micro (1m / 5m / 15m) feature builders ──────────────────────────────────


def _add_micro_features(
    out: dict,
    df: pd.DataFrame,
    atr14: pd.Series,
    bb: dict,
    suffix: str,
    include_p2: bool,
    direction: str = "long",
) -> None:
    cl, op, hi, lo, vol = (
        df["close"], df["open"], df["high"], df["low"], df["volume"]
    )
    taker_buy = df["taker_buy_volume"]

    # ── Calm Layer ────────────────────────────────────────────────────────────
    # ATR z-score (also used by the pipeline for black-swan filter)
    atr_mean200 = atr14.rolling(200).mean()
    atr_std200 = atr14.rolling(200).std(ddof=0)
    out[f"atr_zscore{suffix}"] = (atr14 - atr_mean200) / atr_std200.replace(0.0, np.nan)

    compress_cond = atr14 < atr14.rolling(50).mean()
    compress_bars = consecutive_true(compress_cond)
    out[f"compress_bars{suffix}"] = compress_bars

    pct_rank_atr = rolling_pct_rank(atr14, 200)
    out[f"vcdi{suffix}"] = (1.0 - pct_rank_atr) * np.log1p(compress_bars)

    pct_rank_bb = rolling_pct_rank(bb["width"], 200)
    bb_roc5 = bb["width"] / bb["width"].shift(5) - 1.0
    out[f"bb_width_pct{suffix}"] = pct_rank_bb
    out[f"earp{suffix}"] = pct_rank_bb * np.sign(bb_roc5)

    # ── Trigger Layer ─────────────────────────────────────────────────────────
    roc1 = roc(cl, 1)
    roc3 = roc(cl, 3)
    roc5 = roc(cl, 5)
    accel1 = roc_accel(cl, 1)
    accel3 = roc_accel(cl, 3)
    accel5 = roc_accel(cl, 5)
    vol_accel = vol / vol.rolling(5).mean().replace(0.0, np.nan) - 1.0

    out[f"roc1{suffix}"] = roc1
    out[f"roc3{suffix}"] = roc3
    out[f"roc5{suffix}"] = roc5
    out[f"roc_accel1{suffix}"] = accel1
    out[f"roc_accel3{suffix}"] = accel3
    out[f"roc_accel5{suffix}"] = accel5
    out[f"volume_accel{suffix}"] = vol_accel

    # trigger_score: direction-aware spring release signal.
    # LONG: positive accel5 * volume surge = upward burst.
    # SHORT: negate accel5 so positive trigger = downward burst.
    if direction == "long":
        trigger = accel5 * vol_accel.clip(lower=0.0)
    else:
        trigger = (-accel5) * vol_accel.clip(lower=0.0)
    out[f"trigger_score{suffix}"] = trigger

    if direction == "long":
        accel_consensus = (
            (np.sign(accel5) == 1) & (np.sign(accel3) == 1) & (np.sign(accel1) == 1)
        ).astype(float)
    else:
        accel_consensus = (
            (np.sign(accel5) == -1) & (np.sign(accel3) == -1) & (np.sign(accel1) == -1)
        ).astype(float)
    out[f"accel_consensus{suffix}"] = accel_consensus

    # ── Momentum Confirmation Layer ───────────────────────────────────────────
    tbr = taker_buy / vol.replace(0.0, np.nan)
    out[f"taker_buy_ratio{suffix}"] = tbr

    # tbr_trend: direction-aware momentum trend of taker buy/sell ratio.
    # LONG: MA5 - MA15 of TBR; positive = buy pressure building = bullish.
    # SHORT: MA15 - MA5 of TBR (= TSR trend); positive = sell pressure building = bearish.
    # Short window=5 bars, long window=15 bars — balances sensitivity vs noise.
    _tbr_ma5  = tbr.rolling(5).mean()
    _tbr_ma15 = tbr.rolling(15).mean()
    if direction == "long":
        out[f"tbr_trend{suffix}"] = _tbr_ma5 - _tbr_ma15
    else:
        out[f"tbr_trend{suffix}"] = _tbr_ma15 - _tbr_ma5

    if direction == "long":
        consec = consecutive_true(cl > cl.shift(1))
        bsipi = (tbr - 0.5).rolling(10).mean() * np.sqrt(consec + 1.0)
    else:
        consec = consecutive_true(cl < cl.shift(1))
        bsipi = (0.5 - tbr).rolling(10).mean() * np.sqrt(consec + 1.0)
    out[f"bsipi{suffix}"] = bsipi

    # cbfi_ema: (stance - 0.5) * 2 * vol_ratio, EMA5
    vol_ratio = vol / vol.rolling(20).mean().replace(0.0, np.nan)
    if direction == "long":
        stance = ((cl - op) / (hi - lo + 1e-8)).clip(0.0, 1.0)
        cbfi = (stance - 0.5) * 2.0 * vol_ratio
    else:
        stance = ((cl - op) / (hi - lo + 1e-8)).clip(-1.0, 0.0)
        cbfi = (stance + 0.5) * (-2.0) * vol_ratio
    out[f"cbfi_ema{suffix}"] = cbfi.ewm(span=5, adjust=False).mean()

    # mvdi: (roc5 / std(roc5, 50)) / (atr14 / mean(atr14, 50))
    roc5_std = roc5.rolling(50).std(ddof=0).replace(0.0, np.nan)
    atr_mean50 = atr14.rolling(50).mean().replace(0.0, np.nan)
    norm_roc5 = roc5 / roc5_std
    norm_atr = atr14 / atr_mean50
    # mvdi: direction-aware. LONG: positive = upward momentum; SHORT: negate so positive = downward momentum.
    _raw_mvdi = norm_roc5 / norm_atr.replace(0.0, np.nan)
    out[f"mvdi{suffix}"] = _raw_mvdi if direction == "long" else -_raw_mvdi

    # pei: price impact = |Δclose| / volume; EMA(5)/EMA(20) - 1
    impact = cl.diff().abs() / vol.replace(0.0, np.nan)
    pei = impact.ewm(span=5, adjust=False).mean() / impact.ewm(span=20, adjust=False).mean().replace(0.0, np.nan) - 1.0
    out[f"pei{suffix}"] = pei

    # vol_inefficiency: excess volume relative to price move.
    # = (vol / vol_MA5) / (|roc1| + epsilon).  High value means lots of volume
    # absorbed with little price change — signals hidden distribution/accumulation
    # and often precedes a directional break.
    vol_ma5 = vol.rolling(5).mean().replace(0.0, np.nan)
    out[f"vol_inefficiency{suffix}"] = (vol / vol_ma5) / (roc1.abs() + 1e-6)

    # ── Spring Physics Features (Experimental P2) ─────────────────────────────
    compress_int = compress_bars.astype(int)

    # compress_quality: std(returns) during compression window
    ret = cl.pct_change()
    out[f"compress_quality{suffix}"] = _rolling_window_std(ret, compress_int)

    # vol_dry_ratio: mean_vol_in_compression / mean_vol_20_before_compression
    out[f"vol_dry_ratio{suffix}"] = _vol_dry_ratio(vol, compress_int)

    # price_in_range: position of close within compression price range
    out[f"price_in_range{suffix}"] = _price_in_range(cl, compress_int)

    # release_phase: trigger_score trending up (+1) or peaking (-1)
    out[f"release_phase{suffix}"] = np.sign(trigger - trigger.shift(3))

    # ── Statistical Features (P2) ─────────────────────────────────────────────
    out[f"autocorr_lag5{suffix}"] = autocorr_rolling(cl, window=50, lag=5)

    if include_p2:
        out[f"hurst{suffix}"] = hurst_rolling(cl, window=100)
        out[f"sample_entropy{suffix}"] = sample_entropy_rolling(cl, window=50)


# ─── Medium (1h) feature builders ────────────────────────────────────────────


def _add_medium_features(out: dict, df: pd.DataFrame, suffix: str) -> None:
    cl = df["close"]
    mhist = macd_histogram(cl)
    out[f"macd_zone{suffix}"] = np.sign(mhist)


# ─── Macro (4h) feature builders ─────────────────────────────────────────────


def _add_macro_features(
    out: dict,
    df: pd.DataFrame,
    ema9: pd.Series,
    ema21: pd.Series,
    ema55: pd.Series,
    atr14: pd.Series,
    suffix: str,
) -> None:
    cl, hi, lo = df["close"], df["high"], df["low"]

    # EMA200 (used for trend_score and deviation)
    ema200 = ema(cl, 200)
    out[f"ema200{suffix}"] = ema200
    out[f"deviation{suffix}"] = cl / ema200.replace(0.0, np.nan) - 1.0

    # trend_score: mean of 3 sign signals → {-1, -0.33, 0.33, 1}
    s1 = np.sign(ema9 - ema21)
    s2 = np.sign(ema21 - ema55)
    s3 = np.sign(cl - ema200)
    out[f"trend_score{suffix}"] = (s1 + s2 + s3) / 3.0

    # EMA55 slope over 10 bars
    out[f"ema_slope{suffix}"] = (ema55 - ema55.shift(10)) / ema55.shift(10).replace(0.0, np.nan)

    # RSI regime (normalized to [0, 1])
    out[f"rsi_regime{suffix}"] = rsi(cl, 14) / 100.0

    # ADX
    out[f"adx{suffix}"] = adx(hi, lo, cl, 14)


# ─── Helpers for spring physics features ─────────────────────────────────────


def _rolling_window_std(ret: pd.Series, compress_bars: pd.Series) -> pd.Series:
    """Std of returns over the last `compress_bars[i]` bars at each position."""
    n = len(ret)
    result = np.full(n, np.nan)
    ret_arr = ret.values
    cb_arr = compress_bars.values.astype(int)
    for i in range(n):
        w = cb_arr[i]
        if w >= 2 and i >= w:
            result[i] = float(np.std(ret_arr[i - w + 1: i + 1], ddof=0))
    return pd.Series(result, index=ret.index)


def _vol_dry_ratio(volume: pd.Series, compress_bars: pd.Series) -> pd.Series:
    """Mean volume during compression / mean volume in 20 bars before compression."""
    n = len(volume)
    result = np.full(n, np.nan)
    vol_arr = volume.values
    cb_arr = compress_bars.values.astype(int)
    for i in range(n):
        w = cb_arr[i]
        if w >= 2 and i >= w + 20:
            mean_compress = vol_arr[i - w + 1: i + 1].mean()
            mean_pre = vol_arr[i - w - 20: i - w].mean()
            if mean_pre > 0:
                result[i] = mean_compress / mean_pre
    return pd.Series(result, index=volume.index)


def _price_in_range(close: pd.Series, compress_bars: pd.Series) -> pd.Series:
    """Position of close within compression range [0=bottom, 1=top]."""
    n = len(close)
    result = np.full(n, np.nan)
    cl_arr = close.values
    cb_arr = compress_bars.values.astype(int)
    for i in range(n):
        w = cb_arr[i]
        if w >= 2 and i >= w:
            window = cl_arr[i - w + 1: i + 1]
            lo = window.min()
            hi = window.max()
            rng = hi - lo
            if rng > 1e-10:
                result[i] = (cl_arr[i] - lo) / rng
    return pd.Series(result, index=close.index)
