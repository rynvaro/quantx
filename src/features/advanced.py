"""
Advanced feature engineering for multi-horizon prediction.

Implements high-order derivatives and cross-timeframe features based on
trading psychology and market microstructure.

Design philosophy:
- Capture acceleration and jerk (2nd/3rd derivatives) to detect momentum shifts
- Multi-scale features to match prediction horizons (M15/M30/H1/H2)
- Volume-price interaction to measure buying/selling pressure
"""

import numpy as np
import pandas as pd
from typing import Dict


# ── Price Dynamics (高阶导数) ──────────────────────────────────────────────

def price_velocity(close: pd.Series, period: int) -> pd.Series:
    """Price velocity: (close - close[t-period]) / close[t-period]"""
    return close.pct_change(period)


def price_acceleration(close: pd.Series, period: int) -> pd.Series:
    """
    Price acceleration: velocity change rate.

    acc[t] = (velocity[t] - velocity[t-period]) / period

    Captures momentum shift: positive acc = accelerating up, negative = decelerating.
    """
    vel = price_velocity(close, period)
    return vel.diff(period) / period


def price_jerk(close: pd.Series, period: int) -> pd.Series:
    """
    Price jerk: acceleration change rate (3rd derivative).

    jerk[t] = (acc[t] - acc[t-period]) / period

    Captures turning points: when jerk changes sign, acceleration peaks/troughs.
    """
    acc = price_acceleration(close, period)
    return acc.diff(period) / period


# ── Volume Dynamics ────────────────────────────────────────────────────────

def volume_acceleration(volume: pd.Series, period: int) -> pd.Series:
    """Volume change acceleration (2nd derivative of volume)."""
    vol_chg = volume.pct_change(period)
    return vol_chg.diff(period) / period


def volume_price_momentum(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """
    Volume-weighted price momentum.

    vpm = sum(ret[i] * vol[i]) / sum(vol[i])  over last `period` bars

    Positive when volume concentrates on up-moves, negative on down-moves.
    """
    ret = close.pct_change()
    vol_ret = ret * volume
    return vol_ret.rolling(period).sum() / volume.rolling(period).sum()


def volume_price_correlation(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """
    Rolling correlation between price change and volume.

    High positive: volume confirms price move (healthy trend).
    Near zero: volume-price divergence (potential reversal).
    """
    ret = close.pct_change()
    return ret.rolling(period).corr(volume)


# ── Volatility Dynamics ────────────────────────────────────────────────────

def volatility_acceleration(close: pd.Series, period: int) -> pd.Series:
    """
    Volatility acceleration: change rate of realized volatility.

    vol[t] = std(returns, period)
    vol_acc[t] = (vol[t] - vol[t-period]) / period

    Positive = volatility expanding (risk increasing).
    """
    ret = close.pct_change()
    vol = ret.rolling(period).std()
    return vol.diff(period) / period


def volatility_regime(close: pd.Series, short_period: int, long_period: int) -> pd.Series:
    """
    Volatility regime: short-term vol / long-term vol.

    > 1.0: current volatility above average (high-risk regime)
    < 1.0: current volatility below average (calm regime)
    """
    ret = close.pct_change()
    vol_short = ret.rolling(short_period).std()
    vol_long = ret.rolling(long_period).std()
    return vol_short / vol_long.replace(0.0, np.nan)


# ── Buying/Selling Pressure ────────────────────────────────────────────────

def buy_sell_ratio_acceleration(
    taker_buy_volume: pd.Series,
    volume: pd.Series,
    period: int
) -> pd.Series:
    """
    Acceleration of buy/sell ratio.

    buy_ratio = taker_buy_volume / volume
    acc = (buy_ratio[t] - buy_ratio[t-period]) / period

    Positive = buying pressure accelerating.
    """
    buy_ratio = taker_buy_volume / volume.replace(0.0, np.nan)
    return buy_ratio.diff(period) / period


def net_buying_pressure(
    taker_buy_volume: pd.Series,
    volume: pd.Series,
    period: int
) -> pd.Series:
    """
    Net buying pressure over rolling window.

    nbp = sum(taker_buy - taker_sell) / sum(volume)
        = sum(2*taker_buy - volume) / sum(volume)

    Range: [-1, 1]. Positive = net buying, negative = net selling.
    """
    taker_sell = volume - taker_buy_volume
    net_buy = taker_buy_volume - taker_sell
    return net_buy.rolling(period).sum() / volume.rolling(period).sum()


# ── Technical Indicator Derivatives ────────────────────────────────────────

def rsi_velocity(close: pd.Series, rsi_period: int, vel_period: int) -> pd.Series:
    """
    RSI velocity: change rate of RSI.

    Captures momentum shift in overbought/oversold indicator.
    """
    from .indicators import rsi
    rsi_val = rsi(close, rsi_period)
    return rsi_val.diff(vel_period) / vel_period


def macd_acceleration(close: pd.Series, period: int) -> pd.Series:
    """
    MACD histogram acceleration.

    Captures turning points in MACD momentum.
    """
    from .indicators import macd_histogram
    macd_hist = macd_histogram(close)
    return macd_hist.diff(period) / period


def bollinger_squeeze_rate(close: pd.Series, period: int, lookback: int) -> pd.Series:
    """
    Bollinger Band squeeze rate: change rate of band width.

    Negative = bands squeezing (volatility contracting, breakout imminent).
    Positive = bands expanding (volatility increasing).
    """
    from .indicators import bollinger_bands
    bb = bollinger_bands(close, period)
    width = bb["width"]
    return width.diff(lookback) / lookback


# ── Multi-Timeframe Resonance ──────────────────────────────────────────────

def cross_timeframe_momentum_alignment(
    close_fast: pd.Series,
    close_slow: pd.Series,
    period: int
) -> pd.Series:
    """
    Momentum alignment between two timeframes.

    alignment = sign(ret_fast) * sign(ret_slow) * sqrt(|ret_fast * ret_slow|)

    Positive when both timeframes move in same direction (resonance).
    Magnitude reflects strength of alignment.
    """
    ret_fast = close_fast.pct_change(period)
    ret_slow = close_slow.pct_change(period)
    sign_product = np.sign(ret_fast) * np.sign(ret_slow)
    magnitude = np.sqrt((ret_fast * ret_slow).abs())
    return sign_product * magnitude


def volatility_cascade(
    close_1m: pd.Series,
    close_3m: pd.Series,
    close_5m: pd.Series,
    period: int
) -> pd.Series:
    """
    Volatility cascade: ratio of fine-scale to coarse-scale volatility.

    cascade = vol_1m / (vol_3m + vol_5m)

    High cascade = turbulence at fine scale (noise, choppy market).
    Low cascade = smooth trend across scales.
    """
    ret_1m = close_1m.pct_change()
    ret_3m = close_3m.pct_change()
    ret_5m = close_5m.pct_change()

    vol_1m = ret_1m.rolling(period).std()
    vol_3m = ret_3m.rolling(period).std()
    vol_5m = ret_5m.rolling(period).std()

    return vol_1m / (vol_3m + vol_5m).replace(0.0, np.nan)


# ── Feature Builder ────────────────────────────────────────────────────────

def compute_advanced_features(
    df: pd.DataFrame,
    timeframe: str,
    include_derivatives: bool = True,
    include_volume: bool = True,
    include_volatility: bool = True,
) -> pd.DataFrame:
    """
    Compute advanced features for a single timeframe.

    Parameters
    ----------
    df : DataFrame with OHLCV columns
    timeframe : '1m', '3m', '5m', etc.
    include_derivatives : compute price/volume acceleration/jerk
    include_volume : compute volume-price interaction features
    include_volatility : compute volatility dynamics features

    Returns
    -------
    DataFrame with new feature columns (suffix = f"_{timeframe}")
    """
    result = df.copy()
    close = df["close"]
    volume = df["volume"]

    # Determine period based on timeframe (for derivatives)
    # Use 3-5 bars as base period for acceleration
    base_period = 3

    if include_derivatives:
        # Price dynamics
        result[f"price_vel_{base_period}_{timeframe}"] = price_velocity(close, base_period)
        result[f"price_acc_{base_period}_{timeframe}"] = price_acceleration(close, base_period)
        result[f"price_jerk_{base_period}_{timeframe}"] = price_jerk(close, base_period)

        # Volume dynamics
        result[f"vol_acc_{base_period}_{timeframe}"] = volume_acceleration(volume, base_period)

    if include_volume and "taker_buy_volume" in df.columns:
        # Volume-price interaction
        result[f"vpm_{base_period}_{timeframe}"] = volume_price_momentum(
            close, volume, base_period
        )
        result[f"vpc_{base_period*2}_{timeframe}"] = volume_price_correlation(
            close, volume, base_period * 2
        )

        # Buying/selling pressure
        taker_buy = df["taker_buy_volume"]
        result[f"bsr_acc_{base_period}_{timeframe}"] = buy_sell_ratio_acceleration(
            taker_buy, volume, base_period
        )
        result[f"nbp_{base_period*2}_{timeframe}"] = net_buying_pressure(
            taker_buy, volume, base_period * 2
        )

    if include_volatility:
        # Volatility dynamics
        result[f"vol_acc_{base_period*2}_{timeframe}"] = volatility_acceleration(
            close, base_period * 2
        )
        result[f"vol_regime_{timeframe}"] = volatility_regime(
            close, short_period=base_period*2, long_period=base_period*10
        )

    return result
