"""
Binance Futures REST fetcher.

All functions return DataFrames with the same schema as load_klines(),
so they can be passed directly to compute_features().
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

_BASE_URL = "https://fapi.binance.com"
_SYMBOL   = "ETHUSDT"

# Minimum bars per timeframe required to produce stable indicator values.
# Drives by the longest lookback: rolling_pct_rank(window=200) for micro TFs,
# EMA(200) for 4h.
_DEFAULT_LIMIT: dict[str, int] = {
    "1m":  600,   # 10 h — covers rolling(200) on 1m
    "5m":  600,   # 50 h — covers rolling(200) on 5m
    "15m": 400,   # ~4 days — covers rolling(200) on 15m
    "1h":  250,   # ~10 days — covers MACD warmup
    "4h":  280,   # ~47 days — covers EMA(200) on 4h
}

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

_FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
]


def _get(path: str, params: dict, timeout: int = 30) -> dict | list:
    url = _BASE_URL + path
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_klines(
    interval: str,
    limit: Optional[int] = None,
    symbol: str = _SYMBOL,
) -> pd.DataFrame:
    """
    Fetch the latest `limit` completed 5m bars from Binance Futures.

    Returns a DataFrame with the same schema as load_klines():
        index   : open_time (UTC DatetimeIndex)
        columns : open, high, low, close, volume, quote_volume,
                  taker_buy_volume, taker_buy_quote_volume, count
    The last (potentially incomplete) bar is always dropped.
    """
    if limit is None:
        limit = _DEFAULT_LIMIT.get(interval, 500)

    # Fetch one extra bar so we can safely drop the in-progress bar
    raw = _get(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit + 1},
    )

    df = pd.DataFrame(raw, columns=_KLINE_COLS)

    # Convert types
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in _FLOAT_COLS:
        df[c] = df[c].astype(float)
    df["count"] = df["count"].astype(int)

    # Drop bars whose close_time is in the future (still in progress)
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"].astype("int64") // 10**6 < now_ms]

    df = df.set_index("open_time").sort_index()
    keep = _FLOAT_COLS + ["count"]
    return df[keep]


def fetch_funding_rate(symbol: str = _SYMBOL) -> float:
    """
    Return the most recently settled funding rate for the symbol.
    Uses /fapi/v1/fundingRate (historical endpoint, always settled values).
    Returns 0.0 on any error rather than crashing the signal pipeline.
    """
    try:
        data = _get(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1},
        )
        if data and isinstance(data, list):
            return float(data[-1]["fundingRate"])
    except Exception:
        pass
    return 0.0
