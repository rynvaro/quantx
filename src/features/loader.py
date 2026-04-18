"""
Load raw Binance CSV files into pandas DataFrames.
All timestamps are UTC, index is open_time.
"""

import pandas as pd
from pathlib import Path

# ── Column layout from Binance kline CSV (no header row) ─────────────────────
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

_KLINE_FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
]

_KLINE_INT_COLS = ["count"]

# Columns exposed after loading
KLINE_KEEP = _KLINE_FLOAT_COLS + _KLINE_INT_COLS


def load_klines(interval: str, data_root: Path) -> pd.DataFrame:
    """
    Load all monthly kline CSVs for a given interval.

    Returns a DataFrame indexed by open_time (UTC), sorted ascending,
    with columns: open, high, low, close, volume, quote_volume,
                  taker_buy_volume, taker_buy_quote_volume, count.
    """
    folder = data_root / "klines" / interval
    files = sorted(folder.glob("ETHUSDT-*.csv"))
    if not files:
        raise FileNotFoundError(f"No kline CSV files in {folder}")

    dtype = {c: "float64" for c in _KLINE_FLOAT_COLS}
    dtype["count"] = "int64"

    def _read_kline_csv(f):
        # Binance changed format: pre-2024 files have no header, 2024+ files do
        first = f.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        has_header = first.startswith("open_time")
        if has_header:
            return pd.read_csv(f, dtype=dtype)
        else:
            return pd.read_csv(f, header=None, names=_KLINE_COLS, dtype=dtype)

    dfs = [_read_kline_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = (
        df.drop(columns=["close_time", "ignore"])
          .set_index("open_time")
          .sort_index()
    )
    df = df[~df.index.duplicated(keep="last")]
    return df[KLINE_KEEP]


def load_funding_rate(data_root: Path) -> pd.DataFrame:
    """
    Load all monthly fundingRate CSVs.

    Returns a DataFrame indexed by calc_time (UTC) with column: funding_rate.
    The funding rate is published every 8 hours.
    """
    folder = data_root / "funding_rate"
    files = sorted(folder.glob("ETHUSDT-fundingRate-*.csv"))
    if not files:
        raise FileNotFoundError(f"No fundingRate CSV files in {folder}")

    dfs = [pd.read_csv(f, usecols=["calc_time", "last_funding_rate"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.set_index("calc_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"last_funding_rate": "funding_rate"})
    return df
