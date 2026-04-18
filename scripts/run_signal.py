#!/usr/bin/env python3
"""
Live signal monitor — Phase 2.

Fetches the latest ETH/USDT 5m bar, computes features, runs the model,
and prints the calibrated signal probability.

Usage
-----
# Run once immediately
python scripts/run_signal.py --once

# Loop every 5 minutes (waits for bar close + 5s)
python scripts/run_signal.py

Options
-------
--threshold FLOAT        Signal threshold (default 0.60)
--long-model-dir PATH    Path to long model directory (default models/long/)
--short-model-dir PATH   Path to short model directory (default models/short/)
--once                   Run once and exit instead of looping
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live.fetcher import fetch_klines, fetch_funding_rate
from src.live.indicator import compute_live_features
from src.live.signal import load_signal_models, predict

_INTERVALS = ["1m", "5m", "15m", "1h", "4h"]
_BAR_SECONDS = 5 * 60   # 5-minute bar
_POST_CLOSE_DELAY = 5   # seconds after bar close before fetching


# ── Session state ─────────────────────────────────────────────────────────────

class _Session:
    def __init__(self, threshold: float):
        self.threshold  = threshold
        self.bars_checked = 0
        self.long_signals:  list[tuple[str, float]] = []  # (timestamp_str, prob)
        self.short_signals: list[tuple[str, float]] = []
        self.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def record(self, ts: str, long_prob: float, short_prob: float) -> None:
        self.bars_checked += 1
        if long_prob == long_prob and long_prob >= self.threshold:   # NaN-safe
            self.long_signals.append((ts, long_prob))
        if short_prob == short_prob and short_prob >= self.threshold:
            self.short_signals.append((ts, short_prob))

    def print_summary(self) -> None:
        n_long  = len(self.long_signals)
        n_short = len(self.short_signals)
        n_bars  = self.bars_checked
        print("─" * 58)
        print(f"  运行开始 {self.started_at}")
        print(
            f"  已检查 {n_bars} bars  |  "
            f"LONG 触发 {n_long} 次  |  SHORT 触发 {n_short} 次"
        )
        if self.long_signals:
            recent = "  ".join(
                f"{ts[-8:-4]} ({p:.2f})" for ts, p in self.long_signals[-3:]
            )
            print(f"  最近 LONG : {recent}")
        if self.short_signals:
            recent = "  ".join(
                f"{ts[-8:-4]} ({p:.2f})" for ts, p in self.short_signals[-3:]
            )
            print(f"  最近 SHORT: {recent}")
        print("─" * 58)


# ── Core run-once logic ────────────────────────────────────────────────────────

def run_once(
    long_models:  tuple,  # (model_cls, model_max, model_min, calibrator, feature_cols)
    short_models: tuple,
    threshold: float,
    session: _Session,
) -> None:
    """Fetch → features → infer both directions → print for the current bar."""
    # 1. Fetch data (parallel — all 5 intervals + funding rate concurrently)
    try:
        with ThreadPoolExecutor(max_workers=len(_INTERVALS) + 1) as pool:
            kline_futures = {pool.submit(fetch_klines, tf): tf for tf in _INTERVALS}
            funding_future = pool.submit(fetch_funding_rate)

            klines: dict = {}
            for fut in as_completed(kline_futures):
                tf = kline_futures[fut]
                klines[tf] = fut.result()   # raises on fetch error

            funding_rate = funding_future.result()
    except Exception as exc:
        print(f"[ERROR] Fetch failed: {exc}")
        return

    # 2. Compute features (shared for both directions)
    feature_row = compute_live_features(klines, funding_rate)
    if feature_row.empty:
        print("[WARN] Feature row is empty — skipping.")
        return

    bar_ts = feature_row.index[-1]
    ts_str = bar_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    price  = float(klines["5m"]["close"].iloc[-1])

    # 3. Infer — long and short independently
    def _infer(models: tuple) -> float:
        model_cls, _, _, calibrator, feature_cols = models
        try:
            return predict(feature_row, model_cls, calibrator, feature_cols)
        except Exception as exc:
            print(f"[ERROR] Inference failed: {exc}")
            return float("nan")

    long_prob  = _infer(long_models)
    short_prob = _infer(short_models)

    # 4. Print signal lines
    def _fmt(prob: float) -> str:
        if prob != prob:  # NaN
            return "NaN (warmup)          "
        mark = "\u2713" if prob >= threshold else " "
        label = "FIRE" if prob >= threshold else "--  "
        return f"{label}  prob={prob:.4f} {mark}"

    print(f"\n{ts_str}  price={price:.2f}")
    print(f"  LONG  {_fmt(long_prob)}")
    print(f"  SHORT {_fmt(short_prob)}")

    # 5. Update session and print summary
    session.record(ts_str, long_prob, short_prob)
    session.print_summary()


# ── Timing helper ─────────────────────────────────────────────────────────────

def _seconds_to_next_bar() -> float:
    """Seconds until the next 5m bar close + post-close delay."""
    now = time.time()
    bar_close = (now // _BAR_SECONDS + 1) * _BAR_SECONDS
    return max(0.0, bar_close - now + _POST_CLOSE_DELAY)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Live ETH/USDT signal monitor")
    parser.add_argument("--threshold",       type=float, default=0.60)
    parser.add_argument("--long-model-dir",  type=Path,  default=Path("models/long"))
    parser.add_argument("--short-model-dir", type=Path,  default=Path("models/short"))
    parser.add_argument(
        "--once", action="store_true",
        help="Run once immediately and exit",
    )
    args = parser.parse_args()

    print("Loading models …")
    try:
        long_models  = load_signal_models(args.long_model_dir)
        short_models = load_signal_models(args.short_model_dir)
    except Exception as exc:
        print(f"[ERROR] Failed to load models: {exc}")
        sys.exit(1)
    print(
        f"Models loaded. Threshold={args.threshold}  "
        f"Features(long)={len(long_models[4])}  Features(short)={len(short_models[4])}"
    )

    session = _Session(threshold=args.threshold)

    if args.once:
        run_once(long_models, short_models, args.threshold, session)
        return

    # Loop mode: wait for each bar close
    print("Loop mode — waiting for 5m bar close …  (Ctrl+C to stop)\n")
    try:
        while True:
            deadline = time.time() + _seconds_to_next_bar()
            while True:
                remaining = int(deadline - time.time())
                if remaining <= 0:
                    break
                m, s = divmod(remaining, 60)
                label = f"{m}m {s:02d}s" if m else f"{s}s"
                print(f"  [next bar in {label:>6}]  ", end="\r", flush=True)
                time.sleep(1)
            run_once(long_models, short_models, args.threshold, session)
    except KeyboardInterrupt:
        print("\n\n[停止]")
        session.print_summary()


if __name__ == "__main__":
    main()
