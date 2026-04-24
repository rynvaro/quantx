"""
Event-driven backtest engine for the Calm-Trigger/Spring Model.

Each eligible 5m bar is a potential entry signal. The engine:
  1. Loads the features parquet and model artifacts
  2. Generates signals (calibrated probability >= threshold)
  3. Simulates entry at the NEXT 5m bar's open (no look-ahead)
  4. Manages position with take-profit and stop-loss
  5. Records every trade and computes portfolio statistics

Assumptions
-----------
- One position at a time (no pyramiding), long or short
- Entry: market order at signal bar's close + slippage
- Exit: either TP/SL hit within horizon, or market close at horizon end
- Position sizing: fixed margin per trade (25% of current equity)
- Leverage: 75x
- Fee: 0.04% taker per side (entry + exit)
- Slippage: 0.03% per side (conservative estimate for ETH/USDT perp)
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    # Signal
    signal_threshold: float = 0.60   # calibrated probability cutoff

    # Capital
    initial_equity: float  = 312.0   # USDT
    margin_pct:     float  = 0.25    # fraction of INITIAL equity per trade (fixed sizing)
    leverage:       float  = 75.0
    # Fixed margin per trade = initial_equity * margin_pct (does NOT compound).
    # This gives realistic PnL without runaway compounding in backtests.

    # Exit rules
    take_profit:    float  = 0.008   # 0.8% from entry (ETH price move)
    stop_loss:      float  = 0.008   # 0.8% adverse move → close position
    horizon_bars:   int    = 6       # max hold = 6 × 5m = 30min

    # Costs
    fee_rate:       float  = 0.0004  # taker 0.04% per side
    slippage:       float  = 0.0003  # 0.03% per side

    # Direction: 'long' or 'short'
    direction:      str        = "long"

    # Filter: only trade during these UTC hours (None = all hours)
    active_hours:   list[int] | None = None


# ─── Trade record ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:   pd.Timestamp
    exit_time:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    margin:       float          # USDT posted as margin
    notional:     float          # margin × leverage
    pnl:          float          # net PnL in USDT (after fees + slippage)
    exit_reason:  str            # 'tp' | 'sl' | 'timeout'
    signal_prob:  float          # model probability at entry


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Vectorised-friendly backtest engine.

    The simulation iterates over signals only (not every bar),
    making it fast even for 5-year datasets.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.cfg = config or BacktestConfig()
        self._model_cls: XGBClassifier | None = None
        self._model_max: XGBRegressor  | None = None
        self._calibrator = None
        self._feat_cols: list[str] = []

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_models(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)

        self._model_cls = XGBClassifier()
        self._model_cls.load_model(model_dir / "model_cls.json")

        self._model_max = XGBRegressor()
        self._model_max.load_model(model_dir / "model_max.json")

        with open(model_dir / "calibrator.pkl", "rb") as f:
            self._calibrator = pickle.load(f)

        self._feat_cols = (model_dir / "feature_cols.txt").read_text().splitlines()
        log.info("Models loaded from %s  (%d features)", model_dir, len(self._feat_cols))

    # ── Signal generation ─────────────────────────────────────────────────────

    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a Series of calibrated probabilities for all eligible rows.
        Rows not in df (filtered out) are not scored.
        """
        X = df[self._feat_cols].values.astype(np.float32)
        raw = self._model_cls.predict_proba(X)[:, 1]
        cal = self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return pd.Series(cal, index=df.index, name="signal_prob")

    # ── Core simulation ───────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        model_dir: Path,
        start_date: str | None = None,
        end_date:   str | None = None,
    ) -> tuple[list[Trade], pd.DataFrame]:
        """
        Run the full backtest.

        Parameters
        ----------
        df         : features parquet DataFrame (must contain price_open/high/low/close columns)
        model_dir  : directory containing model artifacts
        start_date : optional ISO date to restrict backtest start
        end_date   : optional ISO date to restrict backtest end

        Returns
        -------
        (trades, equity_curve)
          trades       : list of Trade objects
          equity_curve : DataFrame with columns [equity, drawdown] indexed by time
        """
        if self._model_cls is None:
            self.load_models(model_dir)

        # Check if parquet contains embedded OHLC columns
        has_ohlc = all(c in df.columns for c in ["price_open", "price_high", "price_low", "price_close"])
        if not has_ohlc:
            log.warning("price_open/high/low/close columns not found in DataFrame. "
                        "TP/SL simulation will be close-only (less accurate). "
                        "Re-run build_features.py to embed OHLC into the parquet.")

        # ── Filter eligible rows ──────────────────────────────────────────────
        mask = (df["is_weekend"] == 0) & (df["is_black_swan"] == 0) & df["y_class"].notna()
        if start_date:
            mask &= df.index >= pd.Timestamp(start_date, tz="UTC")
        if end_date:
            mask &= df.index <= pd.Timestamp(end_date, tz="UTC")
        if self.cfg.active_hours:
            mask &= df.index.hour.isin(self.cfg.active_hours)

        eligible = df[mask].copy()
        log.info("Backtest window: %s → %s  (%d eligible bars)",
                 eligible.index[0].date(), eligible.index[-1].date(), len(eligible))

        # ── Score all eligible bars ───────────────────────────────────────────
        log.info("Generating signals …")
        probs = self._generate_signals(eligible)
        signal_mask = probs >= self.cfg.signal_threshold
        signal_times = probs[signal_mask].index
        log.info("Signals: %d  (threshold=%.2f)", len(signal_times), self.cfg.signal_threshold)

        # ── Simulate trades ───────────────────────────────────────────────────
        trades: list[Trade] = []
        equity = self.cfg.initial_equity
        in_position = False
        position_end: pd.Timestamp | None = None

        for sig_time in signal_times:
            # Skip if still in a previous position
            if in_position and sig_time <= position_end:
                continue

            # Entry: use NEXT bar's open (no look-ahead bias)
            # Find the next bar after signal
            future_bars = df.index[df.index > sig_time]
            if len(future_bars) == 0:
                continue
            entry_bar = future_bars[0]

            if has_ohlc:
                entry_price_raw = df.at[entry_bar, "price_open"]
            else:
                entry_price_raw = np.nan
            if np.isnan(entry_price_raw):
                continue

            if self.cfg.direction == "long":
                # Buy: fill at next open + slippage; profit from price rising
                entry_price = entry_price_raw * (1 + self.cfg.slippage)
                tp_price = entry_price * (1 + self.cfg.take_profit)
                sl_price = entry_price * (1 - self.cfg.stop_loss)
            else:
                # Sell short: fill at next open - slippage; profit from price falling
                entry_price = entry_price_raw * (1 - self.cfg.slippage)
                tp_price = entry_price * (1 - self.cfg.take_profit)
                sl_price = entry_price * (1 + self.cfg.stop_loss)

            margin   = self.cfg.initial_equity * self.cfg.margin_pct  # fixed, no compounding
            notional = margin * self.cfg.leverage
            fee_in   = notional * self.cfg.fee_rate

            # ── Simulate exit over next horizon_bars (starting from entry_bar) ───
            # Find future bars after entry
            future_idx = df.index[df.index > entry_bar][:self.cfg.horizon_bars]
            if len(future_idx) == 0:
                continue

            exit_price  = None
            exit_time   = None
            exit_reason = "timeout"

            for bar_ts in future_idx:
                if has_ohlc:
                    bar_high  = df.at[bar_ts, "price_high"]
                    bar_low   = df.at[bar_ts, "price_low"]
                    bar_close = df.at[bar_ts, "price_close"]
                else:
                    bar_close = np.nan
                    bar_high  = bar_close
                    bar_low   = bar_close

                if self.cfg.direction == "long":
                    # SL: price drops to sl_price; TP: price rises to tp_price
                    # Check SL first (pessimistic)
                    if bar_low <= sl_price:
                        exit_price  = sl_price * (1 - self.cfg.slippage)
                        exit_time   = bar_ts
                        exit_reason = "sl"
                        break
                    elif bar_high >= tp_price:
                        exit_price  = tp_price * (1 - self.cfg.slippage)
                        exit_time   = bar_ts
                        exit_reason = "tp"
                        break
                else:
                    # Short — SL: price rises to sl_price; TP: price falls to tp_price
                    # Check SL first (pessimistic)
                    if bar_high >= sl_price:
                        exit_price  = sl_price * (1 + self.cfg.slippage)
                        exit_time   = bar_ts
                        exit_reason = "sl"
                        break
                    elif bar_low <= tp_price:
                        exit_price  = tp_price * (1 + self.cfg.slippage)
                        exit_time   = bar_ts
                        exit_reason = "tp"
                        break
            else:
                # Timeout: exit at last bar's close
                last_ts = future_idx[-1]
                last_close = df.at[last_ts, "price_close"] if has_ohlc else np.nan
                if not np.isnan(last_close):
                    # Long timeout: cover at close - slippage (we sell to exit)
                    # Short timeout: cover at close + slippage (we buy to exit)
                    slip_dir = -1 if self.cfg.direction == "long" else +1
                    exit_price = last_close * (1 + slip_dir * self.cfg.slippage)
                else:
                    exit_price = entry_price
                exit_time   = last_ts
                exit_reason = "timeout"

            if exit_price is None or np.isnan(exit_price):
                continue

            fee_out = notional * self.cfg.fee_rate
            if self.cfg.direction == "long":
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price
            pnl = notional * ret - fee_in - fee_out

            equity += pnl

            trade = Trade(
                entry_time=entry_bar,  # actual entry time (next bar after signal)
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                margin=margin,
                notional=notional,
                pnl=pnl,
                exit_reason=exit_reason,
                signal_prob=float(probs.at[sig_time]),
            )
            trades.append(trade)

            in_position  = True
            position_end = exit_time

            # Liquidation check
            if equity <= 0:
                log.warning("Equity wiped out at %s. Stopping simulation.", sig_time)
                break

        log.info("Backtest complete: %d trades | final equity=%.2f USDT", len(trades), equity)
        equity_curve = _build_equity_curve(trades, self.cfg.initial_equity)
        return trades, equity_curve


# ─── Analytics ────────────────────────────────────────────────────────────────

def _build_equity_curve(trades: list[Trade], initial_equity: float) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["equity", "drawdown"])

    rows = []
    eq = initial_equity
    peak = initial_equity
    for t in trades:
        eq += t.pnl
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        rows.append({"time": t.exit_time, "equity": eq, "drawdown": dd})

    curve = pd.DataFrame(rows).set_index("time")
    return curve


def print_stats(trades: list[Trade], equity_curve: pd.DataFrame, initial_equity: float) -> None:
    """Print a comprehensive backtest report."""
    if not trades:
        print("No trades executed.")
        return

    pnls    = np.array([t.pnl for t in trades])
    wins    = pnls[pnls > 0]
    losses  = pnls[pnls <= 0]
    reasons = [t.exit_reason for t in trades]

    final_eq   = equity_curve["equity"].iloc[-1]
    total_ret  = (final_eq - initial_equity) / initial_equity * 100
    max_dd     = equity_curve["drawdown"].min() * 100
    win_rate   = len(wins) / len(pnls) * 100

    tp_count  = reasons.count("tp")
    sl_count  = reasons.count("sl")
    to_count  = reasons.count("timeout")

    # Annualised return
    days = (trades[-1].exit_time - trades[0].entry_time).days
    ann_ret = ((final_eq / initial_equity) ** (365 / max(days, 1)) - 1) * 100

    # Profit factor
    gross_win  = wins.sum()  if len(wins)   > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    pf = gross_win / gross_loss

    # Avg hold time
    hold_times = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]

    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)
    print(f"  Period         : {trades[0].entry_time.date()} → {trades[-1].exit_time.date()}")
    print(f"  Total trades   : {len(trades)}")
    print(f"  Win rate       : {win_rate:.1f}%")
    print(f"  Exit breakdown : TP={tp_count}  SL={sl_count}  Timeout={to_count}")
    print()
    print(f"  Avg PnL/trade  : {pnls.mean():.2f}U")
    print(f"  Avg win        : {wins.mean():.2f}U" if len(wins) > 0 else "  Avg win  : n/a")
    print(f"  Avg loss       : {losses.mean():.2f}U" if len(losses) > 0 else "  Avg loss : n/a")
    print(f"  Profit factor  : {pf:.2f}")
    print(f"  Avg hold time  : {np.mean(hold_times):.1f} min")
    print()
    print(f"  Initial equity : {initial_equity:.2f}U")
    print(f"  Final equity   : {final_eq:.2f}U")
    print(f"  Total return   : {total_ret:+.1f}%")
    print(f"  Annualised ret : {ann_ret:+.1f}%")
    print(f"  Max drawdown   : {max_dd:.1f}%")
    print("=" * 60 + "\n")
