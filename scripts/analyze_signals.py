"""
Comprehensive live signal log analyzer.

Parses one or two signal log files, fetches Binance klines, and produces:
  1. Signal summary (count, rate, prob distribution)
  2. Outcome analysis (TP/SL/OPEN, win rate, R-ratio, simulated PnL)
  3. Pattern breakdown (win rate per pattern type)
  4. Time-of-day distribution (signal frequency + win rate by hour)
  5. Missed moves analysis (big price moves where model stayed silent)
  6. Version comparison (v1 vs v1.0.1 in overlapping window)

Usage
-----
  # Single log
  python scripts/analyze_signals.py --log data/signal_v1.log

  # Compare two versions
  python scripts/analyze_signals.py --log data/signal_v1.log --compare data/signal_v1.0.1.log

  # Custom horizon
  python scripts/analyze_signals.py --log data/signal_v1.log --horizon 6
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Binance fetch ─────────────────────────────────────────────────────────────

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _fetch_klines_raw(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    all_bars = []
    t = start_ms
    base = "https://fapi.binance.com/fapi/v1/klines"
    while t <= end_ms:
        url = f"{base}?symbol={symbol}&interval={interval}&startTime={t}&limit=1000"
        with urllib.request.urlopen(url, timeout=20, context=_SSL_CTX) as resp:
            data = json.loads(resp.read())
        if not data:
            break
        all_bars.extend(data)
        last_ts = data[-1][0]
        if last_ts >= end_ms:
            break
        t = last_ts + _interval_ms(interval)
    return all_bars


def _interval_ms(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]


def fetch_5m_klines(start_utc: datetime, end_utc: datetime) -> dict[int, dict]:
    bars = _fetch_klines_raw(
        "ETHUSDT", "5m",
        int(start_utc.timestamp() * 1000),
        int(end_utc.timestamp() * 1000),
    )
    return {b[0]: {"o": float(b[1]), "h": float(b[2]),
                   "l": float(b[3]), "c": float(b[4])} for b in bars}


def fetch_1h_klines(start_utc: datetime, end_utc: datetime) -> list[dict]:
    bars = _fetch_klines_raw(
        "ETHUSDT", "1h",
        int(start_utc.timestamp() * 1000),
        int(end_utc.timestamp() * 1000),
    )
    return [{"ts": b[0], "o": float(b[1]), "h": float(b[2]),
             "l": float(b[3]), "c": float(b[4])} for b in bars]


def parse_log(path: str) -> tuple[list[dict], list[dict]]:
    """
    Parse signal log.  Returns (signals, all_bars).
    signals   — list of FIRE signal dicts
    all_bars  — list of every 5m bar with long_prob/short_prob
    """
    signals: list[dict] = []
    all_bars: list[dict] = []
    current_ts: datetime | None = None
    current_price: float | None = None
    pending_probs: dict = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            # ── bar header ────────────────────────────────────────────────────
            if (line.startswith("2026-") or line.startswith("2025-")) \
                    and "CST" in line and "price=" in line:
                parts = line.split()
                date_str = parts[0]
                open_time_str  = parts[1]
                close_time_str = parts[3]
                price = float(parts[-1].split("=")[1])
                close_dt_cst = datetime.strptime(
                    f"{date_str} {close_time_str}", "%Y-%m-%d %H:%M")
                if close_time_str < open_time_str:
                    close_dt_cst += timedelta(days=1)
                current_ts = (close_dt_cst.replace(tzinfo=timezone.utc)
                              - timedelta(hours=8))
                current_price = price
                pending_probs.clear()
            elif current_ts is None:
                continue
            # ── FIRE lines ────────────────────────────────────────────────────
            elif "LONG  FIRE" in line or "SHORT FIRE" in line:
                direction = "LONG" if "LONG  FIRE" in line else "SHORT"
                arrows = line.split("→")
                prob_part = line.split("prob=")[1].split()[0]
                tp_price = float(arrows[1].strip().split()[0])
                sl_price = float(arrows[2].strip().split()[0])
                signals.append({
                    "ts_utc": current_ts,
                    "direction": direction,
                    "entry_price": current_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "prob": float(prob_part),
                    "pattern": None,
                })
            # ── Pattern line ──────────────────────────────────────────────────
            elif line.strip().startswith("Pattern:") and signals:
                if signals[-1].get("pattern") is None:
                    signals[-1]["pattern"] = line.strip().split("Pattern:")[1].strip()
            # ── All-bar prob lines ────────────────────────────────────────────
            elif line.strip().startswith("LONG") and "prob=" in line:
                pending_probs["long"] = float(line.strip().split("prob=")[1].split()[0])
            elif line.strip().startswith("SHORT") and "prob=" in line:
                sp = float(line.strip().split("prob=")[1].split()[0])
                pending_probs["short"] = sp
                # Both sides parsed → emit bar record
                if "long" in pending_probs:
                    all_bars.append({
                        "ts_utc": current_ts,
                        "price": current_price,
                        "long_prob": pending_probs["long"],
                        "short_prob": pending_probs["short"],
                    })
                    pending_probs.clear()

    for sig in signals:
        if sig.get("pattern") is None:
            sig["pattern"] = "NONE"
    return signals, all_bars


# ── Outcome evaluation ────────────────────────────────────────────────────────

def evaluate(signals: list[dict], bar_map: dict[int, dict], horizon: int) -> list[dict]:
    results = []
    for sig in signals:
        entry_ms = int(sig["ts_utc"].timestamp() * 1000)
        direction = sig["direction"]
        tp = sig["tp_price"]
        sl = sig["sl_price"]
        outcome = "OPEN"
        exit_bar = None
        for i in range(1, horizon + 1):
            bar = bar_map.get(entry_ms + i * 300_000)
            if not bar:
                continue
            h, l = bar["h"], bar["l"]
            if direction == "LONG":
                if l <= sl:
                    outcome, exit_bar = "SL", i
                    break
                if h >= tp:
                    outcome, exit_bar = "TP", i
                    break
            else:
                if h >= sl:
                    outcome, exit_bar = "SL", i
                    break
                if l <= tp:
                    outcome, exit_bar = "TP", i
                    break
        results.append({**sig, "outcome": outcome, "exit_bar": exit_bar})
    return results


# ── Missed moves ──────────────────────────────────────────────────────────────

def find_missed_moves(all_bars: list[dict], bars_1h: list[dict],
                      threshold: float = 0.015) -> list[dict]:
    if not bars_1h or not all_bars:
        return []
    bar_lookup = {int(b["ts_utc"].timestamp() * 1000): b for b in all_bars}
    missed = []
    for bar in bars_1h:
        roc = (bar["c"] - bar["o"]) / bar["o"]
        if abs(roc) < threshold:
            continue
        direction = "DOWN" if roc < 0 else "UP"
        probs = []
        for offset in range(13):
            b5 = bar_lookup.get(bar["ts"] + offset * 300_000)
            if b5:
                p = b5["short_prob"] if direction == "DOWN" else b5["long_prob"]
                probs.append(p)
        if not probs:
            continue
        ts_cst = datetime.fromtimestamp(bar["ts"] / 1000, tz=timezone.utc) + timedelta(hours=8)
        missed.append({
            "ts_cst": ts_cst,
            "direction": direction,
            "roc_pct": roc * 100,
            "open": bar["o"],
            "close": bar["c"],
            "max_prob": max(probs),
            "avg_prob": sum(probs) / len(probs),
        })
    return sorted(missed, key=lambda x: abs(x["roc_pct"]), reverse=True)


# ── Prob histogram ────────────────────────────────────────────────────────────

def prob_histogram(probs: list[float], buckets: int = 10) -> str:
    if not probs:
        return "(no data)"
    step = 1.0 / buckets
    counts = [0] * buckets
    for p in probs:
        counts[min(int(p / step), buckets - 1)] += 1
    total = len(probs)
    lines = []
    for i, c in enumerate(counts):
        lo, hi = i * step, (i + 1) * step
        bar = "█" * int(c / total * 40)
        lines.append(f"  {lo:.1f}-{hi:.1f}  {bar:<40} {c:4d} ({c/total*100:4.1f}%)")
    return "\n".join(lines)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _win_str(tp: int, sl: int) -> str:
    total = tp + sl
    return "n/a" if total == 0 else f"{tp/total*100:.1f}% ({tp}/{total})"


def _pnl(results: list[dict]) -> float:
    total = 0.0
    for r in results:
        ep = r["entry_price"]
        if ep <= 0:
            continue
        sl_pct = abs(r["sl_price"] - ep) / ep
        tp_pct = abs(r["tp_price"] - ep) / ep
        if r["outcome"] == "TP":
            total += tp_pct * 100
        elif r["outcome"] == "SL":
            total -= sl_pct * 100
    return total


def print_report(label: str, signals: list[dict], results: list[dict],
                 all_bars: list[dict], missed: list[dict], horizon: int) -> None:
    cst = timedelta(hours=8)
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)

    # 1. Session overview
    if all_bars:
        first, last = all_bars[0]["ts_utc"], all_bars[-1]["ts_utc"]
        dur_h = (last - first).total_seconds() / 3600
        fire_rate = len(signals) / len(all_bars) * 100
        long_f  = sum(1 for s in signals if s["direction"] == "LONG")
        short_f = sum(1 for s in signals if s["direction"] == "SHORT")
        print(f"\n[1] SESSION OVERVIEW")
        print(f"  Period       : {(first+cst).strftime('%Y-%m-%d %H:%M')} → "
              f"{(last+cst).strftime('%Y-%m-%d %H:%M')} CST  ({dur_h:.1f}h)")
        print(f"  Total bars   : {len(all_bars)}")
        print(f"  FIRE signals : {len(signals)}  ({fire_rate:.2f}% of bars)  "
              f"LONG={long_f}  SHORT={short_f}")

    # 2. Prob distribution
    if all_bars:
        lp = [b["long_prob"]  for b in all_bars]
        sp = [b["short_prob"] for b in all_bars]
        print(f"\n[2] PROBABILITY DISTRIBUTION (all {len(all_bars)} bars)")
        print(f"  LONG  — max={max(lp):.4f}  avg={sum(lp)/len(lp):.4f}  "
              f"p>0.5:{sum(1 for p in lp if p>0.5)}  p>0.6:{sum(1 for p in lp if p>0.6)}")
        print(f"  SHORT — max={max(sp):.4f}  avg={sum(sp)/len(sp):.4f}  "
              f"p>0.5:{sum(1 for p in sp if p>0.5)}  p>0.6:{sum(1 for p in sp if p>0.6)}")
        print(f"\n  LONG prob histogram:")
        print(prob_histogram(lp))
        print(f"\n  SHORT prob histogram:")
        print(prob_histogram(sp))

    # 3. Outcome analysis
    tp_all = sum(1 for r in results if r["outcome"] == "TP")
    sl_all = sum(1 for r in results if r["outcome"] == "SL")
    op_all = sum(1 for r in results if r["outcome"] == "OPEN")
    long_r  = [r for r in results if r["direction"] == "LONG"]
    short_r = [r for r in results if r["direction"] == "SHORT"]
    l_tp = sum(1 for r in long_r  if r["outcome"] == "TP")
    l_sl = sum(1 for r in long_r  if r["outcome"] == "SL")
    s_tp = sum(1 for r in short_r if r["outcome"] == "TP")
    s_sl = sum(1 for r in short_r if r["outcome"] == "SL")
    tp_bars = [r["exit_bar"] for r in results if r["outcome"] == "TP" and r["exit_bar"]]
    sl_bars = [r["exit_bar"] for r in results if r["outcome"] == "SL" and r["exit_bar"]]
    r_ratios = []
    for s in signals:
        ep = s["entry_price"]
        if ep <= 0: continue
        sl_pct = abs(s["sl_price"] - ep) / ep
        tp_pct = abs(s["tp_price"] - ep) / ep
        if sl_pct > 0:
            r_ratios.append(tp_pct / sl_pct)

    print(f"\n[3] OUTCOME ANALYSIS  (horizon={horizon} bars = {horizon*5} min)")
    print(f"  TP / SL / OPEN : {tp_all} / {sl_all} / {op_all}")
    print(f"  Win rate       : {_win_str(tp_all, sl_all)}")
    print(f"  LONG  win rate : {_win_str(l_tp, l_sl)}")
    print(f"  SHORT win rate : {_win_str(s_tp, s_sl)}")
    if tp_bars:
        print(f"  Avg TP exit    : bar +{sum(tp_bars)/len(tp_bars):.1f}  "
              f"({sum(tp_bars)/len(tp_bars)*5:.0f} min)")
    if sl_bars:
        print(f"  Avg SL exit    : bar +{sum(sl_bars)/len(sl_bars):.1f}  "
              f"({sum(sl_bars)/len(sl_bars)*5:.0f} min)")
    if r_ratios:
        print(f"  Avg R-ratio    : {sum(r_ratios)/len(r_ratios):.2f}x  "
              f"(min={min(r_ratios):.2f}  max={max(r_ratios):.2f})")
    print(f"  Simulated PnL  : {_pnl(results):+.2f}R  (1R = SL% per trade)")

    # 4. Pattern breakdown
    pat_stats: dict[str, dict] = defaultdict(lambda: {"tp": 0, "sl": 0, "open": 0})
    for r in results:
        pat_stats[r.get("pattern") or "NONE"][r["outcome"].lower()] += 1
    print(f"\n[4] PATTERN BREAKDOWN")
    print(f"  {'Pattern':10}  {'TP':>4}  {'SL':>4}  {'OPEN':>4}  {'Win%':>12}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*12}")
    for pat, st in sorted(pat_stats.items()):
        print(f"  {pat:10}  {st['tp']:>4}  {st['sl']:>4}  {st['open']:>4}  "
              f"{_win_str(st['tp'], st['sl']):>12}")

    # 5. Time-of-day
    hour_sig: dict[int, list] = defaultdict(list)
    for r in results:
        hour_sig[(r["ts_utc"] + cst).hour].append(r)
    if hour_sig:
        print(f"\n[5] TIME-OF-DAY ANALYSIS  (CST)")
        print(f"  {'Hour':>4}  {'Count':>5}  {'TP':>4}  {'SL':>4}  {'Win%':>12}")
        print(f"  {'-'*4}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*12}")
        for h in sorted(hour_sig):
            rs = hour_sig[h]
            tp = sum(1 for r in rs if r["outcome"] == "TP")
            sl = sum(1 for r in rs if r["outcome"] == "SL")
            print(f"  {h:>4}  {len(rs):>5}  {tp:>4}  {sl:>4}  {_win_str(tp, sl):>12}")

    # 6. Missed moves
    if missed:
        print(f"\n[6] MISSED MOVES  (1h moves where model prob stayed low)")
        print(f"  {'Time(CST)':>16}  {'Dir':5}  {'Move%':>6}  {'Open':>8}  "
              f"{'Close':>8}  {'MaxProb':>7}  {'AvgProb':>7}")
        print(f"  {'-'*72}")
        for m in missed[:20]:
            flag = "  ⚠️ blind" if m["max_prob"] < 0.3 else ""
            print(f"  {m['ts_cst'].strftime('%m-%d %H:%M'):>16}  {m['direction']:5}  "
                  f"{m['roc_pct']:>+6.2f}%  {m['open']:>8.2f}  {m['close']:>8.2f}  "
                  f"{m['max_prob']:>7.4f}  {m['avg_prob']:>7.4f}{flag}")

    # 7. Signal detail
    print(f"\n[7] SIGNAL DETAIL")
    print(f"  {'Time(CST)':>16}  {'Dir':5}  {'Prob':>6}  {'Entry':>8}  "
          f"{'R':>5}  {'Pattern':10}  {'Out':4}  {'Bar':>4}")
    print(f"  {'-'*72}")
    for r in results:
        ts_cst = r["ts_utc"] + cst
        ep = r["entry_price"]
        sl_pct = abs(r["sl_price"] - ep) / ep if ep > 0 else 0
        tp_pct = abs(r["tp_price"] - ep) / ep if ep > 0 else 0
        rr = f"{tp_pct/sl_pct:.1f}x" if sl_pct > 0 else "n/a"
        bar_info = f"+{r['exit_bar']:2d}" if r["exit_bar"] else "  --"
        pat = (r.get("pattern") or "NONE")[:10]
        print(f"  {ts_cst.strftime('%m-%d %H:%M'):>16}  {r['direction']:5}  "
              f"{r['prob']:.4f}  {ep:>8.2f}  {rr:>5}  {pat:10}  {r['outcome']:4}  {bar_info}")


def print_comparison(label_a: str, res_a: list[dict], bars_a: list[dict],
                     label_b: str, res_b: list[dict], bars_b: list[dict]) -> None:
    if not bars_a or not bars_b:
        return
    start = max(bars_a[0]["ts_utc"], bars_b[0]["ts_utc"])
    end   = min(bars_a[-1]["ts_utc"], bars_b[-1]["ts_utc"])
    if start >= end:
        print("\n[VERSION COMPARISON]  No overlapping time window.")
        return
    cst = timedelta(hours=8)

    def _filt(results, bars):
        return ([r for r in results if start <= r["ts_utc"] <= end],
                [b for b in bars     if start <= b["ts_utc"] <= end])

    ra, ba = _filt(res_a, bars_a)
    rb, bb = _filt(res_b, bars_b)
    tp_a = sum(1 for r in ra if r["outcome"] == "TP")
    sl_a = sum(1 for r in ra if r["outcome"] == "SL")
    tp_b = sum(1 for r in rb if r["outcome"] == "TP")
    sl_b = sum(1 for r in rb if r["outcome"] == "SL")
    lp_a = [b["long_prob"]  for b in ba] or [0]
    sp_a = [b["short_prob"] for b in ba] or [0]
    lp_b = [b["long_prob"]  for b in bb] or [0]
    sp_b = [b["short_prob"] for b in bb] or [0]

    print(f"\n{'='*72}")
    print(f"  VERSION COMPARISON")
    print(f"{'='*72}")
    print(f"  Overlap: {(start+cst).strftime('%Y-%m-%d %H:%M')} → "
          f"{(end+cst).strftime('%Y-%m-%d %H:%M')} CST")
    print()
    print(f"  {'Metric':32}  {label_a:>14}  {label_b:>14}")
    print(f"  {'-'*32}  {'-'*14}  {'-'*14}")

    def row(name, va, vb):
        print(f"  {name:32}  {va:>14}  {vb:>14}")

    row("Bars checked",          str(len(ba)),                    str(len(bb)))
    row("FIRE signals",          str(len(ra)),                    str(len(rb)))
    row("Signal rate (%)",
        f"{len(ra)/len(ba)*100:.2f}%" if ba else "n/a",
        f"{len(rb)/len(bb)*100:.2f}%" if bb else "n/a")
    row("LONG fires",
        str(sum(1 for r in ra if r["direction"] == "LONG")),
        str(sum(1 for r in rb if r["direction"] == "LONG")))
    row("SHORT fires",
        str(sum(1 for r in ra if r["direction"] == "SHORT")),
        str(sum(1 for r in rb if r["direction"] == "SHORT")))
    row("Win rate",              _win_str(tp_a, sl_a),            _win_str(tp_b, sl_b))
    row("Simulated PnL (R)",     f"{_pnl(ra):+.2f}R",            f"{_pnl(rb):+.2f}R")
    row("Max LONG prob",         f"{max(lp_a):.4f}",              f"{max(lp_b):.4f}")
    row("Max SHORT prob",        f"{max(sp_a):.4f}",              f"{max(sp_b):.4f}")
    row("Avg LONG prob",         f"{sum(lp_a)/len(lp_a):.4f}",   f"{sum(lp_b)/len(lp_b):.4f}")
    row("Avg SHORT prob",        f"{sum(sp_a)/len(sp_a):.4f}",   f"{sum(sp_b)/len(sp_b):.4f}")
    row("p>0.5 LONG bars",
        str(sum(1 for p in lp_a if p > 0.5)),
        str(sum(1 for p in lp_b if p > 0.5)))
    row("p>0.5 SHORT bars",
        str(sum(1 for p in sp_a if p > 0.5)),
        str(sum(1 for p in sp_b if p > 0.5)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive signal log analyzer")
    parser.add_argument("--log",      required=True, help="Primary log file")
    parser.add_argument("--compare",  default=None,  help="Second log for version comparison")
    parser.add_argument("--horizon",  type=int, default=6,
                        help="Bars to wait for TP/SL (default=6=30min)")
    parser.add_argument("--move-threshold", type=float, default=0.015,
                        help="1h move threshold for missed-move analysis (default=1.5%%)")
    parser.add_argument("--no-missed-moves", action="store_true",
                        help="Skip missed moves (faster)")
    args = parser.parse_args()

    print(f"Parsing {args.log} ...", flush=True)
    signals_a, bars_a = parse_log(args.log)
    label_a = Path(args.log).stem
    print(f"  {len(signals_a)} FIRE signals, {len(bars_a)} bars", flush=True)

    signals_b, bars_b, label_b = [], [], ""
    if args.compare:
        print(f"Parsing {args.compare} ...", flush=True)
        signals_b, bars_b = parse_log(args.compare)
        label_b = Path(args.compare).stem
        print(f"  {len(signals_b)} FIRE signals, {len(bars_b)} bars", flush=True)

    all_bars_ts = [b["ts_utc"] for b in bars_a + bars_b]
    if not all_bars_ts:
        print("No bars found. Exiting.")
        sys.exit(1)

    fetch_start = min(all_bars_ts) + timedelta(minutes=5)
    fetch_end   = max(all_bars_ts) + timedelta(hours=3)

    print(f"Fetching 5m klines {fetch_start.strftime('%Y-%m-%d %H:%M')} → "
          f"{fetch_end.strftime('%Y-%m-%d %H:%M')} UTC ...", flush=True)
    bar_map_5m = fetch_5m_klines(fetch_start, fetch_end)
    print(f"  {len(bar_map_5m)} bars", flush=True)

    bars_1h: list[dict] = []
    if not args.no_missed_moves:
        print("Fetching 1h klines ...", flush=True)
        bars_1h = fetch_1h_klines(fetch_start - timedelta(hours=1), fetch_end)
        print(f"  {len(bars_1h)} 1h bars", flush=True)

    results_a = evaluate(signals_a, bar_map_5m, args.horizon)
    results_b = evaluate(signals_b, bar_map_5m, args.horizon) if signals_b else []

    ref_bars = bars_a if not bars_b else sorted(bars_a + bars_b,
                                                 key=lambda x: x["ts_utc"])
    missed = find_missed_moves(ref_bars, bars_1h, args.move_threshold)

    print_report(label_a, signals_a, results_a, bars_a, missed, args.horizon)
    if signals_b:
        print_report(label_b, signals_b, results_b, bars_b, missed, args.horizon)
        print_comparison(label_a, results_a, bars_a, label_b, results_b, bars_b)

    print(f"\n{'='*72}")
    print("Done.")


if __name__ == "__main__":
    main()
