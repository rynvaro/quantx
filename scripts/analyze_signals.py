"""
Analyze live signal log: fetch Binance klines and evaluate TP/SL outcomes.
Usage: python3 scripts/analyze_signals.py [--log data/signal.log] [--horizon 30]
"""
import argparse
import sys
from datetime import datetime, timezone, timedelta
import urllib.request
import json


def parse_log(path):
    signals = []
    current_ts = None
    current_price = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("2026-") and "CST" in line and "price=" in line:
                parts = line.split()
                date_str = parts[0]
                open_time_str = parts[1]
                close_time_str = parts[3]
                price = float(parts[-1].split("=")[1])
                close_dt_cst = datetime.strptime(f"{date_str} {close_time_str}", "%Y-%m-%d %H:%M")
                open_dt_cst  = datetime.strptime(f"{date_str} {open_time_str}",  "%Y-%m-%d %H:%M")
                if close_time_str < open_time_str:
                    close_dt_cst += timedelta(days=1)
                close_dt_utc = close_dt_cst.replace(tzinfo=timezone.utc) - timedelta(hours=8)
                current_ts = close_dt_utc
                current_price = price
            elif ("LONG  FIRE" in line or "SHORT FIRE" in line) and current_ts:
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
                })
    return signals


def fetch_klines(start_utc, end_utc):
    """Fetch 5m ETHUSDT klines using urllib (no requests dependency)."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms   = int(end_utc.timestamp() * 1000)
    all_bars = []
    t = start_ms
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    while t <= end_ms:
        url = f"{base_url}?symbol=ETHUSDT&interval=5m&startTime={t}&limit=1000"
        with urllib.request.urlopen(url, timeout=20, context=ctx) as resp:
            data = json.loads(resp.read())
        if not data:
            break
        all_bars.extend(data)
        t = data[-1][0] + 5 * 60 * 1000
        if data[-1][0] >= end_ms:
            break
    return all_bars


def evaluate(signals, all_bars, horizon):
    bar_map = {}
    for b in all_bars:
        bar_map[b[0]] = {"h": float(b[2]), "l": float(b[3])}

    results = []
    for sig in signals:
        entry_ms = int(sig["ts_utc"].timestamp() * 1000)
        direction = sig["direction"]
        tp = sig["tp_price"]
        sl = sig["sl_price"]
        outcome = "OPEN"
        exit_bar = None
        for i in range(1, horizon + 1):
            check_ms = entry_ms + i * 5 * 60 * 1000
            bar = bar_map.get(check_ms)
            if not bar:
                continue
            h, l = bar["h"], bar["l"]
            if direction == "LONG":
                # Check SL first (conservative — if both hit in same bar, SL wins)
                if l <= sl:
                    outcome = "SL"
                    exit_bar = i
                    break
                if h >= tp:
                    outcome = "TP"
                    exit_bar = i
                    break
            else:  # SHORT
                if h >= sl:
                    outcome = "SL"
                    exit_bar = i
                    break
                if l <= tp:
                    outcome = "TP"
                    exit_bar = i
                    break
        results.append({**sig, "outcome": outcome, "exit_bar": exit_bar})
    return results


def print_report(results, horizon):
    cst = timedelta(hours=8)
    tp_count   = sum(1 for r in results if r["outcome"] == "TP")
    sl_count   = sum(1 for r in results if r["outcome"] == "SL")
    open_count = sum(1 for r in results if r["outcome"] == "OPEN")
    resolved   = tp_count + sl_count

    long_r  = [r for r in results if r["direction"] == "LONG"]
    short_r = [r for r in results if r["direction"] == "SHORT"]
    long_tp  = sum(1 for r in long_r  if r["outcome"] == "TP")
    long_sl  = sum(1 for r in long_r  if r["outcome"] == "SL")
    short_tp = sum(1 for r in short_r if r["outcome"] == "TP")
    short_sl = sum(1 for r in short_r if r["outcome"] == "SL")

    print("=" * 70)
    print(f"SIGNAL OUTCOME ANALYSIS  (horizon={horizon} bars = {horizon*5} min)")
    print("=" * 70)
    print(f"Total signals : {len(results)}")
    print(f"  TP hit      : {tp_count}")
    print(f"  SL hit      : {sl_count}")
    print(f"  Still open  : {open_count}")
    if resolved:
        print(f"Win rate      : {tp_count/resolved*100:.1f}%  ({tp_count}/{resolved} resolved)")
    print(f"LONG  TP/SL   : {long_tp}/{long_sl}  ({long_tp/(long_tp+long_sl)*100:.1f}% win)" if (long_tp+long_sl) else "LONG  TP/SL   : 0/0")
    print(f"SHORT TP/SL   : {short_tp}/{short_sl}  ({short_tp/(short_tp+short_sl)*100:.1f}% win)" if (short_tp+short_sl) else "SHORT TP/SL   : 0/0")

    # Avg exit bar for TP and SL
    tp_bars = [r["exit_bar"] for r in results if r["outcome"] == "TP" and r["exit_bar"]]
    sl_bars = [r["exit_bar"] for r in results if r["outcome"] == "SL" and r["exit_bar"]]
    if tp_bars:
        print(f"Avg TP exit   : bar +{sum(tp_bars)/len(tp_bars):.1f}  ({sum(tp_bars)/len(tp_bars)*5:.0f} min)")
    if sl_bars:
        print(f"Avg SL exit   : bar +{sum(sl_bars)/len(sl_bars):.1f}  ({sum(sl_bars)/len(sl_bars)*5:.0f} min)")

    print("\n--- DETAIL ---")
    print(f"{'Time(CST)':>16}  {'Dir':5}  {'Prob':>6}  {'Entry':>8}  {'TP':>8}  {'SL':>8}  {'Out':4}  {'Bar':>4}")
    print("-" * 70)
    for r in results:
        ts_cst = r["ts_utc"] + cst
        bar_info = f"+{r['exit_bar']:2d}" if r["exit_bar"] else "  --"
        print(f"{ts_cst.strftime('%m-%d %H:%M'):>16}  {r['direction']:5}  {r['prob']:.4f}  {r['entry_price']:>8.2f}  {r['tp_price']:>8.2f}  {r['sl_price']:>8.2f}  {r['outcome']:4}  {bar_info}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/signal.log")
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    print(f"Parsing {args.log}...", flush=True)
    signals = parse_log(args.log)
    print(f"Found {len(signals)} FIRE signals", flush=True)

    start = signals[0]["ts_utc"] + timedelta(minutes=5)
    end   = signals[-1]["ts_utc"] + timedelta(hours=3)
    print(f"Fetching klines {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} UTC...", flush=True)
    bars = fetch_klines(start, end)
    print(f"Fetched {len(bars)} bars", flush=True)

    results = evaluate(signals, bars, args.horizon)
    print_report(results, args.horizon)


if __name__ == "__main__":
    main()
