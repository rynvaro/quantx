# Backtest Selected Parameters

## Long model (`models/long/`)
- TP=0.008, SL=0.008
- Win rate 78.4% | PF 3.69 | Max DD -20.5% | Avg PnL 17.09U
- Reason: SL=0.006 caused -53% DD (排除); 在 SL=0.008 组里 TP=0.008 胜率+PF 双最高

## Short model (`models/short/`)
- TP=0.008, SL=0.010
- Win rate 79.7% | PF 4.59 | Max DD -7.4% | Avg PnL 19.19U
- Reason: 所有组合 PF 最高; short DD 整体远低于 long，SL 略宽(0.010)可接受

## Common settings
- threshold: 0.60
- horizon: 6 bars (30min)
- leverage: 75x, margin_pct: 25%, fee: 0.04% taker, slippage: 0.03%
- backtest window: 2024-04-01 → (latest data)

## Grid range tested (both directions)
- TP: 0.008, 0.010, 0.012
- SL: 0.006/0.008 (long), 0.008/0.010 (short)
- Note: re-run grid when model is retrained or features change
