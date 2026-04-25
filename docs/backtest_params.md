# Backtest Selected Parameters

> 最后更新：2026-04-25（v1 重训练后，bug 修复版本）

## Long model (`models/v1/long/`)
- TP=0.008, SL=0.008
- Win rate 79.1% | PF 3.72 | Max DD -6.3% | Avg PnL 18.08U
- Backtest window: 2025-01-01 → 2025-12-31

## Short model (`models/v1/short/`)
- TP=0.008, SL=0.010
- Win rate 78.5% | PF 4.06 | Max DD -11.9% | Avg PnL 18.93U
- Backtest window: 2025-01-01 → 2025-12-31
- **注意**：全时段（2021起）回测 SHORT 会在 2021 年牛市初期爆仓，建议仅在震荡/下行市场使用

## Common settings
- threshold: 0.60
- horizon: 30 bars (150min)
- leverage: 75x, margin_pct: 25%, fee: 0.04% taker, slippage: 0.03%

## Bug fixes applied (vs old model)
- HORIZON: 15 → 30（标签窗口与回测持仓窗口对齐）
- y_max/y_min: 仅正样本训练 → 全样本训练（消除分布外预测）
- 回测入场价: price_close → next_bar_open（消除 look-ahead bias）
- CTI 特征: long/short 各自计算（修复推理时 SHORT 用 LONG 特征的 bug）

## Note
- 旧模型（修复前）备份在 `models/backups/v1_20260422_pre-bugfix/`
- re-run grid when model is retrained or features change
