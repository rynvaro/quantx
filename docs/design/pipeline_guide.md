# QuantX Pipeline Guide
## ETH/USDT 永续合约 Calm-Trigger/Spring 机器学习交易系统

> 本文档记录从零开始到可运行回测的全流程，包含每一步的目的、指令、注意事项和依赖关系。
> 最后更新：2026-04-25（v1 分支）

---

## 系统概览

```
原始K线数据
    ↓  Step 1: 下载数据
data/raw/
    ↓  Step 2: 特征工程
data/processed/v1/features_thr005_h30_{long|short}.parquet
    ↓  Step 3: 训练模型
models/v1/{long|short}/ (model_cls.json / model_max.json / model_min.json / calibrator.pkl)
    ↓  Step 4: 回测验证
回测报告 + equity_curve.csv
    ↓  Step 5: 实时信号
run_signal.py → LONG/SHORT 概率 + 形态编码输出
```

---

## Step 1 — 下载原始数据

### 目的
从 Binance 公开 API 下载 ETH/USDT 永续合约历史 K 线和资金费率，作为一切后续工作的原料。

### 对应文件
- **脚本**：`scripts/download_data.py`
- **源码**：`src/features/loader.py`
- **输出**：
  - `data/raw/klines/1m/` — 1分钟 K 线
  - `data/raw/klines/5m/` — 5分钟 K 线（回测 TP/SL 判断用）
  - `data/raw/klines/15m/` — 15分钟 K 线
  - `data/raw/klines/1h/` — 1小时 K 线
  - `data/raw/klines/4h/` — 4小时 K 线
  - `data/raw/funding_rate/` — 资金费率

### 执行指令
```bash
python scripts/download_data.py
```

### 注意事项
- 数据从 **2021-01-01** 开始下载至当前，约5年
- Binance 公开接口有频率限制，脚本内已做限速处理，勿修改并发参数
- 下载完成后建议用 `ls -lh data/raw/klines/5m/` 确认文件大小正常（5m 约 200–300MB）
- 若中断可重新运行，脚本支持断点续传（检测已有文件）

### 依赖
无（第一步）

---

## Step 2 — 特征工程

### 目的
将原始多周期 K 线合并，计算技术指标，生成标签（未来 30 分钟内是否上涨超过 0.5%），过滤异常数据，输出可直接用于训练的 parquet 文件。

### 对应文件
- **脚本**：`scripts/build_features.py`
- **源码模块**：
  - `src/features/loader.py` — 加载原始 K 线
  - `src/features/indicators.py` — 计算单周期技术指标（RSI、ATR、MACD、布林带、EMA 等）
  - `src/features/timeframe.py` — 多周期特征合并（1m/5m/15m/1h/4h 对齐到 5m 时间轴）
  - `src/features/labels.py` — 生成标签（y_class / y_max / y_min）
  - `src/features/pipeline.py` — 上述模块的编排入口
- **输出**：`data/processed/v1/features_thr005_h30_{long|short}.parquet`（约 334MB/方向）

### 执行指令
```bash
python scripts/build_features.py \
    --threshold 0.005 \
    --horizon   30
```

> 默认输出到 `data/processed/v1/`，无需指定路径。

参数说明：
- `--threshold 0.005`：标签阈值，即未来 30 分钟内最高价相对入场价涨幅 ≥ 0.5% 为正样本
- `--horizon 30`：标签时间窗口（分钟），对应 6 根 5m bar

### parquet 内容
| 列类型 | 数量 | 说明 |
|---|---|---|
| 技术指标特征 | 131 列 | RSI、ATR、MACD、BB、EMA 等，每个周期一套 |
| 标签 | 3 列 | `y_class`（0/1）、`y_max`（最大涨幅%）、`y_min`（最大跌幅%）|
| 辅助列 | 2 列 | `is_weekend`（周末过滤）、`is_black_swan`（黑天鹅过滤）|
| 时间列 | index | UTC 时间戳，5m 粒度 |

数据规模：390,024 行，正样本占比约 21.9%

### 注意事项
- **时间对齐**：1m/15m/1h/4h 的指标通过 forward-fill 对齐到 5m 时间轴，不存在 look-ahead
- **黑天鹅过滤**：LUNA 崩盘（2022-05）、FTX 崩盘（2022-11）等极端事件被标记并过滤，防止模型学到不可复现的特征
- **原始 OHLC 未存入 parquet**：回测时 TP/SL 判断需要单独加载 `data/raw/klines/5m/` 的原始数据（见 Step 4）

### 依赖
Step 1 完成（`data/raw/` 目录存在且数据完整）

---

## Step 3 — 模型训练

### 目的
用 Walk-Forward 验证评估模型可信度，然后用全量数据训练生产级模型并保存。

### 对应文件
- **脚本**：`scripts/train_model.py`
- **源码模块**：
  - `src/train/fold.py` — Walk-Forward fold 生成器
  - `src/train/model.py` — 3-head XGBoost 训练 + Platt Scaling 校准
  - `src/train/evaluate.py` — 每折指标计算 + 业务 PnL 估算
  - `src/train/pipeline.py` — 端到端编排（加载 → 过滤 → 训练 → 评估 → 全量重训 → 保存）
- **输出**（保存到 `models/v1/long/`，已加入 `.gitignore`）：
  - `model_cls.json` — XGBClassifier（分类头，判断方向）
  - `model_max.json` — XGBRegressor（预测未来最大涨幅）
  - `model_min.json` — XGBRegressor（预测未来最大跌幅）
  - `calibrator.pkl` — Platt Scaling 校准器（将原始概率映射为可信概率）
  - `feature_cols.txt` — 131 个特征列名（推理时必须按此顺序）

### 执行指令
```bash
# 默认自动使用 data/processed/v1/ 和 models/v1/
python scripts/train_model.py --direction long
python scripts/train_model.py --direction short
```

耗时约 **100–120 秒**（MacBook，M 系列芯片）

### 模型架构
```
输入：131 个技术指标特征

     ┌─────────────────────────────────┐
     │  XGBClassifier (model_cls)       │  → 原始概率 → Platt Scaling → 校准概率 [0,1]
     │  训练集：全部样本                │
     └─────────────────────────────────┘

     ┌─────────────────────────────────┐
     │  XGBRegressor (model_max)        │  → 预测未来最大涨幅（%）
     │  训练集：全部样本（已修复）      │
     └─────────────────────────────────┘

     ┌─────────────────────────────────┐
     │  XGBRegressor (model_min)        │  → 预测未来最大跌幅（%）
     │  训练集：全部样本（已修复）      │
     └─────────────────────────────────┘
```

### Walk-Forward 验证结构
```
数据总范围：2021-01 → 2026-03（约 5 年）
VAL 区间（最后 2 年）：2024-04 → 2026-03，切成 6 个折，每折 4 个月

Fold 1: 训练 2021-01~2024-03 │ 验证 2024-04~2024-07
Fold 2: 训练 2021-01~2024-07 │ 验证 2024-08~2024-11
Fold 3: 训练 2021-01~2024-11 │ 验证 2024-12~2025-03
Fold 4: 训练 2021-01~2025-03 │ 验证 2025-04~2025-07
Fold 5: 训练 2021-01~2025-07 │ 验证 2025-08~2025-11
Fold 6: 训练 2021-01~2025-11 │ 验证 2025-12~2026-03

最终生产模型：用 2021-01 → 2026-03 全量数据重新训练
```

注：训练集和验证集之间有 **5 天 gap**，防止标签泄漏（因为标签窗口是 30 分钟，5 天 gap 足够）。

### 验证结果（已完成的训练跑出来的）
```
AUC-ROC  mean=0.8686  std=0.0079   ← 6 折稳定，说明模型不是靠运气
AUC-PR   mean=0.6796  std=0.0220

业务指标（6 折均值）：
threshold=0.60: 期望 PnL=14.22U/笔, 精度=61.5%, 信号数/折≈4963  ← 推荐
threshold=0.55: 期望 PnL=12.36U/笔, 精度=56.9%, 信号数/折≈5826
threshold=0.65: 期望 PnL=16.06U/笔, 精度=66.0%, 信号数/折≈4166
```

### 注意事项
- **XGBoost 3.x 兼容性**：`early_stopping_rounds` 必须写在构造函数里，不能写在 `fit()` 里（已修复）
- 回归头（model_max / model_min）只在正样本上训练，负样本对它们没有意义
- `calibrator.pkl` 是在所有折的 OOF（Out-of-Fold）分数上拟合的，保证覆盖全时间范围
- `feature_cols.txt` 非常重要：推理时特征必须按完全一致的顺序排列

### 依赖
Step 2 完成（parquet 文件存在）

---

## Step 4 — 回测验证

### 目的
在 OOS（Out-of-Sample）历史数据上模拟真实交易，验证模型是否真的能盈利，量化实际 PnL、胜率、最大回撤等指标。

### 对应文件
- **脚本**：`scripts/run_backtest.py`
- **源码**：`src/backtest/engine.py`
- **输出**：`models/v1/long/equity_curve.csv`（权益曲线，可用 Excel/Python 绘图）

### 执行指令
```bash
# 标准回测（默认使用 data/processed/v1/ 和 models/v1/）
python scripts/run_backtest.py --direction long  --start 2025-01-01
python scripts/run_backtest.py --direction short --start 2025-01-01

# 测试不同 TP/SL
python scripts/run_backtest.py --direction long --start 2025-01-01 --tp 0.010 --sl 0.008

# 测试特定时间段
python scripts/run_backtest.py --direction long --start 2025-01-01 --end 2025-12-31
```

### 完整 CLI 参数
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | `data/processed/v1/features_thr005_h30_long.parquet` | features 文件路径 |
| `--models` | `models/v1/long/` | 模型目录 |
| `--start` | 无（全量） | 回测开始日期 |
| `--end` | 无（全量） | 回测结束日期 |
| `--threshold` | 0.60 | 信号概率阈值 |
| `--tp` | 0.008 | 止盈幅度（0.8% ETH 价格变动） |
| `--sl` | 0.008 | 止损幅度（0.8%） |
| `--equity` | 312.0 | 初始权益（USDT） |
| `--margin-pct` | 0.25 | 每笔保证金占初始权益的比例 |
| `--leverage` | 75.0 | 杠杆倍数 |

### 回测参数说明（资金模型）
```
初始权益：312U
每笔保证金：312 × 25% = 78U（固定，不随盈亏变化）
每笔名义仓位：78 × 75 = 5,850U
手续费：0.04% taker 每边（进出各一次）= 5,850 × 0.04% × 2 = 4.68U
滑点：0.03% 每边 = 5,850 × 0.03% × 2 = 3.51U
总成本/笔：约 8.19U
```

### 已跑出的回测结果（2025 全年）
```
【多头】TP=0.8%  SL=0.8%
总交易数：3,419 笔  胜率：79.1%
出场分布：TP=1662 (49%) / SL=334 (10%) / Timeout=1423 (42%)
平均 PnL/笔：+18.1U  Profit Factor：3.72
初始权益：312U → 最终权益：62,129U  最大回撤：-6.3%

【空头】TP=0.8%  SL=1.0%
总交易数：3,362 笔  胜率：78.5%
出场分布：TP=1742 (52%) / SL=273 (8%) / Timeout=1347 (40%)
平均 PnL/笔：+18.9U  Profit Factor：4.06
初始权益：312U → 最终权益：63,950U  最大回撤：-11.9%
注：空头在 2021 年牛市初期会爆仓，建议观察市场状态后使用
```

详细参数选定理由见 `docs/backtest_params.md`。

### 注意事项
- `--start 2024-04-01` 是关键——必须用 OOS 区间，不能用训练数据段来回测（会高估结果）
- 回测中每笔仓位是**固定 78U**，不随账户盈亏变化（保守基准，真实运营可改为动态仓位）
- 回测用到 `data/raw/klines/5m/` 的原始 OHLC 做 TP/SL 判断，该目录必须存在
- Timeout 47% 说明有近一半信号在 30 分钟内行情未走到 ±0.8%，属于正常现象

### 依赖
Step 3 完成（`models/long/` 目录存在且含所有模型文件）
Step 1 完成（`data/raw/klines/5m/` 存在，用于 TP/SL 模拟）

---

## Step 5 — 实时信号

### 目的
每隔 5 分钟（等 bar 收盘）拉取最新 K 线，实时计算特征，同时推理 long 和 short 两个模型，输出校准概率。超过 threshold 时打印信号并标记形态编码（L1/L2/S1-S4/NONE）。

### 对应文件
- **脚本**：`scripts/run_signal.py`
- **源码**：`src/live/fetcher.py` / `src/live/indicator.py` / `src/live/signal.py` / `src/live/pattern_matcher.py`

### 执行指令
```bash
# 后台守护（推荐生产使用）
make start-bg
make logs   # 查看日志
make stop   # 停止

# 前台运行（调试用）
make start

# 单次运行（快速调试）
python scripts/run_signal.py --once

# 自定义 threshold 或模型目录
python scripts/run_signal.py --threshold 0.65 \
    --long-model-dir models/v1/long \
    --short-model-dir models/v1/short
```

### 输出示例
```
2026-04-25 14:35 → 14:40 CST  price=2318.45
  LONG  ✓✓    prob=0.6523  tp=+0.920% → 2339.79  sl=-0.153% → 2314.90
        Pattern: L1
  SHORT --    prob=0.0289  tp=+0.903% → 2297.51  sl=-0.155% → 2322.05
──────────────────────────────────────────────────────────
  运行开始 2026-04-24 18:53:30 UTC
  已检查 142 bars  |  LONG 触发 1 次  |  SHORT 触发 0 次
  最近 LONG : 14:40 (0.65)
──────────────────────────────────────────────────────────
```

形态编码说明见 `docs/pattern_codes.md`，输出格式说明见 `docs/signal_output_format.md`。

### 注意事项
- 网络需能访问 `fapi.binance.com`（通过代理或直连）
- 两个模型的 `feature_cols.txt` 必须和实盘特征完全对齐（均为 131 列，不含 OHLC）
- 信号概率低（如 0.02-0.09）不代表方向相反，只代表当前 bar 无明确信号
- 形态标记（Pattern）仅在触发信号时显示，NONE 表示未匹配已知成功形态

### 依赖
Step 3 完成（`models/v1/long/` 和 `models/v1/short/` 均存在）

---

## 目录结构总览

```
quantx/
├── data/
│   ├── raw/
│   │   ├── klines/
│   │   │   ├── 1m/     ← Step 1 下载
│   │   │   ├── 5m/     ← Step 1 下载，Step 4 回测时用
│   │   │   ├── 15m/    ← Step 1 下载
│   │   │   ├── 1h/     ← Step 1 下载
│   │   │   └── 4h/     ← Step 1 下载
│   │   └── funding_rate/  ← Step 1 下载
│   └── processed/
│       └── v1/
│           ├── features_thr005_h30_long.parquet   ← Step 2 生成（334MB）
│           └── features_thr005_h30_short.parquet  ← Step 2 生成（334MB）
│
├── models/                ← Step 3 生成（已 gitignore）
│   ├── v1/
│   │   ├── long/
│   │   │   ├── model_cls.json
│   │   │   ├── model_max.json
│   │   │   ├── model_min.json
│   │   │   ├── calibrator.pkl
│   │   │   ├── feature_cols.txt
│   │   │   └── equity_curve.csv   ← Step 4 生成
│   │   └── short/             ← 同上
│   └── backups/               ← 旧版本备份
│
├── src/
│   ├── features/          ← Step 2 源码
│   │   ├── loader.py      — 加载原始 K 线
│   │   ├── indicators.py  — 技术指标计算
│   │   ├── timeframe.py   — 多周期特征合并
│   │   ├── labels.py      — 标签生成
│   │   └── pipeline.py    — 特征工程编排
│   ├── train/             ← Step 3 源码
│   │   ├── fold.py        — Walk-Forward fold 生成器
│   │   ├── model.py       — XGBoost 训练 + Platt Scaling
│   │   ├── evaluate.py    — 指标评估
│   │   └── pipeline.py    — 训练编排
│   └── backtest/          ← Step 4 源码
│       └── engine.py      — 回测引擎
│
├── scripts/
│   ├── download_data.py   ← Step 1 入口
│   ├── build_features.py  ← Step 2 入口
│   ├── train_model.py     ← Step 3 入口
│   ├── run_backtest.py    ← Step 4 入口
│   ├── run_signal.py      ← Step 5 入口（实时信号）
│   └── analyze_importance.py  ← 特征重要性分析
│
└── docs/
    └── pipeline_guide.md  ← 本文档
```

---

## 快速重跑（从零开始）

```bash
# 1. 下载数据（约 30–60 分钟）
python scripts/download_data.py

# 2. 生成特征（约 10–20 分钟，自动输出到 data/processed/v1/）
python scripts/build_features.py --direction long  --threshold 0.005 --horizon 30
python scripts/build_features.py --direction short --threshold 0.005 --horizon 30

# 3. 训练模型（约 2 分钟/方向，自动输出到 models/v1/）
python scripts/train_model.py --direction long
python scripts/train_model.py --direction short

# 4. 回测验证（约 10–15 秒/方向）
python scripts/run_backtest.py --direction long  --start 2025-01-01
python scripts/run_backtest.py --direction short --start 2025-01-01

# 5. 实时信号（每 5 分钟输出一次 LONG/SHORT 概率 + 形态标记）
make start-bg
make logs
```

---

## 关键设计决策记录

| 决策 | 原因 |
|---|---|
| 标签阈值 0.5%，时间窗口 30min | 覆盖手续费（约 0.14%）后仍有足够盈利空间，窗口太长会产生太多假信号 |
| 3-head 模型（分类 + 回归 × 2）| 分类头判断方向，回归头预测目标价位，信息更丰富 |
| 回归头全样本训练（已修复）| 仅正样本训练导致分布外预测，推理时对负样本预测失准 |
| Walk-Forward 6 折验证 | 覆盖牛市/熊市/震荡多种市场状态，AUC std=0.008 表明模型稳健 |
| Platt Scaling 校准 | 原始 XGBoost 概率值不准确，校准后信号 threshold 才有业务含义 |
| 过滤周末 + 黑天鹅 | 周末流动性差模式不同；黑天鹅事件不可重现，让模型学到它们反而有害 |
| 回测固定仓位 78U | 防止复利效应掩盖模型真实表现，保守基准更可信 |
| 回测用 next_bar_open 入场（已修复）| 消除 look-ahead bias，信号 bar 收盘后实际只能在下一根 bar 开盘成交 |
| 形态识别（Pattern Matching）| 仅标记，不过滤，积累数据验证哪些形态真正有效 |

---

## 下一步待办

- [x] 把 5m OHLC（open/high/low/close）写入 parquet，简化回测 IO
- [x] 用不同 TP/SL 参数组合跑对比回测，选最优参数（见 `docs/backtest_params.md`）
- [x] 实时信号脚本（拉 Binance 实时 K 线 → 计算指标 → 输出 LONG/SHORT 概率）
- [x] 空头模型支持（build_features / train_model / run_backtest / run_signal 全链路）
- [ ] 纸交易验证（2–4 周，对比信号频率和方向与回测一致性）
- [ ] 日止损风控（当日亏损 > 3×78U = 234U 时停止交易）
