# v1 → v2 优化计划

## 背景

v1 模型于 2026-04-19/20 上线，运行 26h，产生 43 个 FIRE 信号，实盘胜率 **22%**（TP=9, SL=32）。  
v2 在 v1 基础上添加了 EMA20/60 趋势过滤器，2026-04-21 运行，产生 17 个 FIRE 信号，实盘胜率 **5.9%**（TP=1, SL=16）。

---

## v1 核心诊断

### 根本问题：模型预测的是波动率，不是方向

Feature importance 分析（`models/long/model_cls.json`）显示：

| 排名 | 特征 | 权重 | 类型 |
|------|------|------|------|
| #1 | `bb_width_1m` | 9.5% | 波动率 |
| #2 | `atr14_1m` | 8.9% | 波动率 |
| #3 | `roc1_15m` | 4.2% | 方向（仅15m） |
| #4 | `bb_width_5m` | 4.2% | 波动率 |

- 前两名均为波动率指标，合计权重 **19%**
- 最接近方向性的 `roc1_15m` 权重仅 4%，且只有 15m 尺度
- 完全没有 1h/4h 尺度的净收益率特征
- SHORT 模型的 `hour_of_day` 进入 top20，说明方向学习能力不足，退而用时段做辅助判断

**结果**：高波动时 LONG 和 SHORT 同时 fire，连续触发 8-9 根 bar，全部止损。模型并不知道方向，只知道"现在在震荡"。

### 次要问题

| 问题 | 影响程度 | 说明 |
|------|----------|------|
| CTI 特征 train/inference mismatch | 低（CTI 非零率仅 2.2%） | indicator.py 推理时未传 direction 参数，SHORT 模型收到了 LONG 的 CTI 值（corr=-0.004）。**已在 commit 5f9a576 修复** |
| EMA20/60 过滤器无效 | 中 | diff 均值 3.4 点（价格的 0.15%），噪声级别；EMA 滞后 30-60min，反而在某些情况下强制开了错误方向 |
| y_min 回归头选择性偏差 | 低（方向准了再处理） | model_min 只在 y_class=1 的正例上训练，导致 SL 估计系统性偏小（median=-0.08%，低于 ETH 5m 正常噪声 0.3-0.5%） |

---

## v2 目标

1. 模型能区分方向（不再在同一时段 LONG/SHORT 同时高概率）
2. 减少 cluster 连续触发（相邻 bar 的信号应有独立性）
3. 实盘胜率目标：>40%（盈亏平衡约需 >16%@TP=SL=0.8%）

---

## 改动步骤

### Step 1：备份 v1 文件 ✅（已完成）

```bash
cp -r models/ models_v1_backup/
cp data/processed/features_thr005_h30_long.parquet  data/processed/features_thr005_h30_long_v1.parquet
cp data/processed/features_thr005_h30_short.parquet data/processed/features_thr005_h30_short_v1.parquet
```

---

### Step 2：加入方向性动量特征

**改动文件**：`src/features/timeframe.py`

新增特征（在 5m 时间框架的计算函数中）：

| 特征名 | 计算方式 | 意义 |
|--------|----------|------|
| `ret_net_30m` | `(close - close.shift(6)) / close.shift(6)` | 过去 30min（6根5m bar）净收益率 |
| `ret_net_1h` | `(close - close.shift(12)) / close.shift(12)` | 过去 1h（12根5m bar）净收益率 |
| `ret_net_4h` | `(close - close.shift(48)) / close.shift(48)` | 过去 4h（48根5m bar）净收益率 |
| `up_bars_6` | 过去 6 根 bar 中上涨数量占比（0-1） | 近期方向一致性，抗噪 |

这些特征直接告诉模型"价格在往哪走"，而不是间接用波动率代替。

---

### Step 3：去掉 EMA 过滤器

**改动文件**：`scripts/run_signal.py`

删除 EMA20/60 相关代码，所有信号不再区分 FIRE/BLOK，直接输出 FIRE 或 `--`。

原因：新模型如果加了 1h/4h 动量特征，趋势判断已内化在模型里；外部 EMA 过滤器不仅多余，且因滞后性实际有害。

---

### Step 4：重建 parquet + 重训模型

使用新路径，不覆盖 v1：

```bash
# 生成新特征文件
python scripts/build_features.py --direction long  \
    --output data/processed/features_thr005_h30_long_v2.parquet
python scripts/build_features.py --direction short \
    --output data/processed/features_thr005_h30_short_v2.parquet

# 训练新模型
python scripts/train_model.py --direction long  \
    --input data/processed/features_thr005_h30_long_v2.parquet \
    --output models/long_v2/
python scripts/train_model.py --direction short \
    --input data/processed/features_thr005_h30_short_v2.parquet \
    --output models/short_v2/
```

---

### Step 5：验证 feature importance

训练完成后立即检查新特征是否被模型采用：

```bash
conda run -n base python -c "
from xgboost import XGBClassifier
import numpy as np

m = XGBClassifier()
m.load_model('models/long_v2/model_cls.json')
cols = open('models/long_v2/feature_cols.txt').read().strip().splitlines()
imp = m.feature_importances_
idx = np.argsort(imp)[::-1][:20]
for rank, i in enumerate(idx, 1):
    print(f'{rank:2d}.  {imp[i]:.4f}  {cols[i]}')
"
```

**验收标准**：`ret_net_1h` 或 `ret_net_4h` 进入 top10，且 `bb_width`/`atr` 的相对权重下降。

---

### Step 6：离线回测对比

在相同 OOS 区间（2024-04-01 起）运行回测，对比指标：

| 指标 | v1 | v2 目标 |
|------|----|---------|
| OOS 胜率（离线回测） | ~78% | ≥75% |
| 实盘胜率 | 22% | >40% |
| 信号频率（/天） | ~1.6次 | 1-3次 |
| 连续 cluster 触发 | 严重（最多9连续） | 显著减少 |

---

### Step 7：启动 v2 实盘观察

```bash
python scripts/run_signal.py \
    --long-model-dir  models/long_v2 \
    --short-model-dir models/short_v2
```

运行至少 3 天（约 50+ 信号）再评估实盘胜率。

---

## 暂不做的项目

以下问题等 v2 验证后再评估：

- **y_min 回归头偏差**：等方向预测先准，再优化 SL 估计精度
- **回归头训练范围**：对全部样本训练（而非只有正例）
- **1 分钟步幅推理**（memory.txt 项 H）：需重构整个特征体系，是独立的新版本
- **动态出场/续仓**（memory.txt 项 E）：纸仓阶段后再设计
- **入场价改用下一根 bar open**（memory.txt 项 F）：回测准确性优化，次优先级

---

## 版本对照

| | v1 | v2 |
|--|----|----|
| git tag | `v1.0.0` | 待打 |
| 特征数 | 131 | ~135（+4个动量特征） |
| 1h/4h 净收益率特征 | 无 | 有 |
| EMA 过滤器 | 有（无效） | 去掉 |
| CTI direction bug | 存在 | 已修复（commit 5f9a576） |
| 实盘胜率 | 22%（43信号，Apr 19-20） | 待测 |
