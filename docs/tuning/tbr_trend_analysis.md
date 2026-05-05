# tbr_trend 特征分析与调整方案

> 创建日期：2026-05-05  
> 相关文件：`src/features/timeframe.py` L171–174、`src/live/indicator.py` L60

---

## 一、当前实现

### 计算公式

```python
# src/features/timeframe.py L171–174
tbr = taker_buy_volume / volume          # Taker Buy Ratio，主动买单占比
tbr_trend_{tf} = tbr.rolling(5).mean() - tbr.rolling(15).mean()
```

三个时间框架各自独立计算，生成三个特征：
- `tbr_trend_1m`
- `tbr_trend_5m`  
- `tbr_trend_15m`

### 含义

| 值 | 含义 |
|----|------|
| **正值** | 短期 TBR MA > 长期 TBR MA → 买方动能正在增强 |
| **负值** | 短期 TBR MA < 长期 TBR MA → 买方退场，卖方占主导 |
| **接近 0** | 买卖方力量均衡，无明显方向 |

### 设计意图

注释原文：
> "Positive = buy pressure building (leads price up); negative = selling pressure."

即：正值 → 利好上涨，负值 → 利好下跌。这个语义是**从 LONG 视角**定义的。

---

## 二、核心问题：direction 参数未应用于 tbr_trend

### 其他方向感知特征的做法

`compute_features` 接收 `direction` 参数（`"long"` 或 `"short"`），其他几个特征会根据方向翻转：

```python
# bsipi：买卖方强度指标 — 方向感知
if direction == "long":
    bsipi = (tbr - 0.5).rolling(10).mean() * np.sqrt(consec + 1.0)
else:
    bsipi = (0.5 - tbr).rolling(10).mean() * np.sqrt(consec + 1.0)  # ← 翻转

# cbfi_ema：K 线成交方向强度 — 方向感知
if direction == "long":
    stance = ((cl - op) / (hi - lo + 1e-8)).clip(0.0, 1.0)
    cbfi = (stance - 0.5) * 2.0 * vol_ratio
else:
    stance = ((cl - op) / (hi - lo + 1e-8)).clip(-1.0, 0.0)
    cbfi = (stance + 0.5) * (-2.0) * vol_ratio  # ← 翻转
```

### tbr_trend 的问题

```python
# tbr_trend：方向不感知 ← 问题所在
tbr_trend = tbr.rolling(5).mean() - tbr.rolling(15).mean()
# 无论 direction="long" 还是 "short"，计算结果完全相同
```

LONG 模型和 SHORT 模型拿到的 `tbr_trend` 数值完全一样，但两个模型需要的语义是相反的：

| 模型 | tbr_trend 正值意味着 | 正确行为 |
|------|-------------------|---------|
| LONG 模型 | 买方增强 → 利好 LONG | ✅ 应该提高 LONG 触发概率 |
| SHORT 模型 | 买方增强 → 不利 SHORT | ❌ 应该降低 SHORT 触发概率 |

SHORT 模型必须自己从训练数据里学到"正的 tbr_trend 对我不好"这个反向关系，但实际上它没学对——实盘中 SHORT 模型在 tbr_trend 上升（市场反弹）期间仍然持续触发 SHORT。

### 额外发现：live 推理时 direction 参数未传入 compute_features

`src/live/indicator.py` L60：
```python
feat = compute_features(df_raw, tf, include_p2=False)
# ↑ 缺少 direction 参数，默认 direction="long"
```

即使训练时 SHORT 模型用了 `direction="short"`，实盘推理时特征计算全部用 `direction="long"`，导致**训练和推理的特征不一致**。这是一个独立存在的 bug，与 tbr_trend 问题叠加。

---

## 三、实盘验证数据

**v1.0.1 SHORT 信号表现（135h，18 个信号，May 4 数据）：**

| 指标 | 数值 |
|------|------|
| 全解仓胜率 | **22.2%**（4/18） |
| 失败集中场景 | tbr_trend 上升期间（市场反弹中）进 SHORT |
| 典型失败 | 04-30 03:50–08:40，5 个 SHORT 信号，0 TP |
| v1（无 tbr_trend） SHORT 胜率 | 40.0%（12/30） |

v1.0.1 比 v1 多了 `tbr_trend` 特征，SHORT 胜率反而从 40% 跌至 22.2%，差距 18pp，直接归因于 tbr_trend 对 SHORT 方向的误导。

---

## 四、调整方案

### 方案 A：为 SHORT 添加镜像特征 `tsr_trend`（推荐）

**逻辑**：TSR（Taker Sell Ratio）= 1 - TBR，对应"主动卖单占比趋势"。

```python
# src/features/timeframe.py，在 tbr_trend 计算后添加：
tbr_short_ma5  = tbr.rolling(5).mean()
tbr_short_ma15 = tbr.rolling(15).mean()

if direction == "long":
    out[f"tbr_trend{suffix}"] = tbr_short_ma5 - tbr_short_ma15
else:
    # tsr_trend = -(tbr_trend)，卖方动能趋势，正值=卖压增强=利好 SHORT
    out[f"tbr_trend{suffix}"] = tbr_short_ma15 - tbr_short_ma5
```

这样 SHORT 模型的 `tbr_trend` 正值就变成"卖方动能增强"，语义一致，模型不再需要学反向关系。

**代价**：需要重新训练 SHORT 模型（特征数值变了）。

### 方案 B：SHORT 模型排除 tbr_trend（快速验证）

直接从 SHORT 模型的特征集里删掉 `tbr_trend_{1m,5m,15m}` 三列，重训。

```python
# scripts/train_model.py 或 pipeline 中加入：
if direction == "short":
    feature_cols = [c for c in feature_cols if "tbr_trend" not in c]
```

**代价**：需要重新训练，且放弃了卖方动能信息（可能损失部分信号质量）。

### 同步修复：indicator.py 的 direction 传参

```python
# src/live/indicator.py L60，修改为：
feat = compute_features(df_raw, tf, include_p2=False, direction=direction)
```

这个 bug 需要在任何方案实施时一并修复，否则训练/推理特征不一致会持续影响所有方向感知特征（不只是 tbr_trend）。

---

## 五、优先级与实施顺序

| 步骤 | 操作 | 优先级 |
|------|------|--------|
| 1 | 修复 `indicator.py` direction 传参 | 🔴 立即，影响所有方向感知特征 |
| 2 | 在 `timeframe.py` 为 SHORT 翻转 `tbr_trend` | 🟡 下次重训前 |
| 3 | 重训 SHORT 模型 | 🟡 与步骤 2 一起 |
| 4 | 验证新 SHORT 模型实盘胜率 | 🟢 重训后观察 |

---

## 六、特征重要性参考（v1.0.1 LONG 模型）

| 特征 | 排名 | 权重 |
|------|------|------|
| `tbr_trend_15m` | 32/137 | ~0.8% |
| `tbr_trend_5m` | ~50/137 | ~0.6% |
| `tbr_trend_1m` | ~70/137 | ~0.4% |

三个时间框架的 tbr_trend 合计权重约 1.8%，对 LONG 模型是有效特征（rank 32 在 137 个特征中属于中上游）。这进一步证明问题不是特征无用，而是 SHORT 方向语义定义错了。
