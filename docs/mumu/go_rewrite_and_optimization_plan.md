# 优化与技术选型计划

> 创建日期：2026-05-09  
> 状态：规划中，待实施

---

## 一、特征剪枝

**现在能做，立即可做（预计 1 天）。**

已有条件：
- `models/v1.0.2/long/feature_importance.csv` 已存在
- `scripts/analyze_importance.py` 已能输出 gain 排名

**具体步骤：**
1. 读取 importance.csv，按 gain 排序
2. 设阈值（比如 top-40，或 cumulative gain > 95%）
3. 修改 `feature_cols.txt`（训练时用哪些列的配置文件）
4. 用剪枝后的特征集重新训练，对比回测指标

**注意**：gain importance 对高基数特征有高估偏差。更准确的方式是 Permutation Importance，但需额外实现。现阶段先用 gain 做初步剪枝足够。

---

## 二、重要性分析的数据来源

**完全基于历史数据，与实盘数据两条线完全独立。**

```
data/raw/klines/（历史K线）
  ↓ build_features.py
data/processed/v1.0.2/features_*.parquet   ← 训练数据
  ↓ train_model.py（XGBoost 训练时计算特征重要性）
models/v1.0.2/long/feature_importance.csv  ← 输出在这里
```

重要性分析反映的是**哪些特征在历史数据上对模型决策贡献最大**。

实盘数据（`data/signal_*.log`）通过 `analyze_signals.py` 单独分析，与训练流程无关。

---

## 三、Go 实现推理层，Python 只做训练

**想法合理，是工业界常见的分层架构。难度中等，有一个关键陷阱。**

### 3.1 分层职责划分

| 层 | 现在（Python） | 改用 Go | 收益 |
|----|---------------|---------|------|
| 数据下载 | `download_data.py` | Go HTTP client | ✅ 更快，并发友好 |
| 特征计算（训练用） | `build_features.py` | 保持 Python | 训练离线跑，速度不关键 |
| 特征计算（推理用） | `src/live/indicator.py` | **Go 重写** | ✅ 延迟从 ~50ms → <5ms |
| 模型推理 | `src/live/signal.py` | Go 调用模型 | ✅ 可行（见下） |
| 模型训练 | `src/train/` | 保持 Python | XGBoost/sklearn 无 Go 替代 |
| 回测 | `src/backtest/engine.py` | 保持 Python | 离线跑，速度不关键 |

### 3.2 模型推理方案

XGBoost 模型（`.json` 格式）在 Go 中有两种调用方式：

**方案 A：leaves 库（推荐）**
```
github.com/dmitryikh/leaves
  → 纯 Go 实现，直接读取 XGBoost JSON 模型
  → 无 CGO 依赖，部署简单
  → 支持 XGBoost gbtree/dart，满足当前需求
  → 无需改变训练流程
```

**方案 B：ONNX 中间格式**
```
Python 训练 → 导出 ONNX → Go 用 onnxruntime-go 推理
  → 通用性更强
  → 多一个转换步骤，维护成本略高
```

`leaves` 是最简洁的选择，直接读取现有 `.json` 模型文件。

### 3.3 关键陷阱：特征一致性

这是**唯一的核心风险**：

```
Python 训练时计算的特征（EMA、ATR、布林带等）
  必须与
Go 推理时计算的特征
  完全一致（数值精度、边界处理、NaN 填充方式）

否则：模型在历史数据上表现良好，实盘完全失效
```

现有 `src/live/indicator.py` 就是为解决这个问题存在的——它是训练特征的实盘复制版本。用 Go 重写意味着需要逐个函数验证数值对齐，必须有完整的 Python vs Go 对比测试。

### 3.4 Go 负责边界（最终形态）

```
Go 负责（适合，高价值）：
  ✅ WebSocket 订阅 1m K线
  ✅ 维护滑动窗口 K线缓冲区
  ✅ 特征计算（Go 实现，对齐 Python 训练逻辑）
  ✅ 模型推理（leaves 库加载 XGBoost JSON）
  ✅ 信号输出 / 风控逻辑 / 下单执行

Python 负责（保持不动）：
  ✅ 历史数据下载（一次性，速度不关键）
  ✅ 训练集特征构建（build_features.py）
  ✅ 模型训练（XGBoost）
  ✅ 回测（engine.py）
  ✅ 信号分析（analyze_signals.py）
```

### 3.5 工作量估算

| 任务 | 工作量 |
|------|--------|
| Go 重写特征计算（EMA/ATR/RSI/BB 等核心指标） | 3–4 天 |
| 数值一致性测试（Python vs Go 逐指标对齐验证） | 1–2 天 |
| leaves 集成 + 推理流程 | 1 天 |
| WebSocket K线订阅 + 滑动窗口维护 | 1 天 |
| **合计** | **6–8 天** |

---

## 四、建议实施顺序

```
立即可做（不依赖架构决策）：
  → 特征剪枝（1天）

短期（3–5天）：
  → 1m base 特征重构（解决推理延迟，见 architecture_refactor_plan.md）

中期（等 1m base 稳定后）：
  → 目录整理（1天）
  → Go 推理层（6–8天）

⚠️  Go 推理层必须等 1m base 特征设计稳定后再做
    否则特征刚用 Go 实现完又要改，白费力气
```

---

## 五、实施状态

- [ ] 特征剪枝（top-40 / cumulative 95%）
- [ ] Permutation Importance 实现（比 gain 更准确）
- [ ] 1m base 特征架构（见 architecture_refactor_plan.md）
- [ ] Go 推理层：WebSocket + 滑动窗口
- [ ] Go 推理层：核心指标实现（EMA/ATR/RSI/BB/MACD）
- [ ] Go 推理层：leaves 集成 + 模型推理
- [ ] Python vs Go 特征数值对齐验证测试
