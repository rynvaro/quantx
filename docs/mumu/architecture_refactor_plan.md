# 架构重构计划

> 创建日期：2026-05-09  
> 状态：规划中，待实施

---

## 问题一：目录结构

**现状诊断：**

```
scripts/
  download_data.py     ← 数据下载
  build_features.py    ← 特征构建
  train_model.py       ← 训练
  run_backtest.py      ← 回测
  run_signal.py        ← 实盘
  analyze_*.py         ← 分析工具（2个）
  explore_*.py         ← 探索工具

src/features/
  loader.py            ← 数据加载（理应属于 data 层）
  pipeline.py          ← 特征流水线
  timeframe.py         ← 各周期特征计算
  indicators.py        ← 指标函数库
  labels.py            ← 标签计算
```

**问题**：`loader.py`（数据加载）混在 `src/features/` 里；`download_data.py`（数据下载）在 `scripts/` 里；两者概念上属于同一层但物理位置分散。`scripts/` 里分析工具和执行脚本混在一起，没有区分。

**建议的清晰结构：**

```
quantx/
├── src/
│   ├── data/                ← 【新】数据层（下载 + 存储 + 加载）
│   │   ├── downloader.py    ← 从 scripts/download_data.py 迁入
│   │   ├── loader.py        ← 从 src/features/loader.py 迁入
│   │   └── schema.py        ← 数据格式定义（列名、类型）
│   ├── features/            ← 特征层（纯计算，不含 I/O）
│   │   ├── indicators.py    ← 现有，保持
│   │   ├── timeframe.py     ← 现有，保持
│   │   ├── labels.py        ← 现有，保持
│   │   └── pipeline.py      ← 现有，去掉 loader 依赖
│   ├── train/               ← 现有，保持
│   ├── backtest/            ← 现有，保持
│   └── live/                ← 现有，保持
│
├── scripts/
│   ├── data/                ← 【新子目录】数据操作脚本
│   │   ├── download.py      
│   │   └── build_features.py
│   ├── train_model.py
│   ├── run_backtest.py
│   ├── run_signal.py
│   └── analyze/             ← 【新子目录】分析工具
│       ├── signals.py
│       ├── importance.py
│       └── correlation.py   ← 待实现
```

---

## 问题二：5m 推理的延迟问题

**5m K 线推理存在结构性延迟。** 现有流程：

```
时间轴：|----1m----|----2m----|----3m----|----4m----|----5m----|
                                                              ↑
                                                    5m K线收盘
                                                    → 计算特征
                                                    → 模型推理
                                                    → 发出信号
                                                    → 下一根5m开盘才能入场

实际延迟：最差 = 5分钟（刚开始的行情错过4分钟才知道）
          平均 = ~2.5分钟
```

对于目标持仓 30–60 分钟的短线来说，**2–4 分钟的延迟意味着丢掉最初的 10–20% 行情，且入场价格更差，TP 距离被压缩**。这就是为什么信号分析中反复出现很多 R-ratio 只有 0.3–0.8x 的信号，价格根本没空间走到 TP。

---

## 问题三：改用 1m 为基准的方案

**改用 1m K 线为推理基准，所有特征改为滚动窗口计算（Rolling Window Features）。**

现有做法 vs 改进做法对比：

```
现有（5m K线为基准）：
  "5m 加速度" = 5m K线的 ROC(1)
  "15m 动量"  = 15m K线的 RSI
  缺点：必须等整根 K线收盘

改进（1m K线为基准，滚动回看）：
  "5m 加速度" = 当前1m向前5根1m的价格变化率 → 任意时刻可算
  "15m 动量"  = 当前1m向前15根1m的RSI          → 任意时刻可算
  "1h 趋势"   = 当前1m向前60根1m的EMA排列       → 任意时刻可算
  优点：每根1m收盘即可推理，最大延迟仅1分钟
```

**关键改变的三层含义：**

| 维度 | 现在（5m base） | 改进后（1m base） |
|------|----------------|-----------------|
| 推理频率 | 每5分钟1次 | 每1分钟1次 |
| 最大延迟 | 5分钟 | 1分钟 |
| 特征计算 | 用5m/15m/1h/4h的收盘价 | 全部用1m的收盘价，窗口=5/15/60/240 |
| 训练数据 | 每个5m bar一条样本 | 每个1m bar一条样本（数据量×5） |
| 标签变化 | 未来30根1m内触发TP/SL | 未来N根1m内触发TP/SL（同样逻辑）|

**这个改动的代价：**

1. 数据量 ×5（551,808 → ~275万行），训练时间增加，但可接受
2. 特征代码需要重写 `timeframe.py`——不再有"多周期 K 线"的概念，全部变成"不同窗口长度的1m滚动计算"
3. 实盘推理逻辑简化（只需订阅1m K线，不需要等5m收盘）

**信号逻辑变化：**

```python
# 现在（等5m收盘）
on_5m_close():
    features = compute_features(df_5m[-1], df_15m[-1], df_1h[-1], df_4h[-1])
    signal = model.predict(features)

# 改进后（每1m收盘即推理）
on_1m_close():
    # 从当前1m bar向前看
    window_5m  = df_1m[-5:]   # 过去5根1m
    window_15m = df_1m[-15:]  # 过去15根1m
    window_1h  = df_1m[-60:]  # 过去60根1m
    window_4h  = df_1m[-240:] # 过去240根1m
    features = compute_rolling_features(window_5m, window_15m, window_1h, window_4h)
    signal = model.predict(features)
```

---

## 综合建议

这三个问题合在一起，指向的是**同一个架构升级**：

```
当前架构：多时间周期 K线对齐 → 5m 推理
目标架构：单时间周期（1m）滚动窗口 → 1m 推理
```

**建议分两步实施：**

### 阶段一：目录整理（1天，低风险）

纯文件移动，不改逻辑：
- 新建 `src/data/`，将 `src/features/loader.py` 迁入
- `scripts/download_data.py` 迁入 `scripts/data/download.py`
- `scripts/build_features.py` 迁入 `scripts/data/build_features.py`
- `scripts/analyze_*.py` 迁入 `scripts/analyze/`
- 更新所有 import 路径

### 阶段二：1m base 架构（3–5天，高价值）

- 新写 `src/features/rolling.py`（滚动窗口特征，不依赖多周期 K 线）
- 修改 `labels.py`（标签逻辑不变，只是 base 改为 1m）
- 修改 `src/live/signal.py`（订阅 1m K 线，每分钟推理）
- 新训练数据集 `features_1mbase_v2.0.parquet`

---

## 实施状态

- [ ] 阶段一：目录整理
- [ ] 阶段二：1m base 特征重写（`src/features/rolling.py`）
- [ ] 阶段二：labels.py 适配 1m base
- [ ] 阶段二：实盘信号逻辑改为 1m 订阅
- [ ] 阶段二：重新生成训练数据集
- [ ] 阶段二：重新训练模型，对比回测结果
