# 训练管道设计文档

## 1. 目标

基于 `features_thr005_h30.parquet`（threshold=0.5%，horizon=30min，21.9% 正类率），训练三个独立模型头，构成完整的 Calm-Trigger/Spring 交易信号系统。

---

## 2. 目录结构

```
src/
└── train/
    ├── __init__.py
    ├── fold.py          # walk-forward 折叠生成器
    ├── model.py         # 3-head 训练 + Platt Scaling 校准
    ├── evaluate.py      # 指标计算（分类 + 回归 + 业务指标）
    └── pipeline.py      # 串联主函数

scripts/
└── train_model.py       # CLI 入口

models/                  # 输出目录（.gitignore 已排除）
├── model_cls.json       # XGBClassifier
├── model_max.json       # XGBRegressor (y_max)
├── model_min.json       # XGBRegressor (y_min)
└── calibrator.pkl       # Platt Scaling 校准器
```

---

## 3. 模块设计

### 3.1 fold.py — Walk-Forward 折叠生成器

采用**时间序列 Walk-Forward** 切分，严禁随机切分。

```
|─── train ───|─gap─|─ val ─|─gap─|─── train ───|─gap─|─ val ─| ...
```

参数：
- 折数：6 折
- val 窗口：4 个月/折
- gap（禁区）：5 天

**gap 的必要性**：label 的 horizon=30min，但特征中含有滚动窗口（最长 200 bar），若 train/val 紧邻，val 集最早的特征值仍依赖 train 末尾的原始价格数据，存在间接泄漏风险。5 天 gap 确保完全隔离。

只使用 `is_weekend=0 & is_black_swan=0` 的样本参与折叠，与训练时的过滤逻辑一致。

---

### 3.2 model.py — 3-Head 训练

#### Head 1：XGBClassifier（y_class）

- 目标：预测 30min 内是否涨超 0.5%
- 类别不平衡处理：`scale_pos_weight = neg_count / pos_count ≈ 3.6`，使模型感知正类稀缺性
- Early stopping：在 val fold 上监控 `logloss`，patience=50 rounds

#### Head 2：XGBRegressor（y_max）

- 目标：预测 30min 内最大涨幅（连续值）
- **只在正类样本（y_class=1）上训练**
  - 理由：负类样本（78%）的 y_max 均值仅 0.16%，是随机游走噪声，混入会严重干扰回归头学习真实行情幅度的规律。正类专训确保模型学习的是"真行情时能涨多高"，而非"平均情况"。
  - 正类样本量 85,463，足够支撑 XGBoost 训练。

#### Head 3：XGBRegressor（y_min）

- 目标：预测 30min 内最大回撤（连续值，负数）
- 同样**只在正类样本上训练**
  - 理由同上，且 y_min 用于实盘止损估算，负类的 y_min 分布完全不同，混入会使预测值偏向"更深回撤"，导致止损设置过宽。

#### Platt Scaling 校准器

- XGBClassifier 的 raw score 不是概率，倾向于极端值（接近 0 或 1）
- 用每折 val 集的 OOF（Out-of-Fold）预测结果训练一个 `LogisticRegression`
- 输出的 `calibrator.pkl` 将 raw score 映射为校准概率，供实盘阈值过滤使用

---

### 3.3 evaluate.py — 评估指标

每折 val 集上输出：

**分类指标（Head 1）：**
- AUC-PR（Precision-Recall 曲线面积）— 首要指标，不平衡数据下比 AUC-ROC 更有意义
- AUC-ROC — 次要参考
- 在多个 threshold（0.4 / 0.5 / 0.6）下的 Precision / Recall / F1

**回归指标（Head 2 & 3）：**
- MAE（平均绝对误差）
- Pearson 相关系数

**业务指标（核心）：**
- 在不同 Precision 操作点下的期望净利/笔
  - 公式：`E = Precision × TP_gain + (1-Precision) × FP_loss`
  - TP_gain = 名义仓位 × y_max_pred × 实捕率(0.65) - 手续费
  - FP_loss = -(手续费 + 随机游走损失估算)
- 每折汇总表 + 跨折稳定性（std / mean）

---

### 3.4 pipeline.py — 主函数

串联逻辑：

```
load parquet
  → filter (is_weekend=0, is_black_swan=0, dropna labels)
  → generate walk-forward folds
  → for each fold:
      → train 3 heads
      → collect OOF predictions
      → evaluate on val
  → fit Platt calibrator on all OOF predictions
  → retrain final models on FULL data
  → save models/ + print summary
```

---

### 3.5 scripts/train_model.py — CLI

```bash
python scripts/train_model.py \
    --input data/processed/features_thr005_h30.parquet \
    --output models/ \
    --n-folds 6 \
    --gap-days 5
```

---

## 4. 关键设计决策

### 决策 1：回归头只用正类样本训练

**选择**：y_max、y_min 回归头仅在 y_class=1 的样本（85,463 条）上训练。

**原因**：
- 负类的 y_max 均值 0.164%，是 ETH 随机游走的背景噪声
- 正类的 y_max 均值 0.983%，是真实行情的幅度分布
- 两类混训会使回归头学习一个"加权平均"，严重低估真实行情幅度，导致 y_max 预测值偏低，止盈位设置保守

### 决策 2：最终模型用全量数据重训

**选择**：Walk-Forward 仅用于评估，评估完成后用全部训练数据重训一遍作为部署模型。

**原因**：
- Walk-Forward 的目的是**验证策略是否稳定**，而不是限制模型只用部分数据
- 全量重训让最终模型见过更多市场状态（牛市/熊市/震荡），泛化能力更强
- 评估结果来自 walk-forward，报告的指标是对实盘表现的真实估计，不存在偷看未来

### 决策 3：第一版使用全部 136 个特征

**选择**：不做特征预筛选，直接用全部 136 个特征训练。

**原因**：
- XGBoost 内置 L1/L2 正则，对冗余特征有天然抵抗力，不会像线性模型那样被弱特征严重干扰
- 先跑通、拿到 feature importance 后再做有据可依的剪枝，比凭直觉提前删特征更可靠
- 第一版的目标是建立基线，不是调优

### 决策 4：超参数使用合理默认值

**选择**：`n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`，配合 early stopping。

**原因**：
- 在验证实验流程可行之前调参是浪费时间
- 以上参数是 XGBoost 在金融时序任务上经验证的稳健起点
- Early stopping 会自动找到最优 n_estimators，避免过拟合

---

## 5. 下一步计划（优化方向）

以下优化**不在第一版实现范围内**，待基线模型训练完成、评估指标确认后按优先级推进：

### 5.1 超参数调优（Optuna）
- 用 Optuna 在 walk-forward 的前 4 折上做贝叶斯搜索
- 调参目标：AUC-PR 最大化
- 搜索空间：`max_depth`(3-10), `learning_rate`(0.01-0.2), `subsample`(0.6-1.0), `min_child_weight`(1-20)

### 5.2 特征剪枝
- 基线训练完成后，提取 XGBoost gain-based feature importance
- 删除 importance 排名后 30 个特征（约占 22%）
- 重新训练，对比 AUC-PR 变化，确认是否提升泛化

### 5.3 第四个 Head：行情幅度分级
- 当前 y_max 是连续回归，实盘难以直接使用
- 考虑增加分类 head：y_tier = {small: 0.5-1%, medium: 1-2%, large: >2%}
- 用于动态调整止盈位

### 5.4 回归头训练策略对比实验
- 当前：只用正类训练
- 实验对比：全样本训练，对比两者在 val 集上的 MAE 和业务指标
- 若全样本 MAE 更低，则切换

### 5.5 在线学习 / 滚动重训
- 当前：静态模型，一次训练部署
- 后续：每周用最新 1 个月数据增量微调
- 防止模型对市场状态漂移（regime change）失效
