# 方向感知特征 Bug 全面分析

> 创建日期：2026-05-05  
> 状态：**待修复**（用户指令：先不改代码）  
> 相关文件：`src/features/timeframe.py`、`src/live/indicator.py`

---

## 背景

`compute_features(df, tf, include_p2, direction)` 设计为对 LONG/SHORT 方向感知，但随着迭代加入的特征并未全部遵守这一约定，导致 SHORT 模型收到语义错误或反向的特征值。

### 已正确做方向感知的特征（参照）

| 特征 | 翻转方式 |
|------|---------|
| `accel_consensus` | LONG 看 accel > 0，SHORT 看 accel < 0 |
| `bsipi` | LONG: `(tbr - 0.5)`，SHORT: `(0.5 - tbr)` |
| `cbfi_ema` | LONG: stance clip 到 [0,1]，SHORT: clip 到 [-1,0] 并翻转符号 |
| `cti`（cross-tf） | 在 `_add_cross_tf_features` 中处理 |

---

## Bug 列表

### Bug 1 🔴 CRITICAL — `indicator.py` 缺少 `direction` 传参

**文件**：`src/live/indicator.py`  
**位置**：L60（每个 tf 循环内的 `compute_features` 调用）

```python
# 当前（BUG）：
feat = compute_features(df_raw, tf, include_p2=False)

# 应为：
feat = compute_features(df_raw, tf, include_p2=False, direction=direction)
```

**影响**：实盘推理时，以下所有方向感知特征无论 SHORT 还是 LONG 均按 `direction="long"` 计算：
- `accel_consensus_{1m,5m,15m}`（3 个特征）
- `bsipi_{1m,5m,15m}`（3 个特征）
- `cbfi_ema_{1m,5m,15m}`（3 个特征）

共 **9 个特征**在 SHORT 实盘推理时与训练时特征不一致，形成训练/推理分布偏移（distribution shift）。

**为什么会出现**：`indicator.py` 的 `direction` 参数后加，per-tf 循环内的调用漏更新。因为 `direction` 有默认值 `"long"` 不报错，静默失效，单跑 LONG 模型时无法发现。

---

### Bug 2 🔴 HIGH — `tbr_trend` 对 SHORT 方向语义相反

**文件**：`src/features/timeframe.py`  
**位置**：L174

```python
# 当前（BUG）：两个方向使用完全相同的公式
out[f"tbr_trend{suffix}"] = tbr.rolling(5).mean() - tbr.rolling(15).mean()
```

**影响**：
- LONG 模型：`tbr_trend > 0` = 买方动能增强 = 利好 ✅
- SHORT 模型：`tbr_trend > 0` = 买方动能增强 = **不利**，但模型收到正信号 ❌

v1.0.1 SHORT 胜率从 v1 的 40.0% 跌至 22.2%（-17.8pp），直接归因于此 bug（v1 没有 `tbr_trend` 特征）。

**为什么会出现**：`tbr_trend` 是 v1.0.1 新增特征，加入时直接复制 LONG 写法，未按 `bsipi` 的模式考虑 SHORT 方向翻转。

#### 解决方案（方案 A — 原地翻转）

**核心思路**：对 SHORT 方向，将存入 `tbr_trend` 的值翻转为 `MA15 - MA5`，即"卖方动能趋势（TSR trend）"：
- 概念上：SHORT 下的 `tbr_trend` 存的是 `TSR_trend = -(TBR_trend) = MA15 - MA5`
- 正值含义变为：卖方动能增强 → 利好 SHORT ✅
- 列名仍为 `tbr_trend`（不新增列），两个方向特征集保持完全对齐（均 137 列）

```python
# src/features/timeframe.py，将 L174 替换为：
ma5  = tbr.rolling(5).mean()
ma15 = tbr.rolling(15).mean()
if direction == "long":
    # TBR trend：正值 = 买方动能增强 = 利好 LONG
    out[f"tbr_trend{suffix}"] = ma5 - ma15
else:
    # TSR trend（存入同名列）：正值 = 卖方动能增强 = 利好 SHORT
    # 等价于 -(tbr_trend_long)，即 MA15 - MA5
    out[f"tbr_trend{suffix}"] = ma15 - ma5
```

**为什么不单独创建 `tsr_trend` 新列**：

如果为 SHORT 添加一个新列 `tsr_trend` 而保留 `tbr_trend`：
- SHORT 特征集变为 138 列，LONG 为 137 列 → 两个模型特征集不对称
- SHORT 中 `tbr_trend` 仍然语义反向（未修复）
- 需要在训练/推理管道中单独处理列对齐

方案 A 的原地翻转更干净：同一列名，LONG 和 SHORT 语义对称，无需改动任何其他地方。

**代价**：需要重新训练 SHORT 模型（训练数据中 `tbr_trend` 的数值会变化）。

---

### Bug 3 🟡 MEDIUM — `trigger_score` 对 SHORT 方向语义不一致

**文件**：`src/features/timeframe.py`  
**位置**：L166（`trigger_score` 计算）

```python
# 当前：
trigger = accel5 * vol_accel.clip(lower=0.0)
out[f"trigger_score{suffix}"] = trigger
```

**分析**：
- `accel5 > 0` = 价格向上加速，`vol_accel > 0` = 成交量放大
- `trigger_score > 0` = 价格上涨加速 + 量放 = 利好 LONG，不利 SHORT
- `accel_consensus` 已经为 SHORT 翻转（检查 accel < 0），但 `trigger_score` 用同一个 `accel5` 计算时没有翻转
- SHORT 模型收到正的 `trigger_score` 应该是空头不利信号，但它需要自己从训练数据学到"负值才是好的"——增加了学习难度，且与 `accel_consensus` 的语义对齐方式相矛盾

**建议修复**：
```python
# SHORT 方向应翻转 accel5 的符号
if direction == "long":
    trigger = accel5 * vol_accel.clip(lower=0.0)
else:
    trigger = (-accel5) * vol_accel.clip(lower=0.0)
out[f"trigger_score{suffix}"] = trigger
```

注意：`release_phase = np.sign(trigger - trigger.shift(3))` 基于 `trigger_score`，一并受影响——若 `trigger_score` 翻转，`release_phase` 也会自动修正。

---

### Bug 4 🟡 MEDIUM — `mvdi` 对 SHORT 方向语义反向

**文件**：`src/features/timeframe.py`  
**位置**：L200 附近（`mvdi` 计算）

```python
# 当前：
norm_roc5 = roc5 / roc5_std
norm_atr = atr14 / atr_mean50
out[f"mvdi{suffix}"] = norm_roc5 / norm_atr.replace(0.0, np.nan)
```

**分析**：
- `roc5 > 0` = 5 根 K 线内价格上涨，对 LONG 是正向动量
- SHORT 模型需要的是"下跌动量"，即 `roc5 < 0` 时才是有利信号
- 当前 SHORT 模型看到的 `mvdi > 0` 代表"价格在相对波动率中向上移动"，与 LONG 解读完全相反
- 模型必须自己学到"负的 `mvdi` 才是 SHORT 入场信号"

**建议修复**：
```python
norm_roc5 = roc5 / roc5_std
norm_atr   = atr14 / atr_mean50
raw_mvdi = norm_roc5 / norm_atr.replace(0.0, np.nan)
out[f"mvdi{suffix}"] = raw_mvdi if direction == "long" else -raw_mvdi
```

---

## 特征方向感知状态总表

| 特征 | 是否方向感知 | 严重度 | 说明 |
|------|------------|--------|------|
| `accel_consensus` | ✅ 已翻转 | — | 正确 |
| `bsipi` | ✅ 已翻转 | — | 正确 |
| `cbfi_ema` | ✅ 已翻转 | — | 正确 |
| `cti`（cross-tf） | ✅ 已翻转 | — | 正确 |
| `tbr_trend` | ❌ 未翻转 | 🔴 HIGH | Bug 2 |
| `trigger_score` | ❌ 未翻转 | 🟡 MEDIUM | Bug 3 |
| `release_phase` | ❌ 未翻转 | 🟡 MEDIUM | 依赖 trigger_score，Bug 3 的衍生 |
| `mvdi` | ❌ 未翻转 | 🟡 MEDIUM | Bug 4 |
| `taker_buy_ratio` | — 中性 | 🟢 OK | 原始值，模型可学双向 |
| `vol_inefficiency` | — 中性 | 🟢 OK | 用 `abs(roc1)`，天然无方向 |
| `roc1/3/5` | — 原始值 | 🟢 OK | 模型从标签学方向关系 |
| `pei` | — 中性 | 🟢 OK | 用 `abs(Δclose)`，无方向 |

---

## Bug 来源根因分析

### 根本原因：部分重构（Partial Refactor）

```
第一阶段：只做 LONG 模型
  → compute_features() 没有 direction 参数
  → 所有特征以 LONG 视角定义

第二阶段：加入 SHORT 模型
  → 增加 direction 参数
  → 部分特征翻转（bsipi、cbfi_ema、accel_consensus）
  → 其他特征漏掉（trigger_score、mvdi 等）

第三阶段：v1.0.1 新增特征
  → tbr_trend 直接复制 LONG 写法
  → 未参考 bsipi 的翻转模式

第四阶段：indicator.py 加 direction 参数
  → 外层函数加了 direction
  → per-tf 循环内的 compute_features 调用漏更新
  → direction="long" 默认值掩盖了错误（无报错）
```

### 关键失效机制：默认值静默失效

`direction="long"` 的默认值是最危险的设计之一：

- 忘记传参 → 不报错 → 不打日志 → 功能静默跑错方向
- 开发和测试默认跑 LONG → SHORT 问题不暴露
- 没有对比测试（live 特征值 vs 离线特征值）→ 分布偏移无法检测

---

## 修复优先级

| 步骤 | 操作 | 优先级 | 文件 |
|------|------|--------|------|
| 1 | 修复 `indicator.py` direction 传参（Bug 1） | 🔴 立即 | `src/live/indicator.py` L60 |
| 2 | 翻转 `tbr_trend`（Bug 2） | 🔴 重训前 | `src/features/timeframe.py` L174 |
| 3 | 翻转 `trigger_score`（Bug 3） | 🟡 重训前 | `src/features/timeframe.py` L166 |
| 4 | 翻转 `mvdi`（Bug 4） | 🟡 重训前 | `src/features/timeframe.py` L200 |
| 5 | 重训 SHORT v1.1 模型 | 🟡 修复后 | `scripts/train_model.py` |
| 6 | 验证 SHORT 实盘胜率恢复 | 🟢 重训后 | 实盘观察 |

步骤 2–4 必须**同时**修复再重训，分批修复会产生新的中间版本不一致问题。

---

## 预期改善

修复 Bug 1–4 后，SHORT 模型的方向感知特征从当前 3/10 增加到 7/10：

- 修复前：`accel_consensus`、`bsipi`、`cbfi_ema` 正确（但实盘时 Bug 1 让它们全部失效）
- 修复后：上述 3 个 + `tbr_trend`、`trigger_score`、`release_phase`、`mvdi` 全部正确

理论上 SHORT 胜率应回升并超过 v1 的 40% 基线（v1 缺少这些特征，信号稀少）。
