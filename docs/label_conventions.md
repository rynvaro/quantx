# 标签命名约定：y_max / y_min

## 当前设计（src/features/labels.py）

Long 模型：
- `y_max` = 未来最大上涨幅度（正数，原始价格方向）
- `y_min` = 未来最大下跌幅度（负数，原始价格方向）
- `y_class` = 1 if y_max >= threshold else 0

Short 模型（labels.py 在训练标签阶段已做 flip）：
- `y_max` = 最大有利下跌幅度（正数，= -y_min_raw）
- `y_min` = 最大不利上涨幅度（负数，= -y_max_raw）
- `y_class` = 1 if y_max >= threshold else 0

**两个模型接口统一：`y_max` 永远是止盈方向（正数），`y_min` 永远是止损方向（负数）。**

---

## 易错点（已踩过）

### 1. 命名歧义
`y_max` 字面上像"最大值/最高价"，容易误认为 short 的 `y_max` 表示价格涨幅，
从而错误地把 short 的 tp/sl 对调。**flip 已在 labels.py 完成，消费方禁止再对调。**

### 2. 目标价计算方向不同
```python
# LONG
tp_price = price * (1 + y_max)   # y_max > 0 → 高于当前价 ✓
sl_price = price * (1 + y_min)   # y_min < 0 → 低于当前价 ✓

# SHORT（y_max 已 flip，代表有利下跌幅度）
tp_price = price * (1 - y_max)   # y_max > 0 → 低于当前价 ✓
sl_price = price * (1 - y_min)   # y_min < 0 → 高于当前价 ✓
```

### 3. 消费方一律直接使用，不要再 flip
`run_signal.py` 的正确写法：
```python
_fmt(long_prob,  long_ymax,  long_ymin,  'long')
_fmt(short_prob, short_ymax, short_ymin, 'short')  # 不要对调 ymax/ymin
```

---

## 待改进

当前命名在 short 方向语义反直觉，后续重构时考虑：
- `y_fav` / `y_adv`：favorable / adverse move，方向无关
- `y_up` / `y_down`：保持原始价格方向，消费方自己做方向判断，flip 移出 labels.py
