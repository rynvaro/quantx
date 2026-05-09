# 特征扩展计划

> 版本：v1.0.2+  
> 最后更新：2026-05-09  
> 当前特征数：**130 个**（含 labels/price 列共 146 列）  
> 目标：扩展至 **200–250 个候选特征**，通过筛选确定最终训练集
>
> **注意**：本文档基于 5m base 架构设计。后续迁移到 1m base 时，`_{tf}` 后缀的含义从"K 线周期"变为"聚合窗口长度"，特征经济含义不变，代码需重写。

---

## 一、现有特征清单（130 个）

### 1.1 已实现特征（按类别）

#### 波动类（Volatility）
| 特征名 | 时间周期 | 说明 |
|--------|----------|------|
| `atr14_{tf}` | 1m/5m/15m/1h/4h | Average True Range（14 期，Wilder 平滑） |
| `atr_zscore_{tf}` | 1m/5m/15m | ATR 相对 200 期均值的 z-score |
| `bb_width_{tf}` | 1m/5m/15m/1h/4h | 布林带宽度（归一化） |
| `bb_position_{tf}` | 1m/5m/15m/1h/4h | 价格在布林带中的位置 [0,1] |
| `bb_width_pct_{tf}` | 1m/5m/15m | 布林带宽度的百分位排名（200 期） |

#### 动量类（Momentum）
| 特征名 | 时间周期 | 说明 |
|--------|----------|------|
| `roc1/3/5_{tf}` | 1m/5m/15m | N 期价格变化率 |
| `roc_accel1/3/5_{tf}` | 1m/5m/15m | ROC 的一阶差分（加速度） |
| `rsi14_{tf}` | 1m/5m/15m/1h/4h | RSI（14 期，Wilder 平滑） |
| `macd_hist_{tf}` | 1m/5m/15m/1h/4h | MACD 柱状图 |
| `mvdi_{tf}` | 1m/5m/15m | 动量-波动比（normalized ROC / normalized ATR） |

#### 趋势类（Trend）
| 特征名 | 时间周期 | 说明 |
|--------|----------|------|
| `ema9/21/55_{tf}` | 1m/5m/15m/1h/4h | EMA 绝对值 |
| `ema_cross_{tf}` | 1m/5m/15m/1h/4h | EMA9/EMA21 偏差比 |
| `ema200_4h` | 4h | EMA200 绝对值 |
| `ema_slope_4h` | 4h | EMA55 过去 10 期斜率 |
| `deviation_4h` | 4h | 收盘价相对 EMA200 偏差 |
| `trend_score_4h` | 4h | EMA 三线排列得分（−1 到 +1） |
| `adx_4h` | 4h | ADX 趋势强度 |
| `rsi_regime_4h` | 4h | RSI 归一化到 [0,1] |
| `macd_zone_1h` | 1h | MACD 柱状图符号（±1） |

#### 成交量类（Volume）
| 特征名 | 时间周期 | 说明 |
|--------|----------|------|
| `taker_buy_ratio_{tf}` | 1m/5m/15m | 主动买入量占比 |
| `tbr_trend_{tf}` | 1m/5m/15m | TBR 短期均线差（买压趋势） |
| `volume_accel_{tf}` | 1m/5m/15m | 成交量相对 5 期均值加速度 |
| `vwap_dev_{tf}` | 1m/5m | 价格相对日内 VWAP 偏差 |
| `pei_{tf}` | 1m/5m/15m | 价格冲击效率（Δclose / volume） |
| `vol_inefficiency_{tf}` | 1m/5m/15m | 成交量吸收效率（vol_ratio / |roc1|） |

#### 压缩-释放类（Spring Physics，核心创新）
| 特征名 | 时间周期 | 说明 |
|--------|----------|------|
| `compress_bars_{tf}` | 1m/5m/15m | 连续 ATR 压缩持续 bar 数 |
| `vcdi_{tf}` | 1m/5m/15m | 压缩深度指数（ATR 百分位 × log(压缩时长)） |
| `earp_{tf}` | 1m/5m/15m | BB 宽度方向性排名（压缩 vs 扩张） |
| `trigger_score_{tf}` | 1m/5m/15m | 弹簧释放触发分（加速度 × 成交量爆发） |
| `accel_consensus_{tf}` | 1m/5m/15m | 多时间窗口加速度方向一致性 |
| `compress_quality_{tf}` | 1m/5m/15m | 压缩期间收益率标准差 |
| `vol_dry_ratio_{tf}` | 1m/5m/15m | 压缩期成交量萎缩比 |
| `price_in_range_{tf}` | 1m/5m/15m | 收盘在压缩区间的位置 |
| `release_phase_{tf}` | 1m/5m/15m | 释放阶段（上升 +1 / 峰值 −1） |
| `bsipi_{tf}` | 1m/5m/15m | 买卖压力持续性指数 |
| `cbfi_ema_{tf}` | 1m/5m/15m | 蜡烛体方向成交量流 EMA |

#### 跨周期合成类（Cross-TF）
| 特征名 | 说明 |
|--------|------|
| `cpai` | 1m/5m/15m ROC5 方向一致性 × 均值幅度 |
| `cti` | Calm-Trigger Index（vcdi_5m × trigger_score_5m × 趋势门控） |

#### 统计类（Statistical）
| 特征名 | 时间周期 | 说明 |
|--------|----------|------|
| `autocorr_lag5_{tf}` | 1m/5m/15m | 滚动 50 期、lag=5 的自相关系数 |

#### 时间类（Temporal）
| 特征名 | 说明 |
|--------|------|
| `hour_of_day` | UTC 小时（0–23） |
| `day_of_week` | 星期几（0=周一，6=周日） |

#### 过滤标记（Filter Flags）
| 特征名 | 说明 |
|--------|------|
| `is_weekend` | 是否周末 |
| `is_black_swan` | ATR z-score ≥ 4.0 |

#### 资金费率
| 特征名 | 说明 |
|--------|------|
| `funding_rate` | Binance 8h 资金费率（前向填充） |

---

## 二、待扩展特征（60–80 个候选）

### 2.1 动量类补充

| 特征名（建议） | 计算方式 | 覆盖周期 | 优先级 |
|--------------|---------|---------|--------|
| `rsi7_{tf}` | RSI(7) — 短周期超买超卖 | 5m/15m/1h | ⭐⭐⭐ |
| `rsi21_{tf}` | RSI(21) — 长周期 | 5m/1h/4h | ⭐⭐ |
| `roc10/20_{tf}` | 10/20 期变化率 | 5m/15m/1h | ⭐⭐ |
| `stoch_k_{tf}` | Stochastic %K(14,3) | 5m/15m/1h | ⭐⭐⭐ |
| `stoch_d_{tf}` | Stochastic %D（%K 的 3 期 MA） | 5m/15m/1h | ⭐⭐ |
| `williams_r_{tf}` | Williams %R(14) | 5m/15m | ⭐⭐ |
| `cci_{tf}` | CCI(20)：价格相对 TP 均值的偏差 | 5m/15m/1h | ⭐⭐ |
| `roc_diff_1m_5m` | roc1_1m − roc1_5m（微结构偏差） | 跨周期 | ⭐⭐⭐ |

### 2.2 趋势类补充

| 特征名（建议） | 计算方式 | 覆盖周期 | 优先级 |
|--------------|---------|---------|--------|
| `ema_cross_5_20_{tf}` | EMA5/EMA20 比值 − 1 | 5m/15m/1h | ⭐⭐⭐ |
| `ema200_dev_{tf}` | close/EMA200 − 1 | 1h/4h | ⭐⭐⭐ |
| `adx_{tf}` | ADX(14) | 1h/5m | ⭐⭐⭐ |
| `plus_di_{tf}` | ADX 正向方向指标 | 1h | ⭐⭐ |
| `minus_di_{tf}` | ADX 负向方向指标 | 1h | ⭐⭐ |
| `trend_score_1h` | EMA 三线排列（复用现有逻辑到 1h） | 1h | ⭐⭐⭐ |
| `ichimoku_cloud_pos` | 价格相对云层位置（+1/0/−1） | 1h | ⭐ |

### 2.3 波动类补充

| 特征名（建议） | 计算方式 | 覆盖周期 | 优先级 |
|--------------|---------|---------|--------|
| `atr_ratio_{tf}` | atr14 / atr14.shift(14)（波动趋势） | 5m/1h | ⭐⭐⭐ |
| `atr_pct_{tf}` | atr14 / close（相对波动率） | 5m/1h | ⭐⭐⭐ |
| `bb_squeeze_{tf}` | BB width < BB width 20期均值（压缩布尔） | 5m/15m | ⭐⭐ |
| `kc_position_{tf}` | Keltner Channel 位置（close 相对中线偏差/ATR） | 5m/1h | ⭐⭐ |
| `atr14_1m_vs_5m` | atr14_1m / atr14_5m（微结构波动比） | 跨周期 | ⭐⭐⭐ |
| `hist_vol_20_{tf}` | 20 期收益率标准差（历史波动率） | 5m/1h | ⭐⭐ |
| `parkinson_vol_{tf}` | Parkinson 波动率：(ln(H/L))² / (4ln2) | 5m/1h | ⭐ |

### 2.4 成交量类补充

| 特征名（建议） | 计算方式 | 覆盖周期 | 优先级 |
|--------------|---------|---------|--------|
| `obv_{tf}` | On Balance Volume（方向累积成交量） | 5m/1h | ⭐⭐ |
| `obv_slope_{tf}` | OBV 过去 10 期斜率（归一化） | 5m/1h | ⭐⭐ |
| `vol_ma_ratio_{tf}` | volume / MA(volume, 20) | 5m/15m/1h | ⭐⭐⭐ |
| `vol_pct_rank_{tf}` | 成交量 200 期百分位排名 | 5m/1h | ⭐⭐ |
| `taker_buy_ratio_ma_{tf}` | TBR 的 20 期 MA（中期买压水平） | 5m/1h | ⭐⭐ |
| `vwap_dev_1h` | 过去 1h 累计 VWAP 偏差（vs 日内 VWAP） | 1h | ⭐⭐⭐ |
| `cvd_5m` | Cumulative Volume Delta (近 20 bar TBR 加权方向) | 5m | ⭐⭐⭐ |

### 2.5 价格结构类补充

| 特征名（建议） | 计算方式 | 覆盖周期 | 优先级 |
|--------------|---------|---------|--------|
| `hl_position_{tf}` | (close − low_N) / (high_N − low_N)，N=20 | 5m/1h | ⭐⭐⭐ |
| `body_ratio_{tf}` | |close−open| / (high−low+ε)（蜡烛实体比） | 5m/1m | ⭐⭐ |
| `upper_shadow_{tf}` | (high−max(open,close)) / ATR14（上影线） | 5m | ⭐⭐ |
| `lower_shadow_{tf}` | (min(open,close)−low) / ATR14（下影线） | 5m | ⭐⭐ |
| `gap_{tf}` | open / prev_close − 1（跳空） | 5m/1h | ⭐⭐ |
| `close_vs_open_{tf}` | close / open − 1（bar 内涨跌） | 5m/1m | ⭐⭐ |
| `swing_high_dist_{tf}` | 与最近 N 期高点的距离（ATR 归一化） | 5m/1h | ⭐⭐ |
| `swing_low_dist_{tf}` | 与最近 N 期低点的距离（ATR 归一化） | 5m/1h | ⭐⭐ |

### 2.6 时间类补充

> 注：实盘信号分析发现 02:xx CST = 77–88% WR（最高），13:xx CST = 0% WR（死区）。
> 当前 `hour_of_day` 是 UTC 整数，模型需要自己学时区映射，以下特征显式编码更直接。

| 特征名（建议） | 计算方式 | 优先级 |
|--------------|---------|--------|
| `hour_sin` | sin(2π × hour/24)（时间周期编码） | ⭐⭐⭐ |
| `hour_cos` | cos(2π × hour/24) | ⭐⭐⭐ |
| `hour_cst` | UTC hour + 8（mod 24），CST 小时数 | ⭐⭐⭐ |
| `dow_sin` | sin(2π × dow/7) | ⭐⭐ |
| `dow_cos` | cos(2π × dow/7) | ⭐⭐ |
| `is_asia_session` | CST 01:00–09:00（加密最活跃亚洲时段） | ⭐⭐⭐ |
| `is_eu_session` | CST 15:00–23:00（欧洲时段） | ⭐⭐⭐ |
| `is_us_session` | CST 21:30–04:00（美国时段，含美股开盘） | ⭐⭐⭐ |
| `is_dead_zone` | CST 11:00–15:00（已验证 0% WR 死区） | ⭐⭐⭐ |
| `minutes_to_funding` | 距下一次资金费率结算的分钟数（8h cycle） | ⭐⭐⭐ |

### 2.7 跨周期合成类补充

| 特征名（建议） | 计算方式 | 优先级 |
|--------------|---------|--------|
| `rsi_diff_5m_1h` | rsi14_5m − rsi14_1h（短长 RSI 偏差） | ⭐⭐⭐ |
| `rsi_diff_1h_4h` | rsi14_1h − rsi14_4h | ⭐⭐ |
| `bb_pos_diff_5m_1h` | bb_position_5m − bb_position_1h（BB 位置偏差） | ⭐⭐⭐ |
| `trend_alignment_3tf` | trend_score_4h × sign(ema_cross_1h) × sign(roc5_5m) | ⭐⭐ |
| `vol_accel_vs_1h` | volume_accel_5m 相对 atr_zscore_1h（量价背离） | ⭐⭐ |
| `funding_pressure` | funding_rate × sign（方向性资金费率压力） | ⭐⭐⭐ |

### 2.8 元特征类（Meta-Features，模型输出作为特征）

> 来自 memory.txt 的想法：把一个已训练模型的输出当作另一个模型的输入特征，形成"指标的指标"。

| 特征名（建议） | 计算方式 | 优先级 |
|--------------|---------|--------|
| `vol_regime_score` | 预训练的波动状态分类器输出（如 v1.0.x 的 atr_zscore 综合得分） | ⭐⭐ |
| `bull_bear_score` | 用更长周期斜率（如 EMA200_4h 斜率）代替单独训练的牛熊分类器 | ⭐⭐ |
| `signal_prob_lag1` | 上一根 bar 的模型 long/short 概率（信号持续性） | ⭐ |

> 注：元特征在训练时需要特别注意 data leakage，`signal_prob_lag1` 用的是前一根 bar 的预测，不存在泄露问题。

### 2.9 统计/结构类补充（P2，可选）

| 特征名（建议） | 计算方式 | 优先级 |
|--------------|---------|--------|
| `skewness_20_{tf}` | 20 期收益率偏度 | ⭐ |
| `kurtosis_20_{tf}` | 20 期收益率峰度 | ⭐ |
| `hurst_{tf}` | Hurst 指数（已实现，P2=False 默认关闭） | ⭐ |
| `zscore_close_{tf}` | (close − MA50) / std50（价格 z-score） | ⭐⭐ |
| `rolling_corr_vol_ret_{tf}` | 20 期量价相关系数 | ⭐⭐ |

---

## 三、明确拒绝的特征（v2/v3 已验证有害）

> 以下特征在 feature/v2 和 feature/v3 分支中实测，导致模型概率从 0.60–0.85 暴跌至 0.007–0.008，**零信号**，不得重新添加。

| 特征名 | 拒绝原因 |
|--------|---------|
| `ret_net_30m_5m` | 30分钟净收益率：回顾性强趋势特征，模型学到"涨了很多才下注"，丧失预见性 |
| `ret_net_1h_5m` | 同上，1h 版本 |
| `ret_net_4h_5m` | 同上，4h 版本，更严重 |
| `up_bars_6_5m` | 连续上涨 bar 数：与 ret_net 同类问题，事后特征 |
| `buy_ratio_30m_5m` | 30分钟买入占比：与 taker_buy_ratio + tbr_trend 高度冗余，无增量信息 |
| `buy_ratio_1h_5m` | 同上 |
| `buy_ratio_4h_5m` | 同上 |

**根本原因**：这类"窗口内净方向"特征的计算起点和终点都是已知的历史价格，等价于告诉模型"过去已经发生了什么"，与用 ROC/RSI 这类滚动指标的区别在于：ROC 是当前时刻的瞬时斜率，`ret_net_30m` 是过去整段时间的合计方向——后者在模型中形成了极强但无预测力的"选择偏差"特征。

---

## 四、筛选流程

### 4.1 第一轮：快速过滤（基于相关性）

```
相关矩阵过滤：
  - 计算全部候选特征的 Pearson 相关矩阵
  - 剔除相互相关系数 > 0.90 的冗余特征（保留与 y_class 相关度更高的一个）
  - 工具：scripts/analyze_correlation.py（待实现）
```

### 4.2 第二轮：重要性排名（基于模型）

```
当前工具：scripts/analyze_importance.py（已实现）
  - 输出 XGBoost gain-based 特征重要性
  - 保留 top-N 特征（建议 N=40–60）

补充工具：
  - Permutation Importance（更稳健，不受树深度影响）
  - SHAP 值分析（可解释性更强）
```

### 4.3 第三轮：回测验证

```
对不同特征子集分别训练、回测：
  - 指标：WR、EV、MDD、Sharpe
  - 对比 baseline（当前 v1.0.2 特征集）
  - 保留胜率提升 > +2pp 或 EV 改善 > +0.05R 的特征集
```

---

## 五、UI 构想（Streamlit，阶段三，暂缓）

```
┌─────────────────────────────────────────────────┐
│  QuantX Feature Lab                              │
├──────────────┬──────────────────────────────────┤
│ 特征列表      │  回测结果                          │
│ □ rsi7_5m    │  WR: 38.2%  EV: −0.15R           │
│ ☑ rsi14_5m   │  MDD: −12.3%                      │
│ □ stoch_k_5m │  Sharpe: 0.82                     │
│ ☑ atr14_1m   │                                   │
│ ☑ vcdi_5m    │  [特征重要性图表]                   │
│ □ obv_5m     │                                   │
│ ...          │  [累计 PnL 曲线]                   │
├──────────────┴──────────────────────────────────┤
│  方向: ● LONG  ○ SHORT  ○ BOTH                   │
│  [选择预设子集 ▼]  [分析相关性]  [开始训练]        │
│  [保存为特征集 v1.0.3]  [启动试运行]              │
└─────────────────────────────────────────────────┘
```

**技术选型**：Streamlit（最快，2–3 天出原型）  
**训练触发**：调用 `scripts/train_model.py`，输出到 `models/` 指定目录  
**回测触发**：调用 `scripts/run_backtest.py`，结果写入 JSON 后渲染图表

---

## 六、实施路线图

```
阶段 1（2-3天）  扩展特征库
  → 在 indicators.py 添加新指标函数
  → 在 timeframe.py 集成到各时间周期
  → 重新生成 parquet：make build-features VERSION=v1.0.3

阶段 2（1天）  相关性 + 重要性分析
  → 实现 scripts/analyze_correlation.py
  → 运行重要性分析，输出推荐特征子集清单
  → 对比 2-3 个子集的回测结果

阶段 3（2-3天，可选）  Streamlit UI
  → app.py：特征选择 + 训练触发 + 结果展示
  → make ui 启动页面
```

**当前建议：先做阶段 1 + 阶段 2，优先价值最高的特征（⭐⭐⭐）**

---

## 七、优先级总结

按投入产出比排序，最优先的 15 个新特征：

1. `hour_sin` / `hour_cos`：时段信息无噪编码（直接解决 13:xx 死区问题）
2. `is_asia/eu/us_session`：显式交易时段标记
3. `minutes_to_funding`：资金费率结算前后的行为差异
4. `rsi7_{5m/1h}`：短周期超买超卖（比 RSI14 更敏感）
5. `adx_{5m/1h}`：趋势强度（解决 WITH 顺势信号失效问题）
6. `trend_score_1h`：将现有 4h 逻辑复用到 1h
7. `hl_position_{5m/1h}`：N 期高低点通道位置
8. `atr_ratio_5m`：波动加速/衰减（区分趋势加速 vs 超扩展回归）
9. `stoch_k_{5m/15m}`：随机指标（不同于 RSI 的超买超卖视角）
10. `vol_ma_ratio_{5m/1h}`：成交量相对均值（量能确认）
11. `rsi_diff_5m_1h`：短长周期 RSI 背离
12. `bb_pos_diff_5m_1h`：短长周期 BB 位置背离
13. `funding_pressure`：资金费率方向性压力
14. `cvd_5m`：累计成交量差值（买卖力量对比）
15. `atr14_1m_vs_5m`：微结构波动比（识别异常价格行为）

---

## 八、实施状态

- [ ] 阶段 1：在 `indicators.py` 实现新指标函数（时段/RSI7/ADX/stoch/HL位置等）
- [ ] 阶段 1：在 `timeframe.py` 集成新特征，重新生成 `features_thr005_h30_{long,short}.parquet`
- [ ] 阶段 2：实现 `scripts/analyze_correlation.py`
- [ ] 阶段 2：运行 `scripts/analyze_importance.py`，确定最终特征子集（目标 40–60 个）
- [ ] 阶段 2：重训模型，对比回测 WR / EV / MDD 与 v1.0.2 baseline
- [ ] 阶段 3：Streamlit UI（可选，暂缓）
