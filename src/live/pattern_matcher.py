"""
信号形态识别模块

在信号触发后，判断当前市场状态是否匹配已知的成功形态。
不影响信号触发，仅用于标记和后续分析。
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class PatternMatch:
    """形态匹配结果"""
    code: str  # 形态编码
    name: str  # 形态名称
    confidence: float  # 置信度 0-1
    reason: str  # 匹配原因


@dataclass
class MarketContext:
    """市场上下文特征"""
    # 趋势特征
    ema20: float
    ema60: float
    price: float
    trend_strength: float  # (price - ema60) / ema60

    # 动量特征
    ret_5m: float
    ret_15m: float
    ret_30m: float
    ret_1h: float

    # 波动特征
    atr_ratio: float  # 当前波动 / 历史平均

    # 位置特征
    price_vs_1h_high: float  # (price - 1h_high) / 1h_high
    price_vs_1h_low: float  # (price - 1h_low) / 1h_low


def extract_context(klines: dict[str, pd.DataFrame]) -> MarketContext:
    """从 K线数据提取市场上下文"""
    df_5m = klines["5m"]
    df_1h = klines["1h"]

    current = df_5m.iloc[-1]

    # 计算 EMA
    close_5m = df_5m["close"]
    ema20 = close_5m.ewm(span=20, adjust=False).mean().iloc[-1]
    ema60 = close_5m.ewm(span=60, adjust=False).mean().iloc[-1]

    # 计算收益率
    ret_5m = (current["close"] / df_5m.iloc[-2]["close"] - 1)
    ret_15m = (current["close"] / df_5m.iloc[-4]["close"] - 1)
    ret_30m = (current["close"] / df_5m.iloc[-7]["close"] - 1)
    ret_1h = (current["close"] / df_1h.iloc[-2]["close"] - 1)

    # 计算波动
    atr_5m = (df_5m["high"] - df_5m["low"]).rolling(14).mean()
    current_range = current["high"] - current["low"]
    atr_ratio = current_range / atr_5m.iloc[-1] if atr_5m.iloc[-1] > 0 else 1.0

    # 计算位置
    high_1h = df_5m["high"].iloc[-12:].max()
    low_1h = df_5m["low"].iloc[-12:].min()
    price_vs_1h_high = (current["close"] - high_1h) / high_1h
    price_vs_1h_low = (current["close"] - low_1h) / low_1h

    return MarketContext(
        ema20=ema20,
        ema60=ema60,
        price=current["close"],
        trend_strength=(current["close"] - ema60) / ema60,
        ret_5m=ret_5m,
        ret_15m=ret_15m,
        ret_30m=ret_30m,
        ret_1h=ret_1h,
        atr_ratio=atr_ratio,
        price_vs_1h_high=price_vs_1h_high,
        price_vs_1h_low=price_vs_1h_low,
    )


# ========== 成功形态规则库 ==========

def match_downtrend_start_short(ctx: MarketContext) -> PatternMatch | None:
    """S1: 下跌趋势启动初期顺势 SHORT"""
    if ctx.price > ctx.ema20:
        return None
    if ctx.ret_15m > -0.001:  # 需要明显下跌
        return None
    if ctx.trend_strength > -0.005:  # 需要在 EMA60 下方
        return None

    confidence = min(1.0, abs(ctx.ret_15m) * 100)  # 跌幅越大越可信
    return PatternMatch(
        code="S1",
        name="下跌趋势启动初期顺势",
        confidence=confidence,
        reason=f"价格 {ctx.price:.2f} < EMA20 {ctx.ema20:.2f}, 15m跌幅 {ctx.ret_15m*100:.2f}%",
    )


def match_bounce_rejection_short(ctx: MarketContext) -> PatternMatch | None:
    """S2: 小反弹后重回下跌 SHORT"""
    if ctx.price > ctx.ema60:
        return None
    if ctx.ret_5m > 0.002:  # 当前不能在涨
        return None
    if ctx.ret_30m > 0:  # 30分钟需要是跌的
        return None
    if ctx.price_vs_1h_high > -0.003:  # 需要离1h高点有距离
        return None

    confidence = 0.7
    return PatternMatch(
        code="S2",
        name="小反弹后重回下跌",
        confidence=confidence,
        reason=f"价格 {ctx.price:.2f} < EMA60 {ctx.ema60:.2f}, 30m跌幅 {ctx.ret_30m*100:.2f}%",
    )


def match_spike_reversal_short(ctx: MarketContext) -> PatternMatch | None:
    """S3: 急涨后快速反转 SHORT"""
    if ctx.ret_15m < 0.005:  # 15分钟需要有明显涨幅
        return None
    if ctx.ret_5m > 0:  # 但最近5分钟开始回落
        return None
    if ctx.atr_ratio < 1.5:  # 需要高波动
        return None

    confidence = min(1.0, ctx.ret_15m * 50)
    return PatternMatch(
        code="S3",
        name="急涨后快速反转",
        confidence=confidence,
        reason=f"15m涨幅 {ctx.ret_15m*100:.2f}%, 5m开始回落 {ctx.ret_5m*100:.2f}%",
    )


def match_bounce_top_short(ctx: MarketContext) -> PatternMatch | None:
    """S4: 小反弹顶部快速向下 SHORT"""
    if ctx.ret_15m < 0.003:  # 15分钟需要有反弹
        return None
    if ctx.ret_5m > 0:  # 但最近5分钟开始回落
        return None
    if ctx.price_vs_1h_high > -0.002:  # 需要接近1h高点
        return None

    confidence = 0.75
    return PatternMatch(
        code="S4",
        name="小反弹顶部快速向下",
        confidence=confidence,
        reason=f"15m反弹 {ctx.ret_15m*100:.2f}%, 5m回落 {ctx.ret_5m*100:.2f}%",
    )


def match_low_bounce_long(ctx: MarketContext) -> PatternMatch | None:
    """L1: 低位真实反弹 LONG"""
    if ctx.price > ctx.ema60:
        return None
    if ctx.ret_5m < 0.001:  # 需要有上涨动能
        return None
    if ctx.price_vs_1h_low > 0.005:  # 需要接近1h低点
        return None

    confidence = 0.8
    return PatternMatch(
        code="L1",
        name="低位真实反弹",
        confidence=confidence,
        reason=f"接近1h低点 ({ctx.price_vs_1h_low*100:.2f}%), 5m涨幅 {ctx.ret_5m*100:.2f}%",
    )


def match_local_bounce_long(ctx: MarketContext) -> PatternMatch | None:
    """L2: 下跌后局部反弹 LONG"""
    if ctx.ret_30m > 0:  # 30分钟需要是跌的
        return None
    if ctx.ret_5m < 0.001:  # 但最近5分钟开始反弹
        return None
    if ctx.price > ctx.ema20:  # 不能在 EMA20 上方
        return None

    confidence = 0.6
    return PatternMatch(
        code="L2",
        name="下跌后局部反弹",
        confidence=confidence,
        reason=f"30m跌幅 {ctx.ret_30m*100:.2f}%, 5m反弹 {ctx.ret_5m*100:.2f}%",
    )


# ========== 主匹配函数 ==========

def match_patterns(
    klines: dict[str, pd.DataFrame],
    direction: str,
) -> list[PatternMatch]:
    """
    匹配所有已知成功形态

    Args:
        klines: K线数据字典
        direction: "long" 或 "short"

    Returns:
        匹配到的形态列表（按置信度降序）
    """
    ctx = extract_context(klines)

    matchers = {
        "short": [
            match_downtrend_start_short,
            match_bounce_rejection_short,
            match_spike_reversal_short,
            match_bounce_top_short,
        ],
        "long": [
            match_low_bounce_long,
            match_local_bounce_long,
        ],
    }

    matches = []
    for matcher in matchers.get(direction, []):
        result = matcher(ctx)
        if result:
            matches.append(result)

    # 按置信度降序排序
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches
