#!/usr/bin/env python3
"""
数据探索：统计不同时间窗口和阈值下的样本分布

目的：
1. 找到合适的标签阈值（既不太频繁也不太罕见）
2. 了解不同时间窗口的价格运动特性
3. 为模型设计提供数据支持
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.features.loader import load_klines


def compute_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    计算未来 N 分钟内的最大涨幅和最大跌幅

    Args:
        df: K线数据，必须包含 high, low, close 列
        horizons: 时间窗口列表（分钟数），例如 [15, 30, 60, 120]

    Returns:
        原始 df + 新增列：
        - max_ret_{horizon}m: 未来 {horizon} 分钟内的最大涨幅
        - min_ret_{horizon}m: 未来 {horizon} 分钟内的最大跌幅（负数）
    """
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    for horizon in horizons:
        max_rets = []
        min_rets = []

        for i in range(len(df)):
            if i + horizon >= len(df):
                # 末尾样本没有足够的未来数据
                max_rets.append(np.nan)
                min_rets.append(np.nan)
            else:
                current_price = close[i]
                future_highs = high[i+1 : i+1+horizon]
                future_lows = low[i+1 : i+1+horizon]

                max_ret = (future_highs.max() / current_price - 1) * 100  # 百分比
                min_ret = (future_lows.min() / current_price - 1) * 100   # 百分比（负数）

                max_rets.append(max_ret)
                min_rets.append(min_ret)

        df[f'max_ret_{horizon}m'] = max_rets
        df[f'min_ret_{horizon}m'] = min_rets

    return df


def analyze_threshold_distribution(df: pd.DataFrame, horizons: list[int], thresholds: list[float]):
    """
    统计不同阈值下的样本分布

    Args:
        df: 包含 max_ret 和 min_ret 列的数据
        horizons: 时间窗口列表
        thresholds: 阈值列表（百分比），例如 [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    """
    print("=" * 100)
    print("样本分布统计")
    print("=" * 100)
    print()

    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"时间窗口: {horizon} 分钟")
        print(f"{'='*50}\n")

        max_ret_col = f'max_ret_{horizon}m'
        min_ret_col = f'min_ret_{horizon}m'

        # 移除 NaN
        valid_df = df[[max_ret_col, min_ret_col]].dropna()
        total_samples = len(valid_df)

        print(f"有效样本数: {total_samples:,}\n")

        # 统计表格
        print(f"{'阈值':<8} {'涨超阈值':<12} {'跌超阈值':<12} {'双向波动':<12} {'横盘震荡':<12}")
        print(f"{'(%)':<8} {'样本数(占比)':<12} {'样本数(占比)':<12} {'样本数(占比)':<12} {'样本数(占比)':<12}")
        print("-" * 80)

        for threshold in thresholds:
            # 计算各类样本数
            long_hit = (valid_df[max_ret_col] >= threshold).sum()
            short_hit = (valid_df[min_ret_col] <= -threshold).sum()
            both_hit = ((valid_df[max_ret_col] >= threshold) &
                       (valid_df[min_ret_col] <= -threshold)).sum()
            neither_hit = ((valid_df[max_ret_col] < threshold) &
                          (valid_df[min_ret_col] > -threshold)).sum()

            # 计算占比
            long_pct = long_hit / total_samples * 100
            short_pct = short_hit / total_samples * 100
            both_pct = both_hit / total_samples * 100
            neither_pct = neither_hit / total_samples * 100

            print(f"{threshold:<8.1f} "
                  f"{long_hit:>6,} ({long_pct:>5.1f}%)  "
                  f"{short_hit:>6,} ({short_pct:>5.1f}%)  "
                  f"{both_hit:>6,} ({both_pct:>5.1f}%)  "
                  f"{neither_hit:>6,} ({neither_pct:>5.1f}%)")

        # 统计描述
        print(f"\n描述性统计:")
        print(f"  最大涨幅: mean={valid_df[max_ret_col].mean():.2f}%, "
              f"median={valid_df[max_ret_col].median():.2f}%, "
              f"std={valid_df[max_ret_col].std():.2f}%, "
              f"max={valid_df[max_ret_col].max():.2f}%")
        print(f"  最大跌幅: mean={valid_df[min_ret_col].mean():.2f}%, "
              f"median={valid_df[min_ret_col].median():.2f}%, "
              f"std={valid_df[min_ret_col].std():.2f}%, "
              f"min={valid_df[min_ret_col].min():.2f}%")


def analyze_time_to_target(df: pd.DataFrame, horizons: list[int], threshold: float = 1.0):
    """
    统计达到目标阈值所需的时间

    Args:
        df: K线数据
        horizons: 时间窗口列表
        threshold: 目标阈值（百分比）
    """
    print(f"\n\n{'='*100}")
    print(f"达到 ±{threshold}% 所需时间统计")
    print(f"{'='*100}\n")

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    for horizon in horizons:
        print(f"\n时间窗口: {horizon} 分钟")
        print("-" * 50)

        long_times = []  # 达到 +threshold% 所需时间
        short_times = []  # 达到 -threshold% 所需时间

        for i in range(len(df) - horizon):
            current_price = close[i]
            target_up = current_price * (1 + threshold / 100)
            target_down = current_price * (1 - threshold / 100)

            # 找到第一次触及目标的时间
            for t in range(1, horizon + 1):
                if high[i + t] >= target_up:
                    long_times.append(t)
                    break

            for t in range(1, horizon + 1):
                if low[i + t] <= target_down:
                    short_times.append(t)
                    break

        if long_times:
            print(f"  涨到 +{threshold}%: "
                  f"平均 {np.mean(long_times):.1f} 分钟, "
                  f"中位数 {np.median(long_times):.0f} 分钟, "
                  f"成功率 {len(long_times)/(len(df)-horizon)*100:.1f}%")
        else:
            print(f"  涨到 +{threshold}%: 无样本")

        if short_times:
            print(f"  跌到 -{threshold}%: "
                  f"平均 {np.mean(short_times):.1f} 分钟, "
                  f"中位数 {np.median(short_times):.0f} 分钟, "
                  f"成功率 {len(short_times)/(len(df)-horizon)*100:.1f}%")
        else:
            print(f"  跌到 -{threshold}%: 无样本")


def main():
    print("加载 1 分钟 K 线数据...")
    data_root = Path(__file__).parent.parent / "data" / "raw"
    df = load_klines('1m', data_root)

    # 添加 timestamp 列（如果没有的话）
    if 'timestamp' not in df.columns:
        df['timestamp'] = df.index

    print(f"数据范围: {df.index.min()} 到 {df.index.max()}")
    print(f"总样本数: {len(df):,}\n")

    # 定义要分析的时间窗口和阈值
    horizons = [15, 30, 60, 120]  # M15, M30, H1, H2
    thresholds = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]  # 不同的阈值

    # 计算未来收益
    print("计算未来收益...")
    df = compute_forward_returns(df, horizons)

    # 分析样本分布
    analyze_threshold_distribution(df, horizons, thresholds)

    # 分析达到目标所需时间
    analyze_time_to_target(df, horizons, threshold=1.0)

    print("\n" + "="*100)
    print("分析完成")
    print("="*100)


if __name__ == '__main__':
    main()
