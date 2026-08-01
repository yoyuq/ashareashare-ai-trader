"""A股 OHLCV 数据清洗 (v3.0)

2026 量化数据工程最佳实践: 停牌/涨跌停零价填充会拉低均线/波动率等价格类指标,
生成虚假交易信号。清洗优先:
  - 价格为 0/负 (停牌/数据缺失) → 转 NaN
  - 价格列前向填充 (沿用上一交易日有效价, 不引入虚拟价格)
  - 标注 is_trade (该日是否真实成交)
"""

import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 OHLCV: 零价→NaN→ffill, 标注 is_trade。返回副本, 不修改入参。"""
    if df is None or df.empty:
        return df.copy() if df is not None else df

    out = df.copy()
    price_cols = ["open", "high", "low", "close"]

    # 1. 价格 0/负 → NaN (停牌/数据缺失占位)
    for col in price_cols:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.mask(s <= 0)

    # 2. 标注真实成交日 (close 有效 = 有交易)
    if "is_trade" not in out.columns:
        out["is_trade"] = out["close"].notna().astype(int)

    # 3. 停牌日价格前向填充 (沿用上一有效收盘价, 不虚构)
    for col in price_cols:
        if col in out.columns:
            out[col] = out[col].ffill()

    return out
