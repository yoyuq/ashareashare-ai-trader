"""市场形势因子 (v3.1.2) — 从全市场截面构造"当前形势"信号

现有 `factors/sources.py` 是**个股截面因子** (预测哪只股票跑赢)。
这里补充**市场级因子** — 每个交易日一个数值, 描述市场整体状态,
正是 AI 判断"当前形势"所需的输入。用时间序列相关评估:
因子值(T) 对 市场 N 日前向收益 的秩相关。

市场代理: replay 无指数, 用等权日收益复利构造 (标准截面代理做法)。

核心函数:
  build_market_close_series(df) -> Series   等权市场指数 (index=Timestamp)
  build_market_panel(df)      -> DataFrame  每交易日市场信号 (index=date)
"""

import numpy as np
import pandas as pd


def build_market_close_series(df: pd.DataFrame) -> pd.Series:
    """等权市场指数 — 截面日收益均值复利。

    Args:
        df: 全市场日K (date/symbol/close, 含全部股票)

    Returns:
        Series indexed by Timestamp, close-like 指数值 (首日=1.0)
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    piv = d.pivot_table(index="date", columns="symbol", values="close")
    mkt_ret = piv.pct_change().mean(axis=1)
    mkt_ret = mkt_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1 + mkt_ret).cumprod()


def build_market_panel(
    df: pd.DataFrame, pe_pctile_window: int = 60
) -> pd.DataFrame:
    """构造每交易日的市场形势信号面板。

    Args:
        df: 全市场日K (含 close/amount/peTTM/pbMRQ/pctChg)
        pe_pctile_window: 估值分位回看窗口

    Returns:
        DataFrame indexed by Timestamp, 每列一个市场信号:
          mkt_ret_1d / 5d / 20d      等权市场动量
          breadth_advance            当日上涨家数占比
          breadth_above_ma20         站上MA20家数占比
          breadth_avg_10d            10日平均上涨占比 (持续广度)
          mkt_dispersion_20d         20日收益截面离散度
          mkt_skew_1d                当日收益分布偏度
          mkt_amount_ratio           全市场成交额/20日均 (流动性)
          mkt_median_pe_pctile       中位PE在窗口内分位 (市场估值)
          mkt_median_pb_pctile       中位PB在窗口内分位
          limit_up_ratio             涨停家数占比 (pctChg>=9.5)
          limit_down_ratio           跌停家数占比 (pctChg<=-9.5)
          new_high_20d_ratio         创20日新高家数占比
          new_low_20d_ratio          创20日新低家数占比
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    piv_c = d.pivot_table(index="date", columns="symbol", values="close")
    piv_r = piv_c.pct_change()
    piv_amt = d.pivot_table(index="date", columns="symbol", values="amount")

    mkt_cum = (1 + piv_r.mean(axis=1).fillna(0)).cumprod()

    out = pd.DataFrame(index=piv_c.index)
    out["mkt_ret_1d"] = piv_r.mean(axis=1)
    out["mkt_ret_5d"] = mkt_cum.pct_change(5)
    out["mkt_ret_20d"] = mkt_cum.pct_change(20)
    out["breadth_advance"] = (piv_r > 0).mean(axis=1)
    out["breadth_avg_10d"] = out["breadth_advance"].rolling(10).mean()
    out["mkt_dispersion_20d"] = piv_c.pct_change(20).std(axis=1)
    out["mkt_skew_1d"] = piv_r.skew(axis=1)
    out["mkt_amount_ratio"] = piv_amt.sum(axis=1) / piv_amt.sum(axis=1).rolling(20).mean()

    # 估值水平: 中位数对异常值稳健
    med_pe = piv_c.replace(np.nan, np.nan)
    if "peTTM" in d.columns:
        piv_pe = d.pivot_table(index="date", columns="symbol", values="peTTM")
        med_pe = piv_pe.replace([np.inf, -np.inf], np.nan).median(axis=1)
        out["mkt_median_pe_pctile"] = med_pe.rolling(pe_pctile_window).rank(pct=True)
    if "pbMRQ" in d.columns:
        piv_pb = d.pivot_table(index="date", columns="symbol", values="pbMRQ")
        med_pb = piv_pb.replace([np.inf, -np.inf], np.nan).median(axis=1)
        out["mkt_median_pb_pctile"] = med_pb.rolling(pe_pctile_window).rank(pct=True)

    # 涨跌停占比
    out["limit_up_ratio"] = (piv_r >= 0.095).mean(axis=1)
    out["limit_down_ratio"] = (piv_r <= -0.095).mean(axis=1)

    # 长表计算 MA20 广度 / 20日新高新低 (避免 5107 列宽表 rolling 开销)
    g = d.sort_values("date").groupby("symbol")
    d["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    d["above_ma20"] = d["close"] > d["ma20"]
    d["high20"] = g["close"].transform(lambda s: s.rolling(20).max())
    d["at_high20"] = d["close"] == d["high20"]
    d["low20"] = g["close"].transform(lambda s: s.rolling(20).min())
    d["at_low20"] = d["close"] == d["low20"]
    daily = d.groupby("date").agg(
        breadth_above_ma20=("above_ma20", "mean"),
        new_high_20d_ratio=("at_high20", "mean"),
        new_low_20d_ratio=("at_low20", "mean"),
    )
    out = out.join(daily)

    return out


MARKET_FACTOR_NAMES: list = [
    "mkt_ret_1d", "mkt_ret_5d", "mkt_ret_20d",
    "breadth_advance", "breadth_above_ma20", "breadth_avg_10d",
    "mkt_dispersion_20d", "mkt_skew_1d", "mkt_amount_ratio",
    "mkt_median_pe_pctile", "mkt_median_pb_pctile",
    "limit_up_ratio", "limit_down_ratio",
    "new_high_20d_ratio", "new_low_20d_ratio",
]
