"""候选因子源 (v3.1.2) — 从日K数据计算, 用于 IC 筛选

每个因子 = 一个在股票日K上可计算的数值信号, 目标是预测 N 日前向收益。
数据源: 回放日K (date/open/high/low/close/volume/amount/turn/pctChg/peTTM/pbMRQ/is_trade)。

因子分组:
  baseline  第一波8个技术面因子 (v3.1.2 原始集)
  wave1     第二波: 估值/流动性/反转/风险调整/市场相对强度

wave1 中的 rel_strength_* 读取注入的 `mkt_close` 列 (等权市场指数,
由 evaluate 端通过 `factors.market_situation.build_market_close_series` 注入),
注册表契约保持 fn(df) 单参数不变。
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd

# 因子注册表: name -> 计算函数 (输入单股日K切片, 输出该日因子分 float)
FACTORS: Dict[str, Callable] = {}


def _register(name):
    def deco(fn):
        FACTORS[name] = fn
        return fn
    return deco


@_register("vp_divergence")
def vp_divergence(df: pd.DataFrame) -> float:
    """量价背离: 近5日价涨+量增=正(健康), 价涨+量缩=负(背离/危险)"""
    if len(df) < 10:
        return 0.0
    c = df["close"].values
    v = df["volume"].values
    ret5 = c[-1] / c[-6] - 1 if c[-6] > 0 else 0
    vol5 = np.mean(v[-5:]) / (np.mean(v[-10:-5]) + 1e-9)
    # 价涨 + 量增 → 正; 价涨 + 量缩 → 负 (背离); 价跌 + 量增 → 负 (恐慌放量)
    if ret5 > 0:
        return min(1.0, ret5 * 5) * (1 if vol5 > 1 else -1)
    return min(1.0, abs(ret5) * 5) * (-1 if vol5 > 1 else 0.3)


@_register("turnover_anom")
def turnover_anom(df: pd.DataFrame) -> float:
    """换手率异常: 当日换手相对20日均值放大倍数的对数"""
    if "turn" not in df.columns or len(df) < 21:
        return 0.0
    t = pd.to_numeric(df["turn"], errors="coerce").fillna(0).values
    base = np.mean(t[-21:-1])
    if base <= 0:
        return 0.0
    return float(np.log(t[-1] / base + 1e-6))


@_register("rsi_extreme")
def rsi_extreme(df: pd.DataFrame) -> float:
    """RSI 极端方向化: 超买(>70)→负(回落风险), 超卖(<30)→正(反弹机会)"""
    if len(df) < 15:
        return 0.0
    c = df["close"].values
    delta = np.diff(c[-15:])
    gain = np.where(delta > 0, delta, 0).mean()
    loss = np.where(delta < 0, -delta, 0).mean()
    if loss == 0:
        return -1.0  # 持续上涨, 超买
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    if rsi > 70:
        return -(rsi - 70) / 30
    if rsi < 30:
        return (30 - rsi) / 30
    return 0.0


@_register("momentum_5")
def momentum_5(df: pd.DataFrame) -> float:
    """5日动量"""
    if len(df) < 6:
        return 0.0
    c = df["close"].values
    return float(c[-1] / c[-6] - 1) if c[-6] > 0 else 0.0


@_register("momentum_20")
def momentum_20(df: pd.DataFrame) -> float:
    """20日动量"""
    if len(df) < 21:
        return 0.0
    c = df["close"].values
    return float(c[-1] / c[-21] - 1) if c[-21] > 0 else 0.0


@_register("vol_spike")
def vol_spike(df: pd.DataFrame) -> float:
    """波动率突增: ATR 相对20日均值"""
    if len(df) < 21 or "high" not in df.columns:
        return 0.0
    high, low, close = (df["high"].values, df["low"].values, df["close"].values)
    tr = np.maximum(high - low, np.abs(high - np.append(close[0], close[:-1])))
    atr = np.mean(tr[-5:])
    base = np.mean(tr[-21:-5])
    if base <= 0:
        return 0.0
    return float(np.log(atr / base + 1e-6))


@_register("bias_ma20")
def bias_ma20(df: pd.DataFrame) -> float:
    """偏离MA20 (乖离率)"""
    if len(df) < 21:
        return 0.0
    c = df["close"].values
    ma20 = np.mean(c[-20:])
    return float(c[-1] / ma20 - 1) if ma20 > 0 else 0.0


@_register("volume_ratio")
def volume_ratio(df: pd.DataFrame) -> float:
    """量比 (当日量/5日均量)"""
    if len(df) < 6:
        return 0.0
    v = df["volume"].values
    base = np.mean(v[-6:-1])
    return float(v[-1] / base - 1) if base > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# wave1 — 第二波因子 (估值/流动性/反转/风险调整/市场相对强度)
# 使用回放中此前未利用的列: peTTM / pbMRQ / amount / pctChg
# ═══════════════════════════════════════════════════════════════

@_register("ep")
def ep(df: pd.DataFrame) -> float:
    """盈利收益率 1/PE (winsorized)。PE<=0 (亏损) → 0。高 = 便宜"""
    if "peTTM" not in df.columns:
        return 0.0
    pe = pd.to_numeric(df["peTTM"], errors="coerce").iloc[-1]
    if not np.isfinite(pe) or pe <= 0:
        return 0.0
    return float(1.0 / np.clip(pe, 0.1, 200))


@_register("bp")
def bp(df: pd.DataFrame) -> float:
    """市净率倒数 1/PB (winsorized)。PB<=0 → 0。高 = 便宜"""
    if "pbMRQ" not in df.columns:
        return 0.0
    pb = pd.to_numeric(df["pbMRQ"], errors="coerce").iloc[-1]
    if not np.isfinite(pb) or pb <= 0:
        return 0.0
    return float(1.0 / np.clip(pb, 0.05, 50))


@_register("pe_pct_20d")
def pe_pct_20d(df: pd.DataFrame) -> float:
    """当日PE在自身20日内的分位数 (0=自身最便宜)。高=相对自身变贵 → 预期负IC"""
    if "peTTM" not in df.columns or len(df) < 21:
        return 0.5
    pe = pd.to_numeric(df["peTTM"], errors="coerce")
    cur = pe.iloc[-1]
    hist = pe.iloc[-21:-1]
    hist = hist[(hist > 0) & hist.notna()]
    if len(hist) < 10 or not np.isfinite(cur) or cur <= 0:
        return 0.5
    return float((hist < cur).mean())


@_register("pb_pct_20d")
def pb_pct_20d(df: pd.DataFrame) -> float:
    """当日PB在自身20日内的分位数 (0=自身最便宜)。高=相对自身变贵 → 预期负IC"""
    if "pbMRQ" not in df.columns or len(df) < 21:
        return 0.5
    pb = pd.to_numeric(df["pbMRQ"], errors="coerce")
    cur = pb.iloc[-1]
    hist = pb.iloc[-21:-1]
    hist = hist[(hist > 0) & hist.notna()]
    if len(hist) < 10 or not np.isfinite(cur) or cur <= 0:
        return 0.5
    return float((hist < cur).mean())


@_register("amihud_illiq")
def amihud_illiq(df: pd.DataFrame) -> float:
    """Amihud 非流动性: 近5日 |日收益|/成交额 均值。高 = 流动性差"""
    if "amount" not in df.columns or len(df) < 6:
        return 0.0
    c = pd.to_numeric(df["close"], errors="coerce").values
    a = pd.to_numeric(df["amount"], errors="coerce").values
    d = np.abs(np.diff(c[-6:])) / np.maximum(c[-6:-1], 1e-9)
    amt = a[-5:]
    m = np.isfinite(d) & np.isfinite(amt) & (amt > 0)
    if m.sum() < 3:
        return 0.0
    return float(np.mean(d[m] / amt[m]))


@_register("amount_ratio_5d")
def amount_ratio_5d(df: pd.DataFrame) -> float:
    """成交额比: 当日amount/前5日均额 - 1"""
    if "amount" not in df.columns or len(df) < 6:
        return 0.0
    a = pd.to_numeric(df["amount"], errors="coerce").values
    base = np.mean(a[-6:-1])
    return float(a[-1] / base - 1) if np.isfinite(base) and base > 0 else 0.0


@_register("turn_pct_20d")
def turn_pct_20d(df: pd.DataFrame) -> float:
    """当日换手在自身20日内的分位数 (筹码活跃度)"""
    if "turn" not in df.columns or len(df) < 21:
        return 0.5
    t = pd.to_numeric(df["turn"], errors="coerce").fillna(0).values
    cur = t[-1]
    hist = t[-21:-1]
    return float((hist < cur).mean())


@_register("sharpe_20")
def sharpe_20(df: pd.DataFrame) -> float:
    """风险调整动量: 5日收益 / (20日日收益std × √5)"""
    if len(df) < 21:
        return 0.0
    c = df["close"].values
    rets = np.diff(c[-21:]) / np.maximum(c[-21:-1], 1e-9)
    std = np.std(rets)
    r5 = c[-1] / c[-6] - 1 if c[-6] > 0 else 0.0
    if std < 1e-9:
        return 0.0
    return float(r5 / (std * np.sqrt(5)))


@_register("reversal_1d")
def reversal_1d(df: pd.DataFrame) -> float:
    """短期反转: -1 × 昨日收益。A股散户市反转效应显著 → 预期正IC"""
    if len(df) < 2:
        return 0.0
    c = df["close"].values
    ret = c[-1] / c[-2] - 1 if c[-2] > 0 else 0.0
    return float(-ret)


@_register("vol_ratio_20")
def vol_ratio_20(df: pd.DataFrame) -> float:
    """波动体制切换: 5日ATR/20日ATR - 1"""
    if len(df) < 21 or "high" not in df.columns:
        return 0.0
    high, low, close = (df["high"].values, df["low"].values, df["close"].values)
    tr = np.maximum(high - low, np.abs(high - np.append(close[0], close[:-1])))
    a5 = np.mean(tr[-5:])
    a20 = np.mean(tr[-20:])
    if a20 <= 0:
        return 0.0
    return float(a5 / a20 - 1)


@_register("gap_strength")
def gap_strength(df: pd.DataFrame) -> float:
    """跳空强度: 今日开盘/昨日收盘 - 1"""
    if len(df) < 2 or "open" not in df.columns:
        return 0.0
    op = df["open"].values
    cl = df["close"].values
    prev = cl[-2]
    return float(op[-1] / prev - 1) if prev > 0 else 0.0


@_register("rel_strength_5d")
def rel_strength_5d(df: pd.DataFrame) -> float:
    """市场相对强度: 个股5日收益 - 市场5日收益 (领先/落后大盘)"""
    if "mkt_close" not in df.columns or len(df) < 6:
        return 0.0
    c = df["close"].values
    m = df["mkt_close"].values
    rc = c[-1] / c[-6] - 1 if c[-6] > 0 else 0.0
    rm = m[-1] / m[-6] - 1 if m[-6] > 0 else 0.0
    return float(rc - rm)


@_register("rel_strength_20d")
def rel_strength_20d(df: pd.DataFrame) -> float:
    """市场相对强度: 个股20日收益 - 市场20日收益"""
    if "mkt_close" not in df.columns or len(df) < 21:
        return 0.0
    c = df["close"].values
    m = df["mkt_close"].values
    rc = c[-1] / c[-21] - 1 if c[-21] > 0 else 0.0
    rm = m[-1] / m[-21] - 1 if m[-21] > 0 else 0.0
    return float(rc - rm)


# 因子分组
BASELINE_FACTORS = [
    "vp_divergence", "turnover_anom", "rsi_extreme", "momentum_5",
    "momentum_20", "vol_spike", "bias_ma20", "volume_ratio",
]
WAVE1_FACTORS = [
    "ep", "bp", "pe_pct_20d", "pb_pct_20d", "amihud_illiq", "amount_ratio_5d",
    "turn_pct_20d", "sharpe_20", "reversal_1d", "vol_ratio_20",
    "gap_strength", "rel_strength_5d", "rel_strength_20d",
]
FACTOR_GROUPS = {"baseline": BASELINE_FACTORS, "wave1": WAVE1_FACTORS}


def factor_list(group: str = None) -> list:
    """因子名列表。group=None 返回全部 (默认, 向后兼容)。"""
    if group is None:
        return list(FACTORS.keys())
    return FACTOR_GROUPS.get(group, [])


def compute_factors(df: pd.DataFrame) -> Dict[str, float]:
    """对单只股票在 asof 日计算所有因子分"""
    return {name: fn(df) for name, fn in FACTORS.items()}
