"""市场情绪温度计 — 恐慌/贪婪复合指标 (v3.2)

目标: 让 AI 判断"当前市场情绪处于什么位置", 这是"当前形势"的重要维度。

数据可得性:
  - 外部新闻/融资融券/互动易 (akshare EM 端点) 需要 China proxy, 网络外不可靠;
  - 本模块用**回放日K可复算的微观结构成分**构造 CNN 式 Fear & Greed 复合指标,
    每个成分有清晰的经济学含义, 且可复算/可回测/可实时注入。
  - 这是情绪的真实代理 (量价行为), 非新闻文本情绪。

成分 (每日, 从 build_market_panel 派生, 正=贪婪/热):
  sent_limit_up     涨停占比 z              (连板热度)
  sent_limit_down   -跌停占比 z             (恐慌抛售)
  sent_breadth      上涨家数占比 z           (普涨/普跌)
  sent_newhigh      20日新高-新低占比差 z    (广度推进/退潮)
  sent_volume       全市场成交额比 z         (量能参与度)
  sent_skew         当日收益分布偏度 z       (散户情绪, 正偏=亢奋)
  sent_ret5         市场5日收益 z            (短期温度)
  sent_valuation    中位PE分位 z            (越贵越贪婪, 越便宜越恐慌)

复合: 成分等权 z 均值 → min-max 到 0-100.
标签: ≤20 极度恐慌 / ≤40 恐慌 / ≤60 中性 / ≤80 贪婪 / >80 极度贪婪
"""

import numpy as np
import pandas as pd

from factors.market_situation import build_market_panel  # noqa: E402


SENTIMENT_COMPONENTS = [
    "sent_limit_up", "sent_limit_down", "sent_breadth", "sent_newhigh",
    "sent_volume", "sent_skew", "sent_ret5", "sent_valuation",
]


def build_sentiment_panel(df: pd.DataFrame) -> pd.DataFrame:
    """从全市场日K构造每日情绪成分 + 复合恐慌/贪婪分数。

    Args:
        df: 全市场日K (date/symbol/close/amount/peTTM/pbMRQ/pctChg)

    Returns:
        DataFrame indexed by Timestamp:
          8 个 sent_* 成分 (z-score) + `sent_composite` (0-100 恐慌/贪婪分)
    """
    mp = build_market_panel(df)

    out = pd.DataFrame(index=mp.index)
    out["sent_limit_up"] = mp["limit_up_ratio"]
    out["sent_limit_down"] = -mp["limit_down_ratio"]  # 负号: 跌停多=恐慌
    out["sent_breadth"] = mp["breadth_advance"]
    out["sent_newhigh"] = mp["new_high_20d_ratio"] - mp["new_low_20d_ratio"]
    out["sent_volume"] = mp["mkt_amount_ratio"]
    out["sent_skew"] = mp["mkt_skew_1d"]
    out["sent_ret5"] = mp["mkt_ret_5d"]
    out["sent_valuation"] = mp["mkt_median_pe_pctile"]  # 贵=贪婪

    # z-score (全窗口标准化 — 研究用; 实时注入用滚动窗口, 见 format_live_sentiment)
    z = out.sub(out.mean()).div(out.std().replace(0, np.nan)).fillna(0.0)
    comp = z.mean(axis=1)
    lo, hi = comp.min(), comp.max()
    score = (comp - lo) / (hi - lo + 1e-12) * 100
    out["sent_composite"] = score
    # 平滑口径: 3日滚动均 — 原始分逐日剧烈摆动, 平滑分更代表"当前情绪位置"
    out["sent_composite_smoothed"] = score.rolling(3, min_periods=2).mean()

    for c in SENTIMENT_COMPONENTS:
        out[c] = z[c]
    return out


def sentiment_label(score: float) -> str:
    if score <= 20:
        return "极度恐慌"
    if score <= 40:
        return "恐慌"
    if score <= 60:
        return "中性"
    if score <= 80:
        return "贪婪"
    return "极度贪婪"


def format_sentiment(panel: pd.DataFrame, asof: pd.Timestamp = None) -> str:
    """把最近一日的情绪快照格式化为 LLM 可读的一段中文。"""
    if asof is None:
        asof = panel.index[-1]
    if asof not in panel.index:
        idx = panel.index[panel.index <= asof]
        if len(idx) == 0:
            return "情绪数据不足"
        asof = idx[-1]
    row = panel.loc[asof]
    # 主口径用平滑分 (更稳定), 附原始分
    sc = float(row["sent_composite_smoothed"]) if "sent_composite_smoothed" in row else float(row["sent_composite"])
    raw = float(row["sent_composite"])
    label = sentiment_label(sc)
    parts = []
    # 只报告极端成分 (|z|>0.8), 减少噪音
    for c in SENTIMENT_COMPONENTS:
        v = float(row[c])
        if abs(v) > 0.8:
            zh = {
                "sent_limit_up": "涨停占比", "sent_limit_down": "跌停抛压",
                "sent_breadth": "上涨广度", "sent_newhigh": "新高推进",
                "sent_volume": "成交活跃", "sent_skew": "收益偏态",
                "sent_ret5": "短期动量", "sent_valuation": "估值水平",
            }[c]
            parts.append(f"{zh}{'+' if v > 0 else '-'}{abs(v):.1f}")
    detail = ", ".join(parts) if parts else "各成分接近中性"
    return (
        f"市场情绪温度计 {sc:.0f}/100 ({label}, 平滑3日; 当日 {raw:.0f}), "
        f"特征成分: {detail}"
    )


def live_sentiment_from_snapshot(snap: dict) -> float:
    """从实时市场快照计算 0-100 恐慌/贪婪分 (无历史z, 用固定锚点标定).

    组件锚点 (权重: 广度0.30 + 涨跌幅0.25 + 涨停0.15 + 跌停0.15 + 估值0.15):
      breadth  pct_up 0→0分, 0.5→50, 1→100
      avg_pct  ±3% → 0/100, 0% → 50
      limit_up  涨停占比 5% → 100
      limit_down 跌停占比 5% → 0 (减分)
      median_pe  10~40 映射 0~100 (低PE=便宜=恐慌)
    缺失成分自动降权重 (在可得成分间重归一)。
    """
    n = int(snap.get("n_stocks", 0) or 1)
    scores, weights = [], []
    if "pct_up" in snap:
        scores.append(np.clip(snap["pct_up"], 0, 1) * 100)
        weights.append(0.30)
    if "avg_pct" in snap:
        scores.append(np.clip(snap["avg_pct"] / 3.0, -1, 1) * 50 + 50)
        weights.append(0.25)
    if "limit_up" in snap:
        scores.append(np.clip((snap["limit_up"] / n) * 100 * 2, 0, 100))
        weights.append(0.15)
    if "limit_down" in snap:
        scores.append(100 - np.clip((snap["limit_down"] / n) * 100 * 2, 0, 100))
        weights.append(0.15)
    if snap.get("median_pe") and snap["median_pe"] > 0:
        scores.append(np.clip((snap["median_pe"] - 10) / 30, 0, 1) * 100)
        weights.append(0.15)
    if not scores:
        return float("nan")
    w = np.array(weights)
    w = w / w.sum()
    return float(np.dot(scores, w))


def format_live_sentiment_snapshot(snap: dict) -> str:
    """实时快照 → 一行情绪文本 (用于 scanner 注入)。"""
    sc = live_sentiment_from_snapshot(snap)
    if not np.isfinite(sc):
        return ""
    return f"市场情绪温度计 {sc:.0f}/100 ({sentiment_label(sc)})"


def format_live_sentiment(panel: pd.DataFrame, lookback: int = 60) -> str:
    """实时注入版: 用最近 lookback 日窗口做 z-score (非全窗口), 避免未来信息。"""
    tail = panel.tail(lookback).copy()
    if len(tail) < 20:
        return "情绪数据不足"
    comp = tail[SENTIMENT_COMPONENTS]
    z = comp.sub(comp.mean()).div(comp.std().replace(0, np.nan)).fillna(0.0)
    comp_score = z.mean(axis=1)
    lo, hi = comp_score.min(), comp_score.max()
    score = (comp_score.iloc[-1] - lo) / (hi - lo + 1e-12) * 100
    label = sentiment_label(score)
    # 当前成分相对近60日的位置
    cur = z.iloc[-1]
    hot = [c for c in SENTIMENT_COMPONENTS if abs(cur[c]) > 0.8]
    parts = []
    for c in hot:
        zh = {
            "sent_limit_up": "涨停占比", "sent_limit_down": "跌停抛压",
            "sent_breadth": "上涨广度", "sent_newhigh": "新高推进",
            "sent_volume": "成交活跃", "sent_skew": "收益偏态",
            "sent_ret5": "短期动量", "sent_valuation": "估值水平",
        }[c]
        parts.append(f"{zh}{'+' if cur[c] > 0 else '-'}{abs(cur[c]):.1f}")
    detail = ", ".join(parts) if parts else "各成分接近中性"
    return f"市场情绪温度计 {score:.0f}/100 ({label}), 特征成分: {detail}"
