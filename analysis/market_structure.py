"""市场结构识别 (v3.3) — 判断当前是 抱团/动量 还是 轮动/普涨, 决定进攻 or 防御选股

背景: 进攻 vs 防御 A/B 显示价值是 regime 相关的 —
  抱团/动量牛 (龙头独涨普跌): 进攻赢 (+3.41pp, 2020-12~2021-02)
  轮动/普涨牛 (广度健康): 防御稳 (+0.5pp, 2026-01~02)
广度式 regime 检测 (up_ratio/avg_pct) 在窄幅抱团牛会误判成熊市 →
需补"结构维度": 龙头 vs 中位数 的离散度 (leader gap) 才是关键。

判型 (基于当日截面):
  抱团动量 (→进攻): 龙头极强 (top10>2%) 且广度低 (<0.45) — 龙头独涨普跌
  轮动普涨 (→防御): 广度健康 (up>0.55) 且中位数>0 — 涨幅分散
  熊        (→防御): 普跌 (up<0.35 且 med<-1)
  震荡      (→均衡): 其余

用法: market_structure(df) -> str  (df 含 pct_change)
"""

import numpy as np
import pandas as pd


def market_structure(df: pd.DataFrame, up_thr: float = 0.55,
                     top_thr: float = 2.0) -> str:
    """从当日截面判断市场结构 (PIT 安全, 只用当日数据).

    Args:
        df: 截面 DataFrame, 含 pct_change (当日涨跌%).
        up_thr: 广度阈值 (>0.55 = 普涨).
        top_thr: 龙头阈值 (top10 平均涨幅 >2% 且广度低 = 抱团).

    Returns:
        "抱团动量" / "轮动普涨" / "熊" / "震荡"
    """
    if "pct_change" not in df.columns:
        return "震荡"
    pct = pd.to_numeric(df["pct_change"], errors="coerce").dropna()
    if len(pct) < 50:
        return "震荡"

    up = float((pct > 0).mean())
    med = float(pct.median())
    n10 = max(1, int(0.1 * len(pct)))
    top10 = float(pct.nlargest(n10).mean())
    bottom10 = float(pct.nsmallest(n10).mean())
    leader_gap = top10 - bottom10

    # 抱团/动量: 龙头极强 + 广度低 (龙头独涨, 普跌)
    if top10 > top_thr and up < 0.45:
        return "抱团动量"
    # 轮动/普涨: 广度健康 + 中位数上涨 (涨幅分散)
    if up > up_thr and med > 0:
        return "轮动普涨"
    # 熊: 普跌且中位数显著为负
    if up < 0.35 and med < -1.0:
        return "熊"
    # 震荡
    return "震荡"


def structure_label(structure: str) -> str:
    """结构 → 操作倾向 (进攻/防御/均衡)."""
    return {
        "抱团动量": "进攻 (追强势龙头)",
        "轮动普涨": "防御/均衡 (轮动快时防追高)",
        "熊": "防御 (低估值/现金)",
        "震荡": "均衡",
    }.get(structure, "均衡")


def market_structure_series(cross_sections: dict, window: int = 5) -> dict:
    """从多日截面序列判结构 (最近 window 日多数投票, 平滑噪声).

    Args:
        cross_sections: {date_str: 当日截面 DataFrame}
        window: 回看天数 (多数投票)

    Returns:
        {date_str: structure}
    """
    out = {}
    dates = sorted(cross_sections.keys())
    for i, d in enumerate(dates):
        recent = dates[max(0, i - window + 1): i + 1]
        votes = {}
        for rd in recent:
            s = market_structure(cross_sections[rd])
            votes[s] = votes.get(s, 0) + 1
        out[d] = max(votes, key=votes.get)
    return out
