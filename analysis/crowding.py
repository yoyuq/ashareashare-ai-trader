"""拥挤度风控 (v3.4) — 研究移植: 行业动量+拥挤度惩罚是动量策略的生存前提

证据 (外部):
  国泰君安: 拥挤度因子是"顶部择时类避险因子", 叠加后行业轮动年化 18.56%→26.49%,
            最大回撤 49.28%→28.11%.
  信达证券: 单行业+全市场双拥挤度, 把 2015 动量崩盘从 -11.2% 扭成 +15.3%.
  招商证券: 行业动量在熊市/高波动/拥挤度过高时失效 → 需拥挤度惩罚.

本项目适配:
  单股拥挤度 = 当日换手在自身60日内的分位 (turn_pct_60d, PIT 安全).
  全市场拥挤度 = 换手处于自身60日分位 >95% 的股票占比 (极端活跃广度).
    当占比过高 → 市场情绪过热, 动量崩溃风险 → 防御降仓/暂停追高.

用法:
  market_crowding(df_cs) -> {score, signal, hot_ratio}   # 全市场拥挤信号
  crowding_penalty(df_cs) -> pd.Series (0 到负扣分)       # 单股拥挤惩罚
"""

import numpy as np
import pandas as pd


def market_crowding(df_cs: pd.DataFrame, extreme_thr: float = 0.95) -> dict:
    """全市场拥挤度: turn_pct_60d > extreme_thr 的股票占比 (过热广度).

    Args:
        df_cs: 当日全市场截面, 需含 turn_pct_60d (自身60日换手分位, 0-1).
        extreme_thr: 视为"极端活跃"的自身换手分位阈值.

    Returns:
        {score (0-100, >60=过热), signal (hot/warm/cool), hot_ratio (极端活跃占比)}
        数据不足时返回中性 (score=50, signal=cool), 调用方降级为原策略.
    """
    if df_cs is None or df_cs.empty or "turn_pct_60d" not in df_cs.columns:
        return {"score": 50.0, "signal": "cool", "hot_ratio": 0.0}
    tp = pd.to_numeric(df_cs["turn_pct_60d"], errors="coerce").dropna()
    if len(tp) < 100:
        return {"score": 50.0, "signal": "cool", "hot_ratio": 0.0}
    hot = float((tp > extreme_thr).mean())
    # hot_ratio 0.06 ≈ score 55 (warm), 0.10 ≈ 85 (hot). 线性放大, 封顶 100.
    score = float(np.clip(hot * 800.0, 0, 100))
    if score >= 60:
        signal = "hot"
    elif score >= 40:
        signal = "warm"
    else:
        signal = "cool"
    return {"score": round(score, 1), "signal": signal, "hot_ratio": round(hot, 4)}


def crowding_penalty(df_cs: pd.DataFrame, extreme_thr: float = 0.97,
                     max_penalty: float = 8.0) -> pd.Series:
    """单股拥挤度惩罚 (0 到 -max_penalty 扣分).

    自身60日换手分位 > extreme_thr → 极度拥挤 (放量赶顶/逼空尾部), 按超过程度线性扣分.
    用于在 composite_score 上减分, 让"过热"股在排名中适度降权.
    """
    s = pd.Series(0.0, index=df_cs.index)
    if "turn_pct_60d" not in df_cs.columns:
        return s
    tp = pd.to_numeric(df_cs["turn_pct_60d"], errors="coerce")
    over = (tp > extreme_thr) & tp.notna()
    if not over.any():
        return s
    s.loc[over] = -max_penalty * (tp[over] - extreme_thr) / (1 - extreme_thr)
    return s.clip(-max_penalty, 0.0)


def format_crowding(cd: dict) -> str:
    """拥挤度信号 → 一行可读文本 (供 LLM 上下文/日志)."""
    if not cd:
        return "拥挤度: 数据不可用"
    label = {"hot": "过热(动量崩溃风险)", "warm": "偏热(谨慎追高)",
             "cool": "正常(情绪温和)"}.get(cd.get("signal"), cd.get("signal"))
    return (f"拥挤度: 极端活跃占比 {cd.get('hot_ratio', 0):.1%} → "
            f"{label} (score {cd.get('score', 50)})")
