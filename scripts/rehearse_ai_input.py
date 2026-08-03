"""v3.2 AI 输入排练 — 用回放数据演示 AnalysisWorkflow 各组件实际看到的上下文

背景: Baostock 被封 + EastMoney 代理隧道失败, 实时日K不可用 → 无法直接跑
live AnalysisWorkflow。本脚本用回放缓存 + **真实管线代码** 构造 AI 输入,
展示 v3.2 改动让 AI 多看到了什么 (before/after), 并可选跑一次真实 DeepSeek。

真实代码路径 (非 mock):
  - analysis.indicators.TechnicalAnalyzer.compute_all  → 含估值/反转/流动性指标
  - agent.sub_agents.technical_analyst._build_indicator_summary → LLM 摘要
  - factors.market_sentiment.build_sentiment_panel + format_sentiment → 情绪温度计
  - factors.regime_analysis 合成指数 + MarketRegimeDetector → regime 标签
  - 市场广度快照: 从回放 asof 截面重建 (与 live_market_snapshot 同口径)

用法:
  python scripts/rehearse_ai_input.py [--asof 2026-07-31] [--llm]
  --llm: 用 DeepSeek 对"市场扫描上下文"做一次真实解读 (演示 AI 判断)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

OLD_KEY_FIELDS = [
    "close", "ma_5", "ma_20", "ma_60", "rsi_14",
    "macd_dif", "macd_dea", "bb_pct_20", "atr_14",
    "trend_score", "composite_score",
]
SAMPLE_SYMBOLS = ["sh.600519", "sz.000651", "sh.601398", "sz.000858"]  # 茅台/格力/工行/五粮液


def load_replay(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_breadth_snapshot(df: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """从回放 asof 日截面重建广度快照 (字段同 live_market_snapshot)。"""
    d = df[df["date"] <= asof]
    last = d[d["date"] == d["date"].max()]
    if last.empty:
        return {}
    pct = pd.to_numeric(last["pctChg"], errors="coerce").dropna()
    snap = {"n_stocks": int(len(last))}
    if len(pct):
        snap["pct_up"] = round(float((pct > 0).mean()), 3)
        snap["avg_pct"] = round(float(pct.mean()), 3)
        snap["limit_up"] = int((pct >= 9.5).sum())
        snap["limit_down"] = int((pct <= -9.5).sum())
    pe = pd.to_numeric(last["peTTM"], errors="coerce")
    pos = pe[pe > 0]
    if len(pos):
        snap["median_pe"] = round(float(pos.median()), 2)
    pb = pd.to_numeric(last["pbMRQ"], errors="coerce")
    pos = pb[pb > 0]
    if len(pos):
        snap["median_pb"] = round(float(pos.median()), 2)
    return snap


def indicator_summary(df_row: pd.Series, fields: list) -> dict:
    """与 technical_analyst._build_indicator_summary 同口径的字段抽取。"""
    out = {}
    for k in fields:
        if k not in df_row.index:
            continue
        v = df_row[k]
        if isinstance(v, (int, float)):
            if not np.isfinite(v):
                continue
            out[k] = round(float(v), 2)
        else:
            out[k] = str(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", type=str, default="2026-07-31")
    ap.add_argument("--replay", type=str, default="replay_data/daily_2026-02-21_2026-07-31.parquet")
    ap.add_argument("--llm", action="store_true", help="用 DeepSeek 解读市场上下文")
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof)

    df = load_replay(args.replay)
    print(f"回放: {args.replay} | {df['symbol'].nunique()} 只 | asof={asof.date()}\n")

    # ── 1. 单股指标摘要 (before/after) ──
    from analysis.indicators import TechnicalAnalyzer
    ta = TechnicalAnalyzer()

    print("=" * 78)
    print("1) 个股指标摘要 — AI 技术分析看到什么 (before vs after v3.2)")
    print("=" * 78)
    for sym in SAMPLE_SYMBOLS:
        g = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if len(g) < 60:
            print(f"\n[{sym}] 数据不足 ({len(g)}行), 跳过")
            continue
        try:
            res = ta.compute_all(g, symbol=sym)
        except Exception as e:
            print(f"\n[{sym}] compute_all 失败: {repr(e)[:120]}")
            continue
        ind_df = pd.DataFrame(res.indicators)
        if ind_df.empty:
            print(f"\n[{sym}] 无指标")
            continue
        last = ind_df.iloc[-1]
        before = indicator_summary(last, OLD_KEY_FIELDS)
        after = indicator_summary(last, OLD_KEY_FIELDS + [
            "ep", "bp", "pe_pct_20d", "pb_pct_20d",
            "amihud_illiq", "sharpe_20", "reversal_1d",
            "turn_pct_20d", "vol_regime_20",
        ])
        new_keys = [k for k in after if k not in before]
        print(f"\n[{sym}] close={after.get('close')}")
        print(f"  before({len(before)}字段): {json.dumps(before, ensure_ascii=False)[:220]}")
        print(f"  after ({len(after)}字段) +新增 {len(new_keys)}: "
              f"{json.dumps({k: after[k] for k in new_keys}, ensure_ascii=False)}")

    # ── 2. 市场扫描上下文 (regime + 广度 + 情绪温度计) ──
    print("\n" + "=" * 78)
    print("2) 市场扫描上下文 — AI 判断'当前形势'看到什么")
    print("=" * 78)

    from factors.market_sentiment import build_sentiment_panel, format_sentiment
    from factors.regime_analysis import build_synthetic_index, detect_regime_series

    idx = build_synthetic_index(df)
    regime_series = detect_regime_series(idx, None)
    asof_regime = None
    for d, b in regime_series.items():
        if d <= asof:
            asof_regime = b
        else:
            break
    context = f"当前市场状态: {asof_regime} (asof {asof.date()})\n"

    snap = build_breadth_snapshot(df, asof)
    if snap:
        from analysis.market_breadth import format_snapshot
        context += f"市场形势快照: {format_snapshot(snap)}\n"

    sent_panel = build_sentiment_panel(df)
    sent_txt = format_sentiment(sent_panel, asof)
    if sent_txt:
        context += f"{sent_txt}\n"

    print(context)
    print(f"  情绪温度计细节: 见 reports/market_sentiment.json")

    # ── 3. (可选) DeepSeek 解读 ──
    if args.llm:
        print("\n" + "=" * 78)
        print("3) DeepSeek 对上述上下文的真实市场解读")
        print("=" * 78)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
        system = ("你是A股市场扫描Agent。基于给定的市场状态/广度/情绪数据,"
                  "判断当前市场形势, 给出: 1) 市场所处阶段 2) 主要风险 3) 操作倾向。"
                  "要求: 估值(vs历史)与情绪温度是重要依据, 动量信号需按regime打折 (牛才信)。")
        r = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": context}],
            max_tokens=600,
        )
        print("\n" + r.choices[0].message.content)


if __name__ == "__main__":
    main()
