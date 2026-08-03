"""今日 thinking 深度分析 — 最新交易日全市场 → 规则筛300 → LLM精筛100 → thinking深析

用法: python scripts/daily_thinking.py [--date YYYY-MM-DD]
默认用最新交易日 (回放缓存里最后一个有数据的日期)。
"""

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default="", help="分析日期 (默认最新交易日)")
    args = ap.parse_args()

    from historical_replay import (  # noqa: E402
        load_snapshot_basic, build_daily_data, reconstruct_cross_section,
    )
    from simulation.daily_runner import _flash_screen, _deepseek_analyze, _detect_regime  # noqa: E402
    from analysis.pre_screener import PreScreener  # noqa: E402

    basic, universe = load_snapshot_basic()
    # 用缓存中的最新交易日
    import pandas as pd
    df_all = pd.read_parquet("replay_data/daily_2026-02-21_2026-07-31.parquet",
                             columns=["date", "symbol"])
    latest = str(pd.to_datetime(df_all["date"]).max().date())
    T = args.date or latest
    print(f"分析日期: {T} (最新交易日: {latest})")

    data = build_daily_data(None, date(2026, 2, 21), date(2026, 7, 31))
    df = reconstruct_cross_section(data, basic, T)
    df = df[df["isST"] != 1]
    regime = _detect_regime(df).get("regime", "range_bound")
    print(f"市场状态: {regime} | 截面: {len(df)} 只")

    screened = PreScreener().screen(df, regime=regime, top_n=300).df
    print(f"规则初筛: {len(screened)} → LLM精筛中...")
    df_top = await _flash_screen(screened, top_k=100)
    print(f"LLM精筛: {len(df_top)} → thinking深析中 (~5-8分钟)...")

    # v3.1.2: 宏观/政策/国际上下文注入 (来自 macro_context.py 缓存)
    import json as _json
    macro_text = ""
    try:
        _mc = _json.loads(Path("knowledge/macro_context_latest.json").read_text(encoding="utf-8"))
        _parts = [f"日期: {_mc.get('date', '')}",
                  f"综合宏观分: {_mc.get('composite_macro_score', 50)}",
                  f"推荐: {str(_mc.get('recommendation', ''))[:500]}"]
        macro_text = "\n".join(_parts)
        print(f"📊 已注入宏观背景 (综合分 {_mc.get('composite_macro_score', '?')}/100)")
    except Exception:
        print("⚠️ 无宏观缓存 (先跑 python scripts/macro_context.py)")

    deep = await _deepseek_analyze(df_top.head(100), thinking=True,
                                   macro_context=macro_text)
    print(f"thinking分析: {len(deep)} 只")

    buys = sorted([r for r in deep if r.get("action") == "BUY"],
                  key=lambda r: -float(r.get("final_score", 0) or 0))
    sells = [r for r in deep if r.get("action") == "SELL"]
    holds = len(deep) - len(buys) - len(sells)
    print(f"\n=== {T} thinking 推荐 ===")
    print(f"BUY {len(buys)} | SELL {len(sells)} | HOLD {holds}")
    print("\n【买入 Top10】")
    for r in buys[:10]:
        print(f"  🔴 {r.get('code')} {r.get('name','')} score={r.get('final_score',0)} "
              f"conv={r.get('conviction',0)}")
    print("\n【卖出】")
    for r in sells[:5]:
        print(f"  🔻 {r.get('code')} {r.get('name','')} score={r.get('final_score',0)}")


if __name__ == "__main__":
    asyncio.run(main())
