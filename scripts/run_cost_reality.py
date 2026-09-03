"""成本现实性测试 — 冷落低波精选在真实 1万 小资金费率下是否仍保 alpha (预注册 §十二).

背景: 回测一直用 31bp 全周转成本 = 大资金费率. 但用户实盘是 1 万拆 5 只 × 2000 元, 每笔买卖卡
最低佣金 ¥5, 真实往返约 65bp, 是回测的 2 倍+. 本测试诚实回答: **1万的 edge 会不会被真实佣金抹掉**.

成本模型 (固定现实值, 非调参):
  31bp = 大资金回调基率 (佣金万3×2 + 印花0.05%卖 + 滑点10bp, [[transaction-costs-unified]])
  65bp = 1万小资金真实费率: 每次换仓单只(≈¥2000) 买卖各卡 ¥5 最低佣金(=50bp) + 印花5bp + 滑点10bp
  90bp = 敏感性上限 (更保守滑点/涨跌停手滑预留, 不 gate, 仅参考)
成本按换手比例 (replaced/top_k) 折算, 与既有 harness 结构完全一致; 唯一改动 = 成本常量.

判据 (预注册, 跑完不调, 与头号策略同口径):
  主判据①  非纯牛 8 窗 vs_universe ≥ 0 的窗口 ≥ 7/8   (小资金佣金不吞掉熊/震荡超额)
  主判据②  纯牛 3 窗 总收益 > 0 且 maxdd ≥ −30%       (牛不崩)
  → 65bp 下主判据① 仍 PASS ⇒ 1万 可行 (edge 扛得住小资金成本); 否则 ⇒ 1万 太贵, 事先如实披露.

用法: python scripts/run_cost_reality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import strategy_research_harness as H  # noqa: E402

# §六 客观 regime 重标注 (与 crash_guard 一致, 跑之前写死)
OBJECTIVE = {
    "2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
    "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
    "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
    "2024震荡": "纯牛", "2025-26现期": "震荡",
}

# 预注册成本 (固定现实值)
SCALE_BPS = 31.0   # 大资金基线
SMALL_BPS = 65.0   # 1万真实费率 (主判据口径)
STRESS_BPS = 90.0  # 敏感性上限 (不 gate)

COSTS = [("31bp 大资金基线", SCALE_BPS, False),
         ("65bp 1万真实费率", SMALL_BPS, True),
         ("90bp 敏感性参考", STRESS_BPS, False)]


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _factors(df):
    out = df[["date", "symbol", "turn", "amount", "close"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std20"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["log_mkt"] = np.log1p(fmkt)
    return out


def _cold_lowvol(df):
    f = _factors(df)
    r_turn = f["turn"].groupby(df["date"]).rank(pct=True)
    r_std = f["std20"].groupby(df["date"]).rank(pct=True)
    r_mkt = f["log_mkt"].groupby(df["date"]).rank(pct=True)
    return -(r_turn + r_std + r_mkt)


def _run_window(df, idx, window, cost_bps):
    score = _cold_lowvol(df)
    daily_ret, info = H.backtest_ranking_portfolio(df, score, 5, "monthly", cost_bps=cost_bps)
    m = H.compute_metrics(daily_ret)
    t0, t1 = daily_ret.index.min(), daily_ret.index.max()
    uni = H.window_universe_benchmark(df)
    uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
    uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
    idx_total = H._idx_total(idx, t0, t1)
    m.update({"universe_total": round(uni_total, 4), "index_total": round(idx_total, 4),
              "vs_universe": round(m["total"] - uni_total, 4) if not np.isnan(m["total"]) else np.nan,
              "vs_index": round(m["total"] - idx_total, 4) if not np.isnan(m["total"]) else np.nan,
              "regime": OBJECTIVE[window], "n_rebalance": info["n_rebalance"],
              "turnover_bps_equiv": info["turnover_cost_bps_equiv"] / cost_bps * 10000.0})
    return m


def main() -> int:
    _force_utf8()
    idx = H.load_index()
    results = {c[0]: {} for c in COSTS}

    print("=" * 112)
    print("成本现实性测试 — 冷落低波精选 top5 月再平衡: 31bp 大资金 vs 65bp 1万真实 vs 90bp保守")
    print("=" * 112)

    for window in H.WINDOWS:
        fp = H.WINDOWS[window]
        df = H.load_base_window(fp)
        for label, cost_bps, _ in COSTS:
            results[label][window] = _run_window(df, idx, window, cost_bps)

        # 每窗口并排三种成本
        def pct(x, nd=1):
            return f"{x*100:+.{nd}f}%" if not (isinstance(x, float) and np.isnan(x)) else "  nan "
        vs31 = [results[c[0]][window]["vs_universe"] for c in COSTS]
        tot31 = [results[c[0]][window]["total"] for c in COSTS]
        print(f"{window:<12} {OBJECTIVE[window]:<4} "
              f"vsU: {pct(vs31[0])} | {pct(vs31[1])} | {pct(vs31[2])}   "
              f"总:  {pct(tot31[0])} | {pct(tot31[1])} | {pct(tot31[2])}")

    # 判据裁决 (主口径 = 65bp)
    print("\n" + "=" * 112)
    print("预注册判据 (§十二): 主口径 = 65bp 真实1万费率")
    print("=" * 112)
    for label, cost_bps, is_gate in COSTS:
        res = results[label]
        niu = [w for w in H.WINDOWS if OBJECTIVE[w] == "纯牛"]
        nonniu = [w for w in H.WINDOWS if OBJECTIVE[w] != "纯牛"]
        i_win = sum(1 for w in nonniu if res[w]["vs_universe"] is not None and res[w]["vs_universe"] >= 0)
        i_pass = i_win >= 7
        niu_details = "; ".join(f"{w} tot={res[w]['total']*100:+.1f}% dd={res[w]['maxdd']*100:.1f}%" for w in niu)
        ii_pass = all(not np.isnan(res[w]["total"]) and res[w]["total"] > 0 and res[w]["maxdd"] >= -0.30 for w in niu)
        ii_win = sum(1 for w in niu if res[w]["maxdd"] >= -0.30)
        gate_note = "  【主判据@此成本】" if is_gate else ""
        print(f"\n[{label}] {cost_bps:.0f}bp{gate_note}")
        print(f"  主判据① 非纯牛8窗 vsU≥0: {i_win}/8 → {'PASS (≥7/8)' if i_pass else 'FAIL'}")
        print(f"  主判据② 纯牛3窗 全正&maxdd≥−30%: {'PASS' if ii_pass else 'FAIL'}   ({niu_details})")
        print(f"  平均 vs_universe (11窗): {np.mean([res[w]['vs_universe'] for w in H.WINDOWS if res[w]['vs_universe'] is not None]) * 100:+.2f}pp")

    H.dump_json({label: results[label] for label, _, _ in COSTS},
                ROOT / "reports" / "cost_reality_result.json")
    print(f"\n结果已写 reports/cost_reality_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())