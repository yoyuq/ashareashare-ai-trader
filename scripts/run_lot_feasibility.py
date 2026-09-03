"""1万等权5票整数手可行性执行检验 — 预注册 reports/agent_loop/prereg_lot_feasibility.md.

A = buffer 定型 (分数等权); B = 换仓日剔除 close > (账户净值/K)/100股 的票后走 buffer 臂。
账户净值路径: 窗口起点 1 万元, 按臂内日收益复利 (DCA 现金流不建模, 保守取下界)。
输出: reports/agent_loop/lot_feasibility_result.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import strategy_research_harness as H  # noqa: E402
from run_rotation_vs_hold import TOP_K, _score, month_ends, run_arm  # noqa: E402

START_CAPITAL = 10000.0


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def run_arm_lot(df: pd.DataFrame, score: pd.Series) -> tuple[pd.Series, dict, int]:
    """buffer 臂 + 整数手可行性剔除; 返回 (日收益, info, binding次数)。

    两遍实现: 第一遍跑定型 buffer 得净值路径; 第二遍按路径在每个换仓日
    以 equity/K/100 为股价上限剔除不可行票后重跑 buffer。
    """
    # 第一遍: 定型臂净值路径
    dr_a, _ = run_arm(df, score, "buffer")
    if dr_a.empty:
        return dr_a, {"n_rebalance": 0, "n_replaced": 0}, 0
    eq = (1.0 + dr_a.sort_index()).cumprod() * START_CAPITAL

    # 第二遍: 可行性剔除
    d = df.copy()
    s2 = score.copy()
    close = d.pivot_table(index="date", columns="symbol", values="close").sort_index()
    drop, binding = [], 0
    for T in month_ends(list(close.index)):
        if T not in eq.index or T not in set(close.index):
            continue
        slot = float(eq.loc[T]) / TOP_K
        max_price = slot / 100.0  # 100 股/手
        day_px = close.loc[T]
        infeasible = day_px[day_px > max_price].index
        # binding 统计: 该换仓日 score 排名前 K 中不可行票数
        day_score = s2[d["date"] == T].dropna()
        if len(day_score) >= TOP_K:
            ranked = day_score.sort_values(ascending=False).head(TOP_K)
            syms_top = d.loc[ranked.index, "symbol"]
            n_bad = int(syms_top.isin(set(infeasible)).sum())
            if n_bad:
                binding += n_bad
                for c in infeasible:
                    drop.append((T, c))
    if drop:
        dd = pd.MultiIndex.from_tuples(drop)
        mi = pd.MultiIndex.from_arrays([d["date"].values, d["symbol"].values])
        s2[mi.isin(dd)] = np.nan
    dr_b, info = run_arm(d, s2, "buffer")
    return dr_b, info, binding


def main() -> int:
    _force_utf8()
    out = {}
    print(f"{'window':<12} {'arm':<8} {'total':>8} {'vsU':>8} {'maxdd':>8} {'bind':>5}")
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        uni = H.window_universe_benchmark(df)
        dr_a, info_a = run_arm(df, score, "buffer")
        dr_b, info_b, binding = run_arm_lot(df, score)
        out[window] = {}
        for name, dr, info in (("A_buffer", dr_a, info_a), ("B_lot", dr_b, info_b)):
            if dr.empty:
                out[window][name] = None
                print(f"{window:<12} {name:<8} {'NA':>8}")
                continue
            m = H.compute_metrics(dr)
            t0, t1 = dr.index.min(), dr.index.max()
            uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
            uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
            vsu = round(m["total"] - uni_total, 4)
            out[window][name] = {"total": round(m["total"], 4), "vs_universe": vsu,
                                 "maxdd": round(m["maxdd"], 4), "n_replaced": info["n_replaced"]}
        out[window]["_bind"] = binding
        for name in ("A_buffer", "B_lot"):
            r = out[window][name]
            if r is None:
                print(f"{window:<12} {name:<8} {'NA':>8}")
            else:
                print(f"{window:<12} {name:<8} {r['total']*100:>+7.1f}% {r['vs_universe']*100:>+7.1f}% "
                      f"{r['maxdd']*100:>7.1f}% {binding if name == 'B_lot' else 0:>5}")

    w10 = [w for w in H.WINDOWS if w != "2015牛转股灾"]
    diffs = [out[w]["B_lot"]["vs_universe"] - out[w]["A_buffer"]["vs_universe"] for w in w10]
    dd_diffs = [out[w]["B_lot"]["maxdd"] - out[w]["A_buffer"]["maxdd"] for w in w10]
    wins = sum(1 for x in diffs if x >= -0.01)
    avg_diff = float(np.mean(diffs))
    avg_dd_diff = float(np.mean(dd_diffs))
    gate = bool(avg_diff >= -0.01 and wins >= 8 and avg_dd_diff >= -0.01)
    summary = {"avg_vsU_diff": round(avg_diff, 4), "diff_ge_-1pp_wins": f"{wins}/{len(w10)}",
               "avg_dd_diff": round(avg_dd_diff, 4),
               "binding_total": sum(out[w]["_bind"] for w in H.WINDOWS if "_bind" in out[w]),
               "gate": "PASS" if gate else "FAIL",
               "rule": "avg差≥−1pp 且 ≥−1pp窗数≥8/10 且 dd差≥−1pp (执行可接受线)"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "lot_feasibility_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
