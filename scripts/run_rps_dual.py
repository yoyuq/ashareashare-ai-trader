"""RPS 双周期动量 (120d+50d) 月度 top5 — 预注册 reports/agent_loop/prereg_rps_dual.md.

输出: reports/agent_loop/rps_dual_result.json
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

from run_dca_vs_lumpsum import WINDOWS  # noqa: E402

COST = 65.0 / 1e4
TOP_K = 5
MKT_LO, MKT_HI = 50.0, 200.0  # 亿


def main() -> int:
    out = {}
    for wname, fp in WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {fp}")
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret = close.pct_change(fill_method=None)
        mom120 = close.pct_change(120, fill_method=None)
        mom50 = close.pct_change(50, fill_method=None)
        r120 = mom120.rank(axis=1, pct=True)
        r50 = mom50.rank(axis=1, pct=True)
        fmkt = (df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8)
        df["log_mkt"] = np.log1p(fmkt)
        mkt = df.pivot_table(index="date", columns="symbol", values="log_mkt").reindex(close.index)
        pool = (r120 >= 0.9) & (r50 >= 0.9) & (mkt >= np.log1p(MKT_LO)) & (mkt <= np.log1p(MKT_HI))

        s = pd.Series(close.index)
        me = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
        sel = {}
        for T in me:
            row = mom120.loc[T].where(pool.loc[T]).dropna()
            if len(row) >= TOP_K:
                sel[T] = row.nlargest(TOP_K).index.tolist()
        valid = [T for T in me if T in sel]
        if not valid:
            out[wname] = {"n_valid_months": 0}
            print(f"{wname}: 无有效月份 (因子预热不足)")
            continue
        excess_days = []
        n_periods = 0
        for i, T in enumerate(valid):
            T_next = valid[i + 1] if i + 1 < len(valid) else close.index[-1]
            idx = ret.index[(ret.index > T) & (ret.index <= T_next)]
            if idx.empty:
                continue
            port = ret.loc[idx, sel[T]].mean(axis=1)
            counts = ret.notna().sum(axis=1)
            uni = ret.mean(axis=1).where(counts > 50).reindex(idx)
            prev = sel[valid[i - 1]] if i > 0 else []
            replaced = len(set(sel[T]) - set(prev)) / TOP_K
            port = port - (replaced * COST)
            excess_days.append(port - uni)
            n_periods += 1
        ex = pd.concat(excess_days).dropna()
        total_ex = float((1 + ex).prod() - 1)
        out[wname] = {"n_valid_months": n_periods, "total_excess": round(total_ex, 4)}
        print(f"{wname}: 有效月 {n_periods}, 累计超额 {total_ex:+.1%}")

    act = {k: v for k, v in out.items() if v.get("n_valid_months", 0) >= 3}
    wins = sum(1 for v in act.values() if v["total_excess"] >= 0)
    avg = float(np.mean([v["total_excess"] for v in act.values()])) if act else float("nan")
    summary = {"valid_windows": len(act), "excess_wins": f"{wins}/{len(act)}",
               "avg_excess": round(avg, 4), "PASS": bool(len(act) >= 8 and wins >= 5 and avg > 0)}
    print(json.dumps(summary, ensure_ascii=False))
    fp_out = ROOT / "reports" / "agent_loop" / "rps_dual_result.json"
    fp_out.write_text(json.dumps({"windows": out, "summary": summary},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
