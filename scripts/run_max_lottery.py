"""MAX 彩票效应截面因子 — 预注册 reports/agent_loop/prereg_max_lottery.md.

因子: 过去20日最大单日收益, 月末 MAX最低top5 (空彩票臂) vs MAX最高对照臂, vs 全主板等权, 65bp。
输出: reports/agent_loop/max_lottery_result.json
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


def main() -> int:
    out = {}
    for wname, fp in WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {fp}")
        df = df.sort_values(["symbol", "date"])
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret = close.pct_change(fill_method=None)
        max20 = ret.rolling(20).max()          # 主型: 20日最大单日收益
        limit20 = (ret >= 0.095).rolling(20).sum()  # 披露变体: 触板天数

        s = pd.Series(close.index)
        me = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
        arms = {"low_max": {}, "high_max": {}}
        for arm in arms:
            sel = {}
            for T in me:
                if T not in max20.index:
                    continue
                row = max20.loc[T].dropna()
                if len(row) < 50:
                    continue
                codes_k = (row.nsmallest(TOP_K) if arm == "low_max" else row.nlargest(TOP_K)).index.tolist()
                sel[T] = codes_k
            valid = [T for T in me if T in sel]
            excess_days = []
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
            if not excess_days:
                arms[arm] = {"n_valid_months": 0}
                continue
            ex = pd.concat(excess_days).dropna()
            arms[arm] = {"n_valid_months": len(valid), "total_excess": round(float((1 + ex).prod() - 1), 4)}

        # 披露变体: LIMIT_CNT 最低 top5 (仅汇总, 不 gate)
        lsel = {}
        for T in me:
            if T not in limit20.index:
                continue
            row = limit20.loc[T].dropna()
            if len(row) < 50:
                continue
            lsel[T] = row.nsmallest(TOP_K).index.tolist()
        lvalid = [T for T in me if T in lsel]
        lex_days = []
        for i, T in enumerate(lvalid):
            T_next = lvalid[i + 1] if i + 1 < len(lvalid) else close.index[-1]
            idx = ret.index[(ret.index > T) & (ret.index <= T_next)]
            if idx.empty:
                continue
            port = ret.loc[idx, lsel[T]].mean(axis=1)
            counts = ret.notna().sum(axis=1)
            uni = ret.mean(axis=1).where(counts > 50).reindex(idx)
            prev = lsel[lvalid[i - 1]] if i > 0 else []
            port = port - (len(set(lsel[T]) - set(prev)) / TOP_K) * COST
            lex_days.append(port - uni)
        l_ex = round(float((1 + pd.concat(lex_days).dropna()).prod() - 1), 4) if lex_days else None

        # 冗余披露: low_max 月选票 vs 冷落低波公式票重合 (近似: 冷落=低换手+低波+低市值代理)
        out[wname] = {"low_max": arms["low_max"], "high_max": arms["high_max"],
                      "low_limit_cnt_excess": l_ex}
        print(f"{wname}: MAX最低 {arms['low_max'].get('total_excess', 'NA')} "
              f"MAX最高 {arms['high_max'].get('total_excess', 'NA')} "
              f"触板最少 {l_ex} (有效月 {arms['low_max'].get('n_valid_months', 0)})", flush=True)

    def gate(arm):
        act = {k: v[arm] for k, v in out.items()
               if isinstance(v.get(arm), dict) and v[arm].get("n_valid_months", 0) >= 3}
        wins = sum(1 for v in act.values() if v.get("total_excess", 0) >= 0)
        avg = float(np.mean([v["total_excess"] for v in act.values()])) if act else float("nan")
        return {"valid_windows": len(act), "excess_wins": f"{wins}/{len(act)}",
                "avg_excess": round(avg, 4), "PASS": bool(len(act) >= 5 and wins >= 5 and avg > 0)}

    summary = {"low_max_arm": gate("low_max"), "high_max_arm": gate("high_max")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp_out = ROOT / "reports" / "agent_loop" / "max_lottery_result.json"
    fp_out.write_text(json.dumps({"windows": out, "summary": summary},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
