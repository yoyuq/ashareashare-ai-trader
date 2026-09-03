"""冷落低波: 月度重选 vs 冻结持有 — 实操形态定型 (预注册 reports/agent_loop/prereg_monthly_vs_hold.md).

复用 run_dca_vs_lumpsum 的选券机器; 两臂 65bp; 8 窗。
输出: reports/agent_loop/monthly_vs_hold_result.json
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

from run_dca_vs_lumpsum import WINDOWS, maxdd, run_window  # noqa: E402


def main() -> int:
    out = {}
    for wname, fp in WINDOWS.items():
        valid, sel, nav, dates = run_window(fp)  # nav = 月度重选日收益序列 (65bp 已含)
        t0 = valid[0]
        end = dates[-1]

        # 冻结持有: t0 篮子持有到窗末 (建仓成本一次), 期间日收益 = 固定篮子均值
        close = pd.read_parquet(ROOT / fp.replace("replay_data/", "replay_data/"))
        import strategy_research_harness as H
        df = H.load_base_window(fp)
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret = close.pct_change(fill_method=None)
        basket = sel[t0]
        seg = ret.loc[(ret.index > t0) & (ret.index <= end), basket]
        nav_hold = seg.mean(axis=1)
        nav_hold.iloc[0] -= 65.0 / 1e4  # 建仓成本一次
        nav_hold = nav_hold.fillna(0.0)

        tot_rot = float((1.0 + nav).prod() - 1.0)
        tot_hold = float((1.0 + nav_hold).prod() - 1.0)
        out[wname] = {
            "月度重选": {"total": round(tot_rot, 4), "maxdd": round(maxdd(nav), 4)},
            "冻结持有": {"total": round(tot_hold, 4), "maxdd": round(maxdd(nav_hold), 4)},
            "diff": round(tot_rot - tot_hold, 4),
        }
        print(f"{wname}: 月度重选 {tot_rot:+.1%} (dd {maxdd(nav):.1%}) | "
              f"冻结持有 {tot_hold:+.1%} (dd {maxdd(nav_hold):.1%})", flush=True)

    hold_better = sum(1 for v in out.values() if v["diff"] < 0)
    avg_diff = float(np.mean([v["diff"] for v in out.values()]))
    print(json.dumps({"hold_better": f"{hold_better}/8", "avg_diff": round(avg_diff, 4)},
                     ensure_ascii=False))
    fp_out = ROOT / "reports" / "agent_loop" / "monthly_vs_hold_result.json"
    fp_out.write_text(json.dumps({"windows": out,
                                  "summary": {"hold_better": f"{hold_better}/8",
                                              "avg_diff": round(avg_diff, 4)}},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
