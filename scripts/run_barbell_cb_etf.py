"""哑铃组合: 冷落低波 70% + 转债等权代理 30% vs 100% 冷落低波 (预注册 prereg_barbell_cb_etf.md).

输出: reports/agent_loop/barbell_cb_etf_result.json
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

W_CB, W_STK = 0.3, 0.7


def cb_ew_returns() -> pd.Series:
    daily = pd.read_parquet(ROOT / "replay_data" / "cb_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    close = daily.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    ret = close.pct_change(fill_method=None)
    return ret.mean(axis=1).where(ret.notna().sum(axis=1) >= 3)


def main() -> int:
    cb = cb_ew_returns()
    out = {}
    for wname, fp in WINDOWS.items():
        valid, sel, nav, dates = run_window(fp)
        end = dates[-1]
        idx = nav.index
        cb_seg = cb.reindex(idx)
        if cb_seg.isna().all():
            raise RuntimeError(f"{wname}: 转债等权收益全缺失 (零模拟中止)")
        barbell = W_STK * nav + W_CB * cb_seg.fillna(0.0)
        pure = float((1.0 + nav).prod() - 1.0)
        bb = float((1.0 + barbell).prod() - 1.0)
        out[wname] = {
            "纯股票": {"total": round(pure, 4), "maxdd": round(maxdd(nav), 4)},
            "哑铃70/30": {"total": round(bb, 4), "maxdd": round(maxdd(barbell), 4)},
            "total_diff": round(bb - pure, 4),
            "dd_improved": bool(maxdd(barbell) > maxdd(nav)),
        }
        print(f"{wname}: 纯 {pure:+.1%} (dd {maxdd(nav):.1%}) | 哑铃 {bb:+.1%} "
              f"(dd {maxdd(barbell):.1%})", flush=True)

    dd_wins = sum(1 for v in out.values() if v["dd_improved"])
    ret_ok = sum(1 for v in out.values() if v["total_diff"] >= -0.01)
    verdict = bool(dd_wins >= 5 and ret_ok >= 5)
    summary = {"dd_improved": f"{dd_wins}/8", "ret_within_1pp": f"{ret_ok}/8", "PASS": verdict}
    print(json.dumps(summary, ensure_ascii=False))
    fp_out = ROOT / "reports" / "agent_loop" / "barbell_cb_etf_result.json"
    fp_out.write_text(json.dumps({"windows": out, "summary": summary},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
