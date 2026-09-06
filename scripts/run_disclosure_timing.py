"""财报披露时滞因子 A/B — 预注册 reports/agent_loop/prereg_disclosure_timing.md (C9, 逐字执行).

事件集 E(T) = universe 内 FIRST_APPOINT_DATE ∈ (T, T+30自然日] 的票 (未来30天披露年报).
A 主判据: E 内最早 1/3 等权; B 对照: E 内最晚 1/3; 均月末入次月末出, 65bp×2.
C 披露臂: E 内 T 前已公告推迟变更的票 (FIRST_CHANGE_DATE ≤ T 近似, 预注册披露).
窗分 = 窗内可评月末 (1-4月) 平均净差; gate: A−B ≥0 窗 ≥6/8 且 8窗平均 ≥ +1pp.
输出: reports/agent_loop/disclosure_timing_result.json
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

DS_FP = ROOT / "replay_data" / "disclosure_schedule_history.parquet"
OUT_FP = ROOT / "reports" / "agent_loop" / "disclosure_timing_result.json"

COST = 65.0 / 1e4
UNIVERSE_N = 800
EVENT_WINDOW_DAYS = 30
MIN_ARM = 5  # E 内臂 <5 只 → 该月不计入分母 (预注册)
WINDOWS = ["2018熊", "2019牛", "2020牛转崩", "2021白马转小盘",
           "2022熊", "2023震荡", "2024震荡", "2025-26现期"]
WIN_FILES = {
    "2018熊": "daily_2018-01-01_2018-12-31.parquet",
    "2019牛": "daily_2019-01-01_2019-12-31.parquet",
    "2020牛转崩": "daily_2020-06-01_2021-02-28.parquet",
    "2021白马转小盘": "daily_2021-01-01_2021-12-31.parquet",
    "2022熊": "daily_2022-01-01_2022-12-31.parquet",
    "2023震荡": "daily_2023-01-01_2023-12-31.parquet",
    "2024震荡": "daily_2024-01-01_2024-12-31.parquet",
    "2025-26现期": "daily_2025-10-08_2026-07-31.parquet",
}


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_schedule() -> pd.DataFrame:
    if not DS_FP.exists():
        raise RuntimeError(f"预约披露库缺失: {DS_FP} (零模拟, 先跑 build_disclosure_timing.py)")
    df = pd.read_parquet(DS_FP)
    df["code"] = df["SECURITY_CODE"].astype(str).str.zfill(6)
    df = df[df["code"].str.startswith(("60", "00"))].copy()
    df["sym"] = df["code"].map(lambda c: ("sh." if c.startswith("6") else "sz.") + c)
    # 年报: REPORT_DATE = 12-31
    df = df[df["REPORT_DATE"].dt.month == 12].copy()
    df["fy"] = df["REPORT_DATE"].dt.year
    return df.dropna(subset=["FIRST_APPOINT_DATE"])[
        ["sym", "fy", "FIRST_APPOINT_DATE", "FIRST_CHANGE_DATE", "APPOINT_CHANGE"]]


def month_ends(dates: pd.DatetimeIndex) -> list:
    s = pd.Series(dates)
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def run_window(win: str, sched: pd.DataFrame) -> dict:
    fp = ROOT / "replay_data" / WIN_FILES[win]
    df = pd.read_parquet(fp)
    df["date"] = pd.to_datetime(df["date"])
    df["isST"] = df["isST"].astype(str)
    df["tradestatus"] = df["tradestatus"].astype(str)
    df = df[(df["is_trade"] == 1) & (df["isST"] == "0") & (df["tradestatus"] == "1")]
    df = df[df["symbol"].str.startswith(("sh.60", "sz.00"))]
    for c in ("close", "amount", "peTTM", "pbMRQ"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["peTTM"] > 0) & (df["pbMRQ"] > 0)]
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    px = {d: g.set_index("symbol") for d, g in df.groupby("date")}
    tes = month_ends(dates)

    months = []
    for T, Tn in zip(tes, tes[1:]):
        # 只评年报披露季月末 (1-4月)
        if T.month not in (1, 2, 3, 4):
            continue
        uni = px[T]
        uni = uni[uni["amount"] > 0].nlargest(UNIVERSE_N, "amount")
        if len(uni) < 100:
            continue
        uni_next = px[Tn]
        fwd = (uni_next["close"].reindex(uni.index) / uni["close"] - 1.0).dropna()
        uni_ret = float(fwd.mean())
        fy = T.year - 1
        sc = sched[sched["fy"] == fy].set_index("sym")
        fa = sc["FIRST_APPOINT_DATE"].reindex(uni.index)
        ev = fa.dropna()
        ev = ev[(ev > T) & (ev <= T + pd.Timedelta(days=EVENT_WINDOW_DAYS))]
        ev_syms = ev.index
        if len(ev_syms) < 3 * MIN_ARM:
            continue
        order = ev.loc[ev_syms].sort_values().index
        k = max(1, len(order) // 3)
        a_syms = [s for s in order[:k] if s in fwd.index]
        b_syms = [s for s in order[-k:] if s in fwd.index]
        if len(a_syms) < MIN_ARM or len(b_syms) < MIN_ARM:
            continue
        a_net = float(fwd.loc[a_syms].mean()) - 2 * COST
        b_net = float(fwd.loc[b_syms].mean()) - 2 * COST
        # C 臂: E 内 T 前已公告变更 (FIRST_CHANGE_DATE <= T, 近似披露)
        chg = sc["FIRST_CHANGE_DATE"].reindex(ev_syms)
        c_syms = [s for s in chg[chg.notna() & (chg <= T)].index if s in fwd.index]
        c_net = (float(fwd.loc[c_syms].mean()) - 2 * COST) if len(c_syms) >= MIN_ARM else None
        months.append({
            "T": str(T.date()), "n_event": int(len(ev_syms)), "n_a": len(a_syms),
            "n_b": len(b_syms), "n_c": len(c_syms),
            "a_net": round(a_net, 4), "b_net": round(b_net, 4),
            "c_net": round(c_net, 4) if c_net is not None else None,
            "ab_diff": round(a_net - b_net, 4),
            "a_vs_uni": round(a_net - uni_ret, 4),
        })

    ab = [m["ab_diff"] for m in months]
    return {"window": win, "months": months, "n_scored": len(months),
            "ab_avg": round(float(np.mean(ab)), 4) if ab else None,
            "c_avg": (lambda v: round(float(np.mean(v)), 4) if v else None)(
                [m["c_net"] for m in months if m["c_net"] is not None])}


def main() -> int:
    _force_utf8()
    sched = load_schedule()
    print(f"schedule: {len(sched)} rows, {sched['sym'].nunique()} syms, "
          f"FY {sched['fy'].min()} -> {sched['fy'].max()}")
    rows = []
    for win in WINDOWS:
        try:
            r = run_window(win, sched)
        except FileNotFoundError as e:
            raise RuntimeError(f"窗口 {win} 日线缺失 (零模拟): {e}")
        rows.append(r)
        ab = r["ab_avg"]
        print(f"{win:<14} A-B {ab if ab is not None else 'n/a':>8} "
              f"(months {r['n_scored']}) C {r['c_avg']}")

    scored = [r for r in rows if r["n_scored"] > 0]
    n_pos = sum(1 for r in scored if r["ab_avg"] is not None and r["ab_avg"] >= 0)
    avgs = [r["ab_avg"] for r in scored if r["ab_avg"] is not None]
    overall = float(np.mean(avgs)) if avgs else None
    verdict = {
        "prereg": "prereg_disclosure_timing.md (gate: A−B 6/8窗≥0 且 平均≥+1pp)",
        "windows": rows,
        "n_windows_scored": len(scored),
        "ab_pos_windows": n_pos,
        "ab_overall_avg": round(overall, 4) if overall is not None else None,
        "gate_pass": bool(len(scored) >= 6 and n_pos >= 6
                          and overall is not None and overall >= 0.01),
    }
    OUT_FP.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGATE: {'PASS' if verdict['gate_pass'] else 'FAIL'} "
          f"({n_pos}/{len(scored)} 窗 ≥0, 平均 {overall})")
    print(f"saved -> {OUT_FP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
