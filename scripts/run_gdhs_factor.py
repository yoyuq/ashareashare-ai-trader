"""股东户数变化因子 A/B — 预注册 reports/agent_loop/prereg_gdhs_factor.md (C8, 逐字执行).

信号 (PIT): 月末 T 取 announce_date<=T 最新一条的 holder_num_ratio_pct (%).
A 主判据: 信号<=-10% 集中带等权, 月末入次月末出, 65bp x2.
B/C/D 披露臂: -30%<-s<=-10% / <=-30% / >=+20%.
universe: 主板(60/00) 非ST 可交易 pe>0 pb>0 top-800 by amount.
gate: A vs universe 净差>=0 窗 >=6/8 且 8窗平均 >=+1pp.
输出: reports/agent_loop/gdhs_factor_result.json
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

GDHS_FP = ROOT / "replay_data" / "gdhs_history.parquet"
OUT_FP = ROOT / "reports" / "agent_loop" / "gdhs_factor_result.json"

COST = 65.0 / 1e4  # 单边; 月末入+次月末出 = 2 边
UNIVERSE_N = 800
BASKET_MIN = 10  # 月均 <10 只 => 窗口 degenerate (预注册)
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
ARMS = {
    "A": lambda s: s <= -10.0,
    "B": lambda s: (-30.0 < s) & (s <= -10.0),
    "C": lambda s: s <= -30.0,
    "D": lambda s: s >= 20.0,
}


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_signals() -> pd.DataFrame:
    if not GDHS_FP.exists():
        raise RuntimeError(f"股东户数库缺失: {GDHS_FP} (零模拟, 先跑 build_gdhs_history.py)")
    df = pd.read_parquet(GDHS_FP)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["code"].str.startswith(("60", "00"))].copy()
    df["sym"] = df["code"].map(lambda c: ("sh." if c.startswith("6") else "sz.") + c)
    df = df.dropna(subset=["announce_date", "holder_num_ratio_pct"])
    # 同票同期重复公告 -> 保留最新公告 (预注册 keep=last by announce_date)
    df = df.sort_values(["sym", "announce_date"])
    df = df.drop_duplicates(subset=["sym", "period_end"], keep="last")
    return df[["sym", "announce_date", "holder_num_ratio_pct"]].reset_index(drop=True)


def month_ends(dates: pd.DatetimeIndex) -> list:
    s = pd.Series(dates)
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def signal_at(signals: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    """每票 announce_date<=T 的最新 holder_num_ratio_pct."""
    sub = signals[signals["announce_date"] <= T]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.sort_values("announce_date").groupby("sym")["holder_num_ratio_pct"].last()


def run_window(win: str, signals: pd.DataFrame) -> dict:
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
    pairs = [(tes[i], tes[i + 1]) for i in range(len(tes) - 1)]

    arm_diffs = {a: [] for a in ARMS}
    n_basket, n_uni_miss, n_ev = [], 0, 0
    for T, Tn in pairs:
        uni = px[T]
        uni = uni[uni["amount"] > 0].nlargest(UNIVERSE_N, "amount")
        if len(uni) < 100:
            continue
        uni_next = px[Tn]
        fwd = (uni_next["close"].reindex(uni.index) / uni["close"] - 1.0).dropna()
        uni_ret = float(fwd.mean())
        if np.isnan(uni_ret):
            n_uni_miss += 1
            continue
        sig = signal_at(signals, T).reindex(uni.index).dropna()
        row = {"n_basket": {}, "n_uni": len(uni)}
        for a, mask_fn in ARMS.items():
            sel = sig[mask_fn(sig)].index.intersection(fwd.index)
            if len(sel) == 0:
                arm_diffs[a].append(np.nan)
                row["n_basket"][a] = 0
                continue
            gross = float(fwd.loc[sel].mean())
            net = gross - 2 * COST
            arm_diffs[a].append(net - uni_ret)
            row["n_basket"][a] = len(sel)
            n_ev += len(sel)
        n_basket.append(row)

    a_clean = [x for x in arm_diffs["A"] if not np.isnan(x)]
    n_months = len(a_clean)
    sizes = [r["n_basket"]["A"] for r in n_basket]
    degenerate = (n_months == 0
                  or (float(np.mean(sizes)) < BASKET_MIN if sizes else True))
    out = {"window": win, "n_month_ends": n_months, "degenerate": bool(degenerate),
           "arms": {}, "basket_sizes_last": n_basket[-1] if n_basket else None}
    for a, diffs in arm_diffs.items():
        clean = [x for x in diffs if not np.isnan(x)]
        out["arms"][a] = {
            "diffs": [round(x, 4) for x in clean],
            "avg": round(float(np.mean(clean)), 4) if clean else None,
        }
    return out


def main() -> int:
    _force_utf8()
    signals = load_signals()
    print(f"signals: {len(signals)} rows, {signals['sym'].nunique()} syms, "
          f"{signals['announce_date'].min().date()} -> {signals['announce_date'].max().date()}")
    rows = []
    for win in WINDOWS:
        try:
            r = run_window(win, signals)
        except FileNotFoundError as e:
            raise RuntimeError(f"窗口 {win} 日线缺失 (零模拟): {e}")
        rows.append(r)
        arms = r["arms"]
        fmt = lambda a: (f"{arms[a]['avg']:+.4f}" if arms[a]["avg"] is not None else "  n/a ")
        print(f"{win:<14} A {fmt('A')}  B {fmt('B')}  C {fmt('C')}  D {fmt('D')} "
              f"(months {r['n_month_ends']}{', DEGENERATE' if r['degenerate'] else ''})")

    scored = [r for r in rows if not r["degenerate"]]
    n_pos = sum(1 for r in scored if r["arms"]["A"]["avg"] is not None and r["arms"]["A"]["avg"] >= 0)
    avgs = [r["arms"]["A"]["avg"] for r in scored if r["arms"]["A"]["avg"] is not None]
    overall = float(np.mean(avgs)) if avgs else None
    verdict = {
        "prereg": "prereg_gdhs_factor.md (gate: A臂 6/8窗净差>=0 且 平均>=+1pp)",
        "windows": rows,
        "n_windows_scored": len(scored),
        "a_pos_windows": n_pos,
        "a_overall_avg": round(overall, 4) if overall is not None else None,
        "gate_pass": bool(len(scored) >= 6 and n_pos >= 6
                          and overall is not None and overall >= 0.01),
    }
    OUT_FP.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGATE: {'PASS' if verdict['gate_pass'] else 'FAIL'} "
          f"({n_pos}/{len(scored)} 窗 >=0, 平均 {overall})")
    print(f"saved -> {OUT_FP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
