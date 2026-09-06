"""回购公告事件篮子回测 — 预注册 reports/agent_loop/prereg_buyback_event.md (C6, 逐字执行).

事件: stock_repurchase_em 全市场快照 (冻结 replay_data/repurchase_history.parquet).
binding = 计划金额下限>=1000万 & 占公告前一日总股本比例下限>=1% & 进度∈{董事会预案,
股东大会通过, 实施中} (排除 停止实施/否决/完成实施 行).
universe = 主板(60/00) 非ST 可交易 pe>0 pb>0 top-800 by amount (匹配基准纪律).
篮子 = universe 内 binding 事件 (月末前30自然日) 等权, 月末收盘入, 持有60td收盘出,
65bp/单边成本 (入场扣一次, 与 universe 基准同扣法).
B臂披露: 按占比上限 top10 (不作主判据).
gate: A臂 vs universe 净差 >=0 的窗口 >= 6/8 且 8窗平均 >= +2pp.
输出: reports/agent_loop/buyback_event_result.json
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

REP_FP = ROOT / "replay_data" / "repurchase_history.parquet"
OUT_FP = ROOT / "reports" / "agent_loop" / "buyback_event_result.json"

COST = 65.0 / 1e4          # 单边, 入场扣一次 (prereg 写死)
AMOUNT_MIN = 10_000_000.0  # 计划金额下限 1000万
RATIO_MIN = 0.01           # 占总股本比例下限 1%
HOLD_TD = 60               # 60 交易日 forward
EVENT_LOOKBACK_NATDAYS = 30
UNIVERSE_N = 800
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
PROGRESS_KEEP = ("董事会预案", "股东大会通过", "实施中")


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_binding() -> pd.DataFrame:
    if not REP_FP.exists():
        raise RuntimeError(f"回购库缺失: {REP_FP} (零模拟, 先冻结)")
    df = pd.read_parquet(REP_FP)
    df.columns = [str(c).strip() for c in df.columns]
    code = df["股票代码"].astype(str).str.zfill(6)
    df = df[code.str.startswith(("60", "00"))].copy()
    df["code"] = code[code.str.startswith(("60", "00"))]
    df["announce"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df["amt_low"] = pd.to_numeric(df["计划回购金额区间-下限"], errors="coerce")
    df["ratio_low"] = pd.to_numeric(df["占公告前一日总股本比例-下限"], errors="coerce")
    df["ratio_high"] = pd.to_numeric(df["占公告前一日总股本比例-上限"], errors="coerce")
    df = df.dropna(subset=["announce"])
    b = df[(df["amt_low"] >= AMOUNT_MIN) & (df["ratio_low"] >= RATIO_MIN)
           & (df["实施进度"].isin(PROGRESS_KEEP))].copy()
    b["sym"] = b["code"].map(lambda c: ("sh." if c.startswith("6") else "sz.") + c)
    return b.drop_duplicates(subset=["sym", "announce"])


def month_ends(dates: pd.DatetimeIndex) -> list:
    s = pd.Series(dates)
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def run_window(win: str, binding: pd.DataFrame) -> dict:
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
    pos = {d: i for i, d in enumerate(dates)}

    # per-date universe: top-800 by amount; per-symbol close 前向60td
    df["fwd60"] = df.groupby("symbol")["close"].shift(-HOLD_TD) / df["close"] - 1.0
    close_px = {d: g.set_index("symbol") for d, g in df.groupby("date")}

    tes = month_ends(dates)
    a_diffs, b_diffs, n_ev, n_uni_miss = [], [], 0, 0
    for T in tes:
        if pos[T] + HOLD_TD >= len(dates):
            continue  # 窗尾不足60td持有期, 该月末不评 (剩余月末多数仍在)
        uni = close_px[T]
        uni = uni[uni["amount"] > 0].nlargest(UNIVERSE_N, "amount")
        if len(uni) < 100:
            continue
        uni_ret = float(uni["fwd60"].dropna().mean())
        if np.isnan(uni_ret):
            continue
        t0 = T - pd.Timedelta(days=EVENT_LOOKBACK_NATDAYS)
        ev = binding[(binding["announce"] > t0) & (binding["announce"] <= T)]
        ev = ev[ev["sym"].isin(uni.index)]
        if ev.empty:
            a_diffs.append(np.nan); b_diffs.append(np.nan); n_ev += 0
            continue
        a_ret = float(uni.loc[uni.index.intersection(ev["sym"]), "fwd60"].dropna().mean()) - COST
        a_diffs.append(a_ret - uni_ret)
        # B臂: 占比上限 top10
        top = ev.dropna(subset=["ratio_high"]).nlargest(10, "ratio_high")
        if len(top):
            b_ret = float(uni.loc[uni.index.intersection(top["sym"]), "fwd60"].dropna().mean()) - COST
            b_diffs.append(b_ret - uni_ret)
        else:
            b_diffs.append(np.nan)
        n_ev += len(ev)
    a = [x for x in a_diffs if not np.isnan(x)]
    b = [x for x in b_diffs if not np.isnan(x)]
    return {
        "window": win,
        "a_diffs_monthly": [round(x, 4) for x in a],
        "a_avg": round(float(np.mean(a)), 4) if a else None,
        "b_avg": round(float(np.mean(b)), 4) if b else None,
        "n_binding_events_total": n_ev,
        "n_month_ends": len(a),
    }


def main() -> int:
    _force_utf8()
    binding = load_binding()
    print(f"binding events: {len(binding)} rows, {binding['sym'].nunique()} syms, "
          f"{binding['announce'].min().date()} -> {binding['announce'].max().date()}")
    rows = []
    for win in WINDOWS:
        try:
            r = run_window(win, binding)
        except FileNotFoundError as e:
            raise RuntimeError(f"窗口 {win} 日线缺失 (零模拟): {e}")
        rows.append(r)
        print(f"{win:<14} A vsU avg {r['a_avg'] if r['a_avg'] is not None else 'n/a':>8} "
              f"(months {r['n_month_ends']}, events {r['n_binding_events_total']}) "
              f"B {r['b_avg']}")
    scored = [r for r in rows if r["a_avg"] is not None]
    n_pos = sum(1 for r in scored if r["a_avg"] >= 0)
    overall_avg = float(np.mean([r["a_avg"] for r in scored])) if scored else None
    verdict = {
        "prereg": "prereg_buyback_event.md (gate: 6/8窗净差>=0 且 平均>=+2pp)",
        "windows": rows,
        "n_windows_scored": len(scored),
        "a_pos_windows": n_pos,
        "a_overall_avg": round(overall_avg, 4) if overall_avg is not None else None,
        "gate_pass": bool(len(scored) >= 6 and n_pos >= 6
                          and overall_avg is not None and overall_avg >= 0.02),
    }
    OUT_FP.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGATE: {'PASS' if verdict['gate_pass'] else 'FAIL'} "
          f"({n_pos}/{len(scored)} 窗 >=0, 平均 {overall_avg if overall_avg is not None else 'n/a'})")
    print(f"saved -> {OUT_FP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
