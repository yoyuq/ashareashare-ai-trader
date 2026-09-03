"""业绩预告超预期事件策略 — 预注册 reports/agent_loop/prereg_yjyg_pead.md.

A = 事件篮子 (近30自然日 预增/扭亏/略增 且幅度≥50, 等权月频);
B = 集中臂 (合格事件按幅度 top5)。
基准 = 主板等权; 11 窗, 65bp。输出: reports/agent_loop/yjyg_pead_result.json
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

COST = 65.0 / 1e4
AMT_MIN = 50.0        # 业绩变动幅度下限, 预注册写死
FRESH_DAYS = 30       # 公告新鲜度 (自然日), 写死
GOOD_TYPES = {"预增", "扭亏", "略增"}
SHARD_DIR = ROOT / "replay_data" / "yjyg"

OBJECTIVE = {"2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
             "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
             "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
             "2024震荡": "纯牛", "2025-26现期": "震荡"}


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_events() -> pd.DataFrame:
    frames = []
    for fp in sorted(SHARD_DIR.glob("*.parquet")):
        if fp.stat().st_size == 0:
            continue
        frames.append(pd.read_parquet(fp))
    if not frames:
        raise RuntimeError("yjyg 分片库为空")
    ev = pd.concat(frames, ignore_index=True)
    ev["code"] = ev["股票代码"].astype(str).str.zfill(6)
    ev = ev[ev["code"].str.startswith(("60", "00"))]
    ev["ann_date"] = pd.to_datetime(ev["公告日期"], errors="coerce")
    ev["amp"] = pd.to_numeric(ev["业绩变动幅度"], errors="coerce")
    # 同票同期多次公告取最新 (修正公告覆盖旧值); 跨期同票由新鲜度窗口天然处理
    ev = ev.sort_values("ann_date").drop_duplicates(subset=["code", "period"], keep="last")
    ev = ev.dropna(subset=["ann_date"])
    ev["sym"] = ev["code"].map(lambda c: ("sh." if c.startswith("6") else "sz.") + c)
    good = ev[ev["预告类型"].isin(GOOD_TYPES) & (ev["amp"] >= AMT_MIN)]
    print(f"events: {len(ev)} 归母预告行 → 合格 (预增/扭亏/略增 且幅度≥{AMT_MIN:.0f}): {len(good)}")
    return good


def month_ends(dates: list) -> list:
    s = pd.Series(pd.DatetimeIndex(dates))
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def run_window(df: pd.DataFrame, good: pd.DataFrame, arm: str) -> tuple[float, float, int, int]:
    """返回 (vsU_total, maxdd, 最后月持仓数, 月数)。"""
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    ret = close.pct_change(fill_method=None)
    dates = list(close.index)
    rl = month_ends(dates)
    dates_arr = pd.DatetimeIndex(dates)
    uni = ret.mean(axis=1).where(ret.notna().sum(axis=1) > 50)

    prev: set = set()
    ex_days, dd_days = [], []
    n_last, n_months = 0, 0
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else dates[-1]
        idx = dates_arr[(dates_arr > T) & (dates_arr <= T_next)]
        if idx.empty:
            continue
        ev_in = good[(good["ann_date"] > T - pd.Timedelta(days=FRESH_DAYS)) & (good["ann_date"] <= T)]
        cand = list(dict.fromkeys(ev_in["sym"]))
        avail = set(ret.loc[idx].columns[ret.loc[idx].notna().any()])
        cand = [c for c in cand if c in avail]
        if not cand:
            prev = set()
            continue
        sel = sorted(cand)[:5] if arm == "B_top5" else sorted(cand)
        n_last, n_months = len(sel), n_months + 1
        u_seg = uni.reindex(idx)
        port = ret.loc[idx, sel].mean(axis=1)
        new = len(set(sel) - prev)
        if i > 0 and new:
            port = port.copy()
            port.iloc[0] -= new / len(sel) * COST
        ex_days.append(port - u_seg)
        dd_days.append(port)
        prev = set(sel)

    if not ex_days:
        return float("nan"), float("nan"), 0, 0
    ex = pd.concat(ex_days).dropna()
    vsu = float((1.0 + ex).prod() - 1.0)
    eq = (1.0 + pd.concat(dd_days).dropna()).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return vsu, dd, n_last, n_months


def main() -> int:
    _force_utf8()
    good = load_events()
    out = {}
    print(f"{'window':<12} {'arm':<8} {'vsU':>8} {'maxdd':>8} {'n_hold':>6} {'months':>6}")
    for wname in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[wname])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[wname]}")
        ev_w = good[(good["ann_date"] >= df["date"].min() - pd.Timedelta(days=FRESH_DAYS))
                    & (good["ann_date"] <= df["date"].max())]
        out[wname] = {}
        for arm in ("A_basket", "B_top5"):
            vsu, dd, n_hold, n_m = run_window(df, ev_w, arm)
            out[wname][arm] = {"vsU": None if pd.isna(vsu) else round(vsu, 4),
                               "maxdd": None if pd.isna(dd) else round(dd, 4),
                               "n_hold": n_hold, "n_months": n_m}
            print(f"{wname:<12} {arm:<8} {vsu*100:>+7.1f}% {dd*100:>7.1f}% {n_hold:>6} {n_m:>6}")

    nonniu = [w for w in H.WINDOWS if OBJECTIVE[w] != "纯牛"]
    summary = {}
    for arm in ("A_basket", "B_top5"):
        vals = [out[w][arm]["vsU"] for w in nonniu if out[w][arm]["vsU"] is not None]
        wins = sum(1 for v in vals if v >= 0)
        avg = float(np.mean(vals)) if vals else float("nan")
        # sanity: 事件篮子广度断言 (2016窗持仓>10) + A/B 月数>0
        sanity_ok = out["2016熔断震荡"][arm]["n_hold"] > 10 and all(
            out[w][arm]["n_months"] > 0 for w in H.WINDOWS)
        summary[arm] = {"nonniu_valid": len(vals), "nonniu_vsU_wins": f"{wins}/{len(vals)}",
                        "avg_vsU": round(avg, 4),
                        "sanity": "PASS" if sanity_ok else "FAIL",
                        "gate": "PASS" if (len(vals) >= 8 and wins >= 7 and avg > 0 and sanity_ok) else "FAIL"}
    summary["rule"] = "非纯牛8窗 vsU≥0 ≥7/8 且 avg>0 且 sanity"
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "yjyg_pead_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
