"""限售解禁临近规避 screen — 预注册 reports/agent_loop/prereg_restricted_screen.md.

主臂: 全主板 − blocked(未来30交易日内解禁且占流通市值≥5%) 等权月频, vs 全主板等权, 10窗, 65bp。
披露: blocked 篮子自身收益 (文献负效应方向验证)。
输出: reports/agent_loop/restricted_screen_result.json
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
AHEAD_TD = 30       # 解禁前规避窗口 (交易日), 预注册写死
RATIO_MIN = 0.05    # 占解禁前流通市值比例阈值, 写死
SHARD_DIR = ROOT / "replay_data" / "restricted_release"

WINDOWS = ["2016熔断震荡", "2017漂亮50", "2018熊", "2019牛", "2020牛转崩",
           "2021白马转小盘", "2022熊", "2023震荡", "2024震荡", "2025-26现期"]


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
        d = pd.read_parquet(fp)
        if d.empty:
            continue
        d = d[["股票代码", "解禁时间", "解禁数量", "占解禁前流通市值比例",
               "解禁前20日涨跌幅", "解禁后20日涨跌幅"]]
        frames.append(d)
    if not frames:
        raise RuntimeError("restricted_release 分片库为空")
    ev = pd.concat(frames, ignore_index=True)
    ev["code"] = ev["股票代码"].astype(str).str.zfill(6)
    ev = ev[ev["code"].str.startswith(("60", "00"))]
    ev["free_date"] = pd.to_datetime(ev["解禁时间"], errors="coerce")
    ev["ratio"] = pd.to_numeric(ev["占解禁前流通市值比例"], errors="coerce")
    ev = ev.dropna(subset=["free_date"])
    ev["sym"] = ev["code"].map(lambda c: ("sh." if c.startswith("6") else "sz.") + c)
    return ev.drop_duplicates(subset=["sym", "free_date"])


def month_ends(dates: list) -> list:
    s = pd.Series(pd.DatetimeIndex(dates))
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def main() -> int:
    _force_utf8()
    ev = load_events()
    print(f"events: {len(ev)} rows, {ev['sym'].nunique()} syms, "
          f"{ev['free_date'].min().date()} -> {ev['free_date'].max().date()}")
    big = ev[ev["ratio"] >= RATIO_MIN]
    print(f"big events (ratio>={RATIO_MIN:.0%}): {len(big)}")

    out = {}
    print(f"{'window':<14} {'vsU':>8} {'blocked_n':>9} {'blk_vsU':>8}")
    for wname in WINDOWS:
        df = H.load_base_window(H.WINDOWS[wname])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[wname]}")
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret = close.pct_change(fill_method=None)
        dates = list(close.index)
        rl = month_ends(dates)
        dates_arr = pd.DatetimeIndex(dates)

        prev_blocked: set = set()
        main_days, blk_days = [], []
        n_blk_last = 0
        for i, T in enumerate(rl):
            T_next = rl[i + 1] if i + 1 < len(rl) else dates[-1]
            idx = dates_arr[(dates_arr > T) & (dates_arr <= T_next)]
            if idx.empty:
                continue
            # 未来30交易日 (含 T 自身) 有大额解禁的票
            horizon = dates_arr[(dates_arr >= T) & (dates_arr <= dates[min(len(dates) - 1, dates.index(T) + AHEAD_TD)])]
            ev_in = big[big["free_date"].isin(horizon)]
            blocked = set(ev_in["sym"]) & set(close.columns)
            n_blk_last = len(blocked)
            avail = ret.loc[idx].columns[ret.loc[idx].notna().any()]
            keep = [c for c in avail if c not in blocked]
            bl = [c for c in avail if c in blocked]
            uni = ret.mean(axis=1).where(ret.notna().sum(axis=1) > 50).reindex(idx)
            if keep:
                port = ret.loc[idx, keep].mean(axis=1)
                new_blk = len(blocked - prev_blocked)
                cost = new_blk / max(len(keep) + len(bl), 1) * COST
                port = port.copy()
                port.iloc[0] -= cost
                main_days.append(port - uni)
            if bl:
                blk_days.append(ret.loc[idx, bl].mean(axis=1) - uni)
            prev_blocked = blocked

        def tot(days):
            if not days:
                return None
            ex = pd.concat(days).dropna()
            return round(float((1.0 + ex).prod() - 1.0), 4) if len(ex) else None

        out[wname] = {"main_vsU": tot(main_days), "blocked_vsU": tot(blk_days),
                      "n_blocked_last": n_blk_last}
        print(f"{wname:<14} {str(out[wname]['main_vsU']):>8} {n_blk_last:>9} "
              f"{str(out[wname]['blocked_vsU']):>8}")

    act = {w: v["main_vsU"] for w, v in out.items() if v["main_vsU"] is not None}
    wins = sum(1 for v in act.values() if v >= 0)
    avg = float(np.mean(list(act.values()))) if act else float("nan")
    blk_vals = [v["blocked_vsU"] for v in out.values() if v["blocked_vsU"] is not None]
    summary = {"main_gate": {"valid_windows": len(act), "vsU_wins": f"{wins}/{len(act)}",
                             "avg_vsU": round(avg, 4),
                             "PASS": bool(len(act) >= 6 and wins >= 6 and avg > 0)},
               "blocked_basket_avg_vsU": round(float(np.mean(blk_vals)), 4) if blk_vals else None}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "restricted_screen_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
