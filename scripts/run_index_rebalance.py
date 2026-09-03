"""指数定期调样事件测试 (国证2000 调入) — 预注册 reports/agent_loop/prereg_index_rebalance.md.

事件 = 定期调样 (6/12月第二个周五次交易日) 主板调入票;
持有 = T_eff−10td 收盘 → T_eff 收盘, 65bp 往返; 超额 vs 主板等权同窗口。
输出: reports/agent_loop/index_rebalance_result.json
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import strategy_research_harness as H  # noqa: E402

COST_RT = 65.0 / 1e4
LEAD_TD = 10


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def scheduled_dates() -> set:
    """2021→2026 各 6/12 月第二个周五的次交易日历日期 (与生效日比对用, 粗粒度: 生效日所在窗口)。"""
    out = set()
    for y in range(2021, 2027):
        for m in (6, 12):
            d = date(y, m, 1)
            # 第一个周五
            while d.weekday() != 4:
                d = d.replace(day=d.day + 1)
            second_friday = d.replace(day=d.day + 7)
            out.add(second_friday)  # 生效日 = 次交易日, 只用于月份级过滤
    return out


def load_events() -> pd.DataFrame:
    import akshare as ak
    d = ak.index_detail_hist_adjust_cni(symbol="399303")
    d["start"] = pd.to_datetime(d["开始日期"])
    d["code"] = d["样本代码"].astype(str).str.zfill(6)
    ev = d[(d["start"] >= "2021-12-01") & (d["start"] <= "2026-08-31")].copy()
    # 定期调样: 生效日为 6/12 月 (第二个周五次交易日必然落在 6/12 月内)
    ev = ev[ev["start"].dt.month.isin((6, 12))]
    # 主板可执行子集: 00 开头, 排除创业板 300
    ev = ev[ev["code"].str.startswith("00") & ~ev["code"].str.startswith("300")]
    ev = ev.drop_duplicates(["code", "start"])
    print(f"定期调入事件 (主板): {len(ev)}, 生效期: {sorted(ev['start'].dt.strftime('%Y-%m').unique())}")
    return ev[["code", "start"]]


def main() -> int:
    _force_utf8()
    ev = load_events()
    # 预载所需窗口面板
    panels = {}
    for wname, fp in H.WINDOWS.items():
        df = H.load_base_window(fp)
        if not df.empty:
            panels[wname] = df
    rng = [(p["date"].min(), p["date"].max()) for p in panels.values()]

    def panel_for(t: pd.Timestamp):
        for wname, df in panels.items():
            d0, d1 = df["date"].min(), df["date"].max()
            if d0 <= t <= d1:
                return wname, df
        return None, None

    rows = []
    pivots = {}  # wname -> (close, uni_daily) 一次性构建, 防逐事件重建
    for wname, df in panels.items():
        cl = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        uni_daily = df.groupby("date")["ret"].mean()
        pivots[wname] = (cl, uni_daily)

    for _, r in ev.iterrows():
        t_eff = r["start"]
        wname, df = panel_for(t_eff)
        if df is None:
            continue
        close, uni_daily = pivots[wname]
        dates = pd.DatetimeIndex(close.index)
        if t_eff not in set(dates):
            continue
        ti = dates.get_loc(t_eff)
        if ti < LEAD_TD:
            continue
        t_in = dates[ti - LEAD_TD]
        seg = dates[(dates >= t_in) & (dates <= t_eff)]  # 含入场日 t_in (收盘买入)
        sym = ("sz." if r["code"].startswith(("0", "3")) else "sh.") + r["code"]
        if sym not in close.columns:
            continue
        px = close.loc[seg, sym].dropna()
        if len(px) < LEAD_TD or t_in not in px.index:
            continue  # 上市不满/停牌缺失 → 事件不可执行, 剔除并计数
        ret = float(px.iloc[-1] / px.iloc[0] - 1.0) - COST_RT
        u = uni_daily.reindex(seg).dropna()
        uret = float((1.0 + u).prod() - 1.0) if len(u) else np.nan
        rows.append({"eff": t_eff, "code": r["code"], "ret": ret, "uret": uret,
                     "excess": ret - uret if not np.isnan(uret) else np.nan,
                     "window": wname, "n_days": len(seg)})
    if not rows:
        raise RuntimeError("无有效事件")
    e = pd.DataFrame(rows)
    skipped = len(ev) - len(e)
    print(f"有效事件 {len(e)} (剔除不可执行/缺价 {skipped})")

    valid = e.dropna(subset=["excess"])
    n = len(valid)
    avg = float(valid["excess"].mean())
    win = float((valid["excess"] > 0).mean())
    # sanity
    days_ok = bool(valid["n_days"].between(10, 11).all())
    per_cohort = valid.groupby(valid["eff"].dt.strftime("%Y-%m"))["excess"].agg(["count", "mean"])
    sanity_ok = days_ok and (per_cohort["count"] > 0).all()
    gate = bool(n >= 500 and avg >= 0.005 and win >= 0.55 and sanity_ok)
    summary = {"n_events": n, "avg_excess": round(avg, 4), "win_rate": round(win, 4),
               "sanity": "PASS" if sanity_ok else "FAIL", "gate": "PASS" if gate else "FAIL",
               "rule": "N≥500 且 avg超额≥+0.5pp 且 胜率≥55% 且 sanity"}
    print(per_cohort.to_string())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "index_rebalance_result.json"
    fp.write_text(json.dumps({"summary": summary,
                              "by_cohort": {k: {"n": int(v["count"]), "avg": round(float(v["mean"]), 4)}
                                            for k, v in per_cohort.iterrows()},
                              "by_window": {w: {"n": int(len(g)), "avg": round(float(g["excess"].mean()), 4),
                                                "win": round(float((g["excess"] > 0).mean()), 4)}
                                            for w, g in valid.groupby("window")}},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
