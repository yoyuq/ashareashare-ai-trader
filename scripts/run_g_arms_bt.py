"""G 臂回测: G1 龙虎榜机构跟随 / G2 二月小盘效应 / G3 双低+强赎计数回避.
预注册: reports/calendar_lhb_preregistration.md (跑之前写死).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "g_arms_result.json"
COST_STOCK = 31.0
COST_CB_MAIN, COST_CB_LOW = 20.0, 5.0
TOP_K_CB = 10
REDEEM_WINDOW, REDEEM_TH, REDEEM_COUNT = 30, 130.0, 10
GATE_YEARS_MIN = 8  # /12 for G1, /10 for G2


# ---------------------------------------------------------------- G1 LHB ----
def run_g1() -> dict:
    lhb = pd.read_parquet(ROOT / "replay_data" / "lhb_history.parquet")
    lhb["上榜日"] = pd.to_datetime(lhb["上榜日"])
    lhb["code"] = lhb["代码"].astype(str).str.zfill(6)
    inst = lhb[lhb["解读"].astype(str).str.contains("机构买入", na=False)].copy()
    inst = inst.drop_duplicates(subset=["code", "上榜日"])
    inst["year"] = inst["上榜日"].dt.year

    # 载入全部年度日线文件
    frames = []
    for fp in sorted(ROOT.glob("replay_data/daily_*.parquet")):
        if fp.stem.endswith("_idx"):
            continue
        try:
            df = pd.read_parquet(fp, columns=["date", "symbol", "open", "close", "is_trade", "isST"])
        except Exception as e:
            print(f"[warn] 跳过损坏文件 {fp.name}: {type(e).__name__}")
            continue
        frames.append(df)
    d = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "symbol"])
    d["date"] = pd.to_datetime(d["date"])
    d["isST"] = d["isST"].astype(str)
    d = d[(d["is_trade"] == 1) & (d["isST"] == "0") & d["symbol"].str.startswith(("sh.60", "sz.00"))]
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)

    # symbol ↔ code 映射
    d["code"] = d["symbol"].str[3:]
    cal = d[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    cal["pos"] = np.arange(len(cal))

    d = d.merge(cal.rename(columns={"date": "date", "pos": "pos"}), on="date", how="left")
    open_px = d.pivot_table(index="pos", columns="code", values="open")
    close_px = d.pivot_table(index="pos", columns="code", values="close")
    date_by_pos = cal.set_index("pos")["date"]

    picks = []
    for _, r in inst.iterrows():
        code, dT = r["code"], r["上榜日"]
        if code not in open_px.columns:
            continue
        # T+1 交易日
        i0 = int(date_by_pos[date_by_pos > dT].index.min()) if (date_by_pos > dT).any() else None
        if i0 is None:
            continue
        i5 = i0 + 5  # 持有5交易日 → T+6开盘卖
        if i5 not in open_px.index:
            continue
        e = open_px.at[i0, code]
        x = open_px.at[i5, code]
        cT = close_px.at[i0 - 1, code] if (i0 - 1) in close_px.index else np.nan
        if not (np.isfinite(e) and np.isfinite(x) and e > 0):
            continue
        picks.append({"code": code, "date": dT.date(), "year": dT.year,
                      "gap_open": round(e / cT - 1, 4) if np.isfinite(cT) and cT > 0 else np.nan,
                      "net_ret": round(x / e - 1 - COST_STOCK / 1e4, 4)})
    p = pd.DataFrame(picks)
    if p.empty:
        return {"G1_龙虎榜机构跟随": {"n_trades": 0}}
    yearly = {int(y): {"n": int(len(g)), "mean": round(float(g["net_ret"].mean()), 4),
                       "median": round(float(g["net_ret"].median()), 4),
                       "win": round(float((g["net_ret"] > 0).mean()), 4)}
              for y, g in p.groupby("year")}
    return {"G1_龙虎榜机构跟随": {
        "n_trades": int(len(p)),
        "mean_net": round(float(p["net_ret"].mean()), 4),
        "median_net": round(float(p["net_ret"].median()), 4),
        "win_rate": round(float((p["net_ret"] > 0).mean()), 4),
        "gap_open_median": round(float(p["gap_open"].median()), 4),
        "pos_years": f"{sum(1 for v in yearly.values() if v['mean'] > 0)}/{len(yearly)}",
        "by_year": yearly,
        "PASS": bool(p["net_ret"].mean() > 0 and p["net_ret"].median() > 0
                     and sum(1 for v in yearly.values() if v["mean"] > 0) >= GATE_YEARS_MIN),
    }}, p


# ----------------------------------------------------------------- G2 FEB ----
def run_g2() -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from strategy_research_harness import load_fundamentals, pt_fundamental_col

    fund = load_fundamentals()
    years = range(2015, 2025)
    out = {"pool": [], "top10": [], "index": []}
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx = idx[idx["symbol"] == "sh.000001"][["date", "close"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])

    for y in years:
        fp = ROOT / "replay_data" / f"daily_{y}-01-01_{y}-12-31.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])
        df["isST"] = df["isST"].astype(str)
        df = df[(df["is_trade"] == 1) & (df["isST"] == "0") & df["symbol"].str.startswith(("sh.60", "sz.00"))]
        feb = df[(df["date"].dt.month == 2)]
        if feb.empty:
            continue
        dates = sorted(feb["date"].unique())
        t0, t1 = dates[0], dates[-1]
        day0 = df[df["date"] == t0].copy()
        day0["mktcap"] = day0["close"] * pt_fundamental_col(day0.assign(date=t0), fund, "totalShare")
        pool = day0.dropna(subset=["mktcap"])
        if len(pool) < 50:
            continue
        small = pool[pool["mktcap"] <= pool["mktcap"].quantile(1 / 3.0)]
        px = df.pivot_table(index="date", columns="symbol", values="close")

        def port_ret(universe: list[str]) -> float:
            seg = px.loc[t0:t1, universe]
            r = (seg.iloc[-1] / seg.iloc[0] - 1).dropna()
            return float(r.mean()) - COST_STOCK / 1e4

        out["pool"].append({"year": y, "ret": round(port_ret(list(small["symbol"])), 4)})
        top10 = list(small.nsmallest(10, "mktcap")["symbol"])
        out["top10"].append({"year": y, "ret": round(port_ret(top10), 4)})
        seg = idx[(idx["date"] >= t0) & (idx["date"] <= t1)]
        out["index"].append({"year": y, "ret": round(float(seg["close"].iloc[-1] / seg["close"].iloc[0] - 1), 4)})

    def summ(key):
        arr = [x["ret"] for x in out[key]]
        pos = sum(1 for v in arr if v > 0)
        return {"n_years": len(arr), "pos_years": f"{pos}/{len(arr)}",
                "avg": round(float(np.mean(arr)), 4)}

    s_pool, s_top10 = summ("pool"), summ("top10")
    s_pool["PASS"] = bool(s_pool["n_years"] >= 10 and int(s_pool["pos_years"].split("/")[0]) >= 8
                          and s_pool["avg"] > 0)
    return {"G2_二月小盘效应": {"pool": s_pool, "top10_可执行投影": s_top10,
                          "detail": out}}


# ------------------------------------------------------------------ G3 CB ----
def run_g3() -> dict:
    daily = pd.read_parquet(ROOT / "replay_data" / "cb_daily.parquet")
    value = pd.read_parquet(ROOT / "replay_data" / "cb_value.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    value["date"] = pd.to_datetime(value["日期"])
    value["symbol"] = value["symbol"].astype(str)

    def wide(df, col):
        return df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").sort_index()

    close, prem, cv = wide(daily, "close"), wide(value, "转股溢价率"), wide(value, "转股价值")
    cols = close.columns.intersection(prem.columns)
    close, prem, cv = close[cols], prem[cols], cv[cols]
    valid = close.notna() & prem.notna()
    close, prem, cv = close.where(valid), prem.where(valid), cv.where(valid)
    ret = close.pct_change(fill_method=None)

    # 强赎计数: 滚动30日 转股价值>=130 的天数 >= 10 → 禁选
    hot = (cv >= REDEEM_TH).rolling(REDEEM_WINDOW, min_periods=1).sum()
    redeem_mask = hot >= REDEEM_COUNT

    score = -(close + prem * 100.0).where(~redeem_mask)
    dates = close.index
    s = pd.Series(dates)
    rl = s.groupby([s.dt.isocalendar().year.astype(int), s.dt.isocalendar().week.astype(int)])\
          .apply(lambda x: x.iloc[-1]).tolist()
    sel = {}
    for T in rl:
        row = score.loc[T].dropna()
        sel[T] = row.nlargest(TOP_K_CB).index.tolist() if not row.empty else []

    WINDOWS = {
        "2018熊": ("2018-01-01", "2018-12-31"), "2019牛": ("2019-01-01", "2019-12-31"),
        "2020牛转崩": ("2020-06-01", "2021-02-28"), "2021白马转小盘": ("2021-01-01", "2021-12-31"),
        "2022熊": ("2022-01-01", "2022-12-31"), "2023震荡": ("2023-01-01", "2023-12-31"),
        "2024震荡": ("2024-01-01", "2024-12-31"), "2025-26现期": ("2025-10-08", "2026-07-31"),
    }
    bench = ret.mean(axis=1).where(ret.notna().sum(axis=1) >= 3)

    def metrics(r):
        r = r.dropna()
        total = float((1 + r).prod() - 1)
        eq = (1 + r).cumprod()
        return {"total": round(total, 4), "maxdd": round(float((eq / eq.cummax() - 1).min()), 4)}

    res = {}
    for w, (t0, t1) in WINDOWS.items():
        port = pd.Series(np.nan, index=dates)
        cost = pd.Series(0.0, index=dates)
        for i, T in enumerate(rl):
            T_next = rl[i + 1] if i + 1 < len(rl) else dates[-1]
            picked = sel[T]
            if not picked:
                continue
            seg = ret.loc[(ret.index > T) & (ret.index <= T_next), picked]
            if seg.empty:
                continue
            port.loc[seg.index] = seg.mean(axis=1).values
            prev = sel[rl[i - 1]] if i > 0 else []
            cost.loc[seg.index[0]] += len(set(picked) - set(prev)) / TOP_K_CB * COST_CB_MAIN / 1e4
        port = port - cost
        m = metrics(port[t0:t1])
        b = metrics(bench[t0:t1])
        res[w] = {"G3": m, "等权": b, "vs": round(m["total"] - b["total"], 4)}

    wins = sum(1 for v in res.values() if v["G3"]["total"] >= v["等权"]["total"])
    avg = float(np.mean([v["vs"] for v in res.values()]))
    dd_ok = all(v["G3"]["maxdd"] >= -0.30 for v in res.values())
    return {"G3_双低强赎回避": {"windows": res, "wins": f"{wins}/8", "avg_vs": round(avg, 4),
                             "dd_ok": dd_ok,
                             "PASS": bool(wins >= 6 and avg > 0 and dd_ok)}}


def main() -> None:
    result = {"preregistration": "reports/calendar_lhb_preregistration.md"}
    r1, picks = run_g1()
    result.update(r1)
    picks.to_csv(ROOT / "reports" / "g1_lhb_trades.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(r1, ensure_ascii=False)[:400], flush=True)
    r2 = run_g2()
    result.update(r2)
    print(json.dumps(r2, ensure_ascii=False)[:400], flush=True)
    r3 = run_g3()
    result.update(r3)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "windows"}
                      for k, v in r3.items()}, ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
