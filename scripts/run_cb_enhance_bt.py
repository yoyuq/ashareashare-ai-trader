"""双低增强臂 E1/E2/E3/E4 回测 (预注册: reports/cb_enhance_preregistration.md).

复用 cb_daily/cb_value 面板与 A 臂口径; 新增过滤池: 偏债(转股价值<100) / 正股动量(转股价值
20日动量>0) / 小盘(发行规模最小1/3); E4 = 高活跃隔夜 (T-1收盘买, T开盘卖)。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "cb_enhance_result.json"

COST_MAIN = 20.0
COST_LOW = 5.0
TOP_K = 10
TOP_K_OVERNIGHT = 20
MOM_WINDOW = 20
SIZE_BOTTOM_FRAC = 1 / 3.0

WINDOWS = {
    "2018熊":         ("2018-01-01", "2018-12-31"),
    "2019牛":         ("2019-01-01", "2019-12-31"),
    "2020牛转崩":     ("2020-06-01", "2021-02-28"),
    "2021白马转小盘": ("2021-01-01", "2021-12-31"),
    "2022熊":         ("2022-01-01", "2022-12-31"),
    "2023震荡":       ("2023-01-01", "2023-12-31"),
    "2024震荡":       ("2024-01-01", "2024-12-31"),
    "2025-26现期":    ("2025-10-08", "2026-07-31"),
}


def load_panel() -> dict:
    daily = pd.read_parquet(ROOT / "replay_data" / "cb_daily.parquet")
    value = pd.read_parquet(ROOT / "replay_data" / "cb_value.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    value["date"] = pd.to_datetime(value["日期"])
    value["symbol"] = value["symbol"].astype(str)

    def wide(df, col):
        return df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").sort_index()

    close, open_, vol = wide(daily, "close"), wide(daily, "open"), wide(daily, "volume")
    prem, cv = wide(value, "转股溢价率"), wide(value, "转股价值")
    cols = close.columns.intersection(prem.columns)
    close, open_, vol, prem, cv = (w[cols] for w in (close, open_, vol, prem, cv))
    valid = close.notna() & prem.notna()
    close, open_, vol, prem, cv = (w.where(valid) for w in (close, open_, vol, prem, cv))
    ret = close.pct_change(fill_method=None)
    amt = vol * (open_ + close) / 2.0
    size_fp = ROOT / "replay_data" / "cb_size.parquet"
    size = (pd.read_parquet(size_fp).set_index("symbol")["size_yi"]
            if size_fp.exists() else None)  # symbol -> 发行规模(亿)
    return {"close": close, "open": open_, "prem": prem, "cv": cv, "ret": ret, "amt": amt, "size": size}


def rebalance_dates(dates: pd.DatetimeIndex, freq: str = "weekly") -> list:
    s = pd.Series(dates)
    if freq == "weekly":
        grp = [s.dt.isocalendar().year.astype(int), s.dt.isocalendar().week.astype(int)]
    else:
        grp = [s.dt.year, s.dt.month]
    return s.groupby(grp).apply(lambda x: x.iloc[-1]).tolist()


def run_filtered_double_low(panel: dict, pool_mask: pd.DataFrame, cost_bps: float,
                            top_k: int = TOP_K, freq: str = "weekly") -> pd.Series:
    close, prem, ret = panel["close"], panel["prem"], panel["ret"]
    score = -(close + prem * 100.0)
    score = score.where(pool_mask)
    dates = close.index
    rl = rebalance_dates(dates, freq)
    sel = {}
    for T in rl:
        row = score.loc[T].dropna()
        sel[T] = row.nlargest(top_k).index.tolist() if not row.empty else []
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
        cost.loc[seg.index[0]] += len(set(picked) - set(prev)) / top_k * cost_bps / 1e4
    return port - cost


def run_overnight(panel: dict, cost_bps: float, top_k: int = TOP_K_OVERNIGHT) -> pd.Series:
    """E4: T-1 收盘成交额代理 top20, T-1 收盘买 → T 开盘卖. 收益 = open/prev_close - 1."""
    open_, close, amt = panel["open"], panel["close"], panel["amt"]
    dates = close.index
    overnight = open_ / close.shift(1) - 1.0
    port = pd.Series(np.nan, index=dates)
    for i in range(1, len(dates)):
        T_prev, T = dates[i - 1], dates[i]
        row = amt.loc[T_prev].dropna()
        if row.empty:
            continue
        picked = row.nlargest(top_k).index.tolist()
        r = overnight.loc[T, picked].mean()
        port.loc[T] = r - 2 * cost_bps / 1e4
    return port


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"total": np.nan, "sharpe": np.nan, "maxdd": np.nan, "n_days": 0}
    total = float((1.0 + r).prod() - 1.0)
    n = len(r)
    std = float(r.std())
    sharpe = float(r.mean() / std * np.sqrt(252.0)) if std > 1e-12 else np.nan
    eq = (1.0 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"total": round(total, 4), "sharpe": round(sharpe, 3), "maxdd": round(maxdd, 4), "n_days": n}


def main() -> None:
    panel = load_panel()
    close, cv, amt = panel["close"], panel["cv"], panel["amt"]
    bench_ew = panel["ret"].mean(axis=1).where(panel["ret"].notna().sum(axis=1) >= 3)

    # 池定义 (预注册)
    pools = {"E1": cv < 100.0}
    cv_mom = cv.pct_change(MOM_WINDOW, fill_method=None)
    pools["E2"] = cv_mom > 0
    size = panel["size"]
    if size is not None:
        common = close.columns.intersection(size.index.astype(str))
        thr = size.loc[common].quantile(SIZE_BOTTOM_FRAC)
        pools["E3"] = pd.DataFrame(False, index=close.index, columns=close.columns)
        small = [c for c in close.columns if c in set(size.loc[common][size.loc[common] <= thr].index)]
        pools["E3"][small] = True
    else:
        pools["E3"] = None

    arms = {
        "E1_偏债双低周": ("pool", "E1"),
        "E2_动量双低周": ("pool", "E2"),
        "E3_小盘双低周": ("pool", "E3"),
        "E4_高活跃隔夜": ("overnight", None),
    }

    result = {"preregistration": "reports/cb_enhance_preregistration.md", "windows": {}}
    for wname, (t0, t1) in WINDOWS.items():
        win = {}
        seg_ew = bench_ew[(bench_ew.index >= t0) & (bench_ew.index <= t1)]
        ew_total = float((1.0 + seg_ew.dropna()).prod() - 1.0)
        win["benchmark_转债等权"] = {"total": round(ew_total, 4)}
        for aname, (kind, key) in arms.items():
            for tag, cost in (("main20", COST_MAIN), ("low5", COST_LOW)):
                if kind == "pool":
                    if pools[key] is None:
                        m = {"total": np.nan}
                    else:
                        m = metrics(run_filtered_double_low(panel, pools[key], cost)
                                    .loc[t0:t1])
                else:
                    r = run_overnight(panel, cost)
                    m = metrics(r.loc[t0:t1])
                    m["daily_pos_rate"] = round(float((r.loc[t0:t1] > 0).mean()), 4)
                m["vs_等权"] = round(m["total"] - ew_total, 4) if not np.isnan(m["total"]) else np.nan
                win[f"{aname}_{tag}"] = m
        result["windows"][wname] = win
        print(f"{wname}: E1={win['E1_偏债双低周_main20']['total']}, E2={win['E2_动量双低周_main20']['total']}, "
              f"E3={win['E3_小盘双低周_main20']['total']}, E4={win['E4_高活跃隔夜_main20']['total']}, "
              f"等权={ew_total:+.4f}", flush=True)

    summary = {}
    for aname, (kind, _) in arms.items():
        for tag in ("main20", "low5"):
            k = f"{aname}_{tag}"
            wins = sum(1 for w in result["windows"].values()
                       if w[k]["total"] >= w["benchmark_转债等权"]["total"])
            avg_vs = float(np.mean([w[k]["vs_等权"] for w in result["windows"].values()]))
            dd_ok = all(w[k]["maxdd"] >= -0.30 for w in result["windows"].values())
            s = {"wins_vs_等权": f"{wins}/8", "avg_vs_等权": round(avg_vs, 4), "maxdd_ok": dd_ok}
            if kind == "overnight":
                pos_w = sum(1 for w in result["windows"].values() if w[k]["daily_pos_rate"] > 0)
                s["daily_pos_windows"] = f"{pos_w}/8"
                if tag == "main20":
                    s["PASS"] = bool(pos_w >= 6 and avg_vs > 0)
            elif tag == "main20":
                s["PASS"] = bool(wins >= 6 and avg_vs > 0 and dd_ok)
            summary[k] = s
    result["summary"] = summary
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
