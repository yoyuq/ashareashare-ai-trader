"""ETF 动量轮动定论测试 (预注册: reports/cb_enhance_preregistration.md F1).

池 (固定10只, 不挑后验赢家): 510300/510500/512010/512690/515030/512480/512800/512660/512000/511010
规则: 每月第一个交易日, 池内(国债除外) 60日动量 top1; 最强动量<=0 → 511010。切换成本 20bp。
判据: >=6/8 窗口 total >= 池等权(月再平衡) 且平均超额>0。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "etf_momentum_result.json"

POOL = ["sh510300", "sh510500", "sh512010", "sh512690", "sh515030",
        "sh512480", "sh512800", "sh512660", "sh512000"]
SAFE = "sh511010"
MOM = 60
COST = 20.0  # bp per switch (round trip)

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


def fetch_close() -> pd.DataFrame:
    cols = {}
    import akshare as ak
    for sym in POOL + [SAFE]:
        cache = ROOT / "replay_data" / f"etf_{sym}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            df = ak.fund_etf_hist_sina(symbol=sym)
            df["date"] = pd.to_datetime(df["date"])
            df.to_parquet(cache)
        s = df.set_index("date")["close"].sort_index()
        cols[sym] = s
    return pd.DataFrame(cols)


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"total": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    total = float((1.0 + r).prod() - 1.0)
    std = float(r.std())
    sharpe = float(r.mean() / std * np.sqrt(252.0)) if std > 1e-12 else np.nan
    eq = (1.0 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return {"total": round(total, 4), "sharpe": round(sharpe, 3), "maxdd": round(maxdd, 4)}


def main() -> None:
    px = fetch_close()
    ret = px.pct_change(fill_method=None)
    mom = px.pct_change(MOM, fill_method=None)

    dates = px.index
    # 每月第一个交易日调仓
    s = pd.Series(dates)
    rebal = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[0]).tolist()
    # 信号用 T-1 收盘动量 (T 开盘执行, 简化为 T 收盘计收益, 成本在切换日记)
    hold = {}
    for T in rebal:
        i = dates.get_loc(T)
        if i == 0:
            hold[T] = SAFE
            continue
        m = mom.iloc[i - 1][POOL].dropna()
        hold[T] = m.idxmax() if (not m.empty and m.max() > 0) else SAFE

    port = pd.Series(np.nan, index=dates)
    cost = pd.Series(0.0, index=dates)
    rl = sorted(rebal)
    for j, T in enumerate(rl):
        i = dates.get_loc(T)
        T_next = rl[j + 1] if j + 1 < len(rl) else dates[-1]
        cur = hold[T]
        # T 收盘→次日收益: 持有 cur 从 T 收盘起
        seg = ret[cur].loc[(ret.index > T) & (ret.index <= T_next)]
        port.loc[seg.index] = seg.values
        prev = hold[rl[j - 1]] if j > 0 else None
        if prev is not None and prev != cur:
            cost.loc[seg.index[0]] = COST / 1e4
    port = port - cost.reindex(port.index, fill_value=0.0)

    # 基准: 池等权 (月再平衡)
    ew = ret[POOL].mean(axis=1)

    result = {"preregistration": "reports/cb_enhance_preregistration.md",
              "pool": POOL + [SAFE], "mom_days": MOM, "windows": {}}
    for wname, (t0, t1) in WINDOWS.items():
        m = metrics(port[t0:t1])
        e = metrics(ew[t0:t1])
        m["vs_池等权"] = round(m["total"] - e["total"], 4)
        result["windows"][wname] = {"F1_动量轮动": m, "benchmark_池等权": e}
        print(f"{wname}: F1={m['total']:+.4f} 池等权={e['total']:+.4f} 超额={m['vs_池等权']:+.4f}", flush=True)

    wins = sum(1 for w in result["windows"].values()
               if w["F1_动量轮动"]["total"] >= w["benchmark_池等权"]["total"])
    avg = float(np.mean([w["F1_动量轮动"]["vs_池等权"] for w in result["windows"].values()]))
    result["summary"] = {"wins_vs_池等权": f"{wins}/8", "avg_vs": round(avg, 4),
                         "PASS": bool(wins >= 6 and avg > 0)}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
