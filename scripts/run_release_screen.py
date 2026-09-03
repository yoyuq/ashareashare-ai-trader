"""限售解禁事件研究 — screen 有效性 (预注册 reports/agent_loop/prereg_release_screen.md).

事件收益 = 解禁日 T_r−1 收盘 → T_r+20 收盘; 超额 = 个股 − 同窗全主板截面均值。
输出: reports/agent_loop/release_screen_result.json
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

HORIZON = 20


def main() -> int:
    rel = pd.read_parquet(ROOT / "replay_data" / "release.parquet")
    rel["解禁时间"] = pd.to_datetime(rel["解禁时间"])
    rel["code"] = rel["股票代码"].astype(str).str.zfill(6)
    rel = rel[rel["code"].str.startswith(("60", "00"))].copy()
    rel = rel.drop_duplicates(subset=["code", "解禁时间"])
    rel = rel[~rel["股票简称"].astype(str).str.contains("ST", na=False)]

    # 主板日线全集 (2016→2024)
    frames = []
    for y in range(2016, 2025):
        fp = ROOT / "replay_data" / f"daily_{y}-01-01_{y}-12-31.parquet"
        if not fp.exists():
            print(f"[warn] 缺 {fp.name}, 相关年份事件将缺失")
            continue
        d = pd.read_parquet(fp, columns=["date", "symbol", "close", "is_trade", "isST"])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "symbol"])
    d["date"] = pd.to_datetime(d["date"])
    d["isST"] = d["isST"].astype(str)
    d = d[(d["is_trade"] == 1) & (d["isST"] == "0") & d["symbol"].str.startswith(("sh.60", "sz.00"))]
    d = d.sort_values(["symbol", "date"])
    d["code"] = d["symbol"].str[3:]

    cal = d[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    cal["pos"] = np.arange(len(cal))
    d = d.merge(cal, on="date", how="left")
    pos_by_date = cal.set_index("date")["pos"]
    close = d.pivot_table(index="pos", columns="code", values="close").sort_index()

    # 截面 20 日收益 (pos → pos+20), 全主板均值
    fwd = close.shift(-HORIZON) / close - 1.0
    cross_mean = fwd.mean(axis=1)

    rows = []
    for _, r in rel.iterrows():
        code, tr = r["code"], r["解禁时间"]
        if code not in close.columns:
            continue
        dt_ok = pos_by_date[pos_by_date.index.normalize() == tr.normalize()]
        # 解禁日可能非交易日 → 取其后第一个交易日
        after = pos_by_date[pos_by_date.index > tr]
        if after.empty:
            continue
        p_r = int(after.iloc[0])
        p0 = p_r - 1
        p1 = p_r + HORIZON
        if p0 not in close.index or p1 not in close.index:
            continue
        c0, c1 = close.at[p0, code], close.at[p1, code]
        if not (np.isfinite(c0) and np.isfinite(c1) and c0 > 0):
            continue
        ev = c1 / c0 - 1.0
        ex = ev - cross_mean.loc[p0] if np.isfinite(cross_mean.loc[p0]) else np.nan
        rows.append({"code": code, "date": tr, "year": tr.year,
                     "ratio": float(r["占解禁前流通市值比例"]) if pd.notna(r["占解禁前流通市值比例"]) else np.nan,
                     "ev": ev, "excess": ex})
    e = pd.DataFrame(rows)
    if e.empty:
        raise RuntimeError("无有效事件样本")
    big = e[e["ratio"] >= 0.10]

    yearly = {int(y): {"n": int(len(g)), "mean_ex": round(float(g["excess"].mean()), 4),
                       "neg": bool(g["excess"].mean() < 0)}
              for y, g in e.groupby("year")}
    neg_years = sum(1 for v in yearly.values() if v["neg"])
    summary = {
        "n_events": int(len(e)),
        "mean_excess": round(float(e["excess"].mean()), 4),
        "median_excess": round(float(e["excess"].median()), 4),
        "neg_years": f"{neg_years}/{len(yearly)}",
        "big_release_n": int(len(big)),
        "big_mean_excess": round(float(big["excess"].mean()), 4) if len(big) else None,
        "PASS": bool(e["excess"].mean() < 0 and e["excess"].median() < 0
                     and neg_years >= 7 and len(yearly) >= 9),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = ROOT / "reports" / "agent_loop" / "release_screen_result.json"
    out.write_text(json.dumps({"summary": summary, "yearly": yearly},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
