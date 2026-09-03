"""研发强度截面因子 — 预注册 reports/agent_loop/prereg_rd_intensity.md.

因子: 研发费用TTM/营业收入TTM, 半年调仓 (5月/9月首个交易日), top5, vs 全主板等权, 65bp。
输出: reports/agent_loop/rd_intensity_result.json
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
from run_dca_vs_lumpsum import WINDOWS  # noqa: E402

COST = 65.0 / 1e4
TOP_K = 5
SHARD_DIR = ROOT / "replay_data" / "rd_expense"

# 报告期 → 生效日 (披露滞后写死: 年报次年4-30, 其余按季)
def effective_date(report_date: pd.Timestamp) -> pd.Timestamp:
    q = report_date.quarter
    y = report_date.year
    if q == 4:
        return pd.Timestamp(year=y + 1, month=4, day=30)
    if q == 1:
        return pd.Timestamp(year=y, month=4, day=30)
    if q == 2:
        return pd.Timestamp(year=y, month=8, day=31)
    return pd.Timestamp(year=y, month=10, day=31)  # Q3


def build_factor() -> pd.DataFrame:
    """date(生效日) × code 的 TTM 研发强度面板 (长表)。"""
    frames = []
    for fp in sorted(SHARD_DIR.glob("*.parquet")):
        if fp.stat().st_size == 0:
            continue
        d = pd.read_parquet(fp)
        if d.empty or len(d) < 4:
            continue
        if not {"报告期", "研发费用", "营业总收入"}.issubset(d.columns):
            continue  # 分片缺关键列 (抓取时源数据无此科目), 跳过不造数
        d = d.sort_values("报告期")
        d["报告期"] = pd.to_datetime(d["报告期"])
        d = d.drop_duplicates(subset=["报告期"], keep="last")
        # 去累计: Q1=累计; 其他=累计−同年前一期
        d["year"] = d["报告期"].dt.year
        for col in ("研发费用", "营业总收入"):
            prev = d.groupby("year")[col].shift(1)
            q = d["报告期"].dt.quarter
            d[f"q_{col}"] = np.where(q == 1, d[col], d[col] - prev)
        d = d.dropna(subset=["q_研发费用", "q_营业总收入"])
        # TTM = 最近4个单期之和 (滚动窗口须连续)
        q1 = d["q_研发费用"].rolling(4).sum()
        q2 = d["q_营业总收入"].rolling(4).sum()
        ttm = (q1 / q2).where(q2 > 0)
        d = d.assign(rd=ttm).dropna(subset=["rd"])
        if d.empty:
            continue
        d["eff"] = d["报告期"].map(effective_date)
        # 同一生效日只留最新报告期
        d = d.sort_values("报告期").drop_duplicates(subset=["eff"], keep="last")
        frames.append(pd.DataFrame({"date": d["eff"], "code": fp.stem, "rd": d["rd"].values}))
    if not frames:
        raise RuntimeError("rd_expense 分片库无有效数据")
    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values("date").drop_duplicates(subset=["date", "code"], keep="last")
    return long


def main() -> int:
    fac_long = build_factor()
    print(f"factor rows: {len(fac_long)}, codes: {fac_long['code'].nunique()}, "
          f"span: {fac_long['date'].min().date()} → {fac_long['date'].max().date()}", flush=True)

    out = {}
    for wname, fp in WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {fp}")
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret = close.pct_change(fill_method=None)
        codes = {s[3:]: s for s in close.columns}

        # 调仓日: 5月/9月首个交易日
        s = pd.Series(close.index)
        reb = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[0])
        reb = [t for t in reb if t.month in (5, 9)]

        sel = {}
        for T in reb:
            avail = fac_long[(fac_long["date"] <= T)]
            if avail.empty:
                continue
            latest_date = avail["date"].max()
            snap = avail[avail["date"] == latest_date].set_index("code")["rd"]
            snap = snap.rename(index=codes).dropna()
            snap = snap[snap.index.isin(close.columns)]
            if len(snap) < TOP_K * 5:
                continue
            sel[T] = snap.nlargest(TOP_K).index.tolist()
        valid = [T for T in reb if T in sel]
        if len(valid) < 2:
            out[wname] = {"n_valid_rebalances": len(valid)}
            print(f"{wname}: 有效调仓 {len(valid)} <2, 不计入 gate", flush=True)
            continue
        excess_days = []
        for i, T in enumerate(valid):
            T_next = valid[i + 1] if i + 1 < len(valid) else close.index[-1]
            idx = ret.index[(ret.index > T) & (ret.index <= T_next)]
            if idx.empty:
                continue
            port = ret.loc[idx, sel[T]].mean(axis=1)
            counts = ret.notna().sum(axis=1)
            uni = ret.mean(axis=1).where(counts > 50).reindex(idx)
            prev = sel[valid[i - 1]] if i > 0 else []
            replaced = len(set(sel[T]) - set(prev)) / TOP_K
            port = port - (replaced * COST)
            excess_days.append(port - uni)
        ex = pd.concat(excess_days).dropna()
        total_ex = float((1 + ex).prod() - 1)
        out[wname] = {"n_valid_rebalances": len(valid), "total_excess": round(total_ex, 4)}
        print(f"{wname}: 调仓 {len(valid)}, 累计超额 {total_ex:+.1%}", flush=True)

    act = {k: v for k, v in out.items() if v.get("n_valid_rebalances", 0) >= 2}
    wins = sum(1 for v in act.values() if v.get("total_excess", 0) >= 0)
    avg = float(np.mean([v["total_excess"] for v in act.values()])) if act else float("nan")
    summary = {"valid_windows": len(act), "excess_wins": f"{wins}/{len(act)}",
               "avg_excess": round(avg, 4),
               "PASS": bool(len(act) >= 5 and wins >= 5 and avg > 0)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp_out = ROOT / "reports" / "agent_loop" / "rd_intensity_result.json"
    fp_out.write_text(json.dumps({"windows": out, "summary": summary},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
