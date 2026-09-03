"""冷落低波精选执行方式比较 — 预注册 reports/agent_loop/prereg_rotation_vs_hold.md.

三臂: monthly(月频全换) / hold(冻结) / buffer(keep-zone=10 保留), 65bp, 11窗, vs 匹配 universe。
输出: reports/agent_loop/rotation_vs_hold_result.json
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
TOP_K = 5
KEEP_ZONE = 10  # buffer 臂保留区, 预注册写死


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "symbol", "turn", "amount", "close"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std20"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["log_mkt"] = np.log1p(fmkt)
    return out


def _score(df: pd.DataFrame) -> pd.Series:
    f = _factors(df)
    r_turn = f["turn"].groupby(df["date"]).rank(pct=True)
    r_std = f["std20"].groupby(df["date"]).rank(pct=True)
    r_mkt = f["log_mkt"].groupby(df["date"]).rank(pct=True)
    return -(r_turn + r_std + r_mkt)


def month_ends(dates: list) -> list:
    s = pd.Series(pd.DatetimeIndex(dates))
    return s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()


def run_arm(df: pd.DataFrame, score: pd.Series, arm: str) -> tuple[pd.Series, dict]:
    """返回日净收益序列与 info (n_rebalance / replaced 总数)。"""
    d = df.copy()
    d["score"] = score.values
    dates = sorted(d["date"].unique())
    rebal = month_ends(dates)
    rl = [T for T in rebal if T in set(d["date"])]
    close = d.pivot_table(index="date", columns="symbol", values="close").sort_index()
    ret = close.pct_change(fill_method=None)
    # 当日截面 score (月末)
    sel: dict = {}
    day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"] for T, g in d.groupby("date")}
    held: list = []
    n_repl = 0
    for i, T in enumerate(rl):
        if arm == "hold" and i > 0:
            sel[T] = set(held)
            continue
        row = day_score.get(T, pd.Series(dtype=float))
        if len(row) < 50:
            sel[T] = set(held) if held else set()
            continue
        ranked = row.sort_values(ascending=False)
        top5 = ranked.head(TOP_K).index.tolist()
        if arm == "monthly":
            new = top5
        elif arm == "buffer":
            keepzone = ranked.head(KEEP_ZONE).index.tolist()
            new = [c for c in held if c in keepzone]
            for c in top5:
                if len(new) >= TOP_K:
                    break
                if c not in new:
                    new.append(c)
        else:  # hold
            new = top5
        sel[T] = set(new)
        n_repl += len(set(new) - set(held)) if i > 0 or arm != "hold" else TOP_K
        held = new
    # 日收益
    daily = []
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else close.index[-1]
        idx = ret.index[(ret.index > T) & (ret.index <= T_next)]
        if idx.empty or not sel[T]:
            continue
        port = ret.loc[idx, sorted(sel[T])].mean(axis=1)
        if i > 0:
            replaced = len(sel[T] - sel[rl[i - 1]]) / TOP_K
            if replaced:
                port = port.copy()
                port.iloc[0] -= replaced * COST  # 成本只扣换仓日一次 (原误扣全月)
        daily.append(port)
    if not daily:
        return pd.Series(dtype=float), {"n_rebalance": 0, "n_replaced": 0}
    daily_ret = pd.concat(daily).dropna()
    return daily_ret, {"n_rebalance": len(rl), "n_replaced": n_repl}


def main() -> int:
    _force_utf8()
    idx = H.load_index()
    out = {}
    print(f"{'window':<12} {'arm':<8} {'total':>8} {'vsU':>8} {'maxdd':>8} {'repl':>5}")
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        uni = H.window_universe_benchmark(df)
        out[window] = {}
        for arm in ("monthly", "buffer", "hold"):
            dr, info = run_arm(df, score, arm)
            if dr.empty:
                out[window][arm] = {"total": None}
                print(f"{window:<12} {arm:<8} {'NA':>8}")
                continue
            m = H.compute_metrics(dr)
            t0, t1 = dr.index.min(), dr.index.max()
            uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
            uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
            idx_total = H._idx_total(idx, t0, t1)
            vsu = round(m["total"] - uni_total, 4)
            out[window][arm] = {"total": round(m["total"], 4), "vs_universe": vsu,
                                "maxdd": round(m["maxdd"], 4), "vs_index": round(m["total"] - idx_total, 4),
                                **info}
            print(f"{window:<12} {arm:<8} {m['total']*100:>+7.1f}% {vsu*100:>+7.1f}% "
                  f"{m['maxdd']*100:>7.1f}% {info['n_replaced']:>5}")

    def agg(arm, key):
        vals = [out[w][arm].get(key) for w in H.WINDOWS
                if isinstance(out[w].get(arm), dict) and out[w][arm].get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    # sanity: monthly 主判据① (非纯牛8窗 vsU>=0)
    nonniu = [w for w in H.WINDOWS if OBJECTIVE.get(w) != "纯牛"] if False else None
    summary = {}
    for arm in ("monthly", "buffer", "hold"):
        summary[arm] = {"avg_vs_universe": agg(arm, "vs_universe"),
                        "avg_maxdd": agg(arm, "maxdd"),
                        "avg_total": agg(arm, "total"),
                        "total_replaced": sum(out[w].get(arm, {}).get("n_replaced", 0) for w in H.WINDOWS)}
    mo = [out[w]["monthly"]["vs_universe"] for w in H.WINDOWS
          if out[w].get("monthly", {}).get("vs_universe") is not None
          and "牛转崩" not in w and w not in ("2019牛", "2020牛转崩", "2024震荡")]
    # 非纯牛按成本现实性 OBJECTIVE 口径 (熊/崩+震荡 = 8窗)
    OBJECTIVE = {"2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
                 "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
                 "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
                 "2024震荡": "纯牛", "2025-26现期": "震荡"}
    nonniu = [w for w in H.WINDOWS if OBJECTIVE[w] != "纯牛"]
    sanity_wins = sum(1 for w in nonniu if out[w]["monthly"]["vs_universe"] >= 0)
    summary["sanity_monthly_1"] = {"nonniu_vsU_wins": f"{sanity_wins}/{len(nonniu)}",
                                   "PASS_ge_7_of_8": bool(sanity_wins >= 7)}
    # 定型规则
    avg_m, avg_b = summary["monthly"]["avg_vs_universe"], summary["buffer"]["avg_vs_universe"]
    decision = ("buffer" if (avg_b is not None and avg_m is not None
                             and avg_m - avg_b <= 0.01
                             and summary["buffer"]["total_replaced"] < summary["monthly"]["total_replaced"])
                else "monthly")
    summary["decision"] = {"adopt": decision, "rule": "buffer落后monthly≤1.0pp且替换次数更少→buffer; 否则monthly"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "rotation_vs_hold_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
