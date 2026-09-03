"""外部实践者策略机械复现三臂 — 预注册 reports/agent_loop/prereg_external_strategies.md.

GHT (国泰海通低波骨架, 季频100股) / IVW (低波逆向加权, 月频全池) / COLDH (散户冷门机械版, 月频20股)。
65bp, 11窗, vs 主板等权; 对比在持 buffer 臂 (A)。
输出: reports/agent_loop/external_strategies_result.json
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
from run_rotation_vs_hold import _factors, month_ends, run_arm  # noqa: E402

COST = 65.0 / 1e4
QUARTER_MONTHS = (4, 8, 10, 12)


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def run_weighted(df: pd.DataFrame, f_all: pd.DataFrame, rebal: list, weight_fn) -> tuple[pd.Series, dict]:
    """通用加权组合引擎: T 收盘定权重, 赚 (T, T_next] 日收益; 首日扣 单边换手×COST。
    f_all = _factors(df) 整窗预计算 (rolling 须全历史), weight_fn 收单日切片。"""
    d = df.copy()
    dates = sorted(d["date"].unique())
    rl = [T for T in rebal if T in set(d["date"])]
    close = d.pivot_table(index="date", columns="symbol", values="close").sort_index()
    ret = close.pct_change(fill_method=None)
    day_x = {T: g for T, g in f_all.groupby("date")}
    daily = []
    prev_w: pd.Series = pd.Series(dtype=float)
    n_rebal = 0
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else close.index[-1]
        w = weight_fn(day_x.get(T))
        if w is None or len(w) == 0:
            w = prev_w
        if len(w) == 0:
            continue
        n_rebal += 1
        idx = ret.index[(ret.index > T) & (ret.index <= T_next)]
        if idx.empty:
            prev_w = w
            continue
        union = w.index.union(prev_w.index)
        turn = 0.5 * float((w.reindex(union, fill_value=0.0) - prev_w.reindex(union, fill_value=0.0)).abs().sum())
        port = ret.loc[idx, w.index].mul(w, axis=1).sum(axis=1, min_count=1)
        if turn > 0 and i > 0:
            port = port.copy()
            port.iloc[0] -= turn * COST
        daily.append(port)
        prev_w = w
    if not daily:
        return pd.Series(dtype=float), {"n_rebalance": 0}
    return pd.concat(daily).dropna(), {"n_rebalance": n_rebal}


def weight_ght(x: pd.DataFrame) -> pd.Series:
    f = x
    r_mkt = f["log_mkt"].rank(pct=True)
    r_amt = f["amount"].rank(pct=True)
    pool = f[(r_mkt >= 0.20) & (r_amt >= 0.20)].dropna(subset=["std20"])
    pick = pool.nsmallest(100, "std20")
    return pd.Series(1.0 / max(len(pick), 1), index=pick["sym"].values)


def weight_ivw(x: pd.DataFrame) -> pd.Series:
    f = x
    r_std = f["std20"].rank(pct=True)
    pool = f[r_std <= 0.20].dropna(subset=["std20"])
    pool = pool[pool["std20"] > 0]
    if pool.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / pool["std20"]
    s = inv / inv.sum()
    s.index = pool["sym"].values
    return s


def weight_coldh(x: pd.DataFrame) -> pd.Series:
    f = x
    pool = f[(f["turn"] < 1.0) & (f["amount"] < 0.5 * f["ma20a"])].dropna(subset=["std20"])
    pick = pool.nsmallest(20, "std20")
    return pd.Series(1.0 / max(len(pick), 1), index=pick["sym"].values)


def main() -> int:
    _force_utf8()
    out = {}
    print(f"{'window':<12} {'arm':<7} {'total':>8} {'vsU':>8} {'maxdd':>8} {'nreb':>5}")
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        uni = H.window_universe_benchmark(df)
        f_all = _factors(df)
        f_all["date"] = df["date"].values
        f_all["sym"] = df["symbol"].values
        f_all["ma20a"] = df.groupby("symbol")["amount"].rolling(20).mean().reset_index(level=0, drop=True).reindex(df.index)
        out[window] = {}
        # A 臂 (在持 buffer, 冷落低波 top5)
        from run_rotation_vs_hold import _score as cold_score
        dr_a, _ = run_arm(df, cold_score(df), "buffer")
        all_rebal = month_ends(sorted(df["date"].unique()))
        q_rebal = [T for T in all_rebal if T.month in QUARTER_MONTHS]
        arms = {
            "GHT": (run_weighted(df, f_all, q_rebal, weight_ght), q_rebal),
            "IVW": (run_weighted(df, f_all, all_rebal, weight_ivw), all_rebal),
            "COLDH": (run_weighted(df, f_all, all_rebal, weight_coldh), all_rebal),
            "A": ((dr_a, {}), None),
        }
        for name, ((dr, info), _) in arms.items():
            if dr.empty:
                out[window][name] = {"total": None}
                print(f"{window:<12} {name:<7} {'NA':>8}")
                continue
            m = H.compute_metrics(dr)
            t0, t1 = dr.index.min(), dr.index.max()
            uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
            uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
            vsu = round(m["total"] - uni_total, 4)
            out[window][name] = {"total": round(m["total"], 4), "vs_universe": vsu,
                                 "maxdd": round(m["maxdd"], 4), **info}
            print(f"{window:<12} {name:<7} {m['total']*100:>+7.1f}% {vsu*100:>+7.1f}% "
                  f"{m['maxdd']*100:>7.1f}% {info.get('n_rebalance', 0):>5}")

    def agg(arm, key):
        vals = [out[w][arm].get(key) for w in H.WINDOWS
                if isinstance(out[w].get(arm), dict) and out[w][arm].get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {}
    for arm in ("GHT", "IVW", "COLDH"):
        diffs = [out[w][arm]["vs_universe"] - out[w]["A"]["vs_universe"]
                 for w in H.WINDOWS
                 if isinstance(out[w].get(arm), dict) and out[w][arm].get("vs_universe") is not None]
        wins = sum(1 for x in diffs if x >= 0)
        avg_diff = float(np.mean(diffs))
        dd_arm, dd_a = agg(arm, "maxdd"), agg("A", "maxdd")
        gate = bool(avg_diff >= 0.002 and wins >= 6 and dd_arm >= dd_a)
        summary[arm] = {"avg_vs_universe": agg(arm, "vs_universe"),
                        "vsu_pos": f"{sum(1 for w in H.WINDOWS if isinstance(out[w].get(arm), dict) and (out[w][arm].get('vs_universe') or -1) >= 0)}/11",
                        "avg_maxdd": dd_arm, "avg_diff_vs_A": round(avg_diff, 4),
                        "diff_wins": f"{wins}/{len(diffs)}",
                        "gate": "PASS" if gate else "FAIL",
                        "rule": "avg diff≥+0.2pp 且 ≥6/11 且 dd不劣 才值得替换在持A"}
        print(f"\n[{arm}] {json.dumps(summary[arm], ensure_ascii=False)}")
    fp = ROOT / "reports" / "agent_loop" / "external_strategies_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary,
                              "note": "三臂对1万不可执行 (整手门槛), 因子证据用途"},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
