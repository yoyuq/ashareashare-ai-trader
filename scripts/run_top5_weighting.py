"""冷落低波 top5 加权方式 (等权A vs 波动率倒数W1 vs 冷落度倾斜W2)
— 预注册 reports/agent_loop/prereg_top5_weighting.md.

buffer keep-zone=10, 月末, 65bp (0.5×Σ|Δw| 口径), 11窗, vs 主板等权。
输出: reports/agent_loop/top5_weighting_result.json
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
from run_rotation_vs_hold import _factors, _score, month_ends  # noqa: E402

COST = 65.0 / 1e4
TOP_K = 5
KEEP_ZONE = 10


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def run_weighted_top5(df: pd.DataFrame, score: pd.Series, weight_mode: str) -> tuple[pd.Series, dict]:
    """buffer 选券 (与 run_arm 同逻辑) + 加权日收益。weight_mode: equal|invol|tilt。"""
    d = df.copy()
    d["score"] = score.values
    f_all = _factors(df)
    f_all["sym"] = df["symbol"].values
    dates = sorted(d["date"].unique())
    rl = [T for T in month_ends(dates) if T in set(d["date"])]
    close = d.pivot_table(index="date", columns="symbol", values="close").sort_index()
    ret = close.pct_change(fill_method=None)
    day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"] for T, g in d.groupby("date")}
    day_f = {T: g for T, g in f_all.groupby("date")}

    def weights_for(T, held_syms):
        row = day_score.get(T, pd.Series(dtype=float))
        if len(row) < 50:
            return pd.Series(dtype=float)
        ranked = row.sort_values(ascending=False)
        keepzone = ranked.head(KEEP_ZONE).index.tolist()
        new = [c for c in held_syms if c in keepzone]
        for c in ranked.head(TOP_K).index:
            if len(new) >= TOP_K:
                break
            if c not in new:
                new.append(c)
        if not new:
            return pd.Series(dtype=float)
        fx = day_f.get(T)
        fx = fx[fx["sym"].isin(new)].set_index("sym") if fx is not None else None
        if weight_mode == "equal" or fx is None or len(fx) < len(new):
            return pd.Series(1.0 / len(new), index=new)
        if weight_mode == "invol":
            raw = 1.0 / fx["std20"].replace(0, np.nan)
        elif weight_mode == "tilt":
            # 5 只持仓按当日 score 相对排序赋权 5:4:3:2:1 (buffer 保留票也参与排序)
            order = sorted(new, key=lambda c: -float(ranked[c]))
            rank = {c: i for i, c in enumerate(order)}
            raw = pd.Series({c: float(TOP_K - rank[c]) for c in new})
        else:
            raise ValueError(weight_mode)
        raw = raw.dropna()
        if raw.empty or raw.sum() <= 0:
            return pd.Series(1.0 / len(new), index=new)
        return raw / raw.sum()

    daily = []
    prev_w = pd.Series(dtype=float)
    n_rebal = 0
    held: list = []
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else close.index[-1]
        w = weights_for(T, held)
        if len(w) == 0:
            continue
        n_rebal += 1
        held = w.index.tolist()
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


def main() -> int:
    _force_utf8()
    out = {}
    print(f"{'window':<12} {'arm':<6} {'total':>8} {'vsU':>8} {'maxdd':>8} {'nreb':>5}")
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        uni = H.window_universe_benchmark(df)
        out[window] = {}
        for name, mode in (("A", "equal"), ("W1", "invol"), ("W2", "tilt")):
            dr, info = run_weighted_top5(df, score, mode)
            if dr.empty:
                out[window][name] = {"total": None}
                print(f"{window:<12} {name:<6} {'NA':>8}")
                continue
            m = H.compute_metrics(dr)
            t0, t1 = dr.index.min(), dr.index.max()
            uni_seg = uni[(uni.index >= t0) & (uni.index <= t1)].dropna()
            uni_total = float((1.0 + uni_seg).prod() - 1.0) if len(uni_seg) else np.nan
            vsu = round(m["total"] - uni_total, 4)
            out[window][name] = {"total": round(m["total"], 4), "vs_universe": vsu,
                                 "maxdd": round(m["maxdd"], 4), **info}
            print(f"{window:<12} {name:<6} {m['total']*100:>+7.1f}% {vsu*100:>+7.1f}% "
                  f"{m['maxdd']*100:>7.1f}% {info['n_rebalance']:>5}")

    w10 = [w for w in H.WINDOWS if w != "2015牛转股灾"]
    summary = {}
    for arm in ("W1", "W2"):
        diffs = [out[w][arm]["vs_universe"] - out[w]["A"]["vs_universe"] for w in w10]
        wins = sum(1 for x in diffs if x >= 0)
        avg_diff = float(np.mean(diffs))
        dd_a = float(np.mean([out[w]["A"]["maxdd"] for w in w10]))
        dd_w = float(np.mean([out[w][arm]["maxdd"] for w in w10]))
        gate = bool(avg_diff >= 0.002 and wins >= 6 and dd_w >= dd_a)
        summary[arm] = {"avg_diff_vs_A": round(avg_diff, 4), "wins": f"{wins}/{len(w10)}",
                        "avg_dd_A": round(dd_a, 4), "avg_dd_arm": round(dd_w, 4),
                        "gate": "PASS" if gate else "FAIL",
                        "rule": "avg差≥+0.2pp 且 ≥6/10 且 dd不劣"}
        print(f"\n[{arm}] {json.dumps(summary[arm], ensure_ascii=False)}")
    fp = ROOT / "reports" / "agent_loop" / "top5_weighting_result.json"
    fp.write_text(json.dumps({"windows": out, "summary": summary}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
