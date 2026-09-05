"""iter33 IVOL 替换 std20 A/B/C — 预注册 reports/agent_loop/prereg_ivol_swap.md.

A = turn+std20+mkt (定型口径); B = turn+ivol60+mkt (主检验); C = 四因子 (仅记录)。
其余逐字一致 (全主板池内排名, buffer keep-zone=10, top5, 月末收盘, 65bp)。
Gate (只对 B): avg差 ≥ +0.2pp 且 ≥6/11 且 dd差 ≥ −1pp。

输出: reports/agent_loop/ivol_swap_result.json
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
from run_rotation_vs_hold import TOP_K, KEEP_ZONE, COST  # noqa: E402

IVOL_WIN = 60       # 预注册冻结: 60 交易日市场模型滚动窗
IVOL_MIN = 40       # 预注册冻结: min_periods

ARMS = {"A": ("turn", "std", "mkt"), "B": ("turn", "ivol", "mkt"),
        "C": ("turn", "std", "ivol", "mkt")}


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def market_model_ivol(df: pd.DataFrame, idx: pd.DataFrame) -> pd.Series:
    """市场模型残差波动: ivol = sqrt(var_i − β̂²·var_m), 滚动 60d min 40。

    β̂ = cov(r_i, r_m)/var_m (滚动窗内), 等价于对窗内 OLS 回归残差取 std (无截距项,
    截距不影响残差 std)。按 (symbol, date) 对齐市场收益, 只用历史数据 (rolling 无前视)。
    """
    m = idx.set_index("date")["ret"].rename("mret")
    out = df[["date", "symbol", "ret"]].copy()
    out["symbol"] = out["symbol"].astype(str)
    out = out.merge(m, left_on="date", right_index=True, how="left")
    g = out.groupby("symbol", sort=False)
    var_i = g["ret"].transform(lambda s: s.rolling(IVOL_WIN, min_periods=IVOL_MIN).var())
    var_m = g["mret"].transform(lambda s: s.rolling(IVOL_WIN, min_periods=IVOL_MIN).var())
    cov = g.apply(lambda t: t["ret"].rolling(IVOL_WIN, min_periods=IVOL_MIN).cov(t["mret"]))
    cov = cov.reset_index(level=0, drop=True) if cov.index.nlevels > 1 else cov
    beta = cov / var_m.replace(0, np.nan)
    resid_var = (var_i - beta**2 * var_m).clip(lower=0.0)
    return np.sqrt(resid_var)


def score_factors(df: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """因子值: turn / std20 / ivol60 / log_mkt (点内)。"""
    out = df[["date", "symbol", "turn"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std"] = s.reindex(df.index)
    out["ivol"] = market_model_ivol(df, idx).reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["mkt"] = np.log1p(fmkt)
    return out


def score_variant(df: pd.DataFrame, idx: pd.DataFrame, factors: tuple[str, ...]) -> pd.Series:
    f = score_factors(df, idx)
    parts = [f[name].groupby(df["date"]).rank(pct=True) for name in factors]
    return -sum(parts)


def buffer_selections(df: pd.DataFrame, score: pd.Series, rl: list) -> dict:
    d = df.copy()
    d["score"] = score.values
    day_score = {T: g.dropna(subset=["score"]).set_index("symbol")["score"]
                 for T, g in d.groupby("date")}
    held: list = []
    sel: dict = {}
    for T in rl:
        row = day_score.get(T, pd.Series(dtype=float))
        if len(row) < 50:
            sel[T] = set(held)
            continue
        ranked = row.sort_values(ascending=False)
        top5 = ranked.head(TOP_K).index.tolist()
        keepzone = ranked.head(KEEP_ZONE).index.tolist()
        new = [c for c in held if c in keepzone]
        for c in top5:
            if len(new) >= TOP_K:
                break
            if c not in new:
                new.append(c)
        sel[T] = set(new)
        held = new
    return sel


def run_arm(ret_cc: pd.DataFrame, rl: list, sel: dict) -> pd.Series:
    dates = ret_cc.index
    pos = {d: i for i, d in enumerate(dates)}
    daily = []
    for i, T in enumerate(rl):
        T_next = rl[i + 1] if i + 1 < len(rl) else dates[-1]
        if T == dates[-1]:
            break
        t_idx, tn_idx = pos[T], pos[T_next]
        if t_idx + 1 > tn_idx:
            continue
        seg_dates = dates[t_idx + 1: tn_idx + 1]
        if len(seg_dates) == 0:
            continue
        sel_new = sorted(sel[T])
        seg = ret_cc.loc[seg_dates, sel_new].mean(axis=1)
        if i > 0 and rl[i - 1] in sel:
            repl = len(sel[T] - sel[rl[i - 1]]) / TOP_K
            if repl:
                seg = seg.copy()
                seg.iloc[0] -= repl * COST
        daily.append(seg)
    if not daily:
        return pd.Series(dtype=float)
    s = pd.concat(daily)
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def main() -> int:
    _force_utf8()
    idx = H.load_index()
    if idx.empty:
        raise RuntimeError("指数序列为空: replay_data/index_series.parquet")
    out: dict = {}
    for window in H.WINDOWS:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        dates = sorted(df["date"].unique())
        s = pd.Series(pd.DatetimeIndex(dates))
        rl = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
        ret_cc = df.pivot_table(index="date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
        out[window] = {}
        for arm, factors in ARMS.items():
            score = score_variant(df, idx, factors)
            sel = buffer_selections(df, score, rl)
            dr = run_arm(ret_cc, rl, sel)
            if dr.empty:
                raise RuntimeError(f"{arm} 臂空收益: {window} (零模拟, 不兜底)")
            m = H.compute_metrics(dr)
            out[window][arm] = {"total": round(m["total"], 4), "maxdd": round(m["maxdd"], 4),
                                "sel": sel}
        for arm in ("B", "C"):
            ovs = [len(out[window]["A"]["sel"].get(T, set()) & out[window][arm]["sel"].get(T, set())) / 5
                   for T in rl if out[window]["A"]["sel"].get(T)]
            out[window][f"{arm}_overlap"] = round(float(np.mean(ovs)), 3) if ovs else None
        for arm in ("B", "C"):
            out[window][f"{arm}_minus_A"] = round(out[window][arm]["total"] - out[window]["A"]["total"], 4)
        print(f"{window:<12} A={out[window]['A']['total']*100:>+8.1f}% "
              + " ".join(f"{arm}={out[window][arm]['total']*100:>+8.1f}% ({out[window][f'{arm}_minus_A']*100:>+7.2f}, ovl={out[window][f'{arm}_overlap']})"
                         for arm in ("B", "C"))
              + f" ddA={out[window]['A']['maxdd']*100:>6.1f}%")
        for arm in ("A", "B", "C"):
            del out[window][arm]["sel"]

    ddA = np.mean([out[w]["A"]["maxdd"] for w in H.WINDOWS])
    results = {}
    for arm in ("B", "C"):
        diffs = [out[w][f"{arm}_minus_A"] for w in H.WINDOWS]
        wins = sum(1 for x in diffs if x >= 0)
        results[arm] = {"wins": f"{wins}/11", "avg_diff_pp": round(float(np.mean(diffs)) * 100, 2)}
    ddB = np.mean([out[w]["B"]["maxdd"] for w in H.WINDOWS])
    dd_ok = bool((ddB - ddA) >= -0.01)
    gate_pass = bool(int(results["B"]["wins"].split("/")[0]) >= 6
                     and results["B"]["avg_diff_pp"] >= 0.2 and dd_ok)
    summary = {
        "gate": {"arm": "B", **results["B"],
                 "dd_B_avg": round(float(ddB), 4), "dd_A_avg": round(float(ddA), 4),
                 "dd_ok": dd_ok,
                 "required": "avg>=+0.2pp & wins>=6/11 & dd diff>=-1pp",
                 "PASS": gate_pass},
        "all_arms": results,
        "C_note": "仅记录, 不设 gate (预注册防多重比较)",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "ivol_swap_result.json"
    fp.write_text(json.dumps({"prereg": "prereg_ivol_swap.md",
                              "windows": _jsonable(out), "summary": summary},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


if __name__ == "__main__":
    raise SystemExit(main())
