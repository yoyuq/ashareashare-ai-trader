"""交易策略研究回测 — 网络搜集策略的忠实映射 + 回测 (零模拟, 真实日线).

四类时长: 超短线(T+1~3)/短线(3~10日)/中线(2~8周)/长线(月~年).
每策略: fn(df)->score (排名, 越大越优先) 或 entry/exit bool (信号持仓) 或 signal bool (固定持有).
判据见 reports/strategy_research_preregistration.md (预注册, 不事后调参).
用法: python scripts/run_strategy_research.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import strategy_research_harness as H


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# --------------------------------------------------------------------------- #
# 指标 helper (均基于 close 或 high/low, 按 symbol 分组)
# --------------------------------------------------------------------------- #
def _ma(df, n):
    return df.groupby("symbol")["close"].transform(lambda s: s.rolling(n).mean())


def _std(df, n):
    return df.groupby("symbol")["close"].transform(lambda s: s.rolling(n).std())


def _rsi(df, n):
    def f(s):
        d = s.diff()
        gain = d.clip(lower=0).rolling(n).mean()
        loss = (-d.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)
    return df.groupby("symbol")["close"].transform(f)


def _macd_dif(df, fast=12, slow=26):
    def f(s):
        return s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return df.groupby("symbol")["close"].transform(f)


def _macd_dea(df, fast=12, slow=26, sig=9):
    dif = _macd_dif(df, fast, slow)
    return dif.groupby(df["symbol"]).transform(lambda s: s.ewm(span=sig, adjust=False).mean())


def _kdj(df, n=9):
    def f(g):
        low_n = g["low"].rolling(n).min()
        high_n = g["high"].rolling(n).max()
        rsv = (g["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        d = k.ewm(alpha=1 / 3, adjust=False).mean()
        return pd.DataFrame({"k": k, "d": d}, index=g.index)
    out = df.groupby("symbol", group_keys=False).apply(f)
    return out["k"], out["d"]


def _boll(df, n=20, k=2):
    mid = _ma(df, n)
    sd = _std(df, n)
    return mid, mid + k * sd, mid - k * sd


def _cross_up(df, a, b):
    prev_a = a.groupby(df["symbol"]).shift(1)
    prev_b = b.groupby(df["symbol"]).shift(1)
    return (prev_a <= prev_b) & (a > b)


def _cross_down(df, a, b):
    prev_a = a.groupby(df["symbol"]).shift(1)
    prev_b = b.groupby(df["symbol"]).shift(1)
    return (prev_a >= prev_b) & (a < b)


def _float_mkt(df):
    return df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8  # 亿


# --------------------------------------------------------------------------- #
# 策略定义
# --------------------------------------------------------------------------- #
# 排名策略: (名称, 类别, score_fn) — score_fn(df)->Series
RANKING = []


def reg_rank(name, cat, fn):
    RANKING.append({"name": name, "category": cat, "fn": fn})


reg_rank("动量20", "中线", lambda df: df.groupby("symbol")["close"].pct_change(20))
reg_rank("反转20", "中线", lambda df: -df.groupby("symbol")["close"].pct_change(20))
reg_rank("52周新高", "中线", lambda df: df["close"] / df.groupby("symbol")["high"].transform(lambda s: s.rolling(250).max()))
reg_rank("低波", "长线", lambda df: -_std(df, 20))
reg_rank("低换手(冷落)", "长线", lambda df: -df["turn"].astype(float))
reg_rank("小市值", "长线", lambda df: -_float_mkt(df))
reg_rank("低PE(价值)", "长线", lambda df: -df["peTTM"].where(df["peTTM"] > 0))
reg_rank("低PB(价值)", "长线", lambda df: -df["pbMRQ"].where(df["pbMRQ"] > 0))


# 信号持仓策略: (名称, 类别, entry_fn, exit_fn)
SIGNAL = []


def reg_sig(name, cat, entry, exit_):
    SIGNAL.append({"name": name, "category": cat, "entry": entry, "exit": exit_})


def _520_entry(df):
    ma5, ma20 = _ma(df, 5), _ma(df, 20)
    ma20_prev = ma20.groupby(df["symbol"]).shift(1)
    return _cross_up(df, ma5, ma20) & (ma20 >= ma20_prev)


def _520_exit(df):
    ma5, ma20 = _ma(df, 5), _ma(df, 20)
    return _cross_down(df, ma5, ma20)


reg_sig("520金叉", "超短线", _520_entry, _520_exit)


def _macd_entry(df):
    return _cross_up(df, _macd_dif(df), _macd_dea(df))


def _macd_exit(df):
    return _cross_down(df, _macd_dif(df), _macd_dea(df))


reg_sig("MACD金叉", "超短线", _macd_entry, _macd_exit)


def _kdj_entry(df):
    k, d = _kdj(df)
    return _cross_up(df, k, d) & (k < 20)


def _kdj_exit(df):
    k, d = _kdj(df)
    return _cross_down(df, k, d) & (k > 80)


reg_sig("KDJ低位金叉", "超短线", _kdj_entry, _kdj_exit)


def _rsi_entry(df):
    return _rsi(df, 6) < 30


def _rsi_exit(df):
    return _rsi(df, 6) > 70


reg_sig("RSI超卖反弹", "超短线", _rsi_entry, _rsi_exit)


def _boll_low_entry(df):
    _, up, lo = _boll(df)
    return df["close"] < lo


def _boll_low_exit(df):
    mid, _, _ = _boll(df)
    return df["close"] > mid


reg_sig("布林下轨反弹", "超短线", _boll_low_entry, _boll_low_exit)


def _dual2060_entry(df):
    return _cross_up(df, _ma(df, 20), _ma(df, 60))


def _dual2060_exit(df):
    return _cross_down(df, _ma(df, 20), _ma(df, 60))


reg_sig("双均线20/60", "中线", _dual2060_entry, _dual2060_exit)


def _breakout_entry(df):
    hi = df.groupby("symbol")["high"].transform(lambda s: s.rolling(20).max().shift(1))
    return df["close"] > hi


def _breakout_exit(df):
    lo = df.groupby("symbol")["low"].transform(lambda s: s.rolling(20).min().shift(1))
    return df["close"] < lo


reg_sig("平台突破20日新高", "中线", _breakout_entry, _breakout_exit)


def _turtle_entry(df):
    hi = df.groupby("symbol")["high"].transform(lambda s: s.rolling(20).max().shift(1))
    return df["close"] > hi


def _turtle_exit(df):
    lo = df.groupby("symbol")["low"].transform(lambda s: s.rolling(10).min().shift(1))
    return df["close"] < lo


reg_sig("海龟20/10", "中线", _turtle_entry, _turtle_exit)


def _boll_trend_entry(df):
    _, up, _ = _boll(df)
    return df["close"] > up


def _boll_trend_exit(df):
    _, _, lo = _boll(df)
    return df["close"] < lo


reg_sig("布林趋势", "中线", _boll_trend_entry, _boll_trend_exit)


def _rsi50_entry(df):
    r = _rsi(df, 14)
    return _cross_up(df, r, pd.Series(50.0, index=r.index))


def _rsi50_exit(df):
    r = _rsi(df, 14)
    return _cross_down(df, r, pd.Series(50.0, index=r.index))


reg_sig("RSI14破50", "中线", _rsi50_entry, _rsi50_exit)


# 固定持有策略: (名称, 类别, signal_fn, hold_days, exit_price)
FIXED = []


def _bottom3(df):
    r = df.groupby("date")["pctChg"].rank(method="first", ascending=True)
    return r <= 3


FIXED.append({"name": "隔夜反转bottom3", "category": "超短线", "fn": _bottom3, "hold": 1, "exit": "open"})


# 长线基本面 (2018+, point-in-time)
def _roe_rank(df, fund):
    roe = H.pt_fundamental_col(df, fund, "roeAvg")
    return roe


def _growth_rank(df, fund):
    yoy = H.pt_fundamental_col(df, fund, "YOYNI")
    return yoy


def _pbroe_rank(df, fund):
    roe = H.pt_fundamental_col(df, fund, "roeAvg")
    pb = df["pbMRQ"].where(df["pbMRQ"] > 0)
    # 综合: ROE 越高越好 + PB 越低越好 (各自横截面排名相加)
    roe_rank = roe.groupby(df["date"]).rank(pct=True)
    pb_rank = pb.groupby(df["date"]).rank(pct=True, ascending=False)
    return roe_rank + pb_rank


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    _force_utf8()
    idx = H.load_index()
    fund = H.load_fundamentals()
    has_fund = not fund.empty

    strategies = []
    for s in RANKING:
        strategies.append({"name": s["name"], "category": s["category"], "kind": "rank", "top_k": 50, "freq": "monthly", "fn": s["fn"], "fund": False})
    for s in SIGNAL:
        strategies.append({"name": s["name"], "category": s["category"], "kind": "signal", "entry": s["entry"], "exit": s["exit"], "fund": False})
    for s in FIXED:
        strategies.append({"name": s["name"], "category": s["category"], "kind": "fixed", "fn": s["fn"], "hold": s["hold"], "exit": s["exit"], "fund": False})
    if has_fund:
        strategies.append({"name": "高ROE(质量)", "category": "长线", "kind": "rank", "top_k": 50, "freq": "monthly",
                           "fn": lambda df, fund=fund: _roe_rank(df, fund), "fund": True})
        strategies.append({"name": "PB-ROE质量价值", "category": "长线", "kind": "rank", "top_k": 50, "freq": "monthly",
                           "fn": lambda df, fund=fund: _pbroe_rank(df, fund), "fund": True})

    results = {}
    t0 = time.time()
    for window, fp in H.WINDOWS.items():
        year = int(window[:4])
        df = H.load_base_window(fp)
        if df.empty:
            continue
        for st in strategies:
            if st["fund"] and year < 2018:
                continue  # 基本面数据 2018+ 才有, 跳过更早窗口
            try:
                if st["kind"] == "rank":
                    m = H.run_window_ranking(df, idx, st["fn"], st["top_k"], st["freq"], window)
                elif st["kind"] == "signal":
                    m = H.run_window_signal(df, idx, st["entry"], st["exit"], window)
                else:
                    m = H.run_window_fixed_hold(df, idx, st["fn"], st["hold"], st["exit"], window)
            except Exception as e:
                m = {"error": str(e), "regime": H.REGIME.get(window, "")}
            results.setdefault(st["name"], {})[window] = m
        print(f"  [{window}] 完成 {time.time()-t0:.1f}s", flush=True)
        # 增量落盘, 防中途被杀丢结果 (detail 足够复盘, 汇总可在被杀后重建)
        H.dump_json({"detail": results}, ROOT / "reports" / "strategy_research_result.json")
    print(f"[总耗时 {time.time()-t0:.1f}s]")

    # ── 汇总 ──
    print("\n" + "=" * 130)
    print("策略研究回测结果汇总 (真实日线, 成本31bp, 基准=匹配universe等权[月再平衡])")
    print("=" * 130)
    by_cat = {}
    for st in strategies:
        by_cat.setdefault(st["category"], []).append(st["name"])

    summary_rows = []
    for st in strategies:
        ws = results.get(st["name"], {})
        if st["kind"] == "fixed":
            act = [m for m in ws.values() if "avg_net_ret" in m and m.get("avg_net_ret") is not None]
            if not act:
                continue
            n = len(act)
            wins = sum(1 for m in act if m["avg_net_ret"] > 0)
            mean_vu = float(np.mean([m["avg_net_ret"] for m in act]))
            mean_win = float(np.mean([m["win_rate"] for m in act]))
            regime_vu = {}
            for rg in ("纯牛", "熊/崩", "震荡"):
                sub = [m["avg_net_ret"] for m in act if m["regime"] == rg]
                regime_vu[rg] = float(np.mean(sub)) if sub else np.nan
            summary_rows.append({
                "name": st["name"], "category": st["category"], "kind": "fixed", "n": n, "wins": wins,
                "mean_vu": mean_vu, "mean_sharpe": mean_win, "regime_vu": regime_vu,
            })
        else:
            act = [m for m in ws.values() if "vs_universe" in m and m["vs_universe"] is not None and not np.isnan(m["vs_universe"])]
            if not act:
                continue
            n = len(act)
            wins = sum(1 for m in act if m["vs_universe"] > 0)
            mean_vu = float(np.mean([m["vs_universe"] for m in act]))
            mean_sh = float(np.mean([m["sharpe"] for m in act if m["sharpe"] is not None and not np.isnan(m["sharpe"])]))
            regime_vu = {}
            for rg in ("纯牛", "熊/崩", "震荡"):
                sub = [m["vs_universe"] for m in act if m["regime"] == rg]
                regime_vu[rg] = float(np.mean(sub)) if sub else np.nan
            summary_rows.append({
                "name": st["name"], "category": st["category"], "kind": st["kind"], "n": n, "wins": wins,
                "mean_vu": mean_vu, "mean_sharpe": mean_sh, "regime_vu": regime_vu,
            })

    H.dump_json({"strategies": summary_rows, "detail": results}, ROOT / "reports" / "strategy_research_result.json")
    print("\n结果已写 reports/strategy_research_result.json")

    for cat in ("超短线", "短线", "中线", "长线"):
        rows = [r for r in summary_rows if r["category"] == cat]
        if not rows:
            continue
        is_fixed = rows and rows[0]["kind"] == "fixed"
        print(f"\n## {cat}" + ("  (事件固定持有, 均=单笔均净收益/胜率)" if is_fixed else ""))
        print(f"  {'策略':<18}{'窗口':>4}{'赢/窗':>6}{'均值':>10}{'Sharpe/胜率':>11}{'纯牛':>8}{'熊/崩':>8}{'震荡':>8}")
        for r in sorted(rows, key=lambda x: -x["mean_vu"]):
            rv = r["regime_vu"]
            print(f"  {r['name']:<18}{r['n']:>4}{r['wins']:>4}/{r['n']:<3}{r['mean_vu']*100:>+9.1f}%{r['mean_sharpe']:>+10.2f}"
                  f"{_f(rv['纯牛']):>8}{_f(rv['熊/崩']):>8}{_f(rv['震荡']):>8}")


def _f(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "   n/a"
    return f"{x*100:+7.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
