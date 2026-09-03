"""龙头战法 (空间龙头/最高连板) 忠实日线回测 (预注册 §十三).

背景: 用户点名研究 A 股「龙头战法」. 核心 = 每天买市场上**连板高度最高**的股票 (空间板/总龙头), 顺动量持有,
**断板 (不再涨停) 卖出**. 全网教材强调: 龙头只在情绪周期/顺指数时做, 逆势打板必死; 且一字板/涨停封死买不进
需要竞价/封单(L2)价内承接. 本轮用日线忠实映射可测部分, 一字板不可买如实在口径里披露.

信号 (日线忠实映射, 预注册, 跑之前写死):
  连板高度 L(期,日) = 该期自最近一个非涨停日起连续 pctChg≥9.8 的天数 (首板=1, 二板=2…).
  空间龙头(T−1) = 于 T−1 日在「主板(sh.60/sz.00)+非ST+可交易」池中 argmax(L, 其次 amount) 的股票, 且 L≥1.
  买 T 开盘价 (披露: 若 T 一字涨停 o==h==l 则买不进, 列为独立口径).
  卖 断板 = 买入后首个 close pctChg<9.8 的交易日, 以该日收盘价卖 (T+1 已尊重: 买入当日不卖); 窗口末仍持有→末收盘卖.
  成本 31bp/笔.

三口 (预注册):
  A 全样本:   买所有龙头 T 开盘 (含一字, 不可成交, 仅诊断)
  B 可成交:   仅买 T 开盘非一字 (o==h==l 跳过) — 主口径, 诚实可执行子集
  C 情绪门:   B 且仅当 空间龙头连板高度 ≥3 (「情绪冰点不打空间板」, C 情绪周期过滤)

判据 (预注册, 跑完不调, 与前短线同口径): 主口径 B 的单笔均净收益 > 0 且 ≥7/11 窗口 B 单笔均净为正.
若 B 不成立 → 龙头战法(日线可映射形态)无稳定 edge, 与既往 8 条活跃票短线墙一致.

用法: python scripts/run_leader_strategy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import strategy_research_harness as H  # noqa: E402

LU = 9.8  # 涨停阈值 (主板 10% 留 0.2 容差)
COST_BPS = 31.0
MIN_GATE = 3  # C 情绪门: 空间龙头板高 ≥3


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    df["limit"] = df["pctChg"] >= LU
    g = df.groupby("symbol", sort=False)
    flip = (df["limit"] != g["limit"].shift(1)).cumsum()
    df["grp"] = flip
    df["streak"] = df.groupby(["symbol", "grp"])["limit"].cumsum()
    df["streak"] = df["streak"].where(df["limit"], 0).astype(int)
    df["one_word"] = (df["open"] == df["high"]) & (df["low"] == df["high"]) & df["limit"]
    return df


def _leader_table(df: pd.DataFrame) -> pd.DataFrame:
    """每交易日 → 空间龙头 (argmax streak, 平局按 amount). 仅 L≥1."""
    mask = df["streak"] >= 1
    sub = df[mask]
    if sub.empty:
        return pd.DataFrame(columns=["date", "symbol", "streak", "amount"])
    lead = sub.sort_values(["streak", "amount"], ascending=[False, False]).groupby("date").head(1)
    return lead[["date", "symbol", "streak", "amount"]].copy()


def _lhb_map() -> dict:
    """龙虎榜历史库 (预注册 §十四). -> {code: sorted[(date, net_buy)]}.
    point-in-time: 只保留 上榜日 与 净买额; 背测仅用 <= 买入前一日 的记录."""
    path = Path(__file__).parent.parent / "replay_data" / "lhb_history.parquet"
    if not path.exists():
        print("  [WARN] 无 replay_data/lhb_history.parquet — 先跑 scripts/fetch_lhb_history.py 建库 (arm D 将空)")
        return {}
    lh = pd.read_parquet(path)
    if lh.empty or "_code" not in lh.columns:
        return {}
    m = {}
    for code, g in lh.groupby("_code"):
        rows = sorted((pd.Timestamp(d).date() if hasattr(d, "date") and not isinstance(d, str)
                       else pd.Timestamp(d).date() if isinstance(d, str)
                       else d, nb) for d, nb in zip(g["_date"], g["龙虎榜净买额"]) if pd.notna(nb))
        m[code] = rows
    return m


def _has_fund_favor(lhb: dict, sym: str, prev_date, n=5) -> bool:
    """预注册 §十四 gate: 龙头候选票存在 上榜日∈[prev_date-4, prev_date] 且至少一条 净买额>0."""
    sym = sym.split(".")[-1]  # 'sh.600519' -> '600519' (龙虎榜库键为6位code)
    rows = lhb.get(sym)
    if not rows:
        return False
    if hasattr(prev_date, "date"):
        prev_date = prev_date.date()
    elif isinstance(prev_date, str):
        prev_date = pd.Timestamp(prev_date).date()
    from datetime import timedelta
    lo = prev_date - timedelta(days=n - 1)
    for d, nb in rows:
        if d <= prev_date and d >= lo and nb > 0:
            return True
        if d > prev_date:
            break
    return False


def _simulate(df: pd.DataFrame, arm: str, lhb: dict | None = None) -> list:
    """单仓事件模拟: 前一日龙头 → 今日开盘买; 断板收盘卖.
    arm D = B 可成交口径 + 资金认可(龙虎榜净买额>0) gate (预注册 §十四)."""
    lead = _leader_table(df).set_index("date")
    dates = sorted(df["date"].unique())
    # 每 symbol 按 date → row 索引
    day_symbol = df.set_index(["symbol", "date"]).sort_index()
    o = df.groupby(["symbol", "date"])["open"].first()
    close = df.groupby(["symbol", "date"])["close"].first()
    limit_ = df.groupby(["symbol", "date"])["limit"].first()
    oneword = df.groupby(["symbol", "date"])["one_word"].first()

    trades = []
    hold = None  # (symbol, buy_price, buy_date)
    for i, d in enumerate(dates):
        # 用前一个交易日选出的龙头决定今日买
        if i >= 1:
            prev_d = dates[i - 1]
            if hold is None and prev_d in lead.index:
                ldr = lead.loc[prev_d]  # Series (每交易日一个龙头)
                if isinstance(ldr, pd.DataFrame):
                    ldr = ldr.iloc[0]
                sym = ldr["symbol"]
                stre = int(ldr["streak"])
                if arm == "C" and stre < MIN_GATE:
                    pass  # 情绪门: 空间板高度不足, 空仓
                elif not ((sym, d) in o.index):
                    pass  # 禁用日
                else:
                    ok_open = (sym, d) in o.index and bool(not oneword.loc[(sym, d)])
                    if arm == "A":
                        buy = float(o.loc[(sym, d)])
                        if buy > 0:
                            hold = (sym, buy, d)
                    else:  # B / C: 非一字才买
                        if ok_open:
                            if arm == "D" and not (lhb and _has_fund_favor(lhb, sym, pd.Timestamp(prev_d))):
                                continue  # D: 资金认可 gate, 无游资接力信号则空仓
                            buy = float(o.loc[(sym, d)])
                            if buy > 0:
                                hold = (sym, buy, d)
        # 已持仓 → 判断断板 (仅买入日后)
        if hold is not None and d > hold[2]:
            sym = hold[0]
            if not (sym, d) in close.index:
                continue
            c = float(close.loc[(sym, d)])
            lim = bool(limit_.loc[(sym, d)])
            if not lim:  # 断板
                trades.append((sym, hold[1], c))
                hold = None
    # 窗口末仍持仓 → 末收盘卖
    if hold is not None:
        sym = hold[0]
        last_d = dates[-1]
        if (sym, last_d) in close.index:
            trades.append((sym, hold[1], float(close.loc[(sym, last_d)])))
    return trades


def _stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "mean_net": None, "win": None, "total": None}
    cost = COST_BPS / 10000.0
    net = np.array([c / b - 1.0 - cost for _, b, c in trades])
    return {"n": int(len(net)), "mean_net": round(float(net.mean()), 5),
            "win": round(float((net > 0).mean()), 4), "total": round(float((1 + net).prod()) - 1, 4)}


def main() -> int:
    _force_utf8()
    idx = H.load_index()
    arms = ["A", "B", "C", "D"]
    lhb = _lhb_map()
    results = {a: {} for a in arms}
    print("=" * 110)
    print("龙头战法 (空间龙头最高连板) — 忠实日线回测, 31bp, 断板收盘卖")
    print("D = 游资接力(gate: 上榜日∈[T-5,T-1] 净买额>0) 预注册§十四")
    print("=" * 110)
    hdr = f"{'窗口':<12} | {'A全样本':>14} | {'B可成交(主)':>14} | {'C情绪门':>14} | {'D资金认可':>14}"
    print(hdr); print("-" * 110)

    for window in H.WINDOWS:
        df = _prep(H.load_base_window(H.WINDOWS[window]))
        t0, t1 = df["date"].min(), df["date"].max()
        idx_total = H._idx_total(idx, t0, t1)
        line = f"{window:<12} "
        for a in arms:
            trades = _simulate(df, a, lhb if a == "D" else None)
            s = _stats(trades)
            results[a][window] = {"n_trades": s["n"], "mean_net": s["mean_net"], "win": s["win"],
                                  "total": s["total"], "index_total": round(idx_total, 4)}
            nl = f"{s['n']}笔均{s['mean_net']*100:+.2f}%" if s['mean_net'] is not None else f"{s['n']}笔空"
            line += f" | {nl:>14}"
        print(line)

    print("\n" + "=" * 110)
    print("预注册判据: §十三 主口径 B 均净>0 且 ≥7/11 窗口正 | §十四 主口径 D(资金认可) 同样门槛 = 游资 gate 能否救回")
    print("=" * 110)
    for a in arms:
        vals = [results[a][w]["mean_net"] for w in H.WINDOWS if results[a][w]["n_trades"] > 0]
        posw = sum(1 for w in H.WINDOWS if results[a][w]["mean_net"] is not None and results[a][w]["mean_net"] > 0)
        tot = sum(results[a][w]["n_trades"] for w in H.WINDOWS)
        mn = float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else None
        if a == "B":
            jury = "PASS" if (mn is not None and mn > 0 and posw >= 7) else "FAIL"
        elif a == "D":
            jury = "PASS(游资gate救回)" if (mn is not None and mn > 0 and posw >= 7) else "FAIL(仍出血)"
        else:
            jury = "(副,不gate)"
        print(f"[{a}] 总交易 {tot} 笔 | 活跃窗口均单笔净 "
              f"{mn*100 if mn is not None else 0:+.2f}% | 正窗口 {posw}/11 → {jury}")

    # §十四 副指标: D vs B 逐窗口配对差分
    print("\n" + "-" * 110)
    print("副指标 (诊断): arm D(资金认可) vs arm B(可成交) 逐窗口单笔均配对差分 Δ = D−B")
    dels = []
    for w in H.WINDOWS:
        b, d = results["B"][w], results["D"][w]
        if b["n_trades"] and d["n_trades"] and b["mean_net"] is not None and d["mean_net"] is not None:
            delta = d["mean_net"] - b["mean_net"]
            dels.append(delta)
            print(f"  {w:<12} B {b['mean_net']*100:+6.2f}% → D {d['mean_net']*100:+6.2f}%  Δ {delta*100:+6.2f}pp")
    if dels:
        print(f"  均值 Δ = {np.mean(dels)*100:+.2f}pp | D 正窗口 {sum(1 for w in H.WINDOWS if results['D'][w]['mean_net'] and results['D'][w]['mean_net']>0)}/11 vs B 同")

    H.dump_json(results, ROOT / "reports" / "leader_strategy_result.json")
    print(f"\n结果已写 reports/leader_strategy_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())