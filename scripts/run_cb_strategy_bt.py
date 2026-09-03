"""可转债小资金策略回测 (零模拟, 真实日线, 预注册判据见 reports/cb_strategy_preregistration.md).

三臂:
  A 双低轮动 (周, top10)     score = -(close + premium*100)
  B 双低轮动 (月, top10)     同 A
  C 低溢价高活跃不隔夜 (T+0) T-1收盘选 (溢价率<40% + 成交额代理top10), T开盘买T收盘卖

基准: 转债等权 (面板全债日收益等权) + 中证转债 000832 (可得则并入)。
成本: 主口径 20bp 全周转, 敏感性 5bp。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "cb_strategy_result.json"

COST_MAIN = 20.0   # bp 全周转 (主口径)
COST_LOW = 5.0     # bp (敏感性)
TOP_K = 10
PREMIUM_CAP = 40.0  # C 臂溢价率上限 (%)
D_FIRST_DAY_MIN = 128.0  # D 臂首日"封板"近似 (130 上限)
D_TURNOVER_MAX = 0.05    # D 臂首日换手率代理上限 (成交额/发行规模)
D_HOLD = 20              # D 臂持有交易日数

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
REGIME = {
    "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "熊/崩",
    "2021白马转小盘": "纯牛", "2022熊": "熊/崩", "2023震荡": "震荡",
    "2024震荡": "震荡", "2025-26现期": "纯牛",
}


# --------------------------------------------------------------------------- #
def load_panel() -> dict[str, pd.DataFrame]:
    daily = pd.read_parquet(ROOT / "replay_data" / "cb_daily.parquet")
    value = pd.read_parquet(ROOT / "replay_data" / "cb_value.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    value["date"] = pd.to_datetime(value["日期"])
    value["symbol"] = value["symbol"].astype(str)

    def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
        w = df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")
        return w.sort_index()

    close = wide(daily, "close")
    open_ = wide(daily, "open")
    vol = wide(daily, "volume")
    # 东财估值: 转股溢价率 (%)
    prem = wide(value, "转股溢价率")
    # 对齐: 只保留两侧都有价的 (date, bond)
    common_cols = close.columns.intersection(prem.columns)
    close, open_, vol, prem = (w[common_cols] for w in (close, open_, vol, prem))
    valid = close.notna() & prem.notna()
    close = close.where(valid)
    open_ = open_.where(close.notna())
    vol = vol.where(close.notna())
    prem = prem.where(close.notna())
    ret = close.pct_change(fill_method=None)
    amt = (vol * (open_ + close) / 2.0)  # 成交额代理 (仅排名信号)
    return {"close": close, "open": open_, "prem": prem, "ret": ret, "amt": amt, "vol": vol}


def rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    s = pd.Series(dates)
    if freq == "weekly":
        grp = [s.dt.isocalendar().year.astype(int), s.dt.isocalendar().week.astype(int)]
    elif freq == "monthly":
        grp = [s.dt.year, s.dt.month]
    else:
        raise ValueError(freq)
    return s.groupby(grp).apply(lambda x: x.iloc[-1]).tolist()


def select_top(score: pd.DataFrame, T: pd.Timestamp, k: int,
               cap_mask: pd.DataFrame | None = None) -> list[str]:
    row = score.loc[T]
    if cap_mask is not None:
        row = row.where(cap_mask.loc[T])
    row = row.dropna()
    return row.nlargest(k).index.tolist()


def run_ranking_arm(panel: dict, freq: str, cost_bps: float, top_k: int = TOP_K) -> pd.Series:
    """A/B: T 收盘调仓 → 持有至下一调仓日, 赚 T+1..T_next 的日收益; 换仓比例×成本在期首扣."""
    close, prem, ret = panel["close"], panel["prem"], panel["ret"]
    dates = close.index
    score = -(close + prem * 100.0)
    rl = rebalance_dates(dates, freq)
    sel = {T: select_top(score, T, top_k) for T in rl}
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
        replaced = len(set(picked) - set(prev)) / top_k
        cost.loc[seg.index[0]] += replaced * cost_bps / 1e4
    return port - cost


def run_overnight_arm(panel: dict, cost_bps: float, top_k: int = TOP_K) -> pd.Series:
    """C: T-1 收盘选 (溢价率<40% + 成交额代理 top10), T 开盘买 T 收盘卖."""
    open_, close, prem, amt = panel["open"], panel["close"], panel["prem"], panel["amt"]
    dates = close.index
    intraday = close / open_ - 1.0  # T 开→收
    cap_mask = (prem < PREMIUM_CAP) & (prem.notna()) & (open_.notna())
    port = pd.Series(np.nan, index=dates)
    for i in range(1, len(dates)):
        T_prev, T = dates[i - 1], dates[i]
        row = amt.loc[T_prev].where(cap_mask.loc[T_prev]).dropna()
        if row.empty:
            continue
        picked = row.nlargest(top_k).index.tolist()
        port.loc[T] = intraday.loc[T, picked].mean() - 2 * cost_bps / 1e4  # 一回合=双边
    return port


def cb_equal_weight(panel: dict) -> pd.Series:
    r = panel["ret"]
    return r.mean(axis=1).where(r.notna().sum(axis=1) >= 3)  # <3只有效债的日子不算


def run_new_issue_arm(panel: dict, size_yi: dict[str, float]) -> pd.DataFrame:
    """D: 首日收盘≥128 且 首日成交额/发行规模<5% 的券, 上市第2日收盘买, 持有20交易日收盘卖."""
    close, open_, vol = panel["close"], panel["open"], panel["vol"]
    trades = []
    for sym in close.columns:
        s = close[sym].dropna()
        if len(s) < D_HOLD + 2:
            continue
        d0, p0 = s.index[0], float(s.iloc[0])
        if p0 < D_FIRST_DAY_MIN or sym not in size_yi or size_yi[sym] <= 0:
            continue
        amt0 = float(vol.loc[d0, sym]) * float(open_.loc[d0, sym] if not np.isnan(open_.loc[d0, sym]) else p0)
        if amt0 / (size_yi[sym] * 1e8) >= D_TURNOVER_MAX:
            continue
        buy_d, buy_p = s.index[1], float(s.iloc[1])
        sell_d, sell_p = s.index[1 + D_HOLD], float(s.iloc[1 + D_HOLD])
        trades.append({"symbol": sym, "list_date": d0.date(), "buy_date": buy_d.date(),
                       "sell_date": sell_d.date(), "buy_year": buy_d.year,
                       "net_ret": round(sell_p / buy_p - 1.0 - COST_MAIN / 1e4, 4)})
    return pd.DataFrame(trades)


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


def fetch_csi_cb() -> pd.DataFrame | None:
    """中证转债 000832 (东财), 失败返回 None 并如实标注 (不模拟)."""
    cache = ROOT / "replay_data" / "csi_cb_index.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import akshare as ak
    try:
        idx = ak.index_zh_a_hist(symbol="000832", period="daily", start_date="20180101", end_date="20260731")
    except Exception as e:
        print(f"[warn] 中证转债指数获取失败, 仅用转债等权基准: {e}")
        return None
    if idx is None or idx.empty:
        return None
    idx = idx.rename(columns={"日期": "date", "收盘": "close"})
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")["close"].sort_index()
    idx.to_frame().to_parquet(cache)
    return idx.to_frame()


# --------------------------------------------------------------------------- #
def main() -> None:
    panel = load_panel()
    close = panel["close"]
    n_bonds = close.notna().sum(axis=1)
    print(f"panel: {close.shape[0]} days x {close.shape[1]} bonds, "
          f"first valid {n_bonds[n_bonds > 0].index.min().date()}, "
          f"last {close.index.max().date()}")
    bench_ew = cb_equal_weight(panel)
    csi = fetch_csi_cb()

    arms = {
        "A_双低周": lambda c: run_ranking_arm(panel, "weekly", c),
        "B_双低月": lambda c: run_ranking_arm(panel, "monthly", c),
        "C_不隔夜T0": lambda c: run_overnight_arm(panel, c),
    }

    result: dict = {"preregistration": "reports/cb_strategy_preregistration.md",
                    "cost_main_bps": COST_MAIN, "cost_low_bps": COST_LOW,
                    "windows": {}}
    for wname, (t0, t1) in WINDOWS.items():
        win = {"regime": REGIME[wname]}
        ew_seg = bench_ew[(bench_ew.index >= t0) & (bench_ew.index <= t1)]
        win["benchmark_转债等权"] = metrics(ew_seg)
        if csi is not None:
            seg = csi[(csi.index >= t0) & (csi.index <= t1)]["close"].pct_change()
            win["benchmark_中证转债"] = metrics(seg)
        for aname, fn in arms.items():
            for tag, cost in (("main20", COST_MAIN), ("low5", COST_LOW)):
                r = fn(cost)
                seg = r[(r.index >= t0) & (r.index <= t1)]
                m = metrics(seg)
                m["vs_等权"] = round(m["total"] - win["benchmark_转债等权"]["total"], 4) \
                    if not np.isnan(m["total"]) else np.nan
                win[f"{aname}_{tag}"] = m
        result["windows"][wname] = win
        print(f"{wname}: A20={win['A_双低周_main20']['total']}, "
              f"B20={win['B_双低月_main20']['total']}, "
              f"C20={win['C_不隔夜T0_main20']['total']}, 等权={win['benchmark_转债等权']['total']}", flush=True)

    # 判据汇总 (预注册)
    # D 臂 (新券): 需发行规模
    try:
        cov = ak.bond_zh_cov()
        cov["债券代码"] = cov["债券代码"].astype(str)
        size_yi = dict(zip(cov["债券代码"], pd.to_numeric(cov["发行规模"], errors="coerce")))
    except Exception as e:
        print(f"[warn] 发行规模获取失败, D 臂跳过 (不模拟): {e}")
        size_yi = {}
    if size_yi:
        d = run_new_issue_arm(panel, size_yi)
        if not d.empty:
            result["D_新券T1低换手"] = {
                "n_trades": int(len(d)),
                "median_net": round(float(d["net_ret"].median()), 4),
                "mean_net": round(float(d["net_ret"].mean()), 4),
                "win_rate": round(float((d["net_ret"] > 0).mean()), 4),
                "by_year": {int(y): {"n": int(len(g)),
                                     "median": round(float(g["net_ret"].median()), 4),
                                     "win": round(float((g["net_ret"] > 0).mean()), 4)}
                            for y, g in d.groupby("buy_year")},
            }
            d.to_csv(ROOT / "reports" / "cb_new_issue_trades.csv", index=False, encoding="utf-8-sig")
        else:
            result["D_新券T1低换手"] = {"n_trades": 0}
        result["D_新券T1低换手"]["PASS"] = bool(
            result["D_新券T1低换手"].get("n_trades", 0) >= 20
            and result["D_新券T1低换手"].get("median_net", -1) > 0
            and result["D_新券T1低换手"].get("win_rate", 0) > 0.5)

    summary = {}
    for aname in arms:
        for tag, cost in (("main20", COST_MAIN), ("low5", COST_LOW)):
            key = f"{aname}_{tag}"
            wins_vs_ew = sum(1 for w in result["windows"].values()
                             if w[key]["total"] >= w["benchmark_转债等权"]["total"])
            avg_vs = float(np.mean([w[key]["vs_等权"] for w in result["windows"].values()]))
            pos_windows = sum(1 for w in result["windows"].values() if w[key]["total"] > 0)
            dd_ok = all(w[key]["maxdd"] >= -0.30 for w in result["windows"].values())
            summary[key] = {
                "wins_vs_等权": f"{wins_vs_ew}/8",
                "avg_vs_等权": round(avg_vs, 4),
                "pos_windows": f"{pos_windows}/8",
                "maxdd_all_ge_-30%": dd_ok,
            }
    # 预注册判据: A/B 主 = wins≥6/8 且 avg>0 且 dd_ok; C 附加 = 20bp 下 pos≥6/8 且 avg>0
    for k, s in summary.items():
        if k.startswith("C_") and "main20" in k:
            s["PASS"] = bool(int(s["pos_windows"].split("/")[0]) >= 6 and s["avg_vs_等权"] > 0)
        elif "low5" not in k:
            s["PASS"] = bool(int(s["wins_vs_等权"].split("/")[0]) >= 6 and s["avg_vs_等权"] > 0
                             and s["maxdd_all_ge_-30%"])
    result["summary"] = summary
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
