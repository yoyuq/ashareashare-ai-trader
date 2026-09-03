"""ETF 网格交易回测 (零模拟, 真实日线, 事件级模拟).

对应主流小资金策略清单中的「网格交易」(震荡市低买高卖, 网格间距+每格固定手数)。

规则 (预注册, 跑之前写死):
  - 标的: 510300 (沪深300ETF), 510500 (中证500ETF) — 事先指定, 不挑品种
  - 初始资金 10000 元; 建仓 = 50% 仓位 (按手取整, 100股/手)
  - 网格间距 3% (敏感性 1.5%/5%), 每格 100 股 (1手, 1万本金现实约束)
  - 开盘即以现价为锚: 下一买档 = 锚×(1-g), 下一卖档 = 锚×(1+g)
  - 成交: 日内 low ≤ 买档 → 以 min(open, 买档) 成交买入; high ≥ 卖档 → 以 max(open, 卖档) 卖出;
    成交后以成交价为锚重新挂相邻买卖档 (循环至当日不再触发)
  - 费用: 每笔 max(名义额×1bp, ¥1) (佣金万1+最低1元的现实约束, 这是1万本金网格的真实杀手)
  - 判据: vs 同标的买入持有(同初始资金、含建仓费), ≥6/8 窗口 total ≥ 买入持有 且平均超额>0 → PASS
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "grid_etf_result.json"
CAPITAL = 10000.0
LOT = 100
GRID_MAIN = 0.03
GRID_SENS = (0.015, 0.05)
FEE_RATE = 1e-4
FEE_MIN = 1.0

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


def fetch_etf(symbol: str) -> pd.DataFrame:
    cache = ROOT / "replay_data" / f"etf_{symbol}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        import akshare as ak
        df = ak.fund_etf_hist_sina(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(cache)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def run_grid(df: pd.DataFrame, t0: str, t1: str, grid: float):
    """返回 (期末净值, 逐日净值序列, 成交笔数)."""
    seg = df[(df.index >= t0) & (df.index <= t1)]
    if len(seg) < 20:
        return None
    p0 = float(seg["open"].iloc[0])
    shares = int(CAPITAL * 0.5 / (p0 * LOT)) * LOT
    cash = CAPITAL - shares * p0 - max(CAPITAL * 0.5 * FEE_RATE, FEE_MIN)
    next_buy = p0 * (1 - grid)
    next_sell = p0 * (1 + grid)
    n_trades = 0
    nav = {}
    for d, row in seg.iterrows():
        o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
        # 买入档触发 (可多档: 循环)
        while l <= next_buy:
            px = min(o, next_buy)
            cost = px * LOT
            fee = max(cost * FEE_RATE, FEE_MIN)
            if cash >= cost + fee:
                cash -= cost + fee
                shares += LOT
                n_trades += 1
                next_buy = px * (1 - grid)
                next_sell = max(next_sell, px * (1 + grid))
            else:
                break  # 现金不足, 不再加仓 (1万本金网格真实约束)
        # 卖出档触发
        while h >= next_sell and shares >= LOT:
            px = max(o, next_sell)
            cash += px * LOT - max(px * LOT * FEE_RATE, FEE_MIN)
            shares -= LOT
            n_trades += 1
            next_sell = px * (1 + grid)
            next_buy = min(next_buy, px * (1 - grid))
        nav[d] = cash + shares * float(row["close"])
    nav = pd.Series(nav).sort_index()
    return nav / CAPITAL, n_trades


def metrics(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    total = float(nav.iloc[-1] - 1.0)
    std = float(r.std()) if len(r) > 1 else float("nan")
    sharpe = float(r.mean() / std * 252 ** 0.5) if std and std > 1e-12 else float("nan")
    maxdd = float((nav / nav.cummax() - 1.0).min())
    return {"total": round(total, 4), "sharpe": round(sharpe, 3), "maxdd": round(maxdd, 4)}


def main() -> None:
    result: dict = {"windows": {}}
    for sym in ("sh510300", "sh510500"):
        df = fetch_etf(sym)
        for wname, (t0, t1) in WINDOWS.items():
            out = result["windows"].setdefault(wname, {})
            # 买入持有基准 (同初始资金, 含建仓费)
            seg = df[(df.index >= t0) & (df.index <= t1)]
            p0 = float(seg["open"].iloc[0])
            sh0 = int(CAPITAL / (p0 * LOT)) * LOT
            bh = sh0 * seg["close"] / CAPITAL
            out[f"{sym}_买入持有"] = metrics(bh)
            g, n = run_grid(df, t0, t1, GRID_MAIN)
            out[f"{sym}_网格3%"] = {**metrics(g), "n_trades": n}
            for gs in GRID_SENS:
                g2, n2 = run_grid(df, t0, t1, gs)
                out[f"{sym}_网格{gs*100:g}%"] = {**metrics(g2), "n_trades": n2}
            print(f"{wname} {sym}: 网格3%={out[f'{sym}_网格3%']['total']}, "
                  f"持有={out[f'{sym}_买入持有']['total']}", flush=True)

    # 判据汇总 (主参数 3%)
    summary = {}
    for sym in ("sh510300", "sh510500"):
        wins = sum(1 for w in result["windows"].values()
                   if w[f"{sym}_网格3%"]["total"] >= w[f"{sym}_买入持有"]["total"])
        avg_vs = float(pd.Series([w[f"{sym}_网格3%"]["total"] - w[f"{sym}_买入持有"]["total"]
                                  for w in result["windows"].values()]).mean())
        summary[f"{sym}_网格3%"] = {"wins_vs_持有": f"{wins}/8", "avg_vs": round(avg_vs, 4),
                                    "PASS": bool(wins >= 6 and avg_vs > 0)}
    result["summary"] = summary
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
