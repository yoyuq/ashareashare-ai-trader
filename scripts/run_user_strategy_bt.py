"""用户策略回测 (日线近似): 尾盘「涨停基因」超短线.

策略来源: C:/Users/hjl/Desktop/交易策略.txt (7 条), 逐条映射:

  1. 14:30-15:00 尾盘筛选下单        → 用 T 日收盘价近似买价 (尾盘≈收盘)
  2. 主板 + 涨幅3~5% + 近20日有涨停    → 主板=sh.60/sz.00; 涨幅用收盘 pctChg∈[3,5] 代理14:30;
                                       涨停基因=近20日 pctChg≥9.8 (日线可忠实测)
  3. 量比≥1                         → 日线代理: 当日volume / MA5(volume)
  4. 换手率 5~10%                    → 忠实 (turn 列)
  5. 市值 50~200亿                  → 代理: 流通市值 = 成交额/换手率 = amount*100/turn (元)
  6. 站稳分时均线上方 + 强于大盘       → 代理: 收盘≥(高+低)/2 (收在上半区) + pctChg>上证pctChg
  7. 创当天新高回踩分时均线不破 → 进场   → 不可测 (分钟级), 丢弃; 只影响精确买价, 不改选股
     次日早盘冲高就走                 → 代理: T+1 开盘价卖出 (保守); 另报 T+1 收盘作敏感性

不可测/丢弃的部分 (诚实披露): 分时均价线、当天新高回踩的精确买点、次日「冲高」卖点。
这些是日内时序, 现有 replay_data 仅日线, 无法忠实测; 本回测是「选股逻辑 + T+1隔夜」的日线近似。

成本: 31bp 全周转 (佣金万3*2 + 印花税万5卖 + 过户 + 滑点10bp), 与 [[transaction-costs-unified]] 一致。
用法: python scripts/run_user_strategy_bt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

COST_BPS = 31.0
PCT_LO, PCT_HI = 3.0, 5.0
TURN_LO, TURN_HI = 5.0, 10.0
MKT_LO_YI, MKT_HI_YI = 50.0, 200.0
LU_LOOKBACK = 20
LU_PCT = 9.8          # 主板涨停阈值 (10% 限制, 留 0.2 容差)
VOL_LB = 5            # 量比代理 lookback

OUT = ROOT / "reports" / "user_strategy_bt_result.json"

WINDOWS = {
    "2015牛转股灾": "replay_data/daily_2015-01-01_2015-12-31.parquet",
    "2016熔断震荡": "replay_data/daily_2016-01-01_2016-12-31.parquet",
    "2017漂亮50":   "replay_data/daily_2017-01-01_2017-12-31.parquet",
    "2018熊":       "replay_data/daily_2018-01-01_2018-12-31.parquet",
    "2019牛":       "replay_data/daily_2019-01-01_2019-12-31.parquet",
    "2020牛转崩":   "replay_data/daily_2020-06-01_2021-02-28.parquet",
    "2021白马转小盘": "replay_data/daily_2021-01-01_2021-12-31.parquet",
    "2022熊":       "replay_data/daily_2022-01-01_2022-12-31.parquet",
    "2023震荡":     "replay_data/daily_2023-01-01_2023-12-31.parquet",
    "2024震荡":     "replay_data/daily_2024-01-01_2024-12-31.parquet",
    "2025-26现期":  "replay_data/daily_2025-10-08_2026-07-31.parquet",
}

REGIME = {
    "2019牛": "纯牛", "2021白马转小盘": "纯牛", "2025-26现期": "纯牛",
    "2015牛转股灾": "熊/崩", "2018熊": "熊/崩", "2020牛转崩": "熊/崩", "2022熊": "熊/崩",
    "2016熔断震荡": "震荡", "2017漂亮50": "震荡", "2023震荡": "震荡", "2024震荡": "震荡",
}


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _backtest_window(df: pd.DataFrame, idx_pct: pd.Series) -> dict:
    """日线近似回测单个窗口, 返回统计 dict."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # 基础过滤: 主板 + 可交易 + 非ST
    df["isST"] = df["isST"].astype(str)
    df = df[df["is_trade"] == 1]
    df = df[df["isST"] == "0"]
    df = df[df["symbol"].str.startswith("sh.60") | df["symbol"].str.startswith("sz.00")]
    df = df.sort_values(["symbol", "date"])

    # 涨停基因: 近20个交易日有过涨停 (pctChg>=9.8)
    df["limit_up"] = (df["pctChg"] >= LU_PCT).astype(int)
    df["had_lu20"] = df.groupby("symbol")["limit_up"].transform(
        lambda s: s.rolling(LU_LOOKBACK, min_periods=1).max()
    ) > 0

    # 量比代理: 当日量 / MA5量
    df["vol_ratio"] = df["volume"] / df.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(VOL_LB, min_periods=1).mean()
    )

    # 流通市值代理: 成交额/换手率 (amount*100/turn, 元) → 亿
    df["float_mkt_yi"] = df["amount"] * 100.0 / df["turn"] / 1e8

    # 强于大盘 + 站稳分时均线代理
    df = df.merge(idx_pct.rename("idx_pct"), left_on="date", right_index=True, how="left")
    df["strong_close"] = df["close"] >= (df["high"] + df["low"]) / 2.0

    # 次日价格 (T+1)
    df["next_open"] = df.groupby("symbol")["open"].shift(-1)
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)

    # ── 信号 ──
    sig = (
        (df["pctChg"] >= PCT_LO) & (df["pctChg"] <= PCT_HI)
        & df["had_lu20"]
        & (df["vol_ratio"] >= 1.0)
        & (df["turn"] >= TURN_LO) & (df["turn"] <= TURN_HI)
        & (df["float_mkt_yi"] >= MKT_LO_YI) & (df["float_mkt_yi"] <= MKT_HI_YI)
        & (df["pctChg"] > df["idx_pct"])
        & df["strong_close"]
    )
    picks = df[sig & df["next_open"].notna() & (df["next_open"] > 0)].copy()
    if picks.empty:
        return {"n_signal_days": 0, "n_trades": 0, "win_rate": None,
                "avg_net_ret": None, "median_net_ret": None,
                "cum_net_ret_open": 0.0, "cum_net_ret_close": 0.0}

    # 隔夜收益 (T收盘 → T+1开盘/收盘), 净 = 毛 - 31bp
    picks["ret_open"] = picks["next_open"] / picks["close"] - 1.0
    picks["ret_close"] = picks["next_close"] / picks["close"] - 1.0
    cost = COST_BPS / 10000.0
    picks["net_open"] = picks["ret_open"] - cost
    picks["net_close"] = picks["ret_close"] - cost

    daily = picks.groupby("date").agg(
        n=("ret_open", "size"), ret_open=("net_open", "mean"), ret_close=("net_close", "mean")
    )

    win_rate = float((picks["net_open"] > 0).mean())
    return {
        "n_signal_days": int(len(daily)),
        "n_trades": int(len(picks)),
        "avg_per_day": round(float(daily["n"].mean()), 2),
        "win_rate": round(win_rate, 4),
        "avg_net_ret_open": round(float(picks["net_open"].mean()), 4),
        "median_net_ret_open": round(float(picks["net_open"].median()), 4),
        "avg_net_ret_close": round(float(picks["net_close"].mean()), 4),
        "cum_net_ret_open": round(float((1.0 + daily["ret_open"]).prod() - 1.0), 4),
        "cum_net_ret_close": round(float((1.0 + daily["ret_close"]).prod() - 1.0), 4),
    }


def _index_buyhold(idx_close: pd.Series, df: pd.DataFrame) -> float:
    T, T_end = df["date"].min(), df["date"].max()
    seg = idx_close[(idx_close.index >= T) & (idx_close.index <= T_end)]
    if len(seg) < 2:
        return float("nan")
    return float(seg.iloc[-1] / seg.iloc[0] - 1.0)


def main() -> int:
    _force_utf8()
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx = idx[idx["symbol"] == "sh.000001"][["date", "close", "pctChg"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date").sort_index()

    report = {"cost_bps": COST_BPS, "windows": [], "aggregate": {}}
    print("=" * 108)
    print("尾盘「涨停基因」超短线 — 日线近似回测 (T收盘买 / T+1开盘走, 成本31bp)")
    print("=" * 108)

    for label, fp in WINDOWS.items():
        df = pd.read_parquet(ROOT / fp)
        df["date"] = pd.to_datetime(df["date"])
        r = _backtest_window(df, idx["pctChg"])
        bh = _index_buyhold(idx["close"], df)
        r.update({"window": label, "regime": REGIME[label],
                  "sh_buyhold": round(bh, 4) if not np.isnan(bh) else None,
                  "vs_sh_open": round(r["cum_net_ret_open"] - bh, 4) if not np.isnan(bh) else None})
        report["windows"].append(r)

        print(f"\n### {label} [{REGIME[label]}]")
        if r["n_trades"] == 0:
            print("  无信号 (窗口内无满足全部条件的票)")
        else:
            print(f"  信号日 {r['n_signal_days']} | 交易 {r['n_trades']} 笔 | 日均 {r['avg_per_day']} 只 | 胜率 {r['win_rate']*100:.1f}%")
            print(f"  单笔均收益(净): T+1开盘 {r['avg_net_ret_open']*100:+.2f}% | 中位 {r['median_net_ret_open']*100:+.2f}% | T+1收盘 {r['avg_net_ret_close']*100:+.2f}%")
            print(f"  窗口累计(净): T+1开盘 {r['cum_net_ret_open']*100:+.1f}% | T+1收盘 {r['cum_net_ret_close']*100:+.1f}%")
            print(f"  上证买入持有 {r['sh_buyhold']*100:+.1f}% | 相对(开盘口径) {r['vs_sh_open']*100:+.1f}pp")

    # ── 汇总 ──
    ws = report["windows"]
    active = [w for w in ws if w["n_trades"] > 0]
    if active:
        agg = {
            "total_trades": sum(w["n_trades"] for w in active),
            "mean_win_rate": round(float(np.mean([w["win_rate"] for w in active])), 4),
            "mean_avg_net_ret_open": round(float(np.mean([w["avg_net_ret_open"] for w in active])), 4),
            "mean_avg_net_ret_close": round(float(np.mean([w["avg_net_ret_close"] for w in active])), 4),
            "pos_cum_open_windows": sum(1 for w in active if w["cum_net_ret_open"] > 0),
            "n_active_windows": len(active),
            "beat_sh_windows": sum(1 for w in active if w["vs_sh_open"] is not None and w["vs_sh_open"] > 0),
        }
        report["aggregate"] = agg
        print("\n" + "=" * 108)
        print("汇总 (日线近似, 仅供方向判断, 非真日内回测):")
        print(f"  活跃窗口 {agg['n_active_windows']}/{len(ws)} | 总交易 {agg['total_trades']} 笔")
        print(f"  平均胜率 {agg['mean_win_rate']*100:.1f}% | 平均单笔净收益(开盘口径) {agg['mean_avg_net_ret_open']*100:+.2f}%")
        print(f"  累计正收益窗口 {agg['pos_cum_open_windows']}/{agg['n_active_windows']} | 跑赢上证窗口 {agg['beat_sh_windows']}/{agg['n_active_windows']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
