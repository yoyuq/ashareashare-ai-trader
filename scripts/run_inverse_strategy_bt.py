"""反策略回测 (日线近似): 尾盘买「弱势票」次日低开企稳走 (隔夜反转).

假设: 原「涨停基因强票」策略证伪, 根因是「强势票隔夜反转」—— 涨3-5%+高换手+放量的票
次日早盘被闷。反向假设: 弱势票 (放量下跌/冲高回落) 尾盘被砸, 次日企稳回升, 有正隔夜反转溢价。

逐条镜像 (相对 scripts/run_user_strategy_bt.py):
  原: 涨幅3~5% + 涨停基因 + 强于大盘 + 站稳分时均线(收盘上半区)
  反: 跌幅(<0, 排除跌停) + (去涨停基因) + 弱于大盘 + 收在低位(收盘下半区)
  保留: 主板/换手5-10%/市值50-200亿/量比≥1 (可交易性结构, 非方向)

卖出「次日低开企稳走」: 报两个口径
  - T+1 开盘 = 「低开」点 (预期偏负, 是砸出来的坑)
  - T+1 收盘 = 「企稳回升后走」 (这条策略真正兑现的价格)

成本 31bp 全周转; 日线近似, 分时均价线/精确回踩买点/盘中企稳点不可测 (诚实披露)。
用法: python scripts/run_inverse_strategy_bt.py
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
TURN_LO, TURN_HI = 5.0, 10.0
MKT_LO_YI, MKT_HI_YI = 50.0, 200.0
LIMIT_DOWN_PCT = -9.8   # 主板跌停阈值 (排除破位/封死跌停)
VOL_LB = 5

OUT = ROOT / "reports" / "inverse_strategy_bt_result.json"

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
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["isST"] = df["isST"].astype(str)
    df = df[df["is_trade"] == 1]
    df = df[df["isST"] == "0"]
    df = df[df["symbol"].str.startswith("sh.60") | df["symbol"].str.startswith("sz.00")]
    df = df.sort_values(["symbol", "date"])

    # 量比代理 + 流通市值代理
    df["vol_ratio"] = df["volume"] / df.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(VOL_LB, min_periods=1).mean()
    )
    df["float_mkt_yi"] = df["amount"] * 100.0 / df["turn"] / 1e8

    df = df.merge(idx_pct.rename("idx_pct"), left_on="date", right_index=True, how="left")
    df["weak_close"] = df["close"] < (df["high"] + df["low"]) / 2.0  # 收在低位 (分时均线下方)

    df["next_open"] = df.groupby("symbol")["open"].shift(-1)
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)
    df["next_high"] = df.groupby("symbol")["high"].shift(-1)   # 完美卖点上界 (T+1 日内最高)

    # ── 反策略信号: 弱势票 ──
    sig = (
        (df["pctChg"] < 0.0) & (df["pctChg"] > LIMIT_DOWN_PCT)   # 收跌, 排除跌停
        & (df["vol_ratio"] >= 1.0)                                 # 放量
        & (df["turn"] >= TURN_LO) & (df["turn"] <= TURN_HI)        # 换手 5-10%
        & (df["float_mkt_yi"] >= MKT_LO_YI) & (df["float_mkt_yi"] <= MKT_HI_YI)
        & (df["pctChg"] < df["idx_pct"])                           # 弱于大盘
        & df["weak_close"]                                         # 收在低位
    )
    picks = df[sig & df["next_open"].notna() & (df["next_open"] > 0)].copy()
    if picks.empty:
        return {"n_signal_days": 0, "n_trades": 0, "win_rate": None,
                "avg_net_ret": None, "median_net_ret": None,
                "cum_net_ret_open": 0.0, "cum_net_ret_close": 0.0}

    picks["ret_open"] = picks["next_open"] / picks["close"] - 1.0
    picks["ret_close"] = picks["next_close"] / picks["close"] - 1.0
    cost = COST_BPS / 10000.0
    picks["net_open"] = picks["ret_open"] - cost
    picks["net_close"] = picks["ret_close"] - cost
    picks["net_high"] = picks["next_high"] / picks["close"] - 1.0 - cost   # 完美卖点(日内最高)上界

    daily = picks.groupby("date").agg(
        n=("ret_open", "size"), ret_open=("net_open", "mean"), ret_close=("net_close", "mean")
    )

    win_rate = float((picks["net_close"] > 0).mean())   # 企稳走口径的胜率
    return {
        "n_signal_days": int(len(daily)),
        "n_trades": int(len(picks)),
        "avg_per_day": round(float(daily["n"].mean()), 2),
        "win_rate": round(win_rate, 4),
        "avg_net_ret_open": round(float(picks["net_open"].mean()), 4),
        "median_net_ret_open": round(float(picks["net_open"].median()), 4),
        "avg_net_ret_close": round(float(picks["net_close"].mean()), 4),
        "median_net_ret_close": round(float(picks["net_close"].median()), 4),
        "avg_net_ret_high": round(float(picks["net_high"].mean()), 4),
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
    print("=" * 112)
    print("反策略: 尾盘买「弱势票」次日低开企稳走 — 日线近似 (T收盘买 / 成本31bp)")
    print("=" * 112)

    for label, fp in WINDOWS.items():
        df = pd.read_parquet(ROOT / fp)
        df["date"] = pd.to_datetime(df["date"])
        r = _backtest_window(df, idx["pctChg"])
        bh = _index_buyhold(idx["close"], df)
        r.update({"window": label, "regime": REGIME[label],
                  "sh_buyhold": round(bh, 4) if not np.isnan(bh) else None,
                  "vs_sh_close": round(r["cum_net_ret_close"] - bh, 4) if not np.isnan(bh) else None})
        report["windows"].append(r)

        print(f"\n### {label} [{REGIME[label]}]")
        if r["n_trades"] == 0:
            print("  无信号")
        else:
            print(f"  信号日 {r['n_signal_days']} | 交易 {r['n_trades']} 笔 | 日均 {r['avg_per_day']} 只 | 胜率(收盘口径) {r['win_rate']*100:.1f}%")
            print(f"  单笔均收益(净): 低开(T+1开盘) {r['avg_net_ret_open']*100:+.2f}% | 企稳(T+1收盘) {r['avg_net_ret_close']*100:+.2f}% | 完美卖点(T+1最高) {r['avg_net_ret_high']*100:+.2f}%")
            print(f"  中位(净):          低开 {r['median_net_ret_open']*100:+.2f}% | 企稳 {r['median_net_ret_close']*100:+.2f}%")
            print(f"  窗口累计(净): T+1开盘 {r['cum_net_ret_open']*100:+.1f}% | T+1收盘 {r['cum_net_ret_close']*100:+.1f}%")
            print(f"  上证买入持有 {r['sh_buyhold']*100:+.1f}% | 相对(收盘口径) {r['vs_sh_close']*100:+.1f}pp")

    ws = report["windows"]
    active = [w for w in ws if w["n_trades"] > 0]
    if active:
        agg = {
            "total_trades": sum(w["n_trades"] for w in active),
            "mean_win_rate": round(float(np.mean([w["win_rate"] for w in active])), 4),
            "mean_avg_net_ret_open": round(float(np.mean([w["avg_net_ret_open"] for w in active])), 4),
            "mean_avg_net_ret_close": round(float(np.mean([w["avg_net_ret_close"] for w in active])), 4),
            "pos_cum_close_windows": sum(1 for w in active if w["cum_net_ret_close"] > 0),
            "n_active_windows": len(active),
            "beat_sh_windows": sum(1 for w in active if w["vs_sh_close"] is not None and w["vs_sh_close"] > 0),
        }
        report["aggregate"] = agg
        print("\n" + "=" * 112)
        print("汇总 (日线近似, 仅供方向判断):")
        print(f"  活跃窗口 {agg['n_active_windows']}/{len(ws)} | 总交易 {agg['total_trades']} 笔")
        print(f"  平均胜率(收盘口径) {agg['mean_win_rate']*100:.1f}%")
        print(f"  平均单笔净收益: 低开 {agg['mean_avg_net_ret_open']*100:+.2f}% | 企稳 {agg['mean_avg_net_ret_close']*100:+.2f}%")
        print(f"  累计正收益窗口(收盘口径) {agg['pos_cum_close_windows']}/{agg['n_active_windows']} | 跑赢上证 {agg['beat_sh_windows']}/{agg['n_active_windows']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
