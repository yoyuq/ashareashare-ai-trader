"""打板/首板转二板 (一进二) + 首板低开分歧 — 日线忠实回测 (预注册 §十一).

背景: 之前尾盘超短(涨3-5%活跃票)0/11 证伪, 根因=活跃票隔夜失血。这次换「打板/竞价型」:
首板(T-1 涨停) → 次日(T)开盘买, 回避隔夜腿。选股逻辑日线可忠实测(涨停/高开/成交额都在日线里),
盘中执行细节另行用东财历史分钟线单窗口测。

信号 (预注册 §十一):
  首板 = T-1 pctChg>=9.8 且 T-2 pctChg<9.8 (主板/非ST/可交易)
  A 一进二(高开接力):  gap = open/close[T-1]-1 ∈ (0, +6%]
  B 首板低开分歧:      gap ∈ [-6%, 0)
买 T 开盘(9:30 近似 9:25 竞价, 披露); 卖三口径: T收盘(主) / T+1开盘(隔夜) / T最高(完美上界)。

判据 (预注册, 跑完不调): 主口径 = T收盘卖出单笔均净 > 0 且 >=7/11 窗口单笔均净为正。
用法: python scripts/run_first_board_bt.py
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
LU_PCT = 9.8          # 主板涨停阈值 (10% 限制, 留 0.2 容差)
GAP_LO, GAP_HI = -0.06, 0.06  # 高开/低开幅度区间

OUT = ROOT / "reports" / "first_board_bt_result.json"

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


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["isST"] = df["isST"].astype(str)
    df = df[df["is_trade"] == 1]
    df = df[df["isST"] == "0"]
    df = df[df["symbol"].str.startswith("sh.60") | df["symbol"].str.startswith("sz.00")]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = df.groupby("symbol", sort=False)
    df["prev_pct"] = g["pctChg"].shift(1)
    df["prev2_pct"] = g["pctChg"].shift(2)
    df["prev_close"] = g["close"].shift(1)
    df["next_open"] = g["open"].shift(-1)

    df["first_board"] = (df["prev_pct"] >= LU_PCT) & (df["prev2_pct"] < LU_PCT)
    df["gap"] = df["open"] / df["prev_close"] - 1.0
    df = df[df["prev_close"].notna() & (df["prev_close"] > 0)]
    return df


def _stat(picks: pd.DataFrame) -> dict:
    if picks.empty:
        return {"n_trades": 0, "win_rate": None, "avg_close": None,
                "avg_next_open": None, "avg_high": None, "median_close": None}
    cost = COST_BPS / 10000.0
    net_close = picks["close"] / picks["open"] - 1.0 - cost
    net_next = picks["next_open"] / picks["open"] - 1.0 - cost
    net_high = picks["high"] / picks["open"] - 1.0 - cost
    return {
        "n_trades": int(len(picks)),
        "win_rate": round(float((net_close > 0).mean()), 4),
        "avg_close": round(float(net_close.mean()), 4),
        "avg_next_open": round(float(net_next.mean()), 4),
        "avg_high": round(float(net_high.mean()), 4),
        "median_close": round(float(net_close.median()), 4),
    }


def main() -> int:
    _force_utf8()
    report = {"cost_bps": COST_BPS, "windows": {}, "A_一进二": {}, "B_首板低开分歧": {}}

    print("=" * 108)
    print("打板/首板转二板 (一进二) + 首板低开分歧 — 日线忠实回测 (买 T 开盘, 成本 31bp)")
    print("=" * 108)

    for label, fp in WINDOWS.items():
        df = pd.read_parquet(ROOT / fp)
        df = _prep(df)

        sigA = df["first_board"] & (df["gap"] > 0) & (df["gap"] <= GAP_HI)
        sigB = df["first_board"] & (df["gap"] >= GAP_LO) & (df["gap"] < 0)

        pickA = df[sigA & df["next_open"].notna() & (df["next_open"] > 0)].copy()
        pickB = df[sigB & df["next_open"].notna() & (df["next_open"] > 0)].copy()

        sA, sB = _stat(pickA), _stat(pickB)
        report["windows"][label] = {"A": sA, "B": sB, "n_first_board": int(df["first_board"].sum())}

        print(f"\n### {label}")
        print(f"  首板信号日数 {report['windows'][label]['n_first_board']} 个 (T-1 涨停且 T-2 未涨停)")
        print(f"  [A 一进二 高开接力] 交易 {sA['n_trades']} 笔 | 胜率(T收盘) {sA['win_rate']*100 if sA['win_rate'] is not None else 0:.1f}%")
        print(f"    单笔均净: T收盘 {sA['avg_close']*100 if sA['avg_close'] is not None else 0:+.2f}% | "
              f"T+1开盘 {sA['avg_next_open']*100 if sA['avg_next_open'] is not None else 0:+.2f}% | "
              f"T最高(上界) {sA['avg_high']*100 if sA['avg_high'] is not None else 0:+.2f}%")
        print(f"  [B 首板低开分歧]   交易 {sB['n_trades']} 笔 | 胜率(T收盘) {sB['win_rate']*100 if sB['win_rate'] is not None else 0:.1f}%")
        print(f"    单笔均净: T收盘 {sB['avg_close']*100 if sB['avg_close'] is not None else 0:+.2f}% | "
              f"T+1开盘 {sB['avg_next_open']*100 if sB['avg_next_open'] is not None else 0:+.2f}% | "
              f"T最高(上界) {sB['avg_high']*100 if sB['avg_high'] is not None else 0:+.2f}%")

    # 汇总 + 判据
    def _agg(arm: str, key: str):
        vals = [w[arm][key] for w in report["windows"].values()]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    def _pos_win(arm: str, key: str):
        return sum(1 for w in report["windows"].values()
                   if w[arm][key] is not None and w[arm][key] > 0)

    for arm in ("A", "B"):
        name = "A_一进二" if arm == "A" else "B_首板低开分歧"
        agg = {
            "total_trades": sum(w[arm]["n_trades"] for w in report["windows"].values()),
            "mean_win_rate": _agg(arm, "win_rate"),
            "mean_avg_close": _agg(arm, "avg_close"),
            "mean_avg_next_open": _agg(arm, "avg_next_open"),
            "mean_avg_high": _agg(arm, "avg_high"),
            "pos_close_windows": _pos_win(arm, "avg_close"),
            "n_active_windows": sum(1 for w in report["windows"].values() if w[arm]["n_trades"] > 0),
        }
        report[name] = agg
        print(f"\n[{name}] 汇总")
        print(f"  总交易 {agg['total_trades']} 笔 | 活跃窗口 {agg['n_active_windows']}/11")
        print(f"  平均胜率(T收盘) {agg['mean_win_rate']*100 if agg['mean_win_rate'] is not None else 0:.1f}%")
        print(f"  平均单笔净: T收盘 {agg['mean_avg_close']*100 if agg['mean_avg_close'] is not None else 0:+.2f}% | "
              f"T+1开盘 {agg['mean_avg_next_open']*100 if agg['mean_avg_next_open'] is not None else 0:+.2f}% | "
              f"T最高 {agg['mean_avg_high']*100 if agg['mean_avg_high'] is not None else 0:+.2f}%")
        print(f"  判据(主=T收盘): 均净 {agg['mean_avg_close']*100 if agg['mean_avg_close'] is not None else 0:+.2f}% "
              f"{'>0' if (agg['mean_avg_close'] or 0) > 0 else '<=0'} | 正窗口 {agg['pos_close_windows']}/11 "
              f"{'PASS(>=7/11)' if agg['pos_close_windows'] >= 7 else 'FAIL'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
