"""抓取指数基准日线序列 (Baostock, 免代理) — 作为"跑赢市场"的市值加权真实基准。

理论背景 (路线2 结构性风险溢价): 之前所有 A/B 的基线是"top-800 流动性等权", 这已经吃掉小盘溢价。
要回答"跑赢市场" (而非"跑赢等权"), 需要市值加权指数作基准。本脚本抓:
  sh.000001 上证综指  — 最常被引用的"市场"
  sh.000300 沪深300   — 大盘核心资产基准

全区间一次拉取 (2 次查询, 无节流风险), 后续按窗口切片。

用法: python scripts/fetch_index_series.py
输出: replay_data/index_series.parquet (symbol=sh.000001/sh.000300)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scripts.historical_replay import _bs_query, _optimize_dtypes  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "replay_data" / "index_series.parquet"
INDEXES = ["sh.000001", "sh.000300"]


def main() -> int:
    import baostock as bs

    bs.login()
    try:
        frames = []
        for sym in INDEXES:
            df = _bs_query(sym, "2015-01-01", "2026-07-31")
            if df.empty:
                print(f"[{sym}] 未拉到数据!")
                continue
            df["symbol"] = sym
            frames.append(df)
            print(f"[{sym}] {len(df)} 行 | {df['date'].min()} -> {df['date'].max()}")
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    if not frames:
        print("未拉到任何指数数据!")
        return 1
    big = pd.concat(frames, ignore_index=True)
    big = _optimize_dtypes(big)
    big.to_parquet(OUT, index=False)
    print(f"完成: {big['symbol'].nunique()} 只指数 | {len(big)} 行 → {OUT}")
    # 快速校验各窗口覆盖率
    for sym in INDEXES:
        s = big[big.symbol == sym].set_index("date")["close"]
        print(f"  {sym}: {len(s)} 行 | {s.index.min().date()} -> {s.index.max().date()} | "
              f"首收 {s.iloc[0]:.0f} 末收 {s.iloc[-1]:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
