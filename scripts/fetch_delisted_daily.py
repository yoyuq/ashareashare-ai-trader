"""抓取退市股全量日K (Baostock, 免代理) — 修复幸存者偏差 + 暴雷潮测试。

背景: 冷落 beta 5/5 跑赢上证, 但 replay_data 各窗口由 load_snapshot_basic() 的「当前股票池」
抓取, 只含 status=1 (存活) 的票, 1178 只退市股 (status=0) 全部缺失 → 幸存者偏差。
尤其 2025(200只)/2026H1(144只) 退市潮正是冷落票「低换手+快死」高发区, 冷落溢价可能被高估。

本脚本抓取 outDate>=2018-01-01 的退市股在 2018-01-01~2026-07-31 的日K (与窗口同口径: 不复权,
含 turn/isST/peTTM/pbMRQ), 落成面板供后续「补回退市股 → 重测冷落 tilt + 排雷」使用。

用法: python scripts/fetch_delisted_daily.py          # 全量 (~978 只, 约 5-10 分钟)
输出: replay_data/delisted_daily.parquet
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_long_replay import robust_fetch  # noqa: E402

ROOT = Path(__file__).parent.parent
DELIST = ROOT / "replay_data" / "delisted_stocks.parquet"
OUT = ROOT / "replay_data" / "delisted_daily.parquet"
START = date(2018, 1, 1)
END = date(2026, 7, 31)


def main() -> int:
    if not DELIST.exists():
        print("缺少 replay_data/delisted_stocks.parquet (先跑退市股名单抓取)")
        return 1
    d = pd.read_parquet(DELIST)
    # 仍在 2018-2026 期间交易过的退市股: outDate >= 2018-01-01 且 ipoDate <= 2026-07-31
    d = d.copy()
    d["outDate"] = pd.to_datetime(d["outDate"], errors="coerce")
    d["ipoDate"] = pd.to_datetime(d["ipoDate"], errors="coerce")
    mask = (d["outDate"].isna() | (d["outDate"] >= pd.Timestamp("2018-01-01"))) & \
           (d["ipoDate"].isna() | (d["ipoDate"] <= pd.Timestamp("2026-07-31")))
    d = d[mask]
    symbols = sorted(d["code"].astype(str).unique())
    print(f"退市股总数 {len(d)} | 2018 后仍交易的 {len(symbols)} 只 | 窗口 {START}~{END}")
    if not symbols:
        print("无待抓取退市股")
        return 1
    robust_fetch(symbols, START, END, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
