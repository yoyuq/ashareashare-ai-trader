"""重建长窗口全市场日K (Baostock, 免代理) — 供多 regime 因子验证使用

现有 replay_data/daily_2026-02-21_2026-07-31.parquet 仅覆盖 ~110 个交易日,
大概率单一 regime。本脚本用 Baostock (免代理, 稳定) 重拉更长窗口
(默认 2025-10-08 ~ 2026-07-31, 约 200 个交易日), 覆盖牛熊转换。

复用 historical_replay.build_daily_data 的拉取/缓存逻辑 (dtype 优化, 含 peTTM/pbMRQ)。

用法:
  python scripts/build_long_replay.py                      # 默认 2025-10-08 ~ 2026-07-31
  python scripts/build_long_replay.py --start 2025-10-08 --end 2026-07-31
输出:
  replay_data/daily_{start}_{end}.parquet  (全市场, symbol=category, 数值=float32)
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.historical_replay import build_daily_data, load_snapshot_basic  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="重建长窗口全市场日K")
    ap.add_argument("--start", default="2025-10-08", help="开始日期 YYYY-MM-DD")
    ap.add_argument("--end", default="2026-07-31", help="结束日期 YYYY-MM-DD")
    args = ap.parse_args()

    basic, universe = load_snapshot_basic()
    if not universe:
        print("缺少 full_market_cache.json 快照, 无法确定股票池")
        return 1
    print(f"股票池: {len(universe)} 只 | 窗口 {args.start} ~ {args.end}")

    y0, m0, d0 = map(int, args.start.split("-"))
    y1, m1, d1 = map(int, args.end.split("-"))
    build_daily_data(universe, date(y0, m0, d0), date(y1, m1, d1))
    print("长窗口数据构建完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
