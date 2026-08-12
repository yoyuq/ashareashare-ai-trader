"""顺序补齐 2019/2024 数据窗口 (等 2018 抓取完成后再跑, 避免并发触发 Baostock 黑名单).

Baostock 对并发查询敏感 (脚本注释: 上次 830 只快速拉取触发黑名单 10001011).
因此三个年度必须**串行**抓取: 本脚本轮询等待 2018 parquet 落地 → 抓 2019 → 抓 2024.
任一时刻至多一个 fetch 进程在跑, 查询速率恒定.

用法:
  python scripts/fetch_remaining_windows.py            # 后台运行
输出:
  replay_data/daily_2019-01-01_2019-12-31.parquet
  replay_data/daily_2024-01-01_2024-12-31.parquet
"""

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_long_replay import robust_fetch  # noqa: E402
from scripts.historical_replay import load_snapshot_basic  # noqa: E402

_2018 = Path("replay_data/daily_2018-01-01_2018-12-31.parquet")
_WINDOWS = [
    ("2019-01-01", "2019-12-31"),
    ("2024-01-01", "2024-12-31"),
]
_POLL = 60  # 秒


def main():
    basic, universe = load_snapshot_basic()
    if not universe:
        print("缺少股票池快照")
        return 1

    # 等 2018 完成 (外层已单独跑). 最多等 3 小时.
    waited = 0
    while not _2018.exists():
        print(f"[等2018] {waited//60}min 尚未落地, 再等 {_POLL}s...")
        time.sleep(_POLL)
        waited += _POLL
        if waited > 3 * 3600:
            print("超时: 2018 未完成, 不再阻塞, 直接抓后续窗口")
            break

    for start, end in _WINDOWS:
        out = Path(f"replay_data/daily_{start}_{end}.parquet")
        if _is_complete(out, universe):
            print(f"跳过: {out} 已完整 ({len(universe)} 只)")
            continue
        if out.exists():
            print(f"续传: {out} 已存在但不完整 — robust_fetch 将补抓剩余")
        y0, m0, d0 = map(int, start.split("-"))
        y1, m1, d1 = map(int, end.split("-"))
        print(f"\n=== 抓取 {start} ~ {end} ===")
        robust_fetch(universe, date(y0, m0, d0), date(y1, m1, d1), out)
    print("\n全部窗口抓取完成.")
    return 0


def _is_complete(out: Path, universe: list) -> bool:
    """parquet 已存在且 symbol 数 >= universe 95% 才算完整 (防半途文件被误跳过)."""
    if not out.exists():
        return False
    try:
        import pandas as pd
        prev = pd.read_parquet(out)
        n = prev["symbol"].nunique()
        return n >= len(universe) * 0.95
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())