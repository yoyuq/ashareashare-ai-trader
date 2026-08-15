"""串行补齐 2015/2016/2017/2021/2022/2023 全市场日K (Baostock, 免代理).

背景: 冷落 beta 已证 5/5 窗口(2018熊/2019牛/2020牛转崩/2024震荡/2025-26)跑赢上证,
但缺 2015-16股灾尾部 / 2017漂亮50大盘独涨 / 2021-23 三个 regime 段, 无法回答
"beta 在全部市场情况是否都有用"。本脚本补齐这 6 个年度窗口, 全程真实数据零模拟。

Baostock 对并发查询敏感 (脚本注释: 快速拉取触发黑名单 10001011), 故**串行**抓取,
任一时刻至多一个 fetch 进程, 节流 0.08s/查询 + 每200只重连, 断点续跑。

优化: 用 query_stock_basic 拿全市场 ipoDate, 按窗口起始日过滤「尚未上市」的票, 免无谓查询
(IPO 前无日K)。这与回测本身一致 —— run_cold_tilt_rebalance.py 的 _liquid_universe 后续
会把「窗口首日无数据」的票剔除, 故此过滤不改变回测结果, 纯省时。缺 ipoDate 的 symbol 一律保留。

用法: python scripts/fetch_full_cycle_windows.py   (后台运行, 约 5-7 小时)
输出: replay_data/daily_{YYYY-01-01}_{YYYY-12-31}.parquet (每年一个)
"""

import socket
import sys
import time
from datetime import date
from pathlib import Path

# 防 Baostock 单次查询网络挂起: 30s 超时 → 抛异常 → robust_fetch 按 symbol 捕获跳过
socket.setdefaulttimeout(30)

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_long_replay import robust_fetch  # noqa: E402
from scripts.historical_replay import load_snapshot_basic  # noqa: E402

_WINDOWS = [
    ("2015", "2015-01-01", "2015-12-31"),  # 疯牛 + 股灾流动性尾部
    ("2016", "2016-01-01", "2016-12-31"),  # 熔断 + 震荡
    ("2017", "2017-01-01", "2017-12-31"),  # 漂亮50 大盘独涨 (小盘逆向年)
    ("2021", "2021-01-01", "2021-12-31"),  # 核心资产白马 → 小盘切换
    ("2022", "2022-01-01", "2022-12-31"),  # 熊市
    ("2023", "2023-01-01", "2023-12-31"),  # 震荡
]


def _fetch_ipo_map() -> dict:
    """query_stock_basic 一次拉取全市场上市日期 → {code: 'YYYY-MM-DD'}。

    用于按窗口过滤「当时尚未上市」的票。Baostock 只覆盖沪深 (bj 北交所可能缺失),
    缺 ipoDate 的 symbol 一律保留 (不误杀, 拉回空数据由 robust_fetch 自然跳过)。
    """
    import baostock as bs

    ipo: dict = {}
    try:
        lg = bs.login()
        if lg.error_code != "0":
            print(f"[ipo] login 失败 {lg.error_code}, 不启用按窗口过滤", flush=True)
            return ipo
        rs = bs.query_stock_basic()
        while rs.error_code == "0" and rs.next():
            d = dict(zip(rs.fields, rs.get_row_data()))
            if d.get("ipoDate"):
                ipo[d["code"]] = d["ipoDate"]
        try:
            bs.logout()
        except Exception:
            pass
    except Exception as e:
        print(f"[ipo] query_stock_basic 异常 {e}, 不启用按窗口过滤", flush=True)
    return ipo


def _is_complete(out: Path, universe: list) -> bool:
    if not out.exists():
        return False
    try:
        import pandas as pd
        prev = pd.read_parquet(out)
        return prev["symbol"].nunique() >= len(universe) * 0.95
    except Exception:
        return False


def main():
    basic, universe = load_snapshot_basic()
    if not universe:
        print("缺少 full_market_cache.json 快照, 无法确定股票池")
        return 1
    print(f"股票池: {len(universe)} 只 (来自最近交易日快照)")

    ipo_map = _fetch_ipo_map()
    print(f"ipoDate 映射: {len(ipo_map)} 只 (缺日期者按保守保留)", flush=True)

    for label, start, end in _WINDOWS:
        # 按窗口过滤: 只抓「窗口起始日前已上市」的票。与回测的「首日无数据即剔除」一致, 不改结果。
        uni = [s for s in universe if ipo_map.get(s) is None or ipo_map[s] <= start]
        print(f"[{label}] 上市早于 {start}: {len(uni)}/{len(universe)} 只", flush=True)

        out = Path(f"replay_data/daily_{start}_{end}.parquet")
        if _is_complete(out, uni):
            print(f"[{label}] 跳过: {out} 已完整")
            continue
        if out.exists():
            print(f"[{label}] 续传: {out} 已存在但不完整")

        y0, m0, d0 = map(int, start.split("-"))
        y1, m1, d1 = map(int, end.split("-"))
        print(f"\n=== [{label}] 抓取 {start} ~ {end} ===")
        robust_fetch(uni, date(y0, m0, d0), date(y1, m1, d1), out)
        # 抓完一个窗口后短暂冷却, 降低黑名单风险
        time.sleep(5)

    print("\n全部 6 个窗口抓取完成.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
