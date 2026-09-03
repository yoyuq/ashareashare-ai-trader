"""缓存空壳引导 — 用 Baostock 全市场股票列表 (免代理, 真实数据) 重建 full_market_cache.

场景: full_market_cache.json 为空壳 (count=0) 时 refresh_market_cache.py 死锁
(它从旧缓存读列表)。本脚本用 Baostock query_stock_basic 取全部上市 A 股代码,
复用 refresh_market_cache.fetch_tencent 拉腾讯实时 → 写缓存。零模拟: 列表与行情均为真实源。

用法: python scripts/bootstrap_market_cache.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import baostock as bs
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.full_market_cache import write_full_market_cache  # noqa: E402
from refresh_market_cache import fetch_tencent  # noqa: E402
from timeutil import last_trade_date  # noqa: E402


def main() -> int:
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    try:
        day = last_trade_date().isoformat()
        rs = bs.query_all_stock(day)
        if rs.error_code != "0":
            raise RuntimeError(f"query_all_stock 失败: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()
    # 只留 A 股主板/创业板/科创板 (sh.6/sz.0/sz.3), 剔除指数 (sh.000) 与北交所 (本策略范围外)
    a = df[df["code"].str.match(r"(sh\.6|sz\.0|sz\.3)")]
    syms = a["code"].tolist()
    if len(syms) < 1000:
        raise RuntimeError(f"股票列表异常 ({len(syms)} 只), 中止不写缓存")
    print(f"Baostock 列表: {len(syms)} 只 A 股 (基准日 {day})")

    t0 = time.time()
    quotes = fetch_tencent(syms)
    print(f"腾讯拉取: {len(quotes)}/{len(syms)} 只, 耗时 {time.time()-t0:.0f}s")
    new_data = [q for q in quotes.values() if q["price"] > 0]
    if len(new_data) < 1000:
        raise RuntimeError(f"有效行情异常 ({len(new_data)} 只), 中止不写缓存")
    write_full_market_cache(pd.DataFrame(new_data), day, "tencent_realtime")
    print(f"缓存已重建: {day} | {len(new_data)} 只")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
