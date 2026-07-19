"""
历史数据批量下载脚本

用途:
  批量下载全市场10年+日K线历史数据,写入PostgreSQL
  支持断点续传(已有数据跳过)

运行:
  python scripts/download_history.py --start 2015-01-01 --pool default
  python scripts/download_history.py --start 2020-01-01 --pool small_cap --workers 4
"""

import asyncio
import sys
from argparse import ArgumentParser
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data.router import get_data_router
from data.providers.base import DataFrequency, DataRequest


async def download_stock_history(
    symbol: str,
    start_date: date,
    end_date: date,
    router,
    semaphore: asyncio.Semaphore,
) -> bool:
    """下载单只股票历史K线"""
    async with semaphore:
        try:
            request = DataRequest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=DataFrequency.DAILY,
                adjust="qfq",
            )
            result = await router.get_daily_kline(request)
            return result is not None and not result.data.empty
        except Exception as e:
            logger.debug(f"下载失败 {symbol}: {e}")
            return False


async def download_all(
    symbols: List[str],
    start_date: date,
    end_date: date,
    max_concurrent: int = 3,
):
    """批量下载(带并发控制)"""
    router = get_data_router(cross_validation=False)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for sym in symbols:
        tasks.append(
            download_stock_history(sym, start_date, end_date, router, semaphore)
        )

    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="下载历史K线",
    ):
        result = await coro
        results.append(result)

    success = sum(results)
    logger.info(f"下载完成: {success}/{len(symbols)} 成功")


def get_symbol_pool(pool_name: str) -> List[str]:
    """
    获取股票池标的列表

    优先从数据库/缓存读取,否则实时获取
    """
    import yaml

    config_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pools = config.get("stock_pools", {})

    if pool_name == "all":
        # 全市场: 通过AKShare获取
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        return [f"sh.{c}" if c.startswith("6") else f"sz.{c}"
                for c in df["代码"].tolist()]

    if pool_name in pools:
        pool = pools[pool_name]
        if "symbols" in pool:
            return pool["symbols"]
        # 从指数成分股获取
        symbols = []
        for idx_code in pool.get("source", []):
            idx_symbols = _get_index_constituents(idx_code)
            symbols.extend(idx_symbols)
        return list(set(symbols))

    # 默认: 通用池
    return _get_index_constituents("hs300") + _get_index_constituents("csi500")


def _get_index_constituents(index_id: str) -> List[str]:
    """获取指数成分股列表"""
    import akshare as ak

    index_map = {
        "hs300": "000300",
        "csi500": "000905",
        "csi1000": "000852",
        "chinext": "399006",
    }
    code = index_map.get(index_id, index_id)

    try:
        df = ak.index_stock_cons_weight_csindex(symbol=code)
        if df.empty:
            return []
        return [
            f"sh.{c}" if c.startswith("6") else f"sz.{c}"
            for c in df["成分券代码"].tolist()
        ]
    except Exception:
        return []


def main():
    parser = ArgumentParser(description="批量下载A股历史K线数据")
    parser.add_argument(
        "--start", type=str, default="2015-01-01",
        help="起始日期 (默认: 2015-01-01)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="结束日期 (默认: 今天)"
    )
    parser.add_argument(
        "--pool", type=str, default="default",
        choices=["default", "small_cap", "limit_up", "etf", "all"],
        help="股票池 (默认: default)"
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="并发下载数 (默认: 3, 注意API限流)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只列出股票,不下载"
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = (
        datetime.strptime(args.end, "%Y-%m-%d").date()
        if args.end else date.today()
    )

    # 获取股票列表
    symbols = get_symbol_pool(args.pool)
    logger.info(f"股票池 [{args.pool}]: {len(symbols)} 只股票")

    if args.dry_run:
        for i, s in enumerate(symbols[:20]):
            logger.info(f"  {i+1}. {s}")
        if len(symbols) > 20:
            logger.info(f"  ... 及其他 {len(symbols)-20} 只")
        return

    # 批量下载
    asyncio.run(download_all(symbols, start_date, end_date, args.workers))


if __name__ == "__main__":
    main()
