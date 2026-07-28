#!/usr/bin/env python3
"""
独立调度器运行脚本 (v2.10)

后台运行, 定时执行:
  - 9:25  早盘买入
  - 15:07 盘后分析
  - 15:12 盘后卖出检查

使用:
    python scripts/run_scheduler.py           # 前台运行
    python scripts/run_scheduler.py --once    # 立即执行一次完整流程后退出

Windows 后台运行 (开机自启):
    将 run_scheduler.bat 放入 shell:startup 文件夹
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from scripts.scheduler import TaskScheduler


async def run_once():
    """立即执行一次完整流程"""
    from scripts.run_daily_analysis import DailyPipeline
    from scripts.shared import load_watchlist
    from scripts.evening_sell import run_evening_sell
    from scripts.morning_buy import run_morning_buy

    now = datetime.now()
    hour = now.hour

    if 9 <= hour < 15:
        # 盘中 → 只做早盘买入
        logger.info("盘中模式: 执行早盘买入")
        await run_morning_buy(min_confidence=0.45)
    else:
        # 盘后 → 分析 + 卖出检查
        logger.info("盘后模式: 执行分析+卖出检查")
        symbols = load_watchlist("all_industries")
        pipeline = DailyPipeline(symbols, enable_notify=False, force_no_llm=True)
        await pipeline.run()
        await run_evening_sell(dry_run=False)


async def main():
    parser = argparse.ArgumentParser(description="A股自动交易调度器 v2.10")
    parser.add_argument("--once", action="store_true", help="立即执行一次后退出")
    args = parser.parse_args()

    if args.once:
        await run_once()
        return

    scheduler = TaskScheduler()

    # 注册任务
    async def _daily_analysis():
        from scripts.run_daily_analysis import DailyPipeline
        from scripts.shared import load_watchlist
        logger.info("=" * 50)
        logger.info("盘后自动分析")
        symbols = load_watchlist("all_industries")
        pipeline = DailyPipeline(symbols, enable_notify=False, force_no_llm=True)
        await pipeline.run()

    async def _evening_sell():
        from scripts.evening_sell import run_evening_sell
        logger.info("=" * 50)
        logger.info("盘后卖出检查")
        await run_evening_sell(dry_run=False)

    async def _morning_buy():
        from scripts.morning_buy import run_morning_buy
        logger.info("=" * 50)
        logger.info("早盘自动买入")
        await run_morning_buy(min_confidence=0.45)

    scheduler.add_daily("盘后分析", _daily_analysis, hour=15, minute=7)
    scheduler.add_daily("盘后卖出", _evening_sell, hour=15, minute=12)
    scheduler.add_daily("早盘买入", _morning_buy, hour=9, minute=25)

    logger.info("调度器启动, 等待触发...")
    logger.info("  盘后分析: 15:07 (交易日)")
    logger.info("  盘后卖出: 15:12")
    logger.info("  早盘买入: 09:25")
    logger.info("  Ctrl+C 停止")

    scheduler.start()
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("收到停止信号")
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
