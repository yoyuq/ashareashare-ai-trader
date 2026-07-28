"""
每日自动化完整流程 — Daily Runner

一键执行: 盘后分析 → 早盘买入 → 收盘总结

使用:
    python -m simulation.daily_runner           # 完整流程 (先分析,再买入,再总结)
    python -m simulation.daily_runner --reset   # 重置账户后重新开始
    python -m simulation.daily_runner --no-llm  # 规则引擎模式
"""

import argparse
import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


async def run_full_day(
    pool: str = "default",
    no_llm: bool = False,
    min_confidence: float = 0.50,
    reset: bool = False,
    skip_analyze: bool = False,
) -> dict:
    """完整的每日交易流程"""

    today = date.today().isoformat()
    logger.info(f"========== Daily Runner: {today} ==========")

    if reset:
        from simulation.portfolio import PortfolioManager
        manager = PortfolioManager()
        manager.reset()
        logger.info("账户已重置: 初始资金 10,000")

    # Phase 1: 盘后分析
    if not skip_analyze:
        logger.info("[Phase 1/3] 盘后分析...")
        from scripts.evening_summary import run_analysis
        result = await run_analysis(pool=pool, no_llm=no_llm)
        logger.info(f"分析完成: {result.get('debate', {}).get('summary', 'N/A')}")
    else:
        logger.info("[Phase 1/3] 跳过分析 (--skip-analyze)")

    # Phase 2: 早盘买入
    logger.info("[Phase 2/3] 早盘买入...")
    from scripts.morning_buy import run_morning_buy
    buy_result = await run_morning_buy(min_confidence=min_confidence)
    logger.info(f"买入: {buy_result.get('executed', 0)}/{buy_result.get('candidates', 0)} 成交")

    # Phase 3: 收盘总结
    logger.info("[Phase 3/3] 收盘总结...")
    from scripts.evening_summary import run_summary
    summary_result = await run_summary()
    logger.info(f"总结: 总资产 {summary_result.get('total_value', 0):,.2f}")

    logger.info(f"========== Daily Runner 完成 ==========")

    return {
        "date": today,
        "analysis": {"status": "ok"},
        "buy": buy_result,
        "summary": summary_result,
    }


async def main():
    parser = argparse.ArgumentParser(description="A股模拟交易 -- 每日自动化流程")
    parser.add_argument("--pool", type=str, default="all_industries")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--reset", action="store_true", help="重置账户")
    parser.add_argument("--skip-analyze", action="store_true", help="跳过分析(使用已有报告)")
    args = parser.parse_args()

    result = await run_full_day(
        pool=args.pool,
        no_llm=args.no_llm,
        min_confidence=args.min_confidence,
        reset=args.reset,
        skip_analyze=args.skip_analyze,
    )

    print(f"\n{'='*60}")
    print(f"Daily Run Complete: {result['date']}")
    print(f"  Total Value: {result['summary'].get('total_value', 'N/A')}")
    print(f"  Positions: {result['summary'].get('positions', 'N/A')}")
    print(f"  Exits: {result['summary'].get('exits', 'N/A')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
