"""
内置任务调度器 (v2.9)

轻量级 cron — 无需外部 crontab, 在 FastAPI 进程内调度。
支持:
  - 盘后分析 (默认 15:30 北京时间, 周一至周五)
  - 收盘总结 (默认 15:05)
  - 自定义间隔任务

使用:
    from scripts.scheduler import TaskScheduler
    scheduler = TaskScheduler()
    scheduler.start()  # 后台运行

或在 FastAPI 启动时自动启动:
    uvicorn.run(app, ...)  # app 的 lifespan 自动启动 scheduler
"""

import asyncio
import os
import time
from datetime import date, datetime, time as dtime
from typing import Callable, Dict, List, Optional

from loguru import logger


class TaskScheduler:
    """
    轻量任务调度器

    只做一件事: 在指定时间点触发 callback。
    使用 asyncio 实现,无需外部依赖。
    """

    def __init__(self):
        self._tasks: List[dict] = []
        self._running = False
        self._task_handle: Optional[asyncio.Task] = None

    def add_daily(
        self,
        name: str,
        callback: Callable,
        hour: int = 15,
        minute: int = 30,
        weekdays: List[int] = None,  # 0=Mon, 6=Sun
    ):
        """
        添加每日定时任务

        Args:
            name: 任务名称
            callback: async callable
            hour/minte: 触发时间 (北京时间)
            weekdays: 哪些工作日运行, None = 全部
        """
        self._tasks.append({
            "name": name,
            "callback": callback,
            "time": dtime(hour, minute),
            "weekdays": weekdays or list(range(5)),  # Mon-Fri
            "last_run": None,
        })
        logger.info(f"[Scheduler] 注册任务: {name} @ {hour:02d}:{minute:02d}")

    async def _loop(self):
        """主循环 — 每分钟检查一次"""
        logger.info("[Scheduler] 调度器启动")
        self._running = True

        while self._running:
            now = datetime.now()
            current_time = now.time()
            current_weekday = now.weekday()

            for task in self._tasks:
                # 检查工作日
                if current_weekday not in task["weekdays"]:
                    continue

                # 检查时间 — v2.14: 宽松窗口 + 追赶机制
                target = task["time"]
                target_minutes = target.hour * 60 + target.minute
                current_minutes = current_time.hour * 60 + current_time.minute
                # 窗口: 目标时间到目标时间+5分钟 (避免错过, 同时防重复)
                in_window = target_minutes <= current_minutes < target_minutes + 5

                if in_window:
                    today = now.date()
                    if task["last_run"] != today:
                        task["last_run"] = today
                        delay = current_minutes - target_minutes
                        if delay > 1:
                            logger.info(f"[Scheduler] 追赶: {task['name']} (延迟{delay}分钟)")
                        else:
                            logger.info(f"[Scheduler] 触发: {task['name']}")
                        try:
                            await task["callback"]()
                        except Exception as e:
                            logger.error(f"[Scheduler] 任务{task['name']}失败: {e}", exc_info=True)

            await asyncio.sleep(60)

    def start(self):
        """启动调度器 (异步,非阻塞)"""
        if self._task_handle is not None:
            logger.warning("[Scheduler] 已在运行中")
            return
        self._task_handle = asyncio.create_task(self._loop())

    async def stop(self):
        """停止调度器"""
        self._running = False
        if self._task_handle:
            self._task_handle.cancel()
            try:
                await self._task_handle
            except asyncio.CancelledError:
                pass
            self._task_handle = None
            logger.info("[Scheduler] 已停止")


# ═══════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════

_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()

        # 收盘分析: 15:05 (A股 15:00 收盘)
        async def _daily_analysis():
            from scripts.run_daily_analysis import DailyPipeline
            from scripts.shared import load_watchlist
            logger.info("=== 盘后自动分析 (v2.10) ===")
            symbols = load_watchlist("all_industries")
            pipeline = DailyPipeline(symbols, enable_notify=False, force_no_llm=True)
            await pipeline.run()

        # 盘后卖出检查: 15:10 (分析完成后)
        async def _evening_sell():
            from scripts.evening_sell import run_evening_sell
            logger.info("=== 盘后卖出检查 (自动) ===")
            await run_evening_sell(dry_run=False)

        # 早盘买入: 9:25 (集合竞价后)
        async def _morning_buy():
            from scripts.morning_buy import run_morning_buy
            logger.info("=== 早盘自动买入 ===")
            await run_morning_buy(min_confidence=0.45)

        _scheduler.add_daily("盘后分析", _daily_analysis, hour=15, minute=7)
        _scheduler.add_daily("盘后卖出", _evening_sell, hour=15, minute=12)
        _scheduler.add_daily("早盘买入", _morning_buy, hour=9, minute=25)

    return _scheduler
