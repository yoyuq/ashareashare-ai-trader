"""
增量计算调度器 — v2.0新增

解决全量重算的性能问题:
  - 每日收盘后仅计算新增1天的指标(而非重算10年历史)
  - 自动检测上次计算截止日期,增量追加
  - N日滚动指标(如MA20)需要最近N天的历史数据做窗口计算

核心逻辑:
  1. 查询DB: 上次计算到哪天? → last_date
  2. 下载增量: last_date → today
  3. 窗口重算: today-N → today (仅重算最后N天)
  4. 写入DB: 仅写入新增行
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .indicators import TechnicalAnalyzer


class IncrementalComputer:
    """
    增量计算调度器

    适用场景:
      - 日K线技术指标: 每日新增1条K线,仅需重算最后60天(N=截至最长周期)
      - 分钟K线: 每5分钟新增,重算最后240根(20小时)
      - K线向量编码: 每日编码新增K线的embedding

    使用方式:
        computer = IncrementalComputer()
        result = await computer.compute_daily(
            symbol="sh.600000",
            fetch_func=fetch_kline_from_db,
            compute_func=analyzer.compute_all,
        )
    """

    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size
        self._last_dates: Dict[str, date] = {}

    # ═══════════════════════════════════════════════════════════════
    # 主入口: 增量计算日K线指标
    # ═══════════════════════════════════════════════════════════════

    async def compute_daily(
        self,
        symbol: str,
        fetch_func: Callable[[str, date, date], pd.DataFrame],
        compute_func: Callable[[pd.DataFrame], Any],
        target_date: Optional[date] = None,
        max_lookback_days: int = 120,
    ) -> Tuple[Any, bool]:
        """
        增量计算日K线指标

        Args:
            symbol: 股票代码
            fetch_func: 数据获取函数(symbol, start, end) → DataFrame
            compute_func: 计算函数(DataFrame) → 结果
            target_date: 目标日期(默认: 今天)
            max_lookback_days: 最长回溯天数(确保滚动窗口计算准确)

        Returns:
            (计算结果, 是否全量重算)
        """
        if target_date is None:
            target_date = date.today()

        # 1. 获取上次计算截止日
        last_date = self._last_dates.get(symbol)
        is_full = last_date is None

        if is_full:
            # 首次计算: 拉全量
            start_date = target_date - timedelta(days=365 * 10)  # 10年
            df = await fetch_func(symbol, start_date, target_date)
        else:
            if last_date >= target_date:
                logger.debug(f"{symbol}: 数据已是最新({last_date}),跳过")
                return None, False

            # 增量: last_date+1 → target_date
            start_date = last_date + timedelta(days=1)
            new_df = await fetch_func(symbol, start_date, target_date)

            if new_df.empty:
                logger.debug(f"{symbol}: 无新数据({start_date}→{target_date})")
                return None, False

            # 拉取窗口数据(确保滚动指标准确)
            window_start = target_date - timedelta(days=max_lookback_days)
            old_df = await fetch_func(symbol, window_start, last_date)

            df = pd.concat([old_df, new_df]).drop_duplicates(
                subset=["date"], keep="last"
            ).sort_values("date")

        if df.empty:
            return None, False

        # 2. 执行计算
        result = compute_func(df)

        # 3. 更新截止日期
        self._last_dates[symbol] = target_date

        logger.info(f"{symbol}: {'全量' if is_full else '增量'}计算完成, "
                   f"数据范围: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

        return result, is_full

    # ═══════════════════════════════════════════════════════════════
    # 批量增量计算
    # ═══════════════════════════════════════════════════════════════

    async def compute_batch(
        self,
        symbols: List[str],
        fetch_func: Callable[[str, date, date], pd.DataFrame],
        compute_func: Callable[[pd.DataFrame], Any],
        target_date: Optional[date] = None,
        max_concurrent: int = 5,
    ) -> Dict[str, Tuple[Any, bool]]:
        """
        批量增量计算(带并发控制)

        Returns:
            {symbol: (result, is_full_recompute)}
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _compute_one(sym: str):
            async with semaphore:
                try:
                    return await self.compute_daily(
                        sym, fetch_func, compute_func, target_date
                    )
                except Exception as e:
                    logger.error(f"增量计算失败 {sym}: {e}")
                    return None, False

        tasks = [_compute_one(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"{sym}: 异常 {result}")
                output[sym] = (None, False)
            else:
                output[sym] = result

        success = sum(1 for v in output.values() if v[0] is not None)
        logger.info(f"批量计算完成: {success}/{len(symbols)}")

        return output

    # ═══════════════════════════════════════════════════════════════
    # 增量策略
    # ═══════════════════════════════════════════════════════════════

    def needs_full_recompute(
        self,
        symbol: str,
        force: bool = False,
        max_staleness_days: int = 3,
    ) -> bool:
        """判断是否需要全量重算"""
        if force:
            return True
        last = self._last_dates.get(symbol)
        if last is None:
            return True
        return (date.today() - last).days > max_staleness_days

    def set_last_date(self, symbol: str, dt: date):
        """手动设置截止日期(用于恢复/修正)"""
        self._last_dates[symbol] = dt

    def get_last_date(self, symbol: str) -> Optional[date]:
        """查询截止日期"""
        return self._last_dates.get(symbol)

    def reset(self, symbol: Optional[str] = None):
        """重置进度(强制下次全量重算)"""
        if symbol:
            self._last_dates.pop(symbol, None)
        else:
            self._last_dates.clear()

    # ═══════════════════════════════════════════════════════════════
    # 数据异常检测
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def detect_anomalies(
        df: pd.DataFrame,
        price_col: str = "close",
        vol_col: str = "volume",
    ) -> pd.DataFrame:
        """
        数据异常自动检测

        检测类型:
          1. 价格跳空: 单日涨跌幅 > 15% (非ST,大概率数据错误)
          2. 成交量异常: 单日量是20日均量的10倍+
          3. 价格不变: 连续3天收盘价完全一致(停牌/数据缺失)
          4. OHLC矛盾: low > high / open > high 等不可能组合
          5. 负值检测: 价格/成交量为负
        """
        anomalies = pd.DataFrame(index=df.index)

        # 1. 价格跳空异常
        ret = df[price_col].pct_change()
        anomalies["price_spike"] = abs(ret) > 0.15  # >15%涨跌(除ST外可疑)

        # 2. 成交量异常
        vol_ma20 = df[vol_col].rolling(20).mean()
        anomalies["volume_spike"] = df[vol_col] > (vol_ma20 * 10)

        # 3. 价格停滞(连续3天相同)
        anomalies["price_stuck"] = (
            (df[price_col] == df[price_col].shift(1)) &
            (df[price_col] == df[price_col].shift(2))
        )

        # 4. OHLC矛盾
        anomalies["ohlc_error"] = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )

        # 5. 负值检测
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                anomalies[f"negative_{col}"] = df[col] < 0

        # 汇总
        anomaly_cols = [c for c in anomalies.columns if not c.startswith("negative_")]
        anomalies["has_anomaly"] = anomalies[anomaly_cols].any(axis=1)

        anomaly_count = anomalies["has_anomaly"].sum()
        if anomaly_count > 0:
            logger.warning(
                f"数据异常检测: {anomaly_count}/{len(df)} 行存在异常"
            )

        return anomalies
