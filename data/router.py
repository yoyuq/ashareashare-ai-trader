"""
数据路由器 — 多源路由 + 自动降级 + 交叉验证

职责:
1. 按优先级路由数据请求到多个数据源
2. 主源不可用时自动降级到备用源
3. (v2.1) 关键数据双源交叉验证 — 同一策略在AKShare和Baostock上回测结果偏差过大→报警
"""

import asyncio
import hashlib
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from .providers.akshare_provider import AKShareProvider
from .providers.base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource


class DataRouter:
    """
    多源数据路由器

    路由策略:
    - 优先级: AKShare(1) → Baostock(2) → 其他备用
    - 降级: 主源连续失败3次 → 自动切换到备用源 → 5分钟后尝试恢复
    - v2.1: 关键数据(回测用)双源交叉验证
    """

    def __init__(self, cross_validation: bool = True):
        self.cross_validation = cross_validation
        self._providers: Dict[DataSource, DataProvider] = {}
        self._priority: List[DataSource] = []
        self._failure_counts: Dict[DataSource, int] = {}
        self._downgraded_until: Dict[DataSource, datetime] = {}
        self._max_failures = 3
        self._downgrade_duration = timedelta(minutes=5)

    def register(self, provider: DataProvider, priority: int = 0):
        """注册数据源"""
        self._providers[provider.source] = provider
        if provider.source not in self._priority:
            self._priority.append(provider.source)
        self._priority.sort(key=lambda s: priority)  # 按优先级排序

    # ═══════════════════════════════════════════════════════════════
    # 公共API
    # ═══════════════════════════════════════════════════════════════

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        """获取日K线(自动路由+降级)"""
        return await self._route_with_fallback("get_daily_kline", request)

    async def get_minute_kline(self, request: DataRequest) -> DataResult:
        """获取分钟K线"""
        return await self._route_with_fallback("get_minute_kline", request)

    async def get_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表"""
        for source in self._active_sources():
            provider = self._providers.get(source)
            if provider is None:
                continue
            try:
                df = await provider.get_stock_list()
                if not df.empty:
                    self._on_success(source)
                    return df
            except Exception as e:
                logger.warning(f"{source.value} 获取股票列表失败: {e}")
                self._on_failure(source)
        raise RuntimeError("所有数据源均无法获取股票列表")

    async def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情(只用主源)"""
        for source in self._active_sources():
            provider = self._providers.get(source)
            if provider is None:
                continue
            try:
                df = await provider.get_realtime_quote(symbols)
                if not df.empty and "error" not in df.columns:
                    self._on_success(source)
                    return df
            except Exception as e:
                logger.warning(f"{source.value} 获取实时行情失败: {e}")
                self._on_failure(source)
        return pd.DataFrame({"error": ["所有数据源不可用"]})

    # ═══════════════════════════════════════════════════════════════
    # 路由逻辑
    # ═══════════════════════════════════════════════════════════════

    async def _route_with_fallback(self, method: str, request: DataRequest) -> DataResult:
        """带自动降级的路由"""
        errors = []
        for source in self._active_sources():
            provider = self._providers.get(source)
            if provider is None:
                continue
            try:
                func = getattr(provider, method)
                result = await func(request)
                if result is not None and not result.data.empty:
                    self._on_success(source)
                    return result
                else:
                    errors.append(f"{source.value}: 返回空数据")
            except Exception as e:
                logger.warning(f"{source.value}.{method}({request.symbol}) 失败: {e}")
                errors.append(f"{source.value}: {e}")
                self._on_failure(source)
                continue

        raise RuntimeError(
            f"所有数据源均无法获取{request.symbol}的{method}数据: {'; '.join(errors)}"
        )

    def _active_sources(self) -> List[DataSource]:
        """返回当前可用的数据源列表(已排除降级中的)"""
        now = datetime.now()
        active = []
        for source in self._priority:
            if source not in self._providers:
                continue
            # 检查是否在降级期间
            until = self._downgraded_until.get(source)
            if until and now < until:
                continue
            active.append(source)
        return active

    def _on_success(self, source: DataSource):
        """数据源成功——重置失败计数"""
        self._failure_counts[source] = 0

    def _on_failure(self, source: DataSource):
        """数据源失败——累计失败计数,超过阈值则降级"""
        count = self._failure_counts.get(source, 0) + 1
        self._failure_counts[source] = count
        if count >= self._max_failures:
            self._downgraded_until[source] = datetime.now() + self._downgrade_duration
            logger.warning(
                f"数据源 {source.value} 连续失败{count}次,降级至"
                f" {self._downgraded_until[source].strftime('%H:%M:%S')}"
            )

    # ═══════════════════════════════════════════════════════════════
    # v2.1: 交叉验证
    # ═══════════════════════════════════════════════════════════════

    async def cross_validate_daily(
        self, request: DataRequest
    ) -> Tuple[DataResult, Optional[str]]:
        """
        双源交叉验证日K线数据

        Returns:
            (result, warning_message)
            - result: 主源数据(始终返回)
            - warning_message: 若两源偏差过大则返回警告字符串
        """
        if not self.cross_validation or len(self._active_sources()) < 2:
            result = await self.get_daily_kline(request)
            return result, None

        # 从两个数据源分别获取
        primary = self._active_sources()[0]
        secondary = self._active_sources()[1] if len(self._active_sources()) > 1 else None

        if secondary is None:
            result = await self.get_daily_kline(request)
            return result, None

        primary_provider = self._providers[primary]
        secondary_provider = self._providers[secondary]

        try:
            result1 = await primary_provider.get_daily_kline(request)
        except Exception as e:
            # 主源失败,直接用备源
            logger.warning(f"交叉验证: 主源失败,降级到{secondary.value}")
            result2 = await secondary_provider.get_daily_kline(request)
            return result2, f"主源{primary.value}不可用,使用{secondary.value}"

        try:
            result2 = await secondary_provider.get_daily_kline(request)
        except Exception:
            # 备源不可用,仍返回主源数据
            return result1, f"备源{secondary.value}不可用"

        # 比较两个数据源的close序列相关性
        warning = self._compare_results(result1, result2)
        return result1, warning

    def _compare_results(
        self, r1: DataResult, r2: DataResult
    ) -> Optional[str]:
        """比较两个数据源的close序列,偏差过大则报警"""
        df1 = r1.data.set_index("date")["close"]
        df2 = r2.data.set_index("date")["close"]

        # 取交集日期
        common_dates = df1.index.intersection(df2.index)
        if len(common_dates) < 10:
            return "双源数据交集日期不足10天,跳过交叉验证"

        s1 = df1.loc[common_dates]
        s2 = df2.loc[common_dates]

        # 计算日收益率相关性
        ret1 = s1.pct_change().dropna()
        ret2 = s2.pct_change().dropna()
        corr = ret1.corr(ret2)

        # 计算价格偏差
        mape = (abs(s1 - s2) / s1).mean()

        if corr < 0.95:
            return (
                f"⚠️ 双源数据严重不一致! 收益率相关性仅{corr:.3f},"
                f" 平均偏差{mape:.2%}。请检查数据源。"
            )
        elif corr < 0.99:
            return (
                f"⚡ 双源数据存在差异: 相关性{corr:.3f}, 平均偏差{mape:.2%}"
            )

        return None  # 数据一致

    # ═══════════════════════════════════════════════════════════════
    # 缓存键
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def cache_key(request: DataRequest) -> str:
        """生成请求的唯一缓存键"""
        raw = (
            f"{request.symbol}:{request.start_date}:{request.end_date}:"
            f"{request.frequency.value}:{request.adjust}"
        )
        return hashlib.md5(raw.encode()).hexdigest()

    # ═══════════════════════════════════════════════════════════════
    # 健康状态
    # ═══════════════════════════════════════════════════════════════

    @property
    def status(self) -> Dict[str, Any]:
        """返回所有数据源状态"""
        return {
            source.value: {
                "healthy": self._providers[source].is_healthy(),
                "failures": self._failure_counts.get(source, 0),
                "downgraded_until": str(self._downgraded_until.get(source, "-")),
            }
            for source in self._providers
        }


# ═══════════════════════════════════════════════════════════════
# 全局单例工厂
# ═══════════════════════════════════════════════════════════════

_router_instance: Optional[DataRouter] = None


def get_data_router(cross_validation: bool = True) -> DataRouter:
    """获取DataRouter单例(自动注册默认数据源)"""
    global _router_instance
    if _router_instance is None:
        _router_instance = DataRouter(cross_validation=cross_validation)
        _router_instance.register(AKShareProvider(), priority=1)
        _router_instance.register(BaostockProvider(), priority=2)
        logger.info("DataRouter初始化完成: AKShare(主) → Baostock(备)")
    return _router_instance
