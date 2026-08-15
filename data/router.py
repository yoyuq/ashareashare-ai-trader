"""
数据路由器 — 多源路由 + 自动降级 + 交叉验证

职责:
1. 按优先级路由数据请求到多个数据源
2. 主源不可用时自动降级到备用源
3. (v2.1) 关键数据双源交叉验证 — 同一策略在AKShare和Baostock上回测结果偏差过大→报警
"""

import asyncio
import hashlib
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

# 北京时区 (v3.0): 缓存 TTL / 熔断窗口按 A 股交易日边界
from timeutil import now_cn_naive

from .providers.akshare_provider import AKShareProvider
from .providers.baostock_provider import BaostockProvider
from .providers.eastmoney_provider import EastMoneyProvider
from .providers.tencent_provider import TencentFinanceProvider
from .providers.base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource


class DataRouter:
    """
    多源数据路由器

    路由策略:
    - 优先级: AKShare(1) → Baostock(2) → 其他备用
    - 降级: 主源连续失败3次 → 自动切换到备用源 → 5分钟后尝试恢复
    - v2.1: 关键数据(回测用)双源交叉验证
    """

    _GLOBAL_KEY = "*"  # v5.6 P0-7: 非标的维度 (股票列表/实时行情) 的失败键

    def __init__(self, cross_validation: bool = True):
        self.cross_validation = cross_validation
        self._providers: Dict[DataSource, DataProvider] = {}
        self._priority: List[DataSource] = []
        self._priorities: Dict[DataSource, int] = {}  # v3.0: 记录注册优先级 (越小越优先)
        # v5.6 P0-7: 失败计数/降级窗口按 (source, symbol) 记录, 避免单只退市/
        # 新股/北交所标的的连续失败误伤整个健康源; 非标的维度用 _GLOBAL_KEY。
        self._failure_counts: Dict[Tuple[DataSource, str], int] = {}
        self._downgraded_until: Dict[Tuple[DataSource, str], datetime] = {}
        self._max_failures = 3
        self._downgrade_duration = timedelta(minutes=5)
        self.last_used_source: str = ""  # v2.10: 追踪实际使用的数据源
        # v5.6 P1-1/P1-3: 分层 TTL 内存缓存 + LRU 上限 + 负缓存 + 并发合并
        self._cache: Dict[str, tuple] = {}  # key → (timestamp, DataResult | [失败源])
        # 按方法分层 TTL: 日K线 1h (盘中可能有新数据); 分钟K线 5min
        self._cache_ttl: Dict[str, timedelta] = {
            "get_daily_kline": timedelta(hours=1),
            "get_minute_kline": timedelta(minutes=5),
        }
        self._cache_max_entries = int(os.getenv("DATA_CACHE_MAX_ENTRIES", "5000"))
        self._negative_cache_ttl = timedelta(seconds=30)  # 全源失败短期负缓存 (防击穿)
        self._inflight: Dict[str, asyncio.Future] = {}  # v5.6 P1-3: 并发合并 (单飞)
        # 实时行情 / 股票列表缓存 (v5.6 P1-1: 此前无缓存, 每次全量走网络)
        self._quote_cache: Dict[str, tuple] = {}  # ",".join(sorted(symbols)) → (ts, df)
        self._quote_cache_ttl = timedelta(seconds=3)
        self._stock_list_cache: Optional[tuple] = None  # (ts, df)
        self._stock_list_cache_ttl = timedelta(minutes=5)  # 内存兜底 5min (Redis 层可用则 24h)

    def register(self, provider: DataProvider, priority: int = 0):
        """注册数据源 (priority 越小越优先)

        v3.0: 修复优先级排序 bug — 此前用单次调用传入的 priority 常量排序,
        所有源得到相同 key 导致排序失效 (实际按插入序)。现按各源记录的优先级排序。
        """
        self._providers[provider.source] = provider
        self._priorities[provider.source] = priority
        if provider.source not in self._priority:
            self._priority.append(provider.source)
        self._priority.sort(key=lambda s: self._priorities.get(s, 0))

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
        """获取全市场股票列表 (v3.0: 统一 schema 后返回; v5.6 P1-1: 5min 内存缓存)"""
        # v5.6 P1-1: 内存缓存 (Redis 层可用则 24h, 无 Redis 兜底 5min)
        if self._stock_list_cache is not None:
            ts, df = self._stock_list_cache
            if now_cn_naive() - ts < self._stock_list_cache_ttl:
                return df
        for source in self._active_sources():
            provider = self._providers.get(source)
            if provider is None:
                continue
            try:
                df = await provider.get_stock_list()
                # v3.0: 排除错误占位 DataFrame (如 Tencent 返回 {"error": [...]} 为非空)
                if not df.empty and "error" not in df.columns:
                    self._on_success(source)
                    out = self._normalize_stock_list(df)
                    self._stock_list_cache = (now_cn_naive(), out)  # v5.6 P1-1: 缓存
                    return out
                # v5.6 P0-6: 空数据/错误占位表与"确实无数据"分离 — 计入失败,
                # 触发熔断降级而非静默吞掉 (Tencent 不支持股票列表属永久缺能力,
                # 仍计入但由 5 分钟降级窗口限频, 不会影响其实时行情/K线等已支持维度)。
                logger.warning(f"{source.value} 股票列表返回空/错误占位表")
                self._on_failure(source)
            except Exception as e:
                logger.warning(f"{source.value} 获取股票列表失败: {e}")
                self._on_failure(source)
        raise RuntimeError("所有数据源均无法获取股票列表")

    @staticmethod
    def _normalize_stock_list(df: pd.DataFrame) -> pd.DataFrame:
        """统一股票列表 schema: 保证 symbol(带市场前缀)/name/status 列存在。

        各数据源列不一致 (Baostock: symbol/name/status; AKShare: symbol/name/price;
        Tushare: ts_code/name), 此前直接透传导致 quick_scan 按 Baostock 形状取列
        (df["status"]=="1") 在 fallback 到其他源时 KeyError。保留原列, 补齐缺失列。
        """
        out = df.copy()
        if "symbol" not in out.columns:
            if "code" in out.columns:
                out["symbol"] = out["code"]
            elif "ts_code" in out.columns:
                out["symbol"] = out["ts_code"]
            else:
                out["symbol"] = out.index.astype(str)

        # 统一为带市场前缀: sh./sz./bj. (幂等: 已带前缀则原样保留)
        def _prefix(s: str) -> str:
            s = str(s)
            if s.startswith(("sh.", "sz.", "bj.")):
                return s
            code = s.split(".")[0]
            if code.startswith(("6", "9")):
                return f"sh.{code}"
            if code.startswith(("8", "4")):
                return f"bj.{code}"
            return f"sz.{code}"

        out["symbol"] = [_prefix(x) for x in out["symbol"]]
        if "status" not in out.columns:
            out["status"] = "1"  # 默认视为上市
        if "name" not in out.columns:
            out["name"] = ""
        return out

    async def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情(只用主源; v5.6 P1-1: 3s 缓存防高频/并发重复请求)"""
        cache_key = ",".join(sorted(symbols))
        if cache_key in self._quote_cache:
            ts, df = self._quote_cache[cache_key]
            if now_cn_naive() - ts < self._quote_cache_ttl:
                return df
        for source in self._active_sources():
            provider = self._providers.get(source)
            if provider is None:
                continue
            try:
                df = await provider.get_realtime_quote(symbols)
                if not df.empty and "error" not in df.columns:
                    self._on_success(source)
                    self._quote_cache[cache_key] = (now_cn_naive(), df)  # v5.6 P1-1
                    return df
                # v5.6 P0-6: 空/错误占位表计入失败 (静默失败分离)
                logger.warning(f"{source.value} 实时行情返回空/错误占位表")
                self._on_failure(source)
            except Exception as e:
                logger.warning(f"{source.value} 获取实时行情失败: {e}")
                self._on_failure(source)
        return pd.DataFrame({"error": ["所有数据源不可用"]})

    # ═══════════════════════════════════════════════════════════════
    # 路由逻辑
    # ═══════════════════════════════════════════════════════════════

    async def _route_with_fallback(self, method: str, request: DataRequest) -> DataResult:
        """带自动降级的路由 (v2.10: 缓存 + 数据校验 + 超时; v5.6: 分层 TTL + 负缓存 + 单飞)"""

        # 缓存键必须含复权方式与频率, 否则 qfq/hfq/raw 数据互相污染
        cache_key = (
            f"{method}:{request.symbol}:{request.start_date}:{request.end_date}:"
            f"{request.frequency.value}:{request.adjust}"
        )
        ttl = self._cache_ttl.get(method, self._cache_ttl["get_daily_kline"])

        # 1. 命中缓存 (正缓存按分层 TTL; 负缓存按 30s 短期)
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if isinstance(cached, list):  # 负缓存 (全源失败的源清单)
                if now_cn_naive() - ts < self._negative_cache_ttl:
                    # v5.6 P1-3: 命中负缓存视为再次失败, 计入降级 (保持 3 次降级语义)
                    for src in cached:
                        self._on_failure(src, request.symbol)
                    raise RuntimeError(
                        f"{request.symbol} 的 {method} 近期全源失败 (负缓存, "
                        f"{int(self._negative_cache_ttl.total_seconds())}s 内不重试)"
                    )
                del self._cache[cache_key]
            elif now_cn_naive() - ts < ttl:
                return cached
            else:
                del self._cache[cache_key]

        # 2. 并发合并 (单飞): 相同 key 的并发请求共享同一个 in-flight Future, 防击穿
        inflight = self._inflight.get(cache_key)
        if inflight is not None and not inflight.done():
            logger.debug(f"{cache_key} 已有在途请求, 合并等待")
            return await asyncio.shield(inflight)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._inflight[cache_key] = fut
        try:
            result = await self._fetch(method, request, cache_key)
            if not fut.done():
                fut.set_result(result)
            return result
        except BaseException as e:  # 含 CancelledError — 需同步给等待方
            if not fut.done():
                fut.set_exception(e)
            raise
        finally:
            self._inflight.pop(cache_key, None)
            # 无并发 waiter 时显式取走异常, 避免 "Future exception was never retrieved" 警告
            if fut.done() and not fut.cancelled():
                fut.exception()

    async def _fetch(self, method: str, request: DataRequest, cache_key: str) -> DataResult:
        """实际执行多源降级获取 (不含缓存/单飞逻辑; 失败写负缓存)"""
        errors = []
        failed_sources: List[DataSource] = []
        for source in self._active_sources(request.symbol):
            provider = self._providers.get(source)
            if provider is None:
                continue
            try:
                func = getattr(provider, method)
                result = await asyncio.wait_for(func(request), timeout=30)
                if result is not None and not result.data.empty:
                    # v2.10: 数据质量校验
                    if hasattr(provider, '_validate_response'):
                        is_valid = provider._validate_response(result.data, request)
                        if not is_valid:
                            errors.append(f"{source.value}: 数据校验失败")
                            failed_sources.append(source)
                            self._on_failure(source, request.symbol)
                            continue
                    self._on_success(source, request.symbol)
                    self.last_used_source = source.value
                    self._cache_set(cache_key, result)  # v5.6 P1-3: 缓存 (含 LRU 淘汰)
                    return result
                else:
                    errors.append(f"{source.value}: 返回空数据")
                    failed_sources.append(source)
                    self._on_failure(source, request.symbol)  # v3.0: 空响应同样触发熔断降级
            except asyncio.TimeoutError:
                logger.warning(f"{source.value}.{method}({request.symbol}) 超时(30s)")
                errors.append(f"{source.value}: 超时")
                failed_sources.append(source)
                self._on_failure(source, request.symbol)
            except Exception as e:
                logger.warning(f"{source.value}.{method}({request.symbol}) 失败: {e}")
                errors.append(f"{source.value}: {e}")
                failed_sources.append(source)
                self._on_failure(source, request.symbol)
                continue

        # v5.6 P1-3: 全源失败 → 写负缓存 (短期), 防对坏标的反复击穿
        self._cache_set(cache_key, failed_sources)
        raise RuntimeError(
            f"所有数据源均无法获取{request.symbol}的{method}数据: {'; '.join(errors)}"
        )

    def _cache_set(self, key: str, value: Any) -> None:
        """写缓存 + LRU 淘汰 (超出上限淘汰最旧条目)

        v5.6 P1-3: dict 在 Python 3.7+ 保插入序, 超限时淘汰最旧 key;
        value 为 list 时表示负缓存 (由读路径按 _negative_cache_ttl 判失效)。
        """
        if len(self._cache) >= self._cache_max_entries and key not in self._cache:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest, None)
        self._cache[key] = (now_cn_naive(), value)

    def _active_sources(self, symbol: Optional[str] = None) -> List[DataSource]:
        """返回当前可用的数据源列表(已排除降级中的)

        v5.6 P0-7: 传 symbol 时, 仅该 symbol 被标度降级的源被排除; 全局降级
        (股票列表/实时行情等非标的维度) 始终生效。避免单只退市/新股标的的
        连续失败把整个健康源降级。
        """
        now = now_cn_naive()
        active = []
        for source in self._priority:
            if source not in self._providers:
                continue
            # 全局降级 (方法级失败)
            until_global = self._downgraded_until.get((source, self._GLOBAL_KEY))
            if until_global and now < until_global:
                continue
            # 标度降级 (仅该 symbol 请求)
            if symbol:
                until_sym = self._downgraded_until.get((source, symbol))
                if until_sym and now < until_sym:
                    continue
            active.append(source)
        return active

    def _on_success(self, source: DataSource, symbol: Optional[str] = None):
        """数据源成功——重置失败计数"""
        key = (source, symbol or self._GLOBAL_KEY)
        self._failure_counts[key] = 0
        self._downgraded_until.pop(key, None)

    def _on_failure(self, source: DataSource, symbol: Optional[str] = None):
        """数据源失败——累计失败计数,超过阈值则降级

        v5.6 P0-7: 按 (source, symbol) 记失败; 降级窗口过期后重置计数,
        给源一个全新重试周期 (修复"脆弱源实际永久降级")。
        """
        key = (source, symbol or self._GLOBAL_KEY)
        # 降级窗口已过期 → 清计数, 重新累计 (否则下次失败会立即再触发降级)
        until = self._downgraded_until.get(key)
        if until and now_cn_naive() >= until:
            self._downgraded_until.pop(key, None)
            self._failure_counts[key] = 0
        count = self._failure_counts.get(key, 0) + 1
        self._failure_counts[key] = count
        if count >= self._max_failures:
            self._downgraded_until[key] = now_cn_naive() + self._downgrade_duration
            _scope = f"标的 {symbol}" if symbol else "全局"
            logger.warning(
                f"数据源 {source.value}[{_scope}] 连续失败{count}次,降级至"
                f" {self._downgraded_until[key].strftime('%H:%M:%S')}"
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
        """返回所有数据源状态 (v5.6 P0-7: 含标度降级明细)"""
        out: Dict[str, Any] = {}
        for source in self._providers:
            gkey = (source, self._GLOBAL_KEY)
            sym_downgrades = {
                s: str(t) for (src, s), t in self._downgraded_until.items()
                if src == source and s != self._GLOBAL_KEY
            }
            out[source.value] = {
                "healthy": self._providers[source].is_healthy(),
                "failures": self._failure_counts.get(gkey, 0),
                "downgraded_until": str(self._downgraded_until.get(gkey, "-")),
                "symbol_downgrades": sym_downgrades,
            }
        return out


# ═══════════════════════════════════════════════════════════════
# 全局单例工厂
# ═══════════════════════════════════════════════════════════════

_router_instance: Optional[DataRouter] = None


def get_data_router(cross_validation: bool = True) -> DataRouter:
    """
    获取DataRouter单例(自动注册默认数据源)

    v3.0 优先级:
      0. Tushare   — 可靠性最高(实测99.9%), 配置 TUSHARE_TOKEN 后作为主源
      1. Baostock  — 历史数据稳定, K线首选(无token时)
      2. Tencent   — 实时行情免费无限制, 实时报价首选
      3. EastMoney — 国内稳定的免费源, 实时行情+历史K线
      4. AKShare   — 数据最全, 补充其他源缺失的数据类型
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = DataRouter(cross_validation=cross_validation)
        import os
        # 仅当配置了 TUSHARE_TOKEN 才注册为主源, 避免未配置时反复失败重试
        if os.getenv("TUSHARE_TOKEN"):
            from data.providers import TushareProvider
            _router_instance.register(TushareProvider(), priority=0)
        _router_instance.register(BaostockProvider(), priority=1)
        _router_instance.register(TencentFinanceProvider(), priority=2)
        _router_instance.register(EastMoneyProvider(), priority=3)
        _router_instance.register(AKShareProvider(), priority=4)
        provider_list = os.getenv("DATA_PROVIDER_ORDER", "Tushare→Baostock→Tencent→EastMoney→AKShare")
        logger.info(f"DataRouter初始化完成: {provider_list}")
    return _router_instance
