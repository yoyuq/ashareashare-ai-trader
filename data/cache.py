"""
Redis缓存层

缓存策略:
- 实时行情: 3秒TTL (几乎不缓存,保证实时性)
- 日K线: 1小时TTL (盘中可能有新数据)
- 历史日K线: 24小时TTL (已确定的历史数据不变)
- 股票列表: 1天TTL (每天变化不大)
- 财务数据: 1天TTL
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None  # type: ignore


class CacheLayer:
    """
    Redis缓存层 (带降级: Redis不可用时跳过缓存)

    使用方式:
        cache = CacheLayer()
        await cache.connect()

        # 缓存数据
        await cache.set("key", df, ttl=3600)

        # 读取缓存
        df = await cache.get("key")
    """

    def __init__(self, prefix: str = "ashare"):
        self.prefix = prefix
        self.redis: Optional["aioredis.Redis"] = None  # type: ignore
        self._enabled = REDIS_AVAILABLE

    async def connect(self):
        """连接Redis (失败不阻塞)"""
        if not REDIS_AVAILABLE:
            logger.info("Redis不可用,缓存层降级为直通模式")
            self._enabled = False
            return

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD", None)
        db = int(os.getenv("REDIS_DB", "0"))

        try:
            self.redis = await aioredis.Redis(
                host=host,
                port=port,
                password=password or None,
                db=db,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False,
            )
            await self.redis.ping()
            logger.info(f"Redis连接成功: {host}:{port}")
            self._enabled = True
        except Exception as e:
            logger.warning(f"Redis连接失败({host}:{port}): {e},缓存层降级")
            self._enabled = False
            self.redis = None

    async def disconnect(self):
        """断开Redis连接"""
        if self.redis:
            await self.redis.close()

    @property
    def enabled(self) -> bool:
        return self._enabled and self.redis is not None

    def _make_key(self, key: str) -> str:
        """加上prefix的键名"""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        从缓存读取DataFrame

        Args:
            key: 缓存键

        Returns:
            DataFrame或None(未命中/不可用)
        """
        if not self.enabled or self.redis is None:
            return None

        try:
            data = await self.redis.get(self._make_key(key))
            if data is None:
                return None
            # 反序列化: msgpack → JSON → DataFrame
            import io
            df = pd.read_parquet(io.BytesIO(data))
            return df
        except Exception as e:
            logger.debug(f"缓存读取失败({key}): {e}")
            return None

    async def set(
        self,
        key: str,
        df: pd.DataFrame,
        ttl: int = 3600,
    ) -> bool:
        """
        将DataFrame写入缓存

        Args:
            key: 缓存键
            df: 要缓存的DataFrame
            ttl: 过期时间(秒), 默认1小时

        Returns:
            True if written successfully
        """
        if not self.enabled or self.redis is None:
            return False

        try:
            # 序列化: DataFrame → Parquet bytes
            import io
            buf = io.BytesIO()
            df.to_parquet(buf, compression="zstd")
            buf.seek(0)

            await self.redis.set(
                self._make_key(key),
                buf.read(),
                ex=ttl,
            )
            return True
        except Exception as e:
            logger.debug(f"缓存写入失败({key}): {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        if not self.enabled or self.redis is None:
            return False
        try:
            await self.redis.delete(self._make_key(key))
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        if not self.enabled or self.redis is None:
            return False
        try:
            return bool(await self.redis.exists(self._make_key(key)))
        except Exception:
            return False

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """从缓存读取JSON"""
        if not self.enabled or self.redis is None:
            return None
        try:
            data = await self.redis.get(self._make_key(key))
            if data is None:
                return None
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    async def set_json(self, key: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """将JSON写入缓存"""
        if not self.enabled or self.redis is None:
            return False
        try:
            await self.redis.set(
                self._make_key(key),
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                ex=ttl,
            )
            return True
        except Exception:
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """批量删除匹配pattern的缓存键"""
        if not self.enabled or self.redis is None:
            return 0

        try:
            full_pattern = self._make_key(pattern)
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match=full_pattern, count=100
                )
                if keys:
                    await self.redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            return deleted
        except Exception:
            return 0

    # ═══════════════════════════════════════════════════════════════
    # 便捷缓存方法
    # ═══════════════════════════════════════════════════════════════

    async def get_or_fetch(
        self,
        key: str,
        fetch_func,
        ttl: int = 3600,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        缓存读取-未命中-回源 模式

        Usage:
            df = await cache.get_or_fetch(
                "kline:000001:2024-01:2024-12",
                provider.get_daily_kline,
                ttl=3600,
                request=DataRequest(...),
            )
        """
        # 1. 尝试缓存
        cached = await self.get(key)
        if cached is not None:
            logger.debug(f"缓存命中: {key}")
            return cached

        # 2. 回源获取
        logger.debug(f"缓存未命中: {key},回源获取")
        df = await fetch_func(*args, **kwargs)

        # 3. 写入缓存
        if isinstance(df, pd.DataFrame) and not df.empty:
            await self.set(key, df, ttl=ttl)
        elif hasattr(df, "data") and isinstance(df.data, pd.DataFrame):
            await self.set(key, df.data, ttl=ttl)

        return df


# 全局单例
_cache_instance: Optional[CacheLayer] = None


async def get_cache() -> CacheLayer:
    """获取CacheLayer单例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheLayer()
        await _cache_instance.connect()
    return _cache_instance
