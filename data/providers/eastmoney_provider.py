"""
东方财富数据提供器 — 实时行情 + 历史K线

东方财富API是国内最稳定的免费数据源,无需认证,无频率限制。
作为主数据源的补充,优先用于实时价格查询。

API端点:
  - 实时行情: push2.eastmoney.com/api/qt/ulist.np/get
  - 历史K线: push2his.eastmoney.com/api/qt/stock/kline/get
"""

import asyncio
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource


class EastMoneyProvider(DataProvider):
    """东方财富数据提供器 (实时行情, 优先级最高)"""

    REALTIME_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
    KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    # 批量查询上限
    BATCH_SIZE = 50

    def __init__(self):
        super().__init__(name="EastMoney", source=DataSource.EASTMONEY)
        self._realtime_cache: Dict[str, float] = {}
        self._cache_time: Optional[datetime] = None
        # v2.10: 创建双session — 一个走代理, 一个直连
        self._session = requests.Session()
        self._session.trust_env = False  # 国内站点直连
        self._session_direct = self._session
        # 备用session (使用系统代理)
        self._session_proxy = requests.Session()
        # 重试配置
        self._max_retries = 2
        self._retry_delay = 1.0  # 秒

    # ═══════════════════════════════════════════════════════════════
    # v2.10: 重试 + 双通道请求
    # ═══════════════════════════════════════════════════════════════

    async def _request_with_retry(self, url: str, timeout: int = 10) -> requests.Response:
        """带重试和双通道fallback的HTTP请求

        先尝试直连(国内站点), 失败后尝试代理通道, 各重试2次。
        """
        import time as _time

        last_error = None
        sessions = [self._session_direct, self._session_proxy]
        session_names = ["direct", "proxy"]

        for sess_idx, session in enumerate(sessions):
            for attempt in range(self._max_retries):
                try:
                    resp = await asyncio.to_thread(session.get, url, timeout=timeout)
                    if resp.status_code == 200:
                        return resp
                    last_error = Exception(f"HTTP {resp.status_code}")
                except Exception as e:
                    last_error = e
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(self._retry_delay * (attempt + 1))

        raise last_error or Exception("All retry attempts exhausted")

    # ═══════════════════════════════════════════════════════════════
    # 实时行情
    # ═══════════════════════════════════════════════════════════════

    async def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        """批量获取实时行情

        Returns:
            {symbol: {price, change_pct, high, low, volume, amount, name}}
        """
        # 转换代码格式: sh.600519 → 1.600519
        em_codes = {}
        for sym in symbols:
            code = self._to_em_code(sym)
            if code:
                em_codes[code] = sym

        if not em_codes:
            return {}

        # 检查缓存 (5分钟内复用 — v2.9: 从30s延长到5min,分析场景无需高频刷新)
        now = datetime.now()
        if self._cache_time and (now - self._cache_time).seconds < 300:
            cached = {s: {"price": self._realtime_cache.get(s, 0)}
                     for s in symbols if s in self._realtime_cache}
            if len(cached) == len(symbols):
                return cached

        results = {}
        code_list = list(em_codes.keys())

        # 批量查询 (最多50只)
        for i in range(0, len(code_list), self.BATCH_SIZE):
            batch = code_list[i:i + self.BATCH_SIZE]
            secids = ",".join(batch)
            url = (f"{self.REALTIME_URL}?fltt=2"
                   f"&fields=f2,f3,f4,f12,f14,f15,f16,f17,f18"
                   f"&secids={secids}")

            try:
                resp = await asyncio.to_thread(self._session.get, url, timeout=5)
                data = resp.json()
                if data.get("data") and data["data"].get("diff"):
                    for item in data["data"]["diff"]:
                        code = item.get("f12", "")
                        sym = em_codes.get(f"1.{code}") or em_codes.get(f"0.{code}")
                        if sym:
                            price = item.get("f2", 0)
                            results[sym] = {
                                "price": round(float(price) if price != "-" else 0, 2),
                                "change_pct": item.get("f3", 0),
                                "high": item.get("f15", 0),
                                "low": item.get("f16", 0),
                                "volume": item.get("f17", 0),
                                "amount": item.get("f18", 0),
                                "name": item.get("f14", ""),
                            }
                            # 更新缓存
                            self._realtime_cache[sym] = results[sym]["price"]
            except Exception as e:
                self._healthy = False
                raise RuntimeError(f"EastMoney实时行情失败: {e}")

        self._cache_time = now
        return results

    # ═══════════════════════════════════════════════════════════════
    # 日K线数据
    # ═══════════════════════════════════════════════════════════════

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        """获取日K线数据 (从东方财富历史K线API)

        klt: 101=日线, 102=周线, 103=月线
        fqt: 0=不复权, 1=前复权, 2=后复权
        """
        em_code = self._to_em_code(request.symbol)
        if not em_code:
            raise ValueError(f"无法转换代码: {request.symbol}")

        fqt_map = {"qfq": "1", "hfq": "2", "": "0", None: "0"}
        fqt = fqt_map.get(request.adjust, "1")

        # 扩展日期范围 (API需要)
        beg = request.start_date.strftime("%Y%m%d")
        end = request.end_date.strftime("%Y%m%d")

        url = (f"{self.KLINE_URL}?"
               f"secid={em_code}"
               f"&fields1=f1,f2,f3,f4,f5,f6"
               f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
               f"&klt=101&fqt={fqt}&beg={beg}&end={end}")

        try:
            resp = await self._request_with_retry(url, timeout=15)  # v2.10: 双通道+重试
            data = resp.json()

            if data.get("data") and data["data"].get("klines"):
                rows = []
                for line in data["data"]["klines"]:
                    fields = line.split(",")
                    if len(fields) >= 8:
                        rows.append({
                            "date": fields[0],
                            "open": float(fields[1]),
                            "close": float(fields[2]),
                            "high": float(fields[3]),
                            "low": float(fields[4]),
                            "volume": float(fields[5]),
                            "amount": float(fields[6]),
                            "turnover": float(fields[7]) if len(fields) > 7 else 0,
                        })

                df = pd.DataFrame(rows)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    return DataResult(
                        symbol=request.symbol,
                        source=self.source,
                        frequency=DataFrequency.DAILY,
                        data=df,
                        metadata={"raw_symbol": em_code, "adjust": request.adjust},
                    ).standardize()

            # 无历史数据时, 仅当请求覆盖今天才用实时价构造当天记录
            # v3.0: 历史回测请求若构造"今天"假K线会污染回测数据, 直接返回空
            if request.end_date < date.today():
                return DataResult(
                    symbol=request.symbol,
                    source=self.source,
                    frequency=DataFrequency.DAILY,
                    data=pd.DataFrame(),
                    metadata={"error": "no_data"},
                )
            quotes = await self.get_realtime_quotes([request.symbol])
            if request.symbol in quotes:
                q = quotes[request.symbol]
                df = pd.DataFrame([{
                    "date": pd.Timestamp(date.today()),
                    "open": q["price"], "close": q["price"],
                    "high": q.get("high", q["price"]),
                    "low": q.get("low", q["price"]),
                    "volume": q.get("volume", 0),
                    "amount": q.get("amount", 0),
                    "turnover": 0,
                }])
                return DataResult(
                    symbol=request.symbol,
                    source=self.source,
                    frequency=DataFrequency.DAILY,
                    data=df,
                    metadata={"fallback": "realtime_only"},
                ).standardize()

            # 完全无数据
            return DataResult(
                symbol=request.symbol,
                source=self.source,
                frequency=DataFrequency.DAILY,
                data=pd.DataFrame(),
                metadata={"error": "no_data"},
            )

        except Exception as e:
            self._healthy = False
            raise RuntimeError(f"EastMoney获取{request.symbol}日K线失败: {e}") from e

    async def get_minute_kline(self, request: DataRequest) -> DataResult:
        """分钟K线暂不支持,返回空"""
        return DataResult(
            symbol=request.symbol,
            source=self.source,
            frequency=request.frequency,
            data=pd.DataFrame(),
            metadata={"error": "minute_kline_not_supported"},
        )

    async def get_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表 (从东方财富)"""
        try:
            url = ("https://push2.eastmoney.com/api/qt/clist/get?"
                   "pn=1&pz=5000&po=1&np=1&fltt=2&invt=2"
                   "&fid=f12&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
                   "&fields=f2,f3,f12,f14")
            resp = await asyncio.to_thread(self._session.get, url, timeout=10)
            data = resp.json()
            if data.get("data") and data["data"].get("diff"):
                rows = []
                for item in data["data"]["diff"]:
                    code = item.get("f12", "")
                    prefix = "sh." if code.startswith("6") else "sz."
                    rows.append({
                        "symbol": prefix + code,
                        "name": item.get("f14", ""),
                        "price": item.get("f2", 0),
                        "change_pct": item.get("f3", 0),
                    })
                return pd.DataFrame(rows)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    async def health_check(self) -> bool:
        """健康检查: 测试API连通性"""
        try:
            url = (f"{self.REALTIME_URL}?fltt=2"
                   f"&fields=f2,f12&secids=1.600519")
            resp = await asyncio.to_thread(self._session.get, url, timeout=5)
            data = resp.json()
            ok = (data.get("data") is not None
                  and data["data"].get("diff") is not None)
            self._healthy = ok
            return ok
        except Exception:
            self._healthy = False
            return False

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _to_em_code(symbol: str) -> Optional[str]:
        """sh.600519 → 1.600519, sz.000858 → 0.000858, bj.8xxxxx → 0.8xxxxx

        v3.0: 增加北交所支持 (EastMoney secid 用 0. 前缀)。若映射失败返回 None,
        由上层跳过该源并降级, 不再视为崩溃。
        """
        if symbol.startswith("sh."):
            return "1." + symbol[3:]
        elif symbol.startswith("sz."):
            return "0." + symbol[3:]
        elif symbol.startswith("bj."):
            return "0." + symbol[3:]  # 北交所
        return None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """sh.600519 → 600519"""
        return symbol.replace("sh.", "").replace("sz.", "")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover",
        }
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
