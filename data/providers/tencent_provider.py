"""
腾讯行情数据提供器 — 实时行情 (免费, 无频率限制)

腾讯证券行情API (gtimg.cn) 作为实时报价的备用/主用源。
EastMoney被墙时使用此源获取实时价格。

API端点:
  - 实时行情: https://qt.gtimg.cn/q=<code_list>

返回格式: v_CODE="market~name~code~current~yest_close~open~volume~..."
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import DataFrequency, DataProvider, DataRequest, DataResult, DataSource


class TencentFinanceProvider(DataProvider):
    """腾讯行情数据提供器 (实时行情, 免费无限制)"""

    QUOTE_URL = "https://qt.gtimg.cn/q="

    def __init__(self):
        super().__init__(name="Tencent", source=DataSource.TENCENT)
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.qq.com/",
        })

    # ═══════════════════════════════════════════════════════════════
    # 代码转换
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _to_tencent_code(symbol: str) -> str:
        """转换代码格式: sh.600519 → sh600519, sz.000333 → sz000333"""
        parts = symbol.split(".")
        if len(parts) == 2:
            return f"{parts[0]}{parts[1]}"
        return symbol

    # ═══════════════════════════════════════════════════════════════
    # 实时行情
    # ═══════════════════════════════════════════════════════════════

    async def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        """批量获取实时行情

        Returns:
            {symbol: {price, change_pct, high, low, volume, name}}
        """
        tc_codes = [self._to_tencent_code(s) for s in symbols]
        url = self.QUOTE_URL + ",".join(tc_codes)

        try:
            resp = await asyncio.to_thread(self._session.get, url, timeout=10)
            raw_text = resp.text
        except Exception as e:
            self._healthy = False
            raise RuntimeError(f"Tencent实时行情请求失败: {e}")

        results = {}
        for line in raw_text.strip().split("\n"):
            if not line.strip() or "=" not in line:
                continue
            try:
                # 解析: v_sh600519="1~茅台~600519~1293.35~1289.50~..."
                value_str = line.split('="', 1)[1].rstrip('";\n')
                fields = value_str.split("~")
                if len(fields) < 35:
                    continue

                market = fields[0]   # 1=沪, 51=深
                name = fields[1]
                code = fields[2]
                price = float(fields[3]) if fields[3] else 0
                yest_close = float(fields[4]) if fields[4] else 0
                open_price = float(fields[5]) if fields[5] else 0
                volume = int(fields[6]) if fields[6] else 0
                high = float(fields[33]) if len(fields) > 33 and fields[33] else 0
                low = float(fields[34]) if len(fields) > 34 and fields[34] else 0
                change_pct = round((price / yest_close - 1) * 100, 2) if yest_close > 0 else 0

                # 还原symbol
                prefix = "sh" if market == "1" else "sz"
                symbol = f"{prefix}.{code}"

                results[symbol] = {
                    "price": price,
                    "change_pct": change_pct,
                    "high": high,
                    "low": low,
                    "open": open_price,
                    "yest_close": yest_close,
                    "volume": volume,
                    "name": name,
                }
            except (IndexError, ValueError):
                continue

        self._healthy = True
        return results

    # ═══════════════════════════════════════════════════════════════
    # 日K线 (腾讯不支持K线, 返回空)
    # ═══════════════════════════════════════════════════════════════

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        """腾讯不支持历史K线, 返回空DataResult触发降级"""
        return DataResult(
            symbol=request.symbol,
            source=DataSource.TENCENT,
            frequency=request.frequency,
            data=pd.DataFrame(),
            metadata={"note": "Tencent不支持K线, 请使用Baostock"},
        )

    async def get_minute_kline(self, request: DataRequest) -> DataResult:
        return DataResult(
            symbol=request.symbol,
            source=DataSource.TENCENT,
            frequency=request.frequency,
            data=pd.DataFrame(),
            metadata={"note": "Tencent不支持分钟K线"},
        )

    # ═══════════════════════════════════════════════════════════════
    # 其他
    # ═══════════════════════════════════════════════════════════════

    async def get_stock_list(self) -> pd.DataFrame:
        return pd.DataFrame({"error": ["Tencent不支持股票列表"]})

    async def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """单接口兼容"""
        quotes = await self.get_realtime_quotes(symbols)
        rows = []
        for sym, info in quotes.items():
            rows.append({
                "symbol": sym,
                "price": info.get("price", 0),
                "change_pct": info.get("change_pct", 0),
                "high": info.get("high", 0),
                "low": info.get("low", 0),
                "volume": info.get("volume", 0),
                "name": info.get("name", ""),
            })
        return pd.DataFrame(rows)

    async def health_check(self) -> bool:
        """健康检查 — 尝试获取一只股票的实时行情"""
        try:
            quotes = await self.get_realtime_quotes(["sh.600519"])
            return len(quotes) > 0 and quotes.get("sh.600519", {}).get("price", 0) > 0
        except Exception:
            return False

    def _validate_response(self, df: pd.DataFrame, request: DataRequest) -> bool:
        return not df.empty
