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

                # 还原symbol (v3.0: 北交所 8/4 开头 → bj)
                if code.startswith(("8", "4")):
                    prefix = "bj"
                else:
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
    # 日K线 (web.ifzq.gtimg.cn — v3.2 数据源韧性兜底)
    # ═══════════════════════════════════════════════════════════════

    KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    async def get_daily_kline(self, request: DataRequest) -> DataResult:
        """腾讯历史日K线 (免代理可靠) — Baostock 被封 / EastMoney 不通时的日K兜底。

        返回价量 (date/open/high/low/close/volume), 无 PE/PB (估值指标由
        Baostock 主源提供)。standardize() 统一列名 + clean_ohlcv 清洗。
        """
        code = self._to_tencent_code(request.symbol)
        fq = {"qfq": "qfq", "hfq": "hfq"}.get(request.adjust, "")
        start = request.start_date.strftime("%Y-%m-%d") if request.start_date else ""
        end = request.end_date.strftime("%Y-%m-%d") if request.end_date else ""
        # count 取范围所需 (封顶 3200), 空范围时默认 640 根
        if start and end:
            span = (request.end_date - request.start_date).days
            count = min(max(span + 30, 320), 3200)
        else:
            count = 640
        url = f"{self.KLINE_URL}?param={code},day,{start},{end},{count},{fq}"
        try:
            resp = await asyncio.to_thread(self._session.get, url, timeout=15)
            payload = resp.json()
        except Exception as e:
            self._healthy = False
            raise RuntimeError(f"Tencent日K请求失败({code}): {e}") from e

        data = payload.get("data", {}).get(code, {})
        # 键名随复权方式: qfqday / hfqday / day
        bars = data.get("qfqday") or data.get("hfqday") or data.get("day") or []
        if not bars:
            return DataResult(symbol=request.symbol, source=self.source,
                              frequency=DataFrequency.DAILY, data=pd.DataFrame(),
                              metadata={"note": "Tencent日K无数据"})
        rows = []
        for b in bars:
            try:
                rows.append({
                    "date": pd.Timestamp(b[0]),
                    "open": float(b[1]),
                    "close": float(b[2]),
                    "high": float(b[3]),
                    "low": float(b[4]),
                    "volume": float(b[5]),
                })
            except (IndexError, ValueError):
                continue
        df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
        if request.start_date:
            df = df[df["date"] >= pd.Timestamp(request.start_date)]
        if request.end_date:
            df = df[df["date"] <= pd.Timestamp(request.end_date)]
        df = df.reset_index(drop=True)
        self._healthy = True
        return DataResult(
            symbol=request.symbol, source=self.source,
            frequency=DataFrequency.DAILY, data=df,
            metadata={"note": "Tencent日K (价量, 无PE/PB)"},
        ).standardize()

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
        # v5.6 P0-6: 严格于"仅非空" — 排除错误占位表并校验 OHLC 列齐全
        # (基类还要求 >=5 行, 对短期区间过严, 故此处放宽行数但保留列/占位校验)
        if df is None or df.empty:
            return False
        if "error" in df.columns:
            return False
        required_cols = {"date", "open", "high", "low", "close"}
        return required_cols.issubset(set(df.columns))
