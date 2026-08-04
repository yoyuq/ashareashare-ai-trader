"""市场择时信号 (v3.3) — 指数绝对动量 (双动量) 做仓位 Overlay

研究 + 回测 (reports/factor_backtest_timing.md):
  指数 > MA20 → 持市场 (牛市吃满上涨, +34.8% / 夏普3.98 / 回撤2.6%);
  指数 < MA20 → 现金 (避开熊市, 2026-03-03 顶部转负、6月清仓)。
daily_runner 用它调整仓位上限: risk_off 时强制防御 (少仓/现金), risk_on 正常参与。
数据: Tencent 免代理日K (sh.000001 上证指数)。失败返回 None → 调用方保持原 regime 上限。
"""

import asyncio
from datetime import date, timedelta

import numpy as np
import pandas as pd
from loguru import logger


async def market_ma_signal(ma_window: int = 20, router=None) -> dict:
    """计算上证指数 vs MA 的择时信号 (PIT 安全: 只用历史收盘).

    Returns:
        {signal: 'risk_on'|'risk_off', index_close, ma, window, asof}
        None: 数据获取失败 (调用方降级为原 regime 上限)
    """
    try:
        from data.providers.base import DataFrequency, DataRequest
        if router is None:
            from data import get_data_router
            router = get_data_router()
        req = DataRequest(
            "sh.000001",
            date.today() - timedelta(days=ma_window * 3),
            date.today(), DataFrequency.DAILY, adjust="qfq",
        )
        res = await asyncio.wait_for(router.get_daily_kline(req), timeout=20)
        df = res.data
        if df is None or df.empty or "close" not in df.columns:
            return None
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < ma_window + 5:
            return None
        ma = close.rolling(ma_window).mean()
        last, last_ma = float(close.iloc[-1]), float(ma.iloc[-1])
        if not (np.isfinite(last) and np.isfinite(last_ma)):
            return None
        asof = str(df["date"].iloc[-1])[:10] if "date" in df.columns else "?"
        return {
            "signal": "risk_on" if last > last_ma else "risk_off",
            "index_close": round(last, 1), "ma": round(last_ma, 1),
            "window": ma_window, "asof": asof,
        }
    except Exception as e:
        logger.warning(f"市场择时信号获取失败: {e}")
        return None


def format_timing(sig: dict) -> str:
    """择时信号 → 一行可读文本 (供 LLM 上下文/日志)."""
    if not sig:
        return "市场择时: 数据不可用"
    state = "risk_on(风险开启, 可参与)" if sig["signal"] == "risk_on" else "risk_off(风险关闭, 防御/现金)"
    return (f"市场择时: 上证{sig['index_close']:.0f} vs MA{sig['window']} {sig['ma']:.0f} → "
            f"{state} (asof {sig['asof']})")
