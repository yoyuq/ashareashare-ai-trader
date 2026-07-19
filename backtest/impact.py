"""
🆕 v2.1 冲击成本建模 (Market Impact Cost)

基于 Almgren-Chriss 平方根律:
  ΔP = η × σ × √(Q / V)

其中:
  η = 市场弹性参数 (A股经验值: 0.1~0.3)
  σ = 日波动率
  Q = 委托量(股数)
  V = 日均成交量(股数)

总冲击成本 ∝ Q^(3/2) — 超线性增长

应用:
  - 回测时用真实冲击成本替代固定滑点
  - 估算策略容量上限
  - 大单拆分建议
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ImpactCostResult:
    """冲击成本估算结果"""
    symbol: str
    trade_date: date
    order_qty: int
    daily_volume: int
    volatility: float         # 日波动率(%)
    impact_bps: float          # 冲击成本(bps)
    impact_total_pct: float    # 冲击成本(%)
    capacity_remaining: float  # 剩余容量(%)

    @property
    def is_excessive(self) -> bool:
        """冲击成本是否过高 (>50bps)"""
        return self.impact_bps > 50


class ImpactCostEngine:
    """
    冲击成本估算引擎

    使用方式:
        engine = ImpactCostEngine(eta=0.15)
        cost = engine.estimate("sh.600000", 50000, daily_bar)
        if cost.impact_bps > 30:
            print(f"冲击成本{cost.impact_bps:.1f}bps偏高,建议拆分委托")
    """

    # A股经验参数
    DEFAULT_ETA = 0.15         # 市场弹性 (主板)
    ETA_BY_BOARD = {
        "主板": 0.12,
        "创业板": 0.18,
        "科创板": 0.25,        # 科创板流动性差,冲击成本更高
        "北交所": 0.35,
    }

    def __init__(
        self,
        eta: float = DEFAULT_ETA,
        gamma: float = 0.15,        # 风险厌恶系数
        max_daily_volume_pct: float = 0.01,  # 单日最大成交量占比1%
    ):
        self.eta = eta
        self.gamma = gamma
        self.max_daily_volume_pct = max_daily_volume_pct

    # ═══════════════════════════════════════════════════════════════
    # 单笔冲击成本
    # ═══════════════════════════════════════════════════════════════

    def estimate(
        self,
        symbol: str,
        order_qty: int,
        daily_bar: pd.Series,
        board: str = "主板",
    ) -> ImpactCostResult:
        """
        估算单笔订单的冲击成本

        Args:
            symbol: 股票代码
            order_qty: 委托数量(股数)
            daily_bar: 当日行情(含 close/volume/high/low)
            board: 板块

        Returns:
            ImpactCostResult
        """
        daily_volume = daily_bar.get("volume", 0)
        close = daily_bar.get("close", 0)

        if daily_volume <= 0 or close <= 0:
            return ImpactCostResult(
                symbol=symbol,
                trade_date=date.today(),
                order_qty=order_qty,
                daily_volume=0,
                volatility=0,
                impact_bps=0,
                impact_total_pct=0,
                capacity_remaining=100,
            )

        # 日波动率估算
        if "amplitude" in daily_bar.index:
            # 有振幅字段直接用
            volatility_pct = daily_bar["amplitude"] / 100
        else:
            # (high-low)/close 近似
            volatility_pct = (
                (daily_bar.get("high", close) - daily_bar.get("low", close)) / close
            )

        # 板块调整 η
        eta = self.ETA_BY_BOARD.get(board, self.DEFAULT_ETA)

        # ===== Almgren-Chriss 平方根律 =====
        # ΔP (bps) = η × σ × √(Q/V) × 10000
        volume_ratio = order_qty / max(daily_volume, 1)
        impact_bps = eta * volatility_pct * np.sqrt(volume_ratio) * 10000

        # 总冲击成本 (%)
        impact_pct = impact_bps / 10000 * 100

        # 容量检查
        capacity_remaining = max(
            0,
            (self.max_daily_volume_pct - volume_ratio) / self.max_daily_volume_pct * 100
        )

        return ImpactCostResult(
            symbol=symbol,
            trade_date=date.today(),
            order_qty=order_qty,
            daily_volume=int(daily_volume),
            volatility=round(volatility_pct * 100, 2),
            impact_bps=round(impact_bps, 2),
            impact_total_pct=round(impact_pct, 4),
            capacity_remaining=round(capacity_remaining, 1),
        )

    # ═══════════════════════════════════════════════════════════════
    # 大单拆分建议
    # ═══════════════════════════════════════════════════════════════

    def optimal_split(
        self,
        symbol: str,
        total_qty: int,
        daily_bar: pd.Series,
        max_impact_bps: float = 30,
        board: str = "主板",
    ) -> Tuple[int, int]:
        """
        大单拆分建议: 如何把一个大单拆成多日执行

        Args:
            total_qty: 总委托量
            daily_bar: 当日行情
            max_impact_bps: 可接受的最大单日冲击(bps)

        Returns:
            (每日股数, 需要天数)
        """
        daily_volume = daily_bar.get("volume", 1)
        close = daily_bar.get("close", 1)
        volatility_pct = (daily_bar.get("high", close) - daily_bar.get("low", close)) / close
        eta = self.ETA_BY_BOARD.get(board, self.DEFAULT_ETA)

        # 反解: max_qty_per_day = V * (max_bps / (η × σ × 10000))²
        sigma_inv = eta * volatility_pct * 10000
        if sigma_inv <= 0:
            return total_qty, 1

        max_qty_per_day = daily_volume * (max_impact_bps / sigma_inv) ** 2
        max_qty_per_day = int(max_qty_per_day / 100) * 100  # 整手取整
        max_qty_per_day = max(max_qty_per_day, 100)         # 至少100股

        days_needed = max(1, int(np.ceil(total_qty / max_qty_per_day)))

        return min(max_qty_per_day, total_qty), days_needed

    # ═══════════════════════════════════════════════════════════════
    # 策略容量估算
    # ═══════════════════════════════════════════════════════════════

    def estimate_strategy_capacity(
        self,
        symbol: str,
        daily_bar: pd.Series,
        max_impact_bps: float = 30,
        board: str = "主板",
    ) -> float:
        """
        估算某策略在该标的上的资金容量上限

        Returns:
            最大委托金额(元)
        """
        daily_volume = daily_bar.get("volume", 1)
        close = daily_bar.get("close", 1)
        volatility_pct = (daily_bar.get("high", close) - daily_bar.get("low", close)) / close
        eta = self.ETA_BY_BOARD.get(board, self.DEFAULT_ETA)

        sigma_inv = eta * volatility_pct * 10000
        if sigma_inv <= 0:
            return float("inf")

        max_qty = daily_volume * (max_impact_bps / sigma_inv) ** 2
        max_amount = max_qty * close

        # 不超过日成交量的1%
        max_amount_1pct = daily_volume * self.max_daily_volume_pct * close
        return min(max_amount, max_amount_1pct)

    # ═══════════════════════════════════════════════════════════════
    # 批量估算
    # ═══════════════════════════════════════════════════════════════

    def estimate_batch(
        self,
        orders: List[Dict],
        bars: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        批量估算多笔订单的冲击成本

        Args:
            orders: [{"symbol": "sh.600000", "qty": 50000, "board": "主板"}, ...]
            bars: {symbol: daily_bar, ...}

        Returns:
            DataFrame with impact costs
        """
        results = []
        for order in orders:
            sym = order["symbol"]
            qty = order["qty"]
            board = order.get("board", "主板")

            bar = bars.get(sym)
            if bar is None:
                continue

            result = self.estimate(sym, qty, bar, board)
            results.append({
                "symbol": sym,
                "order_qty": qty,
                "daily_volume": result.daily_volume,
                "volatility_pct": result.volatility,
                "impact_bps": result.impact_bps,
                "is_excessive": result.is_excessive,
                "capacity_remaining_pct": result.capacity_remaining,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            total_impact = df["impact_bps"].sum()
            logger.info(
                f"批量冲击成本合计: {total_impact:.1f}bps, "
                f"超阈值单数: {df['is_excessive'].sum()}/{len(df)}"
            )

        return df
