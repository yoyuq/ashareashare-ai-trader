"""
🆕 v2.5 交易推荐引擎 — 重写版

核心改进:
  1. 全市场扫描 → Top N标的(非仅yaml中的8只)
  2. 每策略独立回测逻辑(非统一MA金叉)
  3. 多维度市场数据(技术+资金+动量+质量+形态)
  4. 真实胜率(基于策略实际回测,不是瞎猜)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 延迟导入避免循环依赖
from .scanner import ScanResult
from .sector_rotation import SectorHeatmap
from .northbound import NorthboundTracker


@dataclass
class TradeRecommendation:
    """单条交易建议"""
    symbol: str
    symbol_name: str = ""
    strategy_id: str = ""
    strategy_name: str = ""
    direction: str = "long"
    entry_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    risk_reward_ratio: float = 1.0

    # 胜率(来自策略实际回测)
    win_rate: float = 0
    profit_factor: float = 1.0
    expected_value: float = 0
    sharpe: float = 1.0
    max_drawdown: float = 15.0
    backtest_period: str = ""
    signal_count: int = 0          # 回测信号数

    # 仓位
    suggested_amount: float = 0
    position_pct: float = 0
    shares: int = 0

    # 评级
    confidence: str = "C"
    quality_score: float = 50

    # 多维度得分
    technical_score: float = 0
    fund_score: float = 0
    momentum_score: float = 0

    # 描述
    signals: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    key_reason: str = ""


@dataclass
class DailyRecommendations:
    """每日推荐"""
    date: str
    regime: str
    regime_confidence: float
    total_scanned: int              # 扫描总数
    total_filtered: int             # 过滤后
    recommendations: List[TradeRecommendation] = field(default_factory=list)
    suggested_total_exposure: float = 0

    def to_markdown(self) -> str:
        if not self.recommendations:
            return "## 今日交易建议\n\n无符合条件的交易机会,建议观望。"

        lines = [
            f"## 📋 今日交易建议",
            f"**{self.date}** | 市场: {self.regime} | 扫描: {self.total_scanned}→{self.total_filtered}只",
            f"**机会**: {len(self.recommendations)}个 | 总仓位: ¥{self.suggested_total_exposure:,.0f}",
            "",
            "| 标的 | 名称 | 策略 | 入场 | 止损 | 止盈 | **胜率** | 盈亏比 | 期望值 | 仓位 | 评级 |",
            "|------|------|------|------|------|------|----------|--------|--------|------|------|",
        ]
        for r in self.recommendations:
            sym = r.symbol.split(".")[-1] if "." in r.symbol else r.symbol
            lines.append(
                f"| {sym} | {r.symbol_name} | {r.strategy_name} | {r.entry_price:.1f} | "
                f"{r.stop_loss:.1f} | {r.take_profit:.1f} | **{r.win_rate:.0%}** | "
                f"{r.profit_factor:.1f}x | {r.expected_value:+.1f}% | "
                f"¥{r.suggested_amount:,.0f} | {r.confidence} |"
            )
        lines.append("")
        for r in self.recommendations:
            lines.append(
                f"### {r.symbol_name}({r.symbol.split('.')[-1]}) — {r.strategy_name}\n"
                f"入场{r.entry_price:.1f} 止损{r.stop_loss:.1f} 止盈{r.take_profit:.1f} | "
                f"**胜率{r.win_rate:.0%}** 盈亏比{r.profit_factor:.1f}x | "
                f"仓位¥{r.suggested_amount:,.0f}({r.position_pct:.0%}) | "
                f"信号:{', '.join(r.signals[:3])}\n"
                f"💡 {r.key_reason}\n"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 每策略的真实回测函数
# ═══════════════════════════════════════════════════════════════

def _backtest_ma_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """双均线趋势策略回测"""
    closes = df["close"].values
    n = len(closes)
    if n < 60:
        return {"signals": 0}

    ma10 = pd.Series(closes).rolling(10).mean().values
    ma30 = pd.Series(closes).rolling(30).mean().values

    trades = []
    in_position = False
    entry_price = 0

    for i in range(60, n):
        if not in_position and ma10[i] > ma30[i] and ma10[i-1] <= ma30[i-1]:
            in_position = True
            entry_price = closes[i]
        elif in_position and ma10[i] < ma30[i] and ma10[i-1] >= ma30[i-1]:
            pnl = (closes[i] / entry_price - 1) * 100
            trades.append(pnl)
            in_position = False
            entry_price = 0

    return _calc_bt_stats(trades)


def _backtest_macd_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """MACD趋势策略回测"""
    closes = pd.Series(df["close"].values)
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    n = len(closes)
    trades = []
    in_position = False
    entry_price = 0

    for i in range(30, n):
        if not in_position and macd.iloc[i] > signal.iloc[i] and macd.iloc[i-1] <= signal.iloc[i-1]:
            in_position = True
            entry_price = closes.iloc[i]
        elif in_position and macd.iloc[i] < signal.iloc[i] and macd.iloc[i-1] >= signal.iloc[i-1]:
            pnl = (closes.iloc[i] / entry_price - 1) * 100
            trades.append(pnl)
            in_position = False

    return _calc_bt_stats(trades)


def _backtest_bollinger_reversal(df: pd.DataFrame) -> Dict[str, Any]:
    """布林带均值回归策略回测 (v2.10: 放宽入场条件)"""
    closes = pd.Series(df["close"].values)
    ma20 = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    mid_lower = ma20 - 1 * std20  # v2.10: -1σ 辅助入场

    # RSI filter
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    rsi = 100 - (100 / (1 + gain.ewm(span=14, adjust=False).mean() /
                        (loss.ewm(span=14, adjust=False).mean() + 1e-10)))

    n = len(closes)
    trades = []
    in_position = False
    entry_price = 0

    for i in range(20, n):
        # v2.10: 三级入场 — 深度超跌(下轨+RSI<35) / 中度超跌(-1σ+RSI<40) / 浅度回调(中轨+RSI<35)
        deep_oversold = closes.iloc[i] <= lower.iloc[i] and rsi.iloc[i] < 35
        moderate_oversold = closes.iloc[i] <= mid_lower.iloc[i] and rsi.iloc[i] < 40
        shallow_pullback = closes.iloc[i] <= ma20.iloc[i] and rsi.iloc[i] < 35
        if not in_position and (deep_oversold or moderate_oversold or shallow_pullback):
            in_position = True
            entry_price = closes.iloc[i]
        elif in_position and (closes.iloc[i] >= ma20.iloc[i] or closes.iloc[i] >= upper.iloc[i]):
            pnl = (closes.iloc[i] / entry_price - 1) * 100
            trades.append(pnl)
            in_position = False
        elif in_position and (closes.iloc[i] / entry_price - 1) < -0.05:  # 5%止损
            pnl = -5.0
            trades.append(pnl)
            in_position = False

    return _calc_bt_stats(trades)


def _backtest_rsi_reversal(df: pd.DataFrame) -> Dict[str, Any]:
    """RSI均值回归回测 (v2.10: 放宽入场RSI<35)"""
    closes = pd.Series(df["close"].values)
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    rsi = 100 - (100 / (1 + gain.ewm(span=14, adjust=False).mean() /
                        (loss.ewm(span=14, adjust=False).mean() + 1e-10)))

    n = len(closes)
    trades = []
    in_position = False
    entry_price = 0

    for i in range(14, n):
        if not in_position and rsi.iloc[i] < 35:  # v2.10: 30→35, 增加信号量
            in_position = True
            entry_price = closes.iloc[i]
        elif in_position and rsi.iloc[i] > 55:
            pnl = (closes.iloc[i] / entry_price - 1) * 100
            trades.append(pnl)
            in_position = False
        elif in_position and (closes.iloc[i] / entry_price - 1) < -0.05:
            pnl = -5.0
            trades.append(pnl)
            in_position = False

    return _calc_bt_stats(trades)


def _backtest_momentum_breakout(df: pd.DataFrame) -> Dict[str, Any]:
    """动量突破策略回测"""
    closes = pd.Series(df["close"].values)
    volumes = pd.Series(df["volume"].values)
    vol_ma20 = volumes.rolling(20).mean()

    n = len(closes)
    trades = []
    in_position = False
    entry_price = 0
    hold_days = 0

    for i in range(20, n):
        if not in_position:
            high_20 = closes.iloc[i-20:i].max()
            vol_surge = volumes.iloc[i] > vol_ma20.iloc[i] * 1.3
            if closes.iloc[i] >= high_20 and vol_surge:
                in_position = True
                entry_price = closes.iloc[i]
                hold_days = 0
        else:
            hold_days += 1
            # 持有5天或下跌5%止损
            if hold_days >= 5 or (closes.iloc[i] / entry_price - 1) < -0.05:
                pnl = (closes.iloc[i] / entry_price - 1) * 100
                trades.append(pnl)
                in_position = False

    return _calc_bt_stats(trades)


def _backtest_limit_up(df: pd.DataFrame) -> Dict[str, Any]:
    """涨停板策略回测 — 只看封板质量"""
    closes = pd.Series(df["close"].values)
    highs = pd.Series(df["high"].values)
    volumes = pd.Series(df["volume"].values)

    n = len(closes)
    trades = []
    in_position = False
    entry_price = 0

    for i in range(5, n):
        prev_close = closes.iloc[i-1]
        # 检测首板: 涨停 + 封板(收盘=最高) + 非连板
        is_limit_up = (closes.iloc[i] / prev_close - 1) >= 0.095
        is_sealed = abs(closes.iloc[i] - highs.iloc[i]) < 0.01
        vol_ok = volumes.iloc[i] > volumes.iloc[i-5:i].mean() * 1.5

        if is_limit_up and is_sealed and vol_ok and not in_position:
            in_position = True
            entry_price = closes.iloc[i]
            hold_days = 0
        elif in_position:
            hold_days += 1
            if hold_days >= 3 or (closes.iloc[i] / entry_price - 1) < -0.03:
                pnl = (closes.iloc[i] / entry_price - 1) * 100
                trades.append(pnl)
                in_position = False

    return _calc_bt_stats(trades)


def _backtest_low_volatility(df: pd.DataFrame) -> Dict[str, Any]:
    """低波动策略回测 — 买入持有低波标的"""
    closes = pd.Series(df["close"].values)
    returns = closes.pct_change().dropna()

    n = len(returns)
    if n < 60:
        return {"signals": 0}

    # 每20天检查: 20日波动是否在最低30%分位
    trades = []
    in_position = False
    entry_price = 0
    hold_days = 0

    for i in range(60, n):
        if not in_position:
            hv20 = returns.iloc[i-20:i].std() * np.sqrt(252) * 100
            hv60 = returns.iloc[max(0, i-60):i].std() * np.sqrt(252) * 100
            if hv20 < hv60 * 0.7:  # 波动率显著低于历史
                in_position = True
                entry_price = closes.iloc[i]
                hold_days = 0
        else:
            hold_days += 1
            hv20_now = returns.iloc[i-20:i].std() * np.sqrt(252) * 100
            hv60_now = returns.iloc[max(0, i-60):i].std() * np.sqrt(252) * 100
            if hold_days >= 20 or hv20_now > hv60_now * 1.3:
                pnl = (closes.iloc[i] / entry_price - 1) * 100
                trades.append(pnl)
                in_position = False

    return _calc_bt_stats(trades)


def _backtest_turtle_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """🆕 v2.10 海龟趋势突破策略回测

    入场: 价格突破20日最高价 (Donchian Channel上轨)
    出场: 价格跌破10日最低价 或 ATR止损
    """
    closes = pd.Series(df["close"].values)
    highs = pd.Series(df["high"].values)
    lows = pd.Series(df["low"].values)

    n = len(closes)
    if n < 60:
        return {"signals": 0}

    # ATR(20)
    tr = pd.concat([
        (highs - lows).abs(),
        (highs - closes.shift()).abs(),
        (lows - closes.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()

    trades = []
    in_position = False
    entry_price = 0
    stop_price = 0

    for i in range(60, n):
        if not in_position:
            # 入场: 突破20日高点
            high_20 = highs.iloc[i-20:i].max()
            if closes.iloc[i] >= high_20 and atr.iloc[i] > 0:
                in_position = True
                entry_price = closes.iloc[i]
                stop_price = entry_price - 2 * atr.iloc[i]  # 2x ATR止损
        else:
            # 出场: 跌破10日低点 或 触发ATR止损
            low_10 = lows.iloc[i-10:i].min()
            hit_stop = closes.iloc[i] <= stop_price
            hit_exit = closes.iloc[i] <= low_10
            if hit_stop or hit_exit:
                pnl = (closes.iloc[i] / entry_price - 1) * 100
                trades.append(pnl)
                in_position = False

    return _calc_bt_stats(trades)


def _backtest_multi_factor(df: pd.DataFrame) -> Dict[str, Any]:
    """🆕 v2.10 多因子选股策略回测

    基于5因子综合评分 (动量+波动率+趋势+成交量+质量代理),
    每月调仓: 评分>60买入, 评分<40卖出, 或持满20天。
    """
    closes = pd.Series(df["close"].values)
    volumes = pd.Series(df["volume"].values)
    highs = pd.Series(df["high"].values)
    lows = pd.Series(df["low"].values)
    n = len(closes)
    if n < 120:
        return {"signals": 0}

    # 每月(约20个交易日)计算一次因子评分
    trades = []
    in_position = False
    entry_price = 0
    hold_days = 0

    for i in range(120, n):
        if i % 20 != 0:  # 每月调仓
            if in_position:
                hold_days += 1
            continue

        # 计算当前截面的因子评分
        window = closes.iloc[max(0, i-60):i]

        # 动量因子: 20日收益率
        mom = (closes.iloc[i] / closes.iloc[max(0, i-20)] - 1) * 100
        mom_score = 0.25 * min(max(mom, -10), 20)  # clip到[-10,20]

        # 波动率因子: 低波偏好
        rets = window.pct_change().dropna()
        hv = rets.std() * np.sqrt(252) * 100
        vol_score = 0.15 * (20 - min(hv, 40)) / 20  # 低波加分

        # 趋势因子
        sma20 = closes.iloc[i-20:i].mean()
        trend_score_raw = 0.30 * ((closes.iloc[i] / sma20 - 1) * 100)

        # 成交量因子
        vol_ratio = volumes.iloc[i] / volumes.iloc[i-20:i].mean()
        vol_score = 0.15 * min(max((vol_ratio - 1) * 5, -2), 3)

        # 质量代理: 60日最大回撤
        peak = highs.iloc[max(0, i-60):i].max()
        dd = (closes.iloc[i] - peak) / peak * 100
        qual_score = 0.15 * min(max(-dd / 5, -3), 3)

        composite = mom_score + vol_score + trend_score_raw + vol_score + qual_score

        if not in_position and composite > 0.5:  # 综合评分>0.5入场
            in_position = True
            entry_price = closes.iloc[i]
            hold_days = 0
        elif in_position and (composite < -0.3 or hold_days >= 20):
            pnl = (closes.iloc[i] / entry_price - 1) * 100
            trades.append(pnl)
            in_position = False

    return _calc_bt_stats(trades)


# 策略ID → 回测函数映射
STRATEGY_BACKTESTERS = {
    "dual_ma_trend": _backtest_ma_trend,
    "macd_trend": _backtest_macd_trend,
    "bollinger_reversal": _backtest_bollinger_reversal,
    "rsi_mean_reversion": _backtest_rsi_reversal,
    "momentum_breakout": _backtest_momentum_breakout,
    "limit_up_chase": _backtest_limit_up,
    "low_volatility": _backtest_low_volatility,
    "turtle_trend": _backtest_turtle_trend,      # 🆕 v2.10
    "multi_factor": _backtest_multi_factor,       # 🆕 v2.10
}


def _calc_bt_stats(trades: List[float]) -> Dict[str, Any]:
    """
    从交易列表计算回测统计 (v2.9: 计入交易成本)

    A股交易成本:
      - 佣金: 0.03% (双向,最低5元 → 对回测简化使用比例)
      - 印花税: 0.05% (仅卖出)
      - 滑点: 0.1% (双向)
      - 单边总成本 ≈ 0.03% + 0.1% = 0.13% (买入)
      - 单边总成本 ≈ 0.03% + 0.05% + 0.1% = 0.18% (卖出)
      - 往返总成本 ≈ 0.31% per round-trip
    """
    COST_PER_RT = 0.0031  # 往返交易成本 0.31%

    if not trades:
        return {"signals": 0, "win_rate": 0, "profit_factor": 1,
                "expected_value": 0, "sharpe": 0, "max_dd": 0,
                "avg_win": 0, "avg_loss": 0, "cost_adjusted": True}

    # 扣除交易成本
    trades_net = [t - COST_PER_RT * 100 for t in trades]  # trades 是百分比,成本也转百分比

    wins = [t for t in trades_net if t > 0]
    losses = [abs(t) for t in trades_net if t <= 0]
    total = len(trades_net)
    wr = len(wins) / total if total > 0 else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = sum(losses) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 1

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 1
    ev = wr * avg_win - (1 - wr) * avg_loss

    # Sharpe
    rets = pd.Series([t / 100 for t in trades_net])
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    # Max DD (从交易序列)
    cumsum = np.cumsum(trades_net)
    peak = np.maximum.accumulate(cumsum)
    dd = np.min(cumsum - peak)
    max_dd = abs(dd) if dd < 0 else 0

    return {
        "signals": total,
        "win_rate": wr,
        "profit_factor": pf,
        "expected_value": ev,
        "sharpe": sr,
        "max_dd": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "cost_adjusted": True,
    }


# ═══════════════════════════════════════════════════════════════
# 推荐引擎 (重写)
# ═══════════════════════════════════════════════════════════════

class RecommendationEngine:
    """v2.5 交易推荐引擎"""

    def __init__(self, router=None, knowledge=None, analyzer=None):
        self.router = router
        self.knowledge = knowledge
        self.analyzer = analyzer

    async def generate(
        self,
        capital: float = 100000,
        top_n: int = 20,
        min_composite: float = 50,
    ) -> DailyRecommendations:
        """
        生成每日交易建议

        流程: 市场状态 → 北向资金 → 板块轮动 → 全市场扫描 → 策略匹配 → 真实回测 → 信号分级 → 风控 → 输出
        """
        # Step 1-3: 并行获取市场状态+北向+板块 (3个独立任务)
        from .northbound import NorthboundTracker
        from .sector_rotation import SectorRotationAnalyzer

        nb_tracker = NorthboundTracker()
        sector_analyzer = SectorRotationAnalyzer()

        # 并行执行, 单个超时10秒
        async def safe_regime():
            try: return await asyncio.wait_for(self._get_regime(), timeout=10)
            except: return ("range_bound", 0.5)

        async def safe_northbound():
            try: return await asyncio.wait_for(nb_tracker.get_snapshot(), timeout=10)
            except: return NorthboundTracker()._empty_snapshot()

        async def safe_sector():
            try: return await asyncio.wait_for(sector_analyzer.analyze(), timeout=10)
            except: return SectorHeatmap(date=date.today().isoformat())

        (regime, regime_conf), nb_snapshot, sector_heatmap = await asyncio.gather(
            safe_regime(), safe_northbound(), safe_sector()
        )
        nb_signal = nb_snapshot.signal

        # Step 4: 全市场扫描 (已有两阶段优化)
        from .scanner import MarketScanner
        scanner = MarketScanner(router=self.router, analyzer=self.analyzer)
        try:
            scan_result = await asyncio.wait_for(scanner.scan(top_n=100), timeout=120)
        except asyncio.TimeoutError:
            logger.warning("全市场扫描超时,使用空结果")
            scan_result = ScanResult(
                scan_date=date.today().isoformat(),
                total_stocks=0, filtered_stocks=0,
            )

        if not scan_result.top_scores:
            return DailyRecommendations(
                date=date.today().isoformat(),
                regime=regime, regime_confidence=regime_conf,
                total_scanned=scan_result.total_stocks,
                total_filtered=scan_result.filtered_stocks,
            )

        # Step 3: 对Top标的逐一生成推荐
        all_recs = []
        for stock in scan_result.top_scores[:50]:
            if stock.composite < min_composite:
                continue
            recs = await self._generate_for_stock(
                stock, capital, regime,
                nb_tracker, sector_heatmap, sector_analyzer, nb_signal,
            )
            all_recs.extend(recs)
            if len(all_recs) >= top_n:
                break

        # Step 4: 按质量分排序
        all_recs.sort(key=lambda r: r.quality_score, reverse=True)
        all_recs = all_recs[:top_n]

        return DailyRecommendations(
            date=date.today().isoformat(),
            regime=regime,
            regime_confidence=regime_conf,
            total_scanned=scan_result.total_stocks,
            total_filtered=scan_result.filtered_stocks,
            recommendations=all_recs,
            suggested_total_exposure=sum(r.suggested_amount for r in all_recs),
        )

    async def _get_regime(self) -> Tuple[str, float]:
        try:
            from data.providers.base import DataFrequency, DataRequest
            if self.router is None:
                return "range_bound", 0.5
            req = DataRequest.recent("sh.000300", days=365)
            result = await self.router.get_daily_kline(req)
            from analysis.regime import MarketRegimeDetector
            r = MarketRegimeDetector().detect(result.data)
            return r.regime.value, r.confidence
        except Exception:
            return "range_bound", 0.5

    async def _generate_for_stock(
        self, stock, capital: float, regime: str,
        nb_tracker=None, sector_heatmap=None, sector_analyzer=None, nb_signal: str = "neutral",
    ) -> List[TradeRecommendation]:
        """为单只标的生成推荐 (v2.8 修复: 显式传参避免作用域bug)"""
        try:
            from data.providers.base import DataFrequency, DataRequest
            req = DataRequest.recent(stock.symbol, days=365 * 2)
            result = await self.router.get_daily_kline(req)
            df = result.data
            if df.empty or len(df) < 60:
                return []

            close = df["close"].values[-1]
            if close <= 0:
                return []

            # 匹配策略
            if self.knowledge:
                strats = self.knowledge.get_strategies_for_regime(regime)
            else:
                strats = []

            recs = []
            for strat in strats:
                sid = strat["id"]

                # 真实回测
                bt_func = STRATEGY_BACKTESTERS.get(sid)
                if bt_func is None:
                    bt_func = _backtest_ma_trend  # fallback

                bt = bt_func(df)
                if bt.get("signals", 0) < 5:
                    continue  # 信号太少,不可靠 (v2.10: 3→5)

                wr = bt["win_rate"]
                pf = bt["profit_factor"]
                ev = bt["expected_value"]
                sr = bt["sharpe"]
                avg_win = bt.get("avg_win", 3.0)
                avg_loss = bt.get("avg_loss", 2.0)

                # 仓位
                from .winrate import WinRateAnalyzer
                pos_amt, shares = WinRateAnalyzer.optimal_position_size(
                    capital, wr, max(avg_win, 1.0),
                    max(avg_loss, 1.0), regime=regime
                )
                if pos_amt <= 0:
                    continue

                # 价位
                atr = self._calc_atr(df)
                atr_pct = atr / close
                entry = close
                stop = close * (1 - atr_pct * 2.5)
                target = close * (1 + atr_pct * 3.5)
                rr = (target - entry) / (entry - stop) if entry > stop else 1.5

                # v2.6: 北向+板块调整评分
                adj_score = stock.composite
                adj_note = ""
                if nb_signal != "neutral":
                    adj_score, adj_note = nb_tracker.adjust_signal(adj_score, stock.symbol)

                if sector_heatmap.sectors:
                    industry = stock.industry or sector_heatmap.sectors[0].name
                    adj_score, s_note = sector_analyzer.adjust_score_by_sector(
                        sector_heatmap, industry, adj_score
                    )
                    adj_note += "; " + s_note if adj_note else s_note

                # 信号分级
                from .risk_controls import SignalGrader
                s_rank = sector_heatmap.get_sector_rank(stock.industry or "") if sector_heatmap.sectors else 0
                signal_card = SignalGrader.grade(
                    adj_score, wr, regime, nb_signal,
                    s_rank
                )
                grade = signal_card.level.label
                direction = signal_card.direction

                # 只保留WATCH以上 (>=50) 或 BUY以上
                if signal_card.level.value[1] < 45:
                    continue

                # 质量评分
                from .multiframe import SignalQualityScorer
                quality, _ = SignalQualityScorer.score(
                    wr, pf, bt["signals"], 6, 3, 0.25
                )
                if grade in ("D", "F"):
                    continue

                # 风险
                risks = self._gen_risks(strat, regime, stock)

                # 理由 (v2.6: 含北向+板块+信号分级)
                signals_str = ", ".join(stock.signals[:3])
                reason_parts = [
                    f"{stock.name}综合{adj_score:.0f}分({signal_card.level.emoji}{signal_card.level.label})",
                    f"{strat.get('name', sid)}胜率{wr:.0%}",
                ]
                if adj_note:
                    reason_parts.append(adj_note)
                if signals_str:
                    reason_parts.append(f"信号: {signals_str}")
                reason = "; ".join(reason_parts)

                recs.append(TradeRecommendation(
                    symbol=stock.symbol,
                    symbol_name=stock.name,
                    strategy_id=sid,
                    strategy_name=strat.get("name", sid),
                    direction=direction,
                    entry_price=round(entry, 2),
                    stop_loss=round(stop, 2),
                    take_profit=round(target, 2),
                    risk_reward_ratio=round(rr, 2),
                    win_rate=round(wr, 3),
                    profit_factor=round(pf, 2),
                    expected_value=round(ev, 2),
                    sharpe=round(sr, 3),
                    max_drawdown=round(bt["max_dd"], 1),
                    backtest_period=f"{df['date'].iloc[0]}~{df['date'].iloc[-1]}",
                    signal_count=bt["signals"],
                    suggested_amount=pos_amt,
                    position_pct=round(pos_amt/capital, 3),
                    shares=shares,
                    confidence=grade,
                    quality_score=quality,
                    technical_score=stock.technical_score,
                    fund_score=stock.fund_flow_score,
                    momentum_score=stock.momentum_score,
                    signals=stock.signals,
                    risks=risks,
                    key_reason=reason,
                ))

            return recs

        except Exception as e:
            logger.warning(f"{stock.symbol}推荐失败: {e}")
            return []

    def _calc_atr(self, df: pd.DataFrame) -> float:
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h[-14:] - l[-14:],
                        np.maximum(abs(h[-14:] - np.roll(c, 1)[-14:]),
                                   abs(l[-14:] - np.roll(c, 1)[-14:])))
        return float(np.mean(tr[-10:])) if len(tr) >= 10 else 0.5

    @staticmethod
    def _gen_risks(strat, regime, stock) -> List[str]:
        risks = []
        if regime in ("strong_bear", "crisis"):
            risks.append(f"市场{regime},系统风险极高")
        if strat.get("category") == "ashare_special":
            risks.append("打板高风险,次日流动性不足可能无法成交")
        if strat.get("category") == "trend_following" and regime == "range_bound":
            risks.append("震荡市趋势策略容易反复止损")
        return risks
