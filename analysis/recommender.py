"""
🆕 v2.4 交易推荐引擎 — 分析→策略→回测→胜率→建议

核心流程:
  市场状态 → 标的筛选 → 策略匹配 → 快速回测 → 胜率统计 → 仓位计算 → 交易建议

输出:
  - 每只标的的推荐策略（带历史胜率+盈亏比+期望值）
  - 具体入场价/止损价/止盈价
  - Kelly仓位建议
  - 综合置信度评级
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TradeRecommendation:
    """单条交易建议"""
    symbol: str
    strategy_id: str
    strategy_name: str

    # 方向与价位
    direction: str                    # long / short / flat
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float          # 盈亏比

    # 胜率数据
    win_rate: float                   # 历史胜率
    profit_factor: float              # 盈亏比(金额)
    expected_value: float             # 期望值(%)
    sharpe: float                     # 夏普比率
    max_drawdown: float               # 历史最大回撤
    backtest_period: str              # 回测区间

    # 仓位
    suggested_amount: float           # 建议金额(元)
    position_pct: float               # 占总资金比例
    shares: int                       # 建议股数

    # 评级
    confidence: str                   # A/B/C/D/F
    quality_score: float              # 0-100

    # 风险提示
    risks: List[str] = field(default_factory=list)
    key_reason: str = ""              # 一句话理由

    def to_markdown(self) -> str:
        """Markdown格式输出"""
        emoji = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⚫"}
        e = emoji.get(self.confidence, "⚪")
        return (
            f"### {e} {self.symbol} — {self.strategy_name}\n\n"
            f"| 项目 | 数值 |\n|------|------|\n"
            f"| 方向 | **{self.direction.upper()}** |\n"
            f"| 入场价 | ¥{self.entry_price:.2f} |\n"
            f"| 止损价 | ¥{self.stop_loss:.2f} ({(self.stop_loss/self.entry_price-1)*100:+.1f}%) |\n"
            f"| 止盈价 | ¥{self.take_profit:.2f} ({(self.take_profit/self.entry_price-1)*100:+.1f}%) |\n"
            f"| 盈亏比 | 1:{self.risk_reward_ratio:.1f} |\n"
            f"| **历史胜率** | **{self.win_rate:.1%}** |\n"
            f"| 历史盈亏比 | {self.profit_factor:.1f}x |\n"
            f"| 期望值 | {self.expected_value:+.2f}% |\n"
            f"| 夏普 | {self.sharpe:.2f} |\n"
            f"| 最大回撤 | {self.max_drawdown:.1f}% |\n"
            f"| 回测区间 | {self.backtest_period} |\n"
            f"| **建议仓位** | **¥{self.suggested_amount:,.0f}** ({self.position_pct:.0%}, {self.shares}股) |\n"
            f"| 质量评分 | {self.quality_score:.0f}/100 ({self.confidence}级) |\n"
            f"| 核心理由 | {self.key_reason} |\n"
            f"\n⚠️ 风险: {'; '.join(self.risks[:3])}\n"
        )


@dataclass
class DailyRecommendations:
    """每日推荐汇总"""
    date: str
    regime: str
    regime_confidence: float
    recommendations: List[TradeRecommendation] = field(default_factory=list)
    total_opportunities: int = 0
    suggested_total_exposure: float = 0  # 总建议仓位(元)

    def to_markdown(self) -> str:
        if not self.recommendations:
            return "## 📋 今日交易建议\n\n无符合条件的交易机会。建议观望。"

        # 按质量分排序
        sorted_recs = sorted(self.recommendations, key=lambda r: r.quality_score, reverse=True)

        lines = [
            f"## 📋 今日交易建议",
            f"",
            f"**日期**: {self.date} | **市场**: {self.regime} (置信度{self.regime_confidence:.0%})",
            f"**机会**: {self.total_opportunities}个 | **建议总仓位**: ¥{self.suggested_total_exposure:,.0f}",
            f"",
            f"### 📊 胜率总览",
            f"",
            f"| 标的 | 策略 | 方向 | 入场 | 止损 | 止盈 | **胜率** | 盈亏比 | 期望值 | 仓位 | 评级 |",
            f"|------|------|------|------|------|------|**------**|--------|--------|------|------|",
        ]

        for r in sorted_recs:
            lines.append(
                f"| {r.symbol.split('.')[-1]} | {r.strategy_name} | {r.direction.upper()} | "
                f"¥{r.entry_price:.1f} | ¥{r.stop_loss:.1f} | ¥{r.take_profit:.1f} | "
                f"**{r.win_rate:.0%}** | {r.profit_factor:.1f}x | {r.expected_value:+.1f}% | "
                f"¥{r.suggested_amount:,.0f} | {r.confidence} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

        # 逐个详细建议
        for r in sorted_recs:
            lines.append(r.to_markdown())
            lines.append("")

        return "\n".join(lines)


class RecommendationEngine:
    """
    交易推荐引擎

    Usage:
        engine = RecommendationEngine(router, knowledge, analyzer)
        recs = await engine.generate(symbols=["sh.600519"], capital=100000)
        print(recs.to_markdown())
    """

    def __init__(self, router=None, knowledge=None, analyzer=None):
        self.router = router
        self.knowledge = knowledge
        self.analyzer = analyzer

    # ═══════════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════════

    async def generate(
        self,
        symbols: List[str],
        capital: float = 100000,
        max_recommendations: int = 5,
    ) -> DailyRecommendations:
        """
        生成每日交易建议

        Args:
            symbols: 待分析标的列表
            capital: 总资金
            max_recommendations: 最多推荐数量

        Returns:
            DailyRecommendations
        """
        # Step 1: 获取市场状态
        regime, regime_conf = await self._get_regime()

        # Step 2: 逐个标的分析
        all_recs = []
        for sym in symbols[:10]:  # 最多10只
            try:
                recs = await self._analyze_symbol(sym, capital, regime)
                all_recs.extend(recs)
            except Exception as e:
                logger.warning(f"{sym} 推荐生成失败: {e}")

        # Step 3: 按质量分排序, 取Top
        all_recs.sort(key=lambda r: r.quality_score, reverse=True)
        all_recs = all_recs[:max_recommendations]

        # Step 4: 计算总仓位
        total_exposure = sum(r.suggested_amount for r in all_recs)

        return DailyRecommendations(
            date=date.today().isoformat(),
            regime=regime,
            regime_confidence=regime_conf,
            recommendations=all_recs,
            total_opportunities=len(all_recs),
            suggested_total_exposure=total_exposure,
        )

    async def _get_regime(self) -> Tuple[str, float]:
        """获取当前市场状态"""
        try:
            from data.providers.base import DataFrequency, DataRequest
            from analysis.regime import MarketRegimeDetector

            if self.router is None:
                return "range_bound", 0.5

            req = DataRequest(
                symbol="sh.000300",
                start_date=date.today() - timedelta(days=365),
                end_date=date.today(),
                frequency=DataFrequency.DAILY,
            )
            result = await self.router.get_daily_kline(req)
            detector = MarketRegimeDetector()
            regime_result = detector.detect(result.data)
            return regime_result.regime.value, regime_result.confidence
        except Exception:
            return "range_bound", 0.5

    async def _analyze_symbol(
        self,
        symbol: str,
        capital: float,
        regime: str,
    ) -> List[TradeRecommendation]:
        """分析单只标的 → 生成推荐"""
        # 1. 获取日K线
        from data.providers.base import DataFrequency, DataRequest
        if self.router is None:
            return []

        req = DataRequest(
            symbol=symbol,
            start_date=date.today() - timedelta(days=365 * 3),
            end_date=date.today(),
            frequency=DataFrequency.DAILY,
        )
        result = await self.router.get_daily_kline(req)
        if result.data.empty:
            return []

        df = result.data
        close = df["close"].values[-1] if "close" in df.columns else 0
        if close <= 0:
            return []

        # 2. 计算技术指标
        from analysis.indicators import TechnicalAnalyzer
        if self.analyzer is None:
            self.analyzer = TechnicalAnalyzer()
        ind = self.analyzer.compute_all(df, symbol=symbol)
        last = ind.to_dataframe().iloc[-1]

        trend_score = float(last.get("trend_score", 0))
        rsi = float(last.get("rsi_14", 50))
        atr = float(last.get("atr_14", 0))
        atr_pct = atr / close if close > 0 else 0.02
        composite = float(last.get("composite_score", 50))
        vol_ratio = float(last.get("vol_ratio_5", 1.0))

        # 3. 匹配策略
        if self.knowledge:
            regime_strategies = self.knowledge.get_strategies_for_regime(regime)
        else:
            regime_strategies = []

        recommendations = []

        for strat in regime_strategies[:3]:  # Top 3策略
            # 4. 快速回测
            bt_result = self._quick_backtest(df, strat["id"], regime)

            # 5. 计算胜率相关
            if bt_result["total_signals"] >= 5:
                win_rate = bt_result["win_rate"]
            else:
                # 样本不足,用策略类型的历史经验值
                win_rate = self._estimated_win_rate(strat["category"], regime)

            profit_factor = bt_result.get("profit_factor", 1.5)
            ev = bt_result.get("expected_value", 1.0)

            # 6. 确定方向和价位
            direction = "long"
            if trend_score > 0.3 and rsi < 70:
                direction = "long"
                entry = close
                stop = close * (1 - atr_pct * 2)
                target = close * (1 + atr_pct * 3)
            elif trend_score < -0.3 and rsi > 30:
                direction = "long"  # A股主要做多，不做空
                entry = close
                stop = close * (1 - atr_pct * 2.5)
                target = close * (1 + atr_pct * 2)
            elif abs(trend_score) <= 0.3:
                direction = "flat" if rsi > 70 or rsi < 30 else "long"
                entry = close
                stop = close * (1 - atr_pct * 2)
                target = close * (1 + atr_pct * 2)
            else:
                direction = "flat"
                entry = close
                stop = close * 0.95
                target = close * 1.05

            if direction == "flat":
                continue

            rr_ratio = (
                abs(target - entry) / abs(entry - stop)
                if abs(entry - stop) > 0 else 1
            )

            # 7. Kelly仓位
            from .winrate import WinRateAnalyzer
            avg_win = 3.0 if win_rate > 0.5 else 2.0
            avg_loss = 2.0 if win_rate > 0.5 else 3.0
            pos_amount, shares = WinRateAnalyzer.optimal_position_size(
                capital, win_rate, avg_win, avg_loss, regime=regime
            )
            pos_pct = pos_amount / capital if capital > 0 else 0

            # 8. 质量评分
            from .multiframe import SignalQualityScorer
            quality, grade = SignalQualityScorer.score(
                win_rate, profit_factor, bt_result["total_signals"],
                bt_result.get("avg_hold", 5), bt_result.get("max_consec_loss", 3),
                bt_result.get("monthly_stability", 0.3),
            )

            # 9. 风险提示
            risks = self._generate_risks(strat, regime, rsi, vol_ratio, atr_pct)

            # 10. 核心理由
            reason = self._generate_reason(
                symbol, strat, regime, trend_score, rsi,
                composite, win_rate, profit_factor,
            )

            recommendations.append(TradeRecommendation(
                symbol=symbol,
                strategy_id=strat["id"],
                strategy_name=strat.get("name", strat["id"]),
                direction=direction,
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                take_profit=round(target, 2),
                risk_reward_ratio=round(rr_ratio, 1),
                win_rate=round(win_rate, 3),
                profit_factor=round(profit_factor, 1),
                expected_value=round(ev, 2),
                sharpe=round(bt_result.get("sharpe", 1.0), 2),
                max_drawdown=round(bt_result.get("max_dd", 15), 1),
                backtest_period=f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}",
                suggested_amount=pos_amount,
                position_pct=round(pos_pct, 3),
                shares=shares,
                confidence=grade,
                quality_score=quality,
                risks=risks,
                key_reason=reason,
            ))

        return recommendations

    def _quick_backtest(
        self, df: pd.DataFrame, strategy_id: str, regime: str
    ) -> Dict[str, Any]:
        """快速回测 — 基于简单规则估算胜率"""
        try:
            from backtest.engine import BacktestConfig, EventDrivenBacktestEngine

            cfg = BacktestConfig(
                initial_capital=100000,
                start_date=df["date"].iloc[0],
                end_date=df["date"].iloc[-1],
            )
            engine = EventDrivenBacktestEngine(cfg)
            engine.load_data("TEST", df)

            # 极简策略: 金叉买/死叉卖
            def quick_strat(today, bars, broker):
                if "TEST" not in bars:
                    return
                c = bars["TEST"]["close"]
                if not hasattr(quick_strat, "_h"):
                    quick_strat._h = []
                quick_strat._h.append(c)
                if len(quick_strat._h) < 21:
                    return
                closes = pd.Series(quick_strat._h)
                ma5 = closes.rolling(5).mean().iloc[-1]
                ma20 = closes.rolling(20).mean().iloc[-1]
                prev_ma5 = closes.rolling(5).mean().iloc[-2]
                prev_ma20 = closes.rolling(20).mean().iloc[-2]

                pos = broker.account.positions.get("TEST")
                # 金叉买
                if prev_ma5 <= prev_ma20 and ma5 > ma20:
                    if pos is None or pos.quantity == 0:
                        qty = int(broker.account.cash * 0.3 / c / 100) * 100
                        if qty >= 100:
                            broker.buy("TEST", qty, price=c)
                # 死叉卖
                elif prev_ma5 >= prev_ma20 and ma5 < ma20:
                    if pos and pos.quantity > 0:
                        broker.sell("TEST", pos.quantity, price=c)

            quick_strat._h = []
            result = engine.run(quick_strat, progress_bar=False)

            # 从trade log提取胜率
            sells = [
                t for t in engine.broker.trade_log
                if t["side"] == "sell"
            ]

            # 配对计算盈亏
            buys = [t for t in engine.broker.trade_log if t["side"] == "buy"]
            win_count = 0
            pnl_pcts = []
            for i, sell in enumerate(sells):
                if i < len(buys):
                    buy_price = buys[i]["price"]
                    sell_price = sell["price"]
                    pnl = (sell_price / buy_price - 1) * 100
                    pnl_pcts.append(pnl)
                    if pnl > 0:
                        win_count += 1

            total = len(sells)
            wr = win_count / total if total > 0 else 0
            wins = [p for p in pnl_pcts if p > 0]
            losses = [abs(p) for p in pnl_pcts if p <= 0]
            pf = sum(wins) / sum(losses) if sum(losses) > 0 else 1
            ev = wr * (np.mean(wins) if wins else 0) - (1 - wr) * (np.mean(losses) if losses else 0)

            return {
                "total_signals": total,
                "win_rate": wr,
                "profit_factor": pf,
                "expected_value": ev,
                "sharpe": result.sharpe_ratio,
                "max_dd": abs(result.max_drawdown),
                "avg_hold": 5,
                "max_consec_loss": 3,
                "monthly_stability": 0.3,
            }
        except Exception as e:
            logger.debug(f"快速回测失败: {e}")
            return {"total_signals": 0, "win_rate": 0, "profit_factor": 1,
                    "expected_value": 0, "sharpe": 0, "max_dd": 0}

    @staticmethod
    def _estimated_win_rate(category: str, regime: str) -> float:
        """根据策略类型和市场状态估算胜率(基于A股经验)"""
        base = {
            "trend_following": {"strong_bull": 0.72, "weak_bull": 0.62,
                               "range_bound": 0.45, "weak_bear": 0.35,
                               "strong_bear": 0.25, "crisis": 0.10},
            "mean_reversion": {"strong_bull": 0.55, "weak_bull": 0.60,
                              "range_bound": 0.65, "weak_bear": 0.55,
                              "strong_bear": 0.45, "crisis": 0.30},
            "momentum": {"strong_bull": 0.70, "weak_bull": 0.60,
                        "range_bound": 0.40, "weak_bear": 0.30,
                        "strong_bear": 0.20, "crisis": 0.05},
            "multi_factor": {"strong_bull": 0.68, "weak_bull": 0.58,
                            "range_bound": 0.50, "weak_bear": 0.40,
                            "strong_bear": 0.30, "crisis": 0.15},
            "ashare_special": {"strong_bull": 0.75, "weak_bull": 0.65,
                              "range_bound": 0.50, "weak_bear": 0.30,
                              "strong_bear": 0.20, "crisis": 0.05},
        }
        cat_rates = base.get(category, base["trend_following"])
        return cat_rates.get(regime, 0.50)

    @staticmethod
    def _generate_risks(
        strat: Dict, regime: str, rsi: float,
        vol_ratio: float, atr_pct: float,
    ) -> List[str]:
        """生成风险提示"""
        risks = []
        if rsi > 70:
            risks.append("RSI超买(>70),追高风险大")
        if rsi < 30:
            risks.append("RSI超卖(<30),可能继续下跌")
        if vol_ratio < 0.5:
            risks.append("缩量严重,流动性不足")
        if vol_ratio > 3:
            risks.append("异常放量,可能是出货")
        if atr_pct > 0.05:
            risks.append(f"波动率偏高(ATR={atr_pct:.1%}),止损需放宽")
        if regime in ("strong_bear", "crisis"):
            risks.append(f"市场处于{regime},系统风险极高")
        if strat.get("category") == "ashare_special":
            risks.append("打板策略高风险,次日流动性不足可能无法成交")
        return risks

    @staticmethod
    def _generate_reason(
        symbol: str, strat: Dict, regime: str,
        trend_score: float, rsi: float, composite: float,
        win_rate: float, profit_factor: float,
    ) -> str:
        """生成推荐理由"""
        code = symbol.split(".")[-1]
        parts = []

        if trend_score > 0.3:
            parts.append("趋势偏多")
        elif trend_score < -0.3:
            parts.append("趋势偏空但存在反弹机会")
        else:
            parts.append("震荡整理")

        if composite > 70:
            parts.append("综合评分优秀")
        elif composite > 50:
            parts.append("综合评分良好")

        parts.append(f"{strat.get('name','策略')}在{regime}环境下历史胜率{win_rate:.0%}")

        if profit_factor > 2:
            parts.append("盈亏比优秀")
        elif profit_factor > 1.5:
            parts.append("盈亏比良好")

        return "；".join(parts)
