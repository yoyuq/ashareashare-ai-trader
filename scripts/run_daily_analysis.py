#!/usr/bin/env python3
"""
每日分析流水线 — A股智能分析Agent (v2.8)

完整的自动化分析流程。Pipeline 做编排，Workflow 做引擎。

使用:
    # LLM模式 (需要DeepSeek API)
    python scripts/run_daily_analysis.py

    # 无LLM模式 (纯规则引擎, 快速)
    python scripts/run_daily_analysis.py --no-llm

    # 指定标的
    python scripts/run_daily_analysis.py --symbols sh.600519,sz.300750

    # 跳过通知推送
    python scripts/run_daily_analysis.py --no-notify
"""

import asyncio
import json
import sys
import time
from argparse import ArgumentParser
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


from scripts.shared import load_watchlist, NAME_MAP, resolve_name, update_fallback_prices


class DailyPipeline:
    """
    每日分析流水线 (v2.8 重构)

    架构: Pipeline(编排) → Workflow(引擎) → Notification(推送)
    LLM模式: AnalysisWorkflow 端到端(含逐股辩论)
    无LLM模式: 规则引擎 + 模板报告
    """

    def __init__(self, symbols: List[str], enable_notify: bool = True,
                 force_no_llm: bool = False):
        self.symbols = symbols
        self.enable_notify = enable_notify
        self.force_no_llm = force_no_llm
        self.today = date.today().isoformat()
        self.started_at = datetime.now()

        self.router = None
        self.knowledge = None
        self.analyzer = None
        self.detector = None
        self.model_router = None
        self.notifications = None

    async def run(self) -> Dict[str, Any]:
        """运行完整流水线"""
        logger.info(f"═══ 每日分析流水线启动 ═══")
        logger.info(f"日期: {self.today} | 标的: {len(self.symbols)}只 | "
                   f"LLM: {'禁用' if self.force_no_llm else '自动'}")

        result = {"date": self.today, "symbols_count": len(self.symbols),
                  "stages": {}, "errors": []}

        # Stage 1: 初始化
        await self._init_components()
        result["stages"]["init"] = "ok"

        # Stage 2: 市场状态
        regime = await self._detect_regime()
        result["market_regime"] = regime
        result["stages"]["regime"] = "ok"

        # Stage 3: 核心分析 — 用 Workflow 做引擎
        wf_result = await self._run_engine(regime)
        result["stages"]["analysis"] = wf_result.get("status", "done")
        result.update({
            "analysis": wf_result.get("analysis", {}),
            "debate": wf_result.get("debate", {}),
            "stock_recommendations": wf_result.get("stock_recommendations", {}),
            "report": wf_result.get("report", ""),
            "cost_total": self._get_cost(),
        })

        # Stage 4: 通知
        if self.enable_notify:
            await self._send_notifications(result)
            result["stages"]["notify"] = "ok"

        # Stage 5: 归档
        self._archive_report(result)

        elapsed = (datetime.now() - self.started_at).total_seconds()
        result["elapsed_seconds"] = round(elapsed, 1)
        logger.info(f"═══ 流水线完成 ({elapsed:.1f}s) ═══")
        return result

    # ──── Stage 1: 初始化 ────

    async def _init_components(self):
        logger.info("[1/5] 初始化组件...")
        from data.router import get_data_router
        self.router = get_data_router()
        logger.info("  OK DataRouter")

        from knowledge.manager import KnowledgeManager
        self.knowledge = KnowledgeManager()
        logger.info("  OK KnowledgeManager")

        from analysis.indicators import TechnicalAnalyzer
        self.analyzer = TechnicalAnalyzer()
        logger.info("  OK TechnicalAnalyzer")

        from analysis.regime import MarketRegimeDetector
        self.detector = MarketRegimeDetector()
        logger.info("  OK MarketRegimeDetector")

        if not self.force_no_llm:
            try:
                from models.router import ModelRouter
                self.model_router = ModelRouter()
                logger.info("  OK ModelRouter")
            except Exception as e:
                logger.warning(f"  ModelRouter不可用, 使用规则引擎: {e}")

        if self.enable_notify:
            from notifications import NotificationService
            self.notifications = NotificationService()
            logger.info(f"  OK NotificationService ({self.notifications.active_channels})")

    # ──── Stage 2: 市场状态 ────

    # v2.10: 多指数投票, 解决单指数置信度低的问题
    REGIME_INDICES = [
        ("sh.000001", "上证综指"),
        ("sz.399001", "深证成指"),
        ("sz.399006", "创业板指"),
        ("sh.000688", "科创50"),
        ("sh.000300", "沪深300"),
        ("sh.000905", "中证500"),
    ]

    async def _detect_regime(self) -> Dict:
        logger.info("[2/5] 检测市场状态 (多指数投票)...")
        from data.providers.base import DataFrequency, DataRequest

        async def _detect_one(sym: str, name: str) -> Optional[Tuple[str, float]]:
            try:
                req = DataRequest(sym, date.today() - timedelta(days=365),
                                date.today(), DataFrequency.DAILY, adjust="qfq")
                result = await asyncio.wait_for(
                    self.router.get_daily_kline(req), timeout=15
                )
                if result.data.empty or len(result.data) < 60:
                    return None
                r = self.detector.detect(result.data)
                return (r.regime.value, r.confidence)
            except Exception as e:
                logger.debug(f"  {name}({sym}) 检测失败: {e}")
                return None

        # 并发获取6个指数
        tasks = [_detect_one(sym, name) for sym, name in self.REGIME_INDICES]
        results = await asyncio.gather(*tasks)

        # 投票
        votes = {}
        valid = 0
        all_scores = {}
        for (sym, name), result in zip(self.REGIME_INDICES, results):
            if result is None:
                continue
            regime, conf = result
            valid += 1
            votes[regime] = votes.get(regime, 0) + 1
            all_scores[name] = {"regime": regime, "confidence": round(conf, 3)}

        if not votes:
            logger.warning("  所有指数检测失败, 使用默认range_bound")
            return {"regime": "range_bound", "confidence": 0.2,
                    "index_details": {}, "error": "all indices failed"}

        # 多数投票
        winner = max(votes, key=votes.get)
        vote_conf = votes[winner] / valid  # 投票一致性作为置信度

        logger.info(f"  {winner} (votes={votes[winner]}/{valid}, "
                   f"conf={vote_conf:.0%}) | {all_scores}")

        return {
            "regime": winner,
            "confidence": round(vote_conf, 3),
            "scores": {"votes": votes, "valid_indices": valid},
            "index_details": all_scores,
            "recommendation": self._regime_advice(winner),
        }

    def _regime_advice(self, regime: str) -> str:
        return {
            "strong_bull": "建议积极布局趋势策略，仓位80%",
            "weak_bull": "精选个股做多，仓位50-70%",
            "range_bound": "均值回归策略为主，仓位30-50%",
            "weak_bear": "减仓防守，仓位20-30%",
            "strong_bear": "空仓或极轻仓",
            "crisis": "空仓观望",
        }.get(regime, "观望")

    # ──── Stage 3: 分析引擎 ────

    async def _run_engine(self, regime: Dict) -> Dict[str, Any]:
        """
        v2.8: 统一分析引擎
        LLM可用 → AnalysisWorkflow 端到端
        LLM不可用 → 规则引擎 (无依赖)
        """
        if self.model_router is not None:
            return await self._llm_mode()
        else:
            logger.info("[3/5] LLM不可用, 使用规则引擎模式")
            return await self._rules_mode(regime)

    async def _llm_mode(self) -> Dict[str, Any]:
        """LLM模式: AnalysisWorkflow 端到端"""
        logger.info("[3/5] 运行 LangGraph 分析引擎 (含逐股辩论)...")
        from agent.orchestration.workflow import AnalysisWorkflow
        wf = AnalysisWorkflow(router=self.model_router, knowledge_manager=self.knowledge)
        state = await wf.run_daily_scan(symbols=self.symbols)

        return self._extract_workflow_result(state, mode="llm")

    async def _rules_mode(self, regime: Dict) -> Dict[str, Any]:
        """规则模式: 直调分析 + 回测 + 规则推荐 (v2.9: 并发处理)"""
        from data.providers.base import DataFrequency, DataRequest
        from analysis.recommender import STRATEGY_BACKTESTERS

        regime_val = regime.get("regime", "range_bound")
        analysis = {}
        stock_recs = {}

        # 并发控制: 最多4只股票同时处理 (baostock查询已加全局锁序列化)
        semaphore = asyncio.Semaphore(4)
        total = len(self.symbols)
        completed = 0

        async def _process_one(sym: str):
            nonlocal completed
            info = {"symbol": sym, "status": "ok", "indicators": {}, "strategies": [],
                    "fundamentals": {}}
            async with semaphore:
                try:
                    req = DataRequest(sym, date.today() - timedelta(days=365*2),
                                    date.today(), DataFrequency.DAILY, adjust="qfq")
                    dr = await self.router.get_daily_kline(req)
                    df = dr.data
                    if df.empty or len(df) < 60:
                        info["status"] = "insufficient_data"; return sym, info, None

                    ind = self.analyzer.compute_all(df, symbol=sym, skip_patterns=True)
                    last = ind.to_dataframe().iloc[-1]
                    indicators = {
                        "rsi_14": round(float(last.get("rsi_14", 50)), 1),
                        "trend_score": round(float(last.get("trend_score", 0)), 2),
                        "composite_score": round(float(last.get("composite_score", 0)), 1),
                        "close": round(float(df["close"].iloc[-1]), 2),
                    }
                    info["indicators"] = indicators

                    # 🆕 v2.10: 基本面数据 (best-effort, 不阻塞)
                    fundamentals = {}
                    try:
                        from data.providers.fundamentals import FundamentalsProvider
                        fp = FundamentalsProvider()
                        val = await asyncio.wait_for(fp.get_valuation(sym), timeout=5)
                        if val:
                            fundamentals = {
                                "pe_ttm": val.get("pe_ttm"),
                                "pb": val.get("pb"),
                                "roe": val.get("roe"),
                            }
                            info["fundamentals"] = fundamentals
                    except (asyncio.TimeoutError, Exception):
                        pass  # 基本面数据获取失败不影响主流程

                    strats = self.knowledge.get_strategies_for_regime(regime_val)
                    for s in strats:  # v2.10: 移除[:5]截断, 测试所有匹配策略
                        bt_func = STRATEGY_BACKTESTERS.get(s["id"])
                        if not bt_func: continue
                        bt = bt_func(df)
                        if bt.get("signals", 0) < 5: continue  # v2.10: 2→5, 提高可靠性
                        info["strategies"].append({
                            "id": s["id"], "name": s.get("name", s["id"]),
                            "win_rate": round(bt["win_rate"], 3),
                            "sharpe": round(bt["sharpe"], 3),
                            "max_dd": round(bt["max_dd"], 1),
                            "signals": bt["signals"],
                        })

                    rec = self._rule_recommend(sym, regime_val, indicators, info["strategies"],
                                              fundamentals)
                    return sym, info, rec
                except Exception as e:
                    info["status"] = "error"; info["error"] = str(e)
                    rec = {"action": "HOLD", "conviction": 0.0,
                           "key_reasons": [f"异常: {e}"], "risks": ["分析失败"]}
                    return sym, info, rec

        # 并发执行
        tasks = [_process_one(sym) for sym in self.symbols]
        for coro in asyncio.as_completed(tasks):
            sym, info, rec = await coro
            analysis[sym] = info
            if rec:
                stock_recs[sym] = rec
            completed += 1
            if completed % 5 == 0 or completed == total:
                logger.info(f"  [{completed}/{total}] 完成...")

        buy_n = sum(1 for r in stock_recs.values() if r.get("action") == "BUY")
        sell_n = sum(1 for r in stock_recs.values() if r.get("action") == "SELL")
        logger.info(f"  完成: BUY:{buy_n} HOLD:{len(stock_recs)-buy_n-sell_n} SELL:{sell_n}")

        debate = {
            "direction": "bullish" if buy_n > sell_n else ("bearish" if sell_n > buy_n else "neutral"),
            "confidence": 0.3, "divergence": 0.5,
            "summary": f"规则引擎: {buy_n}BUY/{len(stock_recs)-buy_n-sell_n}HOLD/{sell_n}SELL",
        }
        report = self._template_report(regime, analysis, debate, stock_recs)

        return {
            "status": f"规则引擎完成 (BUY:{buy_n}/SELL:{sell_n})",
            "analysis": analysis, "debate": debate,
            "stock_recommendations": stock_recs, "report": report,
        }

    def _extract_workflow_result(self, state, mode: str) -> Dict[str, Any]:
        """从 Workflow state 提取结果"""
        report = state.get("final_report", "")
        debate = state.get("debate_result", {})
        stock_recs = state.get("stock_recommendations", {})
        matches = state.get("strategy_matches", {})

        analysis = {}
        for sym in self.symbols:
            analysis[sym] = {
                "symbol": sym, "status": "ok" if matches.get(sym) else "no_data",
                "strategies": [{"id": m["strategy"], "name": m.get("name", m["strategy"]),
                               "fit_score": m.get("fit_score", 0)} for m in matches.get(sym, [])],
                "recommendation": stock_recs.get(sym, {}),
            }

        buy_n = sum(1 for r in stock_recs.values() if r.get("action") == "BUY")
        hold_n = sum(1 for r in stock_recs.values() if r.get("action") == "HOLD")
        sell_n = sum(1 for r in stock_recs.values() if r.get("action") == "SELL")

        for err in state.get("errors", []):
            logger.warning(f"  WARN {err.get('node', '?')}: {err.get('error', '?')}")

        return {
            "status": f"{mode}完成 (BUY:{buy_n} HOLD:{hold_n} SELL:{sell_n})",
            "analysis": analysis, "debate": debate,
            "stock_recommendations": stock_recs, "report": report,
        }

    def _rule_recommend(
        self, sym: str, regime: str, ind: dict, strategies: list,
        fundamentals: dict = None,
    ) -> dict:
        """v2.10 规则推荐引擎: 多因子加权 + 基本面校验 → BUY/HOLD/SELL"""
        rsi = ind.get("rsi_14", 50)
        trend = ind.get("trend_score", 0)
        fundamentals = fundamentals or {}
        composite = ind.get("composite_score", 50)

        reasons, risks = [], []
        # ── 多因子评分 (0-100) ──
        bull_score = 50.0  # 起始中性
        bear_score = 50.0

        # 因子1: 综合评分 (百分位, 50=中位数)
        if composite >= 80:
            bull_score += 12; reasons.append(f"综合{composite:.0f}分(前20%)")
        elif composite >= 65:
            bull_score += 7; reasons.append(f"综合{composite:.0f}分(偏强)")
        elif composite >= 50:
            bull_score += 2
        elif composite < 30:
            bear_score += 10; risks.append(f"综合仅{composite:.0f}分(后30%)")
        elif composite < 40:
            bear_score += 5; risks.append(f"综合偏低({composite:.0f}分)")

        # 因子2: 趋势强度
        if trend > 0.5:
            bull_score += 15; reasons.append(f"趋势强劲({trend:.2f})")
        elif trend > 0.2:
            bull_score += 8; reasons.append(f"趋势偏多({trend:.2f})")
        elif trend > 0:
            bull_score += 3
        elif trend < -0.5:
            bear_score += 15; risks.append(f"趋势走弱({trend:.2f})")
        elif trend < -0.2:
            bear_score += 8; risks.append(f"趋势偏空({trend:.2f})")
        elif trend < 0:
            bear_score += 3

        # 因子3: RSI 位置
        if rsi < 25:
            bull_score += 12; reasons.append(f"RSI={rsi:.0f}深度超卖")
        elif rsi < 35:
            bull_score += 7; reasons.append(f"RSI={rsi:.0f}超卖区间")
        elif rsi < 45:
            bull_score += 3
        elif rsi > 80:
            bear_score += 18; risks.append(f"RSI={rsi:.0f}极度超买")
        elif rsi > 72:
            bear_score += 10; risks.append(f"RSI={rsi:.0f}超买区间")
        elif rsi > 62:
            bear_score += 3

        # 因子4: 策略回测表现 (取最优 + 次优)
        if strategies:
            sorted_strats = sorted(strategies, key=lambda s: s["win_rate"] * max(s["sharpe"], 0.1), reverse=True)
            best = sorted_strats[0]

            # 最佳策略 — 用 OR 替代 AND, 分级加分
            wr_bonus = 0
            if best["win_rate"] >= 0.65:
                wr_bonus += 8
            elif best["win_rate"] >= 0.55:
                wr_bonus += 5
            elif best["win_rate"] >= 0.45:
                wr_bonus += 2
            elif best["win_rate"] < 0.30:
                bear_score += 5; risks.append(f"{best['name']}胜率仅{best['win_rate']:.0%}")

            sr_bonus = 0
            if best["sharpe"] >= 2.0:
                sr_bonus += 6
            elif best["sharpe"] >= 1.0:
                sr_bonus += 3
            elif best["sharpe"] >= 0.5:
                sr_bonus += 1
            elif best["sharpe"] < -0.5:
                bear_score += 3

            total_bonus = wr_bonus + sr_bonus
            if total_bonus >= 10:
                bull_score += total_bonus
                reasons.append(f"{best['name']}胜率{best['win_rate']:.0%}夏普{best['sharpe']:.1f}")
            elif total_bonus >= 5:
                bull_score += total_bonus
                reasons.append(f"{best['name']}胜率{best['win_rate']:.0%}(中等)")

            # 次优策略加分
            if len(sorted_strats) >= 2:
                second = sorted_strats[1]
                if second["win_rate"] >= 0.50 and second["sharpe"] >= 0.8:
                    bull_score += 3

        # 🆕 v2.10: 策略-市场状态适配度
        # 震荡市均值回归策略加分、趋势策略减分; 趋势市反之
        if strategies:
            best_id = best.get("id", "")
            best_category = ""  # 从策略注册表推断类别
            if "bollinger" in best_id or "rsi" in best_id or "reversal" in best_id:
                best_category = "mean_reversion"
            elif "macd" in best_id or "trend" in best_id or "turtle" in best_id:
                best_category = "trend_following"
            elif "momentum" in best_id or "breakout" in best_id:
                best_category = "momentum"

            regime_fit_bonus = 0
            if regime == "range_bound":
                if best_category == "mean_reversion":
                    regime_fit_bonus = 8  # 震荡市均值回归天然适配
                elif best_category == "trend_following":
                    regime_fit_bonus = -5  # 震荡市趋势策略容易反复止损
            elif regime in ("strong_bull", "weak_bull"):
                if best_category == "trend_following":
                    regime_fit_bonus = 6  # 趋势市趋势策略适配
                elif best_category == "mean_reversion":
                    regime_fit_bonus = -4  # 强趋势市均值回归容易逆势亏损

            bull_score += regime_fit_bonus
            if regime_fit_bonus > 0:
                reasons.append(f"{regime}适配+{regime_fit_bonus}")
            elif regime_fit_bonus < 0:
                risks.append(f"{regime}不适配{best.get('name','')}({regime_fit_bonus})")

        # 🆕 v2.10: 因子4.5 基本面校验 (best-effort, 无数据时跳过)
        if fundamentals:
            pe = fundamentals.get("pe_ttm")
            pb = fundamentals.get("pb")
            roe = fundamentals.get("roe")

            if pe is not None:
                if pe < 0:
                    bear_score += 10; risks.append(f"PE={pe:.0f}(亏损)")
                elif pe > 100:
                    bear_score += 8; risks.append(f"PE={pe:.0f}(极度高估)")
                elif pe > 60:
                    bear_score += 4; risks.append(f"PE={pe:.0f}(偏高)")
                elif pe < 10 and pe > 0:
                    bull_score += 5; reasons.append(f"PE={pe:.1f}(深度价值)")

            if pb is not None:
                if pb < 0.5:
                    bear_score += 5; risks.append(f"PB={pb:.2f}(可能价值陷阱)")
                elif pb < 1.0:
                    bear_score += 2  # 破净, 轻度关注

            if roe is not None:
                if roe > 20:
                    bull_score += 6; reasons.append(f"ROE={roe:.0f}%(高盈利)")
                elif roe > 15:
                    bull_score += 3; reasons.append(f"ROE={roe:.0f}%(良好)")
                elif roe < 0:
                    bear_score += 8; risks.append(f"ROE={roe:.0f}%(亏损)")

        # 因子5: 市场状态调节
        regime_modifiers = {
            "strong_bull": (1.15, 0.85),   # 做多加分, 做空减分
            "weak_bull": (1.08, 0.92),
            "range_bound": (1.0, 1.0),      # 中性
            "weak_bear": (0.90, 1.10),
            "strong_bear": (0.80, 1.20),
            "crisis": (0.65, 1.35),
        }
        bull_mult, bear_mult = regime_modifiers.get(regime, (1.0, 1.0))
        bull_score *= bull_mult
        bear_score *= bear_mult

        # ── 判定 (v2.9: 校准后阈值) ──
        net = bull_score - bear_score
        if net >= 20:
            action = "BUY"; conviction = min(0.60 + net * 0.007, 0.90)
        elif net >= 12:
            action = "BUY"; conviction = min(0.50 + net * 0.006, 0.75)
        elif net >= 7:
            action = "BUY"; conviction = min(0.43 + net * 0.005, 0.58)
        elif net <= -20:
            action = "SELL"; conviction = min(0.60 + abs(net) * 0.007, 0.90)
        elif net <= -12:
            action = "SELL"; conviction = min(0.50 + abs(net) * 0.006, 0.75)
        elif net <= -8:
            action = "SELL"; conviction = min(0.43 + abs(net) * 0.005, 0.58)
        else:
            # HOLD 区间: net ∈ (-8, 7)
            action = "HOLD"; conviction = 0.25 + abs(net) * 0.03

        if not reasons: reasons.append("技术指标中性")
        if not risks: risks.append("无显著风险信号")

        # 按回测质量排序取最优策略 (v2.10: 修复策略选择)
        if strategies:
            sorted_s = sorted(strategies,
                key=lambda s: s.get("win_rate", 0) * max(s.get("sharpe", 0.1), 0.1),
                reverse=True)
            best_s = sorted_s[0]
        else:
            best_s = {}

        return {
            "action": action, "conviction": round(min(conviction, 0.95), 2),
            "score": round(composite / 10, 1),
            "key_reasons": reasons[:4], "risks": risks[:3],
            "bull_quality": round(bull_score / 20, 1),
            "bear_quality": round(bear_score / 20, 1),
            "verdict_summary": f"规则: {action}({conviction:.0%}) net={net:+.0f}",
            "best_strategy_id": best_s.get("id", ""),
            "best_strategy_name": best_s.get("name", ""),
            "best_strategy_win_rate": best_s.get("win_rate", 0),
            "best_strategy_sharpe": best_s.get("sharpe", 0),
        }

    # ──── Stage 4: 通知 ────

    async def _send_notifications(self, result: Dict):
        logger.info("[4/5] 推送通知...")
        if not self.notifications: return
        try:
            from notifications import DailySummary
            stock_recs = result.get("stock_recommendations", {})
            picks = []
            for sym, rec in stock_recs.items():
                info = result.get("analysis", {}).get(sym, {})
                ind = info.get("indicators", {}) if isinstance(info, dict) else {}
                picks.append({
                    "symbol": sym.replace("sh.", "").replace("sz.", ""),
                    "name": NAME_MAP.get(sym, sym),
                    "action": rec.get("action", "HOLD"),
                    "score": ind.get("composite_score", 0),
                    "strategy": "",
                    "win_rate": 0,
                })
            picks.sort(key=lambda p: p["score"], reverse=True)

            debate = result.get("debate", {})
            buy_n = sum(1 for p in picks if p["action"] == "BUY")
            s = DailySummary(
                date=self.today,
                regime=result.get("market_regime", {}).get("regime", "unknown"),
                regime_confidence=result.get("market_regime", {}).get("confidence", 0),
                total_scanned=len(self.symbols), recommendations_count=buy_n,
                top_picks=picks[:5],
                debate_direction=debate.get("direction", "neutral"),
                debate_confidence=debate.get("confidence", 0),
                cost_today=result.get("cost_total", 0),
            )
            await self.notifications.send_daily_summary(s)
            logger.info(f"  已推送到: {self.notifications.active_channels}")
        except Exception as e:
            logger.error(f"  失败: {e}")

    # ──── Stage 5: 归档 ────

    def _archive_report(self, result: Dict):
        rp = REPORT_DIR / f"report_{self.today}.md"
        rp.write_text(result.get("report", ""), encoding="utf-8")
        logger.info(f"  报告: {rp}")

        jp = REPORT_DIR / f"data_{self.today}.json"
        # 提取每只股票的分析指标 (含收盘价, 供模拟交易使用)
        analysis_prices = {}
        raw_analysis = result.get("analysis", {})
        for sym, info in raw_analysis.items():
            if isinstance(info, dict) and info.get("status") == "ok":
                ind = info.get("indicators", {})
                strats = info.get("strategies", [])
                # 按回测质量排序选最优 (胜率×夏普), 而非取列表第一个
                sorted_s = sorted(strats,
                    key=lambda s: s.get("win_rate", 0) * max(s.get("sharpe", 0.1), 0.1),
                    reverse=True) if strats else []
                best = sorted_s[0] if sorted_s else {}
                # 次优策略(用于多样性展示)
                second_best = sorted_s[1] if len(sorted_s) >= 2 else {}
                analysis_prices[sym] = {
                    "close": ind.get("close", 0),
                    "rsi_14": ind.get("rsi_14", 50),
                    "trend_score": ind.get("trend_score", 0),
                    "composite_score": ind.get("composite_score", 50),
                    "best_strategy_id": best.get("id", ""),
                    "best_strategy_name": best.get("name", ""),
                    "best_strategy_win_rate": best.get("win_rate", 0),
                    "best_strategy_sharpe": best.get("sharpe", 0),
                    "second_strategy_id": second_best.get("id", ""),
                    "second_strategy_name": second_best.get("name", ""),
                    "second_strategy_win_rate": second_best.get("win_rate", 0),
                    "strategies_checked": len(strats),
                    "fundamentals": info.get("fundamentals", {}),  # v2.10: 基本面数据
                }

        save = {
            "date": result["date"],
            "market_regime": result.get("market_regime"),
            "debate": result.get("debate", {}),
            "stock_recommendations": result.get("stock_recommendations", {}),
            "analysis_prices": analysis_prices,
            "data_source": getattr(self.router, 'last_used_source', 'unknown'),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "cost_total": result.get("cost_total", 0),
        }
        jp.write_text(json.dumps(save, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")

        # 🆕 v2.10: 自动更新兜底价格
        updated = update_fallback_prices(analysis_prices)
        if updated > 0:
            logger.info(f"  兜底价格已更新: {updated}只")

        # 🆕 v2.10: 更新持仓市值 (Mark-to-Market)
        mtm_updated = self._mark_to_market(analysis_prices)
        if mtm_updated > 0:
            logger.info(f"  持仓市值已更新: {mtm_updated}只")

    def _mark_to_market(self, analysis_prices: dict) -> int:
        """更新所有持仓的最新价格并记录快照"""
        try:
            from simulation.portfolio import PortfolioManager
            manager = PortfolioManager()
            state = manager.load()
            updated = 0

            for sym, pos in state.positions.items():
                ap = analysis_prices.get(sym, {})
                close = ap.get("close", 0)
                if close > 0:
                    pos.current_price = close
                    updated += 1

            if updated > 0:
                # 记录当日快照
                total_pos_value = sum(p.market_value for p in state.positions.values())
                total_value = state.cash + total_pos_value
                prev_total = state.total_value if hasattr(state, 'total_value') else state.initial_capital
                daily_pnl = total_value - prev_total

                from simulation.portfolio import DailySnapshot
                snapshot = DailySnapshot(
                    date=self.today,
                    total_value=round(total_value, 2),
                    cash=round(state.cash, 2),
                    position_value=round(total_pos_value, 2),
                    daily_pnl=round(daily_pnl, 2),
                    daily_return_pct=round(daily_pnl / prev_total * 100, 2) if prev_total > 0 else 0,
                    cumulative_return_pct=round((total_value / state.initial_capital - 1) * 100, 2),
                    positions_count=len(state.positions),
                )

                # 避免重复快照
                existing_dates = {s.date for s in (state.daily_snapshots or []) if s.date}
                if self.today not in existing_dates:
                    state.daily_snapshots.append(snapshot)

                manager.save()
            return updated
        except Exception as e:
            logger.debug(f"持仓MTM更新跳过: {e}")
            return 0

    def _get_cost(self) -> float:
        return self.model_router.daily_cost if self.model_router else 0.0

    def _template_report(self, regime: Dict, analysis: Dict,
                         debate: Dict, stock_recs: Dict) -> str:
        """模板报告 (v2.8: 含逐股推荐)"""
        lines = [
            f"# A股每日分析报告",
            f"**{self.today}** | 市场: {regime.get('regime', 'N/A')} "
            f"({regime.get('confidence', 0):.0%})",
            f"**建议**: {regime.get('recommendation', '')}",
            "",
        ]
        if debate:
            lines.append(f"## 市场研判")
            lines.append(f"方向: **{debate.get('direction', 'N/A')}** | "
                        f"置信度: {debate.get('confidence', 0):.0%}")
            lines.append("")

        if stock_recs:
            buy = [(s,r) for s,r in stock_recs.items() if r.get("action")=="BUY"]
            hold = [(s,r) for s,r in stock_recs.items() if r.get("action")=="HOLD"]
            sell = [(s,r) for s,r in stock_recs.items() if r.get("action")=="SELL"]
            lines.append(f"## 逐股推荐 ({len(stock_recs)}只)")
            for label, lst in [("🟢 买入", buy), ("⚪ 持有", hold), ("🔴 卖出", sell)]:
                if not lst: continue
                lines.append(f"\n### {label} ({len(lst)}只)")
                for sym, rec in lst:
                    code = sym.replace("sh.", "").replace("sz.", "")
                    info = analysis.get(sym, {})
                    ind = info.get("indicators", {}) if isinstance(info, dict) else {}
                    lines.append(
                        f"- **{NAME_MAP.get(sym, sym)}({code})** "
                        f"| 确信度: {rec.get('conviction', 0):.0%} "
                        f"| RSI: {ind.get('rsi_14', '-')} "
                        f"| 综合: {ind.get('composite_score', '-')}分"
                    )
                    for reason in rec.get("key_reasons", [])[:2]:
                        lines.append(f"  - {reason}")
                    for risk in rec.get("risks", [])[:1]:
                        lines.append(f"  - ⚠️ {risk}")

        lines.append(f"\n## 标的详情")
        for sym, info in analysis.items():
            if not isinstance(info, dict): continue
            if info.get("status") != "ok":
                lines.append(f"- ❌ {sym}: {info.get('status')}")
                continue
            code = sym.replace("sh.", "").replace("sz.", "")
            ind = info.get("indicators", {})
            strats = info.get("strategies", [])
            lines.append(f"\n### {NAME_MAP.get(sym, sym)}({code})")
            if ind:
                lines.append(
                    f"收盘: ¥{ind.get('close', 0):.2f} | "
                    f"RSI: {ind.get('rsi_14', '-')} | "
                    f"趋势: {ind.get('trend_score', '-')}"
                )
            for s in strats[:2]:
                lines.append(
                    f"- {s.get('name','?')}: 胜率{s.get('win_rate',0):.0%} | "
                    f"夏普{s.get('sharpe',0):.2f} | 回撤{s.get('max_dd',0):.1f}%"
                )

        lines.append(f"\n---\n⚠️ 以上分析仅供参考，不构成投资建议。")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

async def main():
    parser = ArgumentParser(description="A股智能分析Agent — 每日分析流水线")
    parser.add_argument("--symbols", type=str, default="",
                       help="逗号分隔的标的列表")
    parser.add_argument("--pool", type=str, default="all_industries",
                       help="股票池: default / tech")
    parser.add_argument("--no-llm", action="store_true",
                       help="使用规则引擎(无需DeepSeek API)")
    parser.add_argument("--no-notify", action="store_true",
                       help="跳过通知推送")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else load_watchlist(args.pool)
    if not symbols:
        logger.error("无待分析标的"); return

    pipeline = DailyPipeline(
        symbols=symbols,
        enable_notify=not args.no_notify,
        force_no_llm=args.no_llm,
    )
    result = await pipeline.run()

    print("\n" + "=" * 60)
    print(f"Daily Analysis Complete ({result['elapsed_seconds']:.1f}s)")
    print(f"  Date: {result['date']}")
    print(f"  Regime: {result.get('market_regime', {}).get('regime', 'N/A')}")
    stock_recs = result.get("stock_recommendations", {})
    buy_n = sum(1 for r in stock_recs.values() if r.get("action") == "BUY")
    sell_n = sum(1 for r in stock_recs.values() if r.get("action") == "SELL")
    print(f"  Recommendations: {buy_n} BUY / {len(stock_recs)-buy_n-sell_n} HOLD / {sell_n} SELL")
    print(f"  Cost: {result.get('cost_total', 0):.4f} CNY")
    print(f"  Report: reports/report_{result['date']}.md")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
