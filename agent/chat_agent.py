"""
🆕 v2.8 对话式AI Agent — 用户问答 + 工具调用

用户可通过自然语言提问,Agent自动选择工具获取数据并回答:
  - "今天市场怎么样" → 市场总览工具
  - "600519怎么样" → 个股分析工具
  - "帮我回测双均线策略在茅台上的表现" → 回测工具
  - "哪些股票值得关注" → 市场扫描工具
  - "布林带策略的胜率是多少" → 胜率查询工具
"""

import asyncio
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


# ═══════════════════════════════════════════════════════════════
# Agent 工具定义
# ═══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_overview",
            "description": "获取当前A股市场总览,包括市场状态(牛/熊/震荡)、主要指数表现、北向资金流向、板块轮动情况",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_stock",
            "description": "深度分析指定股票,包括技术指标(RSI/MACD/均线/布林带)、K线形态、资金面、综合评分",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码,如600519(茅台),000858(五粮液),300750(宁德)"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "对指定股票和策略运行历史回测,返回胜率、盈亏比、夏普比率、最大回撤等绩效指标",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码如600519"},
                    "strategy": {"type": "string", "description": "策略ID: dual_ma_trend/macd_trend/bollinger_reversal/rsi_mean_reversion/momentum_breakout/low_volatility"},
                    "years": {"type": "integer", "description": "回测年数,默认2"},
                },
                "required": ["symbol", "strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_win_rate",
            "description": "查询指定策略在指定市场状态下的历史胜率、盈亏比、期望值等统计指标",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "description": "策略ID"},
                    "regime": {"type": "string", "description": "市场状态: strong_bull/weak_bull/range_bound/weak_bear/strong_bear"},
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_market",
            "description": "扫描全市场寻找交易机会,返回综合评分最高的股票列表及各维度得分",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "返回前N只,默认10"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_strategies",
            "description": "列出所有可用的交易策略及其适用场景",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_concept",
            "description": "解释A股交易相关的概念、术语或规则,如T+1、涨跌停、印花税、北向资金等",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "要解释的概念"},
                },
                "required": ["concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trading_strategy",
            "description": "联网搜索某个交易策略/主题,提炼成结构化候选知识(概念+陈述+类别)。只做研究,不测试不落库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "要研究的策略/主题,如'海龟交易法则'、'价值投资'、'动量策略'"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "judge_trading_strategy",
            "description": "用真实历史数据回测一条交易规则/事实,给出留/删判断(verified/rejected/inconclusive/not_yet_testable)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "规则短名,如'低PE价值投资'"},
                    "claim": {"type": "string", "description": "规则/事实陈述,如'买入PE<15且PB<2的股票长期跑赢'"},
                    "category": {"type": "string", "description": "rule(交易规则)或fact(事实/定义)"},
                },
                "required": ["concept", "claim", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_backtest_windows",
            "description": "根据策略描述,推荐应该用哪些历史回测区间来测试(结合各回放窗口的市场状态做判断,不跑回测)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "description": "策略描述,如'趋势跟踪/均线交叉/止损/价值投资'"},
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "learn_trading_strategy",
            "description": "完整学习一条策略:联网研究→真实数据回测→记录留/删结论到向量库。自定义学习入口(会持久化)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "要学习的策略/主题"},
                },
                "required": ["topic"],
            },
        },
    },
]


# 可用回放窗口 + 市场状态标注 (供 suggest_backtest_windows 做 LLM 判断)
_WINDOW_REGIME = {
    "2018-01-01_2018-12-31": "熊市(系统性下跌, 检验风控/止损降险)",
    "2018-10-01_2019-12-31": "熊转震荡(触底反弹)",
    "2019-01-01_2019-12-31": "结构性牛市(成长股占优)",
    "2020-06-01_2021-02-28": "牛市(趋势/抱团行情)",
    "2024-01-01_2024-12-31": "震荡(含显著回撤段)",
    "2025-10-08_2026-07-31": "近期震荡/结构行情",
    "2026-02-21_2026-07-31": "近期震荡",
}


# ═══════════════════════════════════════════════════════════════
# 工具执行器
# ═══════════════════════════════════════════════════════════════

class ToolExecutor:
    """执行Agent工具调用,返回结构化数据"""

    def __init__(self, router=None, knowledge=None, analyzer=None):
        self.router = router
        self.knowledge = knowledge
        self.analyzer = analyzer

    async def execute(self, name: str, args: dict) -> str:
        """
        执行工具并返回JSON字符串

        v3.0: 使用字典分发 + ToolRegistry fallback
        新增工具只需: (1)在_dispatch_map注册 (2)实现方法
        """
        try:
            # v3.0: 字典分发 (替代 if/elif 链)
            handler = self._dispatch_map.get(name)
            if handler is not None:
                return await handler(args)

            # fallback: 尝试 ToolRegistry
            from agent.tools.registry import ToolRegistry
            registry_func = ToolRegistry().get_tool(name)
            if registry_func is not None:
                return await registry_func(executor=self, **args)

            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @property
    def _dispatch_map(self) -> dict:
        """
        v3.0: 工具分发字典

        添加新工具: 在dict中添加 name → handler 映射即可
        """
        return {
            "get_market_overview": lambda args: self._market_overview(),
            "analyze_stock": lambda args: self._analyze_stock(args.get("symbol", "")),
            "run_backtest": lambda args: self._backtest(
                args.get("symbol", ""),
                args.get("strategy", "dual_ma_trend"),
                args.get("years", 2),
            ),
            "get_win_rate": lambda args: self._win_rate_query(
                args.get("strategy", ""),
                args.get("regime", "range_bound"),
            ),
            "scan_market": lambda args: self._scan(args.get("top_n", 10)),
            "list_strategies": lambda args: self._list_strategies(),
            "explain_concept": lambda args: self._explain(args.get("concept", "")),
            "search_trading_strategy": lambda args: self._search_strategy(args.get("topic", "")),
            "judge_trading_strategy": lambda args: self._judge_strategy(
                args.get("concept", ""), args.get("claim", ""), args.get("category", "rule")),
            "suggest_backtest_windows": lambda args: self._suggest_windows(args.get("strategy", "")),
            "learn_trading_strategy": lambda args: self._learn_strategy(args.get("topic", "")),
        }

    async def _market_overview(self) -> str:
        """市场总览 — 仅用快速数据"""
        try:
            from data.providers.base import DataFrequency, DataRequest
            from analysis.regime import MarketRegimeDetector

            if self.router is None:
                return json.dumps({"regime": "unknown", "note": "数据路由未初始化"}, ensure_ascii=False)

            # 只用最近60天数据(更快)
            req = DataRequest.recent("sh.000300", days=120)
            result = await self.router.get_daily_kline(req)
            detector = MarketRegimeDetector()
            regime = detector.detect(result.data)

            return json.dumps({
                "regime": regime.regime.value,
                "confidence": round(regime.confidence, 2),
                "scores": {k: round(v, 2) for k, v in regime.scores.items()},
                "recommendation": (
                    "建议积极布局" if regime.regime.value in ("strong_bull","weak_bull")
                    else "建议精选个股" if regime.regime.value == "range_bound"
                    else "建议减仓防守" if regime.regime.value in ("weak_bear",)
                    else "建议空仓观望"
                ),
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": str(e), "regime": "unknown"}, ensure_ascii=False)

    async def _analyze_stock(self, symbol: str) -> str:
        """个股分析"""
        if not symbol:
            return json.dumps({"error": "请提供股票代码"}, ensure_ascii=False)

        code = symbol.strip().replace("sh.", "").replace("sz.", "")
        sym = f"sh.{code}" if code.startswith("6") else f"sz.{code}"

        try:
            from data.providers.base import DataFrequency, DataRequest
            from analysis.indicators import TechnicalAnalyzer

            if self.router is None:
                return json.dumps({"error": "数据路由未初始化"}, ensure_ascii=False)

            req = DataRequest.recent(sym, days=365)
            result = await self.router.get_daily_kline(req)
            df = result.data

            if df.empty:
                return json.dumps({"error": f"未找到{sym}的数据"}, ensure_ascii=False)

            analyzer = self.analyzer or TechnicalAnalyzer()
            ind = analyzer.compute_all(df, symbol=sym)
            last = ind.to_dataframe().iloc[-1]

            # 形态
            patterns = [
                k for k, v in ind.patterns.items()
                if hasattr(v, 'iloc') and len(v) > 0 and v.iloc[-1] > 0
            ]

            # 信号检测
            signals = []
            rsi = float(last.get("rsi_14", 50))
            if rsi > 70: signals.append("RSI超买")
            elif rsi < 30: signals.append("RSI超卖")
            if int(last.get("macd_golden_cross", 0)) > 0: signals.append("MACD金叉")
            if int(last.get("macd_death_cross", 0)) > 0: signals.append("MACD死叉")
            vol_ratio = float(last.get("vol_ratio_5", 1))
            if vol_ratio > 2: signals.append("异常放量")
            elif vol_ratio < 0.5: signals.append("极度缩量")

            return json.dumps({
                "symbol": sym,
                "close": round(float(df["close"].iloc[-1]), 2),
                "indicators": {
                    "rsi_14": round(rsi, 1),
                    "macd_dif": round(float(last.get("macd_dif", 0)), 3),
                    "macd_hist": round(float(last.get("macd_hist", 0)), 3),
                    "ma_5": round(float(last.get("ma_5", 0)), 2),
                    "ma_20": round(float(last.get("ma_20", 0)), 2),
                    "ma_60": round(float(last.get("ma_60", 0)), 2),
                    "bias_ma20": round(float(last.get("bias_ma20", 0)), 2),
                    "atr_14": round(float(last.get("atr_14", 0)), 2),
                    "trend_score": round(float(last.get("trend_score", 0)), 2),
                    "composite_score": round(float(last.get("composite_score", 0)), 1),
                },
                "patterns": patterns[:5],
                "signals": signals,
                "vol_ratio": round(vol_ratio, 2),
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _backtest(self, symbol: str, strategy_id: str, years: int) -> str:
        """回测"""
        code = symbol.strip().replace("sh.", "").replace("sz.", "")
        sym = f"sh.{code}" if code.startswith("6") else f"sz.{code}"

        try:
            from data.providers.base import DataFrequency, DataRequest
            from analysis.recommender import STRATEGY_BACKTESTERS

            if self.router is None:
                return json.dumps({"error": "数据路由未初始化"}, ensure_ascii=False)

            req = DataRequest.recent(sym, days=365 * years)
            result = await self.router.get_daily_kline(req)
            df = result.data

            if df.empty or len(df) < 60:
                return json.dumps({"error": f"{sym}数据不足"}, ensure_ascii=False)

            bt_func = STRATEGY_BACKTESTERS.get(strategy_id)
            if bt_func is None:
                return json.dumps({"error": f"未知策略{strategy_id},可用: {list(STRATEGY_BACKTESTERS.keys())}"}, ensure_ascii=False)

            bt = bt_func(df)
            strat_name = (self.knowledge.get_strategy(strategy_id) or {}).get("name", strategy_id) if self.knowledge else strategy_id

            return json.dumps({
                "symbol": sym,
                "strategy": strat_name,
                "strategy_id": strategy_id,
                "period": f"{df['date'].iloc[0]}~{df['date'].iloc[-1]}",
                "signals": bt.get("signals", 0),
                "win_rate": round(bt.get("win_rate", 0), 3),
                "profit_factor": round(bt.get("profit_factor", 1), 2),
                "expected_value": round(bt.get("expected_value", 0), 2),
                "sharpe": round(bt.get("sharpe", 0), 3),
                "max_drawdown": round(bt.get("max_dd", 0), 1),
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _win_rate_query(self, strategy_id: str, regime: str) -> str:
        """胜率查询"""
        from analysis.recommender import RecommendationEngine

        wr = RecommendationEngine._estimated_win_rate(
            "trend_following" if "trend" in strategy_id or "ma" in strategy_id
            else ("mean_reversion" if "bollinger" in strategy_id or "rsi" in strategy_id
            else ("momentum" if "momentum" in strategy_id else "multi_factor")),
            regime,
        )

        return json.dumps({
            "strategy": strategy_id,
            "regime": regime,
            "estimated_win_rate": round(wr, 3),
            "note": "这是基于策略类型和市场状态的经验估计值,实际胜率请运行回测获取",
        }, ensure_ascii=False)

    async def _scan(self, top_n: int) -> str:
        """
        市场扫描 (v2.2 改进)

        优先尝试轻量快速扫描(数据层可用时),
        数据层不可用时返回预配置精选池作为fallback。
        """
        try:
            # 尝试轻量扫描: 对精选池中的标的做快速打分
            if self.router is not None:
                try:
                    from datetime import timedelta
                    from data.providers.base import DataFrequency, DataRequest
                    from analysis.indicators import TechnicalAnalyzer

                    analyzer = self.analyzer or TechnicalAnalyzer()

                    # 获取精选池
                    import yaml
                    from pathlib import Path
                    cfg_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
                    watchlist = []
                    try:
                        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
                        watchlist = cfg.get("watchlist", {}).get("default", [])
                    except Exception:
                        watchlist = [
                            "sh.600519", "sh.600036", "sz.000858", "sz.300750",
                            "sh.601318", "sz.002415", "sh.600276", "sz.000333",
                        ]

                    # 从共享模块加载完整名称映射 (60+只)
                    try:
                        from scripts.shared import NAME_MAP
                        name_map = NAME_MAP
                    except ImportError:
                        name_map = {
                            "sh.600519": "贵州茅台", "sh.600036": "招商银行",
                            "sz.000858": "五粮液", "sz.300750": "宁德时代",
                            "sh.601318": "中国平安", "sz.002415": "海康威视",
                            "sh.600276": "恒瑞医药", "sz.000333": "美的集团",
                        }

                    stocks = []
                    for sym in watchlist[:max(top_n * 2, 12)]:
                        try:
                            req = DataRequest(
                                sym, date.today() - timedelta(days=90),
                                date.today(), DataFrequency.DAILY,
                            )
                            result = await self.router.get_daily_kline(req)
                            df = result.data
                            if df.empty or len(df) < 20:
                                continue

                            ind = analyzer.compute_all(df, symbol=sym)
                            last = ind.to_dataframe().iloc[-1]
                            score = round(float(last.get("composite_score", 50)), 1)

                            code = sym.replace("sh.", "").replace("sz.", "")
                            stocks.append({
                                "symbol": code,
                                "name": name_map.get(sym, code),
                                "composite_score": score,
                                "rsi_14": round(float(last.get("rsi_14", 50)), 1),
                                "trend": "强势" if float(last.get("trend_score", 0)) > 0.3
                                         else "弱势" if float(last.get("trend_score", 0)) < -0.3
                                         else "震荡",
                                "note": f"综合评分{score}分",
                            })
                        except Exception:
                            continue

                    # 按综合评分排序
                    stocks.sort(key=lambda s: s["composite_score"], reverse=True)
                    top_stocks = stocks[:top_n]

                    if top_stocks:
                        return json.dumps({
                            "top_stocks": top_stocks,
                            "note": f"快速扫描完成, 基于{len(stocks)}只精选池标的的实时技术评分",
                        }, ensure_ascii=False)

                except Exception as e:
                    logger.debug(f"快速扫描失败, fallback到精选池: {e}")

            # Fallback: 预配置精选池
            import yaml
            from pathlib import Path
            cfg_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
            watchlist = []
            try:
                cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
                watchlist = cfg.get("watchlist", {}).get("default", [])
            except Exception:
                pass

            if not watchlist:
                return json.dumps({"error": "股票池为空"}, ensure_ascii=False)

            try:
                from scripts.shared import NAME_MAP as name_map
            except ImportError:
                name_map = {}

            stocks = []
            for sym in watchlist[:top_n]:
                code = sym.replace("sh.", "").replace("sz.", "")
                stocks.append({
                    "symbol": code,
                    "name": name_map.get(sym, code),
                    "composite_score": None,
                    "note": "精选池(离线模式,无实时数据)",
                })

            return json.dumps({
                "top_stocks": stocks,
                "note": "离线模式精选池。连接数据源后可获取实时评分。",
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _list_strategies(self) -> str:
        """策略列表"""
        if self.knowledge is None:
            return json.dumps({"error": "知识库未初始化"}, ensure_ascii=False)

        strategies = self.knowledge.list_strategies()
        result = []
        for s in strategies:
            result.append({
                "id": s["id"],
                "name": s["name"],
                "category": s["category"],
                "description": s.get("description", ""),
                "regimes": s.get("market_regimes", []),
            })

        return json.dumps({"strategies": result}, ensure_ascii=False)

    def _explain(self, concept: str) -> str:
        """概念解释"""
        if not concept:
            return json.dumps({"error": "请提供要查询的概念"}, ensure_ascii=False)

        # 从知识库查询
        if self.knowledge:
            glossary = self.knowledge.get_glossary(concept)
            if glossary and len(glossary) > 10:
                return json.dumps({"concept": concept, "explanation": glossary}, ensure_ascii=False)

            # 查询规则
            rule = self.knowledge.get_rule(f"trading_rules.{concept}")
            if rule:
                return json.dumps({"concept": concept, "rule": str(rule)}, ensure_ascii=False)

        return json.dumps({
            "concept": concept,
            "explanation": f"关于'{concept}'的详细解释请参考A股交易规则。常见概念: T+1(当日买次日卖)、涨跌停(±10%/20%/30%)、印花税(卖出0.05%)、北向资金(沪/深港通境外资金)。",
        }, ensure_ascii=False)

    # ── 自主学习闭环工具 (v5.11: 交互式学习入口) ──

    async def _search_strategy(self, topic: str) -> str:
        """联网研究: 搜策略 → 提炼候选知识 (不测试不落库)。"""
        if not topic:
            return json.dumps({"error": "请提供要研究的主题"}, ensure_ascii=False)
        try:
            from agent.learning.researcher import generate_candidates
            cands = await generate_candidates(topic, n=5, search=True)
            if not cands:
                return json.dumps({"topic": topic, "candidates": [],
                                   "note": "未能提炼出候选知识 (联网可能失败/主题过宽)"}, ensure_ascii=False)
            return json.dumps({
                "topic": topic,
                "candidates": [c.to_dict() for c in cands],
                "note": "以上为候选知识, 可让我用 judge_trading_strategy 逐条测试, 或 learn_trading_strategy 完整学习",
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"研究失败: {e}"}, ensure_ascii=False)

    async def _judge_strategy(self, concept: str, claim: str, category: str) -> str:
        """真实数据测试: 给一条规则/事实的留/删判断。"""
        if not concept or not claim:
            return json.dumps({"error": "请提供 concept(规则短名) 和 claim(规则陈述)"}, ensure_ascii=False)
        try:
            from agent.learning.researcher import KnowledgeCandidate
            from agent.learning import tester as T
            from scripts.learn_external import _default_data_files
            cand = KnowledgeCandidate(concept=concept, claim=claim,
                                      category=("fact" if category == "fact" else "rule"),
                                      testable=claim, source="chat_agent")
            if cand.category == "fact":
                res = await T.test_fact(cand, self.knowledge)
            else:
                res = await T.test_rule(cand, _default_data_files())
            return json.dumps({
                "concept": concept, "category": cand.category,
                "verdict": res.verdict, "reason": res.reason,
                "template": res.template, "windows_tested": res.windows_tested,
                "metric_delta": res.metric_delta,
                "note": "verdict: verified(可保留)/rejected(证伪)/inconclusive(证据不足)/not_yet_testable(暂不可测)",
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"测试失败: {e}"}, ensure_ascii=False)

    async def _suggest_windows(self, strategy: str) -> str:
        """纯 LLM 判断: 根据策略描述推荐回测区间 (不跑回测)。"""
        if not strategy:
            return json.dumps({"error": "请提供策略描述"}, ensure_ascii=False)
        try:
            win_list = "\n".join(f"- {w}: {label}" for w, label in _WINDOW_REGIME.items())
            prompt = (
                "你是回测区间推荐官。根据策略描述, 从下列历史回放窗口挑 2~3 个最相关的测试区间, 并说明理由。\n"
                "原则: 趋势/动量类 → 测牛市(趋势行情); 风控/止损/降险类 → 测熊市; 价值/均值回归类 → 测震荡或熊市。\n\n"
                f"【策略】{strategy}\n\n【可用窗口】\n{win_list}\n\n"
                "严格输出 JSON (无其他文字): {\"windows\": [\"窗口标识1\", \"窗口标识2\"], \"reasons\": \"一句话理由\"}"
            )
            from models.router import get_shared_router
            result = await get_shared_router().route(
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": strategy}],
                task_type="external_research", temperature=0.2, max_tokens=400,
                extra_body={"thinking": {"type": "disabled"}},
            )
            data = {}
            try:
                txt = (result.response or "").strip()
                s, e = txt.find("{"), txt.rfind("}")
                data = json.loads(txt[s:e + 1]) if s >= 0 and e > s else {}
            except json.JSONDecodeError:
                data = {}
            return json.dumps({
                "strategy": strategy,
                "suggested_windows": data.get("windows", []),
                "reasons": data.get("reasons", ""),
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"推荐失败: {e}"}, ensure_ascii=False)

    async def _learn_strategy(self, topic: str) -> str:
        """完整学习: 研究 → 真实测试 → 记向量库 (自定义学习入口, 会持久化)。"""
        if not topic:
            return json.dumps({"error": "请提供要学习的主题"}, ensure_ascii=False)
        try:
            from scripts.learn_external import run_learning_loop, _default_data_files
            report = await run_learning_loop([topic], _default_data_files(), dry_run=False, n=3)
            return json.dumps({
                "topic": topic,
                "learned": report.get("learned", []),
                "already_learned": report.get("already_learned", []),
                "errors": report.get("errors", []),
                "note": "learned=本次测试并记录的结论; already_learned=已学过(复用旧结论)跳过",
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"学习失败: {e}"}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 对话Agent
# ═══════════════════════════════════════════════════════════════

class ChatAgent:
    """
    对话式AI Agent (v2.9: 使用 ModelRouter 统一 LLM 调用)

    通过 ModelRouter 三层漏斗调用 DeepSeek:
      - 支持 function calling (工具调用)
      - 享受预算控制、高峰降级、成本追踪
      - 自动 fallback: Pro → Flash → 本地 (工具调用时跳过本地)
    """

    # 单源提示词 (P2-4): 权威源为 knowledge/prompts/system/chat_assistant.txt,
    # 由 _get_system_prompt() 经 KnowledgeManager 加载。此处仅保留 knowledge 不可用时的
    # 最小兜底, 不再与 .txt 维护双份完整正文 (避免漂移)。
    SYSTEM_PROMPT = (
        "你是A股智能分析助手,专业的量化分析AI。"
        "所有数值来自工具调用结果,不得编造;回答简洁有条理,用中文;"
        "涉及数据分析时先调用工具获取数据;投资建议必须标注风险提示。"
    )

    def __init__(self, router=None, knowledge=None, analyzer=None, model_router=None,
                 sessions_dir: str = "data/sessions"):
        self.executor = ToolExecutor(router, knowledge, analyzer)
        self.conversations: Dict[str, List[Dict]] = {}
        self._model_router = model_router
        self._knowledge = knowledge  # v3.0: 保留knowledge引用用于加载外部提示词
        self._sessions_dir = Path(sessions_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_sessions: set = set()

    def _get_system_prompt(self) -> str:
        """
        v3.0-competition: 从知识库加载系统提示词, fallback到硬编码默认值

        优先加载 knowledge/prompts/system/chat_assistant.txt
        """
        if self._knowledge:
            prompt = self._knowledge.get_system_prompt("chat_assistant")
            if prompt and "Prompt文件缺失" not in prompt:
                # 去除 YAML frontmatter
                if prompt.startswith("---"):
                    try:
                        end_idx = prompt.index("---", 3)
                        prompt = prompt[end_idx + 3:].strip()
                    except ValueError:
                        pass
                return prompt
        return self.SYSTEM_PROMPT

    def _session_path(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._sessions_dir / f"{safe}.json"

    def _load_session(self, session_id: str) -> List[Dict]:
        """从磁盘加载会话历史"""
        if session_id in self._loaded_sessions:
            return self.conversations.get(session_id, [])
        self._loaded_sessions.add(session_id)

        path = self._session_path(session_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    self.conversations[session_id] = data
                    logger.info(f"加载会话 {session_id}: {len(data)} 条消息")
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"会话文件损坏 {session_id}: {e}")
        return []

    def _save_session(self, session_id: str):
        """持久化会话到磁盘"""
        conv = self.conversations.get(session_id)
        if not conv:
            return
        try:
            path = self._session_path(session_id)
            serializable = []
            for msg in conv:
                m = {"role": msg["role"]}
                if msg.get("content"):
                    m["content"] = str(msg["content"])[:4000]
                if msg.get("tool_calls"):
                    m["tool_calls"] = msg["tool_calls"]
                if msg.get("tool_call_id"):
                    m["tool_call_id"] = msg["tool_call_id"]
                serializable.append(m)
            path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            logger.warning(f"保存会话失败 {session_id}: {e}")

    def _get_model_router(self):
        """懒加载 ModelRouter"""
        if self._model_router is None:
            try:
                from models.router import ModelRouter
                self._model_router = ModelRouter()
            except Exception:
                self._model_router = None
        return self._model_router

    async def chat(
        self,
        user_message: str,
        session_id: str = "default",
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        处理用户消息,返回AI回复

        Args:
            user_message: 用户输入
            session_id: 会话ID
            conversation_history: 历史对话
        """
        # 初始化会话 — 优先从磁盘加载
        if session_id not in self.conversations:
            saved = self._load_session(session_id)
            if saved:
                self.conversations[session_id] = saved
            else:
                self.conversations[session_id] = [
                    {"role": "system", "content": self._get_system_prompt()}
                ]

        history = conversation_history or self.conversations[session_id]

        # 构建消息
        messages = history + [{"role": "user", "content": user_message}]

        try:
            # 第一轮: 带 tools 的 LLM 调用
            response = await self._call_llm(messages, with_tools=True)

            # 检查是否需要调用工具
            if response.get("tool_calls"):
                tool_results = await self._execute_tools(response["tool_calls"])

                # 将工具结果注入,再次调用LLM生成最终回复
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": response["tool_calls"],
                })
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr["id"],
                        "content": tr["result"],
                    })

                final_response = await self._call_llm(messages, with_tools=False)
                reply = final_response.get("content", "抱歉,处理出错了。")
            else:
                reply = response.get("content", "抱歉,我无法回答这个问题。")

            # 保存历史
            self.conversations[session_id].append(
                {"role": "user", "content": user_message}
            )
            self.conversations[session_id].append(
                {"role": "assistant", "content": reply}
            )

            # 限制历史长度
            if len(self.conversations[session_id]) > 20:
                self.conversations[session_id] = (
                    [self.conversations[session_id][0]] +  # system prompt
                    self.conversations[session_id][-18:]    # last 9 exchanges
                )

            # 持久化到磁盘
            self._save_session(session_id)

            return reply

        except Exception as e:
            logger.error(f"Agent对话失败: {e}")
            return f"抱歉,处理您的请求时出错了: {e}\n请稍后重试或尝试其他问题。"

    async def _call_llm(self, messages: List[Dict], with_tools: bool = False) -> Dict:
        """
        通过 ModelRouter 调用 LLM (v2.9 重构)

        - 有 ModelRouter: 3-tier 漏斗 + 预算 + 降级
        - 无 ModelRouter: 直接调 DeepSeek API (保留兼容)
        """
        mr = self._get_model_router()

        if mr is not None:
            try:
                if with_tools:
                    result = await mr.route_with_tools(
                        messages=messages,
                        tools=TOOLS,
                        task_type="simple_qa",
                        max_retries=1,
                    )
                else:
                    result = await mr.route(
                        messages=messages,
                        task_type="simple_qa",
                        max_retries=1,
                    )
                return self._parse_route_result(result)
            except Exception as e:
                logger.warning(f"ModelRouter 调用失败, 降级到直连: {e}")

        # Fallback: 直接调 DeepSeek
        return await self._call_llm_direct(messages, with_tools)

    async def _call_llm_direct(self, messages: List[Dict], with_tools: bool = False) -> Dict:
        """直连 DeepSeek API (无 ModelRouter 时的 fallback)"""
        import asyncio

        def _sync_call():
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                timeout=30.0,
            )

            kwargs = {
                "model": os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2000,
            }

            if with_tools:
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            result = {}
            if choice.message.content:
                result["content"] = choice.message.content
            if choice.message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
            return result

        try:
            return await asyncio.to_thread(_sync_call)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return {"content": self._fallback_reply(messages[-1].get("content", ""))}

    @staticmethod
    def _parse_route_result(result) -> Dict:
        """从 RouteResult 解析出 ChatAgent 需要的 dict"""
        content = result.response
        # 检查是否包含 tool_calls
        if content.startswith('{"_tool_calls"'):
            import json
            try:
                data = json.loads(content)
                out = {}
                if data.get("content"):
                    out["content"] = data["content"]
                if data.get("_tool_calls"):
                    out["tool_calls"] = data["_tool_calls"]
                return out
            except json.JSONDecodeError:
                pass
        return {"content": content}

    def _fallback_reply(self, user_msg: str) -> str:
        """No-API fallback"""
        msg = user_msg.lower()
        if any(w in msg for w in ["涨","跌","市场","分析"]):
            return ("当前DeepSeek API暂时不可用。您可以:\n"
                    "1. 在Dashboard的[市场总览]面板查看实时市场数据\n"
                    "2. 在[个股分析]面板输入代码查看技术指标\n"
                    "3. 在[策略回测]面板手动运行回测\n"
                    "4. 检查.env中的DEEPSEEK_API_KEY配置")
        return ("您好!我是A股智能分析助手。\n\n"
                "当前API连接异常,请检查:\n"
                "1. .env文件中的DEEPSEEK_API_KEY\n"
                "2. 代理设置(HTTP_PROXY/HTTPS_PROXY)\n"
                "3. 网络连接\n\n"
                "其他面板(市场总览/个股分析/策略回测)仍可使用。")

    async def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls with timeout"""
        results = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            try:
                # 慢工具(联网/回测/学习)放宽到 300s; 快工具保持 15s
                _slow = {"search_trading_strategy", "judge_trading_strategy",
                         "suggest_backtest_windows", "learn_trading_strategy"}
                timeout = 300.0 if name in _slow else 15.0
                result = await asyncio.wait_for(
                    self.executor.execute(name, args), timeout=timeout
                )
            except asyncio.TimeoutError:
                result = json.dumps(
                    {"error": f"工具{name}执行超时,请稍后重试或减少数据量"},
                    ensure_ascii=False,
                )

            results.append({"id": tc["id"], "result": result})
        return results
