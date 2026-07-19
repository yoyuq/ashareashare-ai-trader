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
]


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
        """执行工具并返回JSON字符串"""
        try:
            if name == "get_market_overview":
                return await self._market_overview()
            elif name == "analyze_stock":
                return await self._analyze_stock(args.get("symbol", ""))
            elif name == "run_backtest":
                return await self._backtest(
                    args.get("symbol", ""),
                    args.get("strategy", "dual_ma_trend"),
                    args.get("years", 2),
                )
            elif name == "get_win_rate":
                return self._win_rate_query(
                    args.get("strategy", ""),
                    args.get("regime", "range_bound"),
                )
            elif name == "scan_market":
                return await self._scan(args.get("top_n", 10))
            elif name == "list_strategies":
                return self._list_strategies()
            elif name == "explain_concept":
                return self._explain(args.get("concept", ""))
            else:
                return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _market_overview(self) -> str:
        """市场总览 — 仅用快速数据"""
        try:
            from data.providers.base import DataFrequency, DataRequest
            from analysis.regime import MarketRegimeDetector

            if self.router is None:
                return json.dumps({"regime": "unknown", "note": "数据路由未初始化"}, ensure_ascii=False)

            # 只用最近60天数据(更快)
            req = DataRequest("sh.000300", date.today()-timedelta(days=120),
                            date.today(), DataFrequency.DAILY)
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

            req = DataRequest(sym, date.today()-timedelta(days=365),
                            date.today(), DataFrequency.DAILY)
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

            req = DataRequest(sym, date.today()-timedelta(days=365*years),
                            date.today(), DataFrequency.DAILY)
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
        """Return watchlist stocks — no API call, instant response"""
        try:
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

            name_map = {
                "sh.600519": "贵州茅台", "sh.600036": "招商银行",
                "sz.000858": "五粮液", "sz.300750": "宁德时代",
                "sh.601318": "中国平安", "sz.002415": "海康威视",
                "sh.600276": "恒瑞医药", "sz.000333": "美的集团",
                "sh.600030": "中信证券", "sz.002594": "比亚迪",
            }

            stocks = []
            for sym in watchlist[:top_n]:
                code = sym.replace("sh.", "").replace("sz.", "")
                stocks.append({
                    "symbol": code,
                    "name": name_map.get(sym, code),
                    "note": "预配置精选池",
                })

            return json.dumps({
                "top_stocks": stocks,
                "note": "预配置精选池(8只核心标的)。全市场深度扫描请使用Dashboard的[交易建议]面板。",
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


# ═══════════════════════════════════════════════════════════════
# 对话Agent
# ═══════════════════════════════════════════════════════════════

class ChatAgent:
    """
    对话式AI Agent

    使用DeepSeek V4作为推理引擎,通过function calling调用分析工具,
    用自然语言回答用户的A股相关问题。
    """

    SYSTEM_PROMPT = """你是A股智能分析助手,一个专业的量化分析AI。

你可以:
- 分析市场状态(牛熊判断)
- 对个股进行技术面深度分析(RSI/MACD/均线/布林带/K线形态)
- 运行策略历史回测,给出胜率和绩效指标
- 扫描全市场寻找评分较高的标的
- 解释A股交易规则和概念

规则:
1. 所有数值来自工具调用结果,不要编造
2. 工具返回错误时如实告知
3. 回答简洁有条理,用中文
4. 涉及数据分析时先调用工具获取数据
5. 关于"哪些股票会上涨": 你无法预测未来涨跌,只能基于技术面和历史数据给出评分和概率,必须强调风险
6. 投资建议必须标注"⚠️风险提示: 历史数据不代表未来表现"

当用户问预测性问题时,明确说明: 量化分析提供的是概率参考而非确定性预测,任何单一信号都不构成投资建议。"""

    def __init__(self, router=None, knowledge=None, analyzer=None):
        self.executor = ToolExecutor(router, knowledge, analyzer)
        self.conversations: Dict[str, List[Dict]] = {}

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
        # 初始化会话
        if session_id not in self.conversations:
            self.conversations[session_id] = [
                {"role": "system", "content": self.SYSTEM_PROMPT}
            ]

        history = conversation_history or self.conversations[session_id]

        # 构建消息
        messages = history + [{"role": "user", "content": user_message}]

        try:
            # 调用DeepSeek (支持function calling)
            response = await self._call_llm(messages)

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

                final_response = await self._call_llm(messages)
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

            return reply

        except Exception as e:
            logger.error(f"Agent对话失败: {e}")
            return f"抱歉,处理您的请求时出错了: {e}\n请稍后重试或尝试其他问题。"

    async def _call_llm(self, messages: List[Dict]) -> Dict:
        """调用DeepSeek API (同步客户端+线程池, 兼容代理)"""
        import asyncio

        def _sync_call():
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                timeout=30.0,
            )

            kwargs = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2000,
            }

            # 只在还没调过工具时传tools
            has_tool_results = any(m.get("role") == "tool" for m in messages)
            if not has_tool_results:
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
                # 15s timeout per tool
                result = await asyncio.wait_for(
                    self.executor.execute(name, args), timeout=15.0
                )
            except asyncio.TimeoutError:
                result = json.dumps(
                    {"error": f"工具{name}执行超时,请稍后重试或减少数据量"},
                    ensure_ascii=False,
                )

            results.append({"id": tc["id"], "result": result})
        return results
