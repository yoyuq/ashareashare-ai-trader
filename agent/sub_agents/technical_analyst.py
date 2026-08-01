"""
TechnicalAnalystAgent — 技术面分析子Agent (v3.0-competition)

负责对单个股票进行6维度技术面深度分析:
  趋势/动量/波动率/成交量/K线形态/支撑阻力

使用:
    agent = TechnicalAnalystAgent(knowledge_manager, model_router)
    result = await agent.run(ctx)
"""

import json
from typing import Any, Dict

from agent.sub_agents.base import AgentContext, AgentResult, BaseAgent


class TechnicalAnalystAgent(BaseAgent):
    """技术面分析Agent — 独立运行时类"""

    agent_name = "technical_analyst"
    agent_icon = "📊"

    default_system_prompt = (
        "你是A股技术分析Agent。你的职责是对指定标的进行多维度技术面深度分析。\n"
        "分析维度: 趋势/动量/波动率/成交量/K线形态/支撑阻力。\n"
        "所有RSI/MACD/ATR等数值必须引用代码计算结果,不得自行编造。"
    )

    async def run(self, ctx: AgentContext) -> AgentResult:
        """执行技术分析"""
        self._start_context(ctx.task_id, **ctx.input_data)

        symbol = ctx.input_data.get("symbol", "unknown")
        indicators = ctx.input_data.get("indicators", {})
        df = ctx.input_data.get("dataframe")

        result_data: Dict[str, Any] = {
            "symbol": symbol,
            "indicators_computed": bool(indicators),
            "narrative": "",
        }

        # 如果没有LLM路由,直接返回指标摘要
        if self.router is None:
            result_data["narrative"] = self._summarize_indicators(indicators)
            return self._finish_context(success=True, **result_data)

        # 构建指标摘要
        summary = self._build_indicator_summary(indicators)

        # 加载提示词
        system_prompt = self.load_prompt()

        try:
            response = await self.router.route(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"标的: {symbol}\n指标数据: {summary}"},
                ],
                task_type="technical_analysis",
            )
            result_data["narrative"] = response.response
            result_data["model_used"] = response.tier
        except Exception as e:
            result_data["narrative"] = self._summarize_indicators(indicators)
            result_data["error"] = str(e)

        return self._finish_context(success=True, **result_data)

    def _build_indicator_summary(self, indicators: Any) -> str:
        """构建指标JSON摘要"""
        if indicators is None:
            return "{}"
        try:
            if hasattr(indicators, "to_dataframe"):
                indicators = indicators.to_dataframe()
            if hasattr(indicators, "iloc"):
                last_row = indicators.iloc[-1] if not indicators.empty else {}
            elif isinstance(indicators, dict):
                last_row = indicators
            else:
                return str(indicators)[:2000]

            key_fields = [
                "close", "ma_5", "ma_20", "ma_60", "rsi_14",
                "macd_dif", "macd_dea", "bb_pct_20", "atr_14",
                "trend_score", "composite_score",
            ]
            summary = {
                k: round(float(v), 2) if isinstance(v, (int, float)) else str(v)
                for k, v in (last_row.items() if isinstance(last_row, dict) else last_row.to_dict().items())
                if k in key_fields
            }
            return json.dumps(summary, ensure_ascii=False)
        except Exception:
            return str(indicators)[:1000]

    def _summarize_indicators(self, indicators: Any) -> str:
        """无LLM时的指标摘要"""
        try:
            return self._build_indicator_summary(indicators)
        except Exception:
            return "指标数据不可用"
