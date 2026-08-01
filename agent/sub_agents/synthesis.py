"""
SynthesisAgent — 综合研判子Agent (v3.0-competition)

负责综合所有子Agent的分析结果,进行多空辩论后输出最终研判报告。

输出结构:
  一、市场环境总览
  二、标的分析 (含辩论摘要)
  三、交易建议 (表格)
  四、风险提示
"""

from typing import Any, Dict

from agent.sub_agents.base import AgentContext, AgentResult, BaseAgent


class SynthesisAgent(BaseAgent):
    """综合研判Agent (总指挥) — 独立运行时类"""

    agent_name = "synthesis"
    agent_icon = "📝"

    default_system_prompt = (
        "你是A股综合研判Agent (总指挥)。你的职责是综合所有子Agent的分析结果,"
        "进行多空辩论后输出最终研判报告。"
        "报告结构: 市场环境总览→标的分析→交易建议→风险提示。"
        "所有数字必须由代码计算,不得自行编造。"
    )

    async def run(self, ctx: AgentContext) -> AgentResult:
        """执行综合研判"""
        self._start_context(ctx.task_id, **ctx.input_data)

        # 收集所有输入
        regime = ctx.input_data.get("regime", "unknown")
        stock_analyses = ctx.input_data.get("stock_analyses", {})
        strategy_matches = ctx.input_data.get("strategy_matches", {})
        backtest_results = ctx.input_data.get("backtest_results", {})
        debate_results = ctx.input_data.get("debate_results", {})
        risk_assessment = ctx.input_data.get("risk_assessment", {})

        result_data: Dict[str, Any] = {
            "report": "",
            "recommendations": [],
        }

        if self.router is None:
            # 无LLM — 使用模板生成报告
            result_data["report"] = self._template_report(
                regime, stock_analyses, strategy_matches,
                backtest_results, debate_results, risk_assessment,
            )
            return self._finish_context(success=True, **result_data)

        system_prompt = self.load_prompt()

        # 构建结构化上下文
        context = {
            "market_regime": regime,
            "stock_analyses": {k: str(v)[:500] for k, v in stock_analyses.items()},
            "strategy_matches": {k: str(v)[:300] for k, v in strategy_matches.items()},
            "backtest_summary": {k: str(v)[:300] for k, v in backtest_results.items()},
            "debate_summary": {k: str(v)[:500] for k, v in debate_results.items()},
            "risk_assessment": str(risk_assessment)[:500],
        }

        try:
            response = await self.router.route(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"综合数据:\n{str(context)[:4000]}"},
                ],
                task_type="daily_synthesis",
            )
            result_data["report"] = response.response
            result_data["model_used"] = response.tier
        except Exception as e:
            result_data["report"] = self._template_report(
                regime, stock_analyses, strategy_matches,
                backtest_results, debate_results, risk_assessment,
            )
            result_data["error"] = str(e)

        return self._finish_context(success=True, **result_data)

    def _template_report(
        self, regime, stock_analyses, strategy_matches,
        backtest_results, debate_results, risk_assessment,
    ) -> str:
        """无LLM时的模板报告"""
        lines = [
            "# A股量化分析日报\n",
            f"## 一、市场环境总览\n当前市场体制: **{regime}**\n",
            f"## 二、标的分析\n分析标的数: {len(stock_analyses)}\n",
        ]
        for sym, analysis in stock_analyses.items():
            lines.append(f"- **{sym}**: {str(analysis)[:200]}")
        lines.append(f"\n## 三、策略匹配\n{str(strategy_matches)[:500]}")
        lines.append(f"\n## 四、回测验证\n{str(backtest_results)[:500]}")
        lines.append(f"\n## ⚠️ 风险提示\n历史数据不代表未来表现。本文仅为量化分析参考，不构成投资建议。")
        return "\n".join(lines)
