"""
MarketScannerAgent — 市场扫描子Agent (v3.0-competition)

负责快速扫描全市场,识别值得关注的标的:
  技术面异动/资金面异动/市场热度/板块联动
"""

from typing import Any, Dict

from agent.sub_agents.base import AgentContext, AgentResult, BaseAgent


class MarketScannerAgent(BaseAgent):
    """市场扫描Agent — 独立运行时类"""

    agent_name = "market_scanner"
    agent_icon = "🔍"

    default_system_prompt = (
        "你是A股市场扫描Agent。你的职责是快速扫描全市场,发现值得关注的标的。\n"
        "扫描维度: 技术面异动/资金面异动/市场热度/板块联动。\n"
        "自动过滤ST/新股/僵尸股。"
    )

    async def run(self, ctx: AgentContext) -> AgentResult:
        """执行市场扫描"""
        self._start_context(ctx.task_id, **ctx.input_data)

        regime = ctx.input_data.get("regime", "unknown")
        symbols = ctx.input_data.get("symbols", [])
        scan_results = ctx.input_data.get("scan_results", [])

        result_data: Dict[str, Any] = {
            "regime": regime,
            "symbols_count": len(symbols),
            "scan_hits": len(scan_results),
            "narrative": "",
        }

        if self.router is None:
            result_data["narrative"] = self._summarize_scan(scan_results, regime)
            return self._finish_context(success=True, **result_data)

        system_prompt = self.load_prompt()

        try:
            response = await self.router.route(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"市场体制: {regime}\n候选标的: {symbols[:50]}\n扫描结果: {str(scan_results)[:2000]}"},
                ],
                task_type="simple_qa",
            )
            result_data["narrative"] = response.response
            result_data["model_used"] = response.tier
        except Exception as e:
            result_data["narrative"] = self._summarize_scan(scan_results, regime)
            result_data["error"] = str(e)

        return self._finish_context(success=True, **result_data)

    def _summarize_scan(self, scan_results: list, regime: str) -> str:
        """无LLM时的扫描摘要"""
        if not scan_results:
            return f"当前市场体制: {regime}。暂无符合筛选条件的标的。"
        return f"当前市场体制: {regime}。扫描发现 {len(scan_results)} 个值得关注的标的。"
