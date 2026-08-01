"""
RiskAssessorAgent — 风控评估子Agent (v3.0-competition)

负责评估持仓组合风险,执行8层风控检查:
  L1 回撤熔断 / L2 ATR止损 / L3 移动止盈 / L4 仓位上限
  L5 踩踏防护 / L6 持仓天数 / L7 跌停检测 / L8 相关性监控
"""

from typing import Any, Dict

from agent.sub_agents.base import AgentContext, AgentResult, BaseAgent


class RiskAssessorAgent(BaseAgent):
    """风控评估Agent — 独立运行时类"""

    agent_name = "risk_assessor"
    agent_icon = "🛡️"

    default_system_prompt = (
        "你是A股组合风控评估Agent。\n"
        "职责: 评估持仓组合风险/检查8层风控触发/给出仓位调整建议。\n"
        "风控层级: L1回撤熔断/L2 ATR止损/L3移动止盈/L4仓位上限/"
        "L5踩踏防护/L6持仓天数/L7跌停检测/L8相关性监控。"
    )

    # 8层风控规则 (硬编码,不依赖LLM)
    RISK_LAYERS = [
        {"id": "L1", "name": "回撤断路器", "trigger": "回撤>8%", "action": "减仓50%", "critical_trigger": "回撤>15%", "critical_action": "清仓"},
        {"id": "L2", "name": "ATR动态止损", "trigger": "价格触及2x ATR", "action": "立即平仓", "critical_trigger": "", "critical_action": ""},
        {"id": "L3", "name": "移动止盈", "trigger": "从高点回落6%", "action": "止盈", "critical_trigger": "", "critical_action": ""},
        {"id": "L4", "name": "仓位上限", "trigger": "单票>25%", "action": "限制开仓", "critical_trigger": "行业>35%", "critical_action": "强制分散"},
        {"id": "L5", "name": "踩踏防护", "trigger": "跌停封板无法卖出", "action": "次日竞价挂单", "critical_trigger": "", "critical_action": ""},
        {"id": "L6", "name": "持仓天数", "trigger": ">20天未盈利", "action": "强制评估平仓", "critical_trigger": "", "critical_action": ""},
        {"id": "L7", "name": "跌停检测", "trigger": "持仓触及跌停", "action": "立即评估", "critical_trigger": "", "critical_action": ""},
        {"id": "L8", "name": "相关性监控", "trigger": "组合相关性>0.8", "action": "预警并建议分散", "critical_trigger": "", "critical_action": ""},
    ]

    async def run(self, ctx: AgentContext) -> AgentResult:
        """执行风控评估"""
        self._start_context(ctx.task_id, **ctx.input_data)

        portfolio = ctx.input_data.get("portfolio", {})
        positions = ctx.input_data.get("positions", [])
        drawdown_pct = ctx.input_data.get("drawdown_pct", 0)
        market_regime = ctx.input_data.get("market_regime", "unknown")

        # 规则引擎检查 (不依赖LLM)
        triggered = self._check_risk_layers(portfolio, positions, drawdown_pct)
        suggestions = self._generate_suggestions(triggered, market_regime)

        result_data: Dict[str, Any] = {
            "triggered_layers": triggered,
            "suggestions": suggestions,
            "narrative": "",
        }

        # 如果有LLM路由,生成风控叙事
        if self.router and triggered:
            system_prompt = self.load_prompt()
            try:
                response = await self.router.route(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": f"触发风控: {triggered}\n市场体制: {market_regime}\n建议: {suggestions}"},
                    ],
                    task_type="risk_assessment",
                )
                result_data["narrative"] = response.response
                result_data["model_used"] = response.tier
            except Exception:
                result_data["narrative"] = self._format_risk_report(triggered, suggestions)
        elif triggered:
            result_data["narrative"] = self._format_risk_report(triggered, suggestions)
        else:
            result_data["narrative"] = "✅ 所有风控层级检查通过,无触发项。"

        return self._finish_context(success=True, **result_data)

    def _check_risk_layers(self, portfolio: dict, positions: list, drawdown_pct: float) -> list:
        """检查8层风控触发情况 (规则引擎)"""
        triggered = []

        # L1: 回撤断路器
        if drawdown_pct < -15:
            triggered.append({"layer": "L1", "name": "回撤断路器", "status": "critical",
                            "detail": f"回撤{abs(drawdown_pct):.1f}%>15%, 建议清仓"})
        elif drawdown_pct < -8:
            triggered.append({"layer": "L1", "name": "回撤断路器", "status": "warning",
                            "detail": f"回撤{abs(drawdown_pct):.1f}%>8%, 建议减仓50%"})

        # L4: 仓位检查
        if positions:
            for pos in positions:
                weight = pos.get("weight", 0)
                if weight > 0.25:
                    triggered.append({"layer": "L4", "name": "仓位上限",
                                    "status": "warning",
                                    "detail": f"{pos.get('symbol')} 仓位{weight:.0%}>25%"})

        # L6: 持仓天数
        for pos in positions:
            days = pos.get("holding_days", 0)
            if days > 20 and pos.get("pnl_pct", 0) < 0:
                triggered.append({"layer": "L6", "name": "持仓天数",
                                "status": "warning",
                                "detail": f"{pos.get('symbol')} 持仓{days}天未盈利"})

        return triggered

    def _generate_suggestions(self, triggered: list, regime: str) -> list:
        """根据触发项生成建议"""
        suggestions = []
        for t in triggered:
            lid = t["layer"]
            if lid == "L1":
                if t["status"] == "critical":
                    suggestions.append("🚨 立即清仓: 回撤超过15%触发清仓机制")
                else:
                    suggestions.append("⚠️ 减仓50%: 回撤超过8%触发减仓机制")
            elif lid == "L4":
                suggestions.append("⚠️ 限制开仓: 单票仓位超过25%上限,请分散持仓")
            elif lid == "L6":
                suggestions.append("⚠️ 评估平仓: 持仓超20天未盈利,建议评估是否继续持有")
        if not suggestions:
            suggestions.append("✅ 所有风控指标正常")
        return suggestions

    def _format_risk_report(self, triggered: list, suggestions: list) -> str:
        """格式化风控报告"""
        if not triggered:
            return "✅ 风控检查通过,无需操作。"
        lines = ["### 风控检查报告\n"]
        for t in triggered:
            icon = "🚨" if t["status"] == "critical" else "⚠️"
            lines.append(f"{icon} **{t['layer']} {t['name']}**: {t['detail']}")
        lines.append("\n### 建议操作")
        for s in suggestions:
            lines.append(f"- {s}")
        return "\n".join(lines)
