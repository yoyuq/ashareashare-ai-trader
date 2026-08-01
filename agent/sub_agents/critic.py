"""
CriticAgent — 对抗性验证器 (v3.0-competition, 2026 Best Practice)

基于2026年AI量化交易最佳实践,独立的Critic Agent负责:
  1. 检测过拟合信号 (look-ahead bias, regime cherry-picking, data snooping)
  2. 评估策略稳健性 (参数敏感度, 交易成本冲击, 体制变化脆弱性)
  3. 输出对抗性检查清单 — 只找漏洞,不修复

核心理念: "Never let the Implementer see prior implementation results"
          — Critic独立验证,防止锚定偏差

与Bull/Bear/Judge的关系:
  - Bull/Bear: 多空视角辩论 (同一量化数据)
  - Judge: 综合裁决 (基于辩论质量)
  - Critic: 元验证 (检查整个分析流程是否有方法论缺陷)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.sub_agents.base import AgentContext, AgentResult, BaseAgent


@dataclass
class FlawReport:
    """漏洞报告"""
    flaw_type: str                      # look_ahead | cherry_picking | overfitting | cost_optimism | fragility
    severity: str                       # critical | high | medium | low
    description: str
    evidence: str                       # 引用具体数据
    impact: str                         # 对策略的影响
    suggested_investigation: str = ""   # 建议进一步调查(但不修复)


@dataclass
class CriticResult:
    """CriticAgent输出"""
    symbol: str
    flaws: List[FlawReport] = field(default_factory=list)
    composite_penalty: float = 0.0      # 复合惩罚分 (越高越差)
    robustness_score: float = 100.0     # 稳健性评分 (100=完美)
    checklist: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "flaws": [
                {
                    "type": f.flaw_type,
                    "severity": f.severity,
                    "description": f.description,
                    "evidence": f.evidence,
                    "impact": f.impact,
                }
                for f in self.flaws
            ],
            "composite_penalty": self.composite_penalty,
            "robustness_score": self.robustness_score,
            "checklist": self.checklist,
        }


class CriticAgent(BaseAgent):
    """
    对抗性验证器 — 独立的策略审计Agent

    检查维度 (不参与决策,只找漏洞):
      1. Look-Ahead Bias: 是否使用了未来数据
      2. Regime Cherry-Picking: 回测期间是否过度集中于有利体制
      3. Overfitting: 参数是否过度优化 (PBO/DSR检测)
      4. Cost Optimism: 交易成本估算是否过于乐观
      5. Fragility: 策略在不同市场体制下的表现差异
    """

    agent_name = "critic"
    agent_icon = "🔍"

    default_system_prompt = (
        "你是一个独立的策略审计师(Critic Agent)。你的唯一职责是找漏洞。\n"
        "你无权修改策略、无权提建议——你只报告问题。\n\n"
        "审查维度:\n"
        "1. Look-Ahead Bias: 策略是否使用了未来信息?\n"
        "2. Regime Cherry-Picking: 回测期间选择是否偏向有利体制?\n"
        "3. Overfitting: 参数是否过度优化?\n"
        "4. Cost Optimism: 交易成本是否被低估?\n"
        "5. Fragility: 策略在压力情景下还能工作吗?\n\n"
        "输出格式: 只输出JSON,列出所有发现的漏洞及其严重程度。"
    )

    # 复合惩罚权重 (基于2026 Best Practice)
    PENALTY_WEIGHTS = {
        "look_ahead": 30.0,      # 最严重的错误
        "cherry_picking": 15.0,
        "overfitting": 12.0,
        "cost_optimism": 10.0,
        "fragility": 8.0,
    }

    SEVERITY_MULTIPLIER = {
        "critical": 1.0,
        "high": 0.7,
        "medium": 0.4,
        "low": 0.2,
    }

    async def run(self, ctx: AgentContext) -> AgentResult:
        """
        执行策略审计

        输入 (ctx.input_data):
          - symbol: 股票代码
          - backtest_results: 回测结果
          - strategy_config: 策略配置
          - regime_history: 市场体制历史
          - indicators: 技术指标
        """
        self._start_context(ctx.task_id, **ctx.input_data)

        symbol = ctx.input_data.get("symbol", "unknown")
        backtest = ctx.input_data.get("backtest_results", {})
        strategy_config = ctx.input_data.get("strategy_config", {})
        regime_history = ctx.input_data.get("regime_history", {})
        indicators = ctx.input_data.get("indicators", {})

        # ── 规则引擎检查 (不依赖LLM) ──
        result = CriticResult(symbol=symbol)
        flaws = []

        # 1. Look-Ahead Bias 检查
        la_flaws = self._check_look_ahead(strategy_config, backtest)
        flaws.extend(la_flaws)

        # 2. Regime Cherry-Picking 检查
        cp_flaws = self._check_cherry_picking(regime_history, backtest)
        flaws.extend(cp_flaws)

        # 3. Overfitting 检查
        of_flaws = self._check_overfitting(backtest, strategy_config)
        flaws.extend(of_flaws)

        # 4. Cost Optimism 检查
        co_flaws = self._check_cost_optimism(backtest, strategy_config)
        flaws.extend(co_flaws)

        # 5. Fragility 检查
        fr_flaws = self._check_fragility(backtest, regime_history)
        flaws.extend(fr_flaws)

        result.flaws = flaws

        # 计算复合惩罚分
        result.composite_penalty = sum(
            self.PENALTY_WEIGHTS.get(f.flaw_type, 5.0) *
            self.SEVERITY_MULTIPLIER.get(f.severity, 0.5)
            for f in flaws
        )
        result.robustness_score = max(0.0, 100.0 - result.composite_penalty)

        # 生成检查清单
        result.checklist = [
            {"check": "Look-Ahead Bias", "status": "✅" if not la_flaws else f"❌ {len(la_flaws)}项"},
            {"check": "Regime Cherry-Picking", "status": "✅" if not cp_flaws else f"⚠️ {len(cp_flaws)}项"},
            {"check": "Overfitting (PBO/DSR)", "status": "✅" if not of_flaws else f"⚠️ {len(of_flaws)}项"},
            {"check": "Cost Optimism", "status": "✅" if not co_flaws else f"⚠️ {len(co_flaws)}项"},
            {"check": "Fragility (Regime Stress)", "status": "✅" if not fr_flaws else f"⚠️ {len(fr_flaws)}项"},
        ]

        # 如果有LLM路由,生成审计叙事
        narrative = ""
        if self.router and flaws:
            system_prompt = self.load_prompt()
            try:
                response = await self.router.route(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": f"标的: {symbol}\n回测数据: {json.dumps(backtest, ensure_ascii=False)[:2000]}\n"
                                   f"策略配置: {json.dumps(strategy_config, ensure_ascii=False)[:1000]}\n"
                                   f"规则引擎已发现{len(flaws)}个漏洞:\n{json.dumps(result.to_dict()['flaws'], ensure_ascii=False, indent=2)}"},
                    ],
                    task_type="risk_assessment",
                )
                narrative = response.response
            except Exception:
                narrative = self._format_flaws(flaws)
        elif flaws:
            narrative = self._format_flaws(flaws)
        else:
            narrative = "✅ 策略审计通过: 未发现方法论缺陷。所有5个维度的检查均为通过。"

        return self._finish_context(
            success=True,
            critic_result=result.to_dict(),
            flaw_count=len(flaws),
            robustness_score=result.robustness_score,
            narrative=narrative,
        )

    # ═══════════════════════════════════════════════════════════════
    # 规则引擎检查方法
    # ═══════════════════════════════════════════════════════════════

    def _check_look_ahead(self, config: dict, backtest: dict) -> List[FlawReport]:
        """检查未来数据泄露"""
        flaws = []
        # 检查是否声明了PIT处理
        if config.get("pit_enabled") is False or config.get("pit_enabled") is None:
            flaws.append(FlawReport(
                flaw_type="look_ahead",
                severity="critical",
                description="策略未启用Point-in-Time数据处理",
                evidence="strategy_config.pit_enabled != True",
                impact="可能使用未来财务数据,导致回测结果虚高",
                suggested_investigation="验证所有因子计算是否使用了决策时点之前的数据",
            ))
        # 检查回测是否包含IPO<60天的股票
        if backtest.get("filter_ipo_days", 0) < 60:
            flaws.append(FlawReport(
                flaw_type="look_ahead",
                severity="high",
                description=f"IPO过滤器仅{backtest.get('filter_ipo_days', 0)}天,建议≥60天",
                evidence="回测参数 filter_ipo_days < 60",
                impact="新股数据不足,技术指标计算不准确",
            ))
        return flaws

    def _check_cherry_picking(self, regime_history: dict, backtest: dict) -> List[FlawReport]:
        """检查体制选择偏差"""
        flaws = []
        if not regime_history:
            return flaws

        # 检查回测期间是否过度集中在有利体制
        regimes = regime_history.get("distribution", {})
        if regimes:
            dominant_regime = max(regimes, key=regimes.get)
            dominant_pct = regimes[dominant_regime] / sum(regimes.values()) if sum(regimes.values()) > 0 else 0
            # 如果某个体制占比>60%,可能是不合理的时间段选择
            if dominant_pct > 0.6:
                flaws.append(FlawReport(
                    flaw_type="cherry_picking",
                    severity="medium",
                    description=f"回测期间{dominant_regime}体制占比{dominant_pct:.0%}>60%",
                    evidence=f"体制分布: {regimes}",
                    impact="策略可能仅在特定市场环境下表现良好,泛化能力存疑",
                ))

        # 检查是否使用了过去5年的完整周期
        years_covered = regime_history.get("years_covered", 0)
        if years_covered < 3:
            flaws.append(FlawReport(
                flaw_type="cherry_picking",
                severity="high",
                description=f"回测仅覆盖{years_covered}年,建议≥5年以包含牛熊完整周期",
                evidence=f"years_covered={years_covered}",
                impact="未经过完整牛熊周期验证",
            ))

        return flaws

    def _check_overfitting(self, backtest: dict, config: dict) -> List[FlawReport]:
        """检查过拟合"""
        flaws = []
        metrics = backtest.get("metrics", backtest)

        # PBO检查
        pbo = backtest.get("overfitting_checks", {}).get("PBO", metrics.get("pbo"))
        if pbo is not None and pbo > 0.5:
            flaws.append(FlawReport(
                flaw_type="overfitting",
                severity="high",
                description=f"过拟合概率(PBO)={pbo:.2f}>0.5,策略可能过拟合",
                evidence=f"PBO={pbo:.2f}",
                impact="回测绩效可能无法在实盘中复现",
            ))

        # DSR检查
        dsr = backtest.get("overfitting_checks", {}).get("DSR", metrics.get("deflated_sharpe"))
        if dsr is not None and dsr < 0.5:
            flaws.append(FlawReport(
                flaw_type="overfitting",
                severity="medium",
                description=f"Deflated Sharpe Ratio={dsr:.2f}<0.5,策略统计显著性不足",
                evidence=f"DSR={dsr:.2f}",
                impact="策略收益可能来自随机性而非真实α",
            ))

        # 参数数量检查
        param_count = self._count_params(config)
        trade_count = backtest.get("total_trades", metrics.get("total_trades", 100))
        if trade_count > 0 and param_count / trade_count > 0.1:
            flaws.append(FlawReport(
                flaw_type="overfitting",
                severity="low",
                description=f"参数数量({param_count})相对交易次数({trade_count})偏多",
                evidence=f"参数/交易比={param_count/trade_count:.3f}",
                impact="可能存在过度参数化风险",
            ))

        return flaws

    def _check_cost_optimism(self, backtest: dict, config: dict) -> List[FlawReport]:
        """检查交易成本估算"""
        flaws = []
        metrics = backtest.get("metrics", backtest)
        cost_pct = config.get("transaction_cost_pct", 0.0031)  # 默认0.31%

        # A股实际成本: 佣金0.03% + 印花税0.05%(卖) + 过户费0.001% + 滑点
        MIN_REALISTIC_COST = 0.002  # 0.2% 是合理的最低估计

        if cost_pct < MIN_REALISTIC_COST:
            flaws.append(FlawReport(
                flaw_type="cost_optimism",
                severity="high",
                description=f"交易成本估算{cost_pct:.2%}过低,实际A股双向成本约0.2%-0.5%",
                evidence=f"config.transaction_cost_pct={cost_pct}",
                impact="策略实盘收益将显著低于回测",
            ))

        # 滑点检查
        if not config.get("slippage_enabled"):
            flaws.append(FlawReport(
                flaw_type="cost_optimism",
                severity="medium",
                description="未启用滑点模拟,大单成交价格可能劣于回测",
                evidence="config.slippage_enabled != True",
                impact="大资金策略的实际成交价可能显著偏离回测价格",
            ))

        return flaws

    def _check_fragility(self, backtest: dict, regime_history: dict) -> List[FlawReport]:
        """检查体制脆弱性"""
        flaws = []
        metrics = backtest.get("metrics", backtest)

        # 检查最大回撤是否超过合理范围
        max_dd = abs(metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0)))
        if max_dd > 35:
            flaws.append(FlawReport(
                flaw_type="fragility",
                severity="critical",
                description=f"最大回撤{max_dd:.1f}%过高,策略在压力情景下可能崩溃",
                evidence=f"最大回撤={max_dd:.1f}%",
                impact="投资者可能在最大回撤时被迫平仓,无法等到策略恢复",
            ))

        # 检查胜率是否过低
        win_rate = metrics.get("win_rate", 0.5)
        if win_rate < 0.35:
            flaws.append(FlawReport(
                flaw_type="fragility",
                severity="medium",
                description=f"胜率仅{win_rate:.1%},连续亏损概率高,可能导致执行偏差",
                evidence=f"胜率={win_rate:.1%}",
                impact="连续亏损可能导致投资者失去信心,提前终止策略",
            ))

        return flaws

    def _count_params(self, config: dict) -> int:
        """递归统计策略参数数量"""
        count = 0
        for k, v in config.items():
            if k in ("name", "description", "category", "version", "note", "id"):
                continue
            if isinstance(v, dict):
                count += self._count_params(v)
            elif isinstance(v, (int, float)):
                count += 1
            elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                count += len(v)
        return count

    def _format_flaws(self, flaws: List[FlawReport]) -> str:
        """格式化漏洞报告"""
        if not flaws:
            return "✅ 策略审计通过: 未发现方法论缺陷。"
        lines = ["### 🔍 CriticAgent 策略审计报告\n"]
        lines.append(f"发现 {len(flaws)} 个潜在问题:\n")
        for i, f in enumerate(flaws, 1):
            icon = {"critical": "🚨", "high": "⚠️", "medium": "📝", "low": "💡"}.get(f.severity, "❓")
            lines.append(f"{icon} **{i}. [{f.flaw_type}] {f.description}**")
            lines.append(f"   证据: {f.evidence}")
            lines.append(f"   影响: {f.impact}")
            lines.append("")
        return "\n".join(lines)
