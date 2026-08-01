"""
EvaluatorAgent — 综合研判质量评估器 (v3.1-deerflow)

DeerFlow「Producer → Evaluator → Revise」反射循环中的 Evaluator 角色:
  - 对 synthesis 产出的最终报告逐项打分 (规则引擎, 无LLM成本)
  - 不通过 → 返回结构化 critique 供 synthesis_revision 回炉
  - 通过 → 循环终止

评估维度:
  1. trade_params 缺失: BUY/SELL 推荐缺少代码计算的入场/止损/止盈/仓位
  2. 参数自相矛盾: 止损>=入场 或 止盈<=入场 或 仓位超出合理区间
  3. 确信度与动作矛盾: BUY 却低确信 / SELL 却高确信
  4. 关键字段缺失: BUY/SELL 无 key_reasons 或 risks
  5. 稳健性过低: Critic 审计的 robustness_score < 阈值
  6. 幻觉数字: 报告 JSON 中 entry/stop/take 与代码计算值明显不符

与 CriticAgent 的关系:
  - Critic: 审计回测方法论 (look-ahead/overfitting/...), 在 synthesis 之前
  - Evaluator: 审计最终报告的产出质量, 在 synthesis 之后驱动回炉
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.sub_agents.base import AgentContext, AgentResult, BaseAgent


@dataclass
class EvaluationIssue:
    """质量缺陷"""
    code: str               # missing_trade_params | param_inconsistency | conviction_contradiction
                            # | missing_reasoning | low_robustness | hallucinated_numbers
    severity: str           # critical | high | medium
    symbol: str
    description: str
    evidence: str = ""


@dataclass
class EvaluationResult:
    """EvaluatorAgent 输出"""
    passed: bool
    score: float            # 0-100 (100=完美)
    issues: List[EvaluationIssue] = field(default_factory=list)
    critique: str = ""      # 供 synthesis_revision 消费的修订反馈

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": [
                {"code": i.code, "severity": i.severity,
                 "symbol": i.symbol, "description": i.description,
                 "evidence": i.evidence}
                for i in self.issues
            ],
            "critique": self.critique,
        }


class EvaluatorAgent(BaseAgent):
    """
    综合研判质量评估器 — 规则引擎, 驱动 synthesis 反思回炉
    """

    agent_name = "evaluator"
    agent_icon = "🧪"

    default_system_prompt = (
        "你是一个报告质量评估器(Evaluator Agent)。你的职责是判定一份股票分析报告"
        "是否达到可执行标准, 并给出可操作的修订批评。"
    )

    # 评分权重 (缺陷越重扣分越多)
    SEVERITY_PENALTY = {
        "critical": 30.0,
        "high": 15.0,
        "medium": 8.0,
    }
    PASS_THRESHOLD = 70.0      # >=70 视为通过
    ROBUSTNESS_MIN = 50.0      # Critic 稳健性下限
    POS_PCT_RANGE = (0.01, 0.30)  # 仓位合理区间

    async def run(self, ctx: AgentContext) -> AgentResult:
        """
        执行报告质量评估

        输入 (ctx.input_data):
          - final_report: synthesis 产出的最终报告文本
          - trading_params: 代码计算的交易参数 {symbol: {entry_price, stop_loss, take_profit, position_pct}}
          - stock_recommendations: 逐股推荐 {symbol: {action, conviction, key_reasons, risks, ...}}
          - critic_results: Critic 审计结果 {symbol: {robustness_score, ...}}
        """
        self._start_context(ctx.task_id, **ctx.input_data)

        final_report = ctx.input_data.get("final_report", "")
        trading_params = ctx.input_data.get("trading_params", {}) or {}
        stock_recs = ctx.input_data.get("stock_recommendations", {}) or {}
        critic_results = ctx.input_data.get("critic_results", {}) or {}

        issues: List[EvaluationIssue] = []
        parsed_report = self._try_parse_json(final_report)

        for sym, rec in stock_recs.items():
            if not isinstance(rec, dict):
                continue
            action = rec.get("action", "HOLD")
            if action not in ("BUY", "SELL"):
                continue  # HOLD 不要求交易参数

            tp = (trading_params.get(sym) or {}) if isinstance(trading_params, dict) else {}

            # 1. trade_params 缺失
            entry = self._num(tp.get("entry_price"))
            stop = self._num(tp.get("stop_loss"))
            take = self._num(tp.get("take_profit"))
            pos = self._num(tp.get("position_pct"))
            if not (entry and entry > 0 and stop and stop > 0
                    and take and take > 0 and pos is not None and pos > 0):
                issues.append(EvaluationIssue(
                    code="missing_trade_params", severity="critical", symbol=sym,
                    description=f"{action} 推荐缺少可执行的交易参数 (entry/stop/take/position)",
                    evidence=f"trading_params={tp}",
                ))
            else:
                # 2. 参数自相矛盾
                if stop >= entry:
                    issues.append(EvaluationIssue(
                        code="param_inconsistency", severity="critical", symbol=sym,
                        description=f"止损({stop}) >= 入场价({entry}), 止损失效",
                        evidence=f"stop_loss={stop} entry_price={entry}",
                    ))
                if take <= entry:
                    issues.append(EvaluationIssue(
                        code="param_inconsistency", severity="critical", symbol=sym,
                        description=f"止盈({take}) <= 入场价({entry}), 止盈失效",
                        evidence=f"take_profit={take} entry_price={entry}",
                    ))
                if not (self.POS_PCT_RANGE[0] <= pos <= self.POS_PCT_RANGE[1]):
                    issues.append(EvaluationIssue(
                        code="param_inconsistency", severity="high", symbol=sym,
                        description=f"仓位比例 {pos:.1%} 超出合理区间 "
                                    f"[{self.POS_PCT_RANGE[0]:.0%}, {self.POS_PCT_RANGE[1]:.0%}]",
                        evidence=f"position_pct={pos}",
                    ))

            # 3. 确信度与动作矛盾 — 低确信度却给出交易动作, 方向与信心矛盾
            conviction = self._num(rec.get("conviction"))
            if conviction is not None and 0 < conviction < 0.3:
                issues.append(EvaluationIssue(
                    code="conviction_contradiction", severity="critical", symbol=sym,
                    description=f"动作={action} 但确信度仅 {conviction:.2f}, 交易信号自相矛盾",
                    evidence=f"action={action} conviction={conviction}",
                ))

            # 4. 关键字段缺失
            if not rec.get("key_reasons"):
                issues.append(EvaluationIssue(
                    code="missing_reasoning", severity="medium", symbol=sym,
                    description=f"{action} 推荐缺少 key_reasons 论据",
                    evidence="key_reasons 为空",
                ))
            if not rec.get("risks"):
                issues.append(EvaluationIssue(
                    code="missing_reasoning", severity="medium", symbol=sym,
                    description=f"{action} 推荐缺少 risks 风险提示",
                    evidence="risks 为空",
                ))

            # 6. 幻觉数字: 报告JSON中的参数 vs 代码计算值
            if parsed_report and entry and entry > 0:
                report_rec = self._find_report_rec(parsed_report, sym)
                if report_rec:
                    r_entry = self._num(report_rec.get("entry_price"))
                    if r_entry > 0 and abs(r_entry - entry) / entry > 0.05:
                        issues.append(EvaluationIssue(
                            code="hallucinated_numbers", severity="high", symbol=sym,
                            description=f"报告声称入场价 {r_entry}, 代码计算为 {entry}, 偏差>5%",
                            evidence=f"report entry_price={r_entry} computed={entry}",
                        ))

        # 5. 稳健性过低 (Critic 已审计过回测方法论)
        for sym, cr in critic_results.items():
            rob = self._num(cr.get("robustness_score")) if isinstance(cr, dict) else None
            if rob is not None and rob < self.ROBUSTNESS_MIN:
                issues.append(EvaluationIssue(
                    code="low_robustness", severity="medium", symbol=sym,
                    description=f"Critic 稳健性 {rob:.0f} < {self.ROBUSTNESS_MIN:.0f}, 分析可信度不足",
                    evidence=f"robustness_score={rob}",
                ))

        # ── 评分 ──
        score = 100.0 - sum(self.SEVERITY_PENALTY.get(i.severity, 5.0) for i in issues)
        score = max(0.0, round(score, 1))
        passed = score >= self.PASS_THRESHOLD and not any(
            i.severity == "critical" for i in issues  # 任一 critical 缺陷即不通过
        )

        critique = self._build_critique(issues, passed)

        return self._finish_context(
            success=True,
            evaluation_result=EvaluationResult(
                passed=passed, score=score, issues=issues, critique=critique
            ).to_dict(),
            passed=passed,
            score=score,
            issue_count=len(issues),
            critique=critique,
        )

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    def _build_critique(self, issues: List[EvaluationIssue], passed: bool) -> str:
        """生成供 synthesis_revision 消费的修订反馈文本"""
        if passed:
            return "报告通过质量评估, 无需修订。"
        lines = [
            "本次综合研判报告未通过质量评估, 请针对以下问题修订后再输出:",
        ]
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. [{issue.severity}] {issue.symbol}: {issue.description}")
            if issue.evidence:
                lines.append(f"   依据: {issue.evidence}")
        lines.append(
            "\n要求: 只输出修订后的完整报告, 结构保持不变; "
            "缺失的 trade_params 必须从代码计算的交易参数中填入; 修正自相矛盾的数值。"
        )
        return "\n".join(lines)

    @staticmethod
    def _num(val) -> Optional[float]:
        try:
            f = float(val)
            return f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict]:
        """尝试把最终报告解析为 JSON (可能包裹在代码块中)"""
        if not text:
            return None
        t = text.strip()
        if "```json" in t:
            t = t.split("```json")[1].split("```")[0]
        elif "```" in t:
            t = t.split("```")[1].split("```")[0]
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, IndexError):
            return None

    @staticmethod
    def _find_report_rec(parsed: Dict, sym: str) -> Optional[Dict]:
        """在报告JSON的 recommendations 列表中找某只股票的推荐"""
        recs = parsed.get("recommendations")
        if not isinstance(recs, list):
            return None
        code = sym.split(".")[-1]
        for r in recs:
            if not isinstance(r, dict):
                continue
            rsym = r.get("symbol") or r.get("code") or ""
            if rsym == sym or rsym == code or sym.endswith(rsym):
                return r
        return None
