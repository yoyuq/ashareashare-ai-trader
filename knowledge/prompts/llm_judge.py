"""
LLM-as-Judge — 真实的质量评估流水线 (v3.0-competition)

使用独立的 LLM 调用作为评估器,从4个维度评估AI回复质量:
  1. 格式合规 (structure):     输出是否遵循指定格式
  2. 数值准确性 (accuracy):    引用数据是否可验证
  3. 逻辑一致性 (logic):       分析推理是否有逻辑漏洞
  4. 风险提示完整性 (risk):    是否包含必要的风险提示

与之前的"关键词匹配"方式不同,LLM-Judge会:
  - 阅读完整的回复内容
  - 对照评估标准逐项打分
  - 给出具体的扣分理由和改进建议
  - 输出结构化JSON评估报告

用法:
    judge = LLMJudge(model_router)
    result = await judge.evaluate(question, reply, expected_points)
    print(f"总分: {result['total_score']}/10")
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── 评估Prompt ──

JUDGE_SYSTEM_PROMPT = """你是一个专业的AI回复质量评估器。你的任务是根据给定的评估标准,对AI助手的回复进行客观评分。

评估规则:
1. 只根据回复内容评分,不评判问题本身
2. 每个维度独立打分,不要互相影响
3. 评分必须有具体依据,引用回复中的原文
4. 对于缺点,必须给出具体的改进建议
5. 只输出JSON,不要输出其他内容

评分维度:
- structure (格式合规, 0-3分):
  3分: 有清晰的标题/段落结构,有编号列表,长度合理(>100字)
  2分: 有基本分段,但结构不够清晰
  1分: 回复较短,缺少结构
  0分: 回复为空或完全无结构

- accuracy (数值准确性, 0-3分):
  3分: 引用了多个具体数值,数值有明确来源/上下文
  2分: 有数值但较少,或部分数值缺乏上下文
  1分: 几乎没有引用数值
  0分: 完全无数据支撑

- logic (逻辑一致性, 0-2分):
  2分: 分析有因果推理链,结论有依据支撑
  1分: 有一定逻辑但不够严密
  0分: 逻辑混乱或自相矛盾

- risk_warning (风险提示, 0-2分):
  2分: 包含风险提示+免责声明,且具体而非笼统
  1分: 提到了风险但不具体
  0分: 完全没有风险提示

输出JSON格式 (不要输出其他内容):
{
  "structure": {"score": 0-3, "reason": "评分理由", "evidence": "引用原文"},
  "accuracy": {"score": 0-3, "reason": "评分理由", "evidence": "引用原文"},
  "logic": {"score": 0-2, "reason": "评分理由", "evidence": "引用原文"},
  "risk_warning": {"score": 0-2, "reason": "评分理由", "evidence": "引用原文"},
  "total_score": 0-10,
  "overall_assessment": "一句话总结",
  "improvements": ["改进建议1", "改进建议2"]
}
"""


@dataclass
class JudgeResult:
    """LLM-Judge 评估结果"""
    structure_score: float = 0.0
    accuracy_score: float = 0.0
    logic_score: float = 0.0
    risk_score: float = 0.0
    total_score: float = 0.0
    overall_assessment: str = ""
    improvements: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "structure": self.structure_score,
            "accuracy": self.accuracy_score,
            "logic": self.logic_score,
            "risk_warning": self.risk_score,
            "total_score": self.total_score,
            "overall_assessment": self.overall_assessment,
            "improvements": self.improvements,
            "details": self.details,
        }


class LLMJudge:
    """
    LLM-as-Judge 评估器

    使用独立的 LLM 调用评估AI回复质量。
    需要 ModelRouter 实例来调用LLM。
    """

    def __init__(self, model_router=None):
        """
        Args:
            model_router: ModelRouter 实例,用于调用LLM
        """
        self._router = model_router

    async def evaluate(
        self,
        question: str,
        reply: str,
        expected_points: Optional[List[str]] = None,
        user_context: Optional[str] = None,
    ) -> JudgeResult:
        """
        评估单条回复

        Args:
            question: 用户问题
            reply: AI回复文本
            expected_points: 期望回答包含的要点 (可选)
            user_context: 额外的用户上下文 (可选)

        Returns:
            JudgeResult 评估结果
        """
        # 构建评估请求
        user_prompt = f"""请评估以下AI回复的质量。

【用户问题】
{question[:500]}

【AI回复】
{reply[:2000] if reply else "(空回复)"}
"""
        if expected_points:
            user_prompt += f"""
【期望回答要点】
{chr(10).join(f'- {p}' for p in expected_points)}
"""
        if user_context:
            user_prompt += f"""
【额外上下文】
{user_context[:500]}
"""
        user_prompt += """
请按照评估标准对AI回复进行评分,只输出JSON。"""

        # 尝试使用 LLM 评估
        if self._router:
            try:
                result = await self._router.route(
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    task_type="simple_qa",  # 评估任务相对简单
                )
                return self._parse_judge_response(result.response, reply)
            except Exception as e:
                # LLM不可用时降级为规则评分
                return self._fallback_evaluate(question, reply, expected_points, str(e))
        else:
            return self._fallback_evaluate(question, reply, expected_points, "无LLM路由")

    def evaluate_sync(
        self,
        question: str,
        reply: str,
        expected_points: Optional[List[str]] = None,
    ) -> JudgeResult:
        """
        同步版本的评估 (用于非async环境)

        使用规则引擎而非LLM,保证在任何环境下都能运行。
        """
        return self._fallback_evaluate(question, reply, expected_points, "sync模式")

    def _parse_judge_response(self, response_text: str, original_reply: str) -> JudgeResult:
        """解析LLM-Judge的JSON输出"""
        judge_text = response_text.strip()

        # 提取JSON
        if "```json" in judge_text:
            judge_text = judge_text.split("```json")[1].split("```")[0]
        elif "```" in judge_text:
            judge_text = judge_text.split("```")[1].split("```")[0]

        try:
            data = json.loads(judge_text)
            return JudgeResult(
                structure_score=float(data.get("structure", {}).get("score", 0)),
                accuracy_score=float(data.get("accuracy", {}).get("score", 0)),
                logic_score=float(data.get("logic", {}).get("score", 0)),
                risk_score=float(data.get("risk_warning", {}).get("score", 0)),
                total_score=float(data.get("total_score", 0)),
                overall_assessment=data.get("overall_assessment", ""),
                improvements=data.get("improvements", []),
                details={
                    "structure_reason": data.get("structure", {}).get("reason", ""),
                    "accuracy_reason": data.get("accuracy", {}).get("reason", ""),
                    "logic_reason": data.get("logic", {}).get("reason", ""),
                    "risk_reason": data.get("risk_warning", {}).get("reason", ""),
                },
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # JSON解析失败,降级
            return self._fallback_evaluate("", original_reply, [], f"JSON解析失败: {e}")

    def _fallback_evaluate(
        self,
        question: str,
        reply: str,
        expected_points: Optional[List[str]],
        reason: str,
    ) -> JudgeResult:
        """
        降级评估 — 使用规则引擎 (比纯关键词匹配更严谨)

        当LLM不可用时,使用基于规则的启发式评估。这仍然比简单的
        关键词匹配好,因为它检查了结构特征和内容模式。
        """
        import re

        if not reply:
            return JudgeResult(
                total_score=0,
                overall_assessment="回复为空",
                improvements=["请确保Agent返回有效的回复内容"],
            )

        # 1. 格式合规 (0-3)
        struct_score = 0.0
        struct_reasons = []
        if len(reply) > 100:
            struct_score += 1.0
            struct_reasons.append("回复长度充足(>100字)")
        if "###" in reply or "##" in reply:
            struct_score += 1.0
            struct_reasons.append("有标题/段落结构")
        if re.search(r'\d+[\.\、]', reply):
            struct_score += 0.5
            struct_reasons.append("有编号列表")
        if "建议" in reply or "结论" in reply or "综合" in reply:
            struct_score += 0.5
            struct_reasons.append("有结论/建议")

        # 2. 数值准确性 (0-3)
        numbers = re.findall(r'\d+\.?\d*%?', reply)
        acc_score = 0.0
        acc_reasons = []
        if len(numbers) >= 5:
            acc_score += 2.0
            acc_reasons.append(f"引用了{len(numbers)}个数值")
        elif len(numbers) >= 2:
            acc_score += 1.0
            acc_reasons.append(f"引用了{len(numbers)}个数值")
        source_kws = ["数据", "指标", "计算", "来源", "显示", "表明"]
        if any(kw in reply for kw in source_kws):
            acc_score += 1.0
            acc_reasons.append("数值有上下文/来源")

        # 3. 逻辑一致性 (0-2)
        logic_score = 0.0
        logic_reasons = []
        # 显式因果推理
        causality = ["因为", "所以", "因此", "由于", "基于", "从而", "导致", "由此"]
        causal_count = sum(1 for kw in causality if kw in reply)
        if causal_count >= 2:
            logic_score += 0.8
            logic_reasons.append(f"有{causal_count}处显式因果推理")
        elif causal_count >= 1:
            logic_score += 0.4
        # 隐式逻辑结构: 多维度编号分析 (1. xxx 2. xxx 3. xxx 或 - xxx)
        numbered_points = len(re.findall(r'(?:^|\n)\s*(?:\d+[\.\、\)]|[-•])\s', reply, re.MULTILINE))
        if numbered_points >= 3:
            logic_score += 0.8
            logic_reasons.append(f"有{numbered_points}个编号分析维度")
        elif numbered_points >= 1:
            logic_score += 0.4
        # 结构化分析模式: 现状/分析/建议/风险 四段式
        structure_sections = sum(1 for kw in ["**现状**", "**分析**", "**建议**", "**风险**"] if kw in reply)
        if structure_sections >= 3:
            logic_score += 0.4
            logic_reasons.append(f"有四段式分析结构({structure_sections}/4段)")
        # 多空分析+综合结论
        has_bull_bear = ("看涨" in reply or "看跌" in reply or "多头" in reply or "空头" in reply)
        has_conclusion = ("建议" in reply or "综合" in reply or "总结" in reply)
        if has_bull_bear and has_conclusion:
            logic_score += 0.4
            logic_reasons.append("有多空分析+综合结论")
        # 分析维度关键词
        dimension_kws = ["技术面", "基本面", "资金面", "情绪面", "市场面", "政策面"]
        dim_hits = sum(1 for kw in dimension_kws if kw in reply)
        if dim_hits >= 2:
            logic_score += 0.4
            logic_reasons.append(f"覆盖{dim_hits}个分析维度")

        # 4. 风险提示 (0-2)
        risk_score = 0.0
        risk_reasons = []
        risk_kws = ["风险", "止损", "回撤", "仓位", "注意", "⚠️", "止盈"]
        risk_hits = sum(1 for kw in risk_kws if kw in reply)
        if risk_hits >= 3:
            risk_score += 1.2
            risk_reasons.append(f"提到了{risk_hits}个风险因素")
        elif risk_hits >= 2:
            risk_score += 0.8
            risk_reasons.append(f"提到了{risk_hits}个风险因素")
        elif risk_hits >= 1:
            risk_score += 0.4
        disclaimers = ["不构成", "历史数据不代表", "仅供参考", "投资建议", "风险提示"]
        if any(d in reply for d in disclaimers):
            risk_score += 1.0
            risk_reasons.append("包含免责声明")

        total = struct_score + acc_score + logic_score + risk_score

        return JudgeResult(
            structure_score=struct_score,
            accuracy_score=acc_score,
            logic_score=logic_score,
            risk_score=risk_score,
            total_score=min(10.0, total),
            overall_assessment=f"规则引擎评估 (LLM不可用: {reason})",
            improvements=(
                ["增加标题结构和编号列表"] if struct_score < 2 else []
            ) + (
                ["引用更多具体数值"] if acc_score < 2 else []
            ) + (
                ["加强因果推理链"] if logic_score < 1 else []
            ) + (
                ["添加风险提示和免责声明"] if risk_score < 1 else []
            ),
            details={
                "structure_reason": "; ".join(struct_reasons) if struct_reasons else "格式有改进空间",
                "accuracy_reason": "; ".join(acc_reasons) if acc_reasons else "缺少数据支撑",
                "logic_reason": "; ".join(logic_reasons) if logic_reasons else "逻辑需要更清晰",
                "risk_reason": "; ".join(risk_reasons) if risk_reasons else "需要添加风险提示",
                "fallback_reason": reason,
            },
        )


# ── 便捷函数 ──

async def judge_reply(
    question: str,
    reply: str,
    model_router=None,
    expected_points: Optional[List[str]] = None,
) -> dict:
    """快速评估 — 返回 dict"""
    judge = LLMJudge(model_router)
    result = await judge.evaluate(question, reply, expected_points)
    return result.to_dict()


def judge_reply_sync(question: str, reply: str, expected_points: Optional[List[str]] = None) -> dict:
    """同步快速评估 — 始终可用"""
    judge = LLMJudge()
    result = judge.evaluate_sync(question, reply, expected_points)
    return result.to_dict()
