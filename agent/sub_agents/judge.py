"""
Judge — 策略裁判子Agent (v3.0-competition)

作为AI+金融智能体的子Agent,负责综合Bull/Bear双方论据,
基于同一份量化数据做出公正裁决,输出结构化JSON判定。

用于: 对抗辩论最终裁决 — 决定BUY/HOLD/SELL动作
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Verdict:
    """裁判裁决"""
    action: str                         # BUY / HOLD / SELL
    conviction: float                   # 确信度 0.0-1.0
    score: float                        # 评分 1-10 (10=强烈看多)
    key_reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    bull_quality: float = 3.0           # 多头论据质量 1-5
    bear_quality: float = 3.0           # 空头论据质量 1-5
    verdict_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "conviction": self.conviction,
            "score": self.score,
            "key_reasons": self.key_reasons,
            "risks": self.risks,
            "bull_quality": self.bull_quality,
            "bear_quality": self.bear_quality,
            "verdict_summary": self.verdict_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        return cls(
            action=d.get("action", "HOLD"),
            conviction=float(d.get("conviction", 0.5)),
            score=float(d.get("score", 5.0)),
            key_reasons=d.get("key_reasons", []),
            risks=d.get("risks", []),
            bull_quality=float(d.get("bull_quality", 3.0)),
            bear_quality=float(d.get("bear_quality", 3.0)),
            verdict_summary=d.get("verdict_summary", ""),
        )


# 裁判系统Prompt
DEFAULT_SYSTEM_PROMPT = "你是公正的策略裁判。只输出JSON,不要输出其他内容。"


def build_judge_prompt(
    symbol: str,
    data_context: str,
    bull_response: str,
    bear_response: str,
) -> str:
    """构建裁判的用户Prompt"""
    return (
        f"你是A股策略裁判。以下是 {symbol} 的多空辩论,双方基于同一份量化数据:\n\n"
        f"=== 原始数据 ===\n{data_context[:2000]}\n\n"
        f"=== 多头论据 ===\n{bull_response[:1200]}\n\n"
        f"=== 空头论据 ===\n{bear_response[:1200]}\n\n"
        "请评估并输出 JSON (不要输出其他内容, 只输出 JSON):\n"
        "{\n"
        f'  "action": "BUY" | "HOLD" | "SELL",\n'
        '  "conviction": 0.0-1.0,\n'
        '  "score": 1-10 (10=强烈看多),\n'
        '  "key_reasons": ["论据1", "论据2", "论据3"],\n'
        '  "risks": ["风险1", "风险2"],\n'
        '  "bull_quality": 1-5 (多头论据质量),\n'
        '  "bear_quality": 1-5 (空头论据质量),\n'
        '  "verdict_summary": "一句话总结裁判意见"\n'
        "}"
    )


def parse_judge_response(response_text: str) -> Verdict:
    """解析LLM裁判的JSON输出"""
    judge_text = response_text.strip()
    # 提取 JSON (可能被包裹在 ```json ... ``` 中)
    if "```json" in judge_text:
        judge_text = judge_text.split("```json")[1].split("```")[0]
    elif "```" in judge_text:
        judge_text = judge_text.split("```")[1].split("```")[0]

    try:
        judge_json = json.loads(judge_text)
        return Verdict.from_dict(judge_json)
    except (json.JSONDecodeError, IndexError):
        # JSON 解析失败, 返回默认HOLD裁决
        return Verdict(
            action="HOLD",
            conviction=0.5,
            score=5.0,
            key_reasons=["JSON解析失败,使用默认裁定"],
            risks=["无法解析裁判输出"],
            verdict_summary="裁判输出格式异常,默认维持HOLD",
        )
