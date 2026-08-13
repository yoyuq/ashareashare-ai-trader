"""历史向量库 — 自主学习闭环的记忆层。

把"学过的知识 + 留/删结论"写进 ChromaDB (learned_knowledge 集合),
并在学习新知识前做 RAG 查重: 向量召回 + LLM 语义判重(确认是否学过、复用旧结论)。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from .researcher import KnowledgeCandidate


@dataclass
class LearnedVerdict:
    already_learned: bool
    matched_concept: str = ""
    prior_verdict: str = ""          # verified | rejected | inconclusive
    reason: str = ""


_DEDUP_PROMPT = """你是学习记忆的查重判官。

判断"新知识"是否与"已学过的知识"语义重复(同一条知识的不同表述也算重复)。
语义重复 = 讲的是同一件事/同一策略, 只是措辞不同。

严格输出 JSON (无其他文字):
{"already_learned": true/false, "matched_concept": "命中的旧知识短名", "reason": "一句话判断依据"}
"""


class KnowledgeHistory:
    def __init__(self, km):
        self.km = km

    def record(self, candidate: KnowledgeCandidate, verdict: str,
               test_result: Optional[dict] = None, date: str = "") -> bool:
        """记录一条学过的知识 + 留/删结论到向量库。"""
        meta = dict(test_result or {})
        if date:
            meta["date"] = date
        meta["source"] = candidate.source
        meta["confidence"] = candidate.confidence
        return self.km.record_learned(
            concept=candidate.concept, claim=candidate.claim, verdict=verdict,
            category=candidate.category, meta=meta,
        )

    def recall(self, concept: str, top_k: int = 5) -> list[dict]:
        """向量召回候选 (供查重)。"""
        return self.km.recall_learned(concept, top_k=top_k)

    async def judge_learned(self, candidate: KnowledgeCandidate, recalled: list[dict]) -> LearnedVerdict:
        """LLM 语义判重: 新知识与召回候选是否同一条知识。"""
        if not recalled:
            return LearnedVerdict(already_learned=False, reason="无召回候选")
        old_txt = "\n".join(f"- [{r['verdict']}] {r['concept']}: {r['doc'][:120]}" for r in recalled)
        user_msg = (f"【新知识】概念: {candidate.concept}\n陈述: {candidate.claim}\n\n"
                    f"【已学过的候选】\n{old_txt}\n\n请判断是否语义重复。")
        from models.router import get_shared_router
        result = await get_shared_router().route(
            messages=[{"role": "system", "content": _DEDUP_PROMPT},
                      {"role": "user", "content": user_msg}],
            task_type="learned_dedup", temperature=0.2, max_tokens=300,
            extra_body={"thinking": {"type": "disabled"}},
        )
        data = self._parse_json((result.response or "").strip())
        already = bool(data.get("already_learned", False))
        matched = data.get("matched_concept", "")
        prior = ""
        if already:
            for r in recalled:
                if matched and r["concept"] == matched:
                    prior = r["verdict"]
                    break
            prior = prior or (recalled[0]["verdict"] if recalled else "")
        return LearnedVerdict(already_learned=already, matched_concept=matched or (recalled[0]["concept"] if recalled else ""),
                              prior_verdict=prior, reason=data.get("reason", ""))

    async def check_learned(self, candidate: KnowledgeCandidate, top_k: int = 5) -> Optional[LearnedVerdict]:
        """查重: 语义命中则返回旧结论, 否则 None (未学过)。"""
        recalled = self.recall(candidate.concept, top_k=top_k)
        if not recalled:
            return None
        v = await self.judge_learned(candidate, recalled)
        return v if v.already_learned else None

    @staticmethod
    def _parse_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            s, e = content.find("{"), content.rfind("}")
            if s >= 0 and e > s:
                try:
                    return json.loads(content[s:e + 1])
                except json.JSONDecodeError:
                    pass
        return {}


# ── 消费路径: 已学知识注入诊断提示词 (LEARNED_KNOWLEDGE_GATE, 默认关) ──

def learned_gate() -> int:
    """读取已学知识注入门槛 LEARNED_KNOWLEDGE_GATE (默认 0 = 关闭, 现状不变).

    0 = 不注入 (默认)
    1 = 只注入 verified 规则
    2 = 只注入 verified 且高置信 (confidence >= 0.75)
    """
    try:
        return int(os.getenv("LEARNED_KNOWLEDGE_GATE", "0"))
    except (TypeError, ValueError):
        return 0


_VERDICT_BADGE = {
    "verified": "✅ 已验证",
    "rejected": "❌ 已证伪(勿用)",
    "inconclusive": "⚠️ 证据不足",
    "not_yet_testable": "◻ 暂不可测",
}


def _format_learned_items(items: list[dict], gate: int) -> str:
    """纯函数: 把召回的知识条目按 gate 过滤并格式化成提示词片段 (供单测)。"""
    if gate <= 0 or not items:
        return ""
    if gate == 1:
        items = [r for r in items if r.get("verdict") == "verified"]
    else:
        items = [r for r in items if r.get("verdict") == "verified"
                 and float(r.get("confidence", 0.5) or 0.5) >= 0.75]
    if not items:
        return ""
    lines = ["## 你已学习的外部交易知识（来自策略研究，经真实数据验证）"]
    for i, r in enumerate(items, 1):
        badge = _VERDICT_BADGE.get(r.get("verdict", ""), r.get("verdict", ""))
        lines.append(f"{i}. [{r.get('category', 'rule')}] {r.get('concept', '')} {badge}\n"
                     f"   → {r.get('doc', '')[:200]}")
    lines.append("")
    return "\n".join(lines)


def merge_revalidation(old_verdict: str, fresh_verdict: str,
                       old_conf: float, old_revalidations: int) -> dict:
    """滚动重测结果合并旧结论 (纯函数, 预注册常量, 禁止事后调参追赢)。

    诚实升级/降级:
    - fresh verified     → 确认 verified, 置信度 +0.10 (封顶 0.95), revalidations+1
    - fresh rejected     → 证伪 rejected (诚实降级), 置信度 -0.25 (保底 0.30), revalidations+1
    - fresh inconclusive → 维持旧结论, 置信度 -0.05 (保底 0.40), revalidations+1 (证据不足不翻转)
    - fresh not_yet_testable → 全不变 (测不了)

    confidence 语义 = 证据强度 (研究员置信度为起点, 每次重测增减), 供 gate 2 过滤。
    """
    conf = float(old_conf or 0.5)
    rv = int(old_revalidations or 0)
    if fresh_verdict == "verified":
        return {"verdict": "verified", "confidence": round(min(0.95, conf + 0.10), 2),
                "revalidations": rv + 1, "action": "confirmed"}
    if fresh_verdict == "rejected":
        return {"verdict": "rejected", "confidence": round(max(0.30, conf - 0.25), 2),
                "revalidations": rv + 1, "action": "falsified"}
    if fresh_verdict == "inconclusive":
        return {"verdict": old_verdict, "confidence": round(max(0.40, conf - 0.05), 2),
                "revalidations": rv + 1, "action": "unchanged"}
    return {"verdict": old_verdict, "confidence": round(conf, 2),
            "revalidations": rv, "action": "untestable"}


def format_learned_for_prompt(query: str = "市场诊断", top_k: int = 8,
                              gate: Optional[int] = None) -> str:
    """已学知识注入入口: gate<=0 时返回 "" (零开销, 不碰 ChromaDB)。

    供 daily_runner.build_diagnostic_system_prompt 注入第四节; 默认关闭,
    LEARNED_KNOWLEDGE_GATE=1/2 才启用 (预注册 A/B 后翻 gate)。
    """
    _gate = gate if gate is not None else learned_gate()
    if _gate <= 0:
        return ""
    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager("knowledge/")
        recalled = km.recall_learned(query, top_k=top_k)
    except Exception as e:
        logger.warning(f"[已学知识注入] 召回失败, 跳过注入: {e}")
        return ""
    return _format_learned_items(recalled, _gate)
