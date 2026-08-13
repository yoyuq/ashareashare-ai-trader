"""P4 — fact 型外部知识回写知识库。

把"已验证(consistent)"的 fact 追加进独立参考文件 `knowledge/reference/learned_facts.md`
(不混入人工维护的规则/reference, 明确标注外部学习来源), 并索引进向量库;
把"与规则库冲突(contradicts)"的 fact 记到人类复核日志, 绝不自动改人工规则;
把"规则库未覆盖(not_covered)"但高置信的 fact 记到待复核文件 (不入库, 留人工确认)。

纪律 (与全程零模拟/禁止事后调参追赢一致):
- 只写 `verified` 且置信度 >= FACT_INSERT_CONF 的 fact (保守, 防污染检索)。
- 写前按字符 Jaccard 对 learned_facts.md 已有条目查重, 近似重复不重写。
- 冲突只记日志, 不自动改 knowledge/rules/*.yaml (人类决定谁对)。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from .tester import TestResult

LEARNED_FACTS_FILE = "knowledge/reference/learned_facts.md"
CONTRADICTIONS_FILE = "reports/learning_contradictions.jsonl"
PENDING_REVIEW_FILE = "reports/facts_pending_review.jsonl"  # not_covered 高置信 → 人工复核
FACT_INSERT_CONF = 0.8        # 预注册门槛: 只有高置信 verified 才写知识库
PENDING_REVIEW_CONF = 0.8     # 预注册门槛: not_covered 高置信才进待复核 (低置信静默跳过)
DUP_JACCARD = 0.5             # 字符 Jaccard 查重阈值 (与 ExperienceMemory 一致)


def _char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _existing_claims() -> list[str]:
    """读取 learned_facts.md 已有条目的 claim (只取「陈述:」行), 供查重。"""
    p = Path(LEARNED_FACTS_FILE)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    claims = []
    for block in text.split("\n## "):
        for line in block.splitlines():
            s = line.strip()
            if s.startswith("陈述:") or s.startswith("- 陈述:"):
                claims.append(s.split(":", 1)[1].strip())
                break
    return claims


def _append_fact(candidate, reason: str) -> None:
    """把一条已验证 fact 追加进 learned_facts.md (幂等: 首次建头)。"""
    p = Path(LEARNED_FACTS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    if not p.exists():
        header = (
            "# 外部学习事实库 (learned_facts) — 自主学习闭环自动沉淀, 人工复核后可删改\n\n"
            "> 来源: agent/learning 研究官 (联网/LLM) → 测试器 fact 一致性核验 → 只收 verified(consistent)。\n"
            "> 每条标注来源/日期/置信度; 与人工规则冲突的记在 reports/learning_contradictions.jsonl。\n"
            "> 规则库未覆盖(not_covered)的高置信事实记在 reports/facts_pending_review.jsonl, 待人工复核。\n\n"
        )
        p.write_text(header, encoding="utf-8")
    entry = (
        f"\n## {candidate.concept}\n\n"
        f"- 陈述: {candidate.claim}\n"
        f"- 来源: {candidate.source or '外部学习'}\n"
        f"- 核验: verified (与规则库一致) — {reason}\n"
        f"- 置信度: {candidate.confidence}\n"
        f"- 日期: {today}\n"
    )
    with p.open("a", encoding="utf-8") as f:
        f.write(entry)


def _log_contradiction(candidate, reason: str) -> None:
    """把与规则库冲突的 fact 记到人类复核日志 (jsonl)。"""
    p = Path(CONTRADICTIONS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "concept": candidate.concept,
        "claim": candidate.claim,
        "source": candidate.source,
        "reason": reason,
        "action_required": "人工复核: 外部知识 vs 规则库冲突, 决定保留哪一方",
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _log_pending_review(candidate, reason: str) -> None:
    """把规则库未覆盖(not_covered)但高置信的 fact 记到待复核文件 (不入库, 留人工确认)。"""
    p = Path(PENDING_REVIEW_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "concept": candidate.concept,
        "claim": candidate.claim,
        "category": candidate.category,
        "source": candidate.source,
        "confidence": candidate.confidence,
        "reason": reason,
        "action_required": "人工复核: 规则库未覆盖该维度, 判断是否值得补进知识库",
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def apply_fact_to_kb(candidate, test_result: TestResult, km=None,
                     insert_conf: float = FACT_INSERT_CONF) -> str:
    """把 fact 型测试结论回写到知识库。返回动作字符串 (纯逻辑 + 文件 IO)。

    - verified + 高置信      → 追加 learned_facts.md + 索引 → "inserted"
    - verified + 低置信      → "skipped_low_conf"
    - verified + 近似重复    → "skipped_dup"
    - rejected (contradicts) → 记人类复核日志 → "contradiction_logged"
    - inconclusive (not_covered) + 高置信 → 记待复核文件(不入库) → "pending_review"
    - inconclusive + 低置信  → "skipped"
    - 其他                   → "skipped"
    """
    if test_result.verdict == "verified":
        if float(candidate.confidence or 0.0) < insert_conf:
            return "skipped_low_conf"
        for c in _existing_claims():
            if _char_jaccard(candidate.claim, c) >= DUP_JACCARD:
                return "skipped_dup"
        _append_fact(candidate, test_result.reason)
        if km is not None:
            try:
                km.index_document(LEARNED_FACTS_FILE, doc_type="learned_facts")
            except Exception as e:
                logger.warning(f"[知识库回写] 索引 learned_facts.md 失败: {e}")
        return "inserted"
    if test_result.verdict == "rejected":
        _log_contradiction(candidate, test_result.reason)
        return "contradiction_logged"
    if test_result.verdict == "inconclusive":
        # not_covered: 规则库没写该维度 (沉默≠冲突)。高置信 → 待人工复核, 不入库。
        if float(candidate.confidence or 0.0) >= PENDING_REVIEW_CONF:
            _log_pending_review(candidate, test_result.reason)
            return "pending_review"
        return "skipped"
    return "skipped"
