"""研究官 — 从外部(联网搜索 + LLM 知识)获取候选 A 股知识。

闭环第一步: 给定一个学习主题, 联网搜索相关资料, 让 LLM 把资料 + 自身
已训练的经济学/交易大师知识提炼成结构化候选知识, 供后续测试/留删/记库。

联网搜索后端可插拔 (SEARCH_PROVIDER=tavily|serper|bing|bocha|none), 读 .env;
未配 key 时优雅降级为纯 LLM 知识 (不报错), 仍能跑通闭环。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
from loguru import logger


@dataclass
class KnowledgeCandidate:
    """一条候选知识 (待测试)。"""
    concept: str                     # 短名, 如 "低PE价值投资"
    claim: str                       # 知识陈述, 如 "买入PE<10且ROE>15%的股票长期跑赢"
    category: str                    # "rule" (交易规则) | "fact" (事实/定义)
    testable: str                    # 可测试形式: rule→规则描述; fact→断言
    source: str = ""                 # 来源 url/标题 或 "llm_prior"
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"concept": self.concept, "claim": self.claim, "category": self.category,
                "testable": self.testable, "source": self.source,
                "confidence": self.confidence, "tags": self.tags}


SEED_TOPICS = [
    "价值投资策略", "成长股投资", "趋势跟踪策略", "均值回归策略", "动量策略",
    "仓位管理", "止损设置", "分散投资",
    "追涨杀跌", "损失厌恶", "处置效应",
]


_SEARCH_PROMPT = """你是研究官, 负责为 A 股交易系统搜集并提炼可测试的交易知识。

给定主题与搜索到的资料片段, 提炼出 3~5 条候选知识。每条:
- category: "rule"(可执行交易规则: 进出场/仓位/止损) 或 "fact"(事实/定义/市场机制)。
- testable: rule 写成一句话规则(如 "RSI<30 买入, RSI>70 卖出"); fact 写成可核验断言。
- 每条要具体、可操作, 不要鸡汤。优先提炼大师策略(价值/成长/趋势/均值回归等)与风控规则。

严格输出 JSON 数组 (无其他文字):
[{"concept":"...","claim":"...","category":"rule|fact","testable":"...","confidence":0.5~0.9}]
"""


def _search_config() -> tuple[str, str]:
    provider = os.getenv("SEARCH_PROVIDER", "none").strip().lower()
    key = os.getenv("SEARCH_API_KEY", "").strip()
    return provider, key


async def web_search(query: str, count: int = 5) -> list[dict]:
    """联网搜索, 返回 [{title, url, snippet}]。未配 provider/key → 返回 [] (降级)。"""
    provider, key = _search_config()
    if provider in ("", "none") or not key:
        logger.info("[研究官] 未配置 SEARCH_PROVIDER/SEARCH_API_KEY, 联网关闭 (仅用 LLM 知识)")
        return []
    try:
        if provider == "bocha":
            return await _bocha_search(query, key, count)
        logger.warning(f"[研究官] 未实现的搜索后端: {provider}")
        return []
    except Exception as e:
        logger.warning(f"[研究官] 联网搜索失败 ({provider}): {e}")
        return []


async def _bocha_search(query: str, key: str, count: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.bochaai.com/v1/web-search",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"query": query, "summary": True, "count": count, "freshness": "noLimit"},
        )
        r.raise_for_status()
        data = r.json()
    pages = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
    return [{"title": p.get("name", ""), "url": p.get("url", ""),
             "snippet": (p.get("summary") or p.get("snippet") or "")[:600]}
            for p in pages]


async def generate_candidates(topic: str, n: int = 5, search: bool = True) -> list[KnowledgeCandidate]:
    """联网搜 + LLM 提炼候选知识。"""
    snippets = await web_search(topic, count=5) if search else []
    search_txt = "\n".join(f"- [{s['title']}]({s['url']}): {s['snippet']}" for s in snippets[:5]) or "(无联网资料)"

    user_msg = f"【主题】{topic}\n【联网资料】\n{search_txt}\n\n请提炼 {n} 条候选知识。"
    from models.router import get_shared_router
    result = await get_shared_router().route(
        messages=[{"role": "system", "content": _SEARCH_PROMPT},
                  {"role": "user", "content": user_msg}],
        task_type="external_research",
        temperature=0.4,
        max_tokens=1200,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = (result.response or "").strip()
    arr = _parse_json_array(content)
    out: list[KnowledgeCandidate] = []
    for item in arr:
        if not item.get("concept") or not item.get("claim"):
            continue
        out.append(KnowledgeCandidate(
            concept=item["concept"], claim=item["claim"],
            category=("rule" if item.get("category") == "rule" else "fact"),
            testable=item.get("testable") or item["claim"],
            source=item.get("source", snippets[0]["url"] if snippets else "llm_prior"),
            confidence=float(item.get("confidence", 0.5)),
            tags=[topic],
        ))
    return out


def _parse_json_array(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, list) else ([obj] if isinstance(obj, dict) else [])
    except json.JSONDecodeError:
        s, e = content.find("["), content.rfind("]")
        if s >= 0 and e > s:
            try:
                return json.loads(content[s:e + 1])
            except json.JSONDecodeError:
                return []
        return []
