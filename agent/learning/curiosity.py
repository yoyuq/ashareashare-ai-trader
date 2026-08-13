"""自动化 — 好奇主题队列 + 学习元报告。

让自主学习闭环"自主轮转"而不是每次重学同一批主题:
- 主题队列持久化到 simulation_data/learning_topic_queue.json (哪些主题学过/多久没学)。
- 好奇选择 pick_next_topics: 未学过的种子主题优先, 已学的按最久未学优先 (轮转)。
- 学习元报告 summarize_learned: 留/删分布、置信度、暂不可测缺口、已滚动重测条目。

全程零模拟: 队列/报告只是调度与统计, 不涉及行情/回测; 实际测试仍走 tester.py 真实数据。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

QUEUE_PATH = "simulation_data/learning_topic_queue.json"


# ── 主题队列 ──

def load_queue(path: str = QUEUE_PATH) -> dict:
    """加载主题队列状态; 不存在/损坏返回空 (不报错, 惰性重建)。"""
    if not os.path.exists(path):
        return {"topics": {}, "order": []}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[好奇队列] 读取失败, 重建空队列: {e}")
        return {"topics": {}, "order": []}


def save_queue(state: dict, path: str = QUEUE_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_next_topics(state: dict, seed_topics: list[str], n: int = 3,
                     today: Optional[str] = None) -> list[str]:
    """好奇选择 (纯函数): 未学种子优先, 已学的按最久未学优先轮转。

    Args:
        state: load_queue 返回的状态 dict。
        seed_topics: 种子主题池 (researcher.SEED_TOPICS)。
        n: 本次要选的主题数。
        today: 当前日期 (ISO), 缺省今天。
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    known = state.get("topics", {})
    # 1) 从未学过的种子主题 → 最高优先 (好奇探索)
    fresh = [t for t in seed_topics if t not in known]
    # 2) 已学过的种子主题 → 按 last_learned 升序 (最旧优先)
    stale = sorted(
        (t for t in seed_topics if t in known),
        key=lambda t: known[t].get("last_learned", ""),
    )
    # 3) 队列里非种子的额外主题 → 排最后
    extra = [t for t in state.get("order", []) if t not in seed_topics and t not in fresh]
    return (fresh + stale + extra)[:n]


def mark_learned(state: dict, topics: list[str], n_learned: int = 1,
                 today: Optional[str] = None) -> dict:
    """标记一批主题已学 (原地修改并返回 state)。"""
    today = today or datetime.now().strftime("%Y-%m-%d")
    topics_map = state.setdefault("topics", {})
    order = state.setdefault("order", [])
    for t in topics:
        e = topics_map.setdefault(t, {"last_learned": today, "n_learned": 0})
        e["last_learned"] = today
        e["n_learned"] = int(e.get("n_learned", 0)) + int(n_learned)
        if t not in order:
            order.append(t)
    return state


# ── 学习元报告 ──

def summarize_learned(records: list[dict]) -> dict:
    """学习元报告 (纯函数): 留/删分布 + 置信度 + 暂不可测缺口 + 已重测条目。

    Args:
        records: KnowledgeManager.list_learned() 返回的列表 (每项含 meta)。
    """
    metas = [r.get("meta", {}) for r in records]
    by_verdict = dict(Counter(m.get("verdict", "?") for m in metas))
    by_category = dict(Counter(m.get("category", "?") for m in metas))
    confs = []
    for m in metas:
        try:
            confs.append(float(m.get("confidence", 0.5) or 0.5))
        except (TypeError, ValueError):
            confs.append(0.5)
    verdict_map = {
        "verified": [m.get("concept", "") for m in metas if m.get("verdict") == "verified"],
        "rejected": [m.get("concept", "") for m in metas if m.get("verdict") == "rejected"],
        "inconclusive": [m.get("concept", "") for m in metas if m.get("verdict") == "inconclusive"],
        "not_yet_testable": [m.get("concept", "") for m in metas if m.get("verdict") == "not_yet_testable"],
    }
    revalidated = [m.get("concept", "") for m in metas
                   if int(m.get("revalidations", 0) or 0) > 0]
    return {
        "total": len(records),
        "by_verdict": by_verdict,
        "by_category": by_category,
        "avg_confidence": round(sum(confs) / len(confs), 2) if confs else 0.0,
        "concepts": verdict_map,
        "revalidated": revalidated,
    }


def format_meta_report(records: list[dict]) -> str:
    """把元报告渲染成可读 markdown (供周度 summary / 日志)。"""
    s = summarize_learned(records)
    lines = [f"## 自主学习闭环 · 学习元报告 (共 {s['total']} 条)"]
    lines.append(f"- 结论分布: {s['by_verdict']}")
    lines.append(f"- 类别分布: {s['by_category']}")
    lines.append(f"- 平均置信度: {s['avg_confidence']}")
    for v, label in [("verified", "✅ 已验证"), ("rejected", "❌ 已证伪"),
                     ("inconclusive", "⚠️ 证据不足"), ("not_yet_testable", "◻ 暂不可测")]:
        cs = s["concepts"].get(v, [])
        if cs:
            lines.append(f"- {label}: {', '.join(cs)}")
    if s["revalidated"]:
        lines.append(f"- 🔁 已滚动重测 (out-of-sample): {', '.join(s['revalidated'])}")
    return "\n".join(lines)
