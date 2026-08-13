"""自主学习闭环 runner — 研究官 → RAG 查重 → 测试 → 记向量库。

用法:
  python scripts/learn_external.py --topics 价值投资策略 --dry-run
  python scripts/learn_external.py --topics 价值投资策略,仓位管理
  python scripts/learn_external.py --topics 价值投资策略 --data-file replay_data/daily_2020-06-01_2021-02-28.parquet

全程零模拟: 测试用真实 replay_data/*.parquet; 缺数据即报错。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger


def _default_data_files() -> list[str]:
    """默认真实回放窗口 (预注册, 非事后挑选)。

    跨市场状态选 3 个窗口, 避免"只在牛市测试→简单规则永远跑输买入持有"的失真:
    - 2018 全年: 熊市 (A股系统性下跌, 检验风险规则降险)
    - 2020-06→2021-02: 牛市 (趋势/动量主场)
    - 2024 全年: 震荡 (含显著回撤段)
    """
    import glob
    candidates = [
        "replay_data/daily_2018-01-01_2018-12-31.parquet",
        "replay_data/daily_2020-06-01_2021-02-28.parquet",
        "replay_data/daily_2024-01-01_2024-12-31.parquet",
    ]
    have = [c for c in candidates if os.path.exists(c)]
    if have:
        return have
    found = sorted(glob.glob("replay_data/daily_*.parquet"))
    found = [f for f in found if not f.endswith("_idx.parquet")]
    return found[:3]


def _all_replay_files() -> list[str]:
    """全部真实回放窗口 (供滚动重测挑选新鲜 out-of-sample 窗口)。"""
    import glob
    found = sorted(glob.glob("replay_data/daily_*.parquet"))
    return [f for f in found if not f.endswith("_idx.parquet")]


def _window_name(path: str) -> str:
    return path.split("daily_")[-1].replace(".parquet", "")


async def run_learning_loop(topics, data_files, dry_run=False, n=5) -> dict:
    from knowledge.manager import KnowledgeManager
    from agent.learning.researcher import generate_candidates
    from agent.learning.knowledge_history import KnowledgeHistory
    from agent.learning import tester as T

    km = KnowledgeManager("knowledge/")
    history = KnowledgeHistory(km)
    report = {"date": datetime.now().isoformat(timespec="seconds"), "topics": topics,
              "learned": [], "already_learned": [], "errors": []}

    for topic in topics:
        logger.info(f"=== 研究主题: {topic} ===")
        try:
            candidates = await generate_candidates(topic, n=n, search=not dry_run)
        except Exception as e:
            logger.error(f"[研究官] 主题 {topic} 候选生成失败: {e}")
            report["errors"].append({"topic": topic, "error": str(e)})
            continue

        # 防概念撞名: 同一主题内多个候选同名会互相覆盖 (record_learned 按 concept 幂等)
        seen = set()
        for cand in candidates:
            if cand.concept in seen:
                logger.info(f"[撞名跳过] {cand.concept} (本主题已处理, 避免向量库覆盖)")
                continue
            seen.add(cand.concept)
            try:
                # 1. RAG 查重: 学过就复用旧结论
                learned = await history.check_learned(cand)
                if learned is not None:
                    logger.info(f"[查重命中] {cand.concept} → 已学过({learned.prior_verdict}), 跳过")
                    report["already_learned"].append({
                        "concept": cand.concept, "prior_verdict": learned.prior_verdict,
                        "matched_concept": learned.matched_concept, "reason": learned.reason})
                    continue

                # 2. 测试 (真实数据留/删)
                if cand.category == "fact":
                    res = await T.test_fact(cand, km)
                else:
                    res = await T.test_rule(cand, data_files)

                # 2b. P4: fact 型回写知识库 (verified 追加 learned_facts.md / 冲突记人类复核日志)
                kb_action = ""
                if cand.category == "fact":
                    try:
                        from agent.learning.kb_writer import apply_fact_to_kb
                        kb_action = apply_fact_to_kb(cand, res, km)
                        if kb_action in ("inserted", "contradiction_logged", "pending_review"):
                            logger.info(f"[知识库回写] {cand.concept} → {kb_action}")
                    except Exception as e:
                        logger.warning(f"[知识库回写] {cand.concept} 失败: {e}")

                # 3. 记录到历史向量库
                ok = history.record(cand, res.verdict, res.to_dict(),
                                    date=datetime.now().strftime("%Y-%m-%d"))
                logger.info(f"[{cand.category}] {cand.concept} → {res.verdict} (记录:{ok}) {res.reason[:60]}")
                report["learned"].append({"concept": cand.concept, "category": cand.category,
                                          "verdict": res.verdict, "reason": res.reason,
                                          "kb_action": kb_action,
                                          "metric_delta": res.metric_delta})
            except Exception as e:
                logger.error(f"[闭环] 候选 {cand.concept} 处理失败: {e}")
                report["errors"].append({"concept": cand.concept, "error": str(e)})

    return report


def _fresh_windows_for_record(md: dict, all_files: list[str]) -> list[str]:
    """为该记录挑选新鲜窗口 (排除其原始 windows_tested), 保证滚动重测是 out-of-sample。"""
    import json as _json
    tested = set()
    wt = md.get("windows_tested")
    if wt:
        try:
            tested = set(_json.loads(wt)) if isinstance(wt, str) else set(wt)
        except (_json.JSONDecodeError, TypeError):
            tested = set()
    fresh = [f for f in all_files if _window_name(f) not in tested]
    return fresh


def run_revalidation(data_files=None, universe_size=20,
                     only_verdicts=("verified", "inconclusive", "rejected")) -> dict:
    """P2 实战反馈闭环: 对已学 rule 在新鲜 out-of-sample 窗口重测, 诚实升级/降级。

    - 复用存储的 template+params (不重译, 避免 LLM 漂移), 同一预注册判据 + 新数据。
    - 新鲜窗口 = 全部 replay 窗口 - 该记录原始 windows_tested; 无新鲜窗口则跳过 (不做 in-sample 复读)。
    - v5.10: 默认也重测 rejected (判据 v2 新增降险通道, 旧 rejected 可能翻 verified); params 经 _sanitize_params 防幻觉。
    - 全程零模拟: 缺数据即报错跳过。
    """
    import json as _json
    from knowledge.manager import KnowledgeManager
    from agent.learning.knowledge_history import merge_revalidation
    from agent.learning import tester as T

    km = KnowledgeManager("knowledge/")
    all_files = data_files or _all_replay_files()
    records = km.list_learned(categories=["rule"], verdicts=list(only_verdicts))
    date_str = datetime.now().strftime("%Y-%m-%d")
    report = {"date": datetime.now().isoformat(timespec="seconds"),
              "data_files": all_files, "revalidated": [], "skipped": [], "errors": []}

    for rec in records:
        concept = rec.get("concept", "")
        md = rec.get("meta", {})
        template = md.get("template")
        if not template or template == "not_yet_testable":
            report["skipped"].append({"concept": concept, "reason": "无模板/暂不可测"})
            continue
        # params 存为 JSON 字符串 (ChromaDB 元数据只收标量)
        params_raw = md.get("params") or "{}"
        try:
            params = _json.loads(params_raw) if isinstance(params_raw, str) else dict(params_raw or {})
        except _json.JSONDecodeError:
            params = {}
        fresh_files = _fresh_windows_for_record(md, all_files)
        if not fresh_files:
            report["skipped"].append({"concept": concept, "reason": "无新鲜 out-of-sample 窗口"})
            continue
        deltas = T._run_rule_deltas(template, params, fresh_files, universe_size)
        if not deltas:
            report["errors"].append({"concept": concept, "error": "无可用窗口完成重测"})
            continue
        fresh = T.apply_keep_criterion(deltas)
        old_verdict = md.get("verdict", "inconclusive")
        old_conf = float(md.get("confidence", 0.5) or 0.5)
        old_rv = int(md.get("revalidations", 0) or 0)
        merged = merge_revalidation(old_verdict, fresh.verdict, old_conf, old_rv)

        # 从 doc 还原 claim (doc = "concept\\nclaim" 截断至 800)
        doc = rec.get("doc", "")
        claim = doc.split("\n", 1)[1] if "\n" in doc else doc
        new_meta = {k: v for k, v in md.items()
                    if k not in ("concept", "category", "verdict", "confidence", "revalidations")}
        new_meta["confidence"] = merged["confidence"]
        new_meta["revalidations"] = merged["revalidations"]
        new_meta["last_revalidated"] = date_str
        new_meta["revalidation_verdict"] = fresh.verdict
        new_meta["fresh_windows"] = _json.dumps(deltas, ensure_ascii=False)
        ok = km.record_learned(concept=concept, claim=claim, verdict=merged["verdict"],
                               category=md.get("category", "rule"), meta=new_meta)
        logger.info(f"[重测] {concept}: {old_verdict} → {merged['verdict']} ({merged['action']}, "
                    f"conf {old_conf:.2f}→{merged['confidence']:.2f}, rv {merged['revalidations']})")
        report["revalidated"].append({
            "concept": concept, "old_verdict": old_verdict, "new_verdict": merged["verdict"],
            "fresh_verdict": fresh.verdict, "action": merged["action"],
            "confidence": merged["confidence"], "revalidations": merged["revalidations"],
            "fresh_windows": deltas,
        })
    return report


def main():
    # Windows 控制台默认 GBK, 打印中文/emoji 会 UnicodeEncodeError; 重配 stdout/stderr 为 UTF-8
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="自主学习闭环")
    ap.add_argument("--topics", default="", help="逗号分隔的主题; 留空用默认种子")
    ap.add_argument("--data-file", action="append", default=[], help="真实回放 parquet (可多次); 缺省用预注册窗口")
    ap.add_argument("--dry-run", action="store_true", help="研究官不联网 (仅 LLM 知识)")
    ap.add_argument("--revalidate", action="store_true",
                    help="滚动重测已学 rule (不联网/不研究, 只对历史结论在新鲜窗口重验证)")
    ap.add_argument("--auto", type=int, metavar="N", default=0,
                    help="自动从好奇队列选 N 个主题学习 (轮转/探索, 学完写回队列+元报告)")
    ap.add_argument("--report", action="store_true", help="仅输出学习元报告 (不学习/不联网)")
    ap.add_argument("--out", default="", help="报告输出 json 路径")
    args = ap.parse_args()

    out = args.out or f"reports/external_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if args.report:
        from knowledge.manager import KnowledgeManager
        from agent.learning.curiosity import format_meta_report
        km = KnowledgeManager("knowledge/")
        text = format_meta_report(km.list_learned())
        print(text)
        Path(out).write_text(text, encoding="utf-8")
        print(f"\n元报告已写入: {out}")
        return

    if args.revalidate:
        data_files = args.data_file or None  # None → 全部 replay 窗口
        logger.info(f"滚动重测模式, 窗口: {data_files or '全部 replay 窗口 (排除原始已测)'}")
        report = run_revalidation(data_files)
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n重测报告已写入: {out}")
        return

    from agent.learning.researcher import SEED_TOPICS

    # 好奇队列自动选题
    if args.auto > 0:
        from agent.learning.curiosity import (load_queue, save_queue, pick_next_topics,
                                              mark_learned, summarize_learned, format_meta_report)
        from knowledge.manager import KnowledgeManager
        state = load_queue()
        topics = pick_next_topics(state, SEED_TOPICS, n=args.auto)
        if not topics:
            topics = SEED_TOPICS[:args.auto]
        logger.info(f"[自动] 好奇队列选出主题: {topics}")
    else:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()] or SEED_TOPICS

    data_files = args.data_file or _default_data_files()
    logger.info(f"主题: {topics}")
    logger.info(f"回放窗口: {data_files}")

    report = asyncio.run(run_learning_loop(topics, data_files, dry_run=args.dry_run))

    # 自动模式: 写回队列 + 追加元报告
    if args.auto > 0:
        from agent.learning.curiosity import mark_learned, save_queue, summarize_learned, format_meta_report
        from knowledge.manager import KnowledgeManager
        mark_learned(state, topics, n_learned=1)
        save_queue(state)
        km = KnowledgeManager("knowledge/")
        report["meta"] = summarize_learned(km.list_learned())
        report["meta_text"] = format_meta_report(km.list_learned())

    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("meta_text"):
        print("\n" + report["meta_text"])
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
