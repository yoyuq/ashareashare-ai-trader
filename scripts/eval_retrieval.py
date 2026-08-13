"""知识库检索质量离线评测 (recall@k 基准)

背景 (P1-17): 检索用自研 md5 加权 n-gram 哈希向量, 无标准 embedding、无 recall@k 基准。
此脚本建立离线评测: 人工标注的查询 → 应命中源 (source 文件名), 对 vector/keyword/hybrid
三路检索分别计算 recall@k (top-k 内是否命中任一 relevant 源)。

用法:
    python scripts/eval_retrieval.py              # 重建索引 + 全量评测
    python scripts/eval_retrieval.py --k 5        # 自定义 top-k

指标口径:
    recall@k = 命中相关源的查询数 / 查询总数 (每条查询至少一个 relevant 源进 top-k 记命中)
    命中判定依据 chunk 的 metadata.source (文件名), 与 _seed_knowledge_base_v2 写入一致。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.manager import KnowledgeManager  # noqa: E402

# 人工标注 gold 集: 查询 → 应命中的 source 文件名 (md 或 yaml, 允许多个 relevant)
GOLD_QUERIES = [
    ("印花税税率是多少", ["trading_rules.yaml"]),
    ("当日买入的股票什么时候能卖 T+1", ["trading_rules.yaml"]),
    ("主板涨跌停幅度限制", ["trading_rules.yaml"]),
    ("券商佣金最低收费多少", ["trading_rules.yaml"]),
    ("PE市盈率如何判断高估还是低估", ["fundamental_analysis.md"]),
    ("2015年股灾大盘跌了多少", ["market_cycle.md"]),
    ("震荡市应该用什么交易策略", ["regime_playbook.md", "registry.yaml"]),
    ("十字星是什么K线形态", ["indicator_guide.yaml"]),
    ("RSI超买超卖的阈值", ["indicator_guide.yaml", "hardened_definitions.yaml"]),
    ("放量和缩量是怎么定义的", ["hardened_definitions.yaml", "trading_rules.yaml"]),
    ("布林带均值回归策略怎么用", ["registry.yaml"]),
    ("北向资金跟随策略", ["registry.yaml"]),
]

MODES = ["vector", "keyword", "hybrid"]


def evaluate(km: KnowledgeManager, k: int = 5) -> dict:
    """对 GOLD_QUERIES 逐条跑三路检索, 返回各模式 recall@k 与逐条明细。"""
    report = {}
    for mode in MODES:
        hits, rows = 0, []
        for query, relevant in GOLD_QUERIES:
            picked = km.retrieve(query, top_k=k, mode=mode)
            got = {r.get("source", "") for r in picked}
            hit = bool(got & set(relevant))
            hits += int(hit)
            rows.append({
                "query": query, "relevant": relevant,
                "got": sorted(got)[:k], "hit": hit,
            })
        report[mode] = {
            "recall_at_k": round(hits / len(GOLD_QUERIES), 3),
            "hits": hits, "total": len(GOLD_QUERIES),
            "rows": rows,
        }
    return report


def print_report(report: dict, k: int) -> None:
    print(f"\n知识库检索 recall@{k} 评测 (共 {len(GOLD_QUERIES)} 条 gold 查询)\n")
    print(f"{'查询':<26} {'vector':<8} {'keyword':<9} {'hybrid':<8}  相关源")
    for i, (query, relevant) in enumerate(GOLD_QUERIES):
        marks = [("hit" if report[m]["rows"][i]["hit"] else "-") for m in MODES]
        print(f"{query:<26} {marks[0]:<8} {marks[1]:<9} {marks[2]:<8}  {'/'.join(relevant)}")
    print("\n" + "-" * 70)
    for m in MODES:
        r = report[m]
        print(f"{m:<8} recall@{k} = {r['recall_at_k']:.3f}  ({r['hits']}/{r['total']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库检索 recall@k 离线评测")
    parser.add_argument("--k", type=int, default=5, help="top-k (默认 5)")
    parser.add_argument("--no-rebuild", action="store_true", help="跳过重建, 用现有索引")
    args = parser.parse_args()

    km = KnowledgeManager()
    if not args.no_rebuild:
        n = km.rebuild_knowledge_index()
        if n == 0:
            print("[失败] chromadb 不可用或索引重建失败, 无法评测")
            return 1
        print(f"[重建] knowledge_base_v2 共 {n} chunks")

    report = evaluate(km, k=args.k)
    print_report(report, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
