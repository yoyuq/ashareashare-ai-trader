"""RAG 检索基准 — 哈希向量 vs 关键词 vs 混合 RRF

一组带已知答案来源的查询, 测三种检索模式的 Precision@1/@3:
  - vector  (纯哈希向量)
  - keyword (纯 BM25 风格关键词)
  - hybrid  (RRF 融合)

用法: python scripts/benchmark_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.manager import KnowledgeManager

# (查询, 期望来源文件)
BENCH = [
    ("牛市 追涨 动量 强势股 选股 操作", "regime_playbook.md"),
    ("危机 现金为王 降仓位 止损 反弹", "regime_playbook.md"),
    ("震荡市 低估值 超跌反弹 追高谨慎", "regime_playbook.md"),
    ("熊市 防御 低估值 高股息 动量毒药", "regime_playbook.md"),
    ("弱牛市 动量减弱 基本面 估值", "regime_playbook.md"),
    ("A股 牛熊周期 历史 大涨 股灾", "market_cycle.md"),
    ("市盈率 市净率 财报 基本面 分析 阈值", "fundamental_analysis.md"),
    ("换手率 术语 定义 指标", "glossary.md"),
    ("资本 资产 匹配 机制", "capital_asset_matching.md"),
    ("止盈 止损 仓位 通用原则", "regime_playbook.md"),
]


def hit_rate(km: KnowledgeManager, mode: str, topk: int) -> float:
    hits = 0
    details = []
    for q, expected in BENCH:
        res = km._retrieve(q, top_k=topk, mode=mode)
        sources = {r.get("source", "") for r in res}
        ok = expected in sources
        hits += 1 if ok else 0
        details.append((ok, q[:18], expected, [r.get("source", "")[:22] for r in res[:2]]))
    for ok, q, exp, got in details:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {q:<20} 期望={exp:<22} 命中={got}")
    return hits / len(BENCH)


def main():
    km = KnowledgeManager()
    km._ensure_knowledge_base_collection()
    print(f"基准: {len(BENCH)} 查询, collection knowledge_base_v2\n")
    for topk in (1, 3):
        print(f"=== Precision@{topk} ===")
        for mode in ("vector", "keyword", "hybrid"):
            rate = hit_rate(km, mode, topk)
            print(f"  [{mode:8}] P@{topk} = {rate:.0%}\n")
        print()


if __name__ == "__main__":
    main()
