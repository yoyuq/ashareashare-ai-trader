"""重建知识库向量索引 (三层联动: YAML 规则 + 参考文档 → ChromaDB)

背景 (P1-17): 知识库三层「各自为政」——改 rules/*.yaml 不联动进向量库, 向量检索
找不到任何规则定义。此脚本把 md + YAML 全量重新索引进 knowledge_base_v2。

用法:
    python scripts/rebuild_knowledge_index.py            # 重建 + 打印 chunk 数
    python scripts/rebuild_knowledge_index.py --check    # 仅校验规则 schema, 不重建
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.manager import KnowledgeManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重建知识库向量索引 (YAML + md)")
    parser.add_argument("--check", action="store_true",
                        help="仅校验 trading_rules.yaml 的 schema, 不重建索引")
    args = parser.parse_args()

    km = KnowledgeManager()

    # 1. 规则 schema 锁定校验 (先校验再重建, 坏结构不进向量库)
    schema_result = km.validate_rules_schema()
    if not schema_result["validated_schema"]:
        print("[跳过] jsonschema 未安装或 schema 文件缺失, 未做 schema 校验")
    elif not schema_result["ok"]:
        print("[失败] trading_rules.yaml 结构违反 schema:")
        for e in schema_result["errors"]:
            print(f"  - {e}")
        return 1
    else:
        print("[通过] trading_rules.yaml 结构符合 schema")

    if args.check:
        return 0

    # 2. 重建向量索引
    n = km.rebuild_knowledge_index()
    if n == 0:
        print("[失败] 向量索引重建返回 0 chunks (chromadb 不可用? 见上方日志)")
        return 1

    # 3. 快速冒烟: 检索一条规则查询, 确认 YAML 层已进向量库
    hit = km.retrieve("印花税税率是多少", top_k=5)
    sources = [r["source"] for r in hit]
    print(f"[完成] 已重建 {n} chunks")
    print(f"[冒烟] 查询「印花税税率是多少」→ 命中源: {sources}")
    if not any(s == "trading_rules.yaml" for s in sources):
        print("[警告] 规则查询未命中 trading_rules.yaml, 检索质量可疑 (见 eval_retrieval.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
