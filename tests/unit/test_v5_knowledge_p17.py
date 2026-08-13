"""v5.6 P1-17 知识库三层联动 + 检索评测 + 规则 schema

覆盖:
  1. YAML→向量库重建流程 (rebuild_knowledge_index + _chunk_yaml 中文标签富化)
  2. 检索质量离线评测 (hybrid/keyword/vector 的 recall@k 下界)
  3. 规则 YAML schema 锁定 (validate_rules_schema / JSON Schema 拒绝坏结构)
"""
import importlib.util
import json

import pytest

from knowledge.manager import KnowledgeManager, _chunk_yaml

HAS_CHROMA = importlib.util.find_spec("chromadb") is not None


def _km():
    return KnowledgeManager()


# ═══════════════════════════════════════════════════════════════
# 1. YAML 分块富化 (纯单元, 无 chroma)
# ═══════════════════════════════════════════════════════════════

def test_chunk_yaml_flattens_leaf_dict_with_zh_label():
    data = {"fees": {"stamp_duty": {"rate": 0.0005, "side": "sell_only"}}}
    chunks = _chunk_yaml(data, "trading_rules.yaml")
    texts = [c for _, c in chunks]
    # 中文标签 (印花税) 让中文查询命中英文键 (fees.stamp_duty)
    assert any("印花税" in t and "fees.stamp_duty" in t for t in texts)
    assert any("sell_only" in t for t in texts)


def test_chunk_yaml_collapses_scalar_list():
    data = {"trading_hours": {"morning": ["09:30", "11:30"]}}
    chunks = _chunk_yaml(data, "x")
    texts = [c for _, c in chunks]
    assert any("trading_hours.morning" in t and "09:30" in t for t in texts)


def test_chunk_yaml_list_of_dicts_one_per_element():
    data = {"strategies": [{"id": "dual_ma_trend", "name": "双均线趋势跟踪"},
                           {"id": "macd_trend", "name": "MACD趋势"}]}
    chunks = _chunk_yaml(data, "registry.yaml")
    texts = [c for _, c in chunks]
    assert any("dual_ma_trend" in t and "双均线趋势跟踪" in t for t in texts)
    assert any("macd_trend" in t for t in texts)
    assert len(chunks) == 2


# ═══════════════════════════════════════════════════════════════
# 2. 重建流程 + 检索 (chroma 依赖)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CHROMA, reason="chromadb 未安装")
def test_rebuild_indexes_yaml_rules():
    """改 YAML → rebuild → 向量库可检索到规则定义 (此前 YAML 不进向量库)。"""
    km = _km()
    n = km.rebuild_knowledge_index()
    assert n > 0, "重建应索引出 chunk"
    # 规则查询应命中 trading_rules.yaml 源
    got = [r["source"] for r in km.retrieve("印花税税率是多少", top_k=5)]
    assert "trading_rules.yaml" in got, f"规则查询未命中规则源, 实际: {got}"


@pytest.mark.skipif(not HAS_CHROMA, reason="chromadb 未安装")
def test_rebuild_index_has_version_marker():
    km = _km()
    km.rebuild_knowledge_index()
    col = km._ensure_knowledge_base_collection()
    assert (col.metadata or {}).get("index_version") == "3"


# ═══════════════════════════════════════════════════════════════
# 3. recall@k 下界 (chroma 依赖)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CHROMA, reason="chromadb 未安装")
def test_hybrid_recall_at_5():
    """混合检索 (关键词加权 2×) 应对 12 条 gold 查询的 recall@5 不低于 0.8。

    实测 hybrid=1.0 / keyword=1.0 / vector=0.833 (确定性哈希嵌入, 无随机)。
    """
    from scripts.eval_retrieval import GOLD_QUERIES, evaluate
    km = _km()
    km.rebuild_knowledge_index()
    report = evaluate(km, k=5)
    assert report["hybrid"]["recall_at_k"] >= 0.8, report["hybrid"]
    assert report["keyword"]["recall_at_k"] >= 0.8, report["keyword"]
    # 纯哈希向量较弱 (短术语查询), 但不应低于 0.6
    assert report["vector"]["recall_at_k"] >= 0.6, report["vector"]
    assert len(GOLD_QUERIES) >= 10, "gold 集应足够大才有评测意义"


# ═══════════════════════════════════════════════════════════════
# 4. 规则 schema 锁定
# ═══════════════════════════════════════════════════════════════

def test_rules_schema_validates_current_file():
    r = _km().validate_rules_schema()
    assert r["validated_schema"] is True, "jsonschema 应已安装"
    assert r["ok"] is True, f"当前 trading_rules.yaml 应通过 schema: {r['errors']}"


def test_rules_schema_rejects_broken_structure():
    """删除费率子键 / 篡改交收类型 → 应被 schema 拒绝 (结构锁定生效)。"""
    import jsonschema
    schema = json.loads(
        (_km().root / "rules" / "trading_rules.schema.json").read_text(encoding="utf-8")
    )
    rules = _km().get_all_rules("trading_rules")

    bad1 = json.loads(json.dumps(rules))
    del bad1["fees"]["stamp_duty"]["rate"]  # 印花税率缺失
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad1, schema)

    bad2 = json.loads(json.dumps(rules))
    bad2["settlement"]["type"] = "T+0"  # 篡改交收制度
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad2, schema)

    bad3 = json.loads(json.dumps(rules))
    bad3["unknown_top_key"] = 1  # 未锁定顶层键
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad3, schema)
