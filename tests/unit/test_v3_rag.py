"""v3.1-deerflow RAG 检索测试 — 中文降级修复 + 向量检索激活验证

背景: 项目实现了 ChromaDB 双通道 RAG (语义 rag_query + K线形态相似), 但
  1) 旧的降级关键词匹配用空格分词, 中文查询永远返回空
  2) chromadb 是声明依赖但常未安装, 导致实际不生效
修复: 中文 n-gram tokenizer (_cn_tokens) + K线中文形态匹配 + 激活向量检索。
"""

import pytest


def _make_km():
    from knowledge.manager import KnowledgeManager
    return KnowledgeManager()


# ═══════════════════════════════════════════════════════════════
# 1. _cn_tokens 中英文 tokenizer
# ═══════════════════════════════════════════════════════════════

def test_cn_tokens_extracts_ngrams():
    from knowledge.manager import _cn_tokens
    toks = _cn_tokens("震荡市适合均线策略")
    assert "均线" in toks
    assert "震荡" in toks
    assert "策略" in toks


def test_cn_tokens_latin_and_mixed():
    from knowledge.manager import _cn_tokens
    toks = _cn_tokens("RSI超买 hammer 金叉")
    assert "rsi" in toks          # 拉丁词转小写
    assert "hammer" in toks
    assert "超买" in toks
    assert "金叉" in toks


def test_cn_tokens_empty_and_short():
    from knowledge.manager import _cn_tokens
    assert _cn_tokens("") == []
    assert _cn_tokens("a") == ["a"]  # 单拉丁词仍返回
    assert _cn_tokens("中") == []    # 单字中文不产生 n-gram


# ═══════════════════════════════════════════════════════════════
# 2. K线形态中文关键词匹配 (_keyword_match_klines)
# ═══════════════════════════════════════════════════════════════

def test_kline_keyword_match_chinese():
    """中文查询应命中对应形态 (旧实现返回空)"""
    km = _make_km()
    results = km._keyword_match_klines("看涨吞没 红三兵", top_k=5)
    patterns = [r["pattern"] for r in results]
    assert "bullish_engulfing" in patterns
    assert "three_soldiers" in patterns


def test_kline_keyword_match_english():
    km = _make_km()
    results = km._keyword_match_klines("hammer", top_k=3)
    assert any(r["pattern"] == "hammer" for r in results)


def test_kline_keyword_match_returns_sorted():
    km = _make_km()
    results = km._keyword_match_klines("晨星 十字星", top_k=5)
    assert results, "应有匹配结果"
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


# ═══════════════════════════════════════════════════════════════
# 3. rag_query 中文降级 (无 chroma 时)
# ═══════════════════════════════════════════════════════════════

def test_rag_query_chinese_fallback(monkeypatch):
    """chroma 不可用 → 中文关键词降级应返回非空 (旧实现空格分词返回空)"""
    from knowledge.manager import KnowledgeManager
    monkeypatch.setattr(KnowledgeManager, "chroma_available", property(lambda self: False))
    km = _make_km()
    txt = km.rag_query("震荡市适合什么均线策略", top_k=3)
    assert txt, "中文降级检索不应返回空"


def test_rag_query_garbage_returns_empty(monkeypatch):
    """无匹配词 → 返回空 (不崩溃)"""
    from knowledge.manager import KnowledgeManager
    monkeypatch.setattr(KnowledgeManager, "chroma_available", property(lambda self: False))
    km = _make_km()
    txt = km.rag_query("zzzzqqqqxxyyzz", top_k=3)
    assert txt == "" or "相关知识库参考" in txt


# ═══════════════════════════════════════════════════════════════
# 4. 向量检索激活 (装 chromadb 后应真正工作)
# ═══════════════════════════════════════════════════════════════

def test_chroma_vector_search_when_available():
    """若 chromadb 已安装, 两条向量检索路径应返回真实结果 (自动播种)"""
    import importlib.util
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb 未安装, 跳过向量检索激活测试")
    km = _make_km()
    assert km.chroma_available is True, "chroma 已安装应可用"

    # 语义 RAG
    txt = km.rag_query("均线策略", top_k=3)
    assert txt, "向量语义检索应返回内容"

    # K线形态向量相似
    results = km.search_similar_klines(
        "RSI=62 趋势=0.5 评分=70 金叉 放量", top_k=3
    )
    assert results, "K线形态向量检索应返回结果"
    assert all("pattern" in r for r in results)
