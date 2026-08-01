"""v3.0 providers 层测试 — 补 tencent/eastmoney/fundamentals 解析逻辑 (覆盖 13-30%)

网络层用 mock 隔离, 只测解析/归一化/财务计算逻辑。
"""

import asyncio

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# Tencent 实时行情解析 (含 v3.0 北交所 bj 修复)
# ═══════════════════════════════════════════════════════════════

def _tencent_line(market, name, code, price, yest, open_px, vol, high, low):
    """构造腾讯行情行 (35+ 字段)"""
    fields = [market, name, code, price, yest, open_px, vol]
    fields += ["0"] * 26  # 6..32 补齐
    fields.append(high)   # 33
    fields.append(low)    # 34
    return f'v_{market}{code}="{"~".join(fields)}";'


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeSession:
    def __init__(self, text):
        self._text = text

    def get(self, url, timeout=10):  # sync: asyncio.to_thread 期望同步调用
        return _FakeResp(self._text)


def _make_provider(text):
    from data.providers.tencent_provider import TencentFinanceProvider
    p = TencentFinanceProvider()
    p._session = _FakeSession(text)
    return p


def test_tencent_quote_parsing_sh():
    text = _tencent_line("1", "贵州茅台", "600519", "1293.35", "1289.50",
                         "1295.00", "123456", "1300.00", "1280.00")
    p = _make_provider(text)
    res = asyncio.run(p.get_realtime_quotes(["sh.600519"]))
    assert "sh.600519" in res
    q = res["sh.600519"]
    assert q["price"] == 1293.35
    assert q["high"] == 1300.0
    assert q["low"] == 1280.0
    assert q["name"] == "贵州茅台"
    assert q["change_pct"] == pytest.approx(0.30, abs=1e-2)  # 解析代码四舍五入到 2 位


def test_tencent_quote_parsing_bj():
    """v3.0: 北交所 8/4 开头代码应还原为 bj. 前缀"""
    text = _tencent_line("1", "北交所股", "831010", "10.00", "9.90",
                         "10.00", "5000", "10.10", "9.80")
    p = _make_provider(text)
    res = asyncio.run(p.get_realtime_quotes(["bj.831010"]))
    assert "bj.831010" in res, f"北交所代码应还原为 bj., 实际 keys={list(res.keys())}"
    assert res["bj.831010"]["price"] == 10.0


# ═══════════════════════════════════════════════════════════════
# fundamentals._yoy_growth (同比/环比)
# ═══════════════════════════════════════════════════════════════

def test_yoy_growth_tongbi():
    """同比: 最新期 vs 一年前同期"""
    from data.providers.fundamentals import FundamentalsProvider
    df = pd.DataFrame({"报告期": ["2025-06-30", "2024-06-30"], "净利润": [200.0, 100.0]})
    g = FundamentalsProvider._yoy_growth(df, "净利润")
    assert g == pytest.approx(100.0, rel=1e-3)


def test_yoy_growth_huanbi_fallback():
    """缺一年前同期 → 退回环比 (上一期)"""
    from data.providers.fundamentals import FundamentalsProvider
    df = pd.DataFrame({"报告期": ["2025-06-30", "2025-03-31"], "净利润": [200.0, 100.0]})
    g = FundamentalsProvider._yoy_growth(df, "净利润")
    assert g == pytest.approx(100.0, rel=1e-3)


def test_yoy_growth_insufficient_data():
    from data.providers.fundamentals import FundamentalsProvider
    g = FundamentalsProvider._yoy_growth(pd.DataFrame({"净利润": [100.0]}), "净利润")
    assert g == 0.0
