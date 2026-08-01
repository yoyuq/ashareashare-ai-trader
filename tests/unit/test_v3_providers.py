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


# ═══════════════════════════════════════════════════════════════
# EastMoney 实时行情 + K线解析 (含 v3.0 北交所/假K线修复)
# ═══════════════════════════════════════════════════════════════

class _FakeJsonResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeJsonSession:
    """sync get() → 带 .json() 的响应 (EastMoney 实时行情用)"""
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, timeout=10):
        return _FakeJsonResp(self._payload)


def _em_quote_payload():
    return {
        "data": {"diff": [
            {"f2": 1293.35, "f3": 0.30, "f12": "600519", "f14": "贵州茅台",
             "f15": 1300.0, "f16": 1280.0, "f17": 123456, "f18": 987654321},
            {"f2": 10.5, "f3": -1.2, "f12": "831010", "f14": "北交所股",
             "f15": 10.8, "f16": 10.1, "f17": 5000, "f18": 500000},
        ]}
    }


def test_eastmoney_to_em_code():
    from data.providers.eastmoney_provider import EastMoneyProvider
    assert EastMoneyProvider._to_em_code("sh.600519") == "1.600519"
    assert EastMoneyProvider._to_em_code("sz.000858") == "0.000858"
    assert EastMoneyProvider._to_em_code("bj.831010") == "0.831010"  # v3.0 北交所
    assert EastMoneyProvider._to_em_code("unknown") is None


def test_eastmoney_realtime_parsing_sh_bj():
    from data.providers.eastmoney_provider import EastMoneyProvider
    p = EastMoneyProvider()
    p._session = _FakeJsonSession(_em_quote_payload())
    res = asyncio.run(p.get_realtime_quotes(["sh.600519", "bj.831010"]))
    assert "sh.600519" in res
    q = res["sh.600519"]
    assert q["price"] == 1293.35
    assert q["change_pct"] == 0.30
    assert q["high"] == 1300.0
    assert q["low"] == 1280.0
    assert q["volume"] == 123456
    assert q["amount"] == 987654321
    assert q["name"] == "贵州茅台"
    # v3.0: 北交所通过 0. 前缀映射回 bj.
    assert "bj.831010" in res
    assert res["bj.831010"]["price"] == 10.5


def test_eastmoney_realtime_unknown_code_skipped():
    from data.providers.eastmoney_provider import EastMoneyProvider
    p = EastMoneyProvider()
    p._session = _FakeJsonSession(_em_quote_payload())
    # 传入无法映射的代码 → 不崩溃, 返回空结果
    res = asyncio.run(p.get_realtime_quotes(["hk.00700"]))
    assert res == {}


def test_eastmoney_kline_parsing():
    from datetime import date
    from data.providers.base import DataFrequency, DataRequest
    from data.providers.eastmoney_provider import EastMoneyProvider

    p = EastMoneyProvider()
    klines = ["2026-01-02,10.0,10.5,11.0,9.5,1000,5000,1.5"]

    async def fake_retry(url, timeout=15):
        return _FakeJsonResp({"data": {"klines": klines}})

    p._request_with_retry = fake_retry
    req = DataRequest("sh.600519", date(2026, 1, 1), date(2026, 1, 31),
                      DataFrequency.DAILY, adjust="qfq")
    res = asyncio.run(p.get_daily_kline(req))
    df = res.data
    assert len(df) == 1
    assert df.iloc[0]["open"] == 10.0
    assert df.iloc[0]["close"] == 10.5
    assert df.iloc[0]["high"] == 11.0
    assert df.iloc[0]["low"] == 9.5
    # v3.0: 成交量统一为"股" (EastMoney 返回"手" ×100)
    assert df.iloc[0]["volume"] == 1000 * 100
    assert df.iloc[0]["amount"] == 5000
    # 清洗管线: 停牌标注列存在
    assert "is_trade" in df.columns
    assert df.iloc[0]["is_trade"] == 1


def test_eastmoney_kline_no_data_historical_returns_empty():
    """v3.0: 历史回测请求无数据 → 返回空, 不再构造"今天"假K线污染回测"""
    from datetime import date
    from data.providers.base import DataFrequency, DataRequest
    from data.providers.eastmoney_provider import EastMoneyProvider

    p = EastMoneyProvider()

    async def fake_retry(url, timeout=15):
        return _FakeJsonResp({"data": None})

    p._request_with_retry = fake_retry
    req = DataRequest("sh.600519", date(2010, 1, 1), date(2010, 1, 31),
                      DataFrequency.DAILY)
    res = asyncio.run(p.get_daily_kline(req))
    assert res.data.empty
    assert res.metadata.get("error") == "no_data"


def test_eastmoney_standardize_columns():
    from data.providers.eastmoney_provider import EastMoneyProvider
    df = pd.DataFrame({"日期": ["2026-01-02"], "开盘": [1.0], "收盘": [2.0]})
    out = EastMoneyProvider()._standardize_columns(df)
    assert list(out.columns) == ["date", "open", "close"]


# ═══════════════════════════════════════════════════════════════
# AKShare 归一化 + K线解析 (v3.0: adjust 空串=不复权, 列名标准化)
# ═══════════════════════════════════════════════════════════════

def test_akshare_normalize_symbol():
    from data.providers.akshare_provider import AKShareProvider
    p = AKShareProvider()
    assert p._normalize_symbol("sh.600519") == "600519"
    assert p._normalize_symbol("sz.000858") == "000858"
    assert p._normalize_symbol("bj.831010") == "831010"
    assert p._normalize_symbol("600519") == "600519"


def test_akshare_standardize_columns():
    from data.providers.akshare_provider import AKShareProvider
    df = pd.DataFrame({
        "日期": ["2026-01-02"], "开盘": [1.0], "收盘": [2.0],
        "最高": [3.0], "最低": [0.5], "成交量": [100], "成交额": [200],
        "涨跌幅": [5.0], "换手率": [1.2],
    })
    out = AKShareProvider()._standardize_columns(df)
    for col in ["date", "open", "close", "high", "low",
                "volume", "amount", "pct_change", "turnover"]:
        assert col in out.columns


def test_akshare_kline_parsing(monkeypatch):
    from datetime import date
    import akshare as ak
    from data.providers.akshare_provider import AKShareProvider
    from data.providers.base import DataFrequency, DataRequest

    p = AKShareProvider()
    raw = pd.DataFrame({
        "日期": ["2026-01-02"], "开盘": [10.0], "收盘": [10.5],
        "最高": [11.0], "最低": [9.5], "成交量": [1000], "成交额": [5000], "换手率": [1.5],
    })
    calls = {}

    def fake_hist(symbol, period, start_date, end_date, adjust):
        calls["adjust"] = adjust
        return raw

    monkeypatch.setattr(ak, "stock_zh_a_hist", fake_hist)
    req = DataRequest("sh.600519", date(2026, 1, 1), date(2026, 1, 31),
                      DataFrequency.DAILY, adjust="")
    res = asyncio.run(p.get_daily_kline(req))
    # v3.0: adjust 空串=不复权, 不再被强制成 qfq
    assert calls["adjust"] == ""
    df = res.data
    assert len(df) == 1
    assert df.iloc[0]["close"] == 10.5
    assert "date" in df.columns
    assert df.iloc[0]["is_trade"] == 1


def test_akshare_stock_list_rename_and_cache(monkeypatch):
    import akshare as ak
    from data.providers.akshare_provider import AKShareProvider

    p = AKShareProvider()
    raw = pd.DataFrame({
        "代码": ["600519"], "名称": ["贵州茅台"], "最新价": [1293.35], "涨跌幅": [0.3],
        "总市值": [1e12], "流通市值": [9e11], "市盈率-动态": [30.0], "市净率": [8.0],
        "60日涨跌幅": [5.0], "年初至今涨跌幅": [10.0],
    })
    monkeypatch.setattr(ak, "stock_zh_a_spot_em", lambda: raw)
    df = asyncio.run(p.get_stock_list())
    assert list(df.columns) == ["symbol", "name", "price", "pct_change", "total_mv",
                                "float_mv", "pe_ttm", "pb", "pct_60d", "pct_ytd"]
    assert df.iloc[0]["symbol"] == "600519"
    assert df.iloc[0]["name"] == "贵州茅台"
    assert df.iloc[0]["price"] == 1293.35
    # 缓存: 24h 内第二次调用返回同一对象, 不再重新请求
    df2 = asyncio.run(p.get_stock_list())
    assert df2 is df
