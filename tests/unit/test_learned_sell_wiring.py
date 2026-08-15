"""落点 B 指向性单测 — daily_runner._apply_learned_sell (低估值规则的高估卖出接线)。

验证 (方案3b 实盘/纸盘链路的确定性卖出信号):
- gate=0 → 零开销返回 [] (不碰 engine/data_router, 现状不变)。
- gate=1 持仓 pe_ttm > 22.5 → 触发 `engine.execute_sell(symbol, exit_reason="估值高估卖出", ...)`。
- gate=1 持仓 pe_ttm <= 22.5 / 无数据 / 缺 pe 列 / NaN / 亏损股(pe<=0) → 跳过不卖。
- 列名防御: Baostock 返回 `peTTM`(驼峰), 降级源(Tencent/EastMoney/AKShare)返回 `pe_ttm`(下划线),
  两种都要能正确读出卖信号, 不能因列名差异静默跳过。
- 回归: 曾误用 `engine.account.positions` (engine 只有 `.state` 属性, 会 AttributeError), 本测用
  `.state.positions` 真实路径钉死。

全部用 fake engine/fake async data_router, 不联网、不调 LLM、不碰真实 ChromaDB。
"""
from __future__ import annotations

import asyncio

import pandas as pd

from simulation.daily_runner import _apply_learned_sell


# ── fake KnowledgeManager (monkeypatch, 复用 test_knowledge_apply 的同款数据) ──

class _FakeKM:
    def __init__(self, path):
        pass

    def list_learned(self, categories=None, verdicts=None):
        # 模拟已按 categories=["rule"] + verdicts=["verified"] 过滤后的结果
        return [
            {"concept": "估值指标筛选", "doc": "估值指标筛选\n低PE低PB选股",
             "meta": {"concept": "估值指标筛选", "category": "rule", "verdict": "verified",
                      "template": "low_pe_value",
                      "params": '{"max_pe": 0.5, "max_pb": 1.0}',  # 幻觉参数, 夹回 max_pe=15.0
                      "confidence": 0.8}},
            {"concept": "均线系统", "doc": "均线系统\n金叉买入",
             "meta": {"concept": "均线系统", "category": "rule", "verdict": "verified",
                      "template": "ma_cross",
                      "params": '{"fast": 5, "slow": 20}',
                      "confidence": 0.85}},
        ]


def _patch_km(monkeypatch):
    monkeypatch.setattr("knowledge.manager.KnowledgeManager", _FakeKM)


# ── fake engine / data_router ──

class _FakePosition:
    def __init__(self, name):
        self.name = name
        self.quantity = 100


class _FakeState:
    def __init__(self, positions):
        self.positions = positions


class _FakeTrade:
    def __init__(self, symbol):
        self.symbol = symbol
        self.price = 12.34


class _FakeEngine:
    def __init__(self, positions):
        self.state = _FakeState(positions)
        self.sell_calls = []

    def execute_sell(self, **kwargs):
        self.sell_calls.append(kwargs)
        return _FakeTrade(kwargs.get("symbol"))


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeRouter:
    """df_by_sym: sym -> 单票日线 DataFrame (或 None 表示无数据)。直接构造列名, 测列名防御。"""

    def __init__(self, df_by_sym):
        self.df_by_sym = df_by_sym

    async def get_daily_kline(self, req):
        return _FakeResult(self.df_by_sym.get(req.symbol))


def _run(engine, router, mtm_pct, gate):
    return asyncio.run(_apply_learned_sell(engine, router, mtm_pct, gate=gate))


def _pe_ttm(pe):
    """Baostock 原始列名 (驼峰)。"""
    return pd.DataFrame({"peTTM": [pe]})


# ── 落点 B 接线 ──

def test_gate0_no_sell_zero_overhead(monkeypatch):
    _patch_km(monkeypatch)
    engine = _FakeEngine({"sh.600519": _FakePosition("贵州茅台")})
    router = _FakeRouter({"sh.600519": _pe_ttm(30.0)})  # 即便高估, gate=0 也不卖
    sold = _run(engine, router, {}, gate=0)
    assert sold == []
    assert engine.sell_calls == []


def test_gate1_sells_overvalued_with_reason(monkeypatch):
    _patch_km(monkeypatch)
    engine = _FakeEngine({"sh.600519": _FakePosition("贵州茅台")})
    router = _FakeRouter({"sh.600519": _pe_ttm(30.0)})  # pe 30 > 22.5 → 卖
    sold = _run(engine, router, {"sh.600519": 5.0}, gate=1)

    assert [s["symbol"] for s in sold] == ["sh.600519"]
    assert sold[0]["name"] == "贵州茅台"
    assert sold[0]["price"] == 12.34

    assert len(engine.sell_calls) == 1
    call = engine.sell_calls[0]
    assert call["symbol"] == "sh.600519"
    assert call["exit_reason"] == "估值高估卖出"
    assert call["pct_change"] == 5.0


def test_gate1_skips_low_pe_missing_and_nan(monkeypatch):
    _patch_km(monkeypatch)
    positions = {
        "sh.600519": _FakePosition("贵州茅台"),   # pe 30 > 22.5 → 卖
        "sh.600036": _FakePosition("招商银行"),   # pe 10 ≤ 22.5 → 不卖
        "sz.000858": _FakePosition("五粮液"),     # 无数据 → 跳过
        "sh.601398": _FakePosition("工商银行"),   # 缺 pe 列 → 跳过
        "sz.300750": _FakePosition("宁德时代"),   # pe=NaN → 跳过
    }
    router = _FakeRouter({
        "sh.600519": _pe_ttm(30.0),
        "sh.600036": _pe_ttm(10.0),
        "sz.000858": None,
        "sh.601398": pd.DataFrame({"close": [1.0]}),
        "sz.300750": _pe_ttm(float("nan")),
    })
    engine = _FakeEngine(positions)
    sold = _run(engine, router, {}, gate=1)

    assert [s["symbol"] for s in sold] == ["sh.600519"]
    assert len(engine.sell_calls) == 1
    assert engine.sell_calls[0]["symbol"] == "sh.600519"


def test_gate1_skips_nonpositive_pe(monkeypatch):
    _patch_km(monkeypatch)
    engine = _FakeEngine({"sh.600000": _FakePosition("浦发银行")})
    router = _FakeRouter({"sh.600000": _pe_ttm(-5.0)})  # 亏损股 pe<=0 → 跳过
    sold = _run(engine, router, {}, gate=1)
    assert sold == []
    assert engine.sell_calls == []


def test_gate1_sells_when_provider_uses_pe_ttm_column(monkeypatch):
    """降级源 (Tencent/EastMoney/AKShare) 返回 pe_ttm 下划线列, 卖出信号仍须生效。"""
    _patch_km(monkeypatch)
    engine = _FakeEngine({"sh.600519": _FakePosition("贵州茅台")})
    router = _FakeRouter({"sh.600519": pd.DataFrame({"pe_ttm": [30.0]})})
    sold = _run(engine, router, {}, gate=1)
    assert [s["symbol"] for s in sold] == ["sh.600519"]
    assert engine.sell_calls[0]["exit_reason"] == "估值高估卖出"
