"""v5.6 假多元化 — 独立对抗调用 + 闸门 A/B 单测.

覆盖: adversarial_risk 独立二次调用解析/裁剪/失败回退; 无 client 走 router(实盘记账);
      _apply_adversarial_gate 在独立对抗分歧>=2级时保守收敛.
"""

import pytest


# ═══════════════════════════════════════════════════════════════
# 独立对抗调用 adversarial_risk
# ═══════════════════════════════════════════════════════════════

from agent.evolution.adversarial import adversarial_risk


class _R:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": type(
            "Comp", (), {"create": self._create})()})()
        self._content = content

    async def _create(self, **kwargs):
        return _R(self._content)


@pytest.mark.asyncio
async def test_adv_risk_parses_int():
    cli = _FakeClient('{"adversarial_risk_level": 4, "adversarial_reason": "背离"}')
    got = await adversarial_risk(cli, "快照", "利弗莫尔", "trend_up")
    assert got == 4


@pytest.mark.asyncio
async def test_adv_risk_extracts_from_fenced_json():
    cli = _FakeClient('```json\n{"adversarial_risk_level": 2}\n```')
    got = await adversarial_risk(cli, "快照", "巴菲特", "panic_bottom")
    assert got == 2


@pytest.mark.asyncio
async def test_adv_risk_extracts_substring():
    # 模型偶发夹带文字, 截取首个 { } 块
    cli = _FakeClient('结论: {"adversarial_risk_level": 5} 完')
    got = await adversarial_risk(cli, "快照", "缠中说禅", "bubble_late")
    assert got == 5


@pytest.mark.asyncio
async def test_adv_risk_clamps_bounds():
    cli = _FakeClient('{"adversarial_risk_level": 99}')
    assert await adversarial_risk(cli, "s", "d", "r") == 5
    cli2 = _FakeClient('{"adversarial_risk_level": -3}')
    assert await adversarial_risk(cli2, "s", "d", "r") == 1


@pytest.mark.asyncio
async def test_adv_risk_none_client_walks_router(monkeypatch):
    """无 client (实盘路径) → 走 ModelRouter, 不再直接返回 None (v5.7 统一记账)."""
    import models.router as mr

    class _FakeRouteResult:
        def __init__(self, response):
            self.response = response

    class _FakeRouter:
        async def route(self, **kwargs):
            return _FakeRouteResult('{"adversarial_risk_level": 3, "adversarial_reason": "中性"}')

    monkeypatch.setattr(mr, "get_shared_router", lambda: _FakeRouter())
    assert await adversarial_risk(None, "s", "d", "r") == 3


@pytest.mark.asyncio
async def test_adv_risk_none_client_router_failure_returns_none(monkeypatch):
    """实盘路径 router 抛错 → 降级 None (失败不拦截)."""
    import models.router as mr

    class _BoomRouter:
        async def route(self, **kwargs):
            raise RuntimeError("no api key")

    monkeypatch.setattr(mr, "get_shared_router", lambda: _BoomRouter())
    assert await adversarial_risk(None, "s", "d", "r") is None


@pytest.mark.asyncio
async def test_adv_risk_missing_key_returns_none():
    cli = _FakeClient('{"foo": 1}')
    assert await adversarial_risk(cli, "s", "d", "r") is None


@pytest.mark.asyncio
async def test_adv_risk_non_json_returns_none():
    cli = _FakeClient("我无法判断")
    assert await adversarial_risk(cli, "s", "d", "r") is None


@pytest.mark.asyncio
async def test_adv_risk_exception_returns_none():
    class Boom:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": type(
                "Comp", (), {"create": self._boom})()})()

        async def _boom(self, **kwargs):
            raise RuntimeError("timeout")

    assert await adversarial_risk(Boom(), "s", "d", "r") is None


# ═══════════════════════════════════════════════════════════════
# 多模型对抗 — 独立模型开关 (v5.8 预注册 A/B 脚手架)
# ═══════════════════════════════════════════════════════════════

import agent.evolution.adversarial as adv


def test_adv_llm_cfg_unset_returns_none(monkeypatch):
    monkeypatch.delenv("ADVERSARIAL_LLM_MODEL", raising=False)
    assert adv._adversarial_llm_cfg() is None


def test_adv_llm_cfg_reads_model_with_defaults(monkeypatch):
    monkeypatch.setenv("ADVERSARIAL_LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    key, base, model = adv._adversarial_llm_cfg()
    assert model == "deepseek-v4-pro"
    assert key == "sk-test"
    assert base == "https://api.deepseek.com/v1"


@pytest.mark.asyncio
async def test_adv_multimodel_uses_dedicated_model(monkeypatch):
    """设 ADVERSARIAL_LLM_MODEL → 对抗票走独立模型, 优先于传入 client (主导)."""
    monkeypatch.setenv("ADVERSARIAL_LLM_MODEL", "glm-4-flash")
    seen = {}

    class _FakeChat:
        def __init__(self):
            self.completions = self

        async def create(self, **kw):
            seen["model"] = kw.get("model")
            return _R('{"adversarial_risk_level": 2}')

    class _FakeAdvClient:
        def __init__(self, **kw):
            seen["api_key"] = kw.get("api_key")
            seen["base_url"] = kw.get("base_url")
            self.chat = _FakeChat()

    monkeypatch.setattr(adv, "AsyncOpenAI", _FakeAdvClient)
    # 传入主导 client (会被忽略): 独立模型返回 2, 而非主导 client 的 5
    got = await adversarial_risk(
        _FakeClient('{"adversarial_risk_level": 5}'), "s", "d", "r")
    assert got == 2
    assert seen["model"] == "glm-4-flash"
    assert seen["base_url"] == "https://api.deepseek.com/v1"


@pytest.mark.asyncio
async def test_adv_multimodel_failure_returns_none(monkeypatch):
    """独立模型调用抛错 → 降级 None (失败不拦截), 不落到主导 client."""
    monkeypatch.setenv("ADVERSARIAL_LLM_MODEL", "deepseek-v4-pro")

    class _Boom:
        def __init__(self, **kw):
            self.chat = type("Chat", (), {"completions": type(
                "Comp", (), {"create": self._boom})()})()

        async def _boom(self, **kw):
            raise RuntimeError("dedicated model down")

    monkeypatch.setattr(adv, "AsyncOpenAI", _Boom)
    assert await adversarial_risk(
        _FakeClient('{"adversarial_risk_level": 5}'), "s", "d", "r") is None


# ═══════════════════════════════════════════════════════════════
# 闸门 _apply_adversarial_gate — 独立对抗分歧裁决
# ═══════════════════════════════════════════════════════════════

from simulation.daily_runner import _apply_adversarial_gate


def test_adv_gate_independent_conservative_converges():
    """独立对抗更保守(5 vs 3, 分歧2) → 风险+1级, 仓位向1.0收敛."""
    out = _apply_adversarial_gate({
        "risk_level": 3, "position_multiplier": 1.3,
        "adversarial_risk_level": 5,
    })
    assert out["adversarial_applied"] == "conservative"
    assert out["risk_level"] == 4            # 3→4
    assert out["position_multiplier"] == pytest.approx(1.15, abs=1e-6)  # 向1.0收敛一半
    assert out["adversarial_divergence"] == 2


def test_adv_gate_independent_noop_when_agrees():
    """分歧<2 → 不干预 (off 模式 adversarial=主导 等效此路径)."""
    out = _apply_adversarial_gate({
        "risk_level": 3, "position_multiplier": 1.2,
        "adversarial_risk_level": 3,
    })
    assert out["risk_level"] == 3
    assert out["position_multiplier"] == 1.2


def test_adv_gate_dampen_when_adv_aggressive():
    """对抗更激进(1 vs 3, 分歧2) → 不追激进, 只降置信(仓位向1.0收敛)."""
    out = _apply_adversarial_gate({
        "risk_level": 3, "position_multiplier": 0.7,
        "adversarial_risk_level": 1,
    })
    assert out["adversarial_applied"] == "dampen"
    assert out["risk_level"] == 3            # 不追激进, 风险不变
    assert out["position_multiplier"] == pytest.approx(0.85, abs=1e-6)  # 0.7→向1.0


def test_adv_gate_bad_values_noop():
    """非法 adversarial → 原样返回, 不崩."""
    out = _apply_adversarial_gate({"risk_level": 3, "position_multiplier": 1.0,
                                   "adversarial_risk_level": "abc"})
    assert out["risk_level"] == 3