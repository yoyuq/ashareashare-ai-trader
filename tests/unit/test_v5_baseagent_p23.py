"""v5.6 P2-3 BaseAgent `success` 语义 + `model_used` 填充 + 并发竞态修复

覆盖:
  - `model_used` 从 outputs 弹出 → `AgentResult.model_used` (不再恒空/泄漏进 data)
  - 捕获异常仍 `success=True` → 有 error 时强制 `success=False`
  - 并发 `_start_context`/`_finish_context` 用 contextvar, 各自上下文不串扰
"""
import asyncio

from agent.sub_agents.base import AgentContext, BaseAgent


class _DummyAgent(BaseAgent):
    """最小具体子类, 只暴露 _start_context/_finish_context 语义"""
    agent_name = "dummy_p23"

    async def run(self, ctx: AgentContext):
        return self._finish_context(success=True)


# ═══════════════════════════════════════════════════════════════
# 1. model_used 填充
# ═══════════════════════════════════════════════════════════════

def test_model_used_popped_into_result_field():
    agent = _DummyAgent()
    agent._start_context(task_id="t1")
    res = agent._finish_context(success=True, narrative="n", model_used="deepseek-v4-flash")
    assert res.model_used == "deepseek-v4-flash"
    assert "model_used" not in res.data          # 不再泄漏进 data
    assert res.data["narrative"] == "n"          # 正常字段保留


def test_model_used_empty_when_absent():
    agent = _DummyAgent()
    agent._start_context(task_id="t2")
    res = agent._finish_context(success=True, narrative="n")
    assert res.model_used == ""


# ═══════════════════════════════════════════════════════════════
# 2. success 语义修正
# ═══════════════════════════════════════════════════════════════

def test_error_forces_success_false():
    agent = _DummyAgent()
    agent._start_context(task_id="t3")
    # 子类 except 分支: result_data["error"]="..." 再 _finish_context(success=True, **result_data)
    res = agent._finish_context(success=True, narrative="fallback", error="LLM timeout")
    assert res.success is False
    assert res.error == "LLM timeout"
    assert res.data["narrative"] == "fallback"   # 回退内容仍保留


def test_no_error_keeps_success_true():
    agent = _DummyAgent()
    agent._start_context(task_id="t4")
    res = agent._finish_context(success=True, narrative="ok")
    assert res.success is True
    assert res.error == ""


# ═══════════════════════════════════════════════════════════════
# 3. 并发竞态 (contextvar 隔离)
# ═══════════════════════════════════════════════════════════════

def test_concurrent_contexts_do_not_cross_contaminate():
    agent = _DummyAgent()

    async def _run(task_id: str, err: str = ""):
        ctx = agent._start_context(task_id=task_id)
        await asyncio.sleep(0.01)                  # 强制交错
        agent._finish_context(success=True, error=err) if err else agent._finish_context(success=True)
        return ctx

    # 两个协程在同一实例上并发跑; 旧实现里 self._context 会被后者覆盖,
    # 导致 A 的 error 落到 B 的 context 上 (ctxB.errors 被污染)。
    async def _main():
        return await asyncio.gather(
            _run("A", err="boomA"),
            _run("B"),
        )

    ctx_a, ctx_b = asyncio.run(_main())

    # A 的 error 必须落在 A 自己的 context 上
    assert ctx_a.task_id == "A"
    assert ctx_a.errors == ["boomA"]
    # B 无 error, context 不能被 A 的 error 污染
    assert ctx_b.task_id == "B"
    assert ctx_b.errors == []
