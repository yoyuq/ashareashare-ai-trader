"""
DeepSeek API 连通性测试 + ModelRouter 端到端验证

注意: 联网测试标记为 network, 默认测试套件跳过 (pytest -m "not network"),
仅在显式 `pytest -m network` 或 `python tests/test_deepseek_api.py` 时运行,
避免每次全量测试都真实调用付费 API。
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY", "")


def _require_api_key():
    if not api_key:
        pytest.skip("未设置 DEEPSEEK_API_KEY, 跳过联网测试")


@pytest.mark.network
async def test_direct_api():
    """测试直接调用 DeepSeek API"""
    _require_api_key()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )

    print("\n--- 测试 1: DeepSeek V4-Flash ---")
    try:
        response = await client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "user", "content": "用一句话介绍A股市场的T+1制度"},
            ],
            max_tokens=300,  # V4 思考模式需留足 token, 否则 content 为空
        )
        _msg = response.choices[0].message
        content = _msg.content or getattr(_msg, "reasoning_content", "") or ""
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        cost = (tokens_in / 1_000_000) * 1.0 + (tokens_out / 1_000_000) * 2.0

        assert content, "DeepSeek 返回空内容"
        print(f"  ✅ 响应: {content[:100]}...")
        print(f"  Tokens: {tokens_in} in + {tokens_out} out")
        print(f"  💰 成本: ¥{cost:.6f}")
    except Exception as e:
        pytest.fail(f"DeepSeek 调用失败: {e}")

    print("\n--- 测试 2: DeepSeek V4-Flash (综合问答) ---")
    try:
        response = await client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "user", "content": "MACD金叉和死叉分别代表什么？用一句话回答"},
            ],
            max_tokens=300,  # V4 思考模式需留足 token, 否则 content 为空
        )
        _msg = response.choices[0].message
        content = _msg.content or getattr(_msg, "reasoning_content", "") or ""
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        cost = (tokens_in / 1_000_000) * 1.0 + (tokens_out / 1_000_000) * 2.0

        assert content, "DeepSeek 返回空内容"
        print(f"  ✅ 响应: {content[:100]}...")
        print(f"  Tokens: {tokens_in} in + {tokens_out} out")
        print(f"  💰 成本: ¥{cost:.6f}")
    except Exception as e:
        pytest.fail(f"DeepSeek 调用失败: {e}")

    return True


@pytest.mark.network
async def test_model_router():
    """测试 ModelRouter 三层路由"""
    _require_api_key()
    print("\n--- 测试 3: ModelRouter 完整路由 ---")
    from models.router import ModelRouter

    router = ModelRouter(daily_budget=1.0)
    print(f"  日预算: ¥{router.daily_budget}, 当前花费: ¥{router.daily_cost:.4f}")

    # 测试简单任务 → 应路由到 LOCAL, 但Ollama不可用时会降级到Flash
    print("\n  📝 简单任务: kline_describe")
    try:
        result = await router.route(
            messages=[{"role": "user", "content": "用一句话描述什么是十字星K线"}],
            task_type="kline_describe",
        )
        assert result.response, "ModelRouter 返回空响应"
        print(f"  ✅ 路由: {result.tier.value} → {result.model_name}")
        print(f"  📄 响应: {result.response[:120]}...")
        print(f"  💰 成本: ¥{result.cost:.6f} | 延迟: {result.latency_ms:.0f}ms")
    except Exception as e:
        pytest.fail(f"ModelRouter 路由失败: {e}")

    # 测试中等任务 → Flash
    print("\n  📊 中等任务: technical_analysis")
    try:
        result = await router.route(
            messages=[{"role": "user", "content": "当前RSI=58.3, MACD刚金叉, 请分析技术面含义"}],
            task_type="technical_analysis",
        )
        assert result.response, "ModelRouter 返回空响应"
        print(f"  ✅ 路由: {result.tier.value} → {result.model_name}")
        print(f"  📄 响应: {result.response[:120]}...")
        print(f"  💰 成本: ¥{result.cost:.6f} | 延迟: {result.latency_ms:.0f}ms")
    except Exception as e:
        pytest.fail(f"ModelRouter 路由失败: {e}")

    # 成本汇总
    print(f"\n  📊 日成本汇总: {router.cost_summary()}")

    return True


async def test_cost_monitor():
    """测试 CostMonitor"""
    print("\n--- 测试 4: CostMonitor ---")
    from models.cost_monitor import CostMonitor

    monitor = CostMonitor(daily_budget=1.0, monthly_budget=15.0)

    # 记录几次调用
    monitor.record_call("flash", input_tokens=50000, output_tokens=20000)
    monitor.record_call("flash", input_tokens=30000, output_tokens=15000)
    monitor.record_call("pro", input_tokens=10000, output_tokens=5000)

    report = monitor.daily_report()
    assert report["total_calls_today"] == 3, f"调用计数错误: {report.get('total_calls_today')}"
    assert report["daily_cost"] > 0, "日成本应大于0"
    print(f"  日花费: ¥{report['daily_cost']:.4f} / ¥{report['daily_budget']}")
    print(f"  月花费: ¥{report['monthly_cost']:.4f} / ¥{report['monthly_budget']}")
    print(f"  今日调用: {report['total_calls_today']} 次")
    for tier, stats in report.get("by_tier", {}).items():
        print(f"    {tier}: {stats['count']}次, ¥{stats['cost']:.4f}, {stats['tokens']} tokens")


@pytest.mark.network
async def test_knowledge_with_llm():
    """测试 KnowledgeManager + LLM 组合"""
    _require_api_key()
    print("\n--- 测试 5: KnowledgeManager + LLM 综合分析 ---")
    from knowledge.manager import KnowledgeManager
    from models.router import ModelRouter

    km = KnowledgeManager()
    router = ModelRouter(daily_budget=1.0)

    # 获取带规则注入的Prompt
    prompt = km.get_system_prompt("technical_analyst")
    assert len(prompt) > 0, "system prompt 为空"
    print(f"  Prompt长度: {len(prompt)} chars (含注入的指标指南)")

    # 硬化口径校验
    defn = km.get_hardened_definition("volume_expansion")
    print(f"  放量定义: {defn['formula']} (阈值={defn['threshold']})")

    is_volume_expanding = km.verify_definition("volume_expansion", 2.0)
    print(f"  量比2.0 → 放量? {is_volume_expanding}")

    # 用LLM做分析
    print("\n  🤖 LLM分析 (用知识库Prompt)...")
    try:
        result = await router.route(
            messages=[
                {"role": "system", "content": prompt[:3000]},  # 前3000字符
                {"role": "user", "content": "sh.600519 当前RSI=62, MACD金叉, 量比1.8, 请做简要技术分析"},
            ],
            task_type="technical_analysis",
        )
        print(f"  ✅ 路由: {result.tier.value}")
        print(f"  📄 分析: {result.response[:200]}...")
        print(f"  💰 成本: ¥{result.cost:.6f}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")


async def main():
    print("=" * 60)
    print("🔌 DeepSeek API 连通性 + ModelRouter 端到端测试")
    print("=" * 60)

    # 测试1-2: 直连API
    api_ok = await test_direct_api()
    if not api_ok:
        print("\n❌ DeepSeek API 连接失败, 请检查 Key 和网络")
        return

    # 测试3: ModelRouter
    await test_model_router()

    # 测试4: CostMonitor
    await test_cost_monitor()

    # 测试5: KnowledgeManager + LLM
    await test_knowledge_with_llm()

    print("\n" + "=" * 60)
    print("✅ 全部测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
