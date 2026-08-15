"""End-to-end test: DeepSeek V4-Flash + Full Pipeline

联网/环境依赖测试, 整文件标记为 network, 默认套件跳过 (pytest -m "not network"),
显式运行: pytest -m network tests/test_e2e.py
"""
import asyncio, sys, os, json, time
from pathlib import Path
from datetime import date, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

pytestmark = pytest.mark.network

async def test_deepseek_available():
    """Test 1: DeepSeek V4-Flash 可用性"""
    print("\n--- Test 1: DeepSeek V4-Flash ---")
    from models.router import DEEPSEEK_FLASH_MODEL
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("  DEEPSEEK_API_KEY 未配置")
        return False
    try:
        from openai import AsyncOpenAI
        c = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        r = await asyncio.wait_for(
            c.chat.completions.create(
                model=DEEPSEEK_FLASH_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            ),
            timeout=15,
        )
        print(f"  Model {DEEPSEEK_FLASH_MODEL} 响应正常: {r.choices[0].message.content[:30]!r}")
        return True
    except Exception as e:
        print(f"  Not available: {e}")
        return False

async def test_model_router_hybrid():
    """Test 2: ModelRouter (v3.0 统一 V4-Flash)"""
    print("\n--- Test 2: ModelRouter ---")
    from models.router import ModelRouter
    router = ModelRouter(daily_budget=1.0)

    # 所有任务统一路由到 flash
    print("  Task: kline_describe (should route FLASH)...")
    try:
        r = await router.route(
            [{"role":"user","content":"什么是十字星K线?一句话回答"}],
            task_type="kline_describe"
        )
        print(f"  Tier: {r.tier} | Model: {r.model_name} | {r.latency_ms:.0f}ms | Cost: {r.cost:.6f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # 分析任务 -> Flash
    print("  Task: technical_analysis (should route FLASH)...")
    try:
        r = await router.route(
            [{"role":"user","content":"RSI=62, MACD金叉, 量比1.8, 简要分析"}],
            task_type="technical_analysis"
        )
        print(f"  Tier: {r.tier} | Model: {r.model_name} | {r.latency_ms:.0f}ms | Cost: {r.cost:.6f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # 决策任务 -> Flash (v3.0 统一)
    print("  Task: daily_synthesis (should route FLASH)...")
    try:
        r = await router.route(
            [{"role":"user","content":"当前市场弱牛, 请给一个简短的综合研判"}],
            task_type="daily_synthesis"
        )
        print(f"  Tier: {r.tier} | Model: {r.model_name} | {r.latency_ms:.0f}ms | Cost: {r.cost:.6f}")
    except Exception as e:
        print(f"  Failed: {e}")

    cs = router.cost_summary()
    print(f"  Total cost: {cs['daily_cost']:.4f} | Calls: {cs['total_calls']}")

async def test_win_rate_analysis():
    """Test 3: Win rate engine"""
    print("\n--- Test 3: Win Rate Analysis ---")
    import numpy as np
    import pandas as pd
    from analysis.winrate import WinRateAnalyzer, SignalTracker

    # Generate mock trade log
    np.random.seed(123)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    sells = pd.DataFrame({
        "date": dates,
        "symbol": ["sh.600000"] * n,
        "side": ["sell"] * n,
        "pnl": np.random.choice([-500, -200, 300, 800, 1500], n, p=[0.15, 0.2, 0.25, 0.25, 0.15]),
        "pnl_pct": np.random.choice([-3, -1.5, 2, 5, 8], n, p=[0.15, 0.2, 0.25, 0.25, 0.15]),
        "holding_days": np.random.randint(1, 15, n),
    })

    close = 10 + np.cumsum(np.random.randn(n) * 0.05)
    prices = pd.DataFrame({"date": dates, "close": close})

    wa = WinRateAnalyzer()
    report = wa.analyze(sells, prices)
    print(f"  Signals: {report.total_signals} | Wins: {report.total_win} | WR: {report.overall_win_rate:.1%}")
    print(f"  ProfitFactor: {report.profit_factor} | EV: {report.expected_value:+.1f}% | MaxConsecLoss: {report.max_consecutive_loss}")
    for s, stats in report.scenario_breakdown.items():
        if stats["total"] > 0:
            print(f"  {s}: WR={stats['win_rate']:.0%} ({stats['wins']}/{stats['total']}) PF={stats['profit_factor']:.1f}")

    # Kelly
    pos, shares = wa.optimal_position_size(100000, report.overall_win_rate, 5.0, 2.5)
    print(f"  Kelly position: {pos:.0f} ({shares} shares)")

    # Signal tracker
    st = SignalTracker()
    for i in range(10):
        st.emit(f"sh.{600000+i}", "long", 10+i, 0.5+0.05*i)
    stats = st.get_tracking_stats()
    print(f"  Active signals: {stats}")

async def test_knowledge_integration():
    """Test 4: KnowledgeManager + WinRate integration"""
    print("\n--- Test 4: KnowledgeBase + WinRate ---")
    from knowledge.manager import KnowledgeManager
    km = KnowledgeManager()

    # Get strategy with capacity limit
    s = km.get_strategy("dual_ma_trend")
    cap = s.get("capacity_limit", 0)
    print(f"  Strategy: {s['name']} | Capacity: {cap:,.0f}")

    # Hardened definition
    d = km.get_hardened_definition("rsi_overbought")
    print(f"  RSI overbought: threshold={d['threshold']}")

    # All regime-appropriate strategies
    strats = km.get_strategies_for_regime("weak_bull")
    print(f"  Weak bull strategies: {[s['id'] for s in strats]}")

async def main():
    print("=" * 55)
    print("E2E Test: DeepSeek V4-Flash + WinRate + Knowledge")
    print("=" * 55)

    ds_ok = await test_deepseek_available()
    await test_model_router_hybrid()
    await test_win_rate_analysis()
    await test_knowledge_integration()

    print("\n" + "=" * 55)
    print("All E2E tests completed!")
    print(f"Dashboard: http://localhost:8501")
    print(f"API:       http://localhost:8000/docs")
    print("=" * 55)

if __name__ == "__main__":
    asyncio.run(main())
