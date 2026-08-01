"""v3.0 行为测试 — 补齐关键层测试盲区

覆盖:
1. 数据清洗 clean_ohlcv (停牌零价→NaN→ffill, 2026 最佳实践)
2. ModelRouter 重试/缓存命中计价
3. daily_runner Phase 2 异常路径 (无分析结果 / dry-run)
4. recommender._calc_bt_stats (逐笔净收益聚合)
"""

import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# 1. 数据清洗 (停牌零价处理)
# ═══════════════════════════════════════════════════════════════

def test_clean_ohlcv_zero_price_to_nan_ffill():
    from data.processors.cleaning import clean_ohlcv

    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4, freq="D"),
        "open": [10.0, 0.0, 12.0, 0.0],
        "high": [11.0, 0.0, 13.0, 0.0],
        "low": [9.0, 0.0, 11.0, 0.0],
        "close": [10.5, 0.0, 12.5, 0.0],
        "volume": [1000, 0, 1200, 0],
    })
    out = clean_ohlcv(df)

    # 停牌日(零价)价格被前向填充为上一有效值
    assert out.loc[1, "close"] == 10.5, "停牌日应沿用上一收盘价"
    assert out.loc[3, "close"] == 12.5
    # 原始 DataFrame 不被修改
    assert df.loc[1, "close"] == 0.0
    # is_trade 标注 (真实成交日=1, 停牌日=0)
    assert out["is_trade"].tolist() == [1, 0, 1, 0]
    # 成交量保持 0 (真停牌)
    assert out.loc[1, "volume"] == 0


def test_clean_ohlcv_no_zero_prices_unchanged():
    from data.processors.cleaning import clean_ohlcv

    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"],
                       "open": [10.0, 11.0], "high": [11.0, 12.0],
                       "low": [9.0, 10.0], "close": [10.5, 11.5]})
    out = clean_ohlcv(df)
    assert out["close"].tolist() == [10.5, 11.5]
    assert out["is_trade"].tolist() == [1, 1]


# ═══════════════════════════════════════════════════════════════
# 2. ModelRouter 行为
# ═══════════════════════════════════════════════════════════════

class _FailingCompletions:
    async def create(self, **kwargs):
        raise ConnectionError("mock network down")


def test_router_retries_then_raises():
    """模型调用失败 → 指数退避重试 → 全部失败抛 RuntimeError"""
    from models.router import ModelRouter

    router = ModelRouter()
    router._deepseek_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FailingCompletions())
    )
    with pytest.raises(RuntimeError, match="所有模型调用均失败"):
        asyncio.run(router.route([{"role": "user", "content": "hi"}], max_retries=1))


def test_cost_cache_hit_discount(monkeypatch):
    """缓存命中输入按 1/50 计价; 无缓存按全价"""
    from models.router import ModelRouter

    monkeypatch.setenv("SKIP_PEAK_HOUR", "1")  # 排除高峰×2 干扰
    router = ModelRouter()

    # 全部输入为缓存命中: 0.1M×0.02 + 0.01M×2 = 0.002 + 0.02 = 0.022
    cost_cached = router._calculate_cost(
        {"input_tokens": 100000, "cache_hit_tokens": 100000, "output_tokens": 10000})
    assert abs(cost_cached - 0.022) < 1e-6, f"缓存命中计价错误: {cost_cached}"

    # 无缓存命中: 0.1M×1 + 0.01M×2 = 0.12
    cost_fresh = router._calculate_cost(
        {"input_tokens": 100000, "cache_hit_tokens": 0, "output_tokens": 10000})
    assert abs(cost_fresh - 0.12) < 1e-6, f"无缓存计价错误: {cost_fresh}"


# ═══════════════════════════════════════════════════════════════
# 3. daily_runner Phase 2 异常路径
# ═══════════════════════════════════════════════════════════════

def test_phase2_no_analysis_graceful(tmp_path, monkeypatch):
    """无分析结果文件 → 优雅返回 no_analysis, 不崩溃"""
    from simulation import daily_runner

    monkeypatch.setattr(daily_runner, "REPORT_DIR", tmp_path)
    monkeypatch.setenv("PORTFOLIO_PATH", str(tmp_path / "portfolio.json"))
    result = asyncio.run(daily_runner.phase2_execute())
    assert result == {"status": "no_analysis"}


def test_phase2_dry_run_skips_trading(tmp_path, monkeypatch):
    """dry_run → 只分析不交易"""
    from simulation import daily_runner

    monkeypatch.setattr(daily_runner, "REPORT_DIR", tmp_path)
    monkeypatch.setenv("PORTFOLIO_PATH", str(tmp_path / "portfolio.json"))
    # 写一个空分析结果文件
    (tmp_path / "deep_analysis_top100.json").write_text(
        json.dumps({"results": [], "market_regime": {"regime": "range_bound"}}),
        encoding="utf-8",
    )
    result = asyncio.run(daily_runner.phase2_execute(dry_run=True))
    assert result["status"] == "dry_run"
    assert result["buys"] == 0 and result["sells"] == 0


# ═══════════════════════════════════════════════════════════════
# 4. recommender._calc_bt_stats (逐笔聚合)
# ═══════════════════════════════════════════════════════════════

def test_calc_bt_stats_returns_raw_trades():
    from analysis.recommender import _calc_bt_stats

    stats = _calc_bt_stats([5.0, -3.0, 2.0, 1.0, -4.0])
    assert stats["signals"] == 5
    assert stats["cost_adjusted"] is True
    assert "trades" in stats, "v3.0 应返回逐笔净收益供真实分布聚合"
    assert len(stats["trades"]) == 5
    assert stats["win_rate"] == pytest.approx(3 / 5, abs=1e-9)


def test_calc_bt_stats_empty():
    from analysis.recommender import _calc_bt_stats

    stats = _calc_bt_stats([])
    assert stats["signals"] == 0
    assert stats["trades"] == []
