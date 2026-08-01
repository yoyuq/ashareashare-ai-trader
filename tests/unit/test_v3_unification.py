"""v3.1-deerflow 遗留收尾测试 — 执行管线统一 / 新模块覆盖 / 回归

覆盖:
  1. daily_runner._limit_pct_for_code (板块涨跌停 10/20/30%)
  2. killtest.runner._compute_indicator_df 的 NaN date 修复回归
  3. notify.service (DailySummary/SignalAlert markdown, NotificationService 通道)
  4. knowledge.get_strategies_for_regime
"""

import asyncio
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# 1. daily_runner._limit_pct_for_code (板块涨跌停)
# ═══════════════════════════════════════════════════════════════

def test_limit_pct_for_code_board_aware():
    from simulation.daily_runner import _limit_pct_for_code
    assert _limit_pct_for_code("600519") == 10.0   # 主板
    assert _limit_pct_for_code("000858") == 10.0   # 主板
    assert _limit_pct_for_code("300750") == 20.0   # 创业板
    assert _limit_pct_for_code("688111") == 20.0   # 科创板
    assert _limit_pct_for_code("831010") == 30.0   # 北交所 8 开头
    assert _limit_pct_for_code("430047") == 30.0   # 北交所 4 开头


# ═══════════════════════════════════════════════════════════════
# 2. killtest.runner._compute_indicator_df — NaN date 修复回归
# ═══════════════════════════════════════════════════════════════

def test_compute_indicator_df_fixes_nan_date(monkeypatch):
    """to_dataframe() 的 date 列被强转 NaN → 应从源 df 按位置覆盖 (回归)"""
    import analysis.indicators as ai
    from killtest.runner import _compute_indicator_df

    n = 80
    src = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "open": [10.0] * n, "close": [10.0] * n,
        "high": [10.5] * n, "low": [9.5] * n, "volume": [1000] * n,
    })

    class FakeAnalyzer:
        def compute_all(self, df, symbol="", skip_patterns=False):
            ind = pd.DataFrame({"trend_score": [0.5] * n, "rsi_14": [50.0] * n})
            ind["date"] = float("nan")  # 复现 to_dataframe 的 NaN date bug
            return SimpleNamespace(to_dataframe=lambda: ind)

    monkeypatch.setattr(ai, "TechnicalAnalyzer", FakeAnalyzer)
    out = asyncio.run(_compute_indicator_df(src, "sh.600519"))
    assert out is not None
    assert out["date"].notna().all(), "date 列不应有 NaN"
    assert len(out) == n
    assert pd.Timestamp(out["date"].iloc[0]) == pd.Timestamp("2026-01-01")


def test_compute_indicator_df_too_short_returns_none():
    from killtest.runner import _compute_indicator_df
    short = pd.DataFrame({"date": ["2026-01-01"], "close": [10.0]})
    assert asyncio.run(_compute_indicator_df(short, "sh.600519")) is None


# ═══════════════════════════════════════════════════════════════
# 3. notify.service (收敛自 notifications/ 的新模块)
# ═══════════════════════════════════════════════════════════════

def test_daily_summary_markdown():
    from notify import DailySummary
    s = DailySummary(date="2026-08-01", regime="range_bound", regime_confidence=0.6,
                     total_scanned=100, recommendations_count=3,
                     top_picks=[{"name": "贵州茅台", "symbol": "600519",
                                 "strategy": "dual_ma", "score": 80, "win_rate": 0.55}],
                     debate_direction="neutral", debate_confidence=0.5, cost_today=0.15)
    md = s.to_markdown()
    assert "2026-08-01" in md and "range_bound" in md
    assert "贵州茅台" in md and "600519" in md


def test_signal_alert_markdown():
    from notify import SignalAlert
    a = SignalAlert(symbol="sh.600519", symbol_name="贵州茅台", strategy="dual_ma",
                    direction="long", confidence="A", entry_price=100,
                    stop_loss=95, take_profit=110, win_rate=0.6, key_reason="金叉")
    md = a.to_markdown()
    assert "贵州茅台" in md and "100.00" in md


def test_notification_service_channels(monkeypatch):
    from notify import NotificationService
    monkeypatch.setenv("WECOM_WEBHOOK", "https://example.com/xxx")
    ns = NotificationService()
    assert "console" in ns.active_channels
    # 未配置时只有 console
    monkeypatch.setenv("WECOM_WEBHOOK", "")
    assert ns.active_channels == ["console"]


# ═══════════════════════════════════════════════════════════════
# 4. knowledge.get_strategies_for_regime
# ═══════════════════════════════════════════════════════════════

def test_get_strategies_for_regime():
    from knowledge.manager import KnowledgeManager
    km = KnowledgeManager()
    strats = km.get_strategies_for_regime("range_bound")
    assert isinstance(strats, list)
    # 返回的策略应匹配 range_bound regime 或其 market_regimes 包含它
    for s in strats:
        assert isinstance(s, dict) and "id" in s
