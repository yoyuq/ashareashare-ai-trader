"""
kill-test 编排器 — 三路信号对比

流程: 拉取历史K线 → 计算指标 → 逐日生成三路信号 → 持久化 →
      N日前向收益结算 → 全市场基线 → 渲染对比报告
"""

import asyncio
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from killtest.arms import LLMArm, RandomArm, RuleArm, Signal
from killtest.report import render
from killtest.tracker import KillTestTracker

# 默认流动性好的股票池 (可被 --universe 覆盖)
DEFAULT_UNIVERSE = [
    "sh.600519", "sz.000858", "sz.300750", "sh.601318", "sh.600036",
    "sh.601899", "sz.002594", "sh.600900", "sh.601166", "sh.600887",
    "sz.000001", "sh.688981", "sz.000333", "sh.601888", "sh.600030",
    "sz.002415", "sh.601012", "sz.000725", "sh.600276", "sz.002475",
]


async def _default_get_kline(router, symbol: str, start: date, end: date) -> pd.DataFrame:
    """默认取数: 走 DataRouter (Baostock/Tencent 免代理)"""
    from data.providers.base import DataFrequency, DataRequest
    req = DataRequest(symbol=symbol, start_date=start, end_date=end,
                      frequency=DataFrequency.DAILY)
    result = await router.get_daily_kline(req)
    return result.data


async def _compute_indicator_df(df: pd.DataFrame, symbol: str,
                                indicator_fn=None) -> Optional[pd.DataFrame]:
    """技术指标 DataFrame (因果, 供 asof 切片)"""
    if df is None or len(df) < 60:
        return None
    if indicator_fn is not None:
        return indicator_fn(df, symbol)  # 测试/自定义注入
    from analysis.indicators import TechnicalAnalyzer
    try:
        res = TechnicalAnalyzer().compute_all(df, symbol=symbol, skip_patterns=True)
        ind_df = res.to_dataframe()
        ind_df = ind_df.copy()
        # v3.1-deerflow 修复: to_dataframe() 的 date 列会被 pandas 强转成 NaN float64
        # (字符串日期与数值指标列对齐失败)。始终用源 df 的日期按位置覆盖。
        if "date" in df.columns and len(df) >= len(ind_df):
            ind_df["date"] = pd.to_datetime(
                df["date"].values[:len(ind_df)], errors="coerce"
            )
        return ind_df
    except Exception:
        return None


def _forward_rets_from_df(df: pd.DataFrame, signal_dates: List[str],
                          lookahead: int) -> List[float]:
    """给定 df, 对每个信号日计算做多前向收益 (基线用)"""
    if df is None or df.empty or "close" not in df.columns:
        return []
    dates = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
    closes = pd.to_numeric(df["close"], errors="coerce")
    rets = []
    for d in signal_dates:
        mask = dates >= pd.Timestamp(d)
        if not mask.any():
            continue
        i = int(np.argmax(mask.values))
        if i + lookahead >= len(closes):
            continue
        entry, exit_px = closes.iloc[i], closes.iloc[i + lookahead]
        if (entry is not None and np.isfinite(entry) and entry > 0
                and exit_px is not None and np.isfinite(exit_px) and exit_px > 0):
            rets.append(float(exit_px / entry - 1.0))
    return rets


async def run_killtest(
    history_days: int = 250,
    symbols: Optional[List[str]] = None,
    arms: tuple = ("rule", "random", "llm"),
    lookahead: int = 5,
    seed: int = 42,
    data_dir: str = "killtest_data",
    report_dir: str = "reports",
    get_kline: Optional[Callable] = None,   # async (symbol, start, end) -> df
    router=None,
    indicator_fn=None,                      # (df, symbol) -> indicator df (测试注入)
) -> Dict:
    """
    运行一次 kill-test 对比

    Returns:
        {arms: {arm: [outcome]}, baseline_mean, baseline_n, report: markdown, signal_counts}
    """
    symbols = symbols or DEFAULT_UNIVERSE
    arms = list(arms)
    tracker = KillTestTracker(data_dir)
    rule_arm, random_arm = RuleArm(), RandomArm(seed)
    llm_arm = LLMArm(report_dir)

    if get_kline is None:
        if router is None:
            from data import get_data_router
            router = get_data_router()
        get_kline = lambda sym, s, e: _default_get_kline(router, sym, s, e)

    # ── 1. 拉取历史 + 前瞻缓冲 ──
    end = date.today()
    start = end - timedelta(days=int(history_days * 1.5) + lookahead + 40)
    price_data: Dict[str, pd.DataFrame] = {}
    indicator_dfs: Dict[str, Optional[pd.DataFrame]] = {}
    for sym in symbols:
        try:
            df = await get_kline(sym, start, end)
            if df is not None and not df.empty:
                price_data[sym] = df
                indicator_dfs[sym] = await _compute_indicator_df(df, sym, indicator_fn)
        except Exception:
            continue

    if not price_data:
        return {"error": "无数据, 检查网络/股票池", "report": ""}

    # ── 2. 测试日窗口 (取历史最后一个交易日的 history_days 个交易日) ──
    all_dates = sorted({str(pd.Timestamp(d).date()) for df in price_data.values()
                        for d in (pd.to_datetime(df["date"]) if "date" in df.columns else df.index)})
    signal_dates = all_dates[-history_days:] if len(all_dates) > history_days else all_dates
    valid_symbols = sorted(price_data.keys())

    # ── 3. 逐日生成三路信号 ──
    emit_counts = {a: 0 for a in arms}
    for d in signal_dates:
        if "rule" in arms:
            sigs = rule_arm.generate({s: d for s, d in indicator_dfs.items()}, d)
        else:
            sigs = []
        if "random" in arms:
            n_buy = sum(1 for s in sigs if s.action == "BUY")
            n_sell = sum(1 for s in sigs if s.action == "SELL")
            sigs += random_arm.generate(valid_symbols, d, n_buy, n_sell)
        if "llm" in arms:
            sigs += llm_arm.load_daily(d)
        for s in sigs:
            if s.arm in arms:
                if tracker.emit(s):
                    emit_counts[s.arm] += 1

    # ── 4. N日前向收益结算 ──
    outcomes: Dict[str, List] = {a: [] for a in arms}
    for a in arms:
        tracker.settle(a, price_data, lookahead=lookahead)
        outcomes[a] = tracker.load_outcomes(a)

    # ── 5. 全市场基线 (每个测试日每只标的的做多前向收益) ──
    baseline_rets: List[float] = []
    for sym, df in price_data.items():
        baseline_rets.extend(_forward_rets_from_df(df, signal_dates, lookahead))

    # ── 6. 渲染报告 ──
    report = render(
        {a: outcomes[a] for a in arms if outcomes[a]},
        {"outcomes": []},  # baseline 直接在 render 内用 universe 均值
        title=f"kill-test 三路信号对比 (lookahead={lookahead}日, 股票池={len(symbols)}只, "
              f"窗口={signal_dates[0]}~{signal_dates[-1]})",
    )
    # 基线均值追加到报告 (仅在存在占位符时替换)
    if baseline_rets:
        report = report.replace(
            "> 基线: 无",
            f"> 全市场基线: {len(baseline_rets)} 条前向收益, 均值 "
            f"{np.mean(baseline_rets)*100:.2f}%",
        )

    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        (Path(report_dir) / "killtest_report.md").write_text(report, encoding="utf-8")

    return {
        "arms": outcomes,
        "baseline_mean": float(np.mean(baseline_rets)) if baseline_rets else 0.0,
        "baseline_n": len(baseline_rets),
        "signal_counts": emit_counts,
        "window": [signal_dates[0], signal_dates[-1]],
        "report": report,
    }
