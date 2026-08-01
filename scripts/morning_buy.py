#!/usr/bin/env python3
"""
早盘自动买入脚本 — A股模拟交易 v2.11

改进:
  - 优先使用 pending_recommendations 中的收盘价 (来自分析时获取的实际价格)
  - 传递完整推荐元数据 (策略名/胜率/夏普/评分) 到持仓
  - 动态止损止盈 (基于综合评分和RSI)
  - 智能仓位分配 (按确信度×评分加权)
  - 🆕 v2.11: 市场状态仓位上限 (熊市自动减仓)
  - 🆕 v2.11: RSI超买过滤 (RSI>70降级, RSI>75跳过)
  - 🆕 v2.11: 行业集中度收紧 (30%→25%)

使用:
    python scripts/morning_buy.py                  # 正常执行
    python scripts/morning_buy.py --dry-run       # 试运行
    python scripts/morning_buy.py --min-confidence 0.50  # 自定义阈值
"""

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from simulation.portfolio import PortfolioManager
from simulation.paper_trader import (
    PaperTradingEngine,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_SINGLE_POSITION_PCT,
)

REPORT_DIR = Path(__file__).parent.parent / "reports"

from scripts.shared import (NAME_MAP, resolve_name, get_fallback_price,
    get_best_price, refresh_fallback_prices_from_tencent,
    calc_kelly_position_pct, calc_atr_stop_loss)

# v2.10: 行业分组 (用于集中度检查)
INDUSTRY_MAP = {
    "sh.600036": "金融", "sh.601318": "金融", "sh.600030": "金融",
    "sh.600048": "地产", "sh.601668": "基建",
    "sh.600519": "消费", "sz.000858": "消费", "sz.000333": "消费",
    "sz.000651": "消费", "sh.601933": "消费", "sh.600754": "消费",
    "sh.603605": "消费", "sz.002572": "消费", "sh.603877": "消费",
    "sz.002415": "科技", "sh.603501": "科技", "sz.002230": "科技",
    "sh.688111": "科技", "sz.300502": "科技", "sh.600050": "科技",
    "sz.002027": "传媒", "sz.300413": "传媒", "sz.300308": "科技",
    "sz.300033": "科技", "sz.300624": "科技", "sz.300433": "科技",
    "sz.300750": "新能源", "sh.601012": "新能源", "sz.002594": "新能源",
    "sh.601127": "新能源", "sz.002459": "新能源", "sz.300014": "新能源",
    "sh.688981": "半导体", "sz.002049": "半导体", "sh.688012": "半导体",
    "sh.603986": "半导体", "sh.688256": "半导体",
    "sz.300124": "制造", "sh.600031": "制造", "sh.600760": "军工",
    "sz.002179": "军工", "sh.688017": "制造",
    "sh.600309": "化工", "sh.601899": "资源", "sz.002460": "资源",
    "sh.600019": "钢铁", "sh.601088": "煤炭", "sh.601225": "煤炭",
    "sh.600028": "石化", "sh.600585": "建材",
    "sh.600009": "交运", "sh.601111": "交运",
    "sh.600900": "电力", "sh.601985": "电力", "sz.300070": "环保",
    "sh.600276": "医药", "sz.300760": "医药", "sz.002714": "农业",
    "sh.600895": "园区",
}
MAX_INDUSTRY_PCT = 0.25  # 单行业最大占比25% (v2.11: 从30%收紧)

# 🆕 v2.11: 市场状态 → 仓位上限映射
REGIME_POSITION_LIMITS = {
    "strong_bull":   {"max_positions": 10, "single_pct": 0.20, "label": "强牛-满仓进攻"},
    "weak_bull":     {"max_positions": 8,  "single_pct": 0.15, "label": "弱牛-积极"},
    "range_bound":   {"max_positions": 6,  "single_pct": 0.12, "label": "震荡-中性"},
    "weak_bear":     {"max_positions": 3,  "single_pct": 0.10, "label": "弱熊-防守"},
    "strong_bear":   {"max_positions": 1,  "single_pct": 0.05, "label": "强熊-空仓"},
    "crisis":        {"max_positions": 1,  "single_pct": 0.05, "label": "危机-最低参与"},  # v2.13: 从不完全空仓
}


def load_market_regime() -> Tuple[str, dict]:
    """从今日分析报告加载市场状态, 返回 (regime_str, params_dict)"""
    today = date.today().isoformat()
    datapath = REPORT_DIR / f"data_{today}.json"
    try:
        if datapath.exists():
            with open(datapath, "r", encoding="utf-8") as f:
                data = json.load(f)
            regime_info = data.get("market_regime", {})
            regime = regime_info.get("regime", "range_bound")
            confidence = regime_info.get("confidence", 0)
            params = REGIME_POSITION_LIMITS.get(regime, REGIME_POSITION_LIMITS["range_bound"])
            logger.info(f"市场状态: {regime} (置信度{confidence:.0%}) → "
                       f"{params['label']}, 最大{params['max_positions']}只")
            return regime, params
    except Exception as e:
        logger.warning(f"读取市场状态失败: {e}, 使用默认震荡市参数")
    params = REGIME_POSITION_LIMITS["range_bound"]
    return "range_bound", params


def check_rsi_filter(rsi: float, trend_score: float) -> Tuple[bool, str]:
    """
    RSI超买过滤 (v2.11)

    Returns:
        (should_skip: bool, reason: str)
    """
    if rsi > 75:
        return True, f"RSI={rsi:.0f}极度超买, 大概率回调"
    if rsi > 70:
        if trend_score <= 0.5:
            return True, f"RSI={rsi:.0f}超买且趋势偏弱(趋势{trend_score:.2f}), 回调风险高"
        else:
            return False, f"RSI={rsi:.0f}超买但趋势强劲(趋势{trend_score:.2f}), 允许买入(降级警告)"
    return False, ""


async def get_live_prices(symbols: List[str]) -> Dict[str, float]:
    """获取实时价格 (v2.10: 双源fallback + 重试)"""
    prices = {}
    # 优先EastMoney(实时), 降级Baostock
    try:
        from data.router import get_data_router
        from data.providers.base import DataFrequency, DataRequest

        router = get_data_router()
        yesterday = date.today() - timedelta(days=1)

        for sym in symbols:
            try:
                req = DataRequest(sym, yesterday - timedelta(days=3),
                                date.today(), DataFrequency.DAILY)
                result = await asyncio.wait_for(
                    router.get_daily_kline(req), timeout=15
                )
                df = result.data
                if not df.empty and len(df) > 0:
                    prices[sym] = float(df["close"].iloc[-1])
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"实时价格获取失败: {e}")
    return prices


async def get_realtime_quotes(symbols: List[str]) -> Dict[str, dict]:
    """🆕 v2.10: 获取实时行情(含涨跌幅), 用于价格偏离检测
    v2.11: 优先Tencent→EastMoney fallback"""
    quotes = {}
    # 优先用Tencent (当前网络可用)
    try:
        from data.providers.tencent_provider import TencentFinanceProvider
        tp = TencentFinanceProvider()
        raw = await asyncio.wait_for(tp.get_realtime_quotes(symbols), timeout=10)
        for sym, info in raw.items():
            quotes[sym] = {
                "price": info.get("price", 0),
                "change_pct": info.get("change_pct", 0),
                "high": info.get("high", 0),
                "low": info.get("low", 0),
            }
        if quotes:
            return quotes
    except (asyncio.TimeoutError, Exception):
        pass
    # Fallback: EastMoney
    try:
        from data.providers.eastmoney_provider import EastMoneyProvider
        em = EastMoneyProvider()
        raw = await asyncio.wait_for(em.get_realtime_quotes(symbols), timeout=10)
        for sym, info in raw.items():
            quotes[sym] = {
                "price": info.get("price", 0),
                "change_pct": info.get("change_pct", 0),
                "high": info.get("high", 0),
                "low": info.get("low", 0),
            }
    except (asyncio.TimeoutError, Exception):
        pass
    return quotes


def calc_dynamic_sl_tp(price: float, composite_score: float, rsi: float) -> Tuple[float, float]:
    """
    动态计算止损止盈

    规则:
      - 高评分(>=80): 止损-5%, 止盈+12%  (高确信, 宽止盈)
      - 中评分(50-80): 止损-7%, 止盈+10% (标准)
      - 低评分(<50): 止损-10%, 止盈+8%   (低确信, 严守止损)
      - RSI超买(>70): 止损收紧2%
    """
    if composite_score >= 80:
        sl_pct, tp_pct = 0.05, 0.12
    elif composite_score >= 50:
        sl_pct, tp_pct = 0.07, 0.10
    else:
        sl_pct, tp_pct = 0.10, 0.08

    # RSI调整
    if rsi > 70:
        sl_pct += 0.02  # 超买时收紧止损

    stop_loss = round(price * (1 - sl_pct), 2)
    take_profit = round(price * (1 + tp_pct), 2)

    return stop_loss, take_profit


async def run_morning_buy(
    min_confidence: float = 0.50,
    dry_run: bool = False,
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> Dict[str, Any]:
    """执行早盘买入"""

    logger.info(f"═══ 早盘模拟买入 v2.11 ═══")
    logger.info(f"置信度阈值: {min_confidence:.0%} | 最大持仓: {max_positions} | "
               f"模式: {'试运行' if dry_run else '执行'}")

    # 1. 初始化
    manager = PortfolioManager()
    engine = PaperTradingEngine(manager)
    state = engine.state

    # 🆕 v2.11: 加载市场状态, 动态调整仓位上限
    regime, regime_params = load_market_regime()
    regime_max_positions = regime_params["max_positions"]
    regime_single_pct = regime_params["single_pct"]

    # 如果用户手动指定了 --max-positions, 以用户为准;
    # 否则使用市场状态决定的仓位上限 (取更保守的值)
    if max_positions == DEFAULT_MAX_POSITIONS:
        max_positions = min(max_positions, regime_max_positions)
    else:
        logger.info(f"用户覆盖仓位上限: {max_positions} (市场建议: {regime_max_positions})")

    if regime_max_positions == 0:
        logger.warning(f"⚠️ 市场状态={regime}, 禁止买入! 跳过全部候选")
        return {"status": "blocked_by_regime", "regime": regime, "buys": 0}

    # 🆕 v2.14: 回撤断路器检查 (PortfolioRiskManager from analysis/risk_controls)
    try:
        from analysis.risk_controls import PortfolioRiskManager
        import pandas as pd
        risk_mgr = PortfolioRiskManager(initial_capital=state.initial_capital)
        # 使用每日快照构建收益率序列
        if len(state.daily_snapshots) >= 5:
            daily_returns = pd.Series([
                s.daily_return_pct / 100 for s in state.daily_snapshots[-60:]
            ])
            risk_state = risk_mgr.update(state.total_value, daily_returns)
            if risk_state.circuit_breaker_active:
                logger.warning(
                    f"🚨 回撤断路器触发! 回撤{risk_state.drawdown_pct:.1f}% "
                    f"超过阈值, 跳过本次所有买入"
                )
                if risk_state.warning_messages:
                    for w in risk_state.warning_messages:
                        logger.warning(f"  {w}")
                return {
                    "status": "blocked_by_drawdown",
                    "drawdown_pct": risk_state.drawdown_pct,
                    "warnings": risk_state.warning_messages,
                    "buys": 0,
                }
    except Exception as e:
        logger.debug(f"回撤断路器检查跳过: {e}")

    # 🆕 v2.12: 刷新兜底价格
    await refresh_fallback_prices_from_tencent()

    # 2. 加载分析数据 (v2.13: 始终加载, 用于ATR和历史数据)
    today = date.today().isoformat()
    datapath = REPORT_DIR / f"data_{today}.json"
    analysis_data = {}
    if datapath.exists():
        with open(datapath, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

    # 从 pending_recommendations 加载推荐 (含价格)
    pending = state.pending_recommendations
    logger.info(f"待处理推荐: {len(pending)}条 | 当前持仓: {len(state.positions)}只 | "
               f"现金: {state.cash:,.2f}")

    if not pending and analysis_data:
        # 从分析报告加载
        recs = analysis_data.get("stock_recommendations", {})
        prices_data = analysis_data.get("analysis_prices", {})

        for sym, rec in recs.items():
            if isinstance(rec, dict) and rec.get("action") == "BUY":
                if rec.get("conviction", 0) >= min_confidence:
                    pinfo = prices_data.get(sym, {})
                    pending.append({
                        "symbol": sym,
                        "conviction": rec.get("conviction", 0),
                        "score": rec.get("score", 0),
                        "composite_score": pinfo.get("composite_score", rec.get("score", 5) * 10),
                        "close_price": pinfo.get("close", 0),
                        "rsi": pinfo.get("rsi_14", 50),
                        "trend_score": pinfo.get("trend_score", 0),
                        "strategy_id": pinfo.get("best_strategy_id", ""),
                        "strategy_name": pinfo.get("best_strategy_name", ""),
                        "win_rate": pinfo.get("best_strategy_win_rate", 0),
                        "sharpe": pinfo.get("best_strategy_sharpe", 0),
                        "key_reasons": rec.get("key_reasons", []),
                        "risks": rec.get("risks", []),
                        "verdict_summary": rec.get("verdict_summary", ""),
                    })
        logger.info(f"从分析报告加载: {len(pending)}条买入候选 (含实际收盘价)")

    if not pending:
        logger.warning("无买入推荐。请先运行: python scripts/evening_summary.py --analyze")
        return {"status": "no_data", "buys": 0}

    # 3. 按确信度×评分排序
    pending.sort(key=lambda r: r.get("conviction", 0) * r.get("score", 0), reverse=True)

    # 4. 收集需要获取实时价格的标的 (pending中没有close_price的)
    symbols_need_price = [r["symbol"] for r in pending if r.get("close_price", 0) <= 0]
    live_prices = {}
    if symbols_need_price:
        live_prices = await get_live_prices(symbols_need_price)
        logger.info(f"获取到 {len(live_prices)} 只实时价格")

    # 🆕 v2.10: 获取实时行情, 检测分析价与实时价的偏离
    all_pending_symbols = [r["symbol"] for r in pending]
    # 🆕 v2.12: 也获取当前持仓的实时行情 (用于MTM)
    all_rt_symbols = list(set(all_pending_symbols + list(state.positions.keys())))
    rt_quotes = await get_realtime_quotes(all_rt_symbols)
    price_gaps = []
    if rt_quotes:
        for rec in pending:
            sym = rec["symbol"]
            close_price = rec.get("close_price", 0)
            rt = rt_quotes.get(sym, {})
            rt_price = rt.get("price", 0)
            if close_price > 0 and rt_price > 0:
                gap = (rt_price / close_price - 1) * 100
                if abs(gap) > 2:
                    price_gaps.append((sym, resolve_name(sym), close_price, rt_price, gap))
        if price_gaps:
            logger.warning(f"⚠️ 价格偏离>2%: {len(price_gaps)}只")
            for sym, name, close, rt, gap in price_gaps:
                logger.warning(f"  {name}({sym}): 分析价{close:.2f} → 实时{rt:.2f} ({gap:+.1f}%)")

    # 🆕 v2.12: 持仓MTM — 用实时行情更新持仓现价
    if rt_quotes and state.positions:
        mtm_prices = {}
        for sym in state.positions:
            rt = rt_quotes.get(sym, {})
            if rt.get("price", 0) > 0:
                mtm_prices[sym] = rt["price"]
        if mtm_prices:
            engine.mark_to_market(mtm_prices)
            logger.info(f"持仓MTM刷新: {len(mtm_prices)}只")

    # 🆕 v2.12: 计算当前行业集中度 (使用最新市值)
    industry_exposure = {}
    for sym, pos in state.positions.items():
        ind = INDUSTRY_MAP.get(sym, "其他")
        industry_exposure[ind] = industry_exposure.get(ind, 0) + pos.market_value
    total_equity = state.total_value

    # 5. 执行买入
    executed = []
    skipped = []
    slots_available = max_positions - len(state.positions)

    for rec in pending:
        if slots_available <= 0:
            skipped.append({"symbol": rec["symbol"], "name": resolve_name(rec["symbol"]),
                          "reason": "已达最大持仓数"})
            continue

        sym = rec["symbol"]
        name = resolve_name(sym)

        # 价格: pending中的close_price > 实时行情 > 实时API > 智能兜底
        price = rec.get("close_price", 0)
        rt_info = rt_quotes.get(sym, {})
        rt_price = rt_info.get("price", 0)
        # v2.10: 优先使用实时价(若偏离<5%)
        if rt_price > 0 and price > 0:
            gap = abs(rt_price / price - 1)
            if gap < 0.05:
                price = rt_price  # 使用实时价
            elif gap < 0.10:
                logger.warning(f"  {name}({sym}): 分析价{price:.2f}实时{rt_price:.2f}偏离{gap:.1%}, 使用分析价")
        elif rt_price > 0:
            price = rt_price
        elif price <= 0:
            price = live_prices.get(sym, 0)
        if price <= 0:
            fb_price, warning = get_fallback_price(sym)
            if fb_price is not None:
                price = fb_price
        if price <= 0:
            continue  # 没有有效价格,跳过

        score = rec.get("composite_score", rec.get("score", 5) * 10)
        rsi = rec.get("rsi", 50)

        # 🆕 v3.1-deerflow 执行管线统一: 优先消费 workflow 已算好的 trade_params
        # (entry/stop/take/position 由代码计算, 消除两条管线各自重算 SL/TP 的漂移)
        tp = rec.get("trade_params", {}) if isinstance(rec, dict) else {}
        use_tp = (isinstance(tp, dict) and tp.get("stop_loss") and tp.get("take_profit")
                  and 0 < tp["stop_loss"] < price < tp["take_profit"])

        # 🆕 v2.13: ATR自适应止损止盈 (替代固定百分比) — trade_params 缺失时回退
        ap_data = analysis_data.get("analysis_prices", {}).get(sym, {})
        atr = ap_data.get("atr_14", 0) if isinstance(ap_data, dict) else 0
        if use_tp:
            stop_loss, take_profit = tp["stop_loss"], tp["take_profit"]
            logger.info(f"  {name}({sym}): 使用 workflow trade_params "
                        f"SL={stop_loss:.2f} TP={take_profit:.2f}")
        else:
            stop_loss, take_profit = calc_atr_stop_loss(price, atr, regime, rsi)

        # 🆕 v2.13: 凯利公式仓位 (替代固定20%) — trade_params 合规仓位优先
        wr = rec.get("win_rate", 0)
        sharpe = rec.get("sharpe", 0)
        kelly_pct = calc_kelly_position_pct(wr, rec.get("conviction", 0),
                                            sharpe=sharpe)
        # 取凯利和市场上限的较小值
        effective_single_pct = min(kelly_pct, regime_single_pct) if kelly_pct > 0 else regime_single_pct
        if use_tp and tp.get("position_pct") and 0 < tp["position_pct"] <= effective_single_pct:
            effective_single_pct = tp["position_pct"]
            logger.info(f"  {name}({sym}): 使用 workflow 仓位 {effective_single_pct:.1%}")
        elif kelly_pct > 0:
            logger.debug(f"  {name}: Kelly={kelly_pct:.1%} WR={wr:.0%} → "
                        f"仓位上限{effective_single_pct:.1%}")

        # 🆕 v2.11: RSI超买过滤
        skip_rsi, rsi_reason = check_rsi_filter(rsi, rec.get("trend_score", 0))
        if skip_rsi:
            skipped.append({"symbol": sym, "name": name, "reason": rsi_reason})
            continue
        elif rsi_reason:
            logger.info(f"  ⚠️ {name}({sym}): {rsi_reason}")

        # 🆕 v2.11: 行业集中度检查 (阈值从30%收紧到25%)
        ind = INDUSTRY_MAP.get(sym, "其他")
        current_ind_pct = industry_exposure.get(ind, 0) / total_equity if total_equity > 0 else 0
        estimated_cost = price * 100  # 至少100股
        new_ind_pct = (industry_exposure.get(ind, 0) + estimated_cost) / total_equity
        if new_ind_pct > MAX_INDUSTRY_PCT and current_ind_pct > 0.15:
            skipped.append({
                "symbol": sym, "name": name,
                "reason": f"{ind}行业占比将达{new_ind_pct:.0%}超{MAX_INDUSTRY_PCT:.0%}上限",
            })
            continue
        # 更新行业敞口 (预估)
        industry_exposure[ind] = industry_exposure.get(ind, 0) + estimated_cost

        # 构建增强推荐
        enhanced_rec = {
            "conviction": rec.get("conviction", 0.5),
            "score": rec.get("score", 0),
            "composite_score": score,
            "key_reasons": rec.get("key_reasons", []),
            "risks": rec.get("risks", []),
            "verdict_summary": rec.get("verdict_summary", ""),
            "strategy_id": rec.get("strategy_id", ""),
            "strategy_name": rec.get("strategy_name", ""),
            "win_rate": rec.get("win_rate", 0),
            "sharpe": rec.get("sharpe", 0),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        if dry_run:
            logger.info(f"[DRY-RUN] {name}({sym}): score={score:.0f} conf={rec.get('conviction',0):.0%} "
                       f"@ {price:.2f} SL={stop_loss:.2f} TP={take_profit:.2f} "
                       f"{rec.get('strategy_name', '')}")
            executed.append({"symbol": sym, "name": name, "price": price,
                           "conviction": rec.get("conviction", 0), "dry_run": True})
            slots_available -= 1
            continue

        trade = engine.execute_buy(
            symbol=sym,
            name=name,
            price=price,
            recommendation=enhanced_rec,
            max_position_pct=effective_single_pct,  # v2.13: Kelly+regime联合决定
            max_positions=max_positions,
        )

        if trade:
            executed.append({
                "symbol": sym, "name": name,
                "quantity": trade.quantity, "price": trade.price,
                "amount": trade.amount,
                "conviction": rec.get("conviction", 0),
                "score": score,
                "strategy": rec.get("strategy_name", ""),
                "sl": stop_loss, "tp": take_profit,
            })
            slots_available -= 1
        else:
            skipped.append({"symbol": sym, "name": name, "reason": "资金不足/已达上限"})

    # 6. 汇总
    total_amount = sum(e.get("amount", 0) for e in executed)
    state = engine.state

    print("\n" + "=" * 60)
    print(f"Morning Buy Summary | {date.today().isoformat()}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'} | Candidates: {len(pending)} -> Executed: {len(executed)}")
    print("-" * 60)

    if executed:
        for e in executed:
            if e.get("dry_run"):
                print(f"  [DRY-RUN] {e['name']}({e['symbol']}) conf={e['conviction']:.0%} "
                      f"score={e.get('score',0):.0f} @{e['price']:.2f}")
            else:
                print(f"  [OK] {e['name']}({e['symbol']}) {e['quantity']}sh @{e['price']:.2f} "
                      f"amt={e['amount']:,.0f} | SL={e.get('sl',0):.2f} TP={e.get('tp',0):.2f}")
                if e.get('strategy'):
                    print(f"       Strategy: {e['strategy']}")

    if skipped:
        for s in skipped[:5]:
            print(f"  [SKIP] {s['name']}({s['symbol']}): {s['reason']}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped)-5} more")

    # 清空已处理的推荐
    if not dry_run:
        state.pending_recommendations = []
        manager.save()

    print(f"\nCash: {state.cash:,.2f} | Positions: {len(state.positions)} | "
          f"Total: {state.total_value:,.2f}")
    print(f"Return: {state.total_return:+,.2f} ({state.total_return_pct:+.2f}%)")
    print("=" * 60)

    return {
        "status": "ok",
        "date": date.today().isoformat(),
        "dry_run": dry_run,
        "candidates": len(pending),
        "executed": len(executed),
        "skipped": len(skipped),
        "total_amount": total_amount,
        "remaining_cash": round(state.cash, 2),
        "total_value": round(state.total_value, 2),
    }


async def main():
    parser = argparse.ArgumentParser(description="A股模拟交易 -- 早盘自动买入 v2.1")
    parser.add_argument("--min-confidence", type=float, default=0.45,
                       help="最低置信度阈值 (默认0.50)")
    parser.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS,
                       help=f"最大持仓数 (默认{DEFAULT_MAX_POSITIONS})")
    parser.add_argument("--dry-run", action="store_true",
                       help="试运行模式")
    args = parser.parse_args()

    await run_morning_buy(
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
        max_positions=args.max_positions,
    )


if __name__ == "__main__":
    asyncio.run(main())
