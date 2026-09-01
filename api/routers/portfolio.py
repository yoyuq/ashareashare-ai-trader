"""APIRouter: 风控/组合/交易统计/持仓实时估值 (v6.0 拆分自 server.py)"""
import asyncio
import json
from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1", tags=["portfolio"])

# 项目根 (api/routers/portfolio.py → 上三级)
_ROOT = Path(__file__).parent.parent.parent


@router.get("/risk/status")
async def risk_status():
    """获取当前组合风控状态 (回撤/波动/断路器/8层风控)"""
    try:
        from simulation.portfolio import PortfolioManager
        from analysis.risk_controls import PortfolioRiskManager
        import pandas as pd

        manager = PortfolioManager()
        state = manager.state
        risk_mgr = PortfolioRiskManager(initial_capital=state.initial_capital)

        # 构建日收益率
        daily_rets = pd.Series(dtype=float)
        if len(state.daily_snapshots) >= 5:
            daily_rets = pd.Series([
                s.daily_return_pct / 100 for s in state.daily_snapshots[-60:]
            ])

        risk_state = risk_mgr.update(state.total_value, daily_rets) if len(daily_rets) > 0 else None

        # 8层风控逐层检查
        layers = {}
        # L1
        layers["L1_circuit_breaker"] = {
            "active": risk_state.circuit_breaker_active if risk_state else False,
            "drawdown_pct": risk_state.drawdown_pct if risk_state else 0,
        }
        # L2-L4 (v3.0: 真实接线状态, 不再硬编码)
        stop_positions = sum(1 for p in state.positions.values() if p.stop_loss > 0)
        layers["L2_atr_stop"] = {
            "enforced": len(state.positions) > 0,
            "positions_with_stop": stop_positions,
            "note": "动态止损经 check_exit_conditions 执行",
        }
        layers["L3_trailing_stop"] = {
            "enforced": len(state.positions) > 0,
            "note": "trailing stop 经 update_dynamic_stops 只上移",
        }
        layers["L4_position_limits"] = {
            "enforced": True,
            "max_positions": 8,
            "single_pct": 0.20,
            "note": "paper_trader.execute_buy 执行持仓数/单票上限",
        }

        # L5: 防踩踏
        if state.positions:
            losers = sum(1 for p in state.positions.values() if p.unrealized_pnl_pct < -3)
            layers["L5_stampede"] = {
                "triggered": losers / len(state.positions) >= 0.5 if state.positions else False,
                "losers": losers,
                "total": len(state.positions),
            }
        else:
            layers["L5_stampede"] = {"triggered": False, "note": "no positions"}

        # L6: 持仓天数
        max_days = 0
        for p in state.positions.values():
            if p.buy_date:
                try:
                    days = (date.today() - date.fromisoformat(p.buy_date)).days
                    max_days = max(max_days, days)
                except (ValueError, TypeError):
                    pass
        layers["L6_holding_days"] = {"max_days": max_days, "warning": max_days >= 20}

        # L7-L8 (v3.0: 真实接线状态)
        layers["L7_limit_down"] = {
            "enforced": True,
            "note": "paper_trader.execute_sell 跌停封板拒卖 (板块感知)",
        }
        layers["L8_correlation"] = {
            "enforced": False,
            "note": "组合相关性分析未实现",
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio": {
                "total_value": round(state.total_value, 2),
                "cash": round(state.cash, 2),
                "position_value": round(state.position_value, 2),
                "total_return_pct": round(state.total_return_pct, 2),
                "positions_count": len(state.positions),
                "trade_count": state.trade_count,
                "win_rate": round(state.win_rate * 100, 1),
            },
            "risk": {
                "drawdown_pct": risk_state.drawdown_pct if risk_state else 0,
                "volatility_pct": risk_state.current_volatility if risk_state else 0,
                "risk_multiplier": risk_state.risk_multiplier if risk_state else 1.0,
                "circuit_breaker_active": risk_state.circuit_breaker_active if risk_state else False,
                "crisis_mode": risk_state.crisis_mode if risk_state else False,
                "suggested_exposure_pct": risk_state.suggested_exposure_pct if risk_state else 0.3,
                "warnings": risk_state.warning_messages if risk_state else [],
            },
            "layers": layers,
        }
    except Exception as e:
        raise HTTPException(500, f"风控状态查询失败: {e}")


@router.get("/portfolio/summary")
async def portfolio_summary():
    """获取模拟交易组合摘要 (含持仓/交易/绩效)"""
    try:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine

        manager = PortfolioManager()
        engine = PaperTradingEngine(manager)
        return engine.get_summary()
    except Exception as e:
        raise HTTPException(500, f"组合摘要查询失败: {e}")


@router.get("/trades/stats")
async def trade_statistics():
    """获取交易绩效统计 (WinRateAnalyzer — 胜率/盈亏比/分场景/信号质量)"""
    try:
        from simulation.portfolio import PortfolioManager
        from analysis.winrate import WinRateAnalyzer
        import pandas as pd

        manager = PortfolioManager()
        state = manager.state

        if not state.trade_history:
            return {"total_trades": 0, "note": "无交易记录"}

        # 转换交易记录
        trades_df = pd.DataFrame([
            {
                "date": t.date, "symbol": t.symbol, "side": t.side,
                "price": t.price, "pnl": t.pnl or 0,
                "pnl_pct": t.pnl_pct or 0, "holding_days": 1,
            }
            for t in state.trade_history
            if t.side == "sell" and t.pnl is not None
        ])

        if trades_df.empty:
            return {"total_trades": 0, "note": "无已平仓交易"}

        analyzer = WinRateAnalyzer()
        report = analyzer.analyze(trades_df, pd.DataFrame())

        return {
            "total_trades": report.total_signals,
            "win_count": report.total_win,
            "win_rate": round(report.overall_win_rate * 100, 1),
            "profit_factor": report.profit_factor,
            "expected_value_pct": report.expected_value,
            "avg_holding_days": round(report.avg_holding_days, 1),
            "max_consecutive_loss": report.max_consecutive_loss,
            "signal_efficiency": report.signal_efficiency,
            "scenario_breakdown": {
                k: {
                    "win_rate": round(v["win_rate"] * 100, 1),
                    "trades": v["total"],
                    "profit_factor": v["profit_factor"],
                }
                for k, v in report.scenario_breakdown.items()
                if v["total"] > 0
            },
        }
    except Exception as e:
        raise HTTPException(500, f"交易统计查询失败: {e}")


@router.get("/portfolio/mtm")
async def portfolio_mtm():
    """持仓实时估值 — 获取当前持仓的实时价格并计算浮动盈亏"""
    import requests

    try:
        # 加载持仓
        portfolio_path = _ROOT / "simulation_data" / "portfolio.json"
        if not portfolio_path.exists():
            return {"error": "portfolio not found", "positions": [], "summary": {}}

        with open(portfolio_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        positions = data.get("positions", {})
        if not positions:
            return {"positions": [], "summary": {"total_value": data.get("account", {}).get("cash", 100000),
                     "cash": data.get("account", {}).get("cash", 100000),
                     "total_return": 0, "total_return_pct": 0}}

        # 获取实时价格 (腾讯行情)
        symbols = list(positions.keys())
        tc_codes = [s.replace("sh.", "sh").replace("sz.", "sz") for s in symbols]
        url = f"https://qt.gtimg.cn/q={','.join(tc_codes)}"

        prices = {}
        names = {}
        try:
            resp = await asyncio.to_thread(
                requests.get, url, timeout=5,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"},
            )
            resp.encoding = "gbk"
            for line in resp.text.strip().split("\n"):
                if "=" not in line or "~" not in line:
                    continue
                try:
                    fields = line.split("=", 1)[1].strip('"').split("~")
                    if len(fields) < 10:
                        continue
                    code = fields[2]
                    for sym in symbols:
                        if sym.endswith(code):
                            price = float(fields[3]) if fields[3] else 0
                            prices[sym] = price
                            names[sym] = fields[1]
                except (ValueError, IndexError):
                    continue
        except Exception:
            pass

        # 计算浮动盈亏
        account = data.get("account", {})
        cash = account.get("cash", 100000.0)
        total_cost_all = 0
        total_mv_all = 0
        pos_list = []

        for sym, pos in positions.items():
            cur_price = prices.get(sym, pos.get("current_price", pos.get("avg_cost", 0)))
            qty = pos.get("quantity", 0)
            avg_cost = pos.get("avg_cost", 0)
            total_cost = pos.get("total_cost", qty * avg_cost)
            mv = qty * cur_price
            pnl = mv - total_cost
            pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0

            total_cost_all += total_cost
            total_mv_all += mv

            pos_list.append({
                "symbol": sym,
                "name": names.get(sym, pos.get("name", "")),
                "quantity": qty,
                "avg_cost": round(avg_cost, 2),
                "current_price": round(cur_price, 2),
                "market_value": round(mv, 2),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "stop_loss": pos.get("stop_loss", 0),
                "take_profit": pos.get("take_profit", 0),
                "cost": round(total_cost, 2),
            })

        # 排序: 盈亏降序
        pos_list.sort(key=lambda x: x["unrealized_pnl"], reverse=True)

        total_value = cash + total_mv_all
        initial_capital = data.get("meta", {}).get("initial_capital", 100000)
        total_return = total_value - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0

        return {
            "positions": pos_list,
            "summary": {
                "total_value": round(total_value, 2),
                "cash": round(cash, 2),
                "position_value": round(total_mv_all, 2),
                "total_cost": round(total_cost_all, 2),
                "total_return": round(total_return, 2),
                "total_return_pct": round(total_return_pct, 2),
                "position_count": len(pos_list),
                "win_count": sum(1 for p in pos_list if p["unrealized_pnl"] > 0),
                "loss_count": sum(1 for p in pos_list if p["unrealized_pnl"] < 0),
            },
            "ts": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "positions": [], "summary": {}}
