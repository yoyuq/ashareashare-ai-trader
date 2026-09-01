"""Tab: 🧪 策略回测 — 单标的策略回测对比"""
from datetime import date, timedelta

import streamlit as st
import plotly.graph_objects as go

from data.symbols import to_symbol
from scripts.shared import NAME_MAP
from web.data import _run_async, knowledge, load_full_market_stocks, router


def render():
    st.subheader("策略回测")

    strats = knowledge.list_strategies()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sid = st.selectbox("策略", [s["id"] for s in strats],
                           format_func=lambda x: next((s["name"] for s in strats if s["id"] == x), x))
    with c2:
        fm_stocks = load_full_market_stocks()
        fm_list = sorted(fm_stocks.keys())
        bts = st.selectbox("标的 (全市场)", fm_list,
                           format_func=lambda x: f"{x} | {NAME_MAP.get(x, '')}",
                           help=f"共 {len(fm_list):,} 只")
    sym = fm_stocks.get(bts, to_symbol(bts))
    with c3:
        yrs = st.slider("回测年数", 1, 5, 3)
    with c4:
        if st.button("▶ 运行", type="primary", use_container_width=True):
            with st.spinner("回测中..."):
                try:
                    from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
                    from data.providers.base import DataRequest, DataFrequency
                    from analysis.recommender import STRATEGY_BACKTESTERS

                    async def bt():
                        today = date.today()
                        req = DataRequest(sym, today - timedelta(days=365 * yrs), today, DataFrequency.DAILY)
                        data = await router.get_daily_kline(req)
                        df = data.data
                        cfg = BacktestConfig(100000, df["date"].min(), df["date"].max())
                        eng = EventDrivenBacktestEngine(cfg)
                        eng.load_data(sym, df)
                        bt_func = STRATEGY_BACKTESTERS.get(sid)
                        sbt = bt_func(df) if bt_func else {}

                        def strat(today_, bars, broker):
                            if sym not in bars:
                                return
                            c = bars[sym]["close"]
                            if not hasattr(strat, "_e"):
                                strat._e = None
                            if strat._e is None:
                                q = int(broker.account.cash * 0.3 / c / 100) * 100
                                if q >= 100:
                                    broker.buy(sym, q)
                                    strat._e = today_
                            else:
                                p = broker.account.positions.get(sym)
                                if p and (today_ - strat._e).days >= 10:
                                    broker.sell(sym, p.quantity)
                                    strat._e = None
                        strat._e = None
                        return eng.run(strat, progress_bar=False), sbt

                    r, sbt = _run_async(bt())

                    # Results
                    pc = st.columns(4)
                    pc[0].metric("总收益", f"{r.total_return:+.1f}%")
                    pc[1].metric("年化", f"{r.annual_return:+.1f}%")
                    pc[2].metric("夏普", f"{r.sharpe_ratio:.2f}")
                    pc[3].metric("最大回撤", f"{r.max_drawdown:.1f}%")
                    pc2 = st.columns(4)
                    pc2[0].metric("索提诺", f"{r.sortino_ratio:.2f}")
                    pc2[1].metric("胜率", f"{r.win_rate:.1%}")
                    pc2[2].metric("VaR95", f"{r.var_95:.1f}%")
                    pc2[3].metric("Avg PnL", f"{r.avg_trade_pnl:+.1f}%" if hasattr(r, 'avg_trade_pnl') else "N/A")

                    # Equity curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=r.equity_curve, name="Equity",
                                             line=dict(color="#58a6ff", width=2), fill="tozeroy",
                                             fillcolor="rgba(88,166,255,0.1)"))
                    fig.update_layout(height=300, template="plotly_white",
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    if sbt:
                        st.caption(f"Strategy Stats: {sbt.get('signals', 0)} 个信号 | "
                                   f"胜率 {sbt.get('win_rate', 0):.1%} | "
                                   f"PF {sbt.get('profit_factor', 1):.1f}x | "
                                   f"Sharpe {sbt.get('sharpe', 0):.2f} | "
                                   f"含交易成本: {sbt.get('cost_adjusted', False)}")
                except Exception as e:
                    st.error(str(e))
