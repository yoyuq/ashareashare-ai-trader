"""
A股智能分析Agent Dashboard v2.14 — Professional Dark Theme
"""
import asyncio, concurrent.futures, sys, os, json
from datetime import date, datetime, timedelta
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

VERSION = "2.14.0"

# ═══════════════════════════════════════════════════════════════
# Professional Dark Theme CSS
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="AShare AI Trader", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Dark theme base */
    .stApp { background-color: #0d1117; }
    .main { background-color: #0d1117; color: #c9d1d9; }
    header { background-color: #0d1117; }
    /* Metric cards */
    [data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px 16px; }
    [data-testid="stMetric"] label { color: #8b949e; font-size: 12px; }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #58a6ff; font-size: 24px; }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    /* Tabs */
    button[data-baseweb="tab"] { color: #8b949e !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important; }
    /* Expanders */
    [data-testid="stExpander"] { background-color: #161b22; border: 1px solid #30363d;
        border-radius: 8px; }
    /* DataFrames */
    [data-testid="stTable"] { background-color: #161b22; }
    .stDataFrame { background-color: #161b22; }
    /* Info/Warning/Success */
    [data-testid="stInfo"] { background-color: #1a2332; color: #58a6ff; }
    [data-testid="stWarning"] { background-color: #2d1f1a; color: #f0883e; }
    [data-testid="stSuccess"] { background-color: #1a2f1a; color: #3fb950; }
    [data-testid="stError"] { background-color: #2d1a1a; color: #f85149; }
    /* General */
    h1, h2, h3 { color: #f0f6fc !important; }
    p, span { color: #c9d1d9; }
    .stCaption { color: #8b949e; }
    /* Status bar */
    .status-bar { background-color: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 8px 16px; margin-bottom: 12px;
        display: flex; align-items: center; gap: 20px; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .status-dot.green { background-color: #3fb950; }
    .status-dot.yellow { background-color: #f0883e; }
    .status-dot.red { background-color: #f85149; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════
def _run_async(coro):
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)

@st.cache_data(ttl=3600, show_spinner=False)
def load_stocks():
    s = {}
    try:
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "symbols.yaml", encoding="utf-8"))
        for sym in cfg.get("watchlist", {}).get("default", []):
            s[sym.replace("sh.","").replace("sz.","")] = sym
    except: pass
    if len(s) < 5:
        s.update({"600519":"sh.600519","300750":"sz.300750","600036":"sh.600036",
                  "002594":"sz.002594","000858":"sz.000858","601318":"sh.601318"})
    return s

@st.cache_resource(show_spinner=False)
def init_components():
    c = {}
    try: from data.router import get_data_router; c["router"] = get_data_router()
    except: c["router_err"] = "init failed"
    try: from analysis.indicators import TechnicalAnalyzer; c["analyzer"] = TechnicalAnalyzer()
    except: pass
    try: from analysis.regime import MarketRegimeDetector; c["detector"] = MarketRegimeDetector()
    except: pass
    try: from knowledge.manager import KnowledgeManager; c["knowledge"] = KnowledgeManager()
    except: pass
    return c

stocks = load_stocks()
comps = init_components()
router = comps.get("router")
analyzer = comps.get("analyzer")
detector = comps.get("detector")
knowledge = comps.get("knowledge")
today = date.today()

# ═══════════════════════════════════════════════════════════════
# Sidebar — System Status + Settings
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 AShare AI Trader")
    st.caption(f"v{VERSION}")

    # Status indicators
    ollama_ok = os.path.exists(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"))
    ds_ok = "sk-" in os.getenv("DEEPSEEK_API_KEY", "")
    db_ok = os.getenv("POSTGRES_HOST", "")

    st.markdown(f"""
    <div style="background-color:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin:8px 0">
    <span style="color:#8b949e;font-size:12px">SYSTEM STATUS</span><br>
    <span style="color:{'#3fb950' if ds_ok else '#f85149'}">●</span> DeepSeek {'ON' if ds_ok else 'OFF'}<br>
    <span style="color:{'#3fb950' if ollama_ok else '#8b949e'}">●</span> Ollama {'ON' if ollama_ok else 'OFF'}<br>
    <span style="color:{'#3fb950' if db_ok else '#8b949e'}">●</span> Database {'ON' if db_ok else 'OFF'}
    </div>
    """, unsafe_allow_html=True)

    # Mode selector
    st.markdown("**NAVIGATION**")
    tab = st.radio("", ["📊 Overview", "🔍 Scanner", "📈 Technical", "🧪 Backtest",
                        "📋 Signals", "💰 Portfolio", "🛡️ Risk"], label_visibility="collapsed")

    st.divider()
    st.caption("Auto-refresh: 60s")
    st.caption(f"Last: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh All"):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# Data Helpers
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_regime():
    try:
        from data.providers.base import DataFrequency, DataRequest
        async def f():
            req = DataRequest("sh.000300", today - timedelta(days=365), today, DataFrequency.DAILY)
            r = await router.get_daily_kline(req)
            return detector.detect(r.data)
        return _run_async(f())
    except: return None

@st.cache_data(ttl=120, show_spinner=False)
def get_stock_quick(sym, days=90):
    try:
        from data.providers.base import DataFrequency, DataRequest
        async def f():
            req = DataRequest(sym, today - timedelta(days=days), today, DataFrequency.DAILY)
            r = await router.get_daily_kline(req)
            d = r.data
            if d.empty: return None
            ind = analyzer.compute_all(d, symbol=sym)
            return {"data": d, "indicators": ind, "last": ind.to_dataframe().iloc[-1],
                    "close": float(d["close"].iloc[-1]),
                    "change": float(d.get("pct_change", pd.Series([0])).iloc[-1]) if "pct_change" in d.columns else 0}
        return _run_async(f())
    except: return None

@st.cache_data(ttl=60, show_spinner=False)
def get_portfolio_state():
    try:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        m = PortfolioManager()
        e = PaperTradingEngine(m)
        return e.get_summary()
    except: return None


# ═══════════════════════════════════════════════════════════════
# Tab 1: OVERVIEW — Market Status + Watchlist
# ═══════════════════════════════════════════════════════════════
if tab == "📊 Overview":
    regime = get_regime()

    # --- Status Bar ---
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    if regime:
        emoji = {"strong_bull":"🟢","weak_bull":"🟡","range_bound":"⚪","weak_bear":"🟠","strong_bear":"🔴","crisis":"💀"}
        sc1.metric("Market Regime", f"{emoji.get(regime.regime.value,'?')} {regime.regime.value}")
        sc2.metric("Confidence", f"{regime.confidence:.0%}")
        sc3.metric("Price Structure", f"{regime.scores.get('price_structure',0):.2f}")
        sc4.metric("Momentum", f"{regime.scores.get('momentum',0):.2f}")
        sc5.metric("Volatility", f"{regime.scores.get('volatility',0):.2f}")
        sc6.metric("Volume", f"{regime.scores.get('volume',0):.2f}")

    # --- Watchlist Table ---
    syms = list(stocks.values())[:20]
    watch_data = []
    for sym in syms:
        info = get_stock_quick(sym, 30)
        if info:
            watch_data.append({
                "Code": sym.replace("sh.","").replace("sz.",""),
                "Name": info.get("name", ""),
                "Price": info["close"],
                "Change%": info.get("change", 0),
                "RSI": info["last"].get("rsi_14", 50),
                "Trend": info["last"].get("trend_score", 0),
                "Composite": info["last"].get("composite_score", 50),
                "Vol Ratio": info["last"].get("vol_ratio_5", 1),
            })

    if watch_data:
        df_watch = pd.DataFrame(watch_data)
        df_watch["Signal"] = df_watch.apply(lambda r: "🟢" if r["Composite"] > 65 else ("🟡" if r["Composite"] > 50 else "🔴"), axis=1)

        st.subheader("Watchlist")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.dataframe(df_watch.sort_values("Composite", ascending=False),
                        column_config={
                            "Change%": st.column_config.NumberColumn(format="%+.2f%%"),
                            "RSI": st.column_config.NumberColumn(format="%.0f"),
                            "Trend": st.column_config.NumberColumn(format="%.2f"),
                            "Composite": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                        }, height=400, hide_index=True, use_container_width=True)
        with c2:
            # Top movers
            top_up = df_watch.nlargest(3, "Change%")
            top_down = df_watch.nsmallest(3, "Change%")
            st.caption("Top Gainers")
            for _, r in top_up.iterrows():
                st.metric(r["Code"], f"¥{r['Price']:.2f}", f"+{r['Change%']:.2f}%")
            st.caption("Top Losers")
            for _, r in top_down.iterrows():
                st.metric(r["Code"], f"¥{r['Price']:.2f}", f"{r['Change%']:.2f}%")

    # --- Strategy Recommendations ---
    if regime:
        strats = knowledge.get_strategies_for_regime(regime.regime.value)
        if strats:
            st.subheader("Recommended Strategies")
            scols = st.columns(min(len(strats), 4))
            for i, s in enumerate(strats[:4]):
                scols[i].info(f"**{s['name']}**  \n`{s['id']}`  \n{s.get('category','')}")


# ═══════════════════════════════════════════════════════════════
# Tab 2: SCANNER — Market-wide opportunity scanning
# ═══════════════════════════════════════════════════════════════
elif tab == "🔍 Scanner":
    st.subheader("Market Scanner")

    c1, c2, c3 = st.columns(3)
    with c1: min_score = st.slider("Min Composite Score", 40, 90, 55)
    with c2: top_n = st.slider("Top N", 5, 50, 15)
    with c3:
        if st.button("🔍 Scan Market", type="primary", use_container_width=True):
            with st.spinner("Scanning..."):
                try:
                    from analysis.recommender import RecommendationEngine
                    engine = RecommendationEngine(router=router, knowledge=knowledge, analyzer=analyzer)
                    async def g(): return await engine.generate(capital=100000, top_n=top_n, min_composite=min_score)
                    recs = _run_async(g())

                    if recs.recommendations:
                        st.success(f"{recs.total_scanned} stocks scanned → {len(recs.recommendations)} opportunities")

                        # Signal Cards
                        cols = st.columns(3)
                        for i, r in enumerate(recs.recommendations[:9]):
                            with cols[i % 3]:
                                grade_color = {"A":"#3fb950","B":"#58a6ff","C":"#f0883e","D":"#f85149"}.get(r.confidence, "#8b949e")
                                st.markdown(f"""
                                <div style="background-color:#161b22;border:1px solid #30363d;border-left:3px solid {grade_color};
                                    border-radius:8px;padding:12px;margin:4px 0">
                                    <span style="color:{grade_color};font-weight:bold">{r.confidence} | {r.symbol_name}</span><br>
                                    <span style="color:#8b949e;font-size:12px">{r.strategy_name}</span><br>
                                    <span style="color:#c9d1d9">Entry: ¥{r.entry_price:.2f} | SL: ¥{r.stop_loss:.2f}</span><br>
                                    <span style="color:#58a6ff">Win Rate: {r.win_rate:.0%} | PF: {r.profit_factor:.1f}x</span><br>
                                    <span style="color:#8b949e;font-size:11px">{r.key_reason[:100]}</span>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No opportunities found matching criteria")
                except Exception as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════
# Tab 3: TECHNICAL — Candlestick + Indicators deep dive
# ═══════════════════════════════════════════════════════════════
elif tab == "📈 Technical":
    st.subheader("Technical Analysis")

    c1, c2 = st.columns([1, 3])
    with c1:
        selected_sym = st.selectbox("Symbol", list(stocks.keys()),
                                    format_func=lambda x: f"{x} ({'sh' if x.startswith('6') else 'sz'})")
    sym = stocks.get(selected_sym, f"sh.{selected_sym}" if selected_sym.startswith("6") else f"sz.{selected_sym}")

    info = get_stock_quick(sym, 365)
    if info:
        df = info["data"]
        ind = info["indicators"]
        last = info["last"]

        # --- KPI Row ---
        mc = st.columns(8)
        mc[0].metric("Price", f"¥{info['close']:.2f}", f"{info.get('change',0):+.2f}%")
        mc[1].metric("RSI(14)", f"{last.get('rsi_14',0):.0f}")
        mc[2].metric("Trend", f"{last.get('trend_score',0):.2f}")
        mc[3].metric("Composite", f"{last.get('composite_score',0):.0f}")
        mc[4].metric("Vol Ratio", f"{last.get('vol_ratio_5',0):.1f}")
        mc[5].metric("MACD", f"{last.get('macd_hist',0):.3f}")
        mc[6].metric("ATR(14)", f"{last.get('atr_14',0):.2f}")
        mc[7].metric("Bias MA20", f"{last.get('bias_ma20',0):.1f}%")

        # --- Candlestick Chart ---
        n = min(120, len(df))
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           row_heights=[0.5, 0.25, 0.25],
                           vertical_spacing=0.05,
                           subplot_titles=("Price & MA", "Volume", "RSI"))

        # Price + MAs
        fig.add_trace(go.Candlestick(
            x=df["date"].values[-n:], open=df["open"].values[-n:],
            high=df["high"].values[-n:], low=df["low"].values[-n:],
            close=df["close"].values[-n:], name="Price",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149"), row=1, col=1)
        for ma, color, label in [("ma_5","#f0883e","MA5"),("ma_20","#58a6ff","MA20"),("ma_60","#a371f7","MA60")]:
            if ma in ind.indicators:
                fig.add_trace(go.Scatter(x=df["date"].values[-n:],
                    y=ind.indicators[ma].values[-n:], name=label,
                    line=dict(color=color, width=1)), row=1, col=1)

        # Volume
        colors_v = ["#3fb950" if df["close"].values[i] >= df["open"].values[i] else "#f85149" for i in range(-n, 0)]
        fig.add_trace(go.Bar(x=df["date"].values[-n:], y=df["volume"].values[-n:],
                            name="Volume", marker_color=colors_v, showlegend=False), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df["date"].values[-n:], y=ind.indicators["rsi_14"].values[-n:],
                                name="RSI(14)", line=dict(color="#a371f7", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f85149", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", opacity=0.5, row=3, col=1)

        fig.update_layout(height=550, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0),
                         hovermode="x unified", showlegend=True,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Indicator Dashboard ---
        st.subheader("Indicator Dashboard")
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.caption("Trend")
            st.metric("MA Alignment", "Bullish" if last.get("ma_bullish",0) else "Bearish")
            st.metric("ADX(14)", f"{last.get('adx_14',0):.1f}")
            st.metric("+DI/-DI", f"{last.get('plus_di_14',0):.1f}/{last.get('minus_di_14',0):.1f}")
        with ic2:
            st.caption("Momentum")
            st.metric("RSI(14)", f"{last.get('rsi_14',0):.0f}")
            st.metric("MACD Hist", f"{last.get('macd_hist',0):.3f}")
            st.metric("ROC(5)", f"{last.get('roc_5',0):.1f}%")
        with ic3:
            st.caption("Volatility")
            st.metric("ATR(14)", f"{last.get('atr_14',0):.2f}")
            st.metric("BB Width", f"{last.get('bb_width_20',0):.2f}")
            st.metric("HV(20)", f"{last.get('hv_20',0):.1f}%")
        with ic4:
            st.caption("Volume")
            st.metric("Vol Ratio(5)", f"{last.get('vol_ratio_5',0):.2f}")
            st.metric("OBV Trend", "Up" if last.get("obv",0) > last.get("obv_ma_20",0) else "Down")
            st.metric("Turnover", f"{last.get('turnover',0):.1f}%")

        # Patterns
        patterns = [k for k, v in ind.patterns.items() if hasattr(v, 'iloc') and v.iloc[-1] > 0]
        if patterns:
            st.caption(f"Patterns: {' | '.join(p[:6] for p in patterns)}")
    else:
        st.warning("No data available for this symbol")


# ═══════════════════════════════════════════════════════════════
# Tab 4: BACKTEST — Strategy comparison
# ═══════════════════════════════════════════════════════════════
elif tab == "🧪 Backtest":
    st.subheader("Strategy Backtest")

    strats = knowledge.list_strategies()
    c1, c2, c3, c4 = st.columns(4)
    with c1: sid = st.selectbox("Strategy", [s["id"] for s in strats],
                                format_func=lambda x: next((s["name"] for s in strats if s["id"]==x), x))
    with c2: bts = st.selectbox("Symbol", list(stocks.keys()))
    sym = stocks.get(bts, f"sh.{bts}" if bts.startswith("6") else f"sz.{bts}")
    with c3: yrs = st.slider("Years", 1, 5, 3)
    with c4:
        if st.button("▶ Run", type="primary", use_container_width=True):
            with st.spinner("Backtesting..."):
                try:
                    from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
                    from data.providers.base import DataRequest, DataFrequency
                    from analysis.recommender import STRATEGY_BACKTESTERS

                    async def bt():
                        req = DataRequest(sym, today - timedelta(days=365*yrs), today, DataFrequency.DAILY)
                        data = await router.get_daily_kline(req)
                        df = data.data
                        cfg = BacktestConfig(100000, df["date"].min(), df["date"].max())
                        eng = EventDrivenBacktestEngine(cfg)
                        eng.load_data(sym, df)
                        bt_func = STRATEGY_BACKTESTERS.get(sid)
                        sbt = bt_func(df) if bt_func else {}

                        def strat(today_, bars, broker):
                            if sym not in bars: return
                            c = bars[sym]["close"]
                            if not hasattr(strat,"_e"): strat._e = None
                            if strat._e is None:
                                q = int(broker.account.cash * 0.3 / c / 100) * 100
                                if q >= 100: broker.buy(sym, q, price=c); strat._e = today_
                            else:
                                p = broker.account.positions.get(sym)
                                if p and (today_ - strat._e).days >= 10:
                                    broker.sell(sym, p.quantity, price=c); strat._e = None
                        strat._e = None
                        return eng.run(strat, progress_bar=False), sbt

                    r, sbt = _run_async(bt())

                    # Results
                    pc = st.columns(4)
                    pc[0].metric("Total Return", f"{r.total_return:+.1f}%")
                    pc[1].metric("Annual", f"{r.annual_return:+.1f}%")
                    pc[2].metric("Sharpe", f"{r.sharpe_ratio:.2f}")
                    pc[3].metric("Max DD", f"{r.max_drawdown:.1f}%")
                    pc2 = st.columns(4)
                    pc2[0].metric("Sortino", f"{r.sortino_ratio:.2f}")
                    pc2[1].metric("Win Rate", f"{r.win_rate:.1%}")
                    pc2[2].metric("VaR 95", f"{r.var_95:.1f}%")
                    pc2[3].metric("Avg PnL", f"{r.avg_trade_pnl:+.1f}%" if hasattr(r, 'avg_trade_pnl') else "N/A")

                    # Equity curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=r.equity_curve, name="Equity",
                        line=dict(color="#58a6ff", width=2), fill="tozeroy",
                        fillcolor="rgba(88,166,255,0.1)"))
                    fig.update_layout(height=300, template="plotly_dark",
                                    margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    if sbt:
                        st.caption(f"Strategy Stats: {sbt.get('signals',0)} signals | "
                                  f"Win {sbt.get('win_rate',0):.1%} | "
                                  f"PF {sbt.get('profit_factor',1):.1f}x | "
                                  f"Sharpe {sbt.get('sharpe',0):.2f} | "
                                  f"Cost adjusted: {sbt.get('cost_adjusted',False)}")
                except Exception as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════
# Tab 5: SIGNALS — AI Recommendations
# ═══════════════════════════════════════════════════════════════
elif tab == "📋 Signals":
    st.subheader("AI Signal Feed")

    # Chat-style signal generation
    if "signal_chat" not in st.session_state:
        st.session_state.signal_chat = []

    for msg in st.session_state.signal_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask AI: e.g. 'Scan for buy signals' or 'Analyze 600519 with MACD strategy'"):
        st.session_state.signal_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    from agent.chat_agent import ChatAgent
                    agent = ChatAgent(router=router, knowledge=knowledge, analyzer=analyzer)
                    async def c(): return await agent.chat(prompt, session_id="dashboard")
                    reply = _run_async(c())
                    st.markdown(reply)
                    st.session_state.signal_chat.append({"role": "assistant", "content": reply})
                except Exception as e: st.error(str(e))

    if st.session_state.signal_chat:
        if st.button("Clear History"):
            st.session_state.signal_chat = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# Tab 6: PORTFOLIO — Paper Trading Dashboard
# ═══════════════════════════════════════════════════════════════
elif tab == "💰 Portfolio":
    st.subheader("Portfolio Dashboard")

    portfolio = get_portfolio_state()
    if not portfolio:
        st.info("No portfolio data. Run `python scripts/morning_buy.py` to start.")
    else:
        # --- KPI Row ---
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        ret_emoji = "🟢" if portfolio["total_return"] >= 0 else "🔴"
        k1.metric("Total Assets", f"¥{portfolio['total_value']:,.0f}")
        k2.metric(f"{ret_emoji} Return", f"{portfolio['total_return_pct']:+.2f}%",
                 f"¥{portfolio['total_return']:+,.0f}")
        k3.metric("Cash", f"¥{portfolio['cash']:,.0f}",
                 f"{portfolio['cash']/max(portfolio['total_value'],1)*100:.0f}%")
        k4.metric("Positions", str(portfolio["position_count"]),
                 f"¥{portfolio['position_value']:,.0f}")
        k5.metric("Win Rate", f"{portfolio['win_rate']:.1f}%",
                 f"{portfolio['win_count']}W/{portfolio['loss_count']}L")
        k6.metric("Total Fees", f"¥{portfolio['total_fees']:,.2f}")

        # --- Equity Curve ---
        snaps = portfolio.get("daily_snapshots", [])
        if snaps:
            chart = pd.DataFrame(snaps)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=chart["date"], y=chart["total_value"],
                name="Total Value", line=dict(color="#58a6ff", width=2),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.1)"), secondary_y=False)
            fig.add_trace(go.Scatter(x=chart["date"], y=chart["cumulative_return_pct"],
                name="Cum. Return %", line=dict(color="#3fb950", width=1.5, dash="dot")),
                secondary_y=True)
            fig.update_layout(height=300, template="plotly_dark",
                            margin=dict(l=0, r=0, t=10, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig.update_yaxes(title_text="Asset Value", secondary_y=False)
            fig.update_yaxes(title_text="Return %", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        # --- Positions Table ---
        positions = portfolio.get("positions", [])
        if positions:
            st.subheader(f"Positions ({len(positions)})")
            pos_df = pd.DataFrame(positions)
            pos_df["allocation"] = pos_df["allocation_pct"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(pos_df[[
                "symbol", "name", "quantity", "avg_cost", "current_price",
                "market_value", "unrealized_pnl", "unrealized_pnl_pct",
                "stop_loss", "take_profit", "strategy",
            ]], column_config={
                "unrealized_pnl": st.column_config.NumberColumn("PnL", format="¥%+.0f"),
                "unrealized_pnl_pct": st.column_config.NumberColumn("PnL%", format="%+.1f%%"),
                "current_price": st.column_config.NumberColumn("Price", format="¥%.2f"),
            }, hide_index=True, use_container_width=True)

            # --- Allocation Pie ---
            pie_data = {p.get("name", p.get("symbol","")): p.get("market_value", 0) for p in positions}
            if portfolio["cash"] > 0:
                pie_data["Cash"] = portfolio["cash"]
            fig_pie = px.pie(names=list(pie_data.keys()), values=list(pie_data.values()),
                            title="Allocation", color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie.update_layout(height=300, template="plotly_dark",
                                 margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- Recent Trades ---
        trades = portfolio.get("recent_trades", [])
        if trades:
            st.subheader("Recent Trades")
            st.dataframe(pd.DataFrame(trades)[:10], hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Tab 7: RISK — 8-Layer Risk Control Overview
# ═══════════════════════════════════════════════════════════════
elif tab == "🛡️ Risk":
    st.subheader("Risk Control Center")

    portfolio = get_portfolio_state()

    # --- Risk Gauges ---
    if portfolio:
        total_v = portfolio["total_value"]
        drawdown_pct = 0
        snaps = portfolio.get("daily_snapshots", [])
        if snaps:
            values = [s["total_value"] for s in snaps]
            peak = max(values)
            drawdown_pct = (total_v / peak - 1) * 100 if peak > 0 else 0

        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Current Drawdown", f"{drawdown_pct:.1f}%",
                  "⚠️ Warning" if drawdown_pct < -5 else ("🚨 Critical" if drawdown_pct < -8 else "✅ Normal"))
        gc2.metric("Win Rate", f"{portfolio['win_rate']:.1f}%")
        gc3.metric("Win/Loss", f"{portfolio['win_count']}/{portfolio['loss_count']}")
        gc4.metric("Total Fees", f"¥{portfolio['total_fees']:.2f}")

    # --- 8 Layers ---
    st.subheader("8-Layer Risk Control Status")
    layers = st.columns(4)
    risk_status = [
        ("L1 Circuit Breaker", "DD>8%: Reduce 50% | DD>15%: Liquidate", "✅"),
        ("L2 ATR Stop Loss", "Dynamic stops based on volatility", "✅"),
        ("L3 Trailing Stop", "Locks profits as price rises", "✅"),
        ("L4 Position Limits", f"Max positions controlled by regime", "✅"),
        ("L5 Anti-Stampede", ">50% positions down 3% triggers alert", "⚡"),
        ("L6 Holding Days", ">20 days stale check", "✅"),
        ("L7 Limit-Down Detect", "Monitor for locked limit-down", "⚡"),
        ("L8 Correlation", "Pair correlation >0.7 warning", "⚡"),
    ]
    for i, (name, desc, status) in enumerate(risk_status):
        with layers[i % 4]:
            color = "#3fb950" if status == "✅" else "#f0883e"
            st.markdown(f"""
            <div style="background-color:#161b22;border:1px solid #30363d;
                border-left:3px solid {color};border-radius:8px;padding:10px;margin:4px 0">
                <span style="color:#f0f6fc;font-weight:bold">{status} {name}</span><br>
                <span style="color:#8b949e;font-size:11px">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    # --- Risk Metrics Chart ---
    if snaps:
        st.subheader("Drawdown History")
        dd_chart = pd.DataFrame(snaps)
        dd_chart["peak"] = dd_chart["total_value"].cummax()
        dd_chart["drawdown"] = (dd_chart["total_value"] / dd_chart["peak"] - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd_chart["date"], y=dd_chart["drawdown"],
            name="Drawdown %", fill="tozeroy", fillcolor="rgba(248,81,73,0.2)",
            line=dict(color="#f85149", width=1.5)))
        fig.add_hline(y=-8, line_dash="dash", line_color="#f0883e", opacity=0.7,
                     annotation_text="Reduce 50%")
        fig.add_hline(y=-15, line_dash="dash", line_color="#f85149", opacity=0.7,
                     annotation_text="Liquidate")
        fig.update_layout(height=300, template="plotly_dark",
                         margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # --- Signal Grader Demo ---
    st.subheader("Signal Grader")
    c1, c2, c3 = st.columns(3)
    with c1: demo_score = st.slider("Composite Score", 0, 100, 65, key="sg_score")
    with c2: demo_wr = st.slider("Win Rate", 0.0, 1.0, 0.55, 0.05, key="sg_wr")
    with c3: demo_regime = st.selectbox("Regime", ["strong_bull","weak_bull","range_bound","weak_bear","strong_bear"], key="sg_regime")

    try:
        from analysis.risk_controls import SignalGrader
        card = SignalGrader.grade(demo_score, demo_wr, demo_regime)
        grade_colors = {"STRONG_BUY":"#3fb950","BUY":"#58a6ff","WATCH":"#f0883e",
                       "HOLD":"#8b949e","SELL":"#f85149","STRONG_SELL":"#f85149"}
        color = grade_colors.get(card.level.label, "#8b949e")
        st.markdown(f"""
        <div style="background-color:#161b22;border:2px solid {color};border-radius:12px;
            padding:16px;text-align:center;margin:8px 0">
            <span style="font-size:32px">{card.level.emoji}</span><br>
            <span style="color:{color};font-size:28px;font-weight:bold">{card.level.label}</span><br>
            <span style="color:#8b949e">Direction: {card.direction} | Score: {card.score:.0f}</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception: pass


# ═══════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════
st.divider()
st.caption(f"AShare AI Trader v{VERSION} | DeepSeek V4 + LangGraph | ⚠️ Research & Education Only")
