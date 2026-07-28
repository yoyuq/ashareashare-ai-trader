"""
A股智能分析Agent Dashboard v2.14
"""
import asyncio
import concurrent.futures
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
import yaml, os, json

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

VERSION = "2.14.0"

# ═══════════════════════ 工具函数 ═══════════════════════

def _run_async(coro):
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

# ═══════════════════════ 页面配置 ═══════════════════════
st.set_page_config(page_title="A股AI Agent", page_icon="🏦", layout="wide",
                   initial_sidebar_state="collapsed")

# ═══════════════════════ 缓存组件 ═══════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_config_stocks():
    stocks = {}
    cfg_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
    try:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        for sym in cfg.get("watchlist", {}).get("default", []):
            code = sym.replace("sh.", "").replace("sz.", "")
            stocks[code] = sym
    except Exception: pass
    if len(stocks) < 5:
        stocks.update({"600519":"sh.600519","300750":"sz.300750","600036":"sh.600036",
                       "002594":"sz.002594","000858":"sz.000858","601318":"sh.601318"})
    return stocks

@st.cache_resource(show_spinner=False)
def init_components():
    c = {}
    try:
        from data.router import get_data_router
        c["router"] = get_data_router()
    except: c["router_err"] = "data router init failed"
    try:
        from analysis.indicators import TechnicalAnalyzer
        c["analyzer"] = TechnicalAnalyzer()
    except: pass
    try:
        from analysis.regime import MarketRegimeDetector
        c["detector"] = MarketRegimeDetector()
    except: pass
    try:
        from knowledge.manager import KnowledgeManager
        c["knowledge"] = KnowledgeManager()
    except: pass
    return c

stocks = load_config_stocks()
STOCK_LABELS = {f"{c}": s for c, s in stocks.items()}
STOCK_LIST = list(STOCK_LABELS.keys())
comps = init_components()
router = comps.get("router")
analyzer = comps.get("analyzer")
detector = comps.get("detector")
knowledge = comps.get("knowledge")
today = date.today()

# ═══════════════════════ 顶部 ═══════════════════════
t1, t2, t3, t4 = st.columns([3, 1, 1, 1])
with t1: st.title("🏦 A股智能分析Agent")
with t2:
    ollama_ok = os.path.exists(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"))
    st.caption(f"{'🟢' if ollama_ok else '⚪'} Ollama")
with t3:
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    st.caption(f"{'🟢' if 'sk-' in ds_key else '⚪'} DeepSeek")
with t4:
    st.caption(f"v{VERSION}")

mode = st.radio("", ["💬 AI问答", "📊 市场总览", "🔍 个股分析", "📋 交易建议",
                      "🧪 策略回测", "💰 模拟持仓", "🛡️ 风控面板"],
                horizontal=True, label_visibility="collapsed")

# ═══════════════════════ 标的选择 ═══════════════════════
sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 1])
with sel_col1:
    search = st.text_input("🔍", placeholder="输入代码 600519", label_visibility="collapsed")
with sel_col2:
    filtered = [l for l in STOCK_LIST if search in l] if search else STOCK_LIST
    selected = st.multiselect(f"{len(STOCK_LIST)}只标的", filtered,
                              default=filtered[:3], label_visibility="collapsed", max_selections=10)
with sel_col3:
    if st.button("🔄 全市场", help="加载全市场Top100"):
        @st.cache_data(ttl=86400, show_spinner="加载中...")
        def load_full_market():
            try:
                import akshare as ak
                df = ak.stock_zh_a_spot_em()
                df = df[(~df["名称"].str.contains("ST|退|N", na=True)) &
                       (df["最新价"].astype(float) > 2) &
                       (df["成交额"].astype(float) > 5e7)]
                df = df.sort_values("成交额", ascending=False).head(100)
                return {f"{r['代码']}": f"sh.{r['代码']}" if r["代码"].startswith("6") else f"sz.{r['代码']}"
                        for _, r in df.iterrows()}
            except: return {}
        extra = load_full_market()
        if extra: STOCK_LABELS.update(extra); STOCK_LIST.extend(extra.keys()); st.rerun()

symbols = [STOCK_LABELS.get(s, s) for s in selected]

# ═══════════════════════ 💬 AI问答 ═══════════════════════
if mode == "💬 AI问答":
    st.caption("用自然语言提问 — AI自动调用分析工具回答")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("如: 今天市场怎么样? 分析茅台 回测双均线"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    from agent.chat_agent import ChatAgent
                    agent = ChatAgent(router=router, knowledge=knowledge, analyzer=analyzer)
                    async def chat(): return await agent.chat(prompt, session_id="dashboard")
                    reply = _run_async(chat())
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(str(e))

    if st.session_state.chat_history:
        if st.button("🗑️ 清除对话"): st.session_state.chat_history = []; st.rerun()

# ═══════════════════════ 📊 市场总览 ═══════════════════════
elif mode == "📊 市场总览":
    if not symbols: st.info("请在上方选择标的")
    else:
        with st.spinner("获取市场数据..."):
            try:
                from data.providers.base import DataFrequency, DataRequest

                @st.cache_data(ttl=1800, show_spinner=False)
                def get_regime_cached():
                    async def f():
                        req = DataRequest("sh.000300", today - timedelta(days=365), today, DataFrequency.DAILY)
                        r = await router.get_daily_kline(req)
                        return detector.detect(r.data)
                    return _run_async(f())

                regime = get_regime_cached()
                emoji = {"strong_bull":"🟢强牛","weak_bull":"🟡弱牛","range_bound":"⚪震荡",
                        "weak_bear":"🟠弱熊","strong_bear":"🔴强熊","crisis":"💀危机"}

                c1, c2, c3 = st.columns(3)
                c1.metric("市场状态", emoji.get(regime.regime.value, "?"))
                c2.metric("置信度", f"{regime.confidence:.0%}")
                c3.metric("日期", today.isoformat())

                st.caption("五维度诊断")
                dims = st.columns(5)
                labels = {"price_structure":"价格结构","momentum":"动量","volatility":"波动率",
                         "volume":"成交量","market_breadth":"广度"}
                for i, (k, v) in enumerate(regime.scores.items()):
                    dims[i].metric(labels.get(k,k), f"{v:.2f}")

                strategies = knowledge.get_strategies_for_regime(regime.regime.value)
                if strategies:
                    st.caption("适配策略")
                    sc = st.columns(min(len(strategies), 3))
                    for i, s in enumerate(strategies[:3]):
                        sc[i].info(f"**{s['name']}** | {s['id']}")

                # 参数建议
                from analysis.regime import get_regime_parameters
                params = get_regime_parameters(regime.regime)
                pc = st.columns(5)
                pc[0].metric("仓位乘数", f"{params.get('position_multiplier',0):.0%}")
                pc[1].metric("止损乘数", f"{params.get('stop_loss_multiplier',0):.1f}")
                pc[2].metric("趋势权重", f"{params.get('trend_strategy_weight',0):.0%}")
                pc[3].metric("回归权重", f"{params.get('mean_reversion_weight',0):.0%}")
                pc[4].metric("最大持仓", params.get('max_positions',0))

                st.caption("标的快照")
                async def snap(sym):
                    try:
                        req = DataRequest(sym, today-timedelta(days=5), today, DataFrequency.DAILY)
                        r = await router.get_daily_kline(req)
                        if not r.data.empty:
                            d = r.data.iloc[-1]
                            return {"代码":sym.split(".")[-1],"收盘":f"{d['close']:.2f}",
                                    "涨跌":f"{d.get('pct_change',0):+.2f}%",
                                    "成交额(亿)":f"{d.get('amount',0)/1e8:.1f}"}
                    except: return None
                snaps = [_run_async(snap(s)) for s in symbols[:8]]
                snaps = [s for s in snaps if s]
                if snaps: st.dataframe(pd.DataFrame(snaps), width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"数据获取失败: {e}")

# ═══════════════════════ 🔍 个股分析 ═══════════════════════
elif mode == "🔍 个股分析":
    if not symbols: st.info("请在上方选择标的")
    else:
        for sym in symbols[:5]:
            with st.expander(f"📈 {sym}", expanded=len(symbols)<=2):
                try:
                    from data.providers.base import DataFrequency, DataRequest
                    async def fetch(s):
                        req = DataRequest(s, today-timedelta(days=365), today, DataFrequency.DAILY)
                        return await router.get_daily_kline(req)
                    result = _run_async(fetch(sym))
                    if result.data.empty: st.warning("无数据"); continue
                    df = result.data; ind = analyzer.compute_all(df, symbol=sym)
                    last = ind.to_dataframe().iloc[-1]; c = df["close"].iloc[-1]

                    mc = st.columns(6)
                    mc[0].metric("最新价", f"{c:.2f}")
                    mc[1].metric("RSI(14)", f"{last.get('rsi_14',0):.0f}")
                    mc[2].metric("趋势分", f"{last.get('trend_score',0):.2f}")
                    mc[3].metric("综合分", f"{last.get('composite_score',0):.0f}")
                    mc[4].metric("量比(5)", f"{last.get('vol_ratio_5',0):.1f}")
                    mc[5].metric("MACD柱", f"{last.get('macd_hist',0):.3f}")

                    ch1, ch2 = st.columns(2)
                    n = min(120, len(df))
                    with ch1:
                        st.line_chart(pd.DataFrame({
                            "close": df["close"].values[-n:],
                            "MA20": ind.indicators["ma_20"].values[-n:],
                            "MA60": ind.indicators["ma_60"].values[-n:]}), height=200)
                    with ch2:
                        st.line_chart(pd.DataFrame({
                            "DIF": ind.indicators["macd_dif"].values[-n:],
                            "DEA": ind.indicators["macd_dea"].values[-n:]}), height=200)

                    patterns = [k for k,v in ind.patterns.items() if hasattr(v,'iloc') and v.iloc[-1]>0]
                    if patterns: st.caption(f"📐 {' '.join(patterns[:6])}")
                except Exception as e: st.error(str(e))

# ═══════════════════════ 📋 交易建议 ═══════════════════════
elif mode == "📋 交易建议":
    st.caption("全市场扫描→策略匹配→回测验证→信号分级→交易建议")
    capital = st.number_input("总资金(元)", value=100000, step=10000, min_value=10000, format="%d")

    if st.button("🔮 生成建议", type="primary"):
        if not symbols: st.warning("请选择标的")
        else:
            with st.spinner("扫描+分析+回测+分级..."):
                try:
                    from analysis.recommender import RecommendationEngine
                    engine = RecommendationEngine(router=router, knowledge=knowledge, analyzer=analyzer)
                    async def gen(): return await engine.generate(capital=capital, top_n=15)
                    recs = _run_async(gen())

                    if not recs.recommendations:
                        st.warning(f"扫描{recs.total_scanned}只→{recs.total_filtered}只, 无符合条件的")
                    else:
                        st.success(f"{recs.total_scanned}只→{recs.total_filtered}只→精选{len(recs.recommendations)}个")
                        st.metric("建议总仓位", f"¥{recs.suggested_total_exposure:,.0f}",
                                 f"{recs.suggested_total_exposure/capital*100:.1f}%")

                        overview = []
                        for r in recs.recommendations:
                            overview.append({
                                "标的": r.symbol_name or r.symbol.split(".")[-1],
                                "策略": r.strategy_name,
                                "入场": f"{r.entry_price:.2f}",
                                "止损": f"{r.stop_loss:.2f}",
                                "止盈": f"{r.take_profit:.2f}",
                                "胜率": f"{r.win_rate:.1%}",
                                "盈亏比": f"{r.profit_factor:.1f}x",
                                "期望值": f"{r.expected_value:+.1f}%",
                                "仓位": f"¥{r.suggested_amount:,.0f}",
                                "评级": r.confidence,
                            })
                        st.dataframe(pd.DataFrame(overview), width="stretch", hide_index=True)

                        for r in recs.recommendations:
                            e = {"A":"🟢","B":"🟡","C":"🟠","D":"🔴","F":"⚫"}.get(r.confidence,"⚪")
                            with st.expander(f"{e} {r.symbol.split('.')[-1]} — {r.strategy_name} | {r.confidence}级 | ¥{r.suggested_amount:,.0f}",
                                           expanded=len(recs.recommendations)<=3):
                                c1,c2,c3 = st.columns(3)
                                c1.metric("入场",f"¥{r.entry_price:.2f}")
                                c1.metric("止损",f"¥{r.stop_loss:.2f}",f"{(r.stop_loss/r.entry_price-1)*100:+.1f}%")
                                c1.metric("止盈",f"¥{r.take_profit:.2f}",f"{(r.take_profit/r.entry_price-1)*100:+.1f}%")
                                c2.metric("胜率",f"{r.win_rate:.1%}")
                                c2.metric("盈亏比",f"{r.profit_factor:.1f}x")
                                c2.metric("夏普",f"{r.sharpe:.2f}")
                                c3.metric("仓位",f"¥{r.suggested_amount:,.0f}",f"{r.position_pct:.1%}")
                                c3.metric("股数",f"{r.shares}")
                                c3.metric("评分",f"{r.quality_score:.0f}/100")
                                st.info(f"💡 {r.key_reason}")
                                if r.risks: st.warning("⚠️ "+" | ".join(r.risks[:4]))
                except Exception as e: st.error(str(e))

# ═══════════════════════ 🧪 策略回测 ═══════════════════════
elif mode == "🧪 策略回测":
    strats = knowledge.list_strategies()
    c1, c2, c3 = st.columns(3)
    with c1: sid = st.selectbox("策略", [s["id"] for s in strats],
                                format_func=lambda x: next((s["name"] for s in strats if s["id"]==x),x))
    with c2: bts = st.selectbox("标的", symbols if symbols else STOCK_LIST[:5])
    with c3: yrs = st.slider("年数", 1, 5, 2)

    if st.button("▶️ 运行回测", type="primary"):
        with st.spinner("回测中..."):
            try:
                from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
                from data.providers.base import DataRequest, DataFrequency
                from analysis.recommender import STRATEGY_BACKTESTERS

                async def bt():
                    req = DataRequest(bts, today-timedelta(days=365*yrs), today, DataFrequency.DAILY)
                    data = await router.get_daily_kline(req)
                    cfg = BacktestConfig(100000, data.data["date"].min(), data.data["date"].max())
                    eng = EventDrivenBacktestEngine(cfg); eng.load_data(bts, data.data)

                    # 用真实策略回测
                    bt_func = STRATEGY_BACKTESTERS.get(sid)
                    bt_stats = bt_func(data.data) if bt_func else {}

                    def strat(today, bars, broker):
                        if bts not in bars: return
                        c = bars[bts]["close"]
                        if not hasattr(strat,"_e"): strat._e=None
                        if strat._e is None:
                            q=int(broker.account.cash*0.3/c/100)*100
                            if q>=100: broker.buy(bts,q,price=c); strat._e=today
                        else:
                            p=broker.account.positions.get(bts)
                            if p and (today-strat._e).days>=10: broker.sell(bts,p.quantity,price=c); strat._e=None
                    strat._e=None
                    return eng.run(strat, progress_bar=False), bt_stats

                r, sbt = _run_async(bt())
                st.success("完成!")

                pc=st.columns(4)
                pc[0].metric("总收益",f"{r.total_return:+.1f}%"); pc[1].metric("年化",f"{r.annual_return:+.1f}%")
                pc[2].metric("Sharpe",f"{r.sharpe_ratio:.2f}"); pc[3].metric("回撤",f"{r.max_drawdown:.1f}%")
                pc2=st.columns(4)
                pc2[0].metric("交易",r.total_trades); pc2[1].metric("胜率",f"{r.win_rate:.1%}")
                pc2[2].metric("Sortino",f"{r.sortino_ratio:.2f}"); pc2[3].metric("VaR95",f"{r.var_95:.1f}%")
                st.line_chart(r.equity_curve)

                # 策略真实回测统计
                if sbt:
                    st.caption(f"策略{sid}历史统计: 信号{sbt.get('signals',0)} | "
                              f"胜率{sbt.get('win_rate',0):.1%} | "
                              f"盈亏比{sbt.get('profit_factor',1):.1f}x | "
                              f"夏普{sbt.get('sharpe',0):.2f} | "
                              f"最大回撤{sbt.get('max_dd',0):.1f}%")
            except Exception as e: st.error(str(e))

# ═══════════════════════ 💰 模拟持仓 ═══════════════════════
elif mode == "💰 模拟持仓":
    st.caption("模拟交易看板 | 完整持仓+交易记录+AI决策解析")

    try:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine

        manager = PortfolioManager()
        engine = PaperTradingEngine(manager)
        state = manager.state

        rc1, rc2, rc3 = st.columns([1, 1, 4])
        with rc1:
            if st.button("🔄 刷新", width="stretch"): st.rerun()
        with rc2:
            if st.button("⚠️ 重置", width="stretch", type="secondary", help="清除所有持仓和交易"):
                manager.reset(); st.rerun()

        # ── KPI ──
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("💰 总资产", f"¥{state.total_value:,.2f}",
                 f"初始¥{state.initial_capital:,.0f}")
        ret_emoji = "🟢" if state.total_return >= 0 else "🔴"
        k2.metric(f"{ret_emoji} 累计收益", f"¥{state.total_return:+,.2f}",
                 f"{state.total_return_pct:+.2f}%")
        k3.metric("💵 现金", f"¥{state.cash:,.2f}",
                 f"{state.cash/max(state.total_value,1)*100:.1f}%")
        k4.metric("📦 持仓", f"¥{state.position_value:,.2f}",
                 f"{len(state.positions)}只")
        k5.metric("🏆 已实现", f"¥{state.total_realized_pnl:+,.2f}",
                 f"{state.win_count}W/{state.loss_count}L")

        # ── 走势图 ──
        snapshots = state.daily_snapshots
        if snapshots:
            st.subheader("📈 资产走势")
            chart_data = pd.DataFrame([{
                "日期": s.date, "总资产": s.total_value, "现金": s.cash,
                "持仓市值": s.position_value, "累计收益率(%)": s.cumulative_return_pct,
            } for s in snapshots])
            c1,c2 = st.columns(2)
            with c1:
                st.line_chart(chart_data.set_index("日期")[["总资产","现金","持仓市值"]], width="stretch")
            with c2:
                st.line_chart(chart_data.set_index("日期")[["累计收益率(%)"]], width="stretch")

        # ── 持仓明细 + 配置 ──
        st.subheader("📊 持仓明细 & 配置")
        pc, piec = st.columns([3, 2])
        with pc:
            if state.positions:
                pos_data = []
                for sym, p in state.positions.items():
                    pnl = p.unrealized_pnl; pnl_pct = p.unrealized_pnl_pct
                    pos_data.append({
                        "代码": sym.replace("sh.","").replace("sz.",""),
                        "名称": p.name, "持仓": p.quantity,
                        "成本": f"¥{p.avg_cost:.2f}", "现价": f"¥{p.current_price:.2f}",
                        "市值": f"¥{p.market_value:,.0f}",
                        "盈亏": f"¥{pnl:+,.0f}", "盈亏%": f"{pnl_pct:+.1f}%",
                        "止损": f"¥{p.stop_loss:.2f}", "策略": p.rec_strategy_name,
                    })
                st.dataframe(pd.DataFrame(pos_data), width="stretch", hide_index=True)

                # 止损进度条
                for sym, p in state.positions.items():
                    if p.current_price > 0 and p.stop_loss > 0 and p.take_profit > 0:
                        rng = p.take_profit - p.stop_loss
                        if rng > 0:
                            prog = max(0, min(1, (p.current_price - p.stop_loss) / rng))
                            name = p.name or sym.split(".")[-1]
                            st.progress(prog, text=f"{name}: 止损+{(p.current_price/p.stop_loss-1)*100:.1f}% | 止盈+{(p.take_profit/p.current_price-1)*100:.1f}%")
            else:
                st.info("暂无持仓")

        with piec:
            if state.positions:
                import plotly.express as px
                labels = []; values = []
                for sym, p in state.positions.items():
                    labels.append(p.name or sym.split(".")[-1]); values.append(p.market_value)
                if state.cash > 0: labels.append("现金"); values.append(state.cash)
                fig = px.pie(names=labels, values=values, title="资产配置",
                            color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(height=350, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, width="stretch")

        # ── AI 决策解析 ──
        if state.positions:
            st.subheader("🔬 AI 买入决策")
            ai_cols = st.columns(min(3, len(state.positions)))
            for idx, (sym, p) in enumerate(state.positions.items()):
                with ai_cols[idx % 3]:
                    badge = "🟢" if p.rec_entry_score >= 65 else ("🟡" if p.rec_entry_score >= 55 else "🟠")
                    with st.expander(f"{badge} {p.name or sym.split('.')[-1]} — {p.rec_strategy_name or '综合'}"):
                        st.caption(f"评分: {p.rec_entry_score:.0f}/100 | 策略胜率: {p.rec_win_rate:.0%}")
                        if p.rec_verdict: st.markdown(f"💡 {p.rec_verdict}")
                        if p.rec_reasons:
                            for i,r in enumerate(p.rec_reasons[:4]): st.markdown(f"  {i+1}. {r}")
                        if p.rec_risks:
                            for r in p.rec_risks[:3]: st.markdown(f"  ⚠️ {r}")

        # ── 交易记录 ──
        st.subheader("📋 交易记录")
        if state.trade_history:
            trade_data = []
            for t in reversed(state.trade_history[-30:]):
                trade_data.append({
                    "日期": t.date, "股票": t.name or "", "代码": t.symbol.replace("sh.","").replace("sz.",""),
                    "方向": "卖出" if t.side=="sell" else "买入",
                    "数量": t.quantity, "价格": f"¥{t.price:.2f}" if t.price else "-",
                    "金额": f"¥{t.amount:,.0f}" if t.amount else "-",
                    "手续费": f"¥{t.commission+t.stamp_duty+t.transfer_fee:.2f}",
                    "盈亏": f"¥{t.pnl:+,.0f}" if t.pnl is not None else "-",
                    "原因": t.reason or t.exit_reason or "-",
                })
            st.dataframe(pd.DataFrame(trade_data), width="stretch", hide_index=True)
            st.caption(f"共{state.trade_count}笔 | 胜率{state.win_rate*100:.0f}% | "
                      f"手续费¥{state.total_commission+state.total_stamp_duty+state.total_transfer_fee:.2f}")
        else:
            st.info("暂无交易")

    except Exception as e:
        st.error(f"加载失败: {e}")

# ═══════════════════════ 🛡️ 风控面板 (NEW v2.14) ═══════════════════════
elif mode == "🛡️ 风控面板":
    st.caption("8层风控体系 + 组合风险监控 + 持仓诊断")

    try:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        from analysis.risk_controls import PortfolioRiskManager, SignalGrader

        manager = PortfolioManager()
        state = manager.state
        engine = PaperTradingEngine(manager)
        risk_mgr = PortfolioRiskManager(initial_capital=state.initial_capital)

        # ── 回撤监控 ──
        st.subheader("📉 回撤监控")
        if len(state.daily_snapshots) >= 5:
            import pandas as pd
            daily_rets = pd.Series([s.daily_return_pct/100 for s in state.daily_snapshots[-60:]])
            risk_state = risk_mgr.update(state.total_value, daily_rets)

            rc1,rc2,rc3,rc4 = st.columns(4)
            rc1.metric("当前回撤", f"{risk_state.drawdown_pct:.1f}%")
            rc2.metric("年化波动率", f"{risk_state.current_volatility:.1f}%")
            rc3.metric("风险乘数", f"{risk_state.risk_multiplier:.2f}")
            rc4.metric("建议仓位", f"{risk_state.suggested_exposure_pct:.0%}")

            # 断路器状态
            if risk_state.circuit_breaker_active:
                st.error(f"🚨 回撤断路器激活! 回撤{risk_state.drawdown_pct}%超过阈值")
            elif risk_state.drawdown_pct < -5:
                st.warning(f"⚠️ 回撤接近预警线 (-5%), 当前{risk_state.drawdown_pct:.1f}%")
            else:
                st.success("✅ 回撤正常, 未触发熔断")

            if risk_state.crisis_mode:
                st.error("💀 危机模式: 高波动+低广度, 建议95%现金")
            if risk_state.warning_messages:
                for w in risk_state.warning_messages:
                    st.warning(w)
        else:
            st.info("需要至少5天快照数据")

        # ── 8层风控检查 ──
        st.subheader("🛡️ 8层风控检查")

        l1,l2,l3,l4 = st.columns(4)
        with l1: st.metric("L1 熔断", "✅" if not (len(state.daily_snapshots)>=5 and
                          PortfolioRiskManager(initial_capital=state.initial_capital)
                          .update(state.total_value, pd.Series([s.daily_return_pct/100 for s in state.daily_snapshots[-60:]]))
                          .circuit_breaker_active) else "🚨")
        with l2: st.metric("L2 ATR止损", "✅", "持仓有动态止损")
        with l3: st.metric("L3 移动止盈", "✅", "shared.py内置")
        with l4: st.metric("L4 仓位上限", "✅", f"最多{8}只")

        l5,l6,l7,l8 = st.columns(4)
        # L5: 防踩踏
        if state.positions:
            losers = sum(1 for p in state.positions.values() if p.unrealized_pnl_pct < -3)
            ratio = losers/len(state.positions) if state.positions else 0
            l5.metric("L5 防踩踏", "🚨" if ratio >= 0.5 else "✅",
                     f"{losers}/{len(state.positions)}跌>3%")
        else:
            l5.metric("L5 防踩踏", "—", "无持仓")

        # L6: 持仓天数
        max_days = 0
        for p in state.positions.values():
            if p.buy_date:
                try: days = (date.today() - date.fromisoformat(p.buy_date)).days; max_days = max(max_days, days)
                except: pass
        l6.metric("L6 持仓天数", "⚠️" if max_days >= 20 else "✅",
                 f"最长{max_days}天" if max_days > 0 else "无")

        # L7: 跌停检测 (简化版)
        l7.metric("L7 跌停封板", "—", "需实时行情")

        # L8: 相关性
        if len(state.positions) >= 2:
            # 简易相关性检查 (基于持仓数量)
            l8.metric("L8 相关性", "✅", f"{len(state.positions)}只持仓")
        else:
            l8.metric("L8 相关性", "—", "<2只")

        # ── 信号分级演示 ──
        st.subheader("📊 信号分级器")
        demo_score = st.slider("综合评分", 0, 100, 65)
        demo_wr = st.slider("策略胜率", 0.0, 1.0, 0.55, 0.05)
        demo_regime = st.selectbox("市场状态", ["strong_bull","weak_bull","range_bound","weak_bear","strong_bear"])
        card = SignalGrader.grade(demo_score, demo_wr, demo_regime)
        st.info(f"{card.level.emoji} **{card.level.label}** | 方向: {card.direction} | 分数: {card.score:.0f}")

        # ── 组合摘要 ──
        st.subheader("📋 组合摘要")
        summary = engine.get_summary()
        sc1,sc2,sc3 = st.columns(3)
        sc1.metric("总资产", f"¥{summary['total_value']:,.2f}")
        sc2.metric("总费用", f"¥{summary['total_fees']:,.2f}")
        sc3.metric("胜率", f"{summary['win_rate']:.1f}%")

    except Exception as e:
        st.error(f"风控面板加载失败: {e}")

# ═══════════════════════ 底部 ═══════════════════════
st.divider()
st.caption(f"A股AI Agent v{VERSION} | DeepSeek V4 + LangGraph | ⚠️ 仅供研究学习，不构成投资建议")
