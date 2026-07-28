"""
A股智能分析Agent Dashboard v2.9 — 优化版
"""
import asyncio
import concurrent.futures
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
import yaml, os, json

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()


# ═══════════════════════ 工具函数 ═══════════════════════

def _run_async(coro):
    """
    Safe async runner for Streamlit (fixes asyncio.run() nested loop issue)

    When Streamlit's internal event loop is already running,
    asyncio.run() throws RuntimeError. This helper:
      - Uses asyncio.run() if no loop is running (normal case)
      - Falls back to a thread-pool execution if already inside a loop
    """
    try:
        asyncio.get_running_loop()
        # Already in an event loop — run in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

# ═══════════════════════ 页面配置 ═══════════════════════
st.set_page_config(page_title="A股AI Agent", page_icon="🏦", layout="wide",
                   initial_sidebar_state="collapsed")

# ═══════════════════════ 快速加载(无网络) ═══════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_config_stocks():
    """从yaml加载精选池 — 毫秒级"""
    stocks = {}
    cfg_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
    try:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        for sym in cfg.get("watchlist", {}).get("default", []):
            code = sym.replace("sh.", "").replace("sz.", "")
            stocks[code] = sym
    except Exception:
        pass
    # 确保最少有标的
    if len(stocks) < 5:
        defaults = {"600519": "sh.600519", "300750": "sz.300750", "600036": "sh.600036",
                    "002594": "sz.002594", "000858": "sz.000858", "601318": "sh.601318"}
        stocks.update(defaults)
    return stocks

@st.cache_resource(show_spinner=False)
def init_components():
    """初始化核心组件 — 首次加载后缓存"""
    c = {}
    try:
        from data.router import get_data_router
        c["router"] = get_data_router()
    except Exception as e: c["router_err"] = str(e)
    try:
        from analysis.indicators import TechnicalAnalyzer
        c["analyzer"] = TechnicalAnalyzer()
    except Exception as e: c["analyzer_err"] = str(e)
    try:
        from analysis.regime import MarketRegimeDetector
        c["detector"] = MarketRegimeDetector()
    except Exception as e: c["detector_err"] = str(e)
    try:
        from knowledge.manager import KnowledgeManager
        c["knowledge"] = KnowledgeManager()
    except Exception as e: c["knowledge_err"] = str(e)
    return c

# 初始化
stocks = load_config_stocks()
STOCK_LABELS = {f"{c}": s for c, s in stocks.items()}
STOCK_LIST = list(STOCK_LABELS.keys())
comps = init_components()

# ═══════════════════════ 顶部导航条 ═══════════════════════
t1, t2, t3 = st.columns([3, 1, 1])
with t1: st.title("🏦 A股智能分析Agent")
with t2:
    # Ollama状态
    ollama_ok = os.path.exists(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"))
    st.caption(f"{'🟢' if ollama_ok else '⚪'} Ollama")
with t3:
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    st.caption(f"{'🟢' if 'sk-' in ds_key else '⚪'} DeepSeek")

mode = st.radio("分析模式", ["💬 AI问答", "📊 市场总览", "🔍 个股分析", "📋 交易建议", "🧪 策略回测", "💰 模拟持仓"],
                horizontal=True, label_visibility="collapsed")

# ═══════════════════════ 标的选择(紧凑) ═══════════════════════
sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 1])
with sel_col1:
    search = st.text_input("🔍 搜索", placeholder="输入代码如 600519 或名称",
                          label_visibility="collapsed")
with sel_col2:
    if search:
        filtered = [l for l in STOCK_LIST if search in l]
    else:
        filtered = STOCK_LIST
    selected = st.multiselect(f"已选 {len(STOCK_LIST)}只标的", filtered,
                              default=filtered[:3] if not search else filtered[:3],
                              label_visibility="collapsed", max_selections=10)
with sel_col3:
    if st.button("🔄 加载全市场", help="从AKShare加载Top100(较慢,约30秒)", width="stretch"):
        @st.cache_data(ttl=86400, show_spinner="加载全市场数据...")
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
            except Exception: return {}
        extra = load_full_market()
        if extra:
            STOCK_LABELS.update(extra)
            STOCK_LIST.extend(extra.keys())
            st.rerun()

symbols = [STOCK_LABELS.get(s, s) for s in selected]

# ═══════════════════════ 内容区 ═══════════════════════
router = comps.get("router")
analyzer = comps.get("analyzer")
detector = comps.get("detector")
knowledge = comps.get("knowledge")

today = date.today()

# ──── AI问答 (新增 v2.8) ────
if mode == "💬 AI问答":
    st.caption('用自然语言提问,AI自动调用分析工具回答。试试问: 市场怎么样 / 分析600519 / 回测茅台双均线 / 有哪些策略')

    # 初始化聊天历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史消息
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if prompt := st.chat_input("输入你的问题,如: 今天市场怎么样? 帮我分析一下茅台"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    from agent.chat_agent import ChatAgent

                    agent = ChatAgent(
                        router=router,
                        knowledge=knowledge,
                        analyzer=analyzer,
                    )

                    async def chat():
                        return await agent.chat(prompt, session_id="dashboard")

                    reply = _run_async(chat())
                    st.markdown(reply)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply}
                    )
                except Exception as e:
                    error_msg = f"出错了: {e}"
                    st.error(error_msg)

    # 清除按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清除对话", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ──── 市场总览 ────
if mode == "📊 市场总览":
    if not symbols:
        st.info("请在上方选择标的后查看")
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

                emoji = {"strong_bull": "🟢强牛","weak_bull":"🟡弱牛","range_bound":"⚪震荡",
                        "weak_bear":"🟠弱熊","strong_bear":"🔴强熊","crisis":"💀危机"}

                c1, c2, c3 = st.columns(3)
                c1.metric("市场状态", emoji.get(regime.regime.value, "?"))
                c2.metric("置信度", f"{regime.confidence:.0%}")
                c3.metric("日期", today.isoformat())

                # 五维度
                st.caption("五维度诊断")
                dims = st.columns(5)
                labels = {"price_structure":"价格结构","momentum":"动量","volatility":"波动率",
                         "volume":"成交量","market_breadth":"广度"}
                for i, (k, v) in enumerate(regime.scores.items()):
                    dims[i].metric(labels.get(k,k), f"{v:.2f}")

                # 策略推荐
                strategies = knowledge.get_strategies_for_regime(regime.regime.value)
                if strategies:
                    st.caption("推荐策略")
                    sc = st.columns(min(len(strategies), 3))
                    for i, s in enumerate(strategies[:3]):
                        sc[i].info(f"**{s['name']}** | {s['id']}")

                # 标的快照
                st.caption(f"标的快照")
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

                snaps = [_run_async(snap(s)) for s in symbols[:8] if _run_async(snap(s))]
                if snaps: st.dataframe(pd.DataFrame(snaps), width="stretch", hide_index=True)

            except Exception as e:
                st.error(f"数据获取失败: {e}")

# ──── 个股分析 ────
elif mode == "🔍 个股分析":
    if not symbols:
        st.info("请在上方选择标的")
    else:
        for sym in symbols[:5]:
            with st.expander(f"📈 {sym}", expanded=len(symbols)<=2):
                try:
                    from data.providers.base import DataFrequency, DataRequest

                    async def fetch(s):
                        req = DataRequest(s, today-timedelta(days=365), today, DataFrequency.DAILY)
                        return await router.get_daily_kline(req)

                    result = _run_async(fetch(sym))
                    if result.data.empty:
                        st.warning("无数据"); continue

                    df = result.data; ind = analyzer.compute_all(df, symbol=sym)
                    last = ind.to_dataframe().iloc[-1]
                    c = df["close"].iloc[-1] if "close" in df.columns else 0

                    mc = st.columns(6)
                    mc[0].metric("最新价", f"{c:.2f}")
                    mc[1].metric("RSI", f"{last.get('rsi_14',0):.0f}")
                    mc[2].metric("趋势", f"{last.get('trend_score',0):.2f}")
                    mc[3].metric("综合", f"{last.get('composite_score',0):.0f}")
                    mc[4].metric("量比", f"{last.get('vol_ratio_5',0):.1f}")
                    mc[5].metric("MACD", f"{last.get('macd_hist',0):.2f}")

                    ch1, ch2 = st.columns(2)
                    n = min(120, len(df))
                    with ch1:
                        st.caption("价格+均线")
                        st.line_chart(pd.DataFrame({
                            "close": df["close"].values[-n:],
                            "MA20": ind.indicators["ma_20"].values[-n:],
                            "MA60": ind.indicators["ma_60"].values[-n:] }), height=200)
                    with ch2:
                        st.caption("MACD")
                        st.line_chart(pd.DataFrame({
                            "DIF": ind.indicators["macd_dif"].values[-n:],
                            "DEA": ind.indicators["macd_dea"].values[-n:] }), height=200)

                    patterns = [k for k,v in ind.patterns.items() if hasattr(v,'iloc') and v.iloc[-1]>0]
                    if patterns: st.caption(f"📐 {' '.join(patterns[:6])}")

                except Exception as e:
                    st.error(str(e))

# ──── 策略回测 ────
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

                async def bt():
                    req = DataRequest(bts, today-timedelta(days=365*yrs), today, DataFrequency.DAILY)
                    data = await router.get_daily_kline(req)
                    cfg = BacktestConfig(100000, data.data["date"].min(), data.data["date"].max())
                    eng = EventDrivenBacktestEngine(cfg); eng.load_data(bts, data.data)
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
                    return eng.run(strat, progress_bar=False)

                r = _run_async(bt())
                st.success("完成!")

                pc=st.columns(4)
                pc[0].metric("总收益",f"{r.total_return:+.1f}%"); pc[1].metric("年化",f"{r.annual_return:+.1f}%")
                pc[2].metric("Sharpe",f"{r.sharpe_ratio:.2f}"); pc[3].metric("回撤",f"{r.max_drawdown:.1f}%")
                pc2=st.columns(4)
                pc2[0].metric("交易",r.total_trades); pc2[1].metric("Sortino",f"{r.sortino_ratio:.2f}")
                pc2[2].metric("VaR",f"{r.var_95:.1f}%"); pc2[3].metric("波动",f"{r.daily_volatility:.0f}%")
                st.line_chart(r.equity_curve)
            except Exception as e:
                st.error(str(e))

# ──── 交易建议 (新增 v2.4) ────
elif mode == "📋 交易建议":
    st.caption("分析→策略匹配→快速回测→胜率统计→交易建议")

    capital = st.number_input("总资金(元)", value=100000, step=10000, min_value=10000,
                              format="%d", key="rec_capital")

    if st.button("🔮 生成交易建议", type="primary"):
        if not symbols:
            st.warning("请在上方选择标的")
        else:
            with st.spinner("分析中: 拉取数据→计算指标→匹配策略→回测验证→生成建议..."):
                try:
                    from analysis.recommender import RecommendationEngine

                    engine = RecommendationEngine(
                        router=router,
                        knowledge=knowledge,
                        analyzer=analyzer,
                    )

                    async def gen():
                        return await engine.generate(capital=capital, top_n=15)

                    with st.spinner("分析中(并行获取市场状态+北向+板块+扫描)..."):
                        recs = _run_async(gen())

                    if not recs.recommendations:
                        st.warning(
                            f"扫描了{recs.total_scanned}只→过滤后{recs.total_filtered}只, "
                            "无符合条件的机会,建议观望。"
                        )
                    else:
                        st.success(
                            f"全市场{recs.total_scanned}只→过滤{recs.total_filtered}只→"
                            f"精选 {len(recs.recommendations)} 个交易机会"
                        )

                        # 总览表
                        st.subheader("📊 胜率总览")
                        overview_data = []
                        for r in recs.recommendations:
                            overview_data.append({
                                "标的": r.symbol_name or r.symbol.split(".")[-1],
                                "代码": r.symbol.split(".")[-1],
                                "策略": r.strategy_name,
                                "入场": f"{r.entry_price:.2f}",
                                "止损": f"{r.stop_loss:.2f}",
                                "止盈": f"{r.take_profit:.2f}",
                                "**胜率**": f"**{r.win_rate:.1%}**",
                                "盈亏比": f"{r.profit_factor:.1f}x",
                                "期望值": f"{r.expected_value:+.1f}%",
                                "信号数": r.signal_count,
                                "建议仓位": f"{r.suggested_amount:,.0f}",
                                "评级": f"{r.confidence}",
                            })
                        st.dataframe(pd.DataFrame(overview_data), width="stretch", hide_index=True)

                        # 仓位汇总
                        st.metric("建议总仓位", f"¥{recs.suggested_total_exposure:,.0f}",
                                 f"{recs.suggested_total_exposure/capital*100:.1f}%总资金")

                        # 逐个详细建议
                        st.subheader("📋 详细建议")
                        for r in recs.recommendations:
                            emoji_map = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⚫"}
                            e = emoji_map.get(r.confidence, "⚪")

                            with st.expander(
                                f"{e} {r.symbol.split('.')[-1]} — {r.strategy_name} "
                                f"| 胜率{r.win_rate:.0%} | {r.confidence}级 | ¥{r.suggested_amount:,.0f}",
                                expanded=(len(recs.recommendations) <= 3)
                            ):
                                c1, c2, c3 = st.columns(3)

                                with c1:
                                    st.metric("入场价", f"¥{r.entry_price:.2f}")
                                    st.metric("止损价", f"¥{r.stop_loss:.2f}",
                                             f"{(r.stop_loss/r.entry_price-1)*100:+.1f}%")
                                    st.metric("止盈价", f"¥{r.take_profit:.2f}",
                                             f"{(r.take_profit/r.entry_price-1)*100:+.1f}%")
                                    st.caption(f"盈亏比: 1:{r.risk_reward_ratio:.1f}")

                                with c2:
                                    st.metric("**历史胜率**", f"**{r.win_rate:.1%}**")
                                    st.metric("历史盈亏比", f"{r.profit_factor:.1f}x")
                                    st.metric("期望值", f"{r.expected_value:+.2f}%")
                                    st.metric("夏普比率", f"{r.sharpe:.2f}")

                                with c3:
                                    st.metric("**建议仓位**", f"¥{r.suggested_amount:,.0f}")
                                    st.metric("仓位比例", f"{r.position_pct:.1%}")
                                    st.metric("建议股数", f"{r.shares}股")
                                    st.metric("质量评分", f"{r.quality_score:.0f}/100 ({r.confidence}级)")

                                st.info(f"💡 {r.key_reason}")

                                if r.risks:
                                    st.warning("⚠️ " + " | ".join(r.risks[:4]))

                                st.caption(f"回测区间: {r.backtest_period} | 最大回撤: {r.max_drawdown:.1f}%")

                except Exception as e:
                    st.error(f"推荐生成失败: {e}")

# ──── 模拟持仓 (v2.9) ────
elif mode == "💰 模拟持仓":
    st.caption("📊 模拟交易看板 | 初始资金¥10,000 | 实时追踪持仓盈亏")

    try:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine

        manager = PortfolioManager()
        state = manager.state

        # ── 刷新按钮 ──
        rc1, rc2, rc3 = st.columns([1, 1, 4])
        with rc1:
            if st.button("🔄 刷新数据", width="stretch"):
                st.rerun()
        with rc2:
            if st.button("⚠️ 重置账户", width="stretch", type="secondary",
                        help="清除所有持仓和交易记录，恢复初始资金¥10,000"):
                manager.reset()
                st.rerun()

        # ── KPI 指标卡片 ──
        k1, k2, k3, k4, k5 = st.columns(5)
        return_emoji = "🟢" if state.total_return >= 0 else "🔴"
        k1.metric("💰 总资产", f"¥{state.total_value:,.2f}",
                 f"初始¥{state.initial_capital:,.0f}")
        k2.metric(f"{return_emoji} 累计收益", f"¥{state.total_return:+,.2f}",
                 f"{state.total_return_pct:+.2f}%")
        k3.metric("💵 可用现金", f"¥{state.cash:,.2f}",
                 f"{state.cash/max(state.total_value,1)*100:.1f}%")
        k4.metric("📦 持仓市值", f"¥{state.position_value:,.2f}",
                 f"{len(state.positions)}只")
        k5.metric("🏆 已实现盈亏", f"¥{state.total_realized_pnl:+,.2f}",
                 f"{state.win_count}W/{state.loss_count}L")

        # ── 资产走势图 ──
        st.subheader("📈 资产走势")
        snapshots = state.daily_snapshots
        if snapshots:
            import pandas as pd
            chart_data = pd.DataFrame([
                {
                    "日期": s.date,
                    "总资产": s.total_value,
                    "现金": s.cash,
                    "持仓市值": s.position_value,
                    "累计收益率(%)": s.cumulative_return_pct,
                }
                for s in snapshots
            ])

            tab_c1, tab_c2 = st.columns(2)
            with tab_c1:
                st.caption("资产曲线")
                st.line_chart(chart_data.set_index("日期")[["总资产", "现金", "持仓市值"]],
                             width="stretch")
            with tab_c2:
                st.caption("累计收益率(%)")
                st.line_chart(chart_data.set_index("日期")[["累计收益率(%)"]],
                             width="stretch")

        # ── 持仓明细 + 资产配置 ──
        st.subheader("📊 持仓明细 & 资产配置")
        pos_col, pie_col = st.columns([3, 2])

        with pos_col:
            if state.positions:
                pos_data = []
                for sym, p in state.positions.items():
                    pnl = p.unrealized_pnl
                    pnl_pct = p.unrealized_pnl_pct
                    pos_data.append({
                        "代码": sym.replace("sh.", "").replace("sz.", ""),
                        "名称": p.name,
                        "持仓(股)": p.quantity,
                        "成本价": f"¥{p.avg_cost:.2f}",
                        "现价": f"¥{p.current_price:.2f}",
                        "市值": f"¥{p.market_value:,.0f}",
                        "盈亏": f"¥{pnl:+,.0f}",
                        "盈亏%": f"{pnl_pct:+.1f}%",
                        "止损": f"¥{p.stop_loss:.2f}",
                        "止盈": f"¥{p.take_profit:.2f}",
                        "策略": p.rec_strategy_name,
                    })

                st.dataframe(pd.DataFrame(pos_data), width="stretch", hide_index=True)

                # 止损止盈进度条
                st.caption("止损/止盈距离 (当前位置)")
                for sym, p in state.positions.items():
                    if p.current_price > 0 and p.stop_loss > 0 and p.take_profit > 0:
                        price_range = p.take_profit - p.stop_loss
                        if price_range > 0:
                            progress = (p.current_price - p.stop_loss) / price_range
                            progress = max(0, min(1, progress))
                            name = p.name or sym.split(".")[-1]
                            sl_dist = (p.current_price/p.stop_loss - 1)*100
                            tp_dist = (p.take_profit/p.current_price - 1)*100
                            st.progress(progress, text=f"{name}: 止损+{sl_dist:.1f}% | 止盈+{tp_dist:.1f}% | 当前¥{p.current_price:.2f}")
            else:
                st.info("暂无持仓。运行晚间分析生成推荐 → 早盘买入自动建仓。")

        with pie_col:
            if state.positions:
                import plotly.express as px
                import plotly.graph_objects as go

                pie_labels = []
                pie_values = []
                for sym, p in state.positions.items():
                    name = p.name or sym.split(".")[-1]
                    pie_labels.append(name)
                    pie_values.append(p.market_value)

                if state.cash > 0:
                    pie_labels.append("现金")
                    pie_values.append(state.cash)

                fig = px.pie(
                    names=pie_labels, values=pie_values,
                    title="资产配置",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, width="stretch")
            else:
                st.metric("现金占比", "100%", "¥100,000")

        # ── 交易记录 ──
        st.subheader("📋 交易记录")

        # ── AI分析解析 (每只持仓的买入逻辑) ──
        if state.positions:
            st.subheader("🔬 AI 买入决策解析")
            st.caption("点击展开每只持仓的AI分析理由、风险提示和策略详情")

            ai_cols = st.columns(min(3, len(state.positions)))
            for idx, (sym, p) in enumerate(state.positions.items()):
                col = ai_cols[idx % 3]
                with col:
                    name = p.name or sym.split(".")[-1]
                    code = sym.replace("sh.", "").replace("sz.", "")
                    score = p.rec_entry_score
                    # Color based on score
                    if score >= 65:
                        badge = "🟢"
                    elif score >= 55:
                        badge = "🟡"
                    else:
                        badge = "🟠"

                    with st.expander(f"{badge} {name} ({code}) — {p.rec_strategy_name or '综合策略'}", expanded=False):
                        # 评分和置信度
                        st.caption(f"**入场评分**: {score:.0f}/100 | **策略胜率**: {p.rec_win_rate:.0%} | **置信度**: {p.rec_confidence}")

                        # AI判断
                        if p.rec_verdict:
                            st.markdown(f"💡 **AI判断**: {p.rec_verdict}")

                        # 关键理由
                        if p.rec_reasons:
                            st.markdown("**📌 买入理由:**")
                            for i, reason in enumerate(p.rec_reasons[:4]):
                                st.markdown(f"  {i+1}. {reason}")

                        # 风险提示
                        if p.rec_risks:
                            st.markdown("**⚠️ 风险提示:**")
                            for risk in p.rec_risks[:3]:
                                st.markdown(f"  - {risk}")

                        # 止盈止损
                        st.caption(f"止损: ¥{p.stop_loss:.2f} | 止盈: ¥{p.take_profit:.2f}" +
                                  f" | 仓位占比: {p.market_value/max(state.total_value,1)*100:.1f}%")
        trade_history = state.trade_history
        if trade_history:
            trade_data = []
            for t in reversed(trade_history[-30:]):
                trade_data.append({
                    "日期": t.date,
                    "股票": t.name or t.symbol.split(".")[-1],
                    "代码": t.symbol.replace("sh.", "").replace("sz.", ""),
                    "方向": "🔴卖出" if t.side == "sell" else "🟢买入",
                    "数量": t.quantity,
                    "价格": f"¥{t.price:.2f}" if t.price else "-",
                    "金额": f"¥{t.amount:,.0f}" if t.amount else "-",
                    "手续费": f"¥{t.commission + t.stamp_duty + t.transfer_fee:.2f}",
                    "盈亏": f"¥{t.pnl:+,.0f}" if t.pnl is not None else "-",
                    "原因": t.reason or t.exit_reason or "-",
                })
            st.dataframe(pd.DataFrame(trade_data), width="stretch", hide_index=True)

            # 交易统计
            if state.trade_count > 0:
                st.caption(
                    f"共{state.trade_count}笔交易 | "
                    f"胜率{state.win_rate*100:.0f}% | "
                    f"累计手续费¥{state.total_commission+state.total_stamp_duty+state.total_transfer_fee:.2f}"
                )
        else:
            st.info("暂无交易记录")

        # ── 快速操作提示 ──
        st.caption("---")
        st.caption("💡 **操作提示**: "
                  "`python scripts/evening_summary.py --analyze` 盘后分析 → "
                  "`python scripts/morning_buy.py` 早盘买入 → "
                  "`python scripts/evening_summary.py --summarize` 收盘总结")

    except Exception as e:
        st.error(f"模拟持仓加载失败: {e}")
        st.info("请先初始化: `python -c \"from simulation.portfolio import PortfolioManager; PortfolioManager()\"`")

# ──── 多空辩论 ────
elif mode == "⚔️ 多空辩论":
    st.caption("Bull vs Bear vs Judge 三方对抗辩论")
    dbs = st.selectbox("标的", symbols if symbols else STOCK_LIST[:5])
    if st.button("⚔️ 辩论", type="primary"):
        st.info("多空辩论需LLM。DeepSeek API就绪,辩论可实时运行。\n\nOllama安装后可降本:简单任务走本地Qwen3-4B。")

# ═══════════════════════ 底部 ═══════════════════════
st.divider()
st.caption("A股AI Agent v2.9 | DeepSeek V4驱动 | ⚠️ 仅供研究")
