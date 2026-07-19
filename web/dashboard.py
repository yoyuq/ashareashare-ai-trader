"""
A股智能分析Agent Dashboard v2.2 — 优化版
"""
import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
import yaml, os, json

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

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

mode = st.radio("分析模式", ["📊 市场总览", "🔍 个股分析", "📋 交易建议", "🧪 策略回测", "⚔️ 多空辩论"],
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
    if st.button("🔄 加载全市场", help="从AKShare加载Top100(较慢,约30秒)", use_container_width=True):
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
                    return asyncio.run(f())

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

                snaps = [asyncio.run(snap(s)) for s in symbols[:8] if asyncio.run(snap(s))]
                if snaps: st.dataframe(pd.DataFrame(snaps), use_container_width=True, hide_index=True)

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

                    result = asyncio.run(fetch(sym))
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

                r = asyncio.run(bt())
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
                        return await engine.generate(symbols, capital=capital)

                    recs = asyncio.run(gen())

                    if not recs.recommendations:
                        st.warning("当前市场环境下未找到符合条件的交易机会,建议观望。")
                    else:
                        st.success(f"找到 {recs.total_opportunities} 个交易机会")

                        # 总览表
                        st.subheader("📊 胜率总览")
                        overview_data = []
                        for r in recs.recommendations:
                            overview_data.append({
                                "标的": r.symbol.split(".")[-1],
                                "策略": r.strategy_name,
                                "方向": r.direction.upper(),
                                "入场": f"{r.entry_price:.2f}",
                                "止损": f"{r.stop_loss:.2f}",
                                "止盈": f"{r.take_profit:.2f}",
                                "**胜率**": f"**{r.win_rate:.1%}**",
                                "盈亏比": f"{r.profit_factor:.1f}x",
                                "期望值": f"{r.expected_value:+.1f}%",
                                "夏普": f"{r.sharpe:.2f}",
                                "建议仓位": f"{r.suggested_amount:,.0f}",
                                "评级": f"{r.confidence}",
                            })
                        st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)

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

# ──── 多空辩论 ────
elif mode == "⚔️ 多空辩论":
    st.caption("Bull vs Bear vs Judge 三方对抗辩论")
    dbs = st.selectbox("标的", symbols if symbols else STOCK_LIST[:5])
    if st.button("⚔️ 辩论", type="primary"):
        st.info("多空辩论需LLM。DeepSeek API就绪,辩论可实时运行。\n\nOllama安装后可降本:简单任务走本地Qwen3-4B。")

# ═══════════════════════ 底部 ═══════════════════════
st.divider()
st.caption("A股AI Agent v2.2 | DeepSeek V4驱动 | ⚠️ 仅供研究")
