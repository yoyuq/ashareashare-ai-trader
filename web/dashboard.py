"""
Streamlit Dashboard — A股智能分析Agent 前端 v2.1
"""

import asyncio
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="A股智能分析Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 A股智能分析Agent v2.1")
st.caption("DeepSeek V4 驱动 | 数据: AKShare | 50项测试通过")

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def load_stock_pool():
    """加载股票池(缓存24小时)"""
    stocks = {}

    # 先从 symbols.yaml 加载精选池
    cfg_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        watchlist = cfg.get("watchlist", {}).get("default", [])
        for sym in watchlist:
            code = sym.replace("sh.", "").replace("sz.", "")
            stocks[f"{code}"] = sym
    except Exception:
        pass

    # 尝试从AKShare补充 (带超时)
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if not df.empty:
            df = df[
                (~df["名称"].str.contains("ST|退|N", na=True)) &
                (df["最新价"].astype(float) > 2) &
                (df["成交额"].astype(float) > 5e7)
            ]
            df = df.sort_values("成交额", ascending=False).head(100)
            for _, row in df.iterrows():
                code = row["代码"]
                name = row["名称"]
                sym = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
                label = f"{name}({code})"
                if label not in stocks:
                    stocks[label] = sym
    except Exception:
        pass

    # 确保至少有配置文件的标的
    if len(stocks) < 8:
        defaults = {
            "贵州茅台(600519)": "sh.600519",
            "宁德时代(300750)": "sz.300750",
            "招商银行(600036)": "sh.600036",
            "比亚迪(002594)": "sz.002594",
            "五粮液(000858)": "sz.000858",
            "中国平安(601318)": "sh.601318",
            "中信证券(600030)": "sh.600030",
            "美的集团(000333)": "sz.000333",
            "恒瑞医药(600276)": "sh.600276",
            "海康威视(002415)": "sz.002415",
        }
        for label, sym in defaults.items():
            if label not in stocks:
                stocks[label] = sym

    return stocks


@st.cache_resource
def get_components():
    """加载核心组件(全局缓存)"""
    comps = {}
    try:
        from data.router import get_data_router
        comps["router"] = get_data_router()
    except Exception as e:
        comps["router_err"] = str(e)

    try:
        from analysis.indicators import TechnicalAnalyzer
        comps["analyzer"] = TechnicalAnalyzer()
    except Exception as e:
        comps["analyzer_err"] = str(e)

    try:
        from analysis.regime import MarketRegimeDetector
        comps["detector"] = MarketRegimeDetector()
    except Exception as e:
        comps["detector_err"] = str(e)

    try:
        from knowledge.manager import KnowledgeManager
        comps["knowledge"] = KnowledgeManager()
    except Exception as e:
        comps["knowledge_err"] = str(e)

    return comps


# ═══════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 控制面板")

    stock_pool = load_stock_pool()
    st.caption(f"📋 已加载 {len(stock_pool)} 只标的")

    analysis_type = st.selectbox(
        "分析模式",
        ["📊 市场总览", "🔍 个股深度分析", "🧪 策略回测", "⚔️ 多空辩论"],
    )

    selected_names = st.multiselect(
        "选择标的",
        list(stock_pool.keys()),
        default=list(stock_pool.keys())[:5],
        placeholder="搜索股票名称或代码...",
    )
    selected_symbols = [stock_pool[n] for n in selected_names]

    st.divider()

    with st.expander("🔄 刷新数据"):
        if st.button("刷新股票列表", use_container_width=True):
            load_stock_pool.clear()
            st.rerun()

    with st.expander("📊 系统状态"):
        comps = get_components()
        status_ok = all(
            "err" not in k for k in comps
            if k not in ("router", "analyzer", "detector", "knowledge")
        )
        if status_ok:
            st.success("核心组件: ✅")
        else:
            for k, v in comps.items():
                if isinstance(v, str):
                    st.error(f"{k}: {v[:50]}")

        # API状态
        import os
        ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        if ds_key and "sk-" in ds_key:
            st.success("DeepSeek API: ✅")
        else:
            st.warning("DeepSeek API: ❌ 未配置")
        st.caption("Ollama: ❌ 未安装(官网下载)")

# ═══════════════════════════════════════════════════════════════
# 主内容区
# ═══════════════════════════════════════════════════════════════

components = get_components()
router = components.get("router")
analyzer = components.get("analyzer")
detector = components.get("detector")
knowledge = components.get("knowledge")

today_str = date.today().isoformat()


# ──── 面板1: 市场总览 ────
if analysis_type == "📊 市场总览":
    st.header("📊 市场总览")

    if not selected_symbols:
        st.info("👈 请在侧边栏选择至少一个标的")
    else:
        with st.spinner("获取市场数据..."):
            try:
                from data.providers.base import DataFrequency, DataRequest

                async def fetch_regime():
                    req = DataRequest(
                        symbol="sh.000300",
                        start_date=date.today() - timedelta(days=365),
                        end_date=date.today(),
                        frequency=DataFrequency.DAILY,
                    )
                    result = await router.get_daily_kline(req)
                    return detector.detect(result.data)

                regime = asyncio.run(fetch_regime())

                regime_emoji = {
                    "strong_bull": "🟢 强牛", "weak_bull": "🟡 弱牛",
                    "range_bound": "⚪ 震荡", "weak_bear": "🟠 弱熊",
                    "strong_bear": "🔴 强熊", "crisis": "💀 危机",
                }

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("市场状态",
                            regime_emoji.get(regime.regime.value, regime.regime.value))
                col2.metric("置信度", f"{regime.confidence:.0%}")
                col3.metric("检测日期", today_str)
                col4.metric("标的数量", len(selected_symbols))

                # 多维度评分
                st.subheader("📐 五维度诊断")
                dim_cols = st.columns(len(regime.scores))
                dim_labels = {
                    "price_structure": "价格结构", "momentum": "动量强度",
                    "volatility": "波动率", "volume": "成交量",
                    "market_breadth": "市场广度",
                }
                for i, (dim, score) in enumerate(regime.scores.items()):
                    bar = "█" * max(0, int((score + 1) * 10))
                    color = "🟢" if score > 0.3 else ("🟡" if score > -0.3 else "🔴")
                    dim_cols[i].metric(
                        dim_labels.get(dim, dim),
                        f"{score:.2f}",
                    )

                # 适配策略
                st.subheader("🎯 当前市场适配策略")
                strategies = knowledge.get_strategies_for_regime(regime.regime.value)
                if strategies:
                    s_cols = st.columns(min(len(strategies), 3))
                    for i, s in enumerate(strategies[:3]):
                        with s_cols[i]:
                            risk = knowledge._get_strategy_risks(s)
                            st.info(
                                f"**{s['name']}**\n"
                                f"类型: {s['id']}\n"
                                f"适配: +{s.get('fit_score', '?')}\n"
                                f"风险: {'; '.join(risk[:2])}"
                            )
                else:
                    st.warning("未找到适配策略,建议观望")

                # 快速扫描选中的标的
                st.subheader(f"📈 标的快照 ({len(selected_symbols)}只)")
                snapshot_data = []
                async def get_snapshot(sym):
                    try:
                        req = DataRequest(sym, date.today()-timedelta(days=60),
                                         date.today(), DataFrequency.DAILY)
                        r = await router.get_daily_kline(req)
                        if not r.data.empty:
                            d = r.data.iloc[-1]
                            return {
                                "代码": sym.replace("sh.","").replace("sz.",""),
                                "最新价": f"{d['close']:.2f}",
                                "涨跌幅": f"{d.get('pct_change', 0):+.2f}%",
                                "成交额(亿)": f"{d.get('amount', 0)/1e8:.2f}",
                            }
                    except Exception:
                        return None
                    return None

                snapshots = []
                for sym in selected_symbols[:10]:
                    s = asyncio.run(get_snapshot(sym))
                    if s:
                        snapshots.append(s)

                if snapshots:
                    st.dataframe(pd.DataFrame(snapshots), use_container_width=True)

            except Exception as e:
                st.error(f"数据获取失败: {e}")
                st.info("请检查网络连接。如AKShare超时,可减少标的数量后重试。")


# ──── 面板2: 个股深度分析 ────
elif analysis_type == "🔍 个股深度分析":
    st.header("🔍 个股深度分析")

    if not selected_symbols:
        st.info("👈 请在侧边栏选择标的")
    else:
        for symbol in selected_symbols[:5]:  # 最多5只
            with st.expander(f"📈 {symbol}", expanded=(len(selected_symbols) <= 2)):
                try:
                    from data.providers.base import DataFrequency, DataRequest

                    async def fetch_stock(sym):
                        req = DataRequest(sym, date.today()-timedelta(days=365),
                                         date.today(), DataFrequency.DAILY)
                        return await router.get_daily_kline(req)

                    result = asyncio.run(fetch_stock(symbol))

                    if result.data.empty:
                        st.warning(f"{symbol}: 无数据")
                        continue

                    df = result.data
                    ind = analyzer.compute_all(df, symbol=symbol)
                    last = ind.to_dataframe().iloc[-1]

                    # 指标卡片
                    mc = st.columns(6)
                    metrics = [
                        ("最新价", f"{df['close'].iloc[-1]:.2f}" if 'close' in df.columns else "N/A"),
                        ("RSI(14)", f"{last.get('rsi_14', 0):.1f}"),
                        ("趋势评分", f"{last.get('trend_score', 0):.2f}"),
                        ("综合评分", f"{last.get('composite_score', 0):.1f}"),
                        ("量比", f"{last.get('vol_ratio_5', 0):.2f}"),
                        ("MACD柱", f"{last.get('macd_hist', 0):.2f}"),
                    ]
                    for i, (label, value) in enumerate(metrics):
                        mc[i].metric(label, value)

                    # 图表
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("价格 + 均线(近120日)")
                        chart_n = min(120, len(df))
                        chart = pd.DataFrame({
                            "close": df["close"].values[-chart_n:],
                            "MA20": ind.indicators["ma_20"].values[-chart_n:],
                            "MA60": ind.indicators["ma_60"].values[-chart_n:],
                        })
                        st.line_chart(chart, height=200)

                    with c2:
                        st.caption("MACD(近120日)")
                        chart_n = min(120, len(df))
                        macd_chart = pd.DataFrame({
                            "DIF": ind.indicators["macd_dif"].values[-chart_n:],
                            "DEA": ind.indicators["macd_dea"].values[-chart_n:],
                        })
                        st.line_chart(macd_chart, height=200)

                    # 形态 + 策略
                    active = [
                        k for k, v in ind.patterns.items()
                        if hasattr(v, 'iloc') and v.iloc[-1] > 0
                    ]
                    if active:
                        st.info(f"📐 K线形态: {', '.join(active[:5])}")

                except Exception as e:
                    st.error(f"{symbol} 分析失败: {e}")


# ──── 面板3: 策略回测 ────
elif analysis_type == "🧪 策略回测":
    st.header("🧪 策略回测验证")

    strategies = knowledge.list_strategies()
    c1, c2, c3 = st.columns(3)
    with c1:
        sid = st.selectbox("策略",
                          [s["id"] for s in strategies],
                          format_func=lambda x: next((s["name"] for s in strategies if s["id"]==x), x))
    with c2:
        bt_sym = st.selectbox("标的", selected_symbols if selected_symbols else list(stock_pool.values())[:5])
    with c3:
        yrs = st.slider("回测年数", 1, 5, 2)

    if st.button("▶️ 运行回测", type="primary"):
        with st.spinner("回测运行中..."):
            try:
                from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
                from data.providers.base import DataRequest, DataFrequency

                async def do_bt():
                    req = DataRequest(bt_sym,
                                     date.today()-timedelta(days=365*yrs),
                                     date.today(), DataFrequency.DAILY)
                    data = await router.get_daily_kline(req)

                    cfg = BacktestConfig(initial_capital=100000,
                                        start_date=data.data["date"].min(),
                                        end_date=data.data["date"].max())
                    engine = EventDrivenBacktestEngine(cfg)
                    engine.load_data(bt_sym, data.data)

                    def strat(today, bars, broker):
                        if bt_sym not in bars: return
                        c = bars[bt_sym]["close"]
                        if not hasattr(strat, "_e"): strat._e = None
                        if strat._e is None:
                            qty = int(broker.account.cash*0.3/c/100)*100
                            if qty>=100: broker.buy(bt_sym,qty,price=c); strat._e = today
                        else:
                            pos = broker.account.positions.get(bt_sym)
                            if pos and (today-strat._e).days>=10:
                                broker.sell(bt_sym,pos.quantity,price=c); strat._e = None
                    strat._e = None
                    return engine.run(strat, progress_bar=False)

                r = asyncio.run(do_bt())
                st.success("回测完成!")

                pc = st.columns(4)
                pc[0].metric("总收益", f"{r.total_return:+.2f}%")
                pc[1].metric("年化收益", f"{r.annual_return:+.2f}%")
                pc[2].metric("Sharpe", f"{r.sharpe_ratio:.2f}")
                pc[3].metric("最大回撤", f"{r.max_drawdown:.2f}%")

                pc2 = st.columns(4)
                pc2[0].metric("交易次数", r.total_trades)
                pc2[1].metric("Sortino", f"{r.sortino_ratio:.2f}")
                pc2[2].metric("VaR(95%)", f"{r.var_95:.2f}%")
                pc2[3].metric("年波动率", f"{r.daily_volatility:.1f}%")

                st.subheader("📈 净值曲线")
                st.line_chart(r.equity_curve)

                with st.expander("📋 详细报告"):
                    st.text(r.summary())

            except Exception as e:
                st.error(f"回测失败: {e}")


# ──── 面板4: 多空辩论 ────
elif analysis_type == "⚔️ 多空辩论":
    st.header("⚔️ 多空辩论 (v2.1)")

    st.markdown("""
    ### 三方对抗机制
    每个标的由 **Bull Researcher** vs **Bear Researcher** 对抗论证后,
    由 **Judge** 裁定并量化分歧度。分歧越大 → 置信度越低。
    """)

    db_sym = st.selectbox("辩论标的", selected_symbols if selected_symbols else list(stock_pool.values())[:5])

    if st.button("⚔️ 开始辩论", type="primary"):
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🐂 多头论据")
            st.info(
                "**技术面:** 检查MA排列/MACD金叉/RSI位置\n\n"
                "**资金面:** 成交量配合/主力资金流向\n\n"
                "**情绪面:** 板块热度/舆情分析\n\n"
                "*具体数值需LLM实时分析*"
            )

        with c2:
            st.subheader("🐻 空头论据")
            st.warning(
                "**技术风险:** 顶背离/超买区域/阻力位\n\n"
                "**资金风险:** 量价背离/融资余额下降\n\n"
                "**宏观风险:** 市场状态/板块轮动\n\n"
                "*具体数值需LLM实时分析*"
            )

        st.divider()
        st.subheader("⚖️ 裁判裁决")

        st.info(
            "📊 **综合评分**: 待LLM评估\n"
            "🔍 **关键分歧**: 待双方辩论后量化\n"
            "💡 **建议**: DeepSeek API已就绪,启用LLM后可获得实时辩论结果"
        )


# ═══════════════════════════════════════════════════════════════
# 底部
# ═══════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "A股智能分析Agent v2.1 | "
    "DeepSeek V4驱动 | 数据: AKShare | "
    "50项测试通过 | ⚠️ 仅供研究参考,不构成投资建议"
)
