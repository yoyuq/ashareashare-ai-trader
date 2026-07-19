"""
Streamlit Dashboard — A股智能分析Agent 前端

功能面板:
  1. 市场总览: 当前市场状态 + 关键指数
  2. 个股分析: 选择标的 → 技术分析 → 策略匹配
  3. 回测验证: 选择策略 → 回测 → 绩效报告
  4. 成本监控: 模型调用成本日/月统计

运行:
  streamlit run web/dashboard.py
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# 添加项目根目录
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
st.caption("AI驱动的量化分析助手 — 市场扫描 | 技术分析 | 回测验证 | 多空辩论")

# ═══════════════════════════════════════════════════════════════
# 侧边栏: 配置
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 配置")

    analysis_type = st.selectbox(
        "分析模式",
        ["市场总览", "个股深度分析", "策略回测", "多空辩论"],
    )

    st.divider()

    # 股票池选择
    watchlist = [
        "sh.600519", "sh.600036", "sz.000858", "sz.300750",
        "sh.601318", "sz.002415", "sh.600276", "sz.000333",
        "sh.600030", "sz.002594", "sh.600809", "sz.300059",
    ]
    selected_symbols = st.multiselect(
        "标的列表",
        watchlist,
        default=watchlist[:3],
    )

    st.divider()

    # 模型设置
    use_debate = st.checkbox("启用多空辩论", value=True)
    use_local = st.checkbox("优先本地模型(Ollama)", value=True)

    st.divider()

    run_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 页面内容区
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_components():
    """加载核心组件(缓存)"""
    from data.router import get_data_router
    from analysis.indicators import TechnicalAnalyzer
    from analysis.regime import MarketRegimeDetector
    from knowledge.manager import KnowledgeManager

    return {
        "router": get_data_router(),
        "analyzer": TechnicalAnalyzer(),
        "detector": MarketRegimeDetector(),
        "knowledge": KnowledgeManager(),
    }


components = load_components()
router = components["router"]
analyzer = components["analyzer"]
detector = components["detector"]
knowledge = components["knowledge"]


# ═══════════════════════════════════════════════════════════════
# 面板1: 市场总览
# ═══════════════════════════════════════════════════════════════

if analysis_type == "市场总览":
    st.header("📊 市场总览")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("分析日期", date.today().isoformat())

    # 获取市场状态
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
            "strong_bull": "🟢",
            "weak_bull": "🟡",
            "range_bound": "⚪",
            "weak_bear": "🟠",
            "strong_bear": "🔴",
            "crisis": "💀",
        }
        emoji = regime_emoji.get(regime.regime.value, "❓")

        with col2:
            st.metric("市场状态", f"{emoji} {regime.regime.value}", delta=f"置信度: {regime.confidence:.0%}")

        with col3:
            from analysis.regime import get_regime_parameters
            params = get_regime_parameters(regime.regime)
            st.metric("建议仓位", f"{params['position_multiplier']:.0%}")

        # 详细评分
        st.subheader("多维度评分")
        score_cols = st.columns(len(regime.scores))
        for i, (dim, score) in enumerate(regime.scores.items()):
            score_val = (score + 1) / 2  # 映射到0~1
            score_cols[i].metric(
                dim.replace("_", " ").title(),
                f"{score:.2f}",
                delta=regime.details.get(dim, ""),
            )

        # 策略建议
        st.subheader("🎯 当前市场适配策略")
        strategies = knowledge.get_strategies_for_regime(regime.regime.value)
        if strategies:
            strat_cols = st.columns(min(len(strategies), 3))
            for i, s in enumerate(strategies[:3]):
                with strat_cols[i]:
                    st.info(
                        f"**{s['name']}**\n\n"
                        f"类型: {s['category']}\n"
                        f"容量: {s.get('capacity_limit', 0):,.0f}元"
                    )
        else:
            st.warning("未找到适配策略")

    except Exception as e:
        st.error(f"市场数据获取失败: {e}")
        st.info("请确保网络连接正常且有AKShare数据源可用")

# ═══════════════════════════════════════════════════════════════
# 面板2: 个股分析
# ═══════════════════════════════════════════════════════════════

elif analysis_type == "个股深度分析":
    st.header("🔍 个股深度分析")

    if not selected_symbols:
        st.warning("请在侧边栏选择至少一个标的")
    else:
        for symbol in selected_symbols:
            with st.expander(f"📈 {symbol}", expanded=len(selected_symbols) <= 2):
                try:
                    from data.providers.base import DataFrequency, DataRequest

                    async def fetch_stock():
                        req = DataRequest(
                            symbol=symbol,
                            start_date=date.today() - timedelta(days=365),
                            end_date=date.today(),
                            frequency=DataFrequency.DAILY,
                        )
                        return await router.get_daily_kline(req)

                    result = asyncio.run(fetch_stock())

                    if result.data.empty:
                        st.warning(f"{symbol}: 无数据")
                        continue

                    df = result.data
                    indicator_result = analyzer.compute_all(df, symbol=symbol)
                    last = indicator_result.to_dataframe().iloc[-1]

                    # 关键指标卡片
                    metric_cols = st.columns(6)
                    metrics = [
                        ("最新价", f"{last.get('close', 0):.2f}"),
                        ("RSI(14)", f"{last.get('rsi_14', 0):.1f}"),
                        ("趋势评分", f"{last.get('trend_score', 0):.2f}"),
                        ("综合评分", f"{last.get('composite_score', 0):.1f}"),
                        ("量比", f"{last.get('vol_ratio_5', 0):.2f}"),
                        ("ATR%", f"{last.get('atr_14', 0) / last.get('close', 1) * 100:.2f}%"),
                    ]
                    for i, (label, value) in enumerate(metrics):
                        metric_cols[i].metric(label, value)

                    # 技术指标可视化
                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        st.subheader("价格 + 均线")
                        chart_df = pd.DataFrame({
                            "close": df["close"].values[-120:],
                            "ma_20": indicator_result.indicators["ma_20"].values[-120:],
                            "ma_60": indicator_result.indicators["ma_60"].values[-120:],
                        })
                        st.line_chart(chart_df)

                    with chart_col2:
                        st.subheader("MACD")
                        macd_df = pd.DataFrame({
                            "DIF": indicator_result.indicators["macd_dif"].values[-120:],
                            "DEA": indicator_result.indicators["macd_dea"].values[-120:],
                            "Hist": indicator_result.indicators["macd_hist"].values[-120:],
                        })
                        st.bar_chart(macd_df["Hist"])

                    # K线形态
                    active_patterns = [
                        k for k, v in indicator_result.patterns.items()
                        if hasattr(v, 'iloc') and v.iloc[-1] > 0
                    ]
                    if active_patterns:
                        st.info(f"📐 检测到K线形态: {', '.join(active_patterns[:5])}")

                    # 匹配策略
                    regime_str = regime.regime.value if 'regime' in dir() else "range_bound"
                    matched = knowledge.get_strategies_for_regime(regime_str)
                    if matched:
                        st.subheader("🎯 推荐策略")
                        for s in matched[:3]:
                            score_val = 0.7 if regime_str in s.get("market_regimes", []) else 0.4
                            st.caption(
                                f"• **{s['name']}** (适配分: {score_val:.2f}) — {s.get('description', '')}"
                            )

                except Exception as e:
                    st.error(f"{symbol} 分析失败: {e}")

# ═══════════════════════════════════════════════════════════════
# 面板3: 策略回测
# ═══════════════════════════════════════════════════════════════

elif analysis_type == "策略回测":
    st.header("🧪 策略回测验证")

    strategies = knowledge.list_strategies()

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_strategy = st.selectbox(
            "选择策略",
            [s["id"] for s in strategies],
            format_func=lambda x: next(
                (s["name"] for s in strategies if s["id"] == x), x
            ),
        )
    with col2:
        backtest_symbol = st.selectbox("回测标的", selected_symbols if selected_symbols else watchlist[:3])
    with col3:
        backtest_years = st.slider("回测年数", 1, 5, 2)

    if st.button("▶️ 运行回测", type="primary"):
        with st.spinner("回测运行中..."):
            try:
                from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
                from data.providers.base import DataFrequency, DataRequest

                async def run_bt():
                    req = DataRequest(
                        symbol=backtest_symbol,
                        start_date=date.today() - timedelta(days=365 * backtest_years),
                        end_date=date.today(),
                        frequency=DataFrequency.DAILY,
                    )
                    data_result = await router.get_daily_kline(req)

                    config = BacktestConfig(
                        initial_capital=100000,
                        start_date=data_result.data["date"].min(),
                        end_date=data_result.data["date"].max(),
                    )
                    engine = EventDrivenBacktestEngine(config)
                    engine.load_data(backtest_symbol, data_result.data)

                    # 简化回测
                    def ma_strategy(today, bars, broker):
                        sym = backtest_symbol
                        if sym not in bars:
                            return
                        bar = bars[sym]
                        c = bar["close"]

                        if not hasattr(ma_strategy, "_entry"):
                            ma_strategy._entry = None

                        if ma_strategy._entry is None:
                            qty = int(broker.account.cash * 0.3 / c / 100) * 100
                            if qty >= 100:
                                broker.buy(sym, qty, price=c)
                                ma_strategy._entry = today
                        else:
                            pos = broker.account.positions.get(sym)
                            if pos and (today - ma_strategy._entry).days >= 10:
                                broker.sell(sym, pos.quantity, price=c)
                                ma_strategy._entry = None

                    ma_strategy._entry = None
                    return engine.run(ma_strategy, progress_bar=False)

                result = asyncio.run(run_bt())

                # 结果展示
                st.success("回测完成!")

                perf_cols = st.columns(4)
                perf_cols[0].metric("总收益", f"{result.total_return:+.2f}%")
                perf_cols[1].metric("年化收益", f"{result.annual_return:+.2f}%")
                perf_cols[2].metric("Sharpe", f"{result.sharpe_ratio:.2f}")
                perf_cols[3].metric("最大回撤", f"{result.max_drawdown:.2f}%")

                perf_cols2 = st.columns(4)
                perf_cols2[0].metric("总交易", result.total_trades)
                perf_cols2[1].metric("Sortino", f"{result.sortino_ratio:.2f}")
                perf_cols2[2].metric("VaR(95%)", f"{result.var_95:.2f}%")
                perf_cols2[3].metric("年波动率", f"{result.daily_volatility:.1f}%")

                # 净值曲线
                st.subheader("📈 净值曲线")
                st.line_chart(result.equity_curve)

                st.caption(result.summary())

            except Exception as e:
                st.error(f"回测失败: {e}")

# ═══════════════════════════════════════════════════════════════
# 面板4: 多空辩论
# ═══════════════════════════════════════════════════════════════

elif analysis_type == "多空辩论":
    st.header("⚔️ 多空辩论")

    st.markdown("""
    ### 辩论机制 (v2.1)

    每个标的经过三方对抗辩论:

    1. **🐂 多头研究员** — 必须找到看涨理由
    2. **🐻 空头研究员** — 必须找到看跌理由
    3. **⚖️ 裁判** — 评估双方论据质量,量化分歧度

    > 分歧越大 → 置信度越低 → 仓位建议越保守
    """)

    debate_symbol = st.selectbox("辩论标的", selected_symbols if selected_symbols else watchlist[:3])

    if st.button("⚔️ 开始辩论", type="primary"):
        st.info("多空辩论需要LLM支持。请确保:\n"
                "1. Ollama已安装并运行 (`ollama serve`)\n"
                "2. 或 DeepSeek API Key已配置 (`.env` 文件)\n\n"
                "辩论将自动路由到可用模型。")

        with st.spinner("辩论进行中..."):
            # 显示辩论框架(无LLM时的展示)
            debate_col1, debate_col2 = st.columns(2)

            with debate_col1:
                st.subheader("🐂 多头论据")
                st.markdown("""
                **技术面多头信号:**
                - 需要计算 MA排列、MACD、RSI等指标
                - 检查是否处于多头排列

                **资金面多头信号:**
                - 需要检查成交量配合
                - 需要检查北向资金流向

                **估值面多头信号:**
                - 需要计算PE/PB分位数
                - 需要比较行业均值
                """)

            with debate_col2:
                st.subheader("🐻 空头论据")
                st.markdown("""
                **技术面风险:**
                - 是否存在顶背离
                - RSI是否超买区域

                **资金面风险:**
                - 是否存在量价背离
                - 融资余额是否下降

                **宏观风险:**
                - 市场状态是否不利于该策略
                - 板块轮动风险
                """)

            st.divider()
            st.subheader("⚖️ 裁判裁决")

            st.info(
                "📊 **综合评分**: 待LLM评估\n\n"
                "🔍 **关键分歧**: 待双方辩论后量化\n\n"
                "💡 **建议**: 启动Ollama或配置DeepSeek API以启用实时辩论"
            )

# ═══════════════════════════════════════════════════════════════
# 底部
# ═══════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "A股智能分析Agent v2.1 | "
    "数据来源: AKShare / Baostock | "
    "模型: Ollama Qwen3-4B + DeepSeek V4 | "
    "⚠️ 本工具仅供研究参考,不构成投资建议"
)
