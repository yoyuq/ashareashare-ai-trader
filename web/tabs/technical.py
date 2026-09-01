"""Tab: 📈 技术分析 — K线 + 指标 + AI (DeepSeek V4-Flash) 分析"""
import os
from datetime import date

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.symbols import to_symbol
from scripts.shared import NAME_MAP
from web.data import get_stock_quick, load_full_market_stocks


def render():
    st.subheader("技术分析")

    # 全市场标的 (5100+只, 支持搜索)
    all_stock_map = load_full_market_stocks()

    c1, c2 = st.columns([1, 3])
    with c1:
        stock_list = sorted(all_stock_map.keys())
        selected_code = st.selectbox(
            "标的 (全市场)",
            stock_list,
            format_func=lambda x: f"{x} | {NAME_MAP.get(x, '')}",
            help=f"共 {len(stock_list):,} 只A股，输入代码即可搜索"
        )
    sym = all_stock_map.get(selected_code, to_symbol(selected_code))

    info = get_stock_quick(sym, 365)
    if info:
        df = info["data"]
        ind = info["indicators"]
        last = info["last"]

        # --- KPI Row ---
        mc = st.columns(8)
        mc[0].metric("现价", f"¥{info['close']:.2f}", f"{info.get('change', 0):+.2f}%")
        mc[1].metric("RSI(14)", f"{last.get('rsi_14', 0):.0f}")
        mc[2].metric("Trend", f"{last.get('trend_score', 0):.2f}")
        mc[3].metric("Composite", f"{last.get('composite_score', 0):.0f}")
        mc[4].metric("Vol Ratio", f"{last.get('vol_ratio_5', 0):.1f}")
        mc[5].metric("MACD", f"{last.get('macd_hist', 0):.3f}")
        mc[6].metric("ATR(14)", f"{last.get('atr_14', 0):.2f}")
        mc[7].metric("Bias MA20", f"{last.get('bias_ma20', 0):.1f}%")

        # --- Candlestick Chart ---
        n = min(120, len(df))
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.25, 0.25],
                            vertical_spacing=0.05,
                            subplot_titles=("价格与均线", "成交量", "RSI"))

        # Price + MAs
        fig.add_trace(go.Candlestick(
            x=df["date"].values[-n:], open=df["open"].values[-n:],
            high=df["high"].values[-n:], low=df["low"].values[-n:],
            close=df["close"].values[-n:], name="现价",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149"), row=1, col=1)
        for ma, color, label in [("ma_5", "#f0883e", "MA5"), ("ma_20", "#58a6ff", "MA20"), ("ma_60", "#a371f7", "MA60")]:
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

        fig.update_layout(height=550, template="plotly_white",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=30, b=0),
                          hovermode="x unified", showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Indicator Dashboard ---
        st.subheader("Indicator Dashboard")
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.caption("趋势")
            st.metric("均线排列", "多头" if last.get("ma_bullish", 0) else "空头")
            st.metric("ADX(14)", f"{last.get('adx_14', 0):.1f}")
            st.metric("+DI/-DI", f"{last.get('plus_di_14', 0):.1f}/{last.get('minus_di_14', 0):.1f}")
        with ic2:
            st.caption("动量")
            st.metric("RSI(14)", f"{last.get('rsi_14', 0):.0f}")
            st.metric("MACD柱", f"{last.get('macd_hist', 0):.3f}")
            st.metric("ROC(5)", f"{last.get('roc_5', 0):.1f}%")
        with ic3:
            st.caption("波动率")
            st.metric("ATR(14)", f"{last.get('atr_14', 0):.2f}")
            st.metric("布林宽度", f"{last.get('bb_width_20', 0):.2f}")
            st.metric("HV(20)", f"{last.get('hv_20', 0):.1f}%")
        with ic4:
            st.caption("成交量")
            st.metric("量比5日", f"{last.get('vol_ratio_5', 0):.2f}")
            st.metric("OBV趋势", "向上" if last.get("obv", 0) > last.get("obv_ma_20", 0) else "向下")
            st.metric("换手率", f"{last.get('turnover', 0):.1f}%")

        # Patterns
        patterns = [k for k, v in ind.patterns.items() if hasattr(v, 'iloc') and v.iloc[-1] > 0]
        if patterns:
            st.caption(f"K线形态: {' | '.join(p[:6] for p in patterns)}")

        # ── AI DeepSeek V4-Pro 技术分析 ──
        st.divider()
        st.subheader("🤖 AI 技术分析 (DeepSeek V4-Pro)")

        ai_col1, ai_col2 = st.columns([3, 1])
        with ai_col2:
            do_ai = st.button("🔮 AI分析", type="primary", use_container_width=True,
                              help="调用 DeepSeek V4-Flash 对该标的进行全面技术分析")
        with ai_col1:
            st.caption("基于实时行情、技术指标、K线形态，由 DeepSeek V4-Flash 给出多维度综合研判")

        if do_ai:
            ds_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not ds_key or "sk-" not in ds_key:
                st.error("❌ 未配置 DEEPSEEK_API_KEY，请在 .env 中设置")
            else:
                with st.spinner("🤖 DeepSeek V4-Flash 分析中..."):
                    try:
                        from openai import OpenAI

                        # 构建分析提示
                        name = info.get("name", sym)
                        close = info["close"]
                        change = info.get("change", 0)

                        prompt = f"""你是一位资深A股技术分析师，请对以下标的进行全面的技术分析：

【标的信息】
- 代码: {sym}
- 名称: {name}
- 现价: ¥{close:.2f} ({change:+.2f}%)
- 分析日期: {date.today().isoformat()}

【趋势指标】
- ADX(14): {last.get('adx_14', 'N/A')}
- +DI/-DI: {last.get('plus_di_14', 'N/A')}/{last.get('minus_di_14', 'N/A')}
- 均线排列: {'多头' if last.get('ma_bullish', 0) else '空头'}
- 均线偏离(MA20): {last.get('bias_ma20', 'N/A')}%
- 趋势评分: {last.get('trend_score', 'N/A')}

【动量指标】
- RSI(14): {last.get('rsi_14', 'N/A')}
- MACD柱: {last.get('macd_hist', 'N/A')}
- ROC(5): {last.get('roc_5', 'N/A')}%
- 综合评分: {last.get('composite_score', 'N/A')}

【波动率】
- ATR(14): {last.get('atr_14', 'N/A')}
- 布林带宽(20): {last.get('bb_width_20', 'N/A')}
- 历史波动率(20): {last.get('hv_20', 'N/A')}%

【成交量】
- 量比(5日): {last.get('vol_ratio_5', 'N/A')}
- OBV趋势: {'向上' if last.get('obv', 0) > last.get('obv_ma_20', 0) else '向下'}
- 换手率: {last.get('turnover', 'N/A')}%

【K线形态】
- 识别到的形态: {', '.join(patterns) if patterns else '无明显形态'}

请从以下角度进行分析（使用中文，简洁专业）：
1. 📊 当前趋势判断（多头/空头/震荡，趋势强度）
2. 📈 多空信号梳理（列举看多信号和看空信号各2-4条）
3. 🎯 关键价位（支撑位、阻力位及依据）
4. ⚠️ 风险提示（需关注的潜在风险）
5. 🔮 综合研判（短期1-2周的操作建议）

请用以下格式输出，每条2-3句话，总计不超过500字。"""

                        client = OpenAI(
                            api_key=ds_key,
                            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                            timeout=60.0,
                        )
                        resp = client.chat.completions.create(
                            model="deepseek-v4-flash",
                            messages=[
                                {"role": "system", "content": "你是一位专业的A股技术分析师，擅长从多维度解读技术指标，给出客观、专业的技术分析意见。回答简洁有力，使用中文。"},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.3,
                            max_tokens=1200,
                        )
                        ai_text = resp.choices[0].message.content

                        # 渲染AI分析结果 (v3.1.2 修复: markdown 不能嵌 HTML, 否则表格/内容丢失)
                        # 标题用 HTML 框, 正文用 st.markdown 真正解析 markdown
                        st.markdown(f"""
                        <div style="background:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:12px;padding:16px 24px;margin-top:8px">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                        <span style="font-size:20px">🤖</span>
                        <span style="color:#58a6ff;font-weight:600;font-size:15px">DeepSeek V4-Flash 技术分析 — {name}({sym})</span>
                        <span style="color:var(--ds-text2,#8b949e);font-size:12px;margin-left:auto">模型: deepseek-v4-flash | 仅供参考，不构成投资建议</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(ai_text)  # 真正解析 markdown (标题/表格/加粗)
                        st.markdown("</div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"AI分析请求失败: {e}")
    else:
        st.warning("该标的数据不可用")
