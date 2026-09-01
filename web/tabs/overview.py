"""Tab: 📊 市场总览 — 实时指数 + 市场状态 + 精选池 + 推荐策略"""
import pandas as pd
import streamlit as st

from web.api_client import api_get
from web.data import get_regime, knowledge
from web.render import render_dataframe


@st.fragment(run_every=10)
def fragment_live_indices():
    """实时指数条 — 嵌入市场总览顶部, 含市场状态"""
    try:
        data = api_get("/api/v1/realtime/market", timeout=15)
        indices = data.get("indices", {})
        ms = data.get("market_status", {})
    except Exception:
        indices = {}
        ms = {}

    if not indices:
        st.caption("实时数据暂不可用")
        return

    # 市场状态条
    if ms and not ms.get("is_open"):
        st.info(f"⏸️ **{ms.get('detail', '市场已收盘')}** — 以下为最近收盘数据 (最近交易日: {ms.get('last_trade_date', 'N/A')})")

    cols = st.columns(len(indices))
    for i, (name, d) in enumerate(indices.items()):
        pct = d["pct"]
        clr = "#f85149" if pct > 0 else ("#3fb950" if pct < 0 else "var(--ds-text2,#8b949e)")
        arr = "▲" if pct > 0 else ("▼" if pct < 0 else "—")
        with cols[i]:
            st.markdown(f"""
            <div style="background:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:6px;padding:5px 6px;text-align:center">
                <div style="font-size:10px;color:var(--ds-text2,#8b949e)">{name}</div>
                <div style="font-size:15px;font-weight:700;color:var(--ds-text-h,#f0f6fc)">{d['price']:.0f}</div>
                <div style="font-size:11px;font-weight:600;color:{clr}">{arr} {pct:+.2f}%</div>
            </div>""", unsafe_allow_html=True)


def render():
    # --- 实时指数 (10秒刷新) ---
    fragment_live_indices()

    regime = get_regime()

    # --- Status Bar ---
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    if regime:
        emoji = {"strong_bull": "🟢强牛", "weak_bull": "🟡弱牛", "range_bound": "⚪震荡",
                 "weak_bear": "🟠弱熊", "strong_bear": "🔴强熊", "crisis": "💀危机"}
        sc1.metric("市场状态", f"{emoji.get(regime.regime.value, '?')} {regime.regime.value}")
        sc2.metric("置信度", f"{regime.confidence:.0%}")
        sc3.metric("价格结构", f"{regime.scores.get('price_structure', 0):.2f}")
        sc4.metric("动量", f"{regime.scores.get('momentum', 0):.2f}")
        sc5.metric("波动率", f"{regime.scores.get('volatility', 0):.2f}")
        sc6.metric("成交量", f"{regime.scores.get('volume', 0):.2f}")

    # --- 精选池 (API实时数据) ---
    try:
        rt = api_get("/api/v1/realtime/market", timeout=15)
        wl = rt.get("watchlist", [])
        if wl:
            df_watch = pd.DataFrame(wl)
            df_watch = df_watch.rename(columns={
                "code": "Code", "name": "Name", "price": "Price", "pct": "Change%"
            })
            df_watch["Signal"] = df_watch["Change%"].apply(
                lambda x: "🟢" if x > 1 else ("🟡" if x > -1 else "🔴")
            )

            st.subheader("精选池")
            c1, c2 = st.columns([3, 1])
            with c1:
                render_dataframe(df_watch.sort_values("Change%", ascending=False), height=400,
                                 col_rename={"Code": "代码", "Name": "名称", "Price": "现价",
                                             "Change%": "涨跌幅", "Signal": "信号"},
                                 formatters={"Price": "¥%.2f", "Change%": "%+.2f%%"})
            with c2:
                top_up = df_watch.nlargest(3, "Change%")
                top_down = df_watch.nsmallest(3, "Change%")
                st.caption("涨幅榜")
                for _, r in top_up.iterrows():
                    st.metric(r["Code"], f"¥{r['Price']:.2f}", f"{r['Change%']:+.2f}%")
                st.caption("跌幅榜")
                for _, r in top_down.iterrows():
                    st.metric(r["Code"], f"¥{r['Price']:.2f}", f"{r['Change%']:.2f}%")
        else:
            st.info("精选池数据加载中，请稍候...")
    except Exception as e:
        st.warning(f"精选池暂不可用: {e}")

    # --- Strategy Recommendations ---
    if regime:
        strats = knowledge.get_strategies_for_regime(regime.regime.value)
        if strats:
            st.subheader("推荐策略")
            scols = st.columns(min(len(strats), 4))
            for i, s in enumerate(strats[:4]):
                scols[i].info(f"**{s['name']}**  \n`{s['id']}`  \n{s.get('category', '')}")
