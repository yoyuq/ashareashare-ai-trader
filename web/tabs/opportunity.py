"""Tab: 🔍 机会扫描 — 市场机会扫描 (RecommendationEngine)"""
import streamlit as st

from web.data import _run_async, analyzer, knowledge, router


def render():
    st.subheader("市场机会扫描")

    c1, c2, c3 = st.columns(3)
    with c1:
        min_score = st.slider("最低综合分", 40, 90, 55)
    with c2:
        top_n = st.slider("返回数量", 5, 50, 15)
    with c3:
        if st.button("🔍 开始扫描", type="primary", use_container_width=True):
            with st.spinner("Scanning..."):
                try:
                    from analysis.recommender import RecommendationEngine
                    engine = RecommendationEngine(router=router, knowledge=knowledge, analyzer=analyzer)

                    async def g():
                        return await engine.generate(capital=100000, top_n=top_n, min_composite=min_score)
                    recs = _run_async(g())

                    if recs.recommendations:
                        st.success(f"扫描{recs.total_scanned}只 → {len(recs.recommendations)}个机会")

                        # Signal Cards
                        cols = st.columns(3)
                        for i, r in enumerate(recs.recommendations[:9]):
                            with cols[i % 3]:
                                grade_color = {"A": "#3fb950", "B": "#58a6ff", "C": "#f0883e",
                                               "D": "#f85149"}.get(r.confidence, "var(--ds-text2,#8b949e)")
                                st.markdown(f"""
                                <div style="background-color:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-left:3px solid {grade_color};
                                    border-radius:8px;padding:12px;margin:4px 0">
                                    <span style="color:{grade_color};font-weight:bold">{r.confidence} | {r.symbol_name}</span><br>
                                    <span style="color:var(--ds-text2,#8b949e);font-size:12px">{r.strategy_name}</span><br>
                                    <span style="color:var(--ds-text,#c9d1d9)">入场: ¥{r.entry_price:.2f} | 止损: ¥{r.stop_loss:.2f}</span><br>
                                    <span style="color:#58a6ff">胜率: {r.win_rate:.0%} | 盈亏比: {r.profit_factor:.1f}x</span><br>
                                    <span style="color:var(--ds-text2,#8b949e);font-size:11px">{r.key_reason[:100]}</span>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("未找到符合条件的交易机会")
                except Exception as e:
                    st.error(str(e))
