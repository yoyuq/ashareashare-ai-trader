"""Tab: 💰 模拟持仓 — Paper Trading 看板 (持仓表/雷达/权益曲线/饼图)"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from web.data import get_portfolio_state
from web.viz_components import make_equity_curve, make_radar_chart


@st.fragment(run_every=30)
def fragment_portfolio_live():
    """实时持仓 — 只刷实时价格字段 (紧凑), 完整表静态渲染, 避免整页刷新"""
    try:
        from web.api_client import api_get
        d = api_get("/api/v1/portfolio/mtm", timeout=5)
        s = d.get("summary", {})
        ps = d.get("positions", [])
        if not s.get("total_value"):
            return

        # 只刷实时变化值: 总资产 + 每只现价/盈亏 (紧凑一行, 原地更新)
        ret_clr = "#f85149" if (s.get("total_return", 0)) >= 0 else "#3fb950"
        st.markdown(
            f"<span style='color:var(--ds-text2,#8b949e);font-size:11px'>实时总资产 </span>"
            f"<span style='color:var(--ds-text-h,#f0f6fc);font-weight:700'>¥{s['total_value']:,.0f}</span> "
            f"<span style='color:{ret_clr}'>{(s.get('total_return_pct', 0)):+.2f}%</span>"
            f"<span style='color:var(--ds-text2,#8b949e);font-size:10px'> · 实时刷新中</span>",
            unsafe_allow_html=True,
        )
        if ps:
            cells = ""
            for p in ps:
                pnl_clr = "#f85149" if p["unrealized_pnl"] > 0 else ("#3fb950" if p["unrealized_pnl"] < 0 else "var(--ds-text2,#8b949e)")
                cells += (f"<span style='font-size:11px;color:var(--ds-text,#c9d1d9);margin-right:10px'>"
                          f"{p['name']} <b style='color:var(--ds-text-h,#f0f6fc)'>¥{p['current_price']:.2f}</b> "
                          f"<span style='color:{pnl_clr}'>{p['unrealized_pnl']:+.0f}</span></span>")
            st.markdown(
                f"<div style='font-size:11px;padding:4px 0'>{cells}</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass


def render():
    st.subheader("模拟持仓看板")

    # 加载本地持仓数据
    portfolio = get_portfolio_state()
    if not portfolio:
        st.info("暂无持仓数据。运行 `python scripts/morning_buy.py` 开始模拟交易。")
    else:
        cash = portfolio.get("cash", 0)
        total_value = portfolio.get("total_value", 0)
        positions = portfolio.get("positions", [])
        initial = portfolio.get("initial_capital", 100000)
        total_return = (total_value / initial - 1) * 100 if initial > 0 else 0
        win_count = sum(1 for p in positions if p.get("unrealized_pnl", 0) > 0)
        loss_count = sum(1 for p in positions if p.get("unrealized_pnl", 0) < 0)  # noqa: F841 (展示对称性保留)

        # ── 实时价格 (紧凑 fragment, 只刷该字段) ──
        fragment_portfolio_live()

        # ── 完整持仓表 (静态, 快照渲染一次, 不随 fragment 刷新) ──
        if positions:
            rows = ""
            for p in positions:
                pnl = float(p.get("unrealized_pnl", 0))
                pnl_clr = "#f85149" if pnl > 0 else ("#3fb950" if pnl < 0 else "var(--ds-text2,#8b949e)")
                rows += (f'<tr><td style="color:var(--ds-text,#c9d1d9)">{p.get("name", "")}</td>'
                         f'<td style="color:var(--ds-text2,#8b949e);font-size:10px">{p.get("symbol", "")}</td>'
                         f'<td style="color:var(--ds-text-h,#f0f6fc)">{p.get("quantity", 0)}</td>'
                         f'<td style="color:var(--ds-text2,#8b949e)">¥{float(p.get("avg_cost", 0)):.2f}</td>'
                         f'<td style="color:var(--ds-text-h,#f0f6fc);font-weight:600">¥{float(p.get("current_price", 0)):.2f}</td>'
                         f'<td style="color:var(--ds-text-h,#f0f6fc)">¥{float(p.get("market_value", 0)):,.0f}</td>'
                         f'<td style="color:{pnl_clr};font-weight:600">¥{pnl:+,.0f} ({float(p.get("unrealized_pnl_pct", 0)):+.1f}%)</td>'
                         f'<td style="color:#f85149;font-size:10px">¥{float(p.get("stop_loss", 0) or 0):.2f}</td>'
                         f'<td style="color:#3fb950;font-size:10px">¥{float(p.get("take_profit", 0) or 0):.2f}</td></tr>')
            st.markdown(f"""
            <div style="background:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:6px;padding:6px;margin-top:4px">
            <table style="width:100%;border-collapse:collapse">
            <thead><tr style="border-bottom:2px solid var(--ds-border,#30363d)">
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">名称</th>
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">代码</th>
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">数量</th>
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">成本</th>
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">现价</th>
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">市值</th>
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">盈亏</th>
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">止损</th>
                <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">止盈</th>
            </tr></thead><tbody>{rows}</tbody></table></div>""", unsafe_allow_html=True)

        # ── v3.1: Multi-dimension score radar chart ──
        if positions:
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                # 根据持仓计算多维度评分
                dim_scores = {
                    "技术面": min(100, max(20, total_return * 2 + 50)),
                    "资金流": min(100, max(20, win_count / max(len(positions), 1) * 100)),
                    "动量": min(100, max(20, 70 if total_return > 0 else 30)),
                    "风控": min(100, max(20, 85 if cash / max(total_value, 1) > 0.2 else 50)),
                    "质量": min(100, max(20, 60 + win_count * 5)),
                }
                fig_radar = make_radar_chart(dim_scores, title="组合多维度评分")
                st.plotly_chart(fig_radar, use_container_width=True)
            with c2:
                snaps = portfolio.get("daily_snapshots", [])
                if snaps:
                    chart = pd.DataFrame(snaps)
                    fig_eq = make_equity_curve(
                        dates=chart["date"].tolist(),
                        nav=chart["total_value"].tolist(),
                        title="权益曲线 vs 沪深300"
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

        # ── 权益曲线 (静态, 每日快照) ──
        snaps = portfolio.get("daily_snapshots", [])
        if snaps and not positions:  # Fallback if no positions but has snapshots
            st.divider()
            st.subheader("权益曲线")
            chart = pd.DataFrame(snaps).sort_values("date")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=chart["date"], y=chart["total_value"],
                                     name="总资产", line=dict(color="#58a6ff", width=2),
                                     fill="tozeroy", fillcolor="rgba(88,166,255,0.1)"), secondary_y=False)
            fig.add_trace(go.Scatter(x=chart["date"], y=chart["cumulative_return_pct"],
                                     name="收益率%", line=dict(color="#f85149", width=1.5, dash="dot")),
                          secondary_y=True)
            fig.update_layout(height=280, template="plotly_white",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig.update_yaxes(title_text="资产(元)", secondary_y=False)
            fig.update_yaxes(title_text="Return%", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        # ── 资产配置饼图 ──
        if positions:
            pie_data = {p.get("name", p.get("symbol", "")): p.get("market_value", 0) for p in positions}
            if portfolio["cash"] > 0:
                pie_data["现金"] = portfolio["cash"]
            fig_pie = px.pie(names=list(pie_data.keys()), values=list(pie_data.values()),
                             title="资产配置", color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie.update_layout(height=260, template="plotly_white",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
