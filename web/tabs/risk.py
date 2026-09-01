"""Tab: 🛡️ 风控中心 — 策略历史回测排名 + 8层风控 + 信号分级器"""
import json
from pathlib import Path

import streamlit as st

from web.data import get_portfolio_state

ROOT = Path(__file__).parent.parent.parent


def render():
    st.subheader("风控中心")

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
        gc1.metric("当前回撤", f"{drawdown_pct:.1f}%",
                   "⚠️ 预警" if drawdown_pct < -5 else ("🚨 危险" if drawdown_pct < -8 else "✅ 正常"))
        gc2.metric("胜率", f"{portfolio['win_rate']:.1f}%")
        gc3.metric("胜/负", f"{portfolio['win_count']}/{portfolio['loss_count']}")
        gc4.metric("总费用", f"¥{portfolio['total_fees']:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 策略历史回测
    # ═══════════════════════════════════════════════════════════════
    st.subheader("策略历史回测排名")
    st.caption("每策略在10只蓝筹股(2022-2025)上的汇总表现 | 运行 python -m backtest.strategy_backtest 刷新")

    bt_path = ROOT / "reports" / "strategy_backtest.json"
    if bt_path.exists():
        try:
            with open(bt_path, "r", encoding="utf-8") as f:
                bt_data = json.load(f)
            results = bt_data.get("results", {})

            if results:
                ranked = sorted(results.items(),
                                key=lambda x: x[1].get("sharpe", -999), reverse=True)

                rows = []
                for sid, r in ranked:
                    wr = r.get("win_rate", 0)
                    sr = r.get("sharpe", 0)
                    dd = r.get("max_drawdown", 0)
                    pf = r.get("profit_factor", 0)
                    trades = r.get("total_trades", 0)

                    of_risk_icon = {"low": "✅", "medium": "⚡", "high": "⚠️", "unknown": "?"}
                    grade_label = {"exc": "A", "good": "B", "ok": "C", "weak": "D"}
                    rows.append({
                        "策略": r.get("name", sid),
                        "胜率%": round(wr * 100, 1),
                        "夏普比率": sr,
                        "最大回撤%": dd,
                        "盈亏比": pf,
                        "总交易": trades,
                        "覆盖": r.get("stocks_tested", 0),
                        "过拟合": of_risk_icon.get(r.get("overfit_risk", ""), "") + r.get("overfit_risk", "?"),
                        "验证SR": r.get("val_sr", 0),
                        "评级": grade_label.get(r.get("grade", ""), r.get("grade", "")),
                    })

                if rows:
                    top_strat = ranked[0][1] if ranked else {}
                    low_risk_count = sum(1 for r in ranked if r[1].get("overfit_risk") == "low")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("最佳策略", top_strat.get("name", ""),
                               f"夏普{top_strat.get('sharpe', 0):.1f}")
                    mc2.metric("最高胜率", f"{max(r['胜率%'] for r in rows):.0f}%")
                    mc3.metric("低过拟合风险", f"{low_risk_count}/{len(ranked)}",
                               "通过交叉验证" if low_risk_count > 0 else "全部过拟合")
                    mc4.metric("总回测笔数", f"{sum(r['总交易'] for r in rows)}")

                    # 用HTML表格渲染, 绕过st.dataframe暗色主题CSS冲突
                    html = '<div style="max-height:420px;overflow-y:auto;border:1px solid var(--ds-border,#30363d);border-radius:8px"><table style="width:100%;border-collapse:collapse;font-size:13px">'
                    html += '<thead><tr style="background:var(--ds-bg2,#161b22);position:sticky;top:0">'
                    for col in ["策略", "胜率%", "夏普", "回撤%", "盈亏比", "交易", "覆盖", "过拟合", "验证SR", "评级"]:
                        html += f'<th style="padding:6px 8px;text-align:left;color:var(--ds-text2,#8b949e);border-bottom:2px solid var(--ds-border,#30363d)">{col}</th>'
                    html += '</tr></thead><tbody>'
                    for r in rows:
                        sr_color = "#3fb950" if r["夏普比率"] >= 1.5 else ("#f0883e" if r["夏普比率"] >= 0 else "#f85149")
                        html += '<tr style="border-bottom:1px solid var(--ds-bg3,#21262d)">'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text,#c9d1d9)">{r["策略"]}</td>'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text,#c9d1d9)">{r["胜率%"]:.1f}%</td>'
                        html += f'<td style="padding:5px 8px;color:{sr_color};font-weight:600">{r["夏普比率"]:+.2f}</td>'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text,#c9d1d9)">{r["最大回撤%"]:.1f}</td>'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text,#c9d1d9)">{r["盈亏比"]:.2f}</td>'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text,#c9d1d9)">{r["总交易"]}</td>'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text2,#8b949e)">{r["覆盖"]}</td>'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text-h,#f0f6fc)">{r["过拟合"]}</td>'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text2,#8b949e)">{r["验证SR"]:+.2f}</td>'
                        html += f'<td style="padding:5px 8px;color:var(--ds-text-h,#f0f6fc)">{r["评级"]}</td>'
                        html += '</tr>'
                    html += '</tbody></table></div>'
                    st.markdown(html, unsafe_allow_html=True)

                    # 展开详情 — 前3策略个股明细
                    with st.expander("查看策略个股明细 (前3策略)"):
                        of_mark = {"low": "OK", "medium": "!!", "high": "XX"}
                        for sid, r in ranked[:3]:
                            details = r.get("stock_details", [])
                            risk = r.get("overfit_risk", "?")
                            st.caption(f"[{of_mark.get(risk, '?')}] {r.get('name', sid)} / "
                                       f"ValSR {r.get('val_sr', 0):+.1f} / {len(details)} stocks")
                            if details:
                                h = '<div style="max-height:250px;overflow-y:auto;border:1px solid var(--ds-border,#30363d);border-radius:6px;margin:4px 0"><table style="width:100%;border-collapse:collapse;font-size:12px">'
                                h += '<thead><tr style="background:var(--ds-bg2,#161b22)">'
                                for col in ["代码", "笔数", "胜率%", "期望%", "夏普", "回撤", "盈亏比"]:
                                    h += f'<th style="padding:3px 6px;text-align:left;color:var(--ds-text2,#8b949e);border-bottom:2px solid var(--ds-border,#30363d)">{col}</th>'
                                h += '</tr></thead><tbody>'
                                for d in details:
                                    wr = d.get("win_rate", 0) * 100
                                    sr = d.get("sharpe", 0)
                                    clr = "#3fb950" if sr >= 1 else ("#f0883e" if sr >= 0 else "#f85149")
                                    h += f'<tr style="border-bottom:1px solid var(--ds-bg3,#21262d)">'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{d.get("symbol", "")}</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text2,#8b949e)">{d.get("trades", 0)}</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{wr:.0f}%</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{d.get("expected_value", 0):+.1f}</td>'
                                    h += f'<td style="padding:2px 6px;color:{clr};font-weight:600">{sr:+.2f}</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{d.get("max_dd", 0):.1f}</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{d.get("profit_factor", 0):.2f}</td>'
                                    h += '</tr>'
                                h += '</tbody></table></div>'
                                st.markdown(h, unsafe_allow_html=True)
                            else:
                                st.info("无明细数据")
        except Exception as e:
            st.warning(f"回测数据加载失败: {e}")
    else:
        st.info("📊 暂无策略回测数据。运行以下命令生成: `python -m backtest.strategy_backtest`")

    # ═══════════════════════════════════════════════════════════════
    # 8层风控状态
    # ═══════════════════════════════════════════════════════════════
    st.subheader("8层风控状态")
    layers = st.columns(4)
    risk_status = [
        ("L1 回撤断路器", "回撤>8%减仓50% | >15%清仓", "✅"),
        ("L2 ATR动态止损", "基于波动率自适应调整", "✅"),
        ("L3 移动止盈", "价格上涨自动锁定利润", "✅"),
        ("L4 仓位上限", "Max positions controlled by regime", "✅"),
        ("L5 防踩踏", "超50%持仓跌>3%触发", "⚡"),
        ("L6 持仓天数", "超20天未达标预警", "✅"),
        ("L7 跌停检测", "监控一字跌停封板", "⚡"),
        ("L8 相关性监控", "持仓对相关性>0.7预警", "⚡"),
    ]
    for i, (name, desc, status) in enumerate(risk_status):
        with layers[i % 4]:
            color = "#3fb950" if status == "✅" else "#f0883e"
            st.markdown(f"""
            <div style="background-color:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);
                border-left:3px solid {color};border-radius:8px;padding:10px;margin:4px 0">
                <span style="color:var(--ds-text-h,#f0f6fc);font-weight:bold">{status} {name}</span><br>
                <span style="color:var(--ds-text2,#8b949e);font-size:11px">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # 信号分级器
    # ═══════════════════════════════════════════════════════════════
    st.subheader("信号分级器")
    c1, c2, c3 = st.columns(3)
    with c1:
        demo_score = st.slider("综合评分", 0, 100, 65, key="sg_score")
    with c2:
        demo_wr = st.slider("策略胜率", 0.0, 1.0, 0.55, 0.05, key="sg_wr")
    with c3:
        demo_regime = st.selectbox("市场状态", ["strong_bull", "weak_bull", "range_bound", "weak_bear", "strong_bear"], key="sg_regime")

    try:
        from analysis.risk_controls import SignalGrader
        card = SignalGrader.grade(demo_score, demo_wr, demo_regime)
        grade_colors = {"STRONG_BUY": "#3fb950", "BUY": "#58a6ff", "WATCH": "#f0883e",
                        "HOLD": "var(--ds-text2,#8b949e)", "SELL": "#f85149", "STRONG_SELL": "#f85149"}
        color = grade_colors.get(card.level.label, "var(--ds-text2,#8b949e)")
        st.markdown(f"""
        <div style="background-color:var(--ds-bg2,#161b22);border:2px solid {color};border-radius:12px;
            padding:16px;text-align:center;margin:8px 0">
            <span style="font-size:32px">{card.level.emoji}</span><br>
            <span style="color:{color};font-size:28px;font-weight:bold">{card.level.label}</span><br>
            <span style="color:var(--ds-text2,#8b949e)">Direction: {card.direction} | Score: {card.score:.0f}</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass
