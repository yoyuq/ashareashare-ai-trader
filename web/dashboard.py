"""
A股智能分析Agent Dashboard v6.0 — 入口 (结构重构 v6.0)

模块结构 (v6.0 拆分, 行为与 v2.14/v3.1 保持一致):
  web/api_client.py  — Dashboard↔API 共享 HTTP 层
  web/data.py        — 缓存加载器 + 全市场抓取 + 组件单例
  web/tabs/*.py      — 每 tab 一个模块, 暴露 render()
本文件: set_page_config + 主题 + sidebar + tab 派发 + footer
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from web.api_client import api_get  # noqa: E402
from web.data import _run_async, init_components  # noqa: E402
from web.theme import apply_theme  # noqa: E402

VERSION = "3.1.0"

# ═══════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="A股智能分析Agent", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

apply_theme()

# 组件单例 (router/analyzer/detector/knowledge) — web.data 导入时已初始化
comps = init_components()


# ═══════════════════════════════════════════════════════════════
# Sidebar — System Status + Settings
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 A股智能分析Agent")
    st.caption(f"v{VERSION}")

    ds_ok = "sk-" in os.getenv("DEEPSEEK_API_KEY", "")

    # AI管道状态
    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager()
        prompts = km.list_all_prompts()
        agent_count = len(prompts)
        strategies = km.list_strategies()
        strat_count = len(strategies)
    except Exception:
        agent_count = 0
        strat_count = 0

    st.markdown(f"""
    <div style="background-color:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:8px;padding:12px;margin:8px 0">
    <span style="color:var(--ds-text2,#8b949e);font-size:12px">系统状态</span><br>
    <span style="color:{'#3fb950' if ds_ok else '#f85149'}">●</span> DeepSeek V4-Flash {'已连接' if ds_ok else '未配置'}<br>
    <span style="color:var(--ds-text2,#8b949e);font-size:12px;margin-top:8px">AI管道</span><br>
    <span style="color:var(--ds-text,#c9d1d9)">├ {agent_count} 个Agent</span><br>
    <span style="color:var(--ds-text,#c9d1d9)">├ {strat_count} 个策略</span><br>
    <span style="color:var(--ds-text,#c9d1d9)">├ 统一模型: V4-Flash</span><br>
    <span style="color:var(--ds-text,#c9d1d9)">└ 多轮辩论迭代</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 持仓速览 (30秒刷新) ──
    @st.fragment(run_every=5)
    def sidebar_portfolio():
        try:
            d = api_get("/api/v1/portfolio/mtm", timeout=3)
            s = d.get("summary", {})
            if s.get("total_value"):
                ret = s.get("total_return", 0)
                clr = "#f85149" if ret >= 0 else "#3fb950"
                st.markdown(f"""
                <div style="background:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:6px;padding:8px 10px;margin:4px 0">
                    <div style="font-size:10px;color:var(--ds-text2,#8b949e)">模拟账户</div>
                    <div style="font-size:16px;font-weight:700;color:var(--ds-text-h,#f0f6fc)">RMB {s['total_value']:,.0f}</div>
                    <div style="font-size:12px;font-weight:600;color:{clr}">{ret:+,.0f} ({s['total_return_pct']:+.2f}%)</div>
                    <div style="font-size:10px;color:var(--ds-text2,#8b949e)">{s.get('position_count',0)}只持仓 | 现金 {s['cash']/max(s['total_value'],1)*100:.0f}%</div>
                </div>""", unsafe_allow_html=True)
        except Exception:
            pass
    sidebar_portfolio()

    st.markdown("**导航**")
    tab = st.radio("tab_nav", ["⚡ 实时行情", "📊 市场总览", "📋 全市场行情", "🔍 机会扫描", "📈 技术分析", "🧪 策略回测",
                               "📋 AI信号", "💰 模拟持仓", "🛡️ 风控中心", "📚 知识库管理"], label_visibility="collapsed")

    st.divider()
    st.caption("自动刷新: 60秒")
    st.caption(f"更新: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 刷新全部"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("全市场AI选股工作流")
    st.caption("5884只 → 规则预筛 → DeepSeek → 自动交易")

    wf_col1, wf_col2 = st.columns(2)
    with wf_col1:
        do_workflow = st.button("🚀 一键执行", type="primary", use_container_width=True,
                                help="运行完整工作流: 全市场分析+自动交易")
    with wf_col2:
        do_dry = st.button("🔍 仅分析", use_container_width=True,
                           help="只做全市场AI分析,不实际交易")

    if do_workflow or do_dry:
        dry = do_dry
        with st.spinner("全市场AI选股工作流运行中..."):
            try:
                from simulation.daily_runner import run_full_day

                async def run():
                    return await run_full_day(no_llm=True, dry_run=dry)
                result = _run_async(run())
                if result:
                    t = result.get("trade", {})
                    s = result.get("summary", {})
                    if dry:
                        st.success(f"分析完成! 筛选出 {t.get('buys', 0)} 只BUY信号 | 耗时见日志")
                    else:
                        st.success(f"工作流完成! 总资产 RMB{s.get('total_value', 0):,.2f} | "
                                   f"收益 {s.get('total_return_pct', 0):+.2f}% | "
                                   f"买{t.get('bought', 0)} 卖{t.get('sold', 0)}")
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"工作流失败: {e}")


# ═══════════════════════════════════════════════════════════════
# Tab 派发
# ═══════════════════════════════════════════════════════════════
if tab == "⚡ 实时行情":
    from web.realtime import realtime_section
    realtime_section()
elif tab == "📊 市场总览":
    from web.tabs import overview
    overview.render()
elif tab == "📋 全市场行情":
    from web.tabs import market_table
    market_table.render()
elif tab == "🔍 机会扫描":
    from web.tabs import opportunity
    opportunity.render()
elif tab == "📈 技术分析":
    from web.tabs import technical
    technical.render()
elif tab == "🧪 策略回测":
    from web.tabs import backtest_tab
    backtest_tab.render()
elif tab == "📋 AI信号":
    from web.tabs import signals
    signals.render()
elif tab == "💰 模拟持仓":
    from web.tabs import portfolio_tab
    portfolio_tab.render()
elif tab == "🛡️ 风控中心":
    from web.tabs import risk
    risk.render()
elif tab == "📚 知识库管理":
    from web.tabs import knowledge
    knowledge.render()


# ═══════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════
st.divider()
st.caption(f"A股智能分析Agent v{VERSION} | DeepSeek V4 + LangGraph | ⚠️ 仅供研究学习，不构成投资建议")
