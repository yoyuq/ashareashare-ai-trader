"""
A股智能分析Agent Dashboard v2.14 — Professional Dark Theme
"""
import asyncio, concurrent.futures, sys, os, json
from datetime import date, datetime, timedelta
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()

# v3.1: Enhanced visualization components
from web.viz_components import make_radar_chart, make_kpi_tiles, make_equity_curve, make_heatmap
from scripts.shared import NAME_MAP

VERSION = "3.1.0"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ═══════════════════════════════════════════════════════════════
# Professional Dark Theme CSS
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="A股智能分析Agent", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* ================================================================
       Theme-aware CSS — Streamlit handles base colors, we handle components
       Uses CSS variables synced to Streamlit's theme via JS
       ================================================================ */

    /* --- Spinner replaces running-man, hide Stop button --- */
    header[data-testid="stHeader"] button[kind="header"] { display: none !important; }
    [data-testid="stStatusWidget"] svg,
    [data-testid="stStatusWidget"] button { display: none !important; }
    [data-testid="stStatusWidget"] { position: relative; margin-right: 8px; }
    [data-testid="stStatusWidget"]::before {
        content: "";
        display: inline-block; width: 20px; height: 20px;
        border: 2.5px solid var(--ds-border); border-top-color: #58a6ff;
        border-radius: 50%;
        animation: ds-spin 0.7s linear infinite;
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
    }
    @keyframes ds-spin { to { transform: translate(-50%, -50%) rotate(360deg); } }

    /* ================================================================
       DARK THEME CSS variables (default / fallback)
       ================================================================ */
    html:not([data-theme="light"]) {
        --ds-bg: #0d1117; --ds-bg2: #161b22; --ds-bg3: #21262d;
        --ds-border: #30363d; --ds-text: #c9d1d9; --ds-text2: #8b949e;
        --ds-text-h: #f0f6fc; --ds-accent: #58a6ff; --ds-green: #3fb950;
        --ds-red: #f85149; --ds-orange: #f0883e; --ds-hover: #1c2128;
    }

    /* ================================================================
       LIGHT THEME CSS variables
       ================================================================ */
    html[data-theme="light"] {
        --ds-bg: #ffffff; --ds-bg2: #f0f2f5; --ds-bg3: #e4e6eb;
        --ds-border: #dadde1; --ds-text: #1c1e21; --ds-text2: #65676b;
        --ds-text-h: #050505; --ds-accent: #1877f2; --ds-green: #00a400;
        --ds-red: #fa383e; --ds-orange: #e67600; --ds-hover: #ebedf0;
    }

    /* --- NOTE: Base page colors (.stApp, .main, sidebar, headings, text)
       are intentionally NOT overridden. Streamlit's built-in theme
       handles them. We only style CUSTOM components below. --- */

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: var(--ds-bg2) !important;
        border: 1px solid var(--ds-border) !important;
        border-radius: 8px; padding: 12px 16px;
    }
    [data-testid="stMetric"] label { color: var(--ds-text2) !important; font-size: 12px; }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--ds-accent) !important; font-size: 24px; }
    [data-testid="stMetric"] div[data-testid="stMetricDelta"] { color: var(--ds-text) !important; }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: var(--ds-bg2) !important;
        border: 1px solid var(--ds-border) !important; border-radius: 8px;
    }
    [data-testid="stExpander"] details { background-color: var(--ds-bg2) !important; }
    [data-testid="stExpander"] details summary { color: var(--ds-text) !important; }
    [data-testid="stExpander"] details div { background-color: var(--ds-bg2) !important; color: var(--ds-text) !important; }
    .streamlit-expanderContent { background-color: var(--ds-bg2) !important; }

    /* Callout boxes */
    [data-testid="stNotification"], [data-testid="stInfo"], [data-testid="stWarning"],
    [data-testid="stSuccess"], [data-testid="stError"],
    .stAlert, div[data-testid="stAlert"], .stNotification {
        background-color: var(--ds-bg2) !important;
        border: 1px solid var(--ds-border) !important;
        color: var(--ds-text) !important; border-radius: 8px !important;
    }
    div[data-testid="stNotificationContent"] { background: transparent !important; }
    .stAlert p, .stAlert span, [data-testid="stNotification"] p { color: var(--ds-text) !important; }

    /* Tabs */
    button[data-baseweb="tab"] { color: var(--ds-text2) !important; background: transparent !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--ds-accent) !important; border-bottom: 2px solid var(--ds-accent) !important;
    }
    div[data-testid="stTabs"] { background: transparent; }
    div.stTabs [data-baseweb="tab-panel"] { background: transparent; }

    /* Input widgets */
    input, textarea, select, .stTextInput input, .stSelectbox select {
        background-color: var(--ds-bg) !important; color: var(--ds-text) !important;
        border: 1px solid var(--ds-border) !important; border-radius: 6px !important;
    }
    input:focus, textarea:focus { border-color: var(--ds-accent) !important; box-shadow: 0 0 0 2px rgba(88,166,255,0.2) !important; }
    div[data-baseweb="select"] > div { background-color: var(--ds-bg) !important; border-color: var(--ds-border) !important; }
    div[data-baseweb="popover"] { background-color: var(--ds-bg2) !important; }
    div[data-baseweb="popover"] li { color: var(--ds-text) !important; }
    div[data-baseweb="popover"] li:hover { background-color: var(--ds-hover) !important; }

    /* Buttons */
    button[kind="primary"], .stButton > button[kind="primary"] { background-color: var(--ds-green) !important; }
    button[kind="secondary"], .stButton > button {
        background-color: var(--ds-bg3) !important; color: var(--ds-text) !important;
        border: 1px solid var(--ds-border) !important; border-radius: 6px !important;
    }
    .stButton > button:hover { background-color: var(--ds-border) !important; border-color: var(--ds-text2) !important; }

    /* Slider */
    div[data-testid="stSlider"] div[role="slider"] { background-color: var(--ds-accent) !important; }
    div[data-testid="stSlider"] div { color: var(--ds-text); }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: var(--ds-bg2) !important;
        border: 1px solid var(--ds-border) !important; border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div { color: var(--ds-text) !important; background: transparent !important; }

    /* Spinner / Progress */
    .stSpinner > div { border-top-color: var(--ds-accent) !important; }
    div[data-testid="stProgressBar"] > div { background-color: var(--ds-bg3); }
    div[data-testid="stProgressBar"] > div > div { background-color: var(--ds-accent); }

    /* Status bar */
    .status-bar { background-color: var(--ds-bg2); border: 1px solid var(--ds-border); border-radius: 8px; padding: 8px 16px; margin-bottom: 12px; display: flex; align-items: center; gap: 20px; }
    .status-bar .label { color: var(--ds-text2); font-size: 12px; }
    .status-bar .value { color: var(--ds-text); font-size: 16px; font-weight: 600; }
    .status-bar .change-positive { color: var(--ds-green); }
    .status-bar .change-negative { color: var(--ds-red); }

    /* Radio buttons (navigation) */
    div[data-testid="stRadio"] > div { gap: 0; }
    div[data-testid="stRadio"] label { padding: 8px 16px; border-radius: 6px; font-size: 13px; }
    div[data-testid="stRadio"] label:hover { background-color: var(--ds-hover); }
</style>

<!-- Theme sync: detect Streamlit theme and set html[data-theme] -->
<script>
(function(){
  function syncTheme() {
    var el = document.querySelector('.stApp') || document.body;
    var bg = getComputedStyle(el).backgroundColor;
    var nums = bg.match(/\d+/g);
    var isDark = nums ? (+nums[0] + +nums[1] + +nums[2]) < 300 : true;
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  }
  syncTheme();
  var mo = new MutationObserver(syncTheme);
  mo.observe(document.documentElement, {attributes: true, subtree: true, attributeFilter: ['class','style']});
  mo.observe(document.body, {attributes: true, attributeFilter: ['class']});
  setInterval(syncTheme, 800);
})();
</script>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════
def _run_async(coro):
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)

@st.cache_data(ttl=3600, show_spinner=False)
def load_stocks():
    s = {}
    try:
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "symbols.yaml", encoding="utf-8"))
        for sym in cfg.get("watchlist", {}).get("default", []):
            s[sym.replace("sh.","").replace("sz.","")] = sym
    except: pass
    if len(s) < 5:
        s.update({"600519":"sh.600519","300750":"sz.300750","600036":"sh.600036",
                  "002594":"sz.002594","000858":"sz.000858","601318":"sh.601318"})
    return s

@st.cache_data(ttl=600, show_spinner=False)
def load_full_market_stocks():
    """从全市场缓存加载标的列表 {代码: 完整symbol}, 用于selectbox"""
    result = {}
    try:
        import json as _json
        cache_p = Path(__file__).parent.parent / "simulation_data" / "full_market_cache.json"
        if cache_p.exists():
            with open(cache_p, "r", encoding="utf-8") as f:
                market_data = _json.load(f)
            for item in market_data.get("data", []):
                code = item.get("code", "")
                if code:
                    prefix = "sh" if code.startswith("6") else "sz"
                    result[code] = f"{prefix}.{code}"
    except Exception:
        pass
    if len(result) < 10:
        result = {k.replace('sh.','').replace('sz.',''): v for k, v in stocks.items()}
    return result

@st.cache_resource(show_spinner=False)
def init_components():
    c = {}
    try: from data.router import get_data_router; c["router"] = get_data_router()
    except: c["router_err"] = "init failed"
    try: from analysis.indicators import TechnicalAnalyzer; c["analyzer"] = TechnicalAnalyzer()
    except: pass
    try: from analysis.regime import MarketRegimeDetector; c["detector"] = MarketRegimeDetector()
    except: pass
    try: from knowledge.manager import KnowledgeManager; c["knowledge"] = KnowledgeManager()
    except: pass
    return c

stocks = load_stocks()
comps = init_components()
router = comps.get("router")
analyzer = comps.get("analyzer")
detector = comps.get("detector")
knowledge = comps.get("knowledge")
today = date.today()

# ═══════════════════════════════════════════════════════════════
# Sidebar — System Status + Settings
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 A股智能分析Agent")
    st.caption(f"v{VERSION}")

    ds_ok = "sk-" in os.getenv("DEEPSEEK_API_KEY", "")
    db_ok = os.getenv("POSTGRES_HOST", "")

    # AI管道状态
    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager()
        prompts = km.list_all_prompts()
        agent_count = len(prompts)
        strategies = km.list_strategies()
        strat_count = len(strategies)
    except:
        agent_count = 0; strat_count = 0

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
            import requests
            d = requests.get(f"{API_BASE_URL}/api/v1/portfolio/mtm", timeout=3,
                             headers={"X-API-Key": os.getenv("API_KEY", "")}).json()
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
        except Exception: pass
    sidebar_portfolio()

    st.markdown("**导航**")
    tab = st.radio("", ["⚡ 实时行情", "📊 市场总览", "📋 全市场行情", "🔍 机会扫描", "📈 技术分析", "🧪 策略回测",
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
                async def run(): return await run_full_day(no_llm=True, dry_run=dry)
                result = _run_async(run())
                if result:
                    t = result.get("trade", {})
                    s = result.get("summary", {})
                    if dry:
                        st.success(f"分析完成! 筛选出 {t.get('buys',0)} 只BUY信号 | 耗时见日志")
                    else:
                        st.success(f"工作流完成! 总资产 RMB{s.get('total_value',0):,.2f} | "
                                   f"收益 {s.get('total_return_pct',0):+.2f}% | "
                                   f"买{t.get('bought',0)} 卖{t.get('sold',0)}")
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"工作流失败: {e}")

# ═══════════════════════════════════════════════════════════════
# Data Helpers
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_regime():
    try:
        from data.providers.base import DataFrequency, DataRequest
        async def f():
            req = DataRequest("sh.000300", today - timedelta(days=365), today, DataFrequency.DAILY)
            r = await router.get_daily_kline(req)
            return detector.detect(r.data)
        return _run_async(f())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

@st.cache_data(ttl=120, show_spinner=False)
def get_stock_quick(sym, days=90):
    try:
        from data.providers.base import DataFrequency, DataRequest
        async def f():
            req = DataRequest(sym, today - timedelta(days=days), today, DataFrequency.DAILY)
            r = await router.get_daily_kline(req)
            d = r.data
            if d.empty: return None
            ind = analyzer.compute_all(d, symbol=sym)
            return {"data": d, "indicators": ind, "last": ind.to_dataframe().iloc[-1],
                    "close": float(d["close"].iloc[-1]),
                    "change": float(d.get("pct_change", pd.Series([0])).iloc[-1]) if "pct_change" in d.columns else 0,
                    "name": NAME_MAP.get(sym, sym.split(".")[-1] if "." in sym else sym)}
        return _run_async(f())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def render_dataframe(df: pd.DataFrame, max_rows: int = 500, height: int = 400,
                     col_rename: dict = None, formatters: dict = None) -> None:
    """用 HTML 表格渲染 DataFrame, 绕过 st.dataframe 在暗色主题下的 CSS 冲突.

    Streamlit 1.60 的 GlideDataEditor 与自定义 .stDataFrame CSS 选择器不兼容,
    导致表格内容不可见。此函数直接生成 GitHub-dark 风格 HTML 表格。

    Args:
        df: 要渲染的 DataFrame
        max_rows: 最多显示行数
        height: 表格滚动区域高度 (px)
        col_rename: {原列名: 中文显示名} 映射
        formatters: {列名: format_spec} 如 {'price': '¥%.2f', 'pct_change': '%+.2f%%'}
    """
    col_rename = col_rename or {}
    formatters = formatters or {}

    df = df.head(max_rows)
    cols = df.columns.tolist()
    # 中文列名 (表头用)
    header_names = [col_rename.get(c, c) for c in cols]
    rows = df.values.tolist()

    html = f'<div style="max-height:{height}px;overflow-y:auto;border:1px solid var(--ds-border,#30363d);border-radius:8px">'
    html += '<table style="width:100%;border-collapse:collapse;font-size:13px">'

    # Header
    html += '<thead><tr style="background:var(--ds-bg3,#21262d);position:sticky;top:0;z-index:10">'
    for name in header_names:
        html += f'<th style="padding:6px 10px;text-align:left;color:var(--ds-text2,#8b949e);border-bottom:2px solid var(--ds-border,#30363d);white-space:nowrap">{name}</th>'
    html += '</tr></thead><tbody>'

    # Body — alternate row colors using CSS vars
    html += '<tr style="background:var(--ds-bg2,#161b22);border-bottom:1px solid var(--ds-bg3,#21262d)">'
    for i, row in enumerate(rows):
        if i > 0:
            bg = "var(--ds-bg,#0d1117)" if i % 2 == 0 else "var(--ds-bg2,#161b22)"
            html += f'<tr style="background:{bg};border-bottom:1px solid var(--ds-bg3,#21262d)">'
        for j, val in enumerate(row):
            col = cols[j] if j < len(cols) else ""
            fmt = formatters.get(col, "")
            # Use explicit formatter if provided
            if fmt and isinstance(val, (int, float)) and not (isinstance(val, float) and pd.isna(val)):
                tv = fmt % val
            elif isinstance(val, float):
                if abs(val) >= 1e8:
                    tv = f"{val:,.0f}"
                elif abs(val) >= 100:
                    tv = f"{val:,.1f}"
                elif abs(val) >= 1:
                    tv = f"{val:.2f}"
                else:
                    tv = f"{val:.4f}"
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                tv = "-"
            else:
                tv = str(val)
            # 右对齐数字列, 左对齐文本列
            align = "right" if isinstance(val, (int, float)) and not (isinstance(val, float) and pd.isna(val)) else "left"
            html += f'<td style="padding:4px 10px;color:var(--ds-text,#c9d1d9);white-space:nowrap;text-align:{align}">{tv}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_portfolio_state():
    """持仓快照 (60s 缓存) — 本地结构 + API MTM 实时价覆盖, 静态渲染不随 fragment 刷"""
    try:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        m = PortfolioManager()
        e = PaperTradingEngine(m)
        s = e.get_summary()
        # 用 API MTM 实时价覆盖 (否则 current_price=成本, 盈亏0)
        try:
            import requests
            d = requests.get(f"{API_BASE_URL}/api/v1/portfolio/mtm", timeout=5,
                             headers={"X-API-Key": os.getenv("API_KEY", "")}).json()
            live_pos = {p.get("symbol"): p for p in d.get("positions", [])}
            for p in s.get("positions", []):
                lp = live_pos.get(p.get("symbol"))
                if lp:
                    p["current_price"] = lp.get("current_price", p.get("current_price"))
                    p["market_value"] = lp.get("market_value", p.get("market_value"))
                    p["unrealized_pnl"] = lp.get("unrealized_pnl", 0)
                    p["unrealized_pnl_pct"] = lp.get("unrealized_pnl_pct", 0)
            s["total_value"] = d.get("summary", {}).get("total_value", s.get("total_value"))
            s["total_return"] = d.get("summary", {}).get("total_return", s.get("total_return"))
            s["total_return_pct"] = d.get("summary", {}).get("total_return_pct", s.get("total_return_pct"))
        except Exception:
            pass
        return s
    except: return None

@st.cache_data(ttl=300, show_spinner="正在加载全市场数据...")
def get_full_market():
    """获取全A股行情。AKShare → 本地缓存 → 东方财富直连 → 分析报告, 闭市时自动回退到缓存数据。"""
    import numpy as np

    cache_path = Path(__file__).parent.parent / "simulation_data" / "full_market_cache.json"
    source = None
    data_date = date.today().isoformat()
    df = None

    # ── 尝试 1: AKShare (交易时段最佳) ──
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        source = "live"
    except Exception:
        pass

    # ── 尝试 2: 本地缓存 (优先于网络请求, 避免闭市时段 API 限流) ──
    if (df is None or df.empty) and cache_path.exists():
        try:
            import json as _json
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = _json.load(f)
            df = pd.DataFrame(cache.get("data", []))
            source = "cached"
            data_date = cache.get("date", "unknown")
        except Exception:
            pass

    # ── 尝试 3: 东方财富 API 直连 (闭市仍可获取前一日收盘数据, 带限速) ──
    if df is None or df.empty:
        try:
            df = _fetch_eastmoney_full_market()
            if df is not None and not df.empty:
                source = "live"
        except Exception:
            pass

    # ── 回退 4: 最新分析报告 ──
    if df is None or df.empty:
        import glob as _glob
        report_files = sorted(_glob.glob(str(Path(__file__).parent.parent / "reports" / "data_*.json")), reverse=True)
        if report_files:
            try:
                import json as _json
                with open(report_files[0], "r", encoding="utf-8") as f:
                    rpt = _json.load(f)
                prices = rpt.get("analysis_prices", {})
                rows = []
                for sym, info in prices.items():
                    code_short = sym.replace("sh.", "").replace("sz.", "").replace("bj.", "")
                    rows.append({
                        "code": code_short, "name": info.get("name", sym),
                        "price": info.get("close", 0), "pct_change": info.get("pct_change", 0),
                        "volume": info.get("volume", 0), "amount": info.get("amount", 0),
                        "turnover": info.get("turnover", 0), "pe_ttm": info.get("pe_ttm", 0),
                        "pb": info.get("pb", 0), "total_mv": info.get("total_mv", 0),
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    source = "report"
                    data_date = rpt.get("date", "unknown")
            except Exception:
                pass

    if df is None or df.empty:
        st.session_state._market_source = None
        return None

    # ── 标准化列名 (akshare 用中文列名) ──
    col_map = {
        "代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_change",
        "涨跌额": "change_amt", "成交量": "volume", "成交额": "amount",
        "振幅": "amplitude", "最高": "high", "最低": "low", "今开": "open", "昨收": "prev_close",
        "量比": "vol_ratio", "换手率": "turnover", "市盈率-动态": "pe_ttm",
        "市净率": "pb", "总市值": "total_mv", "流通市值": "float_mv",
        "60日涨跌幅": "pct_60d", "年初至今涨跌幅": "pct_ytd",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 过滤无效数据
    if "price" in df.columns:
        df = df[df["price"].notna() & (df["price"] > 0)]
    if "name" in df.columns:
        df = df[df["name"].notna() & (df["name"] != "")]

    # 添加交易所标识
    df["exchange"] = df["code"].apply(
        lambda x: "SH" if str(x).startswith(("6", "9")) else "SZ"
    )

    # 市值分位标记
    if "total_mv" in df.columns:
        mv = df["total_mv"].dropna()
        if len(mv) > 100:
            q80 = mv.quantile(0.8); q50 = mv.quantile(0.5); q20 = mv.quantile(0.2)
            df["mv_tier"] = df["total_mv"].apply(
                lambda x: "🟢大盘" if x > q80 else ("🟡中盘" if x > q50 else ("🟠小盘" if x > q20 else "🔴微盘"))
            )

    # ── 缓存成功的全市场数据 ──
    if source == "live" and len(df) > 1000:
        try:
            import json as _json
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {"date": data_date, "count": len(df), "source": "eastmoney",
                          "data": df.to_dict(orient="records")}
            with open(cache_path, "w", encoding="utf-8") as f:
                _json.dump(cache_data, f, ensure_ascii=False, default=str)
        except Exception:
            pass

    st.session_state._market_source = {"source": source, "date": data_date, "count": len(df)}
    return df


def _fetch_eastmoney_full_market():
    """东方财富全市场行情 API 直连 — 闭市时段分页拉取(带重试), 降级时返回部分数据"""
    import requests as _req
    import time as _time

    field_map = {
        "f2": "price", "f3": "pct_change", "f4": "change_amt",
        "f5": "volume", "f6": "amount", "f7": "amplitude", "f8": "turnover",
        "f9": "pe_ttm", "f10": "vol_ratio", "f12": "code", "f14": "name",
        "f15": "high", "f16": "low", "f17": "open", "f18": "prev_close",
        "f20": "total_mv", "f21": "float_mv",
    }
    fields = ",".join(field_map.keys())

    all_items = []
    page_size = 200
    max_pages = 28  # ~5500 / 200

    for page in range(1, max_pages + 1):
        url = (
            f"https://push2.eastmoney.com/api/qt/clist/get?"
            f"pn={page}&pz={page_size}&po=1&np=1&fltt=2&invt=2"
            f"&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            f"&fields={fields}"
        )
        try:
            resp = _req.get(url, timeout=15,
                          headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
            data = resp.json().get("data", {})
            items = data.get("diff") or []
            if not items:
                break
            for item in items:
                row = {}
                for api_key, col_name in field_map.items():
                    val = item.get(api_key)
                    if val == "-" or val is None:
                        val = 0
                    row[col_name] = val
                all_items.append(row)
            if len(items) < page_size:
                break
        except Exception:
            break
        # 闭市时段放慢速度, 避免触发反爬
        _time.sleep(0.3 if len(all_items) > 2000 else 0.15)

    if not all_items:
        return None
    return pd.DataFrame(all_items)

# 行业板块映射 (申万一级 — 基于代码区间)
def _guess_sector(code: str) -> str:
    code_str = str(code)
    if code_str.startswith(("60","68")):  # 上交所
        num = int(code_str[:3]) if len(code_str) >= 3 else 0
        if 36 <= num <= 39: return "银行"
        if 48 == num: return "券商"
        if code_str.startswith("688"): return "科创板"
        if num in (16, 19): return "能源"
        if num in (11, 15): return "交运"
        if num in (17, 18): return "材料"
        if num in (10, 58): return "工业"
        if num in (50, 51): return "消费"
        if num in (55, 56): return "医药"
        if num in (53, 54): return "地产"
        if num in (57, 59): return "可选消费"
        if num in (60, 61): return "金融"
        if num in (63, 64): return "科技"
        if num in (65, 66): return "公用事业"
        return "其他"
    else:  # 深交所
        num = int(code_str[:3]) if len(code_str) >= 3 else 0
        if num in (1, 2): return "主板"
        if num == 3: return "创业板"
        return "深市"


# ═══════════════════════════════════════════════════════════════
# Realtime Fragments — 各模块实时数据 (st.fragment auto-refresh)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=5, show_spinner=False)
def _rt_indices():
    """获取大盘指数实时数据 (5秒缓存)"""
    try:
        import requests
        idx_codes = {"上证":"sh000001","深证":"sz399001","创业板":"sz399006","科创50":"sh000688","沪深300":"sh000300"}
        url = f"https://qt.gtimg.cn/q={','.join(idx_codes.values())}"
        r = requests.get(url, timeout=5, headers={"User-Agent":"Mozilla/5.0","Referer":"https://finance.qq.com/"})
        r.encoding = "gbk"
        result = {}
        for line in r.text.split("\n"):
            if "=" not in line or "~" not in line: continue
            try:
                code = line.split("=",1)[0].replace("v_","").strip()
                fld = line.split("=",1)[1].strip('"').split("~")
                if len(fld)<10: continue
                name = {v:k for k,v in idx_codes.items()}.get(code,code)
                result[name] = {"price":float(fld[3]) if fld[3] else 0, "pct":float(fld[32]) if len(fld)>32 and fld[32] else 0}
            except: pass
        return result
    except: return {}

@st.cache_data(ttl=5, show_spinner=False)
def _rt_watchlist(symbols: list = None):
    """获取指定股票的实时价格 (5秒缓存)"""
    if symbols is None:
        symbols = ["sh600519","sz000858","sh601318","sz300750","sh600036","sz000333","sz000651",
                   "sh601088","sh600900","sh601899","sh600031","sz002594","sz002415","sh600276"]
    try:
        import requests
        url = f"https://qt.gtimg.cn/q={','.join(symbols)}"
        r = requests.get(url, timeout=5, headers={"User-Agent":"Mozilla/5.0","Referer":"https://finance.qq.com/"})
        r.encoding = "gbk"
        result = {}
        for line in r.text.split("\n"):
            if "=" not in line or "~" not in line: continue
            try:
                fld = line.split("=",1)[1].strip('"').split("~")
                if len(fld)<10: continue
                result[fld[2]] = {"name":fld[1],"price":float(fld[3]) if fld[3] else 0,
                                  "pct":float(fld[32]) if len(fld)>32 and fld[32] else 0}
            except: pass
        return result
    except: return {}

@st.fragment(run_every=10)
def fragment_live_indices():
    """实时指数条 — 嵌入市场总览顶部, 含市场状态"""
    import requests, json, os
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/realtime/market", timeout=15)
        data = r.json()
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
        clr = "#f85149" if pct>0 else ("#3fb950" if pct<0 else "var(--ds-text2,#8b949e)")
        arr = "▲" if pct>0 else ("▼" if pct<0 else "—")
        with cols[i]:
            st.markdown(f"""
            <div style="background:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:6px;padding:5px 6px;text-align:center">
                <div style="font-size:10px;color:var(--ds-text2,#8b949e)">{name}</div>
                <div style="font-size:15px;font-weight:700;color:var(--ds-text-h,#f0f6fc)">{d['price']:.0f}</div>
                <div style="font-size:11px;font-weight:600;color:{clr}">{arr} {pct:+.2f}%</div>
            </div>""", unsafe_allow_html=True)

@st.fragment(run_every=10)
def fragment_live_watchlist(symbols: list = None):
    """实时自选列表 — 市场总览中的精选池"""
    import requests
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/realtime/market", timeout=15)
        data = r.json()
        quotes_list = data.get("watchlist", [])
        quotes = {}
        for q in quotes_list:
            quotes[q.get("code", "")] = q
    except Exception:
        quotes = {}

    if not quotes:
        st.caption("实时数据暂不可用")
        return
    items = sorted(quotes.items(), key=lambda x: x[1].get("pct", 0), reverse=True)
    rows = ""
    for code, d in items[:15]:
        pct = d.get("pct", 0)
        clr = "#f85149" if pct>0 else ("#3fb950" if pct<0 else "var(--ds-text2,#8b949e)")
        rows += f'<tr style="border-bottom:1px solid var(--ds-bg3,#21262d)">' \
            f'<td style="padding:2px 6px;font-size:12px;color:var(--ds-text,#c9d1d9)">{d.get("name","")}</td>' \
            f'<td style="padding:2px 6px;font-size:10px;color:var(--ds-text2,#8b949e)">{code}</td>' \
            f'<td style="padding:2px 6px;font-size:12px;color:var(--ds-text-h,#f0f6fc);text-align:right">¥{d.get("price",0):.2f}</td>' \
            f'<td style="padding:2px 6px;font-size:12px;font-weight:600;text-align:right;color:{clr}">{pct:+.2f}%</td></tr>'
    st.markdown(f"""
    <div style="background:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:6px;padding:6px;max-height:300px;overflow-y:auto">
    <table style="width:100%;border-collapse:collapse">
    <thead><tr style="border-bottom:2px solid var(--ds-border,#30363d)">
        <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">名称</th>
        <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:left">代码</th>
        <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:right">现价</th>
        <th style="font-size:10px;color:var(--ds-text2,#8b949e);text-align:right">涨跌</th>
    </tr></thead><tbody>{rows}</tbody></table></div>""", unsafe_allow_html=True)

@st.fragment(run_every=30)
def fragment_portfolio_live():
    """实时持仓 — 只刷实时价格字段 (紧凑), 完整表静态渲染, 避免整页刷新"""
    try:
        import requests
        d = requests.get(f"{API_BASE_URL}/api/v1/portfolio/mtm", timeout=5,
                         headers={"X-API-Key": os.getenv("API_KEY", "")}).json()
        s = d.get("summary", {})
        ps = d.get("positions", [])
        if not s.get("total_value"): return

        # 只刷实时变化值: 总资产 + 每只现价/盈亏 (紧凑一行, 原地更新)
        ret_clr = "#f85149" if (s.get("total_return",0)) >= 0 else "#3fb950"
        st.markdown(
            f"<span style='color:var(--ds-text2,#8b949e);font-size:11px'>实时总资产 </span>"
            f"<span style='color:var(--ds-text-h,#f0f6fc);font-weight:700'>¥{s['total_value']:,.0f}</span> "
            f"<span style='color:{ret_clr}'>{(s.get('total_return_pct',0)):+.2f}%</span>"
            f"<span style='color:var(--ds-text2,#8b949e);font-size:10px'> · 实时刷新中</span>",
            unsafe_allow_html=True,
        )
        if ps:
            cells = ""
            for p in ps:
                pnl_clr = "#f85149" if p["unrealized_pnl"]>0 else ("#3fb950" if p["unrealized_pnl"]<0 else "var(--ds-text2,#8b949e)")
                cells += (f"<span style='font-size:11px;color:var(--ds-text,#c9d1d9);margin-right:10px'>"
                          f"{p['name']} <b style='color:var(--ds-text-h,#f0f6fc)'>¥{p['current_price']:.2f}</b> "
                          f"<span style='color:{pnl_clr}'>{p['unrealized_pnl']:+.0f}</span></span>")
            st.markdown(
                f"<div style='font-size:11px;padding:4px 0'>{cells}</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# Tab 0: 实时行情 — Live auto-refreshing market data
# ═══════════════════════════════════════════════════════════════
if tab == "⚡ 实时行情":
    from web.realtime import realtime_section
    realtime_section()

# ═══════════════════════════════════════════════════════════════
# Tab 1: 市场总览 — Market Status + Watchlist
# ═══════════════════════════════════════════════════════════════
if tab == "📊 市场总览":
    # --- 实时指数 (10秒刷新) ---
    fragment_live_indices()

    regime = get_regime()

    # --- Status Bar ---
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    if regime:
        emoji = {"strong_bull":"🟢强牛","weak_bull":"🟡弱牛","range_bound":"⚪震荡","weak_bear":"🟠弱熊","strong_bear":"🔴强熊","crisis":"💀危机"}
        sc1.metric("市场状态", f"{emoji.get(regime.regime.value,'?')} {regime.regime.value}")
        sc2.metric("置信度", f"{regime.confidence:.0%}")
        sc3.metric("价格结构", f"{regime.scores.get('price_structure',0):.2f}")
        sc4.metric("动量", f"{regime.scores.get('momentum',0):.2f}")
        sc5.metric("波动率", f"{regime.scores.get('volatility',0):.2f}")
        sc6.metric("成交量", f"{regime.scores.get('volume',0):.2f}")

    # --- 精选池 (API实时数据) ---
    import requests
    try:
        rt = requests.get(f"{API_BASE_URL}/api/v1/realtime/market", timeout=15).json()
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
                                 col_rename={"Code":"代码","Name":"名称","Price":"现价",
                                             "Change%":"涨跌幅","Signal":"信号"},
                                 formatters={"Price":"¥%.2f","Change%":"%+.2f%%"})
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
                scols[i].info(f"**{s['name']}**  \n`{s['id']}`  \n{s.get('category','')}")


# ═══════════════════════════════════════════════════════════════
# Tab 2: 全市场行情 — Full Market Real-time Quotes (~5800 stocks)
# ═══════════════════════════════════════════════════════════════
elif tab == "📋 全市场行情":
    st.subheader("全市场实时行情")

    # 先加载数据
    df_market = get_full_market()

    # --- AI Pre-Screening ---
    ai_col1, ai_col2, ai_col3 = st.columns([2, 1, 1])
    with ai_col1:
        st.caption("🚀 用免费大模型(智谱GLM-4.7-Flash)并行预筛全市场, 选出Top100再交给DeepSeek深度分析")
    with ai_col2:
        top_n = st.number_input("筛选数量", 20, 200, 100, step=20, key="screen_top_n")
    with ai_col3:
        do_screen = st.button("🚀 AI预筛", type="primary", use_container_width=True,
                              help="将全市场数据分批发送给免费模型并行分析, 选出最优标的")

    if do_screen:
        if df_market is None:
            st.warning("无法获取全市场数据 — 闭市时段数据源可能不可用。请在工作日 9:30-15:00 重试，或先运行 `python -m simulation.daily_runner` 生成分析报告。")
        else:
            try:
                from analysis.market_screener import BatchScreener, rule_based_prefilter
                from web.progress_tracker import ProgressTracker
                import akshare as ak

                screener = BatchScreener()
                available = [m.name for m in screener.models]
                if not available:
                    st.error("❌ 没有可用的免费模型! 请在 .env 中配置 ZHIPU_API_KEY")
                else:
                    # ── 阶段1: 规则预筛 ──
                    tracker = ProgressTracker(
                        pipeline_name=f"全市场AI预筛 → Top{top_n}",
                        total_stocks=len(df_market),
                    )
                    tracker_placeholder = st.empty()

                    # Stage 1: 数据加载 (already done)
                    tracker.update(stage=0, log_msg=f"数据加载完成: {len(df_market):,} 只")
                    tracker.set_stage_total(0, 1)
                    tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    # Stage 2: 规则预筛
                    tracker.update(stage=1, log_msg="规则预筛启动...")
                    tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    df_filtered = rule_based_prefilter(df_market, top_n=max(300, top_n * 3))
                    tracker.set_stage_total(1, len(df_filtered))
                    tracker.update(stage=1, stocks_done=len(df_filtered),
                                   log_msg=f"规则预筛完成: {len(df_market):,} → {len(df_filtered):,} 只")
                    tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    # Stage 3: LLM精筛
                    batch_size = 50
                    total_batches = (len(df_filtered) + batch_size - 1) // batch_size
                    tracker.total_batches = total_batches
                    tracker.total_stocks = len(df_filtered)
                    tracker.set_stage_total(2, total_batches)
                    tracker.update(stage=2, log_msg=f"LLM精筛: {total_batches} 批 × {batch_size} 只, {len(available)} 模型并行")
                    tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    async def progress_cb(idx, total, stats):
                        model_name = available[idx % len(available)] if available else ""
                        tracker.update(
                            batch_idx=idx + 1,
                            stocks_done=min(stats['processed'], len(df_filtered)),
                            errors=stats['errors'],
                            stage=2,
                            model=model_name,
                            log_msg=f"批次{idx+1}/{total} | 已分析{stats['processed']}只 | 错误{stats['errors']}",
                        )
                        tracker.total_batches = total
                        tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    async def run_screen():
                        return await screener.screen(df_market, top_n=top_n, progress_callback=progress_cb)

                    top_result, stats = _run_async(run_screen())

                    # Stage 4: 完成
                    if top_result:
                        tracker.update(stage=3, stocks_done=len(df_filtered),
                                       log_msg=f"预筛完成! {len(top_result)} 只候选, 耗时{stats['elapsed']}s")
                        tracker.set_stage_total(3, len(top_result))
                        tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                        st.success(f"✅ 预筛完成! 从 {stats['total']:,} 只中选出 Top {len(top_result)} | "
                                   f"耗时 {stats['elapsed']}s | 模型: {', '.join(available)}")

                        df_result = pd.DataFrame(top_result)
                        render_dataframe(df_result, height=500)

                        # ── DeepSeek深度分析 ──
                        st.divider()
                        dc1, dc2 = st.columns([2, 1])
                        with dc1:
                            st.caption("将预筛结果交给 DeepSeek 做深度分析 (技术面+基本面+风险)")
                        with dc2:
                            if st.button("🔬 DeepSeek深度分析", type="secondary", use_container_width=True):
                                # DeepSeek progress tracker
                                deep_tracker = ProgressTracker(
                                    pipeline_name="DeepSeek深度分析",
                                    total_stocks=len(top_result),
                                )
                                deep_tracker.set_stage_total(0, 1)
                                deep_tracker.update(stage=0, log_msg=f"准备分析 {len(top_result)} 只候选股")
                                deep_placeholder = st.empty()

                                deep_batches = (len(top_result) + 19) // 20
                                deep_tracker.total_batches = deep_batches
                                deep_tracker.set_stage_total(1, deep_batches)
                                deep_tracker.update(stage=1, model="DeepSeek-Chat",
                                                    log_msg=f"启动 {deep_batches} 批深度分析")

                                async def deep_progress_cb(idx, total, stats):
                                    deep_tracker.update(
                                        batch_idx=idx + 1, stocks_done=min((idx+1)*20, len(top_result)),
                                        errors=stats.get('errors', 0), stage=1,
                                        log_msg=f"深度分析 {idx+1}/{total} | {min((idx+1)*20,len(top_result))}只完成",
                                    )
                                    deep_tracker.total_batches = total
                                    deep_placeholder.markdown(deep_tracker.render_html(), unsafe_allow_html=True)

                                async def run_deep():
                                    return await screener.screen_with_deepseek(df_market, top_n=top_n)

                                deep_results = _run_async(run_deep())
                                if deep_results:
                                    deep_tracker.update(stage=2, stocks_done=len(deep_results),
                                                        log_msg=f"深度分析完成: {len(deep_results)} 只")
                                    deep_placeholder.markdown(deep_tracker.render_html(), unsafe_allow_html=True)

                                    st.success(f"深度分析完成: {len(deep_results)} 只")
                                    df_deep = pd.DataFrame(deep_results)
                                    render_dataframe(df_deep, height=500)
                    else:
                        tracker.update(log_msg="预筛未返回结果!", errors=1)
                        tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)
                        st.warning("预筛未返回结果, 请检查模型配置和网络")
            except Exception as e:
                st.error(f"预筛失败: {e}")

    if df_market is None:
        st.warning("无法获取全市场数据。可能原因：网络未连接、闭市时段数据源暂时不可用。请在工作日 9:30-15:00 之间查看实时行情。")

        # 引导用户使用缓存
        cache_path_check = Path(__file__).parent.parent / "simulation_data" / "full_market_cache.json"
        if cache_path_check.exists():
            st.caption("💡 提示: 点击下方「刷新全部」清除缓存后重新加载可能会恢复数据。")
            if st.button("🔄 强制刷新数据", key="force_refresh_market"):
                st.cache_data.clear()
                st.rerun()
    else:
        # ── 数据来源横幅 ──
        ms = st.session_state.get("_market_source", {})
        src = ms.get("source", "unknown")
        data_date = ms.get("date", "N/A")
        count = ms.get("count", len(df_market))

        if src == "live":
            st.success(f"✅ 实时行情 — {count:,} 只 | 数据时间: {data_date}")
        elif src == "cached":
            st.info(f"⏸️ 市场已收盘 — 显示最近缓存数据 ({data_date}) | {count:,} 只 | 非实时行情, 仅供参考")
        elif src == "report":
            st.info(f"⏸️ 市场已收盘 — 显示最近分析数据 ({data_date}) | {count:,} 只 | 仅含当日被分析的标的, 非全市场行情")
        else:
            st.caption(f"数据来源: {src} | {count:,} 只 | {data_date}")

        total_stocks = len(df_market)
        # --- Top Bar Metrics ---
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        up_count = len(df_market[df_market["pct_change"] > 0]) if "pct_change" in df_market.columns else 0
        down_count = len(df_market[df_market["pct_change"] < 0]) if "pct_change" in df_market.columns else 0
        flat_count = total_stocks - up_count - down_count
        m1.metric("📊 总计", f"{total_stocks:,}只")
        m2.metric("🟢 上涨", f"{up_count}", f"{up_count/total_stocks*100:.0f}%")
        m3.metric("🔴 下跌", f"{down_count}", f"{down_count/total_stocks*100:.0f}%")
        m4.metric("⚪ 平盘", f"{flat_count}")
        if "amount" in df_market.columns:
            total_amt = df_market["amount"].sum() / 1e8
            m5.metric("💰 成交额", f"{total_amt:,.0f}亿")
        avg_pct = df_market["pct_change"].mean() if "pct_change" in df_market.columns else 0
        m6.metric("📈 均涨跌", f"{avg_pct:+.2f}%")

        # --- Filters ---
        st.divider()
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            search = st.text_input("🔍 搜索代码/名称", placeholder="eg: 茅台 or 600519")
        with f2:
            price_range = st.selectbox("💰 价格", ["全部", "0-10元", "10-30元", "30-100元", "100-500元", "500元以上"])
        with f3:
            mv_range = st.selectbox("🏢 市值", ["全部", "🟢大盘", "🟡中盘", "🟠小盘", "🔴微盘"])
        with f4:
            sort_by = st.selectbox("📊 排序", ["涨跌幅↓", "涨跌幅↑", "最新价↓", "最新价↑", "成交额↓", "市盈率↓", "换手率↓"])
        with f5:
            exchange_filter = st.selectbox("🏛 交易所", ["全部", "SH(沪市)", "SZ(深市)"])

        # --- Apply Filters ---
        df_filtered = df_market.copy()

        if search:
            search_lower = search.lower().strip()
            df_filtered = df_filtered[
                df_filtered["name"].str.contains(search_lower, na=False) |
                df_filtered["code"].astype(str).str.contains(search_lower, na=False)
            ]

        if price_range != "全部":
            lo, hi = 0, 999999
            if price_range == "0-10元": hi = 10
            elif price_range == "10-30元": lo, hi = 10, 30
            elif price_range == "30-100元": lo, hi = 30, 100
            elif price_range == "100-500元": lo, hi = 100, 500
            elif price_range == "500元以上": lo = 500
            df_filtered = df_filtered[(df_filtered["price"] >= lo) & (df_filtered["price"] < hi)]

        if mv_range != "全部" and "mv_tier" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["mv_tier"] == mv_range]

        if exchange_filter != "全部":
            ex = "SH" if "沪" in exchange_filter else "SZ"
            df_filtered = df_filtered[df_filtered["exchange"] == ex]

        # --- Sort ---
        sort_col_map = {
            "涨跌幅↓": ("pct_change", False),
            "涨跌幅↑": ("pct_change", True),
            "最新价↓": ("price", False),
            "最新价↑": ("price", True),
            "成交额↓": ("amount", False),
            "市盈率↓": ("pe_ttm", False),
            "换手率↓": ("turnover", False),
        }
        sort_col, sort_asc = sort_col_map.get(sort_by, ("pct_change", False))
        if sort_col in df_filtered.columns:
            df_filtered = df_filtered.sort_values(sort_col, ascending=sort_asc, na_position="last")

        st.caption(f"显示 {len(df_filtered):,} 只 (全市场 {total_stocks:,} 只)")

        # --- Display Table (HTML渲染, 绕过st.dataframe暗色主题CSS冲突) ---
        display_cols = ["code","name","price","pct_change","volume","amount",
                       "turnover","pe_ttm","pb","total_mv","amplitude","mv_tier"]
        available = [c for c in display_cols if c in df_filtered.columns]
        df_show = df_filtered[available].head(500)
        col_cn = {
            "code": "代码", "name": "名称", "price": "现价", "pct_change": "涨跌幅",
            "volume": "成交量", "amount": "成交额", "turnover": "换手率",
            "pe_ttm": "市盈率", "pb": "市净率", "total_mv": "总市值",
            "amplitude": "振幅", "mv_tier": "市值档",
        }
        fmts = {
            "price": "¥%.2f", "pct_change": "%+.2f%%", "volume": "%.0f",
            "amount": "¥%.0f", "turnover": "%.2f%%", "pe_ttm": "%.1f",
            "pb": "%.2f", "total_mv": "¥%.0f", "amplitude": "%.2f%%",
        }
        render_dataframe(df_show, height=600, col_rename=col_cn, formatters=fmts)

        # --- Top Movers ---
        st.divider()
        l1, l2 = st.columns(2)
        with l1:
            st.subheader("🚀 涨幅榜 TOP10")
            if "pct_change" in df_market.columns:
                top_up = (df_market.nlargest(10, "pct_change")
                          [["code","name","price","pct_change","amount","turnover"]]
                          if all(c in df_market.columns for c in ["code","name","price","pct_change","amount","turnover"])
                          else df_market.nlargest(10, "pct_change")[["code","name","price","pct_change"]])
                render_dataframe(top_up, max_rows=10, height=350,
                                 col_rename={"code":"代码","name":"名称","price":"现价",
                                             "pct_change":"涨跌幅","amount":"成交额","turnover":"换手率"},
                                 formatters={"price":"¥%.2f","pct_change":"%+.2f%%",
                                             "amount":"¥%.0f","turnover":"%.2f%%"})
        with l2:
            st.subheader("📉 跌幅榜 TOP10")
            if "pct_change" in df_market.columns:
                top_down = (df_market.nsmallest(10, "pct_change")
                            [["code","name","price","pct_change","amount","turnover"]]
                            if all(c in df_market.columns for c in ["code","name","price","pct_change","amount","turnover"])
                            else df_market.nsmallest(10, "pct_change")[["code","name","price","pct_change"]])
                render_dataframe(top_down, max_rows=10, height=350,
                                 col_rename={"code":"代码","name":"名称","price":"现价",
                                             "pct_change":"涨跌幅","amount":"成交额","turnover":"换手率"},
                                 formatters={"price":"¥%.2f","pct_change":"%+.2f%%",
                                             "amount":"¥%.0f","turnover":"%.2f%%"})

        # --- Sector Heatmap ---
        st.divider()
        st.subheader("板块热力图")
        df_m = df_market.copy()
        df_m["sector"] = df_m["code"].apply(lambda x: _guess_sector(str(x)))
        sector_stats = df_m.groupby("sector").agg(
            数量=("code","count"),
            均涨幅=("pct_change","mean"),
            总成交额亿=("amount","sum"),
            平均PE=("pe_ttm","mean"),
        ).reset_index()
        sector_stats["总成交额亿"] = sector_stats["总成交额亿"] / 1e8
        sector_stats = sector_stats.sort_values("均涨幅", ascending=False)
        sector_stats["趋势"] = sector_stats["均涨幅"].apply(
            lambda x: "🟢" if x > 1 else ("🟡" if x > -1 else "🔴")
        )

        scols = st.columns(min(len(sector_stats), 4))
        for i, (_, row) in enumerate(sector_stats.iterrows()):
            col = scols[i % 4]
            with col:
                bg = "#1a2f1a" if row["均涨幅"] > 1 else ("#2d1f1a" if row["均涨幅"] < -1 else "#1a2332")
                st.markdown(f"""
                <div style="background:{bg};border:1px solid var(--ds-border,#30363d);border-radius:8px;padding:10px 14px;margin:3px 0">
                    <div style="font-size:14px;color:var(--ds-text-h,#f0f6fc)">{row['趋势']} <b>{row['sector']}</b></div>
                    <div style="font-size:22px;color:{'#3fb950' if row['均涨幅']>0 else '#f85149'}">{row['均涨幅']:+.1f}%</div>
                    <div style="font-size:11px;color:var(--ds-text2,#8b949e)">{row['数量']}只 | 成交{row['总成交额亿']:.0f}亿 | PE{row['平均PE']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Tab 3: 机会扫描 — Market-wide opportunity scanning
# ═══════════════════════════════════════════════════════════════
elif tab == "🔍 机会扫描":
    st.subheader("市场机会扫描")

    c1, c2, c3 = st.columns(3)
    with c1: min_score = st.slider("最低综合分", 40, 90, 55)
    with c2: top_n = st.slider("返回数量", 5, 50, 15)
    with c3:
        if st.button("🔍 开始扫描", type="primary", use_container_width=True):
            with st.spinner("Scanning..."):
                try:
                    from analysis.recommender import RecommendationEngine
                    engine = RecommendationEngine(router=router, knowledge=knowledge, analyzer=analyzer)
                    async def g(): return await engine.generate(capital=100000, top_n=top_n, min_composite=min_score)
                    recs = _run_async(g())

                    if recs.recommendations:
                        st.success(f"扫描{recs.total_scanned}只 → {len(recs.recommendations)}个机会")

                        # Signal Cards
                        cols = st.columns(3)
                        for i, r in enumerate(recs.recommendations[:9]):
                            with cols[i % 3]:
                                grade_color = {"A":"#3fb950","B":"#58a6ff","C":"#f0883e","D":"#f85149"}.get(r.confidence, "var(--ds-text2,#8b949e)")
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


# ═══════════════════════════════════════════════════════════════
# Tab 4: 技术分析 — Candlestick + Indicators deep dive
# ═══════════════════════════════════════════════════════════════
elif tab == "📈 技术分析":
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
    sym = all_stock_map.get(selected_code,
            f"sh.{selected_code}" if selected_code.startswith("6") else f"sz.{selected_code}")

    info = get_stock_quick(sym, 365)
    if info:
        df = info["data"]
        ind = info["indicators"]
        last = info["last"]

        # --- KPI Row ---
        mc = st.columns(8)
        mc[0].metric("现价", f"¥{info['close']:.2f}", f"{info.get('change',0):+.2f}%")
        mc[1].metric("RSI(14)", f"{last.get('rsi_14',0):.0f}")
        mc[2].metric("Trend", f"{last.get('trend_score',0):.2f}")
        mc[3].metric("Composite", f"{last.get('composite_score',0):.0f}")
        mc[4].metric("Vol Ratio", f"{last.get('vol_ratio_5',0):.1f}")
        mc[5].metric("MACD", f"{last.get('macd_hist',0):.3f}")
        mc[6].metric("ATR(14)", f"{last.get('atr_14',0):.2f}")
        mc[7].metric("Bias MA20", f"{last.get('bias_ma20',0):.1f}%")

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
        for ma, color, label in [("ma_5","#f0883e","MA5"),("ma_20","#58a6ff","MA20"),("ma_60","#a371f7","MA60")]:
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
            st.metric("均线排列", "多头" if last.get("ma_bullish",0) else "空头")
            st.metric("ADX(14)", f"{last.get('adx_14',0):.1f}")
            st.metric("+DI/-DI", f"{last.get('plus_di_14',0):.1f}/{last.get('minus_di_14',0):.1f}")
        with ic2:
            st.caption("动量")
            st.metric("RSI(14)", f"{last.get('rsi_14',0):.0f}")
            st.metric("MACD柱", f"{last.get('macd_hist',0):.3f}")
            st.metric("ROC(5)", f"{last.get('roc_5',0):.1f}%")
        with ic3:
            st.caption("波动率")
            st.metric("ATR(14)", f"{last.get('atr_14',0):.2f}")
            st.metric("布林宽度", f"{last.get('bb_width_20',0):.2f}")
            st.metric("HV(20)", f"{last.get('hv_20',0):.1f}%")
        with ic4:
            st.caption("成交量")
            st.metric("量比5日", f"{last.get('vol_ratio_5',0):.2f}")
            st.metric("OBV趋势", "向上" if last.get("obv",0) > last.get("obv_ma_20",0) else "向下")
            st.metric("换手率", f"{last.get('turnover',0):.1f}%")

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

                        # 渲染AI分析结果
                        st.markdown(f"""
                        <div style="background:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:12px;padding:20px 24px;margin-top:8px">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
                        <span style="font-size:20px">🤖</span>
                        <span style="color:#58a6ff;font-weight:600;font-size:15px">DeepSeek V4-Flash 技术分析 — {name}({sym})</span>
                        <span style="color:var(--ds-text2,#8b949e);font-size:12px;margin-left:auto">模型: deepseek-v4-flash | 仅供参考，不构成投资建议</span>
                        </div>
                        <div style="color:var(--ds-text,#c9d1d9);font-size:14px;line-height:1.8;white-space:pre-wrap">{ai_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"AI分析请求失败: {e}")
    else:
        st.warning("该标的数据不可用")


# ═══════════════════════════════════════════════════════════════
# Tab 5: 策略回测 — Strategy comparison
# ═══════════════════════════════════════════════════════════════
elif tab == "🧪 策略回测":
    st.subheader("策略回测")

    strats = knowledge.list_strategies()
    c1, c2, c3, c4 = st.columns(4)
    with c1: sid = st.selectbox("策略", [s["id"] for s in strats],
                                format_func=lambda x: next((s["name"] for s in strats if s["id"]==x), x))
    with c2:
        fm_stocks = load_full_market_stocks()
        fm_list = sorted(fm_stocks.keys())
        bts = st.selectbox("标的 (全市场)", fm_list,
                          format_func=lambda x: f"{x} | {NAME_MAP.get(x, '')}",
                          help=f"共 {len(fm_list):,} 只")
    sym = fm_stocks.get(bts, f"sh.{bts}" if bts.startswith("6") else f"sz.{bts}")
    with c3: yrs = st.slider("回测年数", 1, 5, 3)
    with c4:
        if st.button("▶ 运行", type="primary", use_container_width=True):
            with st.spinner("回测中..."):
                try:
                    from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
                    from data.providers.base import DataRequest, DataFrequency
                    from analysis.recommender import STRATEGY_BACKTESTERS

                    async def bt():
                        req = DataRequest(sym, today - timedelta(days=365*yrs), today, DataFrequency.DAILY)
                        data = await router.get_daily_kline(req)
                        df = data.data
                        cfg = BacktestConfig(100000, df["date"].min(), df["date"].max())
                        eng = EventDrivenBacktestEngine(cfg)
                        eng.load_data(sym, df)
                        bt_func = STRATEGY_BACKTESTERS.get(sid)
                        sbt = bt_func(df) if bt_func else {}

                        def strat(today_, bars, broker):
                            if sym not in bars: return
                            c = bars[sym]["close"]
                            if not hasattr(strat,"_e"): strat._e = None
                            if strat._e is None:
                                q = int(broker.account.cash * 0.3 / c / 100) * 100
                                if q >= 100: broker.buy(sym, q); strat._e = today_
                            else:
                                p = broker.account.positions.get(sym)
                                if p and (today_ - strat._e).days >= 10:
                                    broker.sell(sym, p.quantity); strat._e = None
                        strat._e = None
                        return eng.run(strat, progress_bar=False), sbt

                    r, sbt = _run_async(bt())

                    # Results
                    pc = st.columns(4)
                    pc[0].metric("总收益", f"{r.total_return:+.1f}%")
                    pc[1].metric("年化", f"{r.annual_return:+.1f}%")
                    pc[2].metric("夏普", f"{r.sharpe_ratio:.2f}")
                    pc[3].metric("最大回撤", f"{r.max_drawdown:.1f}%")
                    pc2 = st.columns(4)
                    pc2[0].metric("索提诺", f"{r.sortino_ratio:.2f}")
                    pc2[1].metric("胜率", f"{r.win_rate:.1%}")
                    pc2[2].metric("VaR95", f"{r.var_95:.1f}%")
                    pc2[3].metric("Avg PnL", f"{r.avg_trade_pnl:+.1f}%" if hasattr(r, 'avg_trade_pnl') else "N/A")

                    # Equity curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=r.equity_curve, name="Equity",
                        line=dict(color="#58a6ff", width=2), fill="tozeroy",
                        fillcolor="rgba(88,166,255,0.1)"))
                    fig.update_layout(height=300, template="plotly_white",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    if sbt:
                        st.caption(f"Strategy Stats: {sbt.get('signals',0)} 个信号 | "
                                  f"胜率 {sbt.get('win_rate',0):.1%} | "
                                  f"PF {sbt.get('profit_factor',1):.1f}x | "
                                  f"Sharpe {sbt.get('sharpe',0):.2f} | "
                                  f"含交易成本: {sbt.get('cost_adjusted',False)}")
                except Exception as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════
# Tab 6: AI信号 — AI Recommendations
# ═══════════════════════════════════════════════════════════════
elif tab == "📋 AI信号":
    st.subheader("AI交易信号")

    # Chat-style signal generation
    if "signal_chat" not in st.session_state:
        st.session_state.signal_chat = []

    for msg in st.session_state.signal_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("向AI提问，如: 'Scan for buy signals' or 'Analyze 600519 with MACD strategy'"):
        st.session_state.signal_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    from agent.chat_agent import ChatAgent
                    agent = ChatAgent(router=router, knowledge=knowledge, analyzer=analyzer)
                    async def c(): return await agent.chat(prompt, session_id="dashboard")
                    reply = _run_async(c())
                    st.markdown(reply)
                    st.session_state.signal_chat.append({"role": "assistant", "content": reply})
                except Exception as e: st.error(str(e))

    if st.session_state.signal_chat:
        if st.button("清除历史"):
            st.session_state.signal_chat = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# Tab 7: 模拟持仓 — Paper Trading Dashboard
# ═══════════════════════════════════════════════════════════════
elif tab == "💰 模拟持仓":
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
        loss_count = sum(1 for p in positions if p.get("unrealized_pnl", 0) < 0)

        from web.viz_components import make_radar_chart, make_equity_curve, make_heatmap

        # ── 实时价格 (紧凑 fragment, 只刷该字段) ──
        fragment_portfolio_live()

        # ── 完整持仓表 (静态, 快照渲染一次, 不随 fragment 刷新) ──
        if positions:
            rows = ""
            for p in positions:
                pnl = float(p.get("unrealized_pnl", 0))
                pnl_clr = "#f85149" if pnl > 0 else ("#3fb950" if pnl < 0 else "var(--ds-text2,#8b949e)")
                rows += (f'<tr><td style="color:var(--ds-text,#c9d1d9)">{p.get("name","")}</td>'
                         f'<td style="color:var(--ds-text2,#8b949e);font-size:10px">{p.get("symbol","")}</td>'
                         f'<td style="color:var(--ds-text-h,#f0f6fc)">{p.get("quantity",0)}</td>'
                         f'<td style="color:var(--ds-text2,#8b949e)">¥{float(p.get("avg_cost",0)):.2f}</td>'
                         f'<td style="color:var(--ds-text-h,#f0f6fc);font-weight:600">¥{float(p.get("current_price",0)):.2f}</td>'
                         f'<td style="color:var(--ds-text-h,#f0f6fc)">¥{float(p.get("market_value",0)):,.0f}</td>'
                         f'<td style="color:{pnl_clr};font-weight:600">¥{pnl:+,.0f} ({float(p.get("unrealized_pnl_pct",0)):+.1f}%)</td>'
                         f'<td style="color:#f85149;font-size:10px">¥{float(p.get("stop_loss",0) or 0):.2f}</td>'
                         f'<td style="color:#3fb950;font-size:10px">¥{float(p.get("take_profit",0) or 0):.2f}</td></tr>')
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
            pie_data = {p.get("name", p.get("symbol","")): p.get("market_value", 0) for p in positions}
            if portfolio["cash"] > 0:
                pie_data["现金"] = portfolio["cash"]
            fig_pie = px.pie(names=list(pie_data.keys()), values=list(pie_data.values()),
                            title="资产配置", color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie.update_layout(height=260, template="plotly_white",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Tab 8: 风控中心 — 策略历史回测 + 8层风控
# ═══════════════════════════════════════════════════════════════
elif tab == "🛡️ 风控中心":
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
    # 策略历史回测 (核心新增)
    # ═══════════════════════════════════════════════════════════════
    st.subheader("策略历史回测排名")
    st.caption("每策略在10只蓝筹股(2022-2025)上的汇总表现 | 运行 python -m backtest.strategy_backtest 刷新")

    import json as _json
    bt_path = Path(__file__).parent.parent / "reports" / "strategy_backtest.json"
    if bt_path.exists():
        try:
            with open(bt_path, "r", encoding="utf-8") as f:
                bt_data = _json.load(f)
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
                    grade_label = {"exc":"A","good":"B","ok":"C","weak":"D"}
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
                        "评级": grade_label.get(r.get("grade",""), r.get("grade","")),
                    })

                if rows:
                    top_strat = ranked[0][1] if ranked else {}
                    low_risk_count = sum(1 for r in ranked if r[1].get("overfit_risk") == "low")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("最佳策略", top_strat.get("name", ""),
                              f"夏普{top_strat.get('sharpe',0):.1f}")
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
                            st.caption(f"[{of_mark.get(risk,'?')}] {r.get('name', sid)} / "
                                      f"ValSR {r.get('val_sr',0):+.1f} / {len(details)} stocks")
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
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{d.get("symbol","")}</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text2,#8b949e)">{d.get("trades",0)}</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{wr:.0f}%</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{d.get("expected_value",0):+.1f}</td>'
                                    h += f'<td style="padding:2px 6px;color:{clr};font-weight:600">{sr:+.2f}</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{d.get("max_dd",0):.1f}</td>'
                                    h += f'<td style="padding:2px 6px;color:var(--ds-text,#c9d1d9)">{d.get("profit_factor",0):.2f}</td>'
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
    with c1: demo_score = st.slider("综合评分", 0, 100, 65, key="sg_score")
    with c2: demo_wr = st.slider("策略胜率", 0.0, 1.0, 0.55, 0.05, key="sg_wr")
    with c3: demo_regime = st.selectbox("市场状态", ["strong_bull","weak_bull","range_bound","weak_bear","strong_bear"], key="sg_regime")

    try:
        from analysis.risk_controls import SignalGrader
        card = SignalGrader.grade(demo_score, demo_wr, demo_regime)
        grade_colors = {"STRONG_BUY":"#3fb950","BUY":"#58a6ff","WATCH":"#f0883e",
                       "HOLD":"var(--ds-text2,#8b949e)","SELL":"#f85149","STRONG_SELL":"#f85149"}
        color = grade_colors.get(card.level.label, "var(--ds-text2,#8b949e)")
        st.markdown(f"""
        <div style="background-color:var(--ds-bg2,#161b22);border:2px solid {color};border-radius:12px;
            padding:16px;text-align:center;margin:8px 0">
            <span style="font-size:32px">{card.level.emoji}</span><br>
            <span style="color:{color};font-size:28px;font-weight:bold">{card.level.label}</span><br>
            <span style="color:var(--ds-text2,#8b949e)">Direction: {card.direction} | Score: {card.score:.0f}</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception: pass


elif tab == "📚 知识库管理":
    st.subheader("📚 知识库管理")
    st.caption("三层知识体系: YAML规则 + Markdown文档 + ChromaDB向量检索")

    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager()
    except Exception as e:
        st.error(f"知识库初始化失败: {e}")
        km = None

    if km:
        # ── 知识库概览 ──
        kb_root = km.root
        prompt_files = list((kb_root / "prompts" / "system").glob("*.txt"))
        task_files = list((kb_root / "prompts" / "tasks").glob("*.txt"))
        few_shot_files = list((kb_root / "prompts" / "few_shots").glob("*.json"))
        rule_files = list((kb_root / "rules").glob("*.yaml"))
        ref_files = list((kb_root / "reference").glob("*.md"))
        strategy_files = list((kb_root / "strategies").glob("*.yaml"))

        total = len(prompt_files) + len(task_files) + len(few_shot_files) + len(rule_files) + len(ref_files) + len(strategy_files)

        # Metrics row
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("系统提示词", len(prompt_files))
        mc2.metric("任务提示词", len(task_files))
        mc3.metric("Few-shot", len(few_shot_files))
        mc4.metric("规则文件", len(rule_files))
        mc5.metric("总文件数", total)

        # ChromaDB status
        chroma_ok = km.chroma_available
        st.markdown(f"""
        <div style="background-color:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:8px;padding:12px;margin:8px 0">
            <span style="color:var(--ds-text2,#8b949e)">ChromaDB 向量库: </span>
            <span style="color:{'#3fb950' if chroma_ok else '#f0883e'}">● {'已初始化' if chroma_ok else '未初始化 (首次查询时自动播种)'}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs for different sections ──
        kbt1, kbt2, kbt3, kbt4 = st.tabs(["📋 提示词", "📖 规则与策略", "📝 参考文档", "🔍 检索测试"])

        with kbt1:
            st.subheader("系统提示词")
            for f in sorted(prompt_files):
                content = f.read_text(encoding="utf-8")
                version = "unknown"
                if content.startswith("---"):
                    try:
                        end = content.index("---", 3)
                        fm = content[3:end].strip()
                        for line in fm.split("\n"):
                            if line.startswith("version:"):
                                version = line.split(":", 1)[1].strip()
                    except ValueError:
                        pass
                with st.expander(f"{f.stem}  v{version}  ({len(content):,} 字)"):
                    st.code(content[:2000], language="markdown")

            if task_files:
                st.subheader("任务提示词")
                for f in sorted(task_files):
                    content = f.read_text(encoding="utf-8")
                    with st.expander(f"{f.stem}  ({len(content):,} 字)"):
                        st.code(content[:2000], language="markdown")

        with kbt2:
            st.subheader("规则文件")
            for f in sorted(rule_files):
                content = f.read_text(encoding="utf-8")
                with st.expander(f"{f.name}  ({len(content):,} 字)"):
                    st.code(content[:3000], language="yaml")

            st.subheader("策略注册表")
            for f in sorted(strategy_files):
                content = f.read_text(encoding="utf-8")
                with st.expander(f"{f.name}  ({len(content):,} 字)"):
                    st.code(content[:3000], language="yaml")

            # 策略统计
            try:
                strategies = km.list_strategies()
                if strategies:
                    strat_data = []
                    for s in strategies:
                        strat_data.append({
                            "策略名称": s.get("name", ""),
                            "类别": s.get("category", ""),
                            "适用体制": ", ".join(s.get("market_regimes", [])),
                            "容量上限": f"{s.get('capacity_limit', 0) / 1e6:.0f}M",
                        })
                    render_dataframe(pd.DataFrame(strat_data), height=400)
            except Exception:
                pass

        with kbt3:
            st.subheader("参考文档")
            for f in sorted(ref_files):
                content = f.read_text(encoding="utf-8")
                # Count sections
                sections = content.count("## ")
                with st.expander(f"{f.stem}  ({len(content):,} 字, {sections} 章节)"):
                    st.markdown(content[:5000])

            st.subheader("Few-shot 样例")
            for f in sorted(few_shot_files):
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    examples = data
                    scenario = f.stem
                else:
                    examples = data.get("examples", [])
                    scenario = data.get("scenario", f.stem)
                with st.expander(f"{scenario}  ({len(examples)} 示例)"):
                    st.json(data)

        with kbt4:
            st.subheader("🔍 知识库检索测试")
            st.caption("测试 ChromaDB 向量检索和关键词匹配")

            query = st.text_input("检索查询", placeholder="例如: 锤子线形态、T+1交易规则、MACD金叉...",
                                  key="kb_search_query")
            if query:
                with st.spinner("检索中..."):
                    try:
                        result = km.rag_query(query)
                        st.success(f"检索完成 — 返回 {len(str(result)):,} 字")
                        st.markdown(f"```\n{str(result)[:3000]}\n```")
                    except Exception as e:
                        st.warning(f"检索出错: {e}")

            st.divider()
            st.subheader("📊 提示词版本统计")
            try:
                all_prompts = km.list_all_prompts()
                if all_prompts:
                    vdata = []
                    for p in all_prompts:
                        vdata.append({
                            "名称": p["name"],
                            "版本": p["version"],
                            "更新日期": p["date"],
                            "大小": f"{p['size_chars']:,}字",
                        })
                    render_dataframe(pd.DataFrame(vdata), height=400)
            except Exception as e:
                st.warning(f"无法获取版本信息: {e}")

    else:
        st.info("知识库管理器未初始化,请检查 knowledge/ 目录结构")


# ═══════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════
st.divider()
st.caption(f"A股智能分析Agent v{VERSION} | DeepSeek V4 + LangGraph | ⚠️ 仅供研究学习，不构成投资建议")
