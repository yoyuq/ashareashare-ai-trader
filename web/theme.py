"""Dashboard 暗色主题 CSS/JS — 从 dashboard.py 拆出 (v5.6 P1-14 拆分模块)。

把 Streamlit 自定义组件 (metric card / expander / tab / 输入控件 / 表格等) 的
主题变量与 theme-sync JS 集中于此, 供 dashboard.py 一行调用 `apply_theme()`。
"""
import streamlit as st

THEME_CSS = r"""<style>
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
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
