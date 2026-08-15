"""Dashboard HTML 表格渲染 — 从 dashboard.py 拆出 (v5.6 P1-14 拆分模块)。

`build_html_table()` 是纯函数: DataFrame → HTML 字符串, 便于单测 (尤其 XSS 转义)。
`render_dataframe()` 是 Streamlit 薄封装, 把纯函数结果交给 `st.markdown`。

Streamlit 1.60 的 GlideDataEditor 与自定义 `.stDataFrame` CSS 选择器不兼容,
导致表格内容不可见。此模块直接生成 GitHub-dark 风格 HTML 表格绕开该冲突。
"""
from __future__ import annotations

import html
from typing import Optional

import pandas as pd
import streamlit as st


def _escape(value: object) -> str:
    """HTML 转义 (防 XSS, v5.6 P1-14)。所有进入 <th>/<td> 的文本必须经此函数。"""
    return html.escape(str(value), quote=True)


def _format_cell(val, fmt: str) -> str:
    """按 formatter 或默认数值格式生成单元格文本 (纯文本, 转义交由 _escape 统一处理)。"""
    # None/NaN 优先判空 (v5.6 P1-14 修复: 原实现 float 分支在前, NaN 永远渲染成 "nan")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if fmt and isinstance(val, (int, float)):
        return fmt % val
    if isinstance(val, float):
        if abs(val) >= 1e8:
            return f"{val:,.0f}"
        if abs(val) >= 100:
            return f"{val:,.1f}"
        if abs(val) >= 1:
            return f"{val:.2f}"
        return f"{val:.4f}"
    return str(val)


def build_html_table(
    df: pd.DataFrame,
    max_rows: int = 500,
    height: int = 400,
    col_rename: Optional[dict] = None,
    formatters: Optional[dict] = None,
) -> str:
    """DataFrame → GitHub-dark 风格 HTML 表格字符串 (纯函数)。

    所有单元格文本经 `html.escape` 转义, 防 XSS (v5.6 P1-14)。

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
    header_names = [col_rename.get(c, c) for c in cols]
    rows = df.values.tolist()

    html = f'<div style="max-height:{height}px;overflow-y:auto;border:1px solid var(--ds-border,#30363d);border-radius:8px">'
    html += '<table style="width:100%;border-collapse:collapse;font-size:13px">'

    # Header
    html += '<thead><tr style="background:var(--ds-bg3,#21262d);position:sticky;top:0;z-index:10">'
    for name in header_names:
        html += (f'<th style="padding:6px 10px;text-align:left;color:var(--ds-text2,#8b949e);'
                 f'border-bottom:2px solid var(--ds-border,#30363d);white-space:nowrap">{_escape(name)}</th>')
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
            tv = _escape(_format_cell(val, fmt))
            # 右对齐数字列, 左对齐文本列
            align = "right" if isinstance(val, (int, float)) and not (isinstance(val, float) and pd.isna(val)) else "left"
            html += (f'<td style="padding:4px 10px;color:var(--ds-text,#c9d1d9);'
                     f'white-space:nowrap;text-align:{align}">{tv}</td>')
        html += '</tr>'

    html += '</tbody></table></div>'
    return html


def render_dataframe(df: pd.DataFrame, max_rows: int = 500, height: int = 400,
                     col_rename: dict = None, formatters: dict = None) -> None:
    """Streamlit 封装: 把纯函数 `build_html_table` 的结果交给 `st.markdown`。"""
    st.markdown(build_html_table(df, max_rows, height, col_rename, formatters),
                unsafe_allow_html=True)
