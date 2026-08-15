"""v5.6 P1-14 Dashboard 拆分模块 + HTML 转义 (防 XSS)

覆盖 `web/render.py::build_html_table` 纯函数:
  - 字符串单元格经 html.escape 转义 (防 XSS 注入)
  - 表头 (含 col_rename 中文名) 同样转义
  - 数值列 formatter / 默认格式不被破坏
  - None/NaN 渲染为 "-"
"""
import numpy as np
import pandas as pd

from web.render import build_html_table


def test_string_cell_escapes_script_tag():
    """含 <script> 的字符串值必须被转义, 不得原样进入 HTML."""
    df = pd.DataFrame({"name": ["<script>alert(1)</script>", "正常"]})
    h = build_html_table(df)
    assert "<script>alert(1)</script>" not in h
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in h


def test_string_cell_escapes_html_entities():
    """& / < / > / 引号 均被转义."""
    df = pd.DataFrame({"name": ["茅台 & 五粮液 <贵> 'quote' \"d\""]})
    h = build_html_table(df)
    assert "茅台 &amp; 五粮液" in h
    assert "&lt;贵&gt;" in h
    assert "&#x27;quote&#x27;" in h


def test_header_rename_escaped():
    """col_rename 的中文名作为表头同样转义 (防列名注入)."""
    df = pd.DataFrame({"x": [1]})
    h = build_html_table(df, col_rename={"x": "<b>列</b>"})
    assert "<b>列</b>" not in h
    assert "&lt;b&gt;列&lt;/b&gt;" in h


def test_numeric_formatter_preserved():
    """数值列 formatter 仍生效 (价格 ¥%.2f), 数值不被当作字符串转义破坏."""
    df = pd.DataFrame({"price": [1234.5, 0.5], "pct": [3.21, -1.5]})
    h = build_html_table(df, formatters={"price": "¥%.2f", "pct": "%+.2f%%"})
    assert "¥1234.50" in h
    assert "+3.21%" in h
    assert "-1.50%" in h


def test_none_and_nan_render_dash():
    """None / NaN 渲染为 '-'."""
    df = pd.DataFrame({"a": [None, np.nan, 0.0]})
    h = build_html_table(df)
    assert h.count(">-</td>") == 2  # None + NaN 两处
    assert "0.0000" in h  # 0.0 走默认数值格式


def test_empty_dataframe_no_rows():
    """空 DataFrame 不崩溃, 只渲染表头."""
    h = build_html_table(pd.DataFrame(columns=["a", "b"]))
    assert "<thead>" in h and "<tbody>" in h
