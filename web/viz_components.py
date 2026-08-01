"""
Visualization Components (v3.1)

Additional Plotly chart components for the Streamlit dashboard:
  - radar chart: multi-dimension scoring
  - kpi_tiles: metric row (annual return / Sharpe / max drawdown / win rate)
  - equity_curve: portfolio NAV vs benchmark with drawdown shading
"""

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_radar_chart(
    scores: Dict[str, float],
    title: str = "Multi-Dimension Score",
    max_val: float = 100.0,
) -> go.Figure:
    """
    Radar/spider chart for multi-dimension scoring.

    Args:
        scores: {dimension_name: score_value}
        title: chart title
        max_val: maximum value for each axis

    Returns:
        Plotly Figure
    """
    categories = list(scores.keys())
    values = list(scores.values())

    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        name="Current Score",
        line=dict(color="#58a6ff", width=2),
        fillcolor="rgba(88, 166, 255, 0.2)",
    ))

    # Add benchmark line (midpoint)
    mid = [max_val / 2] * len(categories_closed)
    fig.add_trace(go.Scatterpolar(
        r=mid,
        theta=categories_closed,
        line=dict(color="#30363d", width=1, dash="dash"),
        name="Median (50)",
        showlegend=False,
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_val],
                showticklabels=False,
                gridcolor="#21262d",
            ),
            angularaxis=dict(
                gridcolor="#21262d",
                linecolor="#30363d",
            ),
            bgcolor="#0d1117",
        ),
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,
            font=dict(color="#f0f6fc", size=14),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(t=40, b=20, l=40, r=40),
        height=350,
    )

    return fig


def make_kpi_tiles(metrics: Dict[str, Any]) -> str:
    """
    Generate KPI tile HTML for Streamlit markdown.

    Args:
        metrics: {label: {value, format, color_rule}}

    Returns:
        HTML string for st.markdown()
    """
    tiles_html = ['<div style="display:flex;flex-wrap:wrap;gap:12px;margin:12px 0">']

    for key, cfg in metrics.items():
        val = cfg.get("value", 0)
        fmt = cfg.get("format", ".1f")
        label = cfg.get("label", key)
        suffix = cfg.get("suffix", "")

        # Color logic
        color_rule = cfg.get("color_rule", "")
        if isinstance(val, (int, float)):
            if color_rule == "positive_green":
                color = "#3fb950" if val > 0 else "#f85149"
            elif color_rule == "positive_red":
                color = "#f85149" if val > 0 else "#3fb950"
            elif color_rule == "above_0.5_green":
                color = "#3fb950" if val > 0.5 else "#f0883e"
            else:
                color = "#58a6ff"
        else:
            color = "#58a6ff"

        try:
            formatted_val = f"{float(val):{fmt}}{suffix}"
        except (ValueError, TypeError):
            formatted_val = str(val)

        tiles_html.append(
            f'<div style="background:#161b22;border:1px solid #30363d;'
            f'border-radius:8px;padding:12px 16px;min-width:120px;flex:1">'
            f'<div style="font-size:11px;color:#8b949e;margin-bottom:4px">{label}</div>'
            f'<div style="font-size:22px;font-weight:700;color:{color}">{formatted_val}</div>'
            f'</div>'
        )

    tiles_html.append('</div>')
    return "\n".join(tiles_html)


def make_equity_curve(
    dates: List[str],
    nav: List[float],
    benchmark_nav: Optional[List[float]] = None,
    title: str = "Equity Curve",
) -> go.Figure:
    """
    Portfolio equity curve with benchmark overlay and drawdown shading.

    Args:
        dates: date strings
        nav: portfolio NAV values
        benchmark_nav: optional benchmark NAV values
        title: chart title

    Returns:
        Plotly Figure
    """
    # 按日期排序 (防止快照乱序导致图表异常)
    if dates and nav and len(dates) == len(nav):
        sorted_pairs = sorted(zip(dates, nav), key=lambda x: x[0])
        dates = [p[0] for p in sorted_pairs]
        nav = [p[1] for p in sorted_pairs]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
    )

    # Portfolio NAV
    fig.add_trace(
        go.Scatter(
            x=dates, y=nav,
            mode="lines",
            name="Portfolio",
            line=dict(color="#58a6ff", width=2),
            hovertemplate="%{x}<br>Portfolio: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Benchmark NAV
    if benchmark_nav and len(benchmark_nav) == len(nav):
        fig.add_trace(
            go.Scatter(
                x=dates, y=benchmark_nav,
                mode="lines",
                name="CSI 300",
                line=dict(color="#8b949e", width=1.5, dash="dash"),
                hovertemplate="%{x}<br>CSI300: %{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Initial capital line
    initial = nav[0] if nav else 100000
    fig.add_hline(
        y=initial, line_dash="dot", line_color="#30363d",
        row=1, col=1,
    )

    # Drawdown subplot
    nav_arr = np.array(nav)
    peaks = np.maximum.accumulate(nav_arr)
    drawdowns = (nav_arr - peaks) / peaks * 100

    fig.add_trace(
        go.Scatter(
            x=dates, y=drawdowns,
            mode="lines",
            fill="tozeroy",
            name="Drawdown %",
            line=dict(color="#f85149", width=1),
            fillcolor="rgba(248, 81, 73, 0.15)",
            hovertemplate="%{x}<br>Drawdown: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(color="#f0f6fc", size=14),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#c9d1d9", size=11),
        ),
        hovermode="x unified",
        height=450,
    )

    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#21262d", color="#8b949e")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#21262d", color="#8b949e")
    fig.update_yaxes(title_text="NAV (RMB)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1, tickformat=".0%")

    return fig


def make_heatmap(
    data: List[List[float]],
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Sector Heatmap",
    colorscale: str = "RdYlGn",
) -> go.Figure:
    """
    Heatmap for sector rotation or correlation matrix.

    Args:
        data: 2D list of values
        row_labels: y-axis labels (sectors)
        col_labels: x-axis labels (dates)
        title: chart title
        colorscale: Plotly colorscale name

    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=col_labels,
        y=row_labels,
        colorscale=colorscale,
        zmid=0,
        hovertemplate="%{y}<br>%{x}<br>Return: %{z:.2%}<extra></extra>",
        colorbar=dict(
            title="Return",
            tickformat=".1%",
            outlinecolor="#30363d",
        ),
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(color="#f0f6fc", size=14),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
            side="bottom",
        ),
        yaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
        ),
        height=400,
    )

    return fig
