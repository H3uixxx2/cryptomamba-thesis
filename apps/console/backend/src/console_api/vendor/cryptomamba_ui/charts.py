from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CHART_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "doubleClick": False,
    "showTips": False,
}










def candle_chart(df: pd.DataFrame, title: str, prediction: dict | None = None, height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC candle",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )
    if prediction:
        fig.add_trace(
            go.Scatter(
                x=[df["date"].iloc[-1], prediction["prediction_date"]],
                y=[float(prediction["last_close"]), float(prediction["predicted_close"])],
                mode="lines+markers",
                name="Prediction",
                line=dict(color="#f97316", width=3, dash="dash"),
                marker=dict(size=9),
            )
        )
    fig.update_layout(
        height=height,
        title=title,
        margin=dict(l=12, r=12, t=48, b=12),
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def roi_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#2563eb", "#0f766e"]
    fig.add_trace(
        go.Bar(
            x=sim_df["strategy"],
            y=sim_df["roi_pct"],
            marker_color=colors,
            text=sim_df["roi_pct"].map(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        )
    )
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=28, b=12), yaxis_title="ROI (%)", showlegend=False)
    return fig


# --- Model-comparison charts (demo-friendly benchmark views) ---

_HIGHLIGHT_COLOR = "#1e3a8a"   # CryptoMamba-v (the model under test)
_NEUTRAL_COLOR = "#cbd5e1"     # reference baselines






def grouped_bar_chart(
    categories: list[str],
    series: dict[str, list[float]],
    *,
    title: str,
    y_title: str,
    value_fmt: str = "{:,.2f}",
    colors: list[str] | None = None,
    hline: float | None = None,
    hline_label: str = "",
    height: int = 360,
) -> go.Figure:
    """Generic grouped bar chart (categories x named series). Optional reference hline."""
    palette = colors or ["#94a3b8", "#2563eb", _HIGHLIGHT_COLOR, "#0f766e"]
    fig = go.Figure()
    for index, (name, vals) in enumerate(series.items()):
        fig.add_trace(
            go.Bar(
                name=name,
                x=categories,
                y=vals,
                marker_color=palette[index % len(palette)],
                text=[value_fmt.format(v) for v in vals],
                textposition="outside",
                hovertemplate=f"{name}<br>%{{x}}=%{{y}}<extra></extra>",
            )
        )
    if hline is not None:
        fig.add_hline(
            y=hline,
            line_dash="dot",
            line_color="#b45309",
            annotation_text=hline_label,
            annotation_position="top right",
        )
    fig.update_layout(
        barmode="group",
        height=height,
        margin=dict(l=12, r=12, t=46, b=40),
        title=title,
        yaxis_title=y_title,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
